"""Journaled relocation of external Features; numeric producers and indexes stay intact.

Copy immutable completed files while generation continues. At cutover, briefly
stop only Features, drain new external rows, validate destination reads, replace
root mappings and export manifests. Original files are removed only after the
new mapping is active and a second destination checksum passes. Targets and the
system Features root are never copied, edited or deleted.
"""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

from tools.training.diffusiondrive_v1.features_cache import (
    SCHEMA, GIB, StoragePause, active_mount, digest, directory_sync,
    durable_json, file_digest, safe_path)

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / 'outputs/diffusiondrive_v1/external_features_migration'
CONFIGS = [('baseline', 'features_cache.json'), ('remaining', 'features_remaining_20261015.json')]
DISKS = {
    'cache_a': {'label': 'YMTC', 'mountpoint': '/run/media/aethelind/48EA01DBEA01C5E2',
                'uuid': '48EA01DBEA01C5E2', 'fstype': 'ntfs3', 'budget_gib': 450,
                'minimum_free_gib': 100, 'alias': '/data/datasets/features/ymtc'},
    'cache_b': {'label': 'PSSD', 'mountpoint': '/run/media/aethelind/PSSD',
                'uuid': 'F8E4AE34E4ADF552', 'fstype': 'ntfs3', 'budget_gib': 650,
                'minimum_free_gib': 100, 'alias': '/data/datasets/features/pssd'},
}


def read_json(path):
    return json.loads(Path(path).read_text())


def readonly(path):
    db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


@contextmanager
def locked(path, blocking=False):
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise StoragePause('Another process holds ' + str(path)) from exc
        yield stream


def connection():
    db = sqlite3.connect(STATE / 'migration.sqlite', timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('''CREATE TABLE IF NOT EXISTS files(
        job TEXT, root_id TEXT, sample_id TEXT, relative_path TEXT, sha256 TEXT,
        bytes INTEGER, phase TEXT DEFAULT 'pending', source_stamp TEXT,
        PRIMARY KEY(job,root_id,sample_id))''')
    db.commit()
    return db


def owner(job, root_id):
    return {'schema': SCHEMA, 'root_id': root_id, 'dataset_id': job['dataset_id'],
            'index_id': job['index_id']}


def assert_owner(job, root_id, spec):
    active_mount(spec)
    if read_json(Path(spec['path']) / 'cache_root.json') != owner(job, root_id):
        from tools.training.diffusiondrive_v1.feature_shared_storage import read_registry, check_owner
        check_owner(read_registry(),root_id,spec,job['index_id'],job['dataset_id'],job['producer_sha256'])


def validate_job(job):
    with readonly(Path(job['state']) / 'index.sqlite') as db:
        meta = {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
    if (meta['index_id'] != job['index_id'] or meta['dataset_id'] != job['dataset_id']
            or digest(meta['producer']) != job['producer_sha256']):
        raise StoragePause('Index identity changed: ' + job['name'])


def prepare():
    STATE.mkdir(parents=True, exist_ok=True)
    with locked(STATE / 'launch.lock'):
        if (STATE / 'plan.json').exists():
            raise StoragePause('Plan already exists; use start/resume')
        jobs = []
        for name, filename in CONFIGS:
            config_path = ROOT / 'tools/training/diffusiondrive_v1' / filename
            config = read_json(config_path)
            state = ROOT / config['state_dir']
            with readonly(state / 'index.sqlite') as db:
                meta = {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
            new = json.loads(json.dumps(config))
            job = {'name': name, 'config': str(config_path), 'state': str(state),
                   'dataset_id': config['dataset_id'], 'index_id': meta['index_id'],
                   'producer_sha256': digest(meta['producer']), 'before': config,
                   'destinations': {}, 'aliases': {}}
            for root_id, disk in DISKS.items():
                old = config['cache_roots'][root_id]
                assert_owner(job, root_id, old)
                parent = Path(disk['mountpoint']) / ('diffusiondrive_v1_tensors_' + disk['label']) / 'features'
                spec = {k: disk[k] for k in ('mountpoint', 'uuid', 'fstype', 'budget_gib', 'minimum_free_gib')}
                spec['path'] = str(parent / name)
                mounted = active_mount(spec, writing=True)
                if mounted['free_bytes'] < (disk['budget_gib'] + disk['minimum_free_gib']) * GIB:
                    raise StoragePause('Not enough real capacity for destination budget: ' + disk['label'])
                if spec['path'] == old['path']:
                    raise StoragePause('Source equals destination')
                path = Path(spec['path'])
                if path.exists() and any(path.iterdir()):
                    if ({p.name for p in path.iterdir()} != {'cache_root.json'}
                            or read_json(path/'cache_root.json') != owner(job, root_id)):
                        raise StoragePause('Destination must initially be empty: ' + str(path))
                job['destinations'][root_id] = spec
                alias = dict(spec, path=str(Path(disk['alias']) / name), mountpoint=disk['alias'])
                job['aliases'][root_id] = alias
                new['cache_roots'][root_id] = alias
            job['after'] = new
            jobs.append(job)
        # Reserve both job namespaces within one physical-disk budget.
        for root_id, disk in DISKS.items():
            with readonly(Path(jobs[0]['state']) / 'index.sqlite') as db:
                baseline_bytes = db.execute("SELECT COALESCE(SUM(bytes),0) FROM samples WHERE state='complete' AND root_id=?", (root_id,)).fetchone()[0]
            remainder = disk['budget_gib'] - (baseline_bytes + GIB - 1) // GIB - 1
            jobs[1]['destinations'][root_id]['budget_gib'] = remainder
            jobs[1]['aliases'][root_id]['budget_gib'] = remainder
            jobs[1]['after']['cache_roots'][root_id]['budget_gib'] = remainder
        plan = {'schema': 'b2d-v1-external-features-migration-v1', 'created': time.time(),
                'jobs': jobs, 'disks': DISKS, 'protected_roots': ['cache_system'],
                'targets_untouched': True}
        for job in jobs:
            for root_id, spec in job['destinations'].items():
                path = Path(spec['path']); path.mkdir(parents=True, exist_ok=True)
                durable_json(path / 'cache_root.json', owner(job, root_id))
        durable_json(STATE / 'plan.json', plan)
        db = connection()
        try:
            refresh(plan, db)
        finally:
            db.close()
        return plan


def refresh(plan, db):
    for job in plan['jobs']:
        validate_job(job)
        with readonly(Path(job['state']) / 'index.sqlite') as source:
            rows = source.execute("SELECT sample_id,root_id,relative_path,sha256,bytes FROM samples WHERE state='complete' AND root_id IN ('cache_a','cache_b')").fetchall()
        for row in rows:
            previous = db.execute('SELECT * FROM files WHERE job=? AND root_id=? AND sample_id=?',
                                  (job['name'], row['root_id'], row['sample_id'])).fetchone()
            if previous and any(previous[k] != row[k] for k in ('relative_path','sha256','bytes')):
                raise StoragePause('Completed source row changed: ' + row['sample_id'])
            db.execute('INSERT OR IGNORE INTO files(job,root_id,sample_id,relative_path,sha256,bytes) VALUES(?,?,?,?,?,?)',
                       (job['name'], row['root_id'], row['sample_id'], row['relative_path'], row['sha256'], row['bytes']))
        db.commit()


def stamp(path):
    s = path.stat()
    return json.dumps([s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns])


def verify_file(path, row):
    if path.is_symlink() or path.stat().st_size != row['bytes'] or file_digest(path) != row['sha256']:
        raise StoragePause('Size/SHA mismatch; keep original: ' + str(path))


def copy_verified(source, dest, row):
    """No overwrite of existing files; a committed rename is recoverable on retry."""
    before = stamp(source)
    if dest.exists():
        verify_file(dest, row)
        verify_file(source, row)
        if stamp(source) != before:
            raise StoragePause('Source changed during read')
        return before
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + '.external_migration.partial')
    if partial.is_symlink():
        raise StoragePause('Refuse symlink temporary file')
    # Only this migration owns this deterministic temporary name under its lock.
    partial.unlink(missing_ok=True)
    try:
        checksum = hashlib.sha256(); count = 0
        with source.open('rb') as src, partial.open('xb') as out:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk: break
                checksum.update(chunk); count += len(chunk); out.write(chunk)
            out.flush(); os.fsync(out.fileno())
        if count != row['bytes'] or checksum.hexdigest() != row['sha256'] or stamp(source) != before:
            raise StoragePause('Source checksum/identity changed; no cutover')
        verify_file(partial, row)
        os.replace(partial, dest)
        directory_sync(dest.parent)
        return before
    finally:
        partial.unlink(missing_ok=True)


def safe_delete(source, dest, row):
    """Never remove the last verified copy. Source must be unchanged since copy."""
    verify_file(dest, row)
    if not source.exists(): return
    if source.is_symlink() or stamp(source) != row['source_stamp']:
        raise StoragePause('Source changed since verified copy; retain it: ' + str(source))
    source.unlink()
    directory_sync(source.parent)


def stats(db):
    return {r['phase']: {'files': r['n'], 'gib': round(r['b'] / GIB, 3)}
            for r in db.execute('SELECT phase,COUNT(*) n,SUM(bytes) b FROM files GROUP BY phase')}


def progress(db, phase, **extra):
    result = {'pid': os.getpid(), 'phase': phase, 'updated': time.time(), 'files': stats(db), **extra}
    durable_json(STATE / 'status.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def migration_preflight(state, config=None):
    """Fence launch/run across a multi-config cutover or interrupted cutover."""
    guard = Path(state) / 'external_migration_cutover.json'
    if guard.exists():
        raise StoragePause('Features relocation cutover in progress; resume feature_migration first')
    active = Path(state) / 'external_migration_active.json'
    if active.exists() and config is not None:
        for root_id, identity in read_json(active)['roots'].items():
            actual = config['cache_roots'].get(root_id, {})
            if any(actual.get(k) != v for k,v in identity.items()):
                raise StoragePause('Stale/redirected external root after migration: ' + root_id)


def register_active(job):
    identities = {root_id: {k:job['after']['cache_roots'][root_id][k] for k in ('path','mountpoint','uuid','fstype')}
                  for root_id in job['aliases']}
    durable_json(Path(job['state'])/'external_migration_active.json', {'migration':str(STATE),'roots':identities})


def stopped():
    return (STATE / 'stop.request').exists()


def copy_pending(plan, db, limit=None):
    jobs = {job['name']: job for job in plan['jobs']}
    copied = 0
    rows = db.execute("SELECT * FROM files WHERE phase='pending' ORDER BY job,root_id,sample_id").fetchall()
    if limit is not None:
        # A bounded validation run covers all four source/owner combinations.
        from collections import defaultdict, deque
        buckets = defaultdict(deque)
        for row in rows: buckets[(row['job'],row['root_id'])].append(row)
        rows = []
        while buckets and len(rows) < limit:
            for key in list(buckets):
                rows.append(buckets[key].popleft())
                if not buckets[key]: del buckets[key]
                if len(rows) >= limit: break
    used = {r[0]: r[1] for r in db.execute("SELECT root_id,SUM(bytes) FROM files WHERE phase!='pending' GROUP BY root_id")}
    checked = set()
    for row in rows:
        if stopped() or (limit is not None and copied >= limit): break
        job = jobs[row['job']]; root_id = row['root_id']
        source_spec = job['before']['cache_roots'][root_id]
        destination = job['destinations'][root_id]
        if (row['job'],root_id) not in checked:
            assert_owner(job, root_id, source_spec)
            assert_owner(job, root_id, destination)
            checked.add((row['job'],root_id))
        active_mount(source_spec)
        mounted = active_mount(destination, writing=True)
        if mounted['free_bytes'] - row['bytes'] < destination['minimum_free_gib'] * GIB:
            raise StoragePause('Destination free-space floor reached')
        if used.get(root_id,0) + row['bytes'] > plan['disks'][root_id]['budget_gib'] * GIB:
            raise StoragePause('Destination migration budget reached')
        source = safe_path(source_spec, row['relative_path'])
        dest = safe_path(destination, row['relative_path'])
        source_stamp = copy_verified(source, dest, row)
        db.execute("UPDATE files SET phase='copied',source_stamp=? WHERE job=? AND root_id=? AND sample_id=?",
                   (source_stamp, row['job'], root_id, row['sample_id']))
        db.commit(); copied += 1
        used[root_id] = used.get(root_id,0) + row['bytes']
        if copied % 50 == 0: progress(db, 'copying')
    progress(db, 'copying')
    return copied


def validate_aliases(plan):
    for job in plan['jobs']:
        for root_id, alias in job['aliases'].items():
            spec = job['destinations'][root_id] if plan.get('flat_layout') else alias
            active_mount(spec, writing=True)
            physical = Path(job['destinations'][root_id]['path'])
            target = Path(alias['path'])
            if not target.is_dir() or not os.path.samestat(physical.stat(), target.stat()):
                raise StoragePause('Bind mount missing/different; run mount_features.sh: ' + str(target))
            assert_owner(job, root_id, spec)


def validate_reads(plan):
    """Actual Dataset reads for every migrated root, plus unchanged system root."""
    import torch
    from tools.training.diffusiondrive_v1.features_cache import FeatureCacheDataset
    torch.set_num_threads(1)
    evidence = []
    for job in plan['jobs']:
        with readonly(Path(job['state']) / 'index.sqlite') as db:
            for root_id in job['after']['cache_roots']:
                row = db.execute("SELECT * FROM samples WHERE state='complete' AND root_id=? ORDER BY sample_id LIMIT 1", (root_id,)).fetchone()
                if row is None: continue
                dataset = FeatureCacheDataset(Path(job['state']) / 'index.sqlite', job['after']['cache_roots'],
                                              json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0]),
                                              sample_ids=[row['sample_id']])
                tensors = dataset[0]
                evidence.append({'job': job['name'], 'root_id': root_id, 'sample_id': row['sample_id'],
                                 'sha256': row['sha256'], 'shapes': {k:list(v.shape) for k,v in tensors.items()}})
    durable_json(STATE / 'dataset_validation.json', evidence)


def cutover(plan, db):
    from tools.training.diffusiondrive_v1.feature_service import alive
    validate_aliases(plan)  # Do not stop a working generator for absent mounts.
    record_path = STATE / 'cutover.json'
    record = read_json(record_path) if record_path.exists() else {'resume_jobs': [], 'phase': 'preparing'}
    if record['phase'] == 'committed': return record
    with ExitStack() as locks:
        for job in plan['jobs']:
            locks.enter_context(locked(Path(job['state']) / 'launch.lock', blocking=True))
            current = read_json(job['config'])
            if current != job['before'] and current != job['after']:
                raise StoragePause('Config changed outside migration: ' + job['config'])
        # Persist restart intent before any stop request; recover even after a crash.
        if record['phase'] == 'preparing':
            record['resume_jobs'] = [job['name'] for job in plan['jobs'] if alive(Path(job['state']))]
            record['phase'] = 'stopping'
            durable_json(record_path, record)
        for job in plan['jobs']:
            durable_json(Path(job['state']) / 'external_migration_cutover.json', {'migration': str(STATE)})
            if alive(Path(job['state'])):
                durable_json(Path(job['state']) / 'stop.request', {'requested': time.time(), 'reason': 'external_features_relocation'})
        deadline = time.monotonic() + 180
        while any(alive(Path(job['state'])) for job in plan['jobs']):
            if time.monotonic() > deadline: raise StoragePause('Features did not stop; retain both copies and cutover fence')
            time.sleep(.5)
        for job in plan['jobs']:
            locks.enter_context(locked(Path(job['state']) / 'writer.lock'))
            validate_job(job)
            for spec in job['before']['cache_roots'].values():
                if spec['path'] == job['before']['cache_roots'].get('cache_system', {}).get('path'):
                    continue
                locks.enter_context(locked(Path(spec['path']) / '.writer.lock'))
        refresh(plan, db)
        copy_pending(plan, db)
        if stopped(): raise StoragePause('Stop requested; cutover remains fenced and resumable')
        if db.execute("SELECT COUNT(*) FROM files WHERE phase='pending'").fetchone()[0]:
            raise StoragePause('New source files still pending')
        for job in plan['jobs']:
            with readonly(Path(job['state']) / 'index.sqlite') as source:
                pinned = source.execute("SELECT COUNT(*) FROM samples WHERE state='pending' AND root_id IN ('cache_a','cache_b')").fetchone()[0]
                if pinned: raise StoragePause('Uncommitted external samples must be resolved before cutover')
        validate_reads(plan)
        record['phase'] = 'switching'; durable_json(record_path, record)
        for job in plan['jobs']:
            durable_json(Path(job['state']) / 'config_before_external_migration.json', job['before'])
            durable_json(job['config'], job['after'])
        # Same sample rows/root IDs/relative paths/checksums: no index mutation.
        # Export current collections before removing any source copy.
        from tools.training.diffusiondrive_v1.feature_jobs import export_collection
        export_collection(ROOT / 'tools/training/diffusiondrive_v1/features_jobs.json')
        record['phase'] = 'committed'; durable_json(record_path, record)
        for job in plan['jobs']:
            register_active(job)
            (Path(job['state']) / 'external_migration_cutover.json').unlink(missing_ok=True)
            directory_sync(Path(job['state']))
    return record


def restart(plan, record):
    from tools.training.diffusiondrive_v1.feature_service import alive
    # Recover a crash after commit but before removing the launch fences.
    validate_aliases(plan)
    for job in plan['jobs']:
        if read_json(job['config']) != job['after']:
            raise StoragePause('Committed configuration differs; do not restart or clean')
        with locked(Path(job['state'])/'launch.lock', blocking=True):
            register_active(job)
            (Path(job['state'])/'external_migration_cutover.json').unlink(missing_ok=True)
            directory_sync(Path(job['state']))
    for job in plan['jobs']:
        if job['name'] in record['resume_jobs'] and job['name'] not in record.get('restarted_jobs',[]):
            if not alive(Path(job['state'])):
                stop = Path(job['state'])/'stop.request'
                if stop.exists() and read_json(stop).get('reason') != 'external_features_relocation':
                    raise StoragePause('Manual stop request retained; restart explicitly before migration cleanup')
                subprocess.run([job['after']['python'], '-m', 'tools.training.diffusiondrive_v1.feature_service',
                                'resume', '--config', job['config']], cwd=ROOT, check=True)
            # Do not clean originals merely because Popen succeeded. Confirm
            # the restarted generator can commit through the new mapping.
            expected_pid = read_json(Path(job['state'])/'pid.json')['pid']
            deadline = time.monotonic() + 180
            while True:
                status_path = Path(job['state'])/'status.json'
                status = read_json(status_path) if status_path.exists() else {}
                if status.get('pid') == expected_pid:
                    if status.get('counts',{}).get('success',0) > 0:
                        break
                    if status.get('state') == 'complete':
                        break
                if not alive(Path(job['state'])) or time.monotonic() > deadline:
                    raise StoragePause('Resumed generator has not confirmed a new commit; keep original files')
                time.sleep(.5)
            record.setdefault('restarted_jobs',[]).append(job['name'])
            durable_json(STATE/'cutover.json',record)


def cleanup(plan, db):
    for job in plan['jobs']:
        if read_json(job['config']) != job['after']:
            raise StoragePause('Destination not active; no source removal')
    jobs = {j['name']:j for j in plan['jobs']}
    for index, row in enumerate(db.execute("SELECT * FROM files WHERE phase='copied' ORDER BY job,root_id,sample_id").fetchall()):
        if stopped(): break
        job = jobs[row['job']]; root_id = row['root_id']
        source_spec = job['before']['cache_roots'][root_id]
        destination = job['after']['cache_roots'][root_id]
        assert_owner(job, root_id, source_spec)
        assert_owner(job, root_id, destination)
        active_mount(source_spec, writing=True)
        safe_delete(safe_path(source_spec, row['relative_path']), safe_path(destination, row['relative_path']), row)
        db.execute("UPDATE files SET phase='deleted' WHERE job=? AND root_id=? AND sample_id=?", (row['job'],root_id,row['sample_id']))
        db.commit()
        if index % 50 == 0: progress(db, 'cleaning_verified_sources')


def run(limit=None):
    if (STATE/'flatten_in_progress.json').exists():
        raise StoragePause('Finish feature_migration flatten before copying resumes')
    plan = read_json(STATE / 'plan.json')
    with locked(STATE / 'worker.lock'), ExitStack() as locks:
        acquired = set()
        for job in plan['jobs']:
            for root_id, spec in job['destinations'].items():
                assert_owner(job, root_id, spec)
                # Same locks used by the future generator. Release after copy,
                # before restarting it on the new roots.
                lock_path = Path(spec['path']) / '.writer.lock'
                if lock_path not in acquired:
                    locks.enter_context(locked(lock_path))
                    acquired.add(lock_path)
        db = connection()
        try:
            def request_stop(*_): durable_json(STATE / 'stop.request', {'requested':time.time()})
            signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
            committed = (STATE / 'cutover.json').exists() and read_json(STATE/'cutover.json')['phase']=='committed'
            if not committed:
                refresh(plan, db); progress(db, 'copying')
                copy_pending(plan, db, limit)
                if stopped() or limit is not None:
                    progress(db, 'stopped' if stopped() else 'validation_batch_complete'); return
                progress(db, 'checking_mounts')
                record = cutover(plan, db)
            else:
                record = read_json(STATE/'cutover.json')
            locks.close()
            restart(plan, record)
            progress(db, 'cleaning_verified_sources')
            cleanup(plan, db)
            remaining = db.execute("SELECT COUNT(*) FROM files WHERE phase!='deleted'").fetchone()[0]
            progress(db, 'complete' if remaining == 0 else 'stopped')
        except Exception as exc:
            progress(db, 'paused', error=str(exc))
            raise
        finally:
            db.close()


def main():
    from tools.training.diffusiondrive_v1.feature_service import process_identity
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','flatten','start','resume','run','status','stop','logs'])
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.limit is not None and args.limit < 1: p.error('limit must be positive')
    if args.action=='prepare':
        plan=prepare(); print(json.dumps({'plan':str(STATE/'plan.json'),'jobs':len(plan['jobs'])})); return
    if args.action=='flatten':
        from tools.training.diffusiondrive_v1.feature_flatten import flatten
        flatten(); return
    if args.action=='stop': durable_json(STATE/'stop.request',{'requested':time.time()}); return
    if args.action=='logs':
        from collections import deque
        with (STATE/'worker.log').open() as f: print(''.join(deque(f,maxlen=12)),end='')
        return
    if args.action=='status':
        status=read_json(STATE/'status.json') if (STATE/'status.json').exists() else {}
        stored=read_json(STATE/'pid.json') if (STATE/'pid.json').exists() else {}
        status['running']=bool(stored and process_identity(stored['pid'])==stored)
        if not status['running'] and (STATE/'worker.lock').exists():
            try:
                with locked(STATE/'worker.lock'): pass
            except StoragePause:
                status['running']=True
        if (STATE/'migration.sqlite').exists():
            with connection() as db: status['files']=stats(db)
        print(json.dumps(status,ensure_ascii=False,indent=2)); return
    if args.action=='run': run(args.limit); return
    with locked(STATE/'launch.lock'):
        if (STATE/'pid.json').exists():
            stored=read_json(STATE/'pid.json')
            if process_identity(stored['pid'])==stored:
                raise StoragePause('Migration worker already active')
        with locked(STATE/'worker.lock'): pass
        (STATE/'stop.request').unlink(missing_ok=True)
        plan=read_json(STATE/'plan.json')
        cmd=[plan['jobs'][0]['before']['python'],'-u','-m','tools.training.diffusiondrive_v1.feature_migration','run']
        if args.limit is not None: cmd += ['--limit',str(args.limit)]
        with (STATE/'worker.log').open('ab',buffering=0) as log,open(os.devnull,'rb') as null:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=null,stdout=log,stderr=subprocess.STDOUT,
                                  start_new_session=True,close_fds=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        identity=process_identity(proc.pid)
        if identity is None: raise StoragePause('Migration exited during start; inspect logs')
        durable_json(STATE/'pid.json',identity)
        print(json.dumps({'pid':proc.pid,'command':cmd,'log':str(STATE/'worker.log')}))


if __name__=='__main__':
    main()
