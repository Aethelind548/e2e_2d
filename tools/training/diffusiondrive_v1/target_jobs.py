"""Annotation-only full-dataset scheduling over the frozen V1 target builders."""
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from tools.b2d_adapter.errors import AdapterError
from tools.b2d_adapter.route_index import RouteIndex, MODALITIES
from tools.training.diffusiondrive_v1 import target_cache as core
from tools.training.diffusiondrive_v1.features_cache import (
    StoragePause, GIB, digest, durable_json, file_digest, pin, sample_id, safe_path, directory_sync)

DISCOVERY_VERSION = 'b2d-annotation-target-windows-v1'


class AnnotationIndex(RouteIndex):
    """Reuse RouteIndex references and TemporalSampler with no sensor enumeration.

    Existence is established once during the filename snapshot. Actual annotation
    reads and SHA/stat checks still happen before/after every generated target.
    """
    def __init__(self, route):
        self.route = Path(route).resolve()
        self.refs = {}
        self.files = {m: {} for m in MODALITIES}
        with os.scandir(self.route / 'anno') as entries:
            for entry in entries:
                match = re.fullmatch(r'(\d+)\.json\.gz', entry.name)
                if match and entry.is_file(follow_symlinks=False):
                    frame = int(match[1])
                    if frame in self.files['anno']:
                        raise AdapterError('duplicate_frame_id', f'anno {frame}')
                    self.files['anno'][frame] = Path(entry.path)
        self.frame_ids = tuple(sorted(self.files['anno']))
        if not self.frame_ids:
            raise AdapterError('empty_route', str(self.route))

    def ref(self, frame_id):
        # TemporalSampler reuses each real frame across ~41 overlapping windows.
        # Memoize immutable references to avoid repeated filesystem resolution.
        if frame_id not in self.refs:
            self.refs[frame_id] = super().ref(frame_id)
        return self.refs[frame_id]

    def path(self, ref, modality):
        self.check_ref(ref)
        if modality != 'anno':
            raise AdapterError('targets_sensor_access_forbidden', modality)
        path = self.files['anno'].get(ref.frame_id)
        if path is None:
            raise AdapterError('missing_frame', ref.token)
        return path


def open_index(path):
    db = core.open_index(path)
    db.execute('CREATE TABLE IF NOT EXISTS discoveries(route TEXT PRIMARY KEY,summary TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS bev_counts(sample_id TEXT PRIMARY KEY,pixels_json TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS target_queue ON samples(state,route,frame)')
    db.commit()
    return db


def summary(db):
    states = dict(db.execute('SELECT state,COUNT(*) FROM samples GROUP BY state').fetchall())
    return {'candidates': sum(states.values()), 'states': states,
            'routes_discovered': db.execute('SELECT COUNT(*) FROM discoveries').fetchone()[0],
            'error_codes': dict(db.execute("SELECT error_code,COUNT(*) FROM samples WHERE error_code IS NOT NULL GROUP BY error_code").fetchall())}


def event(row, state, sha, code=None, reason=None):
    return {'sample_id': row['sample_id'], 'route': row['route'], 'frame': row['frame'],
            'status': state, 'error_code': code, 'reason': reason, 'producer_sha': sha,
            'timestamp': time.time(), 'retryable': state == 'failed_retryable'}


def discover(db, config, native, sha, stopped=lambda: False):
    from tools.b2d_adapter.temporal_sampler import TemporalSampler
    root = Path(config['dataset_root']).resolve()
    # Ensure the source is on the expected mounted volume too.
    core.storage_check(dict(config['target_root'], path=str(root)))
    with os.scandir(root) as entries:
        routes = sorted(e.name for e in entries if e.is_dir(follow_symlinks=False) and (root / e.name / 'anno').is_dir())
    if not routes:
        raise StoragePause('No local annotation Routes found')
    pin(db, 'requested_routes', routes)
    pin(db, 'discovery_version', DISCOVERY_VERSION)
    sampler = TemporalSampler(native.geometry)
    for route in routes:
        if stopped(): return False
        if db.execute('SELECT 1 FROM discoveries WHERE route=?', (route,)).fetchone(): continue
        core.storage_check(dict(config['target_root'], path=str(root)))
        try:
            index = AnnotationIndex.from_dataset_root(root, route)
        except AdapterError as exc:
            db.execute('INSERT INTO discoveries VALUES(?,?)', (route, json.dumps({'error_code': exc.code, 'reason': str(exc), 'candidates': 0})))
            db.commit(); continue
        counters = {'pending': 0, 'ineligible': 0}
        for frame in index.frame_ids:
            sid = sample_id(config['dataset_id'], route, frame)
            state, code, reason = 'pending', None, None
            try:
                sampler.sample(index, index.ref(frame), full_scene=False, target_only=True)
            except AdapterError as exc:
                state, code, reason = 'ineligible', exc.code, str(exc)
            db.execute('INSERT OR IGNORE INTO samples(sample_id,dataset_id,route,frame,state,error_code,error,updated) VALUES(?,?,?,?,?,?,?,?)',
                       (sid, config['dataset_id'], route, frame, state, code, reason, time.time()))
            counters[state] += 1
        # A route and its rows are committed together; interrupted discovery repeats safely.
        db.execute('INSERT INTO discoveries VALUES(?,?)', (route, json.dumps(counters)))
        db.commit()
        if db.execute('SELECT COUNT(*) FROM discoveries').fetchone()[0] % 50 == 0:
            print(json.dumps({'event': 'targets_discovery', **summary(db), 'routes_total': len(routes)}), flush=True)
    return True


def failure_state(exc):
    """Only bounded retry for transient access; geometry/qualification is permanent."""
    cause = exc
    while cause is not None:
        if isinstance(cause, OSError):
            if isinstance(cause, gzip.BadGzipFile) and isinstance(exc, AdapterError):
                return 'ineligible'
            if cause.errno in (errno.EAGAIN, errno.EINTR, errno.EBUSY, errno.ESTALE, errno.ETIMEDOUT):
                return 'failed_retryable'
            # Missing files after discovery and device/permission faults need inspection.
            raise StoragePause(str(exc)) from exc
        cause = cause.__cause__
    if isinstance(exc, AdapterError):
        if exc.code in {'dependency_missing', 'map_version_mismatch', 'target_contract_mismatch',
                        'invalid_targets_cache', 'missing_route_map', 'targets_sensor_access_forbidden'}:
            raise StoragePause(str(exc)) from exc
        return 'ineligible'
    raise exc


def pending_rows(db, retry_failed=False):
    predicate = "state IN ('pending','failed_retryable')" if retry_failed else "state='pending'"
    after = None
    while True:
        sql = 'SELECT * FROM samples WHERE ' + predicate
        params = ()
        if after is not None:
            sql += ' AND (route,frame) > (?,?)'; params = after
        batch = db.execute(sql + ' ORDER BY route,frame LIMIT 256', params).fetchall()
        if not batch: return
        for row in batch: yield row
        after = (batch[-1]['route'], batch[-1]['frame'])


def write_target(config, row, producer, reader, builder, native, provider, used):
    """Build with the existing four-task builder and frozen serializer contract."""
    import torch
    spec = config['target_root']; sha = digest(producer)
    ref = reader.index.ref(row['frame']); root = Path(config['dataset_root']).resolve()
    sources, mapped, future = core.source_identity(reader, ref, native, provider, root)
    current = core.current_annotation_source(sources, row['route'], row['frame'])
    reference = core.reference_pose(reader.read(ref), native)
    if row['sources_json'] and (digest(json.loads(row['sources_json'])) != digest(sources)
                                or digest(json.loads(row['map_json'])) != digest(mapped)
                                or digest(json.loads(row['reference_json'])) != digest(reference)):
        raise StoragePause('Pinned target source/reference changed: ' + row['sample_id'])
    def capacity(size):
        if used + size > spec['budget_gib'] * GIB or core.storage_check(spec, True) - size < spec['minimum_free_gib'] * GIB:
            raise StoragePause('Target cache budget/free-space threshold')
    capacity(1024 ** 2)
    relative = str(Path(sha) / row['sample_id'][:2] / (row['sample_id'] + '.pt'))
    dest = safe_path(spec, relative)
    # Recover a completed atomic rename that preceded the SQLite commit. Never
    # overwrite an artifact of unknown identity, even after an interrupted run.
    if dest.exists():
        cached = dict(row, current_anno_sha256=current['sha256'], reference_json=json.dumps(reference),
                      sources_json=json.dumps(sources), map_json=json.dumps(mapped), relative_path=relative,
                      sha256=file_digest(dest), bytes=dest.stat().st_size)
        payload = core.load_target_row(cached, spec, sha)
        return dict(cached, state='complete', error_code=None, error=None, updated=time.time()), payload['provenance']['bev']['class_pixel_counts'], 0
    result = builder.build_targets(ref); core.validate_targets(result.tensors)
    if digest(core.source_identity(reader, ref, native, provider, root)) != digest((sources, mapped, future)):
        raise StoragePause('Target sources changed during build')
    payload = {'schema': core.SCHEMA, 'sample_id': row['sample_id'], 'dataset_id': config['dataset_id'],
               'route': row['route'], 'frame': row['frame'], 'target_producer_sha256': sha,
               'reference_frame': row['frame'], 'reference_pose': reference,
               'coordinate_contract': producer['contract']['coordinates'], 'sources': sources, 'map': mapped,
               'current_annotation_sha256': current['sha256'], 'future_frame_ids': future,
               'targets': result.tensors, 'provenance': result.provenance}
    stream = io.BytesIO(); torch.save(payload, stream); data = stream.getvalue()
    capacity(len(data) + 65536)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + f'.{os.getpid()}.partial')
    try:
        with temp.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        core.storage_check(spec, True); os.replace(temp, dest); directory_sync(dest.parent)
    finally:
        core.storage_check(spec, True); temp.unlink(missing_ok=True)
    cached = dict(row, state='complete', current_anno_sha256=current['sha256'], reference_json=json.dumps(reference),
                  sources_json=json.dumps(sources), map_json=json.dumps(mapped), relative_path=relative,
                  sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), error_code=None, error=None, updated=time.time())
    restored = core.load_target_row(cached, spec, sha)
    if any(not torch.equal(result.tensors[k], restored['targets'][k]) for k in result.tensors):
        raise StoragePause('Target readback differs')
    return cached, result.provenance['bev']['class_pixel_counts'], max(dest.stat().st_size, dest.stat().st_blocks * 512)


def generate(config, state, limit=None, retry_failed=False, discover_only=False):
    import fcntl
    import signal
    import torch
    from tools.b2d_adapter import FrameReader
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1TargetBuilder
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration, bev_provider
    state = Path(state); state.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); db = None; stopped = False
    counts = {'processed': 0, 'success': 0, 'ineligible': 0, 'failed_retryable': 0}
    status = {'pid': os.getpid(), 'state': 'initializing', 'counts': counts}
    def stop(*_):
        nonlocal stopped
        stopped = True
    def should_stop(): return stopped or (state / 'stop.request').exists()
    def update():
        elapsed = time.monotonic() - started
        status.update(updated=time.time(), seconds=elapsed, success_per_minute=counts['success'] * 60 / max(elapsed, .001))
        if db is not None: status['index'] = summary(db)
        durable_json(state / 'status.json', status)
    with (state / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        try:
            torch.set_num_threads(config.get('threads', 1))
            native, model = native_configuration(config), model_configuration(config)
            producer = core.producer_identity(config, model, native); sha = digest(producer)
            spec = config['target_root']; core.storage_check(spec, True)
            root = Path(spec['path']); root.mkdir(parents=True, exist_ok=True)
            db = open_index(state / 'index.sqlite')
            pin(db, 'producer', producer); pin(db, 'dataset_id', config['dataset_id'])
            pin(db, 'job_implementation', {'version': DISCOVERY_VERSION, 'sha256': file_digest(Path(__file__))})
            pin(db, 'target_root', spec); pin(db, 'dataset_root', str(Path(config['dataset_root']).resolve()))
            marker = root / 'target_root.json'
            identity = {'schema': core.SCHEMA, 'state_id': str(state.resolve()), 'root_id': spec['root_id']}
            if marker.exists():
                if json.loads(marker.read_text()) != identity: raise StoragePause('Target root belongs to another index')
            else:
                if any(root.iterdir()): raise StoragePause('Unmarked nonempty target root')
                durable_json(marker, identity)
            namespace = root / sha
            if namespace.exists():
                for p in namespace.rglob('*.partial'): p.unlink()
            status.update(state='discovering', producer_sha256=sha); update()
            if not discover(db, config, native, sha, should_stop):
                status['state'] = 'stopped'; return status
            if discover_only:
                status['state'] = 'discovered'; return status
            used = sum(max(p.stat().st_size, p.stat().st_blocks * 512) for p in root.rglob('*') if p.is_file())
            provider = bev_provider(dict(config, bev_cache_dir=None))  # In-process Town LRU, no per-frame BEV cache.
            rows = pending_rows(db, retry_failed)
            current_route = None; status['state'] = 'running'; update()
            with (state / 'attempts.jsonl').open('a', buffering=1) as journal:
                for raw_row in rows:
                    if should_stop(): status['state'] = 'stopped'; break
                    if limit is not None and counts['processed'] >= limit: status['state'] = 'batch_complete'; break
                    row = dict(raw_row)
                    status.update(last_sample_id=row['sample_id'], current_route=row['route'], current_frame=row['frame'])
                    try:
                        if current_route != row['route']:
                            reader = FrameReader(AnnotationIndex.from_dataset_root(config['dataset_root'], row['route']))
                            builder = DiffusionDriveV1TargetBuilder(reader, native, model, provider)
                            current_route = row['route']
                        cached, pixels, allocated = write_target(config, row, producer, reader, builder, native, provider, used)
                        columns = list(cached)
                        db.execute('INSERT OR REPLACE INTO samples(' + ','.join(columns) + ') VALUES(' + ','.join('?' for _ in columns) + ')', list(cached.values()))
                        db.execute('INSERT OR REPLACE INTO bev_counts VALUES(?,?)', (row['sample_id'], json.dumps(pixels)))
                        db.commit(); used += allocated; counts['success'] += 1
                        record = event(row, 'complete', sha)
                    except (AdapterError, OSError) as exc:
                        classification = failure_state(exc)
                        code = exc.code if isinstance(exc, AdapterError) else 'temporary_file_access'
                        db.execute('UPDATE samples SET state=?,error_code=?,error=?,updated=? WHERE sample_id=?',
                                   (classification, code, str(exc), time.time(), row['sample_id']))
                        db.commit(); counts[classification] += 1
                        record = event(row, classification, sha, code, str(exc)); status['last_error'] = record
                    journal.write(json.dumps(record) + '\n'); print(json.dumps(record), flush=True)
                    counts['processed'] += 1; status['cache_bytes'] = used
                    if counts['processed'] % 10 == 0 or limit is not None: update()
                else: status['state'] = 'complete'
        except Exception as exc:
            status.update(state='paused', last_error={'code': type(exc).__name__, 'reason': str(exc)})
            import traceback
            traceback.print_exc()
        finally:
            update()
            if db is not None: db.close()
    return status


def alignment(config, target_db):
    """Join metadata only; same current annotation + coordinate contract fixes pose.

    Both producers use the same frozen FrameReader/rear-axle geometry on the same
    SHA-verified annotation. No camera/point cloud or tensor files are decoded.
    """
    from tools.training.diffusiondrive_v1.features_cache import feature_producer
    from tools.training.diffusiondrive_v1.feature_jobs import config_and_state, readonly_db
    collection = json.loads((core.ROOT / config['features_collection']).read_text())
    expected = digest(feature_producer(config))
    if collection['producer_sha256'] != expected: raise StoragePause('Feature collection identity differs')
    producer = json.loads(target_db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
    fp = feature_producer(config)
    if digest(fp['native_policy']['parameters']['geometry']) != digest(producer['geometry']['parameters']):
        raise StoragePause('Feature/Target rear-axle coordinate contract differs')
    targets = {r['sample_id']: dict(r) for r in target_db.execute("SELECT * FROM samples WHERE state='complete'")}
    result = {'targets_complete': len(targets), 'paired': 0, 'awaiting_feature': len(targets),
              'feature_only': 0, 'invalid_identity': 0, 'conflicts': []}
    seen = set()
    for job in collection['jobs']:
        cfg, state = config_and_state(core.ROOT / job['config'])
        db = readonly_db(state / 'index.sqlite')
        if cfg['dataset_id'] != config['dataset_id'] or digest(json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])) != expected:
            db.close(); raise StoragePause('Feature job identity differs')
        for r in db.execute("SELECT * FROM samples WHERE state='complete'"):
            sid = r['sample_id']
            if sid in seen: db.close(); raise StoragePause('Duplicate Features Sample ID across jobs')
            seen.add(sid)
            t = targets.get(sid)
            if t is None: result['feature_only'] += 1; continue
            result['awaiting_feature'] -= 1
            valid = (sid == sample_id(config['dataset_id'], r['route'], r['frame'])
                     and t['dataset_id'] == config['dataset_id'] and t['route'] == r['route'] and t['frame'] == r['frame'])
            try:
                fsource = core.current_annotation_source(json.loads(r['source_json']), r['route'], r['frame'])
                tsource = core.current_annotation_source(json.loads(t['sources_json']), t['route'], t['frame'])
                valid = valid and fsource['sha256'] == t['current_anno_sha256'] == tsource['sha256']
            except (AdapterError, ValueError, TypeError): valid = False
            if valid: result['paired'] += 1
            else:
                result['invalid_identity'] += 1; result['conflicts'].append(sid)
        db.close()
    return result


def export(config, state):
    state = Path(state)
    db = sqlite3.connect('file:' + str((state / 'index.sqlite').resolve()) + '?mode=ro', uri=True); db.row_factory = sqlite3.Row
    # One target snapshot keeps report denominators consistent while writer runs.
    db.execute('BEGIN')
    producer = json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
    accepted = [dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY route,frame")]
    durable_json(state / 'target_manifest.json', {'schema': core.SCHEMA, 'producer': producer, 'dataset_id': config['dataset_id'],
                 'root_id': config['target_root']['root_id'], 'accepted': accepted})
    with (state / 'rejections.jsonl').open('w') as journal:
        for row in db.execute("SELECT * FROM samples WHERE state IN ('ineligible','failed_retryable') ORDER BY route,frame"):
            record = event(row, row['state'], digest(producer), row['error_code'], row['error'])
            record['timestamp'] = row['updated']
            journal.write(json.dumps(record) + '\n')
    totals = [0] * 7; contains = {'vehicle': 0, 'pedestrian': 0, 'static': 0}
    for row in db.execute('SELECT pixels_json FROM bev_counts'):
        pixels = json.loads(row[0])
        for i, value in enumerate(pixels): totals[i] += value
        for key, i in [('vehicle',5), ('pedestrian',6), ('static',4)]:
            contains[key] += int(pixels[i] > 0)
    routes = {}
    towns = {}
    for route, classification, count in db.execute('SELECT route,state,COUNT(*) FROM samples GROUP BY route,state'):
        routes.setdefault(route, {})[classification] = count
        town = re.search(r'_(Town\d+(?:HD)?)_', route).group(1)
        bucket = towns.setdefault(town, {}); bucket[classification] = bucket.get(classification, 0) + count
    for buckets in (routes, towns):
        for counts in buckets.values(): counts['acceptance_rate'] = counts.get('complete', 0) / sum(counts.values())
    result = {**summary(db), 'alignment': alignment(config, db), 'routes': routes, 'towns': towns,
              'bev_pixel_counts': totals, 'bev_pixel_fractions': [v / max(sum(totals), 1) for v in totals], 'contains': contains,
              'route_discovery_errors': [dict(r) for r in db.execute("SELECT * FROM discoveries WHERE json_extract(summary,'$.error_code') IS NOT NULL")]}
    durable_json(state / 'generation_report.json', result); db.close()
    return result
