"""Foreground features_only worker or a detached, resumable Ubuntu background job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

from tools.training.diffusiondrive_v1.features_cache import (
    durable_json, connect, index_summary, active_mount, CacheStorage, StoragePause)


def process_identity(pid):
    try:
        parts = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if parts[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': parts[19]}
    except (OSError, IndexError):
        return None


def alive(state):
    try:
        stored = json.loads((state / 'pid.json').read_text())
        actual = process_identity(stored['pid'])
        if actual is not None:
            return actual == stored
        # A restricted /proc view can hide the detached host process. The shared
        # writer lock still provides a reliable active-writer check.
        import fcntl
        with (state / 'writer.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        return False
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'start', 'resume', 'status', 'stop', 'logs', 'export', 'verify'])
    parser.add_argument('--config', type=Path, default=ROOT / 'tools/training/diffusiondrive_v1/features_cache.json')
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--limit', type=int, help='Maximum new/rebuilt sample attempts in this invocation')
    parser.add_argument('--prefer-root', help='Validation only; must name a configured cache root; normal order follows config')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--lines', type=int, default=20)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.prefer_root and args.prefer_root not in config['cache_roots']:
        parser.error('--prefer-root must name a configured cache root')
    state = (args.state_dir or ROOT / config['state_dir']).resolve()
    state.mkdir(parents=True, exist_ok=True)
    if args.action in ('run', 'start', 'resume'):
        from tools.training.diffusiondrive_v1.feature_migration import migration_preflight
        migration_preflight(state, config)
    if args.action == 'run':
        from tools.training.diffusiondrive_v1.feature_storage import validate_extension
        validate_extension(config)
        from tools.training.diffusiondrive_v1.feature_queue import run_pending
        result = run_pending(config, state, args.limit, args.prefer_root, args.retry_failed)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 2 if result['state'] == 'paused' else 0
    if args.action == 'verify':
        from tools.training.diffusiondrive_v1.feature_queue import verify_completed
        result = verify_completed(config, state, args.limit)
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result['invalid'] else 0
    if args.action in ('start', 'resume'):
        from tools.training.diffusiondrive_v1.feature_storage import validate_extension
        validate_extension(config)
        import fcntl
        with (state / 'launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            migration_preflight(state, config)
            if config != json.loads(args.config.read_text()):
                raise StoragePause('Features storage config changed during launch; retry resume')
            if alive(state):
                raise SystemExit('Already running: ' + str(state))
            # A manual foreground worker also owns writer.lock.
            with (state / 'writer.lock').open('a') as writer:
                try:
                    fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise SystemExit('Foreground writer is active')
            # fail early if either mount identity or writable state is wrong
            for spec in config['cache_roots'].values():
                active_mount(spec, writing=True)
            (state / 'stop.request').unlink(missing_ok=True)
            executable = config['python']
            command = [executable, '-u', '-m', 'tools.training.diffusiondrive_v1.feature_service', 'run',
                       '--config', str(args.config.resolve()), '--state-dir', str(state)]
            if args.limit is not None:
                command += ['--limit', str(args.limit)]
            if args.prefer_root:
                command += ['--prefer-root', args.prefer_root]
            if args.retry_failed:
                command += ['--retry-failed']
            with (state / 'worker.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
                process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                           stdin=null, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, close_fds=True)
            identity = process_identity(process.pid)
            if identity is None:
                raise SystemExit('Worker exited during launch; inspect worker.log')
            durable_json(state / 'pid.json', identity)
            launch = {'command': command, 'cwd': str(ROOT), 'pid': process.pid,
                      'started': time.time(), 'log': str(state / 'worker.log'), 'detached_session': True}
            durable_json(state / 'launch.json', launch)
            print(json.dumps(launch, ensure_ascii=False))
        return 0
    if args.action == 'stop':
        # A persistent request is observed between samples, after atomic commit.
        durable_json(state / 'stop.request', {'requested': time.time()})
        print(json.dumps({'stop_requested': True, 'running': alive(state), 'state_dir': str(state)}))
        return 0
    if args.action == 'logs':
        path = state / 'worker.log'
        if path.exists():
            from collections import deque
            with path.open(errors='replace') as f:
                print(''.join(deque(f, maxlen=args.lines)), end='')
        return 0
    if args.action == 'export':
        db = connect(state / 'index.sqlite')
        producer = json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
        rows = [dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY route,frame")]
        db.close()
        durable_json(state / 'features_manifest.json', {'schema': 'b2d-v1-features-only-v1', 'producer': producer,
                     'roots': list(config['cache_roots']), 'samples': rows,
                     'supervision_eligibility': 'not evaluated'})
        print(json.dumps({'samples': len(rows), 'manifest': str(state / 'features_manifest.json')}))
        return 0
    status = json.loads((state / 'status.json').read_text()) if (state / 'status.json').exists() else {}
    status['running'] = alive(state)
    if (state / 'index.sqlite').exists():
        db = connect(state / 'index.sqlite')
        status['index'] = index_summary(db)
        db.close()
    disks = {}
    for root_id, spec in config['cache_roots'].items():
        try:
            disks[root_id] = dict(active_mount(spec), path=spec['path'],
                                 cache_bytes=CacheStorage.measure(Path(spec['path'])),
                                 budget_bytes=int(spec['budget_gib'] * 1024 ** 3),
                                 minimum_free_bytes=int(spec['minimum_free_gib'] * 1024 ** 3))
        except Exception as exc:
            disks[root_id] = {'unavailable': str(exc)}
    status['disks'] = disks
    if not status['running'] and status.get('state') == 'running':
        status['state'] = 'interrupted; resume required'
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except StoragePause as exc:
        print('Features 任务保持安全暂停：' + str(exc), file=sys.stderr)
        raise SystemExit(2)
