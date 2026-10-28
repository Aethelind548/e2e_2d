"""Targets-only foreground/background CPU generator; independent of Features jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')


def main():
    from tools.training.diffusiondrive_v1.target_cache import generate, storage_check
    from tools.training.diffusiondrive_v1.features_cache import durable_json
    from tools.training.diffusiondrive_v1.feature_service import alive, process_identity
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run', 'start', 'resume', 'compact', 'status', 'stop', 'logs', 'export', 'scan', 'alignment'])
    p.add_argument('--config', type=Path, default=ROOT / 'tools/training/diffusiondrive_v1/paired_check.json')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--all-completed-features', action='store_true', help='Snapshot all currently completed Feature indexes, ignoring small config selection')
    mode.add_argument('--all-local-routes', action='store_true', help='Discover original annotations independently of Features')
    p.add_argument('--limit', type=int)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--lines', type=int, default=15)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); state = ROOT / cfg['target_state_dir']
    local = args.all_local_routes or cfg.get('target_discovery') == 'all-local-routes'
    if local and args.all_completed_features: p.error('Full local task cannot use Features-only discovery')
    if args.limit is not None and args.limit < 1: p.error('--limit must be positive')
    state.mkdir(parents=True, exist_ok=True)
    if args.action == 'run':
        from tools.training.diffusiondrive_v1.target_storage import CODEC,codec_scope,storage_preflight
        storage_preflight(cfg,state)
        with codec_scope(write=cfg.get('target_storage_encoding')==CODEC):
            if local:
                from tools.training.diffusiondrive_v1.target_jobs import generate as full_generate
                result = full_generate(cfg, state, args.limit, args.retry_failed)
            else:
                result = generate(cfg, state, args.all_completed_features, args.limit, args.retry_failed)
        print(json.dumps(result), flush=True)
        return 2 if result['state'] == 'paused' else 0
    if args.action == 'scan':
        if not local: p.error('scan requires --all-local-routes or full task config')
        from tools.training.diffusiondrive_v1.target_jobs import generate as full_generate
        result = full_generate(cfg, state, discover_only=True)
        print(json.dumps(result)); return 2 if result['state'] == 'paused' else 0
    if args.action in ('start', 'resume', 'compact'):
        from tools.training.diffusiondrive_v1.target_storage import storage_preflight
        if args.action!='compact':storage_preflight(cfg,state)
        import fcntl
        with (state / 'launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if alive(state): raise SystemExit('Target worker already active')
            with (state / 'writer.lock').open('a') as writer:
                try: fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: raise SystemExit('Foreground target writer active')
            storage_check(cfg['target_root'], True)
            (state / 'stop.request').unlink(missing_ok=True)
            cmd = [cfg['python'], '-u', '-m', 'tools.training.diffusiondrive_v1.target_service', 'run', '--config', str(args.config.resolve())]
            if args.action=='compact':
                from tools.training.diffusiondrive_v1.target_storage import CODEC
                if cfg.get('target_storage_encoding')!=CODEC:raise SystemExit('Enable compact codec in config first')
                cmd=[cfg['python'],'-u','-m','tools.training.diffusiondrive_v1.target_storage','compact',
                     '--config',str(args.config.resolve()),'--resume']
            else:
                if local: cmd += ['--all-local-routes']
                if args.all_completed_features: cmd += ['--all-completed-features']
            if args.limit is not None: cmd += ['--limit', str(args.limit)]
            if args.retry_failed: cmd += ['--retry-failed']
            with (state / 'worker.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
                process = subprocess.Popen(cmd, cwd=ROOT, stdin=null, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, close_fds=True, env=dict(os.environ))
            identity = process_identity(process.pid)
            if identity is None: raise SystemExit('Target worker exited during launch')
            durable_json(state / 'pid.json', identity)
            launch = {'pid': process.pid, 'command': cmd, 'started': time.time(), 'log': str(state / 'worker.log')}
            durable_json(state / 'launch.json', launch); print(json.dumps(launch)); return 0
    if args.action == 'stop':
        durable_json(state / 'stop.request', {'requested': time.time()}); print('Stop requested after current commit'); return 0
    if args.action == 'logs':
        from collections import deque
        path = state / 'worker.log'
        if path.exists():
            with path.open(errors='replace') as f: print(''.join(deque(f, maxlen=args.lines)), end='')
        return 0
    if args.action == 'export':
        if local:
            from tools.training.diffusiondrive_v1.target_jobs import export
            result = export(cfg, state)
            print(json.dumps({'index': {k: result[k] for k in ['candidates','states','routes_discovered']},
                              'alignment': result['alignment'], 'manifest': str(state / 'target_manifest.json')})); return 0
        from tools.training.diffusiondrive_v1.target_cache import SCHEMA
        import sqlite3
        db = sqlite3.connect('file:' + str((state / 'index.sqlite').resolve()) + '?mode=ro', uri=True); db.row_factory = sqlite3.Row
        producer = json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
        durable_json(state / 'target_manifest.json', {'schema': SCHEMA, 'producer': producer, 'dataset_id': cfg['dataset_id'],
            'root_id': cfg['target_root']['root_id'], 'accepted': [dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY route,frame")]})
        db.close(); print(str(state / 'target_manifest.json')); return 0
    if args.action == 'alignment':
        from tools.training.diffusiondrive_v1.target_jobs import alignment
        import sqlite3
        db = sqlite3.connect('file:' + str((state / 'index.sqlite').resolve()) + '?mode=ro', uri=True); db.row_factory = sqlite3.Row
        result = alignment(cfg, db); db.close()
        durable_json(state / 'alignment.json', result); print(json.dumps(result)); return 0
    status = json.loads((state / 'status.json').read_text()) if (state / 'status.json').exists() else {}
    status['running'] = alive(state)
    if (state / 'compaction_status.json').exists():
        status['compaction'] = json.loads((state / 'compaction_status.json').read_text())
        if status['running'] and status['compaction']['state'] == 'compacting':
            status['state'] = 'compacting'
            status['pid'] = status['compaction']['pid']
    if local and (state / 'index.sqlite').exists():
        from tools.training.diffusiondrive_v1.target_jobs import summary
        import sqlite3
        db = sqlite3.connect('file:' + str((state / 'index.sqlite').resolve()) + '?mode=ro', uri=True)
        status['index'] = summary(db); db.close()
    if not status['running'] and status.get('state') in ('running','discovering','initializing'):
        status['state'] = 'interrupted; resume required'
    print(json.dumps(status, indent=2)); return 0


if __name__ == '__main__':
    from tools.training.diffusiondrive_v1.features_cache import StoragePause
    try:
        raise SystemExit(main())
    except StoragePause as exc:
        print('Targets 任务保持安全暂停：' + str(exc), file=sys.stderr)
        raise SystemExit(2)
