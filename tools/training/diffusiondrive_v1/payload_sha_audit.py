"""Resumable, read-only SHA-256 audit of frozen Feature/Target payload files.

The production readers already hash every file they load; this tool exists for the
*scheduled, resumable* full-cache audit that cannot run inside a training job.
It re-computes each payload's size and SHA-256 and compares them with the frozen
index, recording one row per ``sample_id`` in a progress SQLite database so an
interrupted run resumes exactly where it stopped.

Safety rules:
* read-only: no cache writes, no index writes, no generation;
* storage health is checked once at start, then every row stops on EIO;
* default one worker, sequential file reads;
* a row is only marked ``ok`` when size and SHA both match the frozen index.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = 'payload-sha-audit-v1'
CHUNK = 1 << 20


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def open_progress(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS audit ('
               'sample_id TEXT PRIMARY KEY, source TEXT, status TEXT, expected_sha TEXT, '
               'actual_sha TEXT, expected_bytes INTEGER, actual_bytes INTEGER, '
               'error TEXT, seconds REAL, updated REAL)')
    db.commit()
    return db


def index_rows(kind, manifest):
    if kind == 'target':
        for name, source in manifest['target_sources'].items():
            spec = source['target_root']
            producer = source['producer_sha256']
            db = sqlite3.connect((Path(source['state']) / 'index.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
            db.row_factory = sqlite3.Row
            for row in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY sample_id"):
                yield dict(row, _source=name, _spec=spec, _producer=producer)
            db.close()
    else:
        from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
        for name, source in manifest['feature_sources'].items():
            roots = effective_roots(manifest, source)
            db = sqlite3.connect((Path(source['state']) / 'index.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
            db.row_factory = sqlite3.Row
            for row in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY sample_id"):
                record = dict(row, _source=name, _producer=source['producer_sha256'])
                record['_spec'] = roots[record['root_id']]
                yield record
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['target', 'feature'], required=True)
    parser.add_argument('--state', type=Path, required=True, help='progress directory (SQLite lives here)')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight/baseline_pssd_manifest.json')
    parser.add_argument('--limit', type=int, default=0, help='0 = full remaining queue')
    parser.add_argument('--report-every', type=int, default=2000)
    args = parser.parse_args()

    from tools.b2d_adapter.errors import AdapterError
    from tools.training.diffusiondrive_v1.features_cache import StoragePause, active_mount, safe_path
    from tools.training.diffusiondrive_v1.target_cache import storage_check

    manifest = json.loads(args.manifest.read_text())
    state = args.state if args.state.is_absolute() else ROOT / args.state
    state.mkdir(parents=True, exist_ok=True)
    progress_path = state / f'{args.kind}_audit.sqlite'
    db = open_progress(progress_path)

    specs = {}
    if args.kind == 'target':
        for name, source in manifest['target_sources'].items():
            storage_check(source['target_root'])
            specs[name] = source['target_root']
    else:
        from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
        for name, source in manifest['feature_sources'].items():
            roots = effective_roots(manifest, source)
            for root in roots.values():
                active_mount(root)
            specs[name] = roots

    rows = list(index_rows(args.kind, manifest))
    done = {r[0] for r in db.execute('SELECT sample_id FROM audit WHERE status="ok"')}
    pending = [row for row in rows if row['sample_id'] not in done]
    if args.limit:
        pending = pending[:args.limit]
    started = time.monotonic()
    counts = Counter()
    failures = []
    stopped = None
    for number, row in enumerate(pending, 1):
        spec = row['_spec']
        begin = time.monotonic()
        try:
            path = safe_path(spec, row['relative_path'])
            stat = path.stat()
            actual_bytes = int(stat.st_size)
            actual_sha = sha256_file(path)
            status = 'ok' if (actual_bytes == int(row['bytes']) and actual_sha == row['sha256']) else 'mismatch'
            error = None
        except Exception as exc:  # noqa: BLE001 - recorded verbatim, then may abort
            path = None
            actual_bytes = None
            actual_sha = None
            status = 'error'
            error = type(exc).__name__ + ': ' + str(exc)[:300]
            if isinstance(exc, (StoragePause, OSError)):
                stopped = error
        elapsed = round(time.monotonic() - begin, 4)
        db.execute('INSERT OR REPLACE INTO audit VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (row['sample_id'], row['_source'], status, row['sha256'], actual_sha,
                    int(row['bytes']), actual_bytes, error, elapsed, time.time()))
        counts[status] += 1
        if status != 'ok':
            failures.append({'sample_id': row['sample_id'], 'source': row['_source'],
                             'relative_path': row['relative_path'], 'status': status, 'error': error})
        if number % 500 == 0:
            db.commit()
        if number % args.report_every == 0:
            print(json.dumps({'phase': args.kind, 'checked': number, 'of': len(pending),
                              'ok': counts['ok'], 'mismatch': counts['mismatch'],
                              'error': counts['error'],
                              'seconds': round(time.monotonic() - started, 1)}), flush=True)
        if stopped:
            break
    db.commit()
    summary = {
        'schema': VERSION, 'kind': args.kind,
        'manifest': str(args.manifest), 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'progress_db': str(progress_path),
        'index_rows': len(rows), 'already_verified': len(done), 'attempted_this_run': len(pending) if not stopped else counts.total(),
        'counts_this_run': dict(counts),
        'failures': failures[:200], 'failure_count': len(failures),
        'stopped_on_io_error': stopped,
        'remaining_after_run': len(rows) - len(done) - counts['ok'],
        'seconds': round(time.monotonic() - started, 1),
        'per_item_seconds_median': None,
        'note': 'ok means size AND sha256 equal the frozen index; readers still validate on every load',
    }
    durations = [r[0] for r in db.execute('SELECT seconds FROM audit WHERE seconds IS NOT NULL ORDER BY seconds')]
    if durations:
        summary['per_item_seconds_median'] = round(durations[len(durations) // 2], 4)
    db.close()
    (state / f'{args.kind}_audit_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('kind', 'index_rows', 'already_verified',
                                              'counts_this_run', 'failure_count',
                                              'stopped_on_io_error', 'remaining_after_run',
                                              'per_item_seconds_median', 'seconds')}, ensure_ascii=False))
    return 2 if stopped else (1 if failures else 0)


if __name__ == '__main__':
    sys.exit(main())
