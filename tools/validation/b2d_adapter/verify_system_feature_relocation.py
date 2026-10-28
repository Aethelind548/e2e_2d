"""Read-only post-remount verification against frozen Feature index and migration journal.

Only the evidence JSON is written. No cache, producer, index or manifest changes.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import time

from tools.training.diffusiondrive_v1.features_cache import (
    active_mount, digest, durable_json, file_digest, load_feature_row, safe_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--physical-mount', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registration_path = args.state / 'storage_registration.json'
    registration = json.loads(registration_path.read_text())
    assert registration['verified'] and registration['source_files_deleted'] == 0
    started = time.monotonic()
    result = {'status': 'running', 'registration_sha256': file_digest(registration_path),
              'verified_files': 0, 'verified_bytes': 0, 'decoded_examples': [],
              'source_files_deleted': 0, 'cache_mutations': 0}
    durable_json(args.output, result)
    try:
        with sqlite3.connect(f'file:{args.state}/verification.sqlite?mode=ro', uri=True) as journal:
            assert journal.execute('select count(*) from files where verified!=1').fetchone()[0] == 0
            for folder, job in registration['jobs'].items():
                with sqlite3.connect(f'file:{folder}/index.sqlite?mode=ro', uri=True) as index:
                    index.row_factory = sqlite3.Row
                    meta = {r['key']: json.loads(r['value']) for r in index.execute('select * from meta')}
                    assert meta['index_id'] == job['index_id']
                    assert digest(meta['producer']) == registration['producer_sha256']
                    for root_id, relocation in job['relocations'].items():
                        old = relocation['from']; recorded = relocation['to']
                        actual = dict(recorded, mountpoint=str(args.physical_mount),
                                      path=str(args.physical_mount / Path(recorded['path']).relative_to(recorded['mountpoint'])))
                        active_mount(actual)
                        assert file_digest(Path(actual['path']) / 'cache_root.json') == relocation['destination_owner_sha256']
                        rows = [dict(r) for r in index.execute(
                            "select * from samples where state='complete' and root_id=? order by relative_path", (root_id,))]
                        assert len(rows) == job['count']
                        assert digest([(r['sample_id'], r['relative_path'], r['sha256'], r['bytes']) for r in rows]) == job['selected_rows_sha256']
                        picks = {0, len(rows)//4, len(rows)//2, len(rows)*3//4, len(rows)-1}
                        for i, row in enumerate(rows):
                            if i % 250 == 0:
                                active_mount(actual); active_mount(old)
                            source = safe_path(old, row['relative_path'])
                            dest = safe_path(actual, row['relative_path'])
                            recorded_dest = safe_path(recorded, row['relative_path'])
                            evidence = journal.execute('select source,destination,sha256,bytes,verified from files where sample_id=?',
                                                       (row['sample_id'],)).fetchone()
                            assert evidence == (str(source), str(recorded_dest), row['sha256'], row['bytes'], 1)
                            assert source.is_file() and not source.is_symlink() and source.stat().st_size == row['bytes']
                            assert dest.is_file() and not dest.is_symlink() and dest.stat().st_size == row['bytes']
                            before = dest.stat()
                            assert file_digest(dest) == row['sha256'], row['sample_id']
                            after = dest.stat()
                            assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
                            result['verified_files'] += 1; result['verified_bytes'] += row['bytes']
                            if i in picks:
                                tensors = load_feature_row(row, {root_id: actual}, registration['producer_sha256'])
                                assert file_digest(source) == row['sha256']
                                result['decoded_examples'].append({'sample_id': row['sample_id'], 'route': row['route'],
                                    'frame': row['frame'], 'tensors': {k: {'shape': list(t.shape), 'dtype': str(t.dtype)} for k,t in tensors.items()}})
                            if (i+1) % 2000 == 0:
                                result.update(seconds=time.monotonic()-started, physical_root=actual)
                                durable_json(args.output, result)
                                print(json.dumps({'verified': result['verified_files'], 'total': registration['verified_files']}), flush=True)
        assert result['verified_files'] == registration['verified_files']
        assert result['verified_bytes'] == registration['verified_bytes']
        result.update(status='passed', seconds=time.monotonic()-started,
                      verification='All destination SHA256/size vs frozen index and prior journal; all sources present/size checked; five source SHA and tensor decodes.')
    except Exception as exc:
        result.update(status='failed', error=repr(exc), seconds=time.monotonic()-started)
        raise
    finally:
        durable_json(args.output, result)
    print(json.dumps({'status': result['status'], 'files': result['verified_files'], 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
