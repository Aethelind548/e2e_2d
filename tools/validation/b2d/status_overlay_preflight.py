"""Independent Status Overlay preflight.

This tool is intentionally read-only with respect to the running Route Preview
generation, the production Feature/Target caches, manifests and training code.
It validates an experimental ``status_overlay_manifest.jsonl`` against the
frozen pair index and panel identity, and reports whether the production
readers currently recognize the overlay format.

No overlay row is admitted to training by this tool.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

DEFAULT_PANEL = ROOT / 'outputs/diffusiondrive_v1/scratch_attribution/20261022_142557_panel/behavior_panel.json'
DEFAULT_OVERLAY = ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261022_status_upstream/status_overlay_manifest.jsonl'
DEFAULT_READER_SOURCES = (
    ROOT / 'tools/training/diffusiondrive_v1/paired_dataset.py',
    ROOT / 'tools/training/diffusiondrive_v1/cache_only_reader.py',
)
COMMAND_ORDER = ('Left', 'Straight', 'Right', 'Unknown')
VALIDITIES = {'VALID', 'NOT_VERIFIED', 'INVALID', 'AMBIGUOUS', 'FAILED', 'MISSING'}


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finite_vector(value, length):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None, 'not_numeric'
    if array.shape != (length,):
        return None, f'wrong_shape={array.shape}'
    if not np.isfinite(array).all():
        return None, 'nonfinite'
    return array, None


def one_hot(value):
    if value is None:
        return False, 'missing'
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,):
        return False, f'wrong_shape={array.shape}'
    if not np.isfinite(array).all():
        return False, 'nonfinite'
    if not np.isin(array, (0.0, 1.0)).all() or abs(float(array.sum()) - 1.0) > 1e-9:
        return False, 'not_one_hot'
    return True, None


def nested_preview_validity(row):
    preview = ((row.get('diagnostic_metadata') or {}).get('upstream') or {}).get('preview')
    if not isinstance(preview, dict):
        return None
    return preview.get('preview_validity')


def check_rows(rows, panel_rows, pair_index, reader_sources):
    counts = defaultdict(lambda: {'passed': 0, 'failed': 0, 'failures': []})
    seen = set()
    panel_by_sid = {row['sample_id']: row for row in panel_rows}
    index_by_sid = {}
    duplicate_index = set()
    for item in pair_index:
        sid = item['sample_id']
        if sid in index_by_sid:
            duplicate_index.add(sid)
        index_by_sid[sid] = item

    def record(name, ok, sid, reason=None):
        bucket = counts[name]
        if ok:
            bucket['passed'] += 1
        else:
            bucket['failed'] += 1
            bucket['failures'].append({'sample_id': sid, 'reason': reason})

    for row in rows:
        sid = row.get('sample_id')
        record('sample_id_nonempty', isinstance(sid, str) and bool(sid.strip()), sid,
               'missing_or_empty_sample_id')
        if isinstance(sid, str):
            record('sample_id_unique', sid not in seen, sid, 'duplicate_sample_id')
            seen.add(sid)
        identity = row.get('sample_identity') or {}
        record('sample_identity_present', isinstance(identity, dict) and bool(identity), sid,
               'missing_sample_identity')
        route = identity.get('route', identity.get('route_id'))
        frame = identity.get('frame', identity.get('frame_id'))
        record('identity_route_frame_present', bool(route) and frame is not None, sid,
               'missing_route_or_frame')
        panel = panel_by_sid.get(sid)
        record('overlay_in_frozen_panel', panel is not None, sid, 'sample_not_in_panel')
        index = index_by_sid.get(sid)
        record('overlay_index_one_to_one', index is not None and sid not in duplicate_index, sid,
               'sample_not_in_pair_index' if index is None else 'duplicate_pair_index')
        if index is not None:
            same = (index.get('route') == route and int(index.get('frame')) == int(frame)
                    and (identity.get('split') is None or identity.get('split') == index.get('split')))
            record('route_frame_split_match', same, sid, 'route_frame_or_split_mismatch')
            source = row.get('source_identity') or {}
            feature_row = (panel or {}).get('row', {})
            record('feature_target_sha_match',
                   source.get('feature_sha256') == feature_row.get('feature_sha')
                   and source.get('target_sha256') == feature_row.get('target_sha'),
                   sid, 'feature_or_target_sha_mismatch')
        expected = (panel or {}).get('row', {})
        source = row.get('source_identity') or {}
        record('manifest_source_hash_present', bool(source.get('manifest_sha256')), sid,
               'manifest_sha_missing')
        record('feature_producer_hash_present', bool(source.get('feature_producer_sha256')), sid,
               'feature_producer_sha_missing')
        record('target_producer_hash_present', bool(source.get('target_producer_sha256')), sid,
               'target_producer_sha_missing')

        status = row.get('status8')
        validity = row.get('status_validity')
        record('status_validity_known', validity in VALIDITIES, sid, 'unknown_status_validity')
        record('status_dtype_declared_float32', row.get('status_dtype') == 'float32', sid,
               'status_dtype_not_float32')
        record('status_order_v1', row.get('status_order') == list(COMMAND_ORDER) + ['vx', 'vy', 'ax', 'ay'],
               sid, 'status_order_mismatch')
        if status is None:
            record('status8_shape_float32_8', validity == 'INVALID', sid,
                   'status8_null_for_noninvalid' if validity != 'INVALID' else None)
            record('status_command_one_hot', validity == 'INVALID', sid, 'null_without_invalid')
            record('motion4_a0_match', True, sid)
        else:
            array, error = finite_vector(status, 8)
            record('status8_shape_float32_8', error is None, sid, error)
            if array is not None:
                as32 = array.astype(np.float32)
                record('status8_float32_representable',
                       np.allclose(as32.astype(np.float64), array, rtol=0, atol=1e-6),
                       sid, 'not_float32_representable')
                command_ok, command_reason = one_hot(array[:4])
                record('status_command_one_hot', command_ok, sid, command_reason)
                old = row.get('old_status8')
                old_array, old_error = finite_vector(old, 8)
                record('old_status8_present', old_error is None, sid, old_error or 'missing')
                if old_array is not None:
                    record('motion4_a0_match',
                           np.allclose(array[4:], old_array[4:], rtol=0, atol=1e-6),
                           sid, 'motion4_differs_from_frozen_old_status')
                record('invalid_has_no_status8', validity != 'INVALID', sid,
                       'invalid_row_has_status8')
            else:
                record('old_status8_present', False, sid, 'invalid_status8')
                record('motion4_a0_match', False, sid, 'invalid_status8')
        metadata = row.get('diagnostic_metadata') or {}
        upstream = metadata.get('upstream') or {}
        motion_method = upstream.get('motion_method', metadata.get('motion_method'))
        preview_validity = nested_preview_validity(row)
        record('preview_validity_present', preview_validity in VALIDITIES, sid,
               'missing_or_unknown_preview_validity')
        record('navigation_source_present', isinstance(row.get('navigation_source'), str)
               and bool(row['navigation_source'].strip()), sid, 'missing_navigation_source')
        record('fallback_source_explicit', row.get('fallback_source') in (None, 'NEAR_FALLBACK'), sid,
               'invalid_fallback_source')
        record('a0_motion_method', motion_method == 'A0_FROZEN', sid,
               'motion_method_not_a0_frozen')
        record('a2v_not_enabled', 'A2V' not in json.dumps(metadata).upper(), sid, 'a2v_marker_present')
        record('training_not_admitted', row.get('training_admitted') is False, sid,
               'training_admitted_is_true')
        record('training_contract_experimental',
               row.get('training_contract') == 'NOT_ADMITTED_EXPERIMENT_ONLY', sid,
               'unexpected_training_contract')
        if validity == 'VALID' and row.get('navigation_source') == 'NOT_VERIFIED':
            record('valid_status_has_source', False, sid, 'valid_status_with_not_verified_source')
        else:
            record('valid_status_has_source', True, sid)

    reader_text = '\n'.join(Path(path).read_text(errors='replace') for path in reader_sources
                            if Path(path).is_file())
    reader_overlay_refs = reader_text.lower().count('overlay')
    counts['reader_overlay_recognition'] = {
        'passed': 0,
        'failed': 0,
        'failures': [],
        'reader_overlay_refs': reader_overlay_refs,
        'reader_supports_overlay': bool(reader_overlay_refs),
    }
    return counts


def build_report(args):
    panel = json.loads(args.panel.read_text())
    manifest = json.loads(Path(panel['manifest']).read_text())
    if sha256(Path(panel['manifest'])) != panel.get('manifest_sha256'):
        raise ValueError('frozen manifest SHA mismatch')
    rows = load_jsonl(args.overlay)
    panel_rows = [dict(row, split=split) for split in ('validation', 'train') for row in panel[split]]
    pair_index_path = Path(manifest['pair_index'])
    if sha256(pair_index_path) != manifest.get('index_sha256'):
        raise ValueError('pair index SHA mismatch')
    import sqlite3
    with sqlite3.connect(pair_index_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        pair_index = [dict(row) for row in db.execute('SELECT * FROM pairs')]
    counts = check_rows(rows, panel_rows, pair_index, args.reader_source)
    total = len(rows)
    failures = []
    for name, bucket in counts.items():
        for item in bucket.get('failures', []):
            failures.append({'check': name, **item})
    report = {
        'schema': 'status-overlay-preflight-v1',
        'overlay': str(args.overlay.resolve()),
        'overlay_sha256': sha256(args.overlay),
        'panel': str(args.panel.resolve()),
        'panel_sha256': sha256(args.panel),
        'manifest': str(Path(panel['manifest']).resolve()),
        'manifest_sha256': sha256(Path(panel['manifest'])),
        'pair_index': str(pair_index_path.resolve()),
        'pair_index_sha256': sha256(pair_index_path),
        'overlay_rows': total,
        'panel_rows': len(panel_rows),
        'reader_sources': [str(path) for path in args.reader_source],
        'reader_overlay_refs': counts['reader_overlay_recognition'].get('reader_overlay_refs', 0),
        'checks': {name: {k: v for k, v in bucket.items() if k != 'failures'}
                   for name, bucket in counts.items()},
        'failures': failures,
        'failure_count': len(failures),
        'status_validity_counts': dict(Counter(row.get('status_validity') for row in rows)),
        'navigation_source_counts': dict(Counter(row.get('navigation_source') for row in rows)),
        'preview_validity_counts': dict(Counter(nested_preview_validity(row) for row in rows)),
        'status8_present': sum(row.get('status8') is not None for row in rows),
        'status8_null': sum(row.get('status8') is None for row in rows),
        'legal_unknown_count': sum(
            row.get('status8') is not None and one_hot(row['status8'][:4])[0]
            and float(row['status8'][3]) == 1.0 for row in rows),
        'near_fallback_count': sum(row.get('fallback_source') == 'NEAR_FALLBACK' for row in rows),
        'not_verified_count': sum(row.get('status_validity') == 'NOT_VERIFIED' for row in rows),
        'invalid_count': sum(row.get('status_validity') == 'INVALID' for row in rows),
        'valid_count': sum(row.get('status_validity') == 'VALID' for row in rows),
        'training_admitted_count': sum(row.get('training_admitted') is True for row in rows),
        'invalid_samples': [{
            'sample_id': row.get('sample_id'),
            'route': (row.get('sample_identity') or {}).get('route'),
            'frame': (row.get('sample_identity') or {}).get('frame'),
            'status_validity': row.get('status_validity'),
            'preview_validity': nested_preview_validity(row),
            'reason': (row.get('diagnostic_metadata') or {}).get('upstream', {}).get(
                'preview', {}).get('failure_reason'),
        } for row in rows if row.get('status_validity') == 'INVALID'],
        'near_fallback_samples': [row.get('sample_id') for row in rows
                                  if row.get('fallback_source') == 'NEAR_FALLBACK'],
        'training_admitted_samples': [row.get('sample_id') for row in rows
                                      if row.get('training_admitted') is True],
        'reader_accepts_overlay': bool(counts['reader_overlay_recognition'].get('reader_overlay_refs')),
        'admission': 'REJECTED_BY_PREFLIGHT' if not counts['reader_overlay_recognition'].get('reader_overlay_refs') else 'NOT_PROVEN',
        'source_boundary': ('Read-only overlay/panel/manifest/index preflight; '
                            'no Route Preview generation, cache, Feature Producer, '
                            'model, checkpoint or training process is modified.'),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument('--panel', type=Path, default=DEFAULT_PANEL)
    parser.add_argument('--reader-source', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.reader_source:
        args.reader_source = list(DEFAULT_READER_SOURCES)
    if args.output.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    args.output.mkdir(parents=True)
    report = build_report(args)
    (args.output / 'status_overlay_preflight.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    fields = ['check', 'sample_id', 'reason']
    with (args.output / 'status_overlay_preflight_failures.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for failure in report['failures']:
            writer.writerow({key: failure.get(key) for key in fields})
    print(json.dumps({'overlay_rows': report['overlay_rows'],
                      'failure_count': report['failure_count'],
                      'validity': report['status_validity_counts'],
                      'reader_accepts_overlay': report['reader_accepts_overlay'],
                      'admission': report['admission'],
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
