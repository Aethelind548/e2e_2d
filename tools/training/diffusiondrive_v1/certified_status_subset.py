"""Build the navigation certification manifest and the certified Status subset.

The manifest records, for every compared frame, which navigation evidence exists
and why a row is or is not admissible.  The subset builder only copies rows whose
navigation evidence reached ``ONLINE_MATCH_VERIFIED``; today no capture exists, so
it must produce an empty subset and report the blocking reason instead of
fabricating admission.

Rules enforced here:
* rows must exist in the frozen pair index (``sample_id`` identity);
* the 12 geometry-defect routes stay excluded until a new overlay contract;
* A0 Motion4 must be bit-identical between old and new Status;
* missing/duplicate/foreign rows are rejected, never repaired.
"""
import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = 'certified-status-subset-v1'
VERIFIED = 'ONLINE_MATCH_VERIFIED'


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def pair_index(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        return {row['sample_id']: dict(row) for row in db.execute('SELECT * FROM pairs')}
    finally:
        db.close()


def capture_index(capture_dir):
    """Map route id -> captured online plan record (empty when nothing was captured)."""
    index = {}
    if capture_dir is None or not Path(capture_dir).is_dir():
        return index
    for path in sorted(Path(capture_dir).glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('schema') != 'online-global-plan-v1':
            continue
        meta = record.get('metadata') or {}
        route = meta.get('route') or meta.get('route_id')
        if route:
            index[route] = {'path': str(path), 'sha256': record.get('sha256'),
                            'points': record.get('world_point_count'),
                            'source': meta.get('source'),
                            'carla_server_version': meta.get('carla_server_version'),
                            'map_name': meta.get('map_name')}
    return index


def certification_status(row, defect_routes, captures):
    if row['route'] in defect_routes:
        return 'INSUFFICIENT_EVIDENCE', 'EXCLUDED_GEOMETRY_DEFECT'
    if row['route'] not in captures:
        return 'INSUFFICIENT_EVIDENCE', 'NO_ONLINE_PLAN_CAPTURED'
    verified = row.get('certification_status')
    if verified == 'ONLINE_MATCH_VERIFIED':
        return 'ONLINE_MATCH_VERIFIED', row.get('certification_reason') or 'CARLA_PLAN_VERIFIED'
    if verified == 'COMMAND_MISMATCH':
        return 'ONLINE_PLAN_CAPTURED', 'COMMAND_MISMATCH_RECORDED'
    if row.get('xml_preview_status') != 'PASS':
        return 'INSUFFICIENT_EVIDENCE', 'PLAN_COVERAGE_INSUFFICIENT'
    return 'ONLINE_PLAN_CAPTURED', 'ONLINE_MATCH_NOT_YET_VERIFIED'


def build_manifest(comparison_path, pairs, defect_routes, captures, output):
    counts, rows = Counter(), 0
    with Path(output).open('w') as stream:
        for row in read_jsonl(comparison_path):
            pair = pairs.get(row['sample_id'])
            if pair is None:
                counts['not_in_pair_index'] += 1
                continue
            status, reason = certification_status(row, defect_routes, captures)
            record = {
                'schema': 'navigation-certification-manifest-v1',
                'builder_version': VERSION,
                'sample_id': row['sample_id'],
                'route': row['route'],
                'frame': row['frame'],
                'split': pair['split'],
                'xml_id': row.get('xml_id'),
                'xml_source': row.get('xml_source'),
                'online_capture': captures.get(row['route']),
                'plan_sha256': row.get('plan_sha256'),
                'plan_metadata_source': row.get('plan_metadata_source'),
                'rear_projection_distance_m': row.get('rear_projection_distance_m'),
                'xml_preview_status': row.get('xml_preview_status'),
                'xml_command4': row.get('xml_command4'),
                'offline_command4': row.get('offline_command4'),
                'old_status_command4': row.get('old_status_command4'),
                'new_status_command4': row.get('new_status_command4'),
                'point_distance_m': row.get('xml_vs_offline_point_distance_m'),
                'xml_horizon_m': row.get('xml_horizon_m'),
                'certification_status': status,
                'certification_reason': reason,
                'live_agent_source_certified': False,
                'online_input_reproducibility': 'NOT_VERIFIED',
                'training_admitted': False,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            counts[status] += 1
            counts['total'] += 1
            rows += 1
    return {'rows': rows, 'counts': dict(counts)}


def build_subset(manifest_path, pairs, defect_routes, overlay_path, output):
    overlay = {}
    for row in read_jsonl(overlay_path):
        overlay[row['sample_id']] = row
    verified, excluded, rejected = [], Counter(), []
    for record in read_jsonl(manifest_path):
        if record.get('certification_status') != VERIFIED:
            continue
        if record['route'] in defect_routes:
            excluded['geometry_defect'] = excluded.get('geometry_defect', 0) + 1
            continue
        pair = pairs.get(record['sample_id'])
        overlay_row = overlay.get(record['sample_id'])
        if pair is None or overlay_row is None:
            rejected.append({'sample_id': record['sample_id'], 'reason': 'missing_pair_or_overlay'})
            continue
        old = overlay_row.get('old_status8')
        new = overlay_row.get('status8')
        if old is None or new is None:
            rejected.append({'sample_id': record['sample_id'], 'reason': 'missing_status'})
            continue
        if list(old[4:]) != list(new[4:]):
            rejected.append({'sample_id': record['sample_id'], 'reason': 'motion4_changed'})
            continue
        verified.append({
            'schema': 'certified-status-row-v1',
            'sample_id': record['sample_id'],
            'sample_identity': {'sample_id': pair['sample_id'], 'route': pair['route'],
                                'frame': pair['frame'], 'split': pair['split']},
            'status8': new,
            'old_status8': old,
            'a0_motion4': list(old[4:]),
            'navigation_certification': VERIFIED,
            'navigation_evidence': record.get('online_capture'),
            'plan_sha256': record.get('plan_sha256'),
            'certification_status': VERIFIED,
            'certification_builder': VERSION,
            'staged_only': True,
            'training_admitted': False,
            'not_admitted_reason': 'CERTIFICATION_CONTRACT_NOT_APPROVED',
        })
    with Path(output).open('w') as stream:
        for row in verified:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return {'verified_rows': len(verified), 'excluded': dict(excluded),
            'rejected': rejected[:100], 'rejected_count': len(rejected),
            'staged_rows': len(verified),
            'reader_ready': False,
            'blocking_reason': 'CERTIFICATION_CONTRACT_NOT_APPROVED' if verified else 'NO_ONLINE_MATCH_VERIFIED_ROWS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--capture-dir', type=Path, default=None)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--pair-index', type=Path, required=True)
    parser.add_argument('--defect-routes', type=Path, required=True,
                        help='newline-delimited geometry-defect Route names')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    args.output_dir.mkdir(parents=True)
    pairs = pair_index(args.pair_index)
    defect_routes = {line.strip() for line in args.defect_routes.read_text().splitlines() if line.strip()}
    captures = capture_index(args.capture_dir)
    manifest_path = args.output_dir / 'navigation_certification_manifest.jsonl'
    manifest = build_manifest(args.comparison, pairs, defect_routes, captures, manifest_path)
    subset_path = args.output_dir / 'certified_status_subset.jsonl'
    subset = build_subset(manifest_path, pairs, defect_routes, args.overlay, subset_path)
    summary = {
        'schema': VERSION,
        'capture_dir': None if args.capture_dir is None else str(args.capture_dir),
        'captured_routes': len(captures),
        'manifest': manifest,
        'subset': subset,
        'go_no_go': 'NO_GO',
        'blocking_reason': subset['blocking_reason'],
        'policy': 'no row is admitted without captured online plan plus verified match',
    }
    (args.output_dir / 'certified_status_subset_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('captured_routes', 'go_no_go', 'blocking_reason')}, ensure_ascii=False))
    print(json.dumps(manifest['counts'], ensure_ascii=False))
    return 0 if subset['reader_ready'] else 3


if __name__ == '__main__':
    sys.exit(main())
