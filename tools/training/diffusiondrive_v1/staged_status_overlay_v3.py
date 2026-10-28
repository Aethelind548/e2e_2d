"""Build the staged Status overlay v3 candidate (never training-admitted).

The frozen overlay v2 stays untouched.  This tool writes a *candidate* file in a
separate directory where every row keeps its offline status payload but gains the
navigation evidence level derived by ``certification_level_audit``:

``certification_level``         one of the audit levels
``route_source_identity``       association / XML / map identity facts
``session_capture_sha256``      real leaderboard ``set_global_plan`` capture
``planner_capture_sha256``      CARLA planner reconstruction capture
``command_match``               per-frame Command4 equality with the offline row
``preview_point_distance_m``    plan preview point vs offline preview point
``certification_reason``        machine-readable reason code
``source_limitations``          explicit list of what the evidence does not prove
``training_admitted``           always ``false`` in this round

Rows are copied by ``sample_id`` from an existing staged subset; nothing is
interpolated or invented, and any requested row without an audit record is
reported instead of silently defaulted.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

VERSION = 'staged-status-overlay-v3.0.0'
SCHEMA = 'staged-status-overlay-v3'

LIMITATIONS = {
    'CERTIFIED': [
        'session plan captured from a real leaderboard run of the local route XML',
        'Command4 agreement holds at the sampled frames only (5 frames per route)',
        'not a re-run of the original data-collection session',
    ],
    'COMMAND_MATCH_VERIFIED': [
        'Command4 agreement holds but dataset route identity is not fully pinned',
    ],
    'SESSION_PLAN_CAPTURED': [
        'real set_global_plan arguments captured, Command4 differs from the offline overlay',
    ],
    'CARLA_PLANNER_REPRODUCED': [
        'plan rebuilt by the real CARLA planner from the local route XML only',
        'no real leaderboard session capture for this route',
    ],
    'COMMAND_MISMATCH': [
        'available plan disagrees with the offline overlay Command4',
    ],
    'INSUFFICIENT_EVIDENCE': [
        'plan coverage, provenance or coordinate projection insufficient',
    ],
}


def load_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def class_of(status8):
    if not status8 or len(status8) < 4 or 1.0 not in status8[:4]:
        return 'None'
    return ('Left', 'Straight', 'Right', 'Unknown')[status8[:4].index(1.0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--audit-rows', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    subset = load_rows(args.subset)
    audit = {row['sample_id']: row for row in load_rows(args.audit_rows)}
    overlay_sha = hashlib.sha256(Path(args.overlay).read_bytes()).hexdigest()
    overlay = {}
    with args.overlay.open() as stream:
        for line in stream:
            row = json.loads(line)
            overlay[row['sample_id']] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, missing_audit, missing_overlay = [], [], []
    for row in subset:
        sample_id = row['sample_identity']['sample_id'] if 'sample_identity' in row else row['sample_id']
        audit_row = audit.get(sample_id)
        if audit_row is None:
            missing_audit.append(sample_id)
            continue
        frozen = overlay.get(sample_id)
        if frozen is None:
            missing_overlay.append(sample_id)
            continue
        level = audit_row['certification_level']
        record = {
            'schema': SCHEMA,
            'producer_version': VERSION,
            'sample_id': sample_id,
            'split': row.get('split') or row['sample_identity']['split'],
            'route': row['sample_identity']['route'],
            'frame': row['sample_identity']['frame'],
            'status8': frozen['status8'],
            'old_status8': frozen['old_status8'],
            'a0_motion4': frozen['a0_motion4'],
            'status_dtype': frozen['status_dtype'],
            'status_order': frozen['status_order'],
            'navigation_order': frozen['navigation_order'],
            'certification_level': level,
            'route_source_identity': audit_row['route_source_identity'],
            'session_capture_sha256': audit_row['session_capture_sha256'],
            'planner_capture_sha256': audit_row['planner_capture_sha256'],
            'command_match': audit_row['command_match'],
            'preview_point_distance_m': audit_row['preview_point_distance_m'],
            'horizon_m': audit_row['horizon_m'],
            'rear_projection_distance_m': audit_row['rear_projection_distance_m'],
            'certification_reason': audit_row['certification_reason'],
            'source_limitations': LIMITATIONS[level],
            'frozen_overlay_sha256': overlay_sha,
            'training_admitted': False,
            'staged_only': True,
            'training_contract': 'NOT_ADMITTED_EXPERIMENT_ONLY',
        }
        rows.append(record)

    out = args.output_dir / 'staged_status_overlay_v3.jsonl'
    with out.open('w') as stream:
        for record in rows:
            stream.write(json.dumps(record, sort_keys=True) + '\n')

    by_level = Counter(r['certification_level'] for r in rows)
    by_split = Counter((r['split'], r['certification_level']) for r in rows)
    by_class = defaultdict(Counter)
    for record in rows:
        by_class[record['split']][class_of(record['status8'])] += 1
    per_route = defaultdict(set)
    for record in rows:
        per_route[record['split']].add(record['route'])
    train_routes, val_routes = per_route['train'], per_route['validation']
    certified_by_split = Counter(
        r['split'] for r in rows if r['certification_level'] == 'CERTIFIED')
    certified_route_split = {
        'train': sorted({r['route'] for r in rows
                         if r['certification_level'] == 'CERTIFIED' and r['split'] == 'train'}),
        'validation': sorted({r['route'] for r in rows
                              if r['certification_level'] == 'CERTIFIED' and r['split'] == 'validation'}),
    }
    session_routes = sorted({r['route'] for r in rows if r['session_capture_sha256']})

    summary = {
        'version': VERSION,
        'rows_in': len(subset),
        'rows_out': len(rows),
        'missing_audit_rows': len(missing_audit),
        'missing_overlay_rows': len(missing_overlay),
        'level_counts': dict(by_level),
        'level_by_split': {f'{split}:{level}': count for (split, level), count in sorted(by_split.items())},
        'class_by_split': {split: dict(counter) for split, counter in by_class.items()},
        'routes_by_split': {'train': len(train_routes), 'validation': len(val_routes)},
        'route_overlap_train_validation': sorted(train_routes & val_routes),
        'certified_rows_by_split': dict(certified_by_split),
        'certified_routes_by_split': {k: len(v) for k, v in certified_route_split.items()},
        'certified_route_names': certified_route_split,
        'session_capture_routes': session_routes,
        'training_admitted_rows': sum(1 for r in rows if r['training_admitted']),
        'reader_ready': False,
        'blocking_reason': 'CERTIFICATION_CONTRACT_V2_NOT_APPROVED',
    }
    (args.output_dir / 'staged_status_overlay_v3_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in (
        'rows_out', 'level_counts', 'routes_by_split', 'route_overlap_train_validation',
        'certified_rows_by_split', 'training_admitted_rows', 'blocking_reason')}, sort_keys=True))


if __name__ == '__main__':
    main()
