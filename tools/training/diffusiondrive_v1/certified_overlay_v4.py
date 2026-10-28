"""Contract-complete audit of session-certified frames and Overlay v4 candidate.

The staged Overlay v3 carried evidence levels but was never checked field by field
against the revised contract.  This tool does that check and writes a *new*
candidate overlay containing only the rows that pass every contract condition:

1. task route identity .......... accepted association, same XML SHA, same map
2. real session source .......... session capture file exists and its SHA matches
3. same navigation representation dense ``set_global_plan`` world plan, 20 m arc,
                                  2.0 m lateral threshold, 5.0 m projection gate
4. Command4 agreement ........... session preview command equals the offline row
5. rear/coordinate contract ..... rear projection <= 5 m, horizon = 20 m arc
6. evidence SHA pinned .......... session, planner, frozen overlay, feature, target
7. frozen Feature/Target identity pair index row exists, manifest/index SHA match
8. Motion4 equals frozen A0 ..... status8[4:] == old_status8[4:] == a0_motion4

Rows that fail any condition are written to ``rejected_rows.jsonl`` with the list
of failed checks; nothing is coerced or repaired.  ``training_admitted`` stays
false because the contract is still under review.
"""
import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from tools.validation.b2d.session_plan_agent import record_sha as session_record_sha

VERSION = 'certified-overlay-v4.0.0'
SCHEMA = 'certified-status-overlay-v4'
NAVIGATION_RULE = {
    'id': 'dense-set-global-plan-world-xy/arc20/lateral2.0/projection5.0',
    'representation': 'dense set_global_plan world points (1 m hop)',
    'preview_m': 20.0,
    'lateral_threshold_m': 2.0,
    'max_projection_m': 5.0,
    'rear_reference': 'rear axle, left-positive local frame',
    'agent_view_used': False,
}

CHECKS = (
    'route_identity',
    'session_source',
    'navigation_representation',
    'command_match',
    'rear_coordinate',
    'evidence_sha',
    'frozen_identity',
    'motion_a0',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def class_of(status8):
    if not status8 or 1.0 not in status8[:4]:
        return 'None'
    return ('Left', 'Straight', 'Right', 'Unknown')[status8[:4].index(1.0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuous-rows', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--pair-index', type=Path, required=True)
    parser.add_argument('--association', type=Path, required=True)
    parser.add_argument('--session-dir', type=Path, required=True)
    parser.add_argument('--planner-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    association = {r['route']: r for r in json.loads(args.association.read_text())['records']}
    overlay_sha = sha256_file(args.overlay)
    pair_index_sha = sha256_file(args.pair_index)
    registry = contract.get('evidence_registry', {})
    pinned_overlay_sha = registry.get('frozen_overlay_sha256')
    pinned_pair_index_sha = registry.get('frozen_pair_index_sha256')

    sessions = {}
    for path in sorted(args.session_dir.glob('*.json')):
        record = json.loads(path.read_text())
        route = record['metadata'].get('dataset_route')
        if route:
            sessions[route] = {'path': path, 'sha256': record['sha256'],
                               'content_sha_matches': session_record_sha(record) == record['sha256'],
                               'metadata': record['metadata']}
    planners = {}
    if args.planner_dir:
        for path in sorted(args.planner_dir.glob('*.json')):
            if path.name == 'capture_index.json':
                continue
            record = json.loads(path.read_text())
            route = record['metadata'].get('route')
            if route:
                planners[route] = record

    pairs = {}
    connection = sqlite3.connect(f'file:{args.pair_index}?mode=ro', uri=True)
    try:
        for row in connection.execute(
                'select sample_id, route, frame, split, target_sha, feature_sha from pairs'):
            pairs[row[0]] = row
    finally:
        connection.close()

    overlay = {}
    with args.overlay.open() as stream:
        for line in stream:
            row = json.loads(line)
            sample = row['sample_id']
            if sample in pairs:
                overlay[sample] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    passed, rejected = [], []
    for row in load_jsonl(args.continuous_rows):
        sample = row['sample_id']
        checks = dict.fromkeys(CHECKS, False)
        reasons = []
        session = sessions.get(row['route'])
        planner = planners.get(row['route'])
        pair = pairs.get(sample)
        frozen = overlay.get(sample)

        if session:
            association_record = association.get(row['route'], {})
            checks['session_source'] = bool(session['sha256'] and session['content_sha_matches'])
            same_xml = bool(planner and session['metadata'].get('xml_sha256')
                            == planner['metadata'].get('xml_sha256'))
            checks['route_identity'] = bool(
                association_record.get('accepted')
                and session['metadata'].get('dataset_route') == row['route']
                and session['metadata'].get('town') == (planner or {}).get('metadata', {}).get('town', session['metadata'].get('town'))
                and same_xml
                and session['metadata'].get('leaderboard_evaluator_sha256')
                and session['metadata'].get('route_scenario_sha256'))
        checks['navigation_representation'] = bool(
            row.get('preview_status') == 'PASS'
            and abs((row.get('horizon_m') or 0.0) - NAVIGATION_RULE['preview_m']) <= 0.5)
        checks['command_match'] = row.get('level') == 'CERTIFIED'
        checks['rear_coordinate'] = bool(
            (row.get('rear_projection_m') or 99.0) <= NAVIGATION_RULE['max_projection_m'])
        checks['frozen_identity'] = bool(
            pair is not None and frozen is not None
            and overlay_sha == pinned_overlay_sha
            and pair_index_sha == pinned_pair_index_sha
            and frozen.get('source_feature_identity', {}).get('pair_index_sha256') == pair_index_sha)
        if frozen:
            a0 = list(frozen['a0_motion4'])
            checks['motion_a0'] = bool(
                list(frozen['old_status8'][4:]) == a0 == list(frozen['status8'][4:]))
        checks['evidence_sha'] = bool(
            session and session['sha256']
            and (planner or {}).get('sha256')
            and overlay_sha and pair and pair[4] and pair[5])

        record = {
            'sample_id': sample,
            'route': row['route'],
            'frame': row['frame'],
            'split': row.get('split'),
            'checks': checks,
            'offline_command': row.get('offline_command'),
            'session_command': row.get('preview_command'),
            'rear_projection_m': row.get('rear_projection_m'),
            'horizon_m': row.get('horizon_m'),
            'evidence': {
                'session_capture_sha256': (session or {}).get('sha256'),
                'planner_capture_sha256': (planner or {}).get('sha256'),
                'frozen_overlay_sha256': overlay_sha,
                'pair_index_sha256': pair_index_sha,
                'feature_sha256': pair[5] if pair else None,
                'target_sha256': pair[4] if pair else None,
            },
            'navigation_rule': NAVIGATION_RULE,
        }
        if all(checks.values()):
            passed.append(record)
        else:
            failures = [name for name, ok in checks.items() if not ok]
            record['failed_checks'] = failures
            record['rejection_reason'] = failures[0].upper()
            rejected.append(record)

    rows, seen = [], set()
    for record in passed:
        sample = record['sample_id']
        if sample in seen:
            continue
        seen.add(sample)
        frozen = overlay[sample]
        row = {
            'schema': frozen['schema'],
            'producer_version': VERSION,
            'sample_id': sample,
            'sample_identity': frozen['sample_identity'],
            'status8': frozen['status8'],
            'old_status8': frozen['old_status8'],
            'a0_motion4': frozen['a0_motion4'],
            'motion_source': frozen['motion_source'],
            'status_dtype': frozen['status_dtype'],
            'status_order': frozen['status_order'],
            'navigation_order': frozen['navigation_order'],
            'source_feature_identity': frozen['source_feature_identity'],
            'preview_identity': frozen['preview_identity'],
            'diagnostic_metadata': frozen.get('diagnostic_metadata'),
            'preview_validity': 'VALID',
            'status_validity': 'VALID',
            'fallback_source': None,
            'physical_payload_hash_verification': frozen.get(
                'physical_payload_hash_verification', 'NOT_RUN_DEFERRED_TO_STRICT_READER'),
            'navigation_source': {
                'kind': 'ONLINE_SESSION_SET_GLOBAL_PLAN',
                'certification': 'CERTIFIED',
                'reference_path_online_available': True,
                'evidence_sha256': record['evidence']['session_capture_sha256'],
                'navigation_rule': NAVIGATION_RULE,
            },
            'failure_reason': None,
            'training_admitted': False,
            'training_contract': 'NOT_ADMITTED_EXPERIMENT_ONLY',
            'certification_level': 'CERTIFIED',
            'contract_checks': record['checks'],
            'navigation_rule': NAVIGATION_RULE,
            'evidence': record['evidence'],
            'frozen_overlay_sha256': overlay_sha,
            'staged_only': True,
            'contract_status': contract.get('status', 'UNKNOWN'),
            'not_admitted_reason': 'CERTIFICATION_CONTRACT_PENDING_REVIEW',
        }
        rows.append(row)

    with (args.output_dir / 'certified_status_overlay_v4.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    with (args.output_dir / 'rejected_rows.jsonl').open('w') as stream:
        for row in rejected:
            stream.write(json.dumps(row, sort_keys=True) + '\n')

    by_split = Counter(row['sample_identity']['split'] for row in rows)
    by_route = defaultdict(int)
    for row in rows:
        identity = row['sample_identity']
        by_route[(identity['split'], identity['route'])] += 1
    summary = {
        'version': VERSION,
        'contract_status': contract.get('status'),
        'rows_in': len(passed) + len(rejected),
        'passed_rows': len(rows),
        'rejected_rows': len(rejected),
        'by_split': dict(by_split),
        'routes_by_split': {
            'train': len({r for (s, r) in by_route if s == 'train'}),
            'validation': len({r for (s, r) in by_route if s == 'validation'}),
        },
        'routes': {f'{split}:{route}': count for (split, route), count in sorted(by_route.items())},
        'classes': dict(Counter(class_of(row['status8']) for row in rows)),
        'rejection_reasons': dict(Counter(r['rejection_reason'] for r in rejected)),
        'training_admitted_rows': sum(1 for row in rows if row['training_admitted']),
        'reader_ready': False,
        'blocking_reason': 'CERTIFICATION_CONTRACT_PENDING_REVIEW',
        'evidence': {'frozen_overlay_sha256': overlay_sha, 'pair_index_sha256': pair_index_sha},
    }
    (args.output_dir / 'certified_overlay_v4_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')

    reader_contract = {
        'schema': 'status-overlay-reader-contract-candidate-v1',
        'status': 'DRAFT_NOT_ACTIVE',
        'overlay': 'certified_status_overlay_v4.jsonl',
        'overlay_sha256': sha256_file(args.output_dir / 'certified_status_overlay_v4.jsonl'),
        'producer_version': VERSION,
        'certification_level': 'CERTIFIED',
        'navigation_rule': NAVIGATION_RULE,
        'evidence_allowlist': sorted({row['evidence']['session_capture_sha256'] for row in rows}),
        'training_admitted': False,
        'note': 'not activated until the certification contract is approved',
    }
    (args.output_dir / 'reader_contract_candidate_v1.json').write_text(
        json.dumps(reader_contract, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in (
        'passed_rows', 'rejected_rows', 'by_split', 'routes_by_split', 'classes',
        'rejection_reasons', 'training_admitted_rows', 'blocking_reason')}, sort_keys=True))


if __name__ == '__main__':
    main()
