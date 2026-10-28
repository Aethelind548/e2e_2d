"""Audit navigation evidence levels for the status certification chain.

The earlier round labelled 555 frames ``ONLINE_MATCH_VERIFIED`` because a plan
rebuilt by the real CARLA planner from the local route XML agreed with the
offline reconstruction.  That label must not be read as training certification.
This tool re-derives an explicit evidence level per frame:

* ``CERTIFIED``                - session-captured plan, route identity accepted,
                                 per-frame Command4 equal, evidence SHA pinned
* ``COMMAND_MATCH_VERIFIED``    - session-captured plan and Command4 equal, but
                                 dataset route identity is not fully pinned
* ``SESSION_PLAN_CAPTURED``    - real ``set_global_plan`` arguments captured, but
                                 Command4 differs from the offline overlay
* ``CARLA_PLANNER_REPRODUCED`` - planner rebuilt from local XML only; no session
* ``COMMAND_MISMATCH``         - available plan disagrees with the offline overlay
* ``INSUFFICIENT_EVIDENCE``    - no usable plan, excluded route, or poor coverage

Read-only: consumes existing evidence directories and writes one audit output
directory.  It never edits the frozen overlay, pair index or dataset caches.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tools.validation.b2d.route_source_certification import (
    CARLA_PLAN_SOURCES,
    REAR_PROJECTION_TOLERANCE_M,
    dedupe_points,
    preview_command4,
    project_progress,
)

VERSION = 'certification-level-audit-v1'
PREVIEW_M = 20.0
LATERAL_THRESHOLD_M = 2.0
# The offline preview generator uses 5.0 m; certification must use the same rule.
PREVIEW_MAX_PROJECTION_M = 5.0
LEVELS = (
    'CERTIFIED',
    'COMMAND_MATCH_VERIFIED',
    'SESSION_PLAN_CAPTURED',
    'CARLA_PLANNER_REPRODUCED',
    'COMMAND_MISMATCH',
    'INSUFFICIENT_EVIDENCE',
)


def load_defect_routes(path):
    if not path:
        return set()
    p = Path(path)
    if not p.is_file():
        return set()
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


def load_sessions(session_dir):
    """Index session captures by dataset route."""
    sessions = {}
    if not session_dir:
        return sessions
    for path in sorted(Path(session_dir).glob('*.json')):
        if path.name == 'session_capture_index.jsonl':
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        metadata = record.get('metadata', {})
        route = metadata.get('dataset_route')
        if not route:
            continue
        sessions[route] = {
            'path': str(path),
            'sha256': record.get('sha256'),
            'metadata': metadata,
            'record': record,
            'points': dedupe_points([(p[0], p[1]) for p in record['raw']['world_points']]),
            'agent_points': dedupe_points([(p[0], p[1]) for p in record['agent_view']['world_points']]),
        }
    return sessions


def load_planners(planner_dir):
    planners = {}
    if not planner_dir:
        return planners
    for path in sorted(Path(planner_dir).glob('*.json')):
        if path.name == 'capture_index.json':
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        route = record.get('metadata', {}).get('route')
        if route:
            planners[route] = {
                'sha256': record.get('sha256'),
                'points': dedupe_points([(p[0], p[1]) for p in record['world_points']]),
                'metadata': record.get('metadata', {}),
            }
    return planners


def command_of(command4):
    if not command4 or len(command4) < 4 or 1.0 not in command4[:4]:
        return None
    return ('Left', 'Straight', 'Right', 'Unknown')[command4[:4].index(1.0)]


def _preview(points, rear, yaw):
    if len(points) < 2:
        return {'status': 'FAIL', 'reason': 'plan_too_short'}
    return preview_command4(points, rear, yaw, preview_m=PREVIEW_M,
                            lateral_threshold_m=LATERAL_THRESHOLD_M,
                            max_projection_m=PREVIEW_MAX_PROJECTION_M)


def classify_row(row, session, planner, association, defect_routes):
    """Return (level, reason, details) for one comparison row."""
    route = row['route']
    details = {}
    planner_ok = row.get('plan_metadata_source') in CARLA_PLAN_SOURCES and bool(row.get('plan_sha256'))
    details['planner_capture_sha256'] = row.get('plan_sha256')
    details['planner_capture_matches_index'] = bool(
        planner and row.get('plan_sha256') == planner.get('sha256'))

    if route in defect_routes:
        return 'INSUFFICIENT_EVIDENCE', 'EXCLUDED_GEOMETRY_DEFECT', details
    if not planner_ok:
        return 'INSUFFICIENT_EVIDENCE', 'NO_PLANNER_EVIDENCE', details
    if row.get('xml_preview_status') != 'PASS':
        return 'INSUFFICIENT_EVIDENCE', 'PLAN_COVERAGE_INSUFFICIENT', details
    if (row.get('rear_projection_distance_m') or 0.0) > REAR_PROJECTION_TOLERANCE_M:
        return 'INSUFFICIENT_EVIDENCE', 'COORDINATE_PROJECTION_TOO_FAR', details

    offline = row.get('offline_command4')
    plan_command = row.get('xml_command4')
    details['planner_command'] = command_of(plan_command)
    details['offline_command'] = command_of(offline)
    details['preview_point_distance_m'] = row.get('xml_vs_offline_point_distance_m')
    details['horizon_m'] = row.get('xml_horizon_m')

    association_record = association.get(route, {})
    planner_meta = (planner or {}).get('metadata', {})
    session_meta = session['metadata'] if session else {}
    same_xml = bool(
        session_meta.get('xml_sha256')
        and session_meta.get('xml_sha256') == planner_meta.get('xml_sha256'))
    same_map = bool(
        session_meta.get('map_name')
        and session_meta.get('map_name').split('/')[2:3] == (row.get('map_name') or '').split('/')[2:3])
    identity_ok = bool(
        association_record.get('accepted')
        and session_meta.get('dataset_route') == route
        and session_meta.get('town') == row.get('town')
        and same_xml
        and same_map
        and session_meta.get('leaderboard_evaluator_sha256')
        and session_meta.get('route_scenario_sha256'))
    details['route_identity'] = {
        'association_accepted': bool(association_record.get('accepted')),
        'same_route_xml_sha': same_xml,
        'same_map': same_map,
        'evaluator_sha_pinned': bool(session_meta.get('leaderboard_evaluator_sha256')),
        'route_scenario_sha_pinned': bool(session_meta.get('route_scenario_sha256')),
        'xml_id_recorded_by_planner': planner_meta.get('xml_id'),
        'route_id_matches_xml_id': bool(planner_meta.get('route_id') == planner_meta.get('xml_id')),
    }

    if session is None:
        details['session_capture_sha256'] = None
        details['command_match'] = bool(plan_command == offline and offline is not None)
        if not details['command_match']:
            return 'COMMAND_MISMATCH', 'PLANNER_COMMAND_DIFFERS_FROM_OFFLINE', details
        return 'CARLA_PLANNER_REPRODUCED', 'PLANNER_ONLY_NO_SESSION_CAPTURE', details

    details['session_capture_sha256'] = session['sha256']
    session_preview = _preview(session['points'], row['rear_xy'], row['yaw_rad'])
    details['session_command'] = command_of(session_preview.get('command4'))
    details['session_preview_status'] = session_preview.get('status')
    details['session_command_match'] = bool(
        session_preview.get('status') == 'PASS' and session_preview.get('command4') == offline
        and offline is not None)
    if session_preview.get('status') == 'PASS':
        rear_arc, _ = project_progress(session['points'], row['rear_xy'])
        preview_arc, _ = project_progress(session['points'], session_preview['preview_point_xy'])
        details['session_horizon_m'] = round(preview_arc - rear_arc, 3)
        details['session_vs_planner_point_m'] = round(float(math.dist(
            session_preview['preview_point_xy'], row.get('xml_preview_point_xy') or [float('nan')] * 2)), 4) \
            if row.get('xml_preview_point_xy') else None
    if not details['session_command_match']:
        return 'COMMAND_MISMATCH', 'SESSION_COMMAND_DIFFERS_FROM_OFFLINE', details
    details['command_match'] = True
    details['route_identity_ok'] = identity_ok
    if not identity_ok:
        return 'COMMAND_MATCH_VERIFIED', 'ROUTE_IDENTITY_NOT_PINNED', details
    return 'CERTIFIED', 'SESSION_PLAN_COMMAND_VERIFIED', details


def agent_view_probe(session, row, offline):
    """Check whether the agent's downsampled view carries the same Command4."""
    if session is None:
        return None
    probe = _preview(session['agent_points'], row['rear_xy'], row['yaw_rad'])
    return {
        'agent_point_count': len(session['agent_points']),
        'agent_preview_status': probe.get('status'),
        'agent_command': command_of(probe.get('command4')),
        'offline_command': command_of(offline),
        'agent_command_matches_offline': bool(probe.get('status') == 'PASS' and probe.get('command4') == offline),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--association', type=Path, required=True)
    parser.add_argument('--session-dir', type=Path, default=None)
    parser.add_argument('--planner-dir', type=Path, default=None)
    parser.add_argument('--defect-routes', type=Path, default=None)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.comparison.open() if line.strip()]
    association = {r['route']: r for r in json.loads(args.association.read_text())['records']}
    sessions = load_sessions(args.session_dir)
    planners = load_planners(args.planner_dir)
    defect_routes = load_defect_routes(args.defect_routes)

    level_rows = []
    per_route = defaultdict(lambda: {'frames': 0, 'levels': Counter(), 'session': False})
    probe_rows = []
    for row in rows:
        session = sessions.get(row['route'])
        planner = planners.get(row['route'])
        level, reason, details = classify_row(row, session, planner, association, defect_routes)
        record = {
            'sample_id': row['sample_id'],
            'route': row['route'],
            'town': row['town'],
            'frame': row['frame'],
            'split': None,
            'certification_level': level,
            'certification_reason': reason,
            'command_match': details.get('command_match', details.get('session_command_match')),
            'planner_capture_sha256': details.get('planner_capture_sha256'),
            'session_capture_sha256': details.get('session_capture_sha256'),
            'route_source_identity': {
                'accepted_association': bool(association.get(row['route'], {}).get('accepted')),
                'xml_id': row.get('xml_id'),
                'xml_source': row.get('xml_source'),
                'xml_id_matches_dataset_route_id': bool(
                    association.get(row['route'], {}).get('xml_id_matches_route_id')),
            },
            'preview_point_distance_m': details.get('preview_point_distance_m'),
            'horizon_m': details.get('horizon_m'),
            'planner_command': details.get('planner_command'),
            'session_command': details.get('session_command'),
            'offline_command': details.get('offline_command'),
            'rear_projection_distance_m': row.get('rear_projection_distance_m'),
            'details': details,
        }
        level_rows.append(record)
        bucket = per_route[row['route']]
        bucket['frames'] += 1
        bucket['levels'][level] += 1
        bucket['session'] = bucket['session'] or session is not None
        if session is not None:
            probe = agent_view_probe(session, row, row.get('offline_command4'))
            probe_rows.append({'route': row['route'], 'frame': row['frame'], 'sample_id': row['sample_id'], **probe})

    counts = Counter(r['certification_level'] for r in level_rows)
    session_routes = sorted(sessions)
    session_frames = [r for r in level_rows if r['route'] in set(session_routes)]
    summary = {
        'version': VERSION,
        'rows': len(level_rows),
        'routes': len({r['route'] for r in level_rows}),
        'level_counts': dict(counts),
        'level_order': list(LEVELS),
        'session_routes': session_routes,
        'session_frames': len(session_frames),
        'session_level_counts': dict(Counter(r['certification_level'] for r in session_frames)),
        'planner_only_frames': counts.get('CARLA_PLANNER_REPRODUCED', 0),
        'planner_only_routes': sum(1 for r, b in per_route.items() if not b['session']),
        'command_mismatch_frames': counts.get('COMMAND_MISMATCH', 0),
        'insufficient_frames': counts.get('INSUFFICIENT_EVIDENCE', 0),
        'certified_frames': counts.get('CERTIFIED', 0),
        'certified_routes': sum(1 for b in per_route.values() if b['levels'].get('CERTIFIED')),
        'agent_view_probes': len(probe_rows),
        'agent_view_matches': sum(1 for p in probe_rows if p['agent_command_matches_offline']),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'certification_level_rows.jsonl').open('w') as stream:
        for record in level_rows:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
    with (args.output / 'agent_view_probe.jsonl').open('w') as stream:
        for record in probe_rows:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
    (args.output / 'certification_level_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
