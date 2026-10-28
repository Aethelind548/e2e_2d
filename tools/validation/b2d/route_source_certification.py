"""Original task-route source certification for offline Route Preview.

Available independent navigation source: the leaderboard route XMLs shipped with
Bench2Drive (``bench2drive220.xml`` and ``bench2drive_0.0.4_val.xml``).  Each XML
route is an ordered CARLA-world waypoint list, but it is only a *proxy* for the
live agent: ``leaderboard_evaluator.py`` passes ``route_scenario.route`` to
``set_global_plan``, and that plan is densified by CARLA's GlobalRoutePlanner at
run time.  This tool therefore never emits ``CERTIFIED``.

It does two things:

1. Associate dataset routes with XML routes geometrically (ego rear-axle poses vs
   XML polyline).  Route-ID equality alone is recorded but never accepted.
2. For accepted associations, compute the shared 20 m ``preview_command4`` at the
   same frame and rear pose from the XML plan and compare it with the stored
   offline preview and the frozen old/new Status commands.

Read-only: no CARLA, no GPU, no cache or generation writes.
"""
import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.b2d_adapter.coordinate_transform import actor_to_reference  # noqa: E402
from tools.b2d_adapter.route_preview_contract import preview_command4  # noqa: E402

PREFIX = (-1.388632877, 0.0)
TOLERANCE_M = 6.0
COMMAND_ORDER = ['Left', 'Straight', 'Right', 'Unknown']
XML_FILES = ('bench2drive220.xml', 'bench2drive_0.0.4_val.xml')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_xml(directory):
    routes = {}
    for name in XML_FILES:
        path = Path(directory) / name
        root = ET.parse(path).getroot()
        for route in root.findall('route'):
            points = [(float(p.get('x')), float(p.get('y'))) for p in route.findall('waypoints/position')]
            if len(points) >= 2:
                routes.setdefault((route.get('town'), route.get('id')),
                                  {'source': name, 'points': points})
    return routes


def accept_match(coverage, median_m):
    """Acceptance rule for an original-route association (documented, testable)."""
    return bool(coverage >= 0.8 and median_m <= 8.0)


def polyline_distance(points, samples):
    pts = np.asarray(points, dtype=float)
    seg_a, seg_b = pts[:-1], pts[1:]
    ab = seg_b - seg_a
    length2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    out = []
    for sx, sy in samples:
        ap = np.array([sx, sy]) - seg_a
        t = np.clip((ap * ab).sum(axis=1) / length2, 0.0, 1.0)
        proj = seg_a + t[:, None] * ab
        out.append(float(np.min(np.hypot(proj[:, 0] - sx, proj[:, 1] - sy))))
    return out


def dedupe_points(points, epsilon=1e-6):
    """Drop consecutive duplicate plan points (trace_route can repeat waypoints)."""
    result = []
    for x, y in points:
        if not result or abs(x - result[-1][0]) > epsilon or abs(y - result[-1][1]) > epsilon:
            result.append((float(x), float(y)))
    return result


def project_progress(points, xy):
    """Arc length from the polyline start to the closest point, plus the distance."""
    pts = np.asarray(points, dtype=float)
    seg_a, seg_b = pts[:-1], pts[1:]
    ab = seg_b - seg_a
    length = np.hypot(ab[:, 0], ab[:, 1])
    length2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    ap = np.asarray(xy, dtype=float) - seg_a
    t = np.clip((ap * ab).sum(axis=1) / length2, 0.0, 1.0)
    proj = seg_a + t[:, None] * ab
    distances = np.hypot(proj[:, 0] - xy[0], proj[:, 1] - xy[1])
    best = int(np.argmin(distances))
    cumulative = float(np.concatenate(([0.0], np.cumsum(length)))[best] + t[best] * length[best])
    return cumulative, float(distances[best])


def pose_of(reader, index, frame):
    rear = actor_to_reference(reader.read(index.ref(frame)).actor_pose, PREFIX)
    return (float(rear.x), float(rear.y)), float(rear.yaw)


def associate(routes, by_town, source_root, poses):
    from tools.b2d_adapter import FrameReader, RouteIndex
    records, errors = [], []
    for number, route in enumerate(routes, 1):
        town = 'Town' + route.split('_Town')[1].split('_')[0]
        route_id = route.split('_Route')[1].split('_')[0]
        try:
            index = RouteIndex.from_dataset_root(source_root, route)
            frames = index.frame_ids
            step = max(1, len(frames) // poses)
            picks = list(frames[::step])[:poses]
            reader = FrameReader(index)
            samples = [pose_of(reader, index, frame)[0] for frame in picks]
            scored = []
            for xml_id, item in by_town.get(town, []):
                distances = polyline_distance(item['points'], samples)
                scored.append({'xml_id': xml_id, 'source': item['source'],
                               'coverage': sum(d <= TOLERANCE_M for d in distances) / len(distances),
                               'median_m': float(np.median(distances)),
                               'xml_points': len(item['points'])})
            scored.sort(key=lambda x: (-x['coverage'], x['median_m']))
            best = scored[0] if scored else None
            records.append({
                'route': route, 'town': town, 'route_id': route_id,
                'poses_used': len(samples),
                'xml_id_matches_route_id': any(s['xml_id'] == route_id for s in scored),
                'best': best,
                'accepted': bool(best and accept_match(best['coverage'], best['median_m'])),
                'candidates_in_town': len(scored),
            })
        except Exception as exc:  # noqa: BLE001 - record and continue
            errors.append({'route': route, 'error_type': type(exc).__name__, 'error': str(exc)[:200]})
        if number % 200 == 0:
            print(json.dumps({'phase': 'associate', 'routes': number,
                              'accepted': sum(r['accepted'] for r in records)}), flush=True)
    return records, errors


CARLA_PLAN_SOURCES = ('CARLA_GLOBAL_ROUTE_PLANNER_FROM_LOCAL_XML',)
REAR_PROJECTION_TOLERANCE_M = 5.0


def _status(record, xml_preview, offline_command, rear_projection):
    captured = record['best'].get('source') == 'ONLINE_CAPTURED_PLAN'
    if not captured:
        if xml_preview.get('status') != 'PASS':
            return 'INSUFFICIENT_EVIDENCE'
        if offline_command is None or xml_preview.get('command4') != offline_command:
            return 'COMMAND_MISMATCH'
        return 'ONLINE_PENDING'
    if record.get('plan_metadata_source') not in CARLA_PLAN_SOURCES or not record.get('plan_sha256'):
        return 'INSUFFICIENT_EVIDENCE'
    if xml_preview.get('status') != 'PASS' or rear_projection > REAR_PROJECTION_TOLERANCE_M:
        return 'INSUFFICIENT_EVIDENCE'
    return 'ONLINE_MATCH_VERIFIED'


def _reason(record, xml_preview, offline_command, rear_projection):
    if record['best'].get('source') != 'ONLINE_CAPTURED_PLAN':
        return 'XML_PREVIEW_NOT_COVERED' if xml_preview.get('status') != 'PASS' else 'NO_ONLINE_PLAN_CAPTURED'
    if record.get('plan_metadata_source') not in CARLA_PLAN_SOURCES:
        return 'PLAN_PROVENANCE_UNSUPPORTED'
    if not record.get('plan_sha256'):
        return 'PLAN_SHA_MISSING'
    if xml_preview.get('status') != 'PASS':
        return 'PLAN_COVERAGE_INSUFFICIENT'
    if rear_projection > REAR_PROJECTION_TOLERANCE_M:
        return 'COORDINATE_PROJECTION_TOO_FAR'
    return 'CARLA_PLAN_FROM_LOCAL_XML_VERIFIED'


def compare(records, xml, preview_root, manifest, overlay_path, frames_per_route, source_root):
    import pyarrow.parquet as pq
    from tools.b2d_adapter import FrameReader, RouteIndex
    from tools.training.diffusiondrive_v1.features_cache import sample_id as make_sample_id
    dataset_id = manifest['dataset_id']
    overlay = {}
    wanted = {(r['route'], r['best']['xml_id']) for r in records if r['accepted']}
    if overlay_path and Path(overlay_path).is_file():
        with Path(overlay_path).open() as stream:
            for line in stream:
                row = json.loads(line)
                ident = row['sample_identity']
                if ident['route'] in {r['route'] for r in records if r['accepted']}:
                    overlay[make_sample_id(dataset_id, ident['route'], ident['frame'])] = row
    columns = ['frame', 'preview_valid', 'preview_reason', 'command4_json', 'preview_point_xy_json',
               'projection_distance_m', 'motion4_json', 'preview_category', 'route_progress_m']
    rows, errors = [], []
    for record in [r for r in records if r['accepted']]:
        route = record['route']
        key = (None, route) if record['best'].get('source') == 'ONLINE_CAPTURED_PLAN' else (record['town'], record['best']['xml_id'])
        points = dedupe_points(xml[key]['points'])
        if len(points) < 2:
            errors.append({'route': route, 'stage': 'compare', 'error_type': 'ValueError',
                           'error': 'plan has fewer than two distinct points'})
            continue
        path = Path(preview_root) / 'routes' / route / 'frame_route_preview.parquet'
        try:
            table = pq.read_table(path, columns=columns).to_pylist()
            index = RouteIndex.from_dataset_root(source_root, route)
            reader = FrameReader(index)
            available = {row['frame']: row for row in table}
            frame_ids = [f for f in index.frame_ids if f in available]
            step = max(1, len(frame_ids) // frames_per_route)
            picks = frame_ids[::step][:frames_per_route]
            for frame in picks:
                stored = available[frame]
                rear, yaw = pose_of(reader, index, frame)
                xml_preview = preview_command4(points, rear, yaw, max_projection_m=10.0)
                offline_command = json.loads(stored['command4_json']) if stored.get('command4_json') else None
                offline_point = json.loads(stored['preview_point_xy_json']) if stored.get('preview_point_xy_json') else None
                xml_point = xml_preview.get('preview_point_xy')
                sid = make_sample_id(dataset_id, route, frame)
                overlay_row = overlay.get(sid)
                motion = json.loads(stored['motion4_json']) if stored.get('motion4_json') else None
                distance = (float(math.hypot(xml_point[0] - offline_point[0], xml_point[1] - offline_point[1]))
                            if xml_point and offline_point else None)
                rear_arc, rear_projection = project_progress(points, rear)
                preview_arc = project_progress(points, xml_point)[0] if xml_point else None
                horizon = (preview_arc - rear_arc) if preview_arc is not None else None
                rows.append({
                    'route': route, 'town': record['town'], 'xml_id': record['best']['xml_id'],
                    'xml_source': record['best']['source'], 'frame': frame, 'sample_id': sid,
                    'rear_xy': list(rear), 'yaw_rad': yaw,
                    'xml_command4': xml_preview.get('command4'),
                    'xml_preview_status': xml_preview.get('status'),
                    'xml_preview_point_xy': xml_point,
                    'offline_preview_valid': bool(stored['preview_valid']),
                    'offline_command4': offline_command,
                    'offline_preview_point_xy': offline_point,
                    'xml_vs_offline_point_distance_m': distance,
                    'xml_vs_offline_command_match': bool(offline_command is not None
                                                         and xml_preview.get('command4') == offline_command),
                    'old_status_command4': None if overlay_row is None or overlay_row.get('old_status8') is None
                    else overlay_row['old_status8'][:4],
                    'new_status_command4': None if overlay_row is None or overlay_row.get('status8') is None
                    else overlay_row['status8'][:4],
                    'status_available': overlay_row is not None,
                    'xml_rear_arc_m': round(rear_arc, 3),
                    'xml_preview_arc_m': None if preview_arc is None else round(preview_arc, 3),
                    'xml_horizon_m': None if horizon is None else round(horizon, 3),
                    'offline_route_progress_m': stored.get('route_progress_m'),
                    'plan_source': record['best'].get('source'),
                    'plan_metadata_source': record.get('plan_metadata_source'),
                    'plan_sha256': record.get('plan_sha256'),
                    'carla_server_version': record.get('carla_server_version'),
                    'map_name': record.get('map_name'),
                    'rear_projection_distance_m': round(rear_projection, 3),
                    'certification_status': _status(record, xml_preview, offline_command, rear_projection),
                    'certification_reason': _reason(record, xml_preview, offline_command, rear_projection),
                    'speed_mps': None if motion is None else float(motion[0]),
                    'navigation_source_class': 'ORIGINAL_ROUTE_XML_PROXY',
                    'live_agent_source_certified': False,
                    'online_input_reproducibility': 'NOT_VERIFIED',
                })
        except Exception as exc:  # noqa: BLE001
            errors.append({'route': route, 'stage': 'compare', 'error_type': type(exc).__name__,
                           'error': str(exc)[:200]})
    return rows, errors


def summarise(association, rows, errors):
    accepted = [r for r in association if r['accepted']]
    comparable = [r for r in rows if r['offline_command4'] is not None]
    by_command = Counter()
    for row in rows:
        command = row['xml_command4']
        name = COMMAND_ORDER[int(np.argmax(command))] if command else 'NONE'
        by_command[name] += 1
    matched = [r for r in comparable if r['xml_vs_offline_command_match']]
    distances = [r['xml_vs_offline_point_distance_m'] for r in comparable
                 if r['xml_vs_offline_point_distance_m'] is not None]
    low_speed = [r for r in comparable if (r['speed_mps'] is not None and r['speed_mps'] < 0.5)]
    xml_pass = [r for r in comparable if r['xml_preview_status'] == 'PASS']
    xml_pass_match = [r for r in xml_pass if r['xml_vs_offline_command_match']]
    near = [r for r in xml_pass if r['xml_vs_offline_point_distance_m'] is not None
            and r['xml_vs_offline_point_distance_m'] <= 10.0]
    near_match = [r for r in near if r['xml_vs_offline_command_match']]
    confusion = Counter()
    class_frames = Counter()
    class_routes = defaultdict(set)
    for row in comparable:
        if row['xml_command4'] is None or row['offline_command4'] is None:
            continue
        xml_name = COMMAND_ORDER[int(np.argmax(row['xml_command4']))]
        offline_name = COMMAND_ORDER[int(np.argmax(row['offline_command4']))]
        confusion[xml_name + '->' + offline_name] += 1
        class_frames[xml_name] += 1
        class_routes[xml_name].add(row['route'])
    horizons = [r['xml_horizon_m'] for r in rows if r['xml_horizon_m'] is not None]
    statuses = Counter(r['certification_status'] for r in rows)
    by_route = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['frame']):
        by_route[row['route']].append(row)
    switches = {'xml': 0, 'offline': 0, 'new_status': 0, 'old_status': 0}
    aligned = {'xml_offline_same_frame': 0, 'routes_with_any_switch': 0, 'routes_with_aligned_switch': 0}
    for route, route_rows in by_route.items():
        seq = {}
        for key, field in (('xml', 'xml_command4'), ('offline', 'offline_command4'),
                           ('new_status', 'new_status_command4'), ('old_status', 'old_status_command4')):
            seq[key] = [(r['frame'], tuple(r[field]) if r[field] is not None else None) for r in route_rows]
        xml_frames = {frame for (frame, value), (prev_frame, prev) in zip(seq['xml'][1:], seq['xml'][:-1]) if value != prev}
        offline_frames = {frame for (frame, value), (prev_frame, prev) in zip(seq['offline'][1:], seq['offline'][:-1]) if value != prev}
        for key in ('xml', 'offline', 'new_status', 'old_status'):
            values = [value for _, value in seq[key] if value is not None]
            switches[key] += sum(1 for a, b in zip(values, values[1:]) if a != b)
        if xml_frames or offline_frames:
            aligned['routes_with_any_switch'] += 1
            if xml_frames & offline_frames:
                aligned['routes_with_aligned_switch'] += 1
                aligned['xml_offline_same_frame'] += len(xml_frames & offline_frames)
    mismatches = Counter()
    for row in comparable:
        if row['xml_vs_offline_command_match'] or row['xml_command4'] is None or row['offline_command4'] is None:
            continue
        xml_name = COMMAND_ORDER[int(np.argmax(row['xml_command4']))]
        offline_name = COMMAND_ORDER[int(np.argmax(row['offline_command4']))]
        mismatches[xml_name + '->' + offline_name] += 1
    return {
        'schema': 'original-route-certification-summary-v1',
        'dataset_routes': len(association),
        'accepted_routes': len(accepted),
        'accepted_by_town': dict(Counter(r['town'] for r in accepted)),
        'route_id_equal_routes': sum(1 for r in association if r['xml_id_matches_route_id']),
        'frames_compared': len(rows),
        'frames_with_offline_command': len(comparable),
        'command_match_frames': len(matched),
        'command_match_rate': (len(matched) / len(comparable)) if comparable else 0.0,
        'point_distance_median_m': float(np.median(distances)) if distances else None,
        'point_distance_p90_m': float(np.percentile(distances, 90)) if distances else None,
        'frames_by_xml_command': dict(by_command),
        'low_speed_frames': len(low_speed),
        'low_speed_command_match': sum(1 for r in low_speed if r['xml_vs_offline_command_match']),
        'status_rows_available': sum(1 for r in rows if r['status_available']),
        'new_vs_xml_command_match': sum(1 for r in rows if r['new_status_command4'] is not None
                                        and r['xml_command4'] is not None
                                        and r['new_status_command4'] == r['xml_command4']),
        'old_vs_xml_command_match': sum(1 for r in rows if r['old_status_command4'] is not None
                                        and r['xml_command4'] is not None
                                        and r['old_status_command4'] == r['xml_command4']),
        'xml_preview_pass_frames': len(xml_pass),
        'xml_preview_pass_command_match': len(xml_pass_match),
        'xml_preview_pass_command_match_rate': (len(xml_pass_match) / len(xml_pass)) if xml_pass else 0.0,
        'xml_point_distance_le_10m_frames': len(near),
        'xml_point_distance_le_10m_command_match': len(near_match),
        'xml_point_distance_le_10m_command_match_rate': (len(near_match) / len(near)) if near else 0.0,
        'mismatch_pairs': dict(mismatches.most_common(10)),
        'confusion_matrix_xml_to_offline': dict(sorted(confusion.items())),
        'classes': {name: {'frames': class_frames.get(name, 0), 'routes': len(class_routes.get(name, ()))}
                    for name in COMMAND_ORDER},
        'xml_horizon_median_m': float(np.median(horizons)) if horizons else None,
        'xml_horizon_p10_p90_m': [float(np.percentile(horizons, 10)), float(np.percentile(horizons, 90))]
                                 if horizons else None,
        'certification_status_counts': dict(statuses),
        'switch_events': switches,
        'switch_alignment': aligned,
        'online_plan_captured': any(r.get('plan_source') == 'ONLINE_CAPTURED_PLAN' for r in rows),
        'online_match_verified_rows': statuses.get('ONLINE_MATCH_VERIFIED', 0),
        'plan_sources': dict(Counter(r.get('plan_metadata_source') for r in rows)),
        'certified_rows': 0,
        'certification_status': 'NOT_VERIFIED',
        'source_boundary': ('Original leaderboard route XML is an offline ordered-plan proxy. '
                            'The live agent receives route_scenario.route densified by CARLA '
                            'GlobalRoutePlanner, which this tool cannot reproduce without CARLA.'),
        'errors': errors,
    }


def load_captures(capture_dir):
    """Read captured online plans: route -> ordered (x, y) world points."""
    plans = {}
    for path in sorted(Path(capture_dir).glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('schema') != 'online-global-plan-v1':
            continue
        meta = record.get('metadata') or {}
        route = meta.get('route') or meta.get('route_id')
        points = [(float(p[0]), float(p[1])) for p in record.get('world_points', [])]
        if route and len(points) >= 2:
            plans[route] = {'points': points, 'sha256': record.get('sha256'),
                            'source': 'ONLINE_CAPTURED_PLAN', 'xml_id': None,
                            'xml_points': len(points),
                            'metadata_source': meta.get('source'),
                            'carla_server_version': meta.get('carla_server_version'),
                            'map_name': meta.get('map_name')}
    return plans


def captured_records(plans):
    """Build comparison records for captured plans (no XML association needed)."""
    records = []
    for route, plan in sorted(plans.items()):
        town = 'Town' + route.split('_Town')[1].split('_')[0] if '_Town' in route else None
        records.append({'route': route, 'town': town,
                        'route_id': route.split('_Route')[1].split('_')[0] if '_Route' in route else None,
                        'poses_used': 0, 'xml_id_matches_route_id': False,
                        'best': {'xml_id': None, 'source': plan['source'], 'coverage': 1.0,
                                 'median_m': 0.0, 'xml_points': plan['xml_points']},
                        'accepted': True, 'candidates_in_town': 0,
                        'plan_sha256': plan['sha256'],
                        'plan_metadata_source': plan.get('metadata_source'),
                        'carla_server_version': plan.get('carla_server_version'),
                        'map_name': plan.get('map_name')})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path,
                        default=Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/v1'))
    parser.add_argument('--preview-root', type=Path,
                        default=Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/route_preview_v1'))
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight/baseline_pssd_manifest.json')
    parser.add_argument('--overlay', type=Path,
                        default=ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261023_status_integration/overlay/full_status_overlay_manifest.jsonl')
    parser.add_argument('--routes-from', type=Path,
                        default=ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261023_status_integration/navigation/full_route_geometry_audit.jsonl')
    parser.add_argument('--capture-dir', type=Path, default=None,
                        help='captured online plan JSONs; replaces XML association when given')
    parser.add_argument('--poses', type=int, default=5)
    parser.add_argument('--frames-per-route', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    routes = [json.loads(line)['route'] for line in args.routes_from.open() if line.strip()]
    if args.capture_dir is not None:
        captures = load_captures(args.capture_dir)
        if not captures:
            raise SystemExit('no online-global-plan-v1 capture found in ' + str(args.capture_dir))
        xml = {}
        for route, plan in captures.items():
            xml[(None, route)] = {'source': plan['source'], 'points': plan['points']}
        association = captured_records(captures)
        assoc_errors = []
    else:
        xml = load_xml(args.xml_dir)
        by_town = defaultdict(list)
        for (town, xml_id), item in xml.items():
            by_town[town].append((xml_id, item))
        association, assoc_errors = associate(routes, by_town, args.source_root, args.poses)
    rows, compare_errors = compare(association, xml, args.preview_root, manifest, args.overlay,
                                   args.frames_per_route, args.source_root)
    summary = summarise(association, rows, assoc_errors + compare_errors)
    summary['xml_sources'] = {name: sha256(Path(args.xml_dir) / name) for name in XML_FILES}
    summary['association_rule'] = {'poses_sampled': args.poses, 'tolerance_m': TOLERANCE_M,
                                   'accept': 'coverage >= 0.8 and median distance <= 8.0 m'}
    summary['frames_per_route'] = args.frames_per_route
    (args.output / 'original_route_source_association.json').write_text(
        json.dumps({'schema': 'original-route-source-association-v1', 'records': association}, indent=2) + '\n')
    with (args.output / 'original_route_xml_comparison.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    (args.output / 'navigation_certification_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('dataset_routes', 'accepted_routes', 'frames_compared',
                                              'command_match_rate', 'point_distance_median_m',
                                              'new_vs_xml_command_match', 'certification_status')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
