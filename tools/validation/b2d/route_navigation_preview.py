"""Read-only two-level route navigation reconstruction prototype.

Level 1 uses the complete expert pose sequence and the compact HD-map Center
export to recover road-level junction decisions. Level 2 projects the current
rear axle onto the reconstructed path and emits a 20 m preview.

The current V1 command vector has [Left, Straight, Right, Unknown]. UTurn is
therefore emitted only in an explicitly incompatible experimental vector; the
V1-compatible vector maps UTurn to Unknown and marks it non-compatible.

No Feature/Target cache, model, checkpoint, manifest or training process is
written. The reconstructed reference path is offline evidence and is not a
certified online expected-route source.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.b2d_adapter.coordinate_transform import actor_pose, actor_to_reference  # noqa: E402
from tools.b2d_adapter.experimental_route_navigation import (  # noqa: E402
    CompactMap, local_preview, reconstruct,
)
from tools.b2d_adapter.experimental_route_preview import RoutePolyline, bearing_command  # noqa: E402

PANEL = ROOT / 'outputs/diffusiondrive_v1/scratch_attribution/20261022_142557_panel/behavior_panel.json'
CONFIG = ROOT / 'tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_scratch.json'
COMPACT = ROOT / 'outputs/diffusiondrive_v1/route_preview_data_audit/20261022_readonly'
MAPS = Path('/data/datasets/Bench2Drive-V0.0.4/maps')
XML_DIR = Path('/home/aethelind/work/environment-build_test/Bench2Drive/leaderboard/data')
DT = .1
AXLE = (-1.388632877, 0.0)
TOWN_RE = re.compile(r'_(Town\d+[A-Za-z]*)_')
ROUTE_RE = re.compile(r'_Route(\d+)_')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def town_of(route):
    match = TOWN_RE.search(route)
    if not match:
        raise ValueError('town not in route name')
    return match.group(1)


def route_id_of(route):
    match = ROUTE_RE.search(route)
    return match.group(1) if match else None


def xml_route_ids(directory):
    ids = set()
    for path in sorted(Path(directory).glob('*.xml')):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for node in root.findall('route'):
            if node.get('id'):
                ids.add(node.get('id'))
    return ids


def read_route(route_dir):
    paths = sorted((route_dir / 'anno').glob('*.json.gz'))
    poses, z_values, frames, raw_current = [], [], [], {}
    for path in paths:
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                anno = json.load(stream)
        except (OSError, ValueError, EOFError):
            continue
        ego = [b for b in anno.get('bounding_boxes', [])
               if isinstance(b, dict) and b.get('class') == 'ego_vehicle']
        if len(ego) != 1:
            continue
        actor = actor_pose(ego[0])
        rear = actor_to_reference(actor, AXLE)
        frame = int(path.stem.split('.')[0])
        poses.append((rear.x, rear.y, rear.yaw))
        z_values.append(float(ego[0].get('location', [0, 0, 0])[2]))
        frames.append(frame)
        raw_current[frame] = {'anno': anno, 'rear': rear, 'actor': actor}
    return frames, poses, z_values, raw_current


def near_far(row, current):
    anno = current['anno']
    rear = current['rear']
    values = {}
    for prefix in ('near', 'far'):
        try:
            point = (float(anno['x_command_' + prefix]), float(anno['y_command_' + prefix]))
        except (KeyError, TypeError, ValueError):
            values[prefix] = None
            continue
        values[prefix] = bearing_command(point, (rear.x, rear.y), rear.yaw)
    return values


def write_csv(path, rows, fields):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def map_for_town(town, cache):
    if town not in cache:
        compact = COMPACT / f'{town}_centers.npz'
        metadata = COMPACT / f'{town}_center_metadata.json'
        if not compact.is_file() or not metadata.is_file():
            cache[town] = None
        else:
            cache[town] = CompactMap(compact, metadata,
                                     source=f'{town}_HD_map.npz -> audit compact {compact.name}')
            cache[town].sha256 = sha256(compact)
    return cache[town]


def reconstruct_route_path(route, dataset_root, map_cache):
    """Reconstruct one route's ordered path with the existing V1.1 pipeline.

    This is a thin orchestration of ``read_route``, ``map_for_town`` and
    ``reconstruct``; it does not introduce a second RouteBuilder.  It returns
    the path and the exact geometry/provenance evidence needed to decide
    whether a frame may be recovered under the V3 contract.  Raw annotations
    are discarded after pose extraction.
    """
    route_dir = Path(dataset_root) / route
    frames, poses, z_values, raw_current = read_route(route_dir)
    del raw_current
    if len(frames) < 2 or len(frames) != len(poses):
        raise ValueError('route has fewer than two complete ego poses')
    town = town_of(route)
    town_map = map_for_town(town, map_cache)
    if town_map is None:
        raise ValueError('compact map unavailable for ' + town)
    result = reconstruct(town_map, poses, z_values, route_name=route)
    gaps = list(result.get('reference_path_gaps_m') or [])
    max_gap = max(gaps, default=0.0)
    points = result.get('reference_path_xy') or []
    path_valid = bool(len(points) >= 2 and max_gap <= 1.0)
    ambiguous = any(event.get('source') == 'AMBIGUOUS'
                    for event in result.get('events') or [])
    match_rate = float(result.get('match_rate') or 0.0)
    path = (RoutePolyline.from_waypoints(
        points, source='reconstructed_offline', dense=False)
        if len(points) >= 2 else None)
    if path is None or match_rate < .5:
        provenance = 'NOT_VERIFIED'
    elif ambiguous:
        provenance = 'AMBIGUOUS'
    elif path_valid:
        provenance = 'RECONSTRUCTED_ROUTE'
    else:
        # The full route contains a topology gap, but the ordered polyline is
        # still usable for a per-frame local window if that window is clear.
        provenance = 'RECONSTRUCTED_ROUTE_LOCAL'
    return {
        'route': route, 'town': town, 'frames': frames, 'poses': poses,
        'path': path, 'path_valid': path_valid, 'max_gap_m': max_gap,
        'match_rate': match_rate, 'ambiguous': ambiguous,
        'provenance': provenance, 'events': result.get('events') or [],
        'reference_path_xy': points,
    }


def route_decision_history(route_path, frames, poses, raw_current, events, preview_m=20.0):
    """Evaluate pre-junction frames without using future frames in the call."""
    previews = []
    if not events or len(route_path.points_xy) < 2:
        return previews
    for event in events:
        maneuver = event['maneuver']
        if maneuver not in ('Left', 'Right', 'Straight'):
            continue
        entry = int(event['entry_frame'])
        lo = max(0, entry - 60)
        first_preview = first_near = None
        for index in range(lo, entry):
            current = raw_current.get(frames[index])
            if current is None:
                continue
            pose = poses[index]
            p = local_preview(route_path, pose[:2], pose[2], preview_m=preview_m)
            near = near_far(current, current)['near']
            if first_preview is None and p.get('category') == maneuver:
                first_preview = index
            if first_near is None and near and near['category_name'] == maneuver:
                first_near = index
            previews.append({
                'event_index': event['event_index'], 'event_entry_frame': entry,
                'frame_index': index, 'frame': frames[index],
                'event_maneuver': maneuver, 'preview_category': p.get('category'),
                'preview_status': p.get('status'), 'near_category': near['category_name'] if near else None,
                'preview_correct': p.get('category') == maneuver,
                'near_correct': bool(near and near['category_name'] == maneuver),
            })
        event['first_preview_correct_frame'] = first_preview
        event['first_preview_lead_s'] = None if first_preview is None else (entry - first_preview) * DT
        event['first_near_correct_frame'] = first_near
        event['first_near_lead_s'] = None if first_near is None else (entry - first_near) * DT
        current = raw_current.get(frames[entry])
        nav = near_far(current, current) if current is not None else {'near': None, 'far': None}
        event['near_at_entry'] = nav['near']['category_name'] if nav['near'] else None
        event['far_at_entry'] = nav['far']['category_name'] if nav['far'] else None
        event['near_at_entry_matches'] = event['near_at_entry'] == maneuver
        event['far_at_entry_matches'] = event['far_at_entry'] == maneuver
        # A separate geometric check over the whole post-entry window. It is
        # not an independent route source because it uses the same expert
        # trajectory; it also handles long stops before the junction.
        exit_index = int(event['exit_frame'])
        probe = min(len(poses) - 1, exit_index + 50)
        yaw0 = poses[entry][2]
        peak = None
        for index in range(entry, probe + 1):
            delta = math.degrees((poses[index][2] - yaw0 + math.pi) % (2 * math.pi) - math.pi)
            if peak is None or abs(delta) > abs(peak):
                peak = delta
        observed = float(peak or 0.0)
        event['post_entry_peak_delta_deg'] = observed
        event['post_entry_peak_maneuver'] = ('Straight' if abs(observed) < 20 else
                                             'UTurn' if abs(observed) >= 135 else
                                             'Right' if observed > 0 else 'Left')
        event['post_entry_consistency'] = event['post_entry_peak_maneuver'] == maneuver
    return previews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=PANEL)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--compact-map-dir', type=Path, default=COMPACT)
    parser.add_argument('--maps', type=Path, default=MAPS)
    parser.add_argument('--xml-dir', type=Path, default=XML_DIR)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    panel = json.loads(args.panel.read_text())
    config = json.loads(args.config.read_text())
    dataset_root = Path(config['dataset_root'])
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    all_routes = sorted({rec['route'] for split in ('validation', 'train') for rec in panel[split]})
    panel_by_route = defaultdict(list)
    for split in ('validation', 'train'):
        for rec in panel[split]:
            panel_by_route[rec['route']].append((split, rec))
    xml_ids = xml_route_ids(args.xml_dir)
    map_cache = {}
    manifest_rows = []
    path_rows = []
    preview_rows = []
    failure_rows = []
    history_rows = []
    for route_number, route in enumerate(all_routes, 1):
        town = town_of(route)
        town_map = map_for_town(town, map_cache)
        route_dir = dataset_root / route
        frames, poses, z_values, raw_current = read_route(route_dir)
        if town_map is None:
            failure_rows.append({'route': route, 'frame': None, 'stage': 'map',
                                 'reason': 'compact_map_unavailable', 'town': town})
            manifest_rows.append({'route': route, 'town': town, 'route_id': route_id_of(route),
                                  'status': 'NOT_VERIFIED', 'reason': 'compact_map_unavailable',
                                  'source': 'NOT_VERIFIED', 'route_provenance': 'NOT_VERIFIED',
                                  'decisions': []})
            continue
        result = reconstruct(town_map, poses, z_values, route_name=route)
        max_path_gap = max(result['reference_path_gaps_m'], default=0.0)
        path_valid = max_path_gap <= 1.0
        ambiguous = any(event.get('source') == 'AMBIGUOUS' for event in result['events'])
        # Keep provenance explicit at Route level. No original XML route is
        # used for decisions; an XML match is only a sparse cross-reference.
        if not path_valid or result['match_rate'] < .5:
            route_provenance = 'NOT_VERIFIED'
        elif ambiguous:
            route_provenance = 'AMBIGUOUS'
        else:
            route_provenance = 'RECONSTRUCTED_ROUTE'
        for event in result['events']:
            event['provenance'] = 'AMBIGUOUS' if event.get('source') == 'AMBIGUOUS' \
                else 'RECONSTRUCTED_ROUTE'
        # Level 2 uses only the current pose plus the reconstructed map path.
        path = RoutePolyline.from_waypoints(result['reference_path_xy'],
                                            source='reconstructed_offline', dense=False) \
            if len(result['reference_path_xy']) >= 2 and path_valid else None
        if path is not None:
            history = route_decision_history(path, frames, poses, raw_current, result['events'])
            for row in history:
                row['route'] = route
            history_rows.extend(history)
        route_preview = []
        frame_to_index = {frame: index for index, frame in enumerate(frames)}
        for split, rec in panel_by_route[route]:
            frame = int(rec['frame'])
            if frame not in raw_current or path is None:
                route_preview.append({'route': route, 'frame': frame, 'status': 'FAIL',
                                      'reason': 'missing_frame_or_path', 'split': split,
                                      'provenance': 'NOT_VERIFIED',
                                      'reference_path_online_available': False,
                                      'reference_path_valid': path_valid})
                continue
            index = frame_to_index[frame]
            pose = poses[index]
            current = raw_current[frame]
            p = local_preview(path, pose[:2], pose[2])
            nav = near_far(rec, current)
            row = {
                'sample_id': rec['sample_id'], 'split': split, 'primary_class': rec['primary_class'],
                'route': route, 'town': town, 'route_id': route_id_of(route), 'frame': frame,
                'matched_road': None, 'matched_lane': None, 'matched_topology_type': None,
                'preview_status': p.get('status'), 'preview_reason': p.get('reason'),
                'preview_category': p.get('category'), 'uturn': p.get('uturn'),
                'v1_compatible': p.get('v1_compatible'),
                's0_m': p.get('s0_m'), 'projection_distance_m': p.get('projection_distance_m'),
                'target_xy': p.get('target_xy'), 'local_xy_m': p.get('local_xy_m'),
                'lateral_m': p.get('lateral_m'), 'heading_delta_deg': p.get('heading_delta_deg'),
                'command4_v1_order': p.get('command4_v1_order'),
                'command4_uturn_order': p.get('command4_uturn_order'),
                'near_category': nav['near']['category_name'] if nav['near'] else None,
                'near_distance_m': nav['near']['distance_m'] if nav['near'] else None,
                'far_category': nav['far']['category_name'] if nav['far'] else None,
                'far_distance_m': nav['far']['distance_m'] if nav['far'] else None,
                'provenance': route_provenance,
                'reference_path_source': 'RECONSTRUCTED_ROUTE_OFFLINE',
                'reference_path_online_available': False,
                'reference_path_max_gap_m': max_path_gap,
                'reference_path_valid': path_valid,
            }
            # Match for the current frame is already part of Level 1; expose it
            # without feeding it into Level 2.
            match = result['matches'][index]
            if match:
                row.update(matched_road=match.road, matched_lane=match.lane,
                           matched_topology_type=match.topology_type)
            route_preview.append(row)
        preview_rows.extend(route_preview)
        manifest = {
            'schema': 'route-navigation-reconstruction-v1', 'route': route, 'town': town,
            'route_id': route_id_of(route), 'source': 'RECONSTRUCTED_ROUTE',
            'route_provenance': route_provenance,
            'original_route_xml_match': route_id_of(route) in xml_ids,
            'map_source': town_map.source, 'map_compact_sha256': town_map.sha256,
            'frames': len(frames), 'matched_frames': result['matched_frames'],
            'match_rate': result['match_rate'],
            'status': ('PASS' if result['match_rate'] >= .8 and result['events'] and path_valid
                       else 'SUSPECT' if result['match_rate'] >= .5 else 'NOT_VERIFIED'),
            'decision_count': len(result['events']), 'decisions': result['events'],
            'reference_path_point_count': len(result['reference_path_xy']),
            'reference_path_gaps_m': result['reference_path_gaps_m'],
            'reference_path_max_gap_m': max_path_gap,
            'reference_path_valid': path_valid,
            'reference_path_source': 'RECONSTRUCTED_ROUTE_OFFLINE',
            'reference_path_online_available': False,
        }
        manifest_rows.append(manifest)
        if path is not None:
            path_rows.append({'route': route, 'town': town, 'points': result['reference_path_xy'],
                              'gaps_m': result['reference_path_gaps_m'],
                              'source': 'RECONSTRUCTED_ROUTE_OFFLINE',
                              'provenance': 'RECONSTRUCTED_ROUTE',
                              'reference_path_online_available': False})
        # Failures are recorded per unmatched frame and per event.
        matched_by_frame = {}
        for index, match in enumerate(result['matches']):
            matched_by_frame[frames[index]] = match
        for index, frame in enumerate(frames):
            if matched_by_frame.get(frame) is None:
                failure_rows.append({'route': route, 'frame': frame, 'stage': 'level1_map_match',
                                     'reason': 'no_centerline_match', 'town': town})
        for failure in result['failures']:
            failure_rows.append({'route': route, 'frame': failure.get('entry_frame'),
                                 'stage': 'level1_decision', 'reason': failure.get('reason'),
                                 'town': town, 'entry_road': failure.get('entry_road'),
                                 'exit_road': failure.get('exit_road')})
        if max_path_gap > 1.0:
            failure_rows.append({'route': route, 'frame': None, 'stage': 'reference_path',
                                 'reason': 'large_centerline_gap', 'town': town,
                                 'entry_road': None, 'exit_road': None})
        for row in route_preview:
            if row.get('preview_status') != 'PASS':
                failure_rows.append({'route': route, 'frame': row.get('frame'), 'stage': 'level2_preview',
                                     'reason': row.get('preview_reason') or row.get('reason'),
                                     'town': town})
        if route_number % 10 == 0:
            print(json.dumps({'event': 'route_progress', 'routes': route_number,
                              'of': len(all_routes), 'route': route}), flush=True)
    # Add current-frame match metadata by a second, cheap pass over matching
    # frames; Level 1 matching is deterministic and this keeps the first pass
    # memory bounded.
    manifest_path = output / 'route_navigation_manifest.jsonl'
    with manifest_path.open('x') as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'route_reference_paths.jsonl').open('x') as stream:
        for row in path_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'frame_route_preview.jsonl').open('x') as stream:
        for row in preview_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'route_navigation_history.jsonl').open('x') as stream:
        for row in history_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    decisions = []
    for manifest in manifest_rows:
        for decision in manifest.get('decisions', []):
            decision['route'] = manifest['route']
            decisions.append(decision)
    write_csv(output / 'event_consistency.csv', decisions,
              ['route', 'event_index', 'entry_frame', 'exit_frame', 'entry_road', 'exit_road',
               'delta_heading_deg', 'maneuver', 'confidence', 'source', 'provenance',
               'entry_path_preview', 'path_consistent', 'post_entry_peak_delta_deg',
               'post_entry_peak_maneuver', 'post_entry_consistency', 'near_at_entry',
               'far_at_entry', 'near_at_entry_matches', 'far_at_entry_matches',
               'first_preview_lead_s', 'first_near_lead_s'])
    write_csv(output / 'route_navigation_failures.csv', failure_rows,
              ['route', 'frame', 'stage', 'reason', 'town', 'entry_road', 'exit_road'])
    counts = Counter(decision.get('maneuver') for decision in decisions)
    statuses = Counter(manifest.get('status') for manifest in manifest_rows)
    provenance_labels = ('ORIGINAL_ROUTE', 'RECONSTRUCTED_ROUTE', 'AMBIGUOUS', 'NOT_VERIFIED')
    provenance_counts = {label: 0 for label in provenance_labels}
    for manifest in manifest_rows:
        provenance_counts[manifest.get('route_provenance', 'NOT_VERIFIED')] += 1
    preview_provenance_counts = {label: 0 for label in provenance_labels}
    for row in preview_rows:
        preview_provenance_counts[row.get('provenance', 'NOT_VERIFIED')] += 1
    decision_provenance_counts = {label: 0 for label in provenance_labels}
    for decision in decisions:
        decision_provenance_counts[decision.get('provenance', 'NOT_VERIFIED')] += 1
    path_valid_routes = sum(manifest.get('reference_path_valid') is True for manifest in manifest_rows)
    path_gap_routes = sum(manifest.get('reference_path_max_gap_m', 0) > 1.0 for manifest in manifest_rows)
    post_entry_consistent = sum(decision.get('post_entry_consistency') is True for decision in decisions)
    path_consistent = sum(decision.get('path_consistent') is True for decision in decisions)
    near_entry_consistent = sum(decision.get('near_at_entry_matches') is True for decision in decisions)
    far_entry_consistent = sum(decision.get('far_at_entry_matches') is True for decision in decisions)
    preview_leads = [decision.get('first_preview_lead_s') for decision in decisions
                     if decision.get('first_preview_lead_s') is not None]
    near_leads = [decision.get('first_near_lead_s') for decision in decisions
                  if decision.get('first_near_lead_s') is not None]
    summary = {
        'schema': 'route-navigation-preview-summary-v1',
        'panel': str(args.panel.resolve()), 'panel_sha256': sha256(args.panel),
        'config': str(args.config.resolve()), 'config_sha256': sha256(args.config),
        'dataset_root': str(dataset_root),
        'map_source': str(args.maps), 'map_compact_dir': str(args.compact_map_dir),
        'routes': len(all_routes), 'panel_frames': sum(len(v) for v in panel_by_route.values()),
        'manifest_status': dict(statuses), 'decision_counts': dict(counts),
        'provenance_labels': list(provenance_labels),
        'provenance_counts': provenance_counts,
        'preview_provenance_counts': preview_provenance_counts,
        'decision_provenance_counts': decision_provenance_counts,
        'original_route_xml_match_routes': sum(
            manifest.get('original_route_xml_match') is True for manifest in manifest_rows),
        'path_valid_routes': path_valid_routes, 'path_gap_routes': path_gap_routes,
        'decision_total': len(decisions),
        'post_entry_peak_consistency': post_entry_consistent,
        'path_consistency': path_consistent,
        'near_at_entry_consistency': near_entry_consistent,
        'far_at_entry_consistency': far_entry_consistent,
        'preview_lead_s_median': float(np.median(preview_leads)) if preview_leads else None,
        'near_lead_s_median': float(np.median(near_leads)) if near_leads else None,
        'preview_lead_events': len(preview_leads), 'near_lead_events': len(near_leads),
        'preview_rows': len(preview_rows),
        'preview_status_counts': dict(Counter(row.get('preview_status') for row in preview_rows)),
        'preview_category_counts': dict(Counter(row.get('preview_category') for row in preview_rows)),
        'failure_rows': len(failure_rows),
        'history_rows': len(history_rows),
        'uturn_contract': 'V1 order [Left,Straight,Right,Unknown]; UTurn is experimental and v1_compatible=false',
        'source_boundary': 'Level 1 may use complete expert trajectory; Level 2 function receives only current pose and reconstructed map path; path is offline and not a certified online plan',
        'provenance_boundary': 'ORIGINAL_ROUTE is not used because no certified original ordered route is available; XML matches are sparse cross-references only. AMBIGUOUS means a path/maneuver check failed; NOT_VERIFIED means no valid reconstruction/path.',
        'protected': 'read-only; no Feature/Target/cache/model/checkpoint/training changes',
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'event': 'route_preview_done', 'routes': len(all_routes),
                      'previews': len(preview_rows), 'decisions': sum(counts.values()),
                      'output': str(output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
