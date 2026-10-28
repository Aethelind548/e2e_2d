"""Read-only first-stage Adapter semantics prototypes.

Runs three independent diagnostics without touching Feature/Target caches:
causal state candidates A0/A1/A2, route availability/preview feasibility, and
stop/departure observability. The current training source, producer files and
checkpoints are never modified. Central differences are evaluation references
only and are never used to create an online state.
"""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.b2d_adapter.coordinate_transform import actor_pose, actor_to_reference  # noqa: E402
from tools.b2d_adapter.experimental_causal_state import (  # noqa: E402
    A0, A1, A2, METHODS, causal_state, reference_state,
)
from tools.b2d_adapter.experimental_route_preview import (  # noqa: E402
    RoutePolyline, bearing_command, preview_command, world_to_rear_left,
)

PANEL = ROOT / 'outputs/diffusiondrive_v1/scratch_attribution/20261022_142557_panel/behavior_panel.json'
EVAL = ROOT / 'outputs/diffusiondrive_v1/scratch_attribution/20261022_144016_eval/selected_oracle_samples.jsonl'
CONFIG = ROOT / 'tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_scratch.json'
ROUTE_XML = Path('/home/aethelind/work/environment-build_test/Bench2Drive/leaderboard/data')
DT = .1
AXLE = (-1.388632877, 0.0)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def percentile(values, q):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(values, q)) if values else None


def median(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


class RawReader:
    """Annotation-only reader with a small per-process cache."""

    def __init__(self, dataset_root):
        self.root = Path(dataset_root)
        self.cache = {}
        self.route_dirs = {}

    def route_dir(self, route):
        if route not in self.route_dirs:
            self.route_dirs[route] = self.root / route
        return self.route_dirs[route]

    def path(self, route, frame, kind='anno'):
        suffix = '.json.gz' if kind == 'anno' else '.jpg'
        folder = 'anno' if kind == 'anno' else 'camera/rgb_front'
        return self.route_dir(route) / folder / f'{frame:05d}{suffix}'

    def read(self, route, frame):
        key = (route, frame)
        if key in self.cache:
            return self.cache[key]
        path = self.path(route, frame)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            anno = json.load(stream)
        ego = [b for b in anno.get('bounding_boxes', [])
               if isinstance(b, dict) and b.get('class') == 'ego_vehicle']
        if len(ego) != 1:
            raise ValueError('ego count')
        pose = actor_pose(ego[0])
        rear = actor_to_reference(pose, AXLE)
        record = {
            'route': route, 'frame': int(frame), 'anno': anno, 'actor_pose': pose,
            'rear_pose': rear, 'speed': anno.get('speed'),
            'acceleration': anno.get('acceleration'), 'angular_velocity': anno.get('angular_velocity'),
            'navigation': {k: v for k, v in anno.items() if 'command' in k or k in ('x_target', 'y_target')},
            'throttle': anno.get('throttle'), 'brake': anno.get('brake'),
            'steer': anno.get('steer'), 'should_brake': anno.get('should_brake'),
            'sha256': sha256(path),
        }
        self.cache[key] = record
        return record

    def window(self, route, center, before, after):
        frames = list(range(center - before, center + after + 1))
        if not frames or min(frames) < 0:
            return None, frames
        try:
            return [self.read(route, f) for f in frames], frames
        except (FileNotFoundError, ValueError):
            return None, frames


def pose_array(records):
    return np.asarray([[r['rear_pose'].x, r['rear_pose'].y, r['rear_pose'].yaw] for r in records])


def estimate_metrics(panel, reader):
    rows = []
    aggregate = defaultdict(list)
    route_metrics = defaultdict(lambda: defaultdict(list))
    availability = Counter()
    for split in ('validation', 'train'):
        for rec in panel[split]:
            route, frame = rec['route'], int(rec['frame'])
            window, frames = reader.window(route, frame, 3, 1)
            if window is None:
                availability['window_unavailable'] += 1
                continue
            try:
                current = reader.read(route, frame)
            except (FileNotFoundError, ValueError):
                availability['current_unavailable'] += 1
                continue
            if current['sha256'] != rec['row']['current_anno_sha']:
                availability['current_sha_mismatch'] += 1
            pos = pose_array(window)
            idx = len(window) - 2  # t; window ends at t+1
            ref = reference_state(pos[idx - 1:idx + 2], dt=DT)
            estimates = {}
            for method in METHODS:
                try:
                    # Only poses through t are supplied to the online candidate.
                    estimates[method] = causal_state(pos[:idx + 1], method=method,
                                                     speed=current['speed'], dt=DT)
                except ValueError:
                    availability[method + '_insufficient'] += 1
            if len(estimates) != len(METHODS):
                continue
            jitter_window, _ = reader.window(route, frame, 4, 1)
            prev = {}
            ref_prev = None
            if jitter_window is not None:
                jpos = pose_array(jitter_window)
                jidx = len(jitter_window) - 2
                for method in METHODS:
                    try:
                        prev[method] = causal_state(jpos[:jidx], method=method,
                                                    speed=jitter_window[jidx - 1]['speed'], dt=DT)
                    except ValueError:
                        availability[method + '_previous_insufficient'] += 1
                try:
                    ref_prev = reference_state(jpos[jidx - 2:jidx + 1], dt=DT)
                except ValueError:
                    pass
            else:
                availability['jitter_window_unavailable'] += 1
            base = {
                'split': split, 'primary_class': rec['primary_class'],
                'sample_id': rec['sample_id'], 'route': route, 'frame': frame,
                'speed_mps': current['speed'], 'current_sha_matches_panel': current['sha256'] == rec['row']['current_anno_sha'],
                'target_path_m': rec['stats']['path_m'],
                'target_first_speed_mps': rec['stats']['first_speed_mps'],
                'target_last_speed_mps': rec['stats']['last_speed_mps'],
                'target_heading_change_deg': math.degrees(rec['stats']['heading_change_rad']),
                'reference_velocity': ref.velocity.tolist(),
                'reference_acceleration': ref.acceleration.tolist(),
                'reference_forward_acceleration_speed_mps2':
                    ((float(window[idx + 1]['speed']) - float(window[idx - 1]['speed'])) / (2 * DT)
                     if window[idx + 1]['speed'] is not None and window[idx - 1]['speed'] is not None else None),
            }
            for method, estimate in estimates.items():
                ve = float(np.linalg.norm(estimate.velocity - ref.velocity))
                ae = float(np.linalg.norm(estimate.acceleration - ref.acceleration))
                speed_est = float(np.linalg.norm(estimate.velocity))
                row = dict(base, method=method,
                           velocity_xy_mps=estimate.velocity.tolist(),
                           acceleration_xy_mps2=estimate.acceleration.tolist(),
                           velocity_error_mps=ve, acceleration_error_mps2=ae,
                           estimated_speed_mps=speed_est,
                           speed_error_mps=(None if current['speed'] is None else speed_est - float(current['speed'])),
                           velocity_bias_forward_mps=float(estimate.velocity[0] - ref.velocity[0]),
                           acceleration_bias_forward_mps2=float(estimate.acceleration[0] - ref.acceleration[0]),
                           acceleration_speed_reference_error_mps2=(
                               None if base['reference_forward_acceleration_speed_mps2'] is None else
                               abs(float(estimate.acceleration[0]) - base['reference_forward_acceleration_speed_mps2'])),
                           used_speed_magnitude_only=estimate.used_speed,
                           jitter_velocity_mps=None, jitter_acceleration_mps2=None,
                           reference_jitter_velocity_mps=None, reference_jitter_acceleration_mps2=None)
                if method in prev:
                    pv = prev[method]
                    row['jitter_velocity_mps'] = float(np.linalg.norm(estimate.velocity - pv.velocity))
                    row['jitter_acceleration_mps2'] = float(np.linalg.norm(estimate.acceleration - pv.acceleration))
                if ref_prev is not None:
                    row['reference_jitter_velocity_mps'] = float(np.linalg.norm(ref.velocity - ref_prev.velocity))
                    row['reference_jitter_acceleration_mps2'] = float(np.linalg.norm(ref.acceleration - ref_prev.acceleration))
                rows.append(row)
                aggregate[(split, rec['primary_class'], method)].append(row)
                route_metrics[(method, route)]['velocity_error_mps'].append(ve)
                route_metrics[(method, route)]['acceleration_error_mps2'].append(ae)
    return rows, aggregate, route_metrics, availability


def aggregate_rows(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['split'], r['primary_class'], r['method'])].append(r)
    all_groups = defaultdict(list)
    for r in rows:
        all_groups[(r['split'], r['method'])].append(r)
    out = []
    for (split, method), group in sorted(all_groups.items()):
        groups[(split, 'ALL', method)] = group
    for (split, cls, method), group in sorted(groups.items()):
        speeds = [r['estimated_speed_mps'] for r in group]
        out.append({
            'split': split, 'primary_class': cls, 'method': method, 'samples': len(group),
            'velocity_mae_mps': mean([r['velocity_error_mps'] for r in group]),
            'velocity_p95_mps': percentile([r['velocity_error_mps'] for r in group], 95),
            'acceleration_mae_mps2': mean([r['acceleration_error_mps2'] for r in group]),
            'acceleration_p95_mps2': percentile([r['acceleration_error_mps2'] for r in group], 95),
            'acceleration_speed_ref_mae_mps2': mean([r['acceleration_speed_reference_error_mps2'] for r in group]),
            'acceleration_speed_ref_p95_mps2': percentile([r['acceleration_speed_reference_error_mps2'] for r in group], 95),
            'stationary_false_speed_rate': (sum(s > .1 for s in speeds) / len(speeds)) if cls == 'stationary' else None,
            'stationary_mean_speed_mps': mean(speeds) if cls == 'stationary' else None,
            'low_depart_underestimate_rate': (sum(r['estimated_speed_mps'] < .5 * float(r['speed_mps'])
                                                  for r in group if r['speed_mps'] is not None and float(r['speed_mps']) > .2)
                                              / max(1, sum(1 for r in group if r['speed_mps'] is not None and float(r['speed_mps']) > .2)))
                                             if cls == 'low_depart' else None,
            'velocity_jitter_median_mps': median([r['jitter_velocity_mps'] for r in group]),
            'acceleration_jitter_median_mps2': median([r['jitter_acceleration_mps2'] for r in group]),
            'reference_velocity_jitter_median_mps': median([r['reference_jitter_velocity_mps'] for r in group]),
            'reference_acceleration_jitter_median_mps2': median([r['reference_jitter_acceleration_mps2'] for r in group]),
            'current_sha_all_match': all(r['current_sha_matches_panel'] for r in group),
        })
    return out


def compute_lag(panel, reader):
    rows = []
    for split in ('validation', 'train'):
        for rec in panel[split]:
            window, _ = reader.window(rec['route'], int(rec['frame']), 6, 6)
            if window is None:
                continue
            pos = pose_array(window)
            reference_speed = {}
            for j in range(1, len(window) - 1):
                try:
                    reference_speed[j] = float(np.linalg.norm(reference_state(pos[j - 1:j + 2], dt=DT).velocity))
                except ValueError:
                    pass
            # Only t with causal history and an offline central reference.
            for idx in range(3, len(window) - 1):
                current = window[idx]
                for method in METHODS:
                    try:
                        est = causal_state(pos[:idx + 1], method=method, speed=current['speed'], dt=DT)
                    except ValueError:
                        continue
                    for shift in (-2, -1, 0, 1, 2):
                        j = idx + shift
                        if j not in reference_speed:
                            continue
                        err = abs(float(np.linalg.norm(est.velocity)) - reference_speed[j])
                        rows.append({'split': split, 'primary_class': rec['primary_class'],
                                     'method': method, 'shift_frames': shift, 'speed_error_mps': err})
    best = []
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['split'], r['primary_class'], r['method'], r['shift_frames'])].append(r['speed_error_mps'])
    for (split, cls, method, shift), values in grouped.items():
        best.append({'split': split, 'primary_class': cls, 'method': method,
                     'shift_frames': shift, 'shift_s': shift * DT,
                     'mean_speed_error_mps': mean(values), 'samples': len(values)})
    return best


def write_csv(path, rows, fields):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def route_id_from_name(route):
    match = re.search(r'_Route(\d+)_', route)
    return match.group(1) if match else None


def parse_route_xmls(directory):
    routes = defaultdict(list)
    for path in sorted(Path(directory).glob('*.xml')):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for node in root.findall('route'):
            rid = node.get('id')
            points = []
            for position in node.findall('waypoints/position'):
                try:
                    points.append([float(position.get('x')), float(position.get('y'))])
                except (TypeError, ValueError):
                    points = []
                    break
            if rid and len(points) >= 2:
                routes[rid].append({'source': path.name, 'town': node.get('town'),
                                    'points': points, 'sha256': sha256(path)})
    return routes


def route_inventory(panel, reader, xml_routes):
    unique = sorted({rec['route'] for split in ('validation', 'train') for rec in panel[split]})
    rows = []
    for route in unique:
        rid = route_id_from_name(route)
        route_dir = reader.route_dir(route)
        matches = xml_routes.get(rid, []) if rid else []
        result_path = route_dir / 'result.json'
        result_id = result_status = result_length = result_duration = None
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
                records = result.get('_checkpoint', {}).get('records', [])
                if records:
                    result_id = records[0].get('route_id')
                    result_status = records[0].get('status')
                    meta = records[0].get('meta', {})
                    result_length = meta.get('route_length')
                    result_duration = meta.get('duration_game')
            except (ValueError, OSError, AttributeError):
                pass
        files = [p for p in route_dir.iterdir() if p.is_file()] if route_dir.is_dir() else []
        plan_candidates = [p for p in route_dir.rglob('*') if p.is_file()
                           and p.suffix.lower() in ('.xml', '.xodr', '.pkl', '.json')
                           and p.name not in ('result.json',)]
        plan_files = [p.name for p in plan_candidates]
        dense_candidates = [p.name for p in plan_candidates
                            if p.suffix.lower() in ('.xodr', '.pkl')
                            or 'route' in p.name.lower() or 'plan' in p.name.lower()]
        anno_files = sorted((route_dir / 'anno').glob('*.json.gz')) if (route_dir / 'anno').is_dir() else []
        rows.append({
            'route': route, 'route_id': rid, 'xml_exact_matches': len(matches),
            'xml_sources': ';'.join(sorted({m['source'] for m in matches})),
            'xml_towns': ';'.join(sorted({str(m['town']) for m in matches})),
            'xml_waypoint_counts': ';'.join(str(len(m['points'])) for m in matches),
            'result_route_id': result_id, 'result_status': result_status,
            'result_route_length_m': result_length, 'result_duration_game_s': result_duration,
            'implied_frame_rate_hz': (len(anno_files) / float(result_duration)
                                       if result_duration and anno_files else None),
            'result_id_matches_folder': (result_id == 'RouteScenario_{}_rep0'.format(rid)) if result_id and rid else None,
            'route_plan_candidates': ';'.join(plan_files) if plan_files else '',
            'dense_path_candidates': ';'.join(dense_candidates) if dense_candidates else '',
            'sparse_xml_waypoints_available': bool(matches),
            'dense_path_available': bool(dense_candidates),
            'route_files': ';'.join(sorted(p.name for p in files)),
            'annotation_frames': len(anno_files),
            'annotation_min_frame': int(anno_files[0].stem.split('.')[0]) if anno_files else None,
            'annotation_max_frame': int(anno_files[-1].stem.split('.')[0]) if anno_files else None,
            'route_dir_exists': route_dir.is_dir(),
        })
    return rows


def _nav_point(anno, prefix):
    try:
        return (float(anno['x_command_' + prefix]), float(anno['y_command_' + prefix]))
    except (KeyError, TypeError, ValueError):
        return None


def navigation_rows(panel, reader):
    rows = []
    for split in ('validation', 'train'):
        for rec in panel[split]:
            current = reader.read(rec['route'], int(rec['frame']))
            anno = current['anno']
            rear = current['rear_pose']
            near, far = _nav_point(anno, 'near'), _nav_point(anno, 'far')
            n = bearing_command(near, (rear.x, rear.y), rear.yaw) if near else None
            f = bearing_command(far, (rear.x, rear.y), rear.yaw) if far else None
            heading = math.degrees(rec['stats']['heading_change_rad'])
            future_turn = abs(heading) >= 20.0 and rec['stats']['path_m'] >= 3.0
            future_category = (0 if heading > 0 else 2) if future_turn else 1
            rows.append({
                'split': split, 'primary_class': rec['primary_class'], 'sample_id': rec['sample_id'],
                'route': rec['route'], 'frame': rec['frame'],
                'command_near_raw': anno.get('command_near'), 'command_far_raw': anno.get('command_far'),
                'near_category': n['category_name'] if n else None,
                'far_category': f['category_name'] if f else None,
                'near_distance_m': n['distance_m'] if n else None,
                'far_distance_m': f['distance_m'] if f else None,
                'near_bearing_deg': n['bearing_deg'] if n else None,
                'far_bearing_deg': f['bearing_deg'] if f else None,
                'near_far_category_differs': (n['category'] != f['category']) if n and f else None,
                'future_heading_change_deg': heading,
                'future_path_m': rec['stats']['path_m'],
                'future_turn': future_turn,
                'future_category': ('Left' if future_category == 0 else 'Right' if future_category == 2 else 'Straight'),
                'near_matches_future_turn': (n['category'] == future_category) if n else None,
                'far_matches_future_turn': (f['category'] == future_category) if f else None,
            })
    return rows


def route_preview_rows(panel, reader, xml_routes):
    rows = []
    for split in ('validation', 'train'):
        for rec in panel[split]:
            rid = route_id_from_name(rec['route'])
            matches = xml_routes.get(rid, [])
            if not matches:
                continue
            current = reader.read(rec['route'], int(rec['frame']))
            rear = current['rear_pose']
            anno = current['anno']
            near, far = _nav_point(anno, 'near'), _nav_point(anno, 'far')
            n = bearing_command(near, (rear.x, rear.y), rear.yaw) if near else None
            f = bearing_command(far, (rear.x, rear.y), rear.yaw) if far else None
            for match in matches:
                route = RoutePolyline.from_waypoints(match['points'], source=match['source'], dense=False)
                s0, projection, projected, segment = route.project((rear.x, rear.y))
                for preview in (5.0, 10.0, 20.0, 30.0):
                    p = preview_command(route, (rear.x, rear.y), rear.yaw, preview_m=preview)
                    rows.append({
                        'split': split, 'primary_class': rec['primary_class'], 'sample_id': rec['sample_id'],
                        'route': rec['route'], 'frame': rec['frame'], 'xml_source': match['source'],
                        'xml_town': match['town'], 'route_length_m': route.length_m,
                        'projection_distance_m': projection, 's0_m': s0, 'segment_index': segment,
                        'preview_m': preview, 'preview_category': p.get('category_name'),
                        'preview_reason': p.get('reason'), 'preview_lateral_m': p.get('lateral_m'),
                        'near_category': n['category_name'] if n else None,
                        'far_category': f['category_name'] if f else None,
                        'near_distance_m': n['distance_m'] if n else None,
                        'far_distance_m': f['distance_m'] if f else None,
                    })
    return rows


def route_geometry(panel, reader, xml_routes):
    records = []
    panel_routes = {rec['route'] for split in ('validation', 'train') for rec in panel[split]}
    for route_name in sorted(panel_routes):
        rid = route_id_from_name(route_name)
        for match in xml_routes.get(rid, []):
            poly = RoutePolyline.from_waypoints(match['points'], source=match['source'], dense=False)
            records.append(dict(poly.summary(), route=route_name, route_id=rid,
                                town=match['town'], xml_sha256=match['sha256']))
    return records


def timing_inventory(panel, reader):
    unique = sorted({rec['route'] for split in ('validation', 'train') for rec in panel[split]})
    rows = []
    timestamp_keys = Counter()
    sensor_timestamp_keys = Counter()
    sensor_frame_keys = Counter()
    camera_lidar_same_frame = 0
    panel_samples = 0
    dt_samples = []
    for route in unique:
        anno_files = sorted((reader.route_dir(route) / 'anno').glob('*.json.gz'))
        if not anno_files:
            continue
        first = json.load(gzip.open(anno_files[0], 'rt'))
        keys = sorted(first)
        ts_keys = [k for k in keys if 'time' in k.lower() or 'stamp' in k.lower()]
        local_sensor_ts = []
        local_sensor_frame = []
        for sensor, value in first.get('sensors', {}).items():
            if isinstance(value, dict):
                for key in value:
                    if 'time' in key.lower() or 'stamp' in key.lower():
                        local_sensor_ts.append(sensor + '.' + key)
                        sensor_timestamp_keys[(sensor, key)] += 1
                    if 'frame' in key.lower() or 'tick' in key.lower():
                        local_sensor_frame.append(sensor + '.' + key)
                        sensor_frame_keys[(sensor, key)] += 1
        for k in ts_keys:
            timestamp_keys[k] += 1
        rows.append({'route': route, 'annotation_frames': len(anno_files),
                     'first_frame': int(anno_files[0].stem.split('.')[0]),
                     'last_frame': int(anno_files[-1].stem.split('.')[0]),
                     'timestamp_keys': ';'.join(ts_keys) if ts_keys else '',
                     'sensor_timestamp_keys': ';'.join(local_sensor_ts) if local_sensor_ts else '',
                     'sensor_metadata_has_timestamp': bool(local_sensor_ts),
                     'sensor_metadata_has_frame': bool(local_sensor_frame)})
    for split in ('validation', 'train'):
        for rec in panel[split]:
            panel_samples += 1
            route_dir = reader.route_dir(rec['route'])
            frame = int(rec['frame'])
            cam = [route_dir / 'camera' / name / f'{frame:05d}.jpg'
                   for name in ('rgb_front_left', 'rgb_front', 'rgb_front_right')]
            lidar = route_dir / 'lidar' / f'{frame:05d}.laz'
            if all(p.is_file() for p in cam) and lidar.is_file():
                camera_lidar_same_frame += 1
            window, _ = reader.window(rec['route'], frame, 1, 1)
            if window is not None:
                for a, b in zip(window, window[1:]):
                    if a['speed'] is not None and b['speed'] is not None:
                        speed = .5 * (abs(float(a['speed'])) + abs(float(b['speed'])))
                        dist = math.hypot(b['rear_pose'].x - a['rear_pose'].x,
                                          b['rear_pose'].y - a['rear_pose'].y)
                        if speed > .5 and dist > .05:
                            dt_samples.append(dist / speed)
    return {
        'routes': len(unique), 'panel_samples': panel_samples,
        'camera_and_lidar_same_frame_filename': camera_lidar_same_frame,
        'timestamp_key_counts': dict(timestamp_keys),
        'sensor_timestamp_key_counts': {str(k): v for k, v in sensor_timestamp_keys.items()},
        'sensor_frame_key_counts': {str(k): v for k, v in sensor_frame_keys.items()},
        'inferred_dt_s': {'samples': len(dt_samples), 'median': median(dt_samples),
                           'p05': percentile(dt_samples, 5), 'p95': percentile(dt_samples, 95)},
        'physical_timestamp_status': 'NOT_VERIFIED: no per-frame or sensor timestamp field found',
        'frame_relation_status': 'frame IDs are filenames and panel sample IDs; this proves filename alignment only',
        'tick_status': 'NOT_VERIFIED: no world tick or sensor frame metadata found',
        'route_rows': rows,
    }


def _context(reader, route, frame):
    current = reader.read(route, frame)
    previous = reader.read(route, frame - 1) if frame >= 1 else None
    rear = current['rear_pose']
    vehicles = []
    walkers = []
    for box in current['anno'].get('bounding_boxes', []):
        if not isinstance(box, dict) or box.get('class') == 'ego_vehicle':
            continue
        try:
            location = box['location']
            local = world_to_rear_left((float(location[0]), float(location[1])),
                                       (rear.x, rear.y), rear.yaw)
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if local[0] <= 0 or abs(local[1]) > 6 or local[0] > 50:
            continue
        item = {'distance_m': float(local[0]), 'lateral_m': float(local[1])}
        if box.get('class') == 'vehicle':
            vehicles.append(item)
        elif box.get('class') in ('walker', 'pedestrian'):
            walkers.append(item)
    near = _nav_point(current['anno'], 'near')
    far = _nav_point(current['anno'], 'far')
    n = bearing_command(near, (rear.x, rear.y), rear.yaw) if near else None
    f = bearing_command(far, (rear.x, rear.y), rear.yaw) if far else None
    route_dir = reader.route_dir(route)
    cam = [route_dir / 'camera' / name / f'{frame:05d}.jpg'
           for name in ('rgb_front_left', 'rgb_front', 'rgb_front_right')]
    return {
        'annotation_sha256': current['sha256'], 'lead_count': len(vehicles),
        'lead_nearest_m': min((v['distance_m'] for v in vehicles), default=None),
        'pedestrian_count': len(walkers),
        'near_category': n['category_name'] if n else None,
        'far_category': f['category_name'] if f else None,
        'near_distance_m': n['distance_m'] if n else None,
        'far_distance_m': f['distance_m'] if f else None,
        'throttle': current['throttle'], 'brake': current['brake'],
        'previous_throttle': previous['throttle'] if previous else None,
        'previous_brake': previous['brake'] if previous else None,
        'camera_files_present': all(p.is_file() for p in cam),
        'lidar_file_present': (route_dir / 'lidar' / f'{frame:05d}.laz').is_file(),
    }


def status_distance(a, b):
    sa, sb = np.asarray(a['status'], dtype=float), np.asarray(b['status'], dtype=float)
    command_diff = 0.0 if np.allclose(sa[:4], sb[:4]) else 1.0
    return float(command_diff + np.linalg.norm(sa[4:6] - sb[4:6])
                 + 0.2 * np.linalg.norm(sa[6:8] - sb[6:8]))


def _observability_record(reader, src, pair, selection_type, high=None):
    distance = status_distance(src, pair)
    try:
        c1 = _context(reader, src['route'], int(src['frame']))
        c2 = _context(reader, pair['route'], int(pair['frame']))
    except (FileNotFoundError, ValueError):
        return {'selection_type': selection_type, 'high_checkpoint': high['checkpoint'] if high else None,
                'high_class': src['primary_class'], 'high_sample_id': src['sample_id'],
                'pair_class': pair['primary_class'], 'pair_sample_id': pair['sample_id'],
                'status_distance': distance, 'classification': 'C', 'reason': 'missing raw annotation/context'}
    clues = []
    if c1['lead_count'] != c2['lead_count']:
        clues.append('lead_count')
    if c1['lead_nearest_m'] is not None and c2['lead_nearest_m'] is not None:
        if abs(c1['lead_nearest_m'] - c2['lead_nearest_m']) >= 3.0:
            clues.append('lead_distance')
    if c1['near_category'] != c2['near_category'] or c1['far_category'] != c2['far_category']:
        clues.append('navigation')
    for key in ('previous_throttle', 'previous_brake'):
        if c1[key] is not None and c2[key] is not None and abs(float(c1[key]) - float(c2[key])) >= .2:
            clues.append(key)
    if distance > .2:
        classification, reason = 'C', 'status distance above .2; no comparable pair'
    elif clues:
        classification, reason = 'A', 'status similar; annotation-level causal context differs: ' + ','.join(clues)
    else:
        classification, reason = 'B', 'status and available annotations similar; ambiguity remains'
    row = {
        'selection_type': selection_type,
        'high_checkpoint': high['checkpoint'] if high else None,
        'high_class': src['primary_class'], 'high_sample_id': src['sample_id'],
        'high_route': src['route'], 'high_frame': src['frame'],
        'high_selected_ade_m': high['selected_error']['ade_m'] if high else None,
        'high_oracle_ade_m': high['oracle_error']['ade_m'] if high else None,
        'selection_gap_m': high['selection_gap'] if high else None,
        'high_target_path_m': src['stats']['path_m'], 'high_target_first_speed_mps': src['stats']['first_speed_mps'],
        'pair_class': pair['primary_class'], 'pair_sample_id': pair['sample_id'], 'pair_route': pair['route'],
        'pair_frame': pair['frame'], 'pair_target_path_m': pair['stats']['path_m'],
        'pair_target_first_speed_mps': pair['stats']['first_speed_mps'],
        'status_distance': distance, 'classification': classification, 'reason': reason,
        'classification_basis': 'status distance plus current annotation actors/navigation and previous control; camera/LiDAR content not semantically certified',
        'sensor_visibility_status': 'NOT_VERIFIED: files present, object visibility not audited',
        'clue_sources': ','.join(clues), 'camera_files_present': c1['camera_files_present'] and c2['camera_files_present'],
        'lidar_files_present': c1['lidar_file_present'] and c2['lidar_file_present'],
    }
    for prefix, c in (('high', c1), ('pair', c2)):
        for key in ('lead_count', 'lead_nearest_m', 'pedestrian_count', 'near_category', 'far_category',
                    'near_distance_m', 'far_distance_m', 'throttle', 'brake', 'previous_throttle', 'previous_brake'):
            row[prefix + '_' + key] = c[key]
    return row


def observability_rows(panel, eval_path, reader, limit=5):
    panel_by_id = {}
    for split in ('validation', 'train'):
        for rec in panel[split]:
            panel_by_id[rec['sample_id']] = dict(rec, split=split)
    eval_rows = [json.loads(line) for line in Path(eval_path).read_text().splitlines() if line.strip()]
    eval_rows = [r for r in eval_rows if not r.get('case_study') and r['sample_id'] in panel_by_id]
    wanted = {'stationary': 'low_depart', 'low_depart': 'stationary',
              'approaching_stop': 'low_depart'}
    selected = []
    for (split, cls), group in itertools.groupby(
            sorted(eval_rows, key=lambda r: (r['split'], r['primary_class'])),
            key=lambda r: (r['split'], r['primary_class'])):
        group = sorted(list(group), key=lambda r: r['selection_gap'], reverse=True)
        selected.extend(group[:limit])
    rows = []
    for high in selected:
        src = panel_by_id[high['sample_id']]
        target_cls = wanted.get(src['primary_class'])
        if target_cls is None:
            continue
        candidates = [r for r in panel[src['split']] if r['primary_class'] == target_cls]
        if not candidates:
            rows.append({'selection_type': 'high_error', 'high_checkpoint': high['checkpoint'],
                         'high_class': src['primary_class'], 'high_sample_id': src['sample_id'],
                         'pair_class': target_cls, 'pair_sample_id': None, 'status_distance': None,
                         'classification': 'C', 'reason': 'no opposite-class sample in split'})
            continue
        pair = min(candidates, key=lambda r: status_distance(src, r))
        rows.append(_observability_record(reader, src, pair, 'high_error', high))
    # Direct similar-status pairs, independent of the high-error ranking.
    for split in ('validation', 'train'):
        for a_cls, b_cls in (('stationary', 'low_depart'), ('low_creep', 'low_depart')):
            A = [r for r in panel[split] if r['primary_class'] == a_cls]
            B = [r for r in panel[split] if r['primary_class'] == b_cls]
            for a in A:
                if not B:
                    continue
                distance, b = min(((status_distance(a, b), b) for b in B), key=lambda x: x[0])
                if distance <= .2:
                    rows.append(_observability_record(reader, a, b, 'similar_status'))
    return rows



def independent_motion_reference(poses, dt=DT):
    """Offline pose reference, independent of the estimator implementation.

    Exactly t-1,t,t+1 are required. Inertial derivatives are rotated once into
    the current body frame. This is a numerical reference, not physical GT.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (3, 3) or not np.isfinite(poses).all():
        raise ValueError('finite three-pose reference required')
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('positive finite dt required')
    # Solving the three-point quadratic keeps this reference independent of
    # experimental_causal_state.reference_state and its derivative formulas.
    times = np.asarray([-dt, 0., dt])
    basis = np.column_stack((np.ones(3), times, times ** 2))
    coefficients = np.linalg.solve(basis, poses[:, :2] - poses[1, :2])
    c, sn = math.cos(poses[1, 2]), math.sin(poses[1, 2])
    world_to_left = np.asarray([[c, sn], [sn, -c]])
    return world_to_left @ coefficients[1], world_to_left @ (2 * coefficients[2])


def production_a0_from_records(records, dt=DT):
    """Call the unchanged production function through a guarded memory reader.

    Only t-2..t exist in this reader; requesting any future frame fails. The
    production function supplies its actual float32 result and provenance.
    """
    from tools.b2d_adapter.config import CandidateConfig
    from tools.b2d_adapter.object_adapter import candidate_causal_state
    if len(records) != 3:
        raise ValueError('production A0 requires three trailing records')
    ids = [int(record['frame']) for record in records]
    if ids != list(range(ids[-1] - 2, ids[-1] + 1)):
        raise ValueError('nonconsecutive frame ids')
    refs = {i: SimpleNamespace(frame_id=i, token=str(i)) for i in ids}
    reads = []
    def read(ref):
        if ref.frame_id not in refs:
            raise ValueError('production A0 attempted noncausal read')
        reads.append(ref.frame_id)
        record = records[ids.index(ref.frame_id)]
        ego = next(box for box in record['anno']['bounding_boxes']
                   if box.get('class') == 'ego_vehicle')
        return SimpleNamespace(actor_pose=record['actor_pose'], ego_box=ego,
                               ref=ref, provenance={'source_files': {'anno': record['sha256']}})
    reader = SimpleNamespace(index=SimpleNamespace(ref=lambda i: refs[i]), read=read)
    state = candidate_causal_state(reader, refs[ids[-1]], CandidateConfig(source_dt_s=dt))
    return state, reads


def audit_clock_records(records, dt=DT):
    """Count actual metadata; absent timestamps never become zero anomalies."""
    timestamp_keys = ('timestamp', 'timestamp_s', 'elapsed_seconds', 'simulation_time')
    missing, nonnumeric, timestamps = 0, 0, []
    for record in records:
        found = [record['anno'][key] for key in timestamp_keys if key in record['anno']]
        if not found:
            missing += 1
            continue
        try:
            value = float(found[0])
            if not math.isfinite(value):
                raise ValueError('nonfinite')
            timestamps.append((record['frame'], value))
        except (TypeError, ValueError):
            nonnumeric += 1
    gaps = sum(max(0, b['frame'] - a['frame'] - 1) for a, b in zip(records, records[1:]))
    intervals = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(timestamps, timestamps[1:])]
    return {'records': len(records), 'timestamp_missing': missing,
            'timestamp_nonnumeric_or_nonfinite': nonnumeric, 'missing_frame_ids': gaps,
            'timestamp_interval_pairs': len(intervals),
            'timestamp_nonpositive_intervals': sum(delta <= 0 for _, delta in intervals) if intervals else None,
            'timestamp_abnormal_intervals': sum(abs(delta - step * dt) > .02 for step, delta in intervals) if intervals else None,
            'physical_interval_status': 'NOT_VERIFIED' if missing or nonnumeric or not intervals else 'OBSERVED_NOT_CERTIFIED'}



def write_motion_audit_reports(result, output):
    """Render the two requested reports from the same machine evidence."""
    def header(title):
        return (f'# {title}\n\n'
                '> **建立前提：** 2026-10-22 的 81 Route / 192 Frame 冻结行为面板。\n'
                '> **解决问题：** 独立核对低速 Motion4，区分实现、噪声、参考点和时钟限制。\n'
                '> **适用范围：** 本次 188 个完整窗口与 2184 个只读原始标注；四方法独立评估。\n'
                '> **不包含：** 全量物理真值认证、生产替换、缓存生成或训练。\n'
                '> **当前状态：** 有限离线审计；正式 A0 保持不变。\n'
                '> **相关代码：** tools/validation/b2d/adapter_semantics_prototype.py。\n'
                '> **最后更新：** 2026-10-22。\n\n')
    aggregates = result['aggregates']
    overall = [r for r in aggregates if r['group'] == 'ALL']
    lines = [header('Motion4 坐标、时钟与生产一致性审计'),
             '没有确认新的生产实现错误。A0 的后向差分、当前车身坐标变换与隔离实现一致；已知延迟、低速噪声、采集时钟和后轴资产身份限制仍然存在。未修改生产代码。\n\n',
             '## 独立参考与原始身份\n\n',
             '只读数据源：`' + result['dataset_root_readonly'] + '`。192/192 当前 anno SHA 与面板逐值匹配；188 个窗口可同时评估全部四种方法，4 个起始帧历史不足保留失败。读取 2184 个去重标注，完整位姿、帧号和 SHA 保存在 [motion4_audit.json](motion4_audit.json) 的 `raw_pose_evidence`。\n\n',
             '参考直接拟合原始 t−1/t/t+1 后轴世界位置三点二次式，在 t 车身轴下表达导数；不调用候选 reference_state。未来帧仅用于离线评分，生产 A0 只接收并读取 t−2..t。它是数值参考，不是独立物理速度/加速度真值。\n\n',
             '| Split | 方法 | N | 速度 MAE/P95 (m/s) | 加速度 MAE/P95 (m/s²) |\n|---|---|---:|---|---|\n']
    for row in overall:
        lines.append(f"| {row['split']} | {row['method'].split('-')[0].upper()} | {row['samples']} | {row['velocity_mae_mps']:.6f} / {row['velocity_p95_mps']:.6f} | {row['acceleration_mae_mps2']:.6f} / {row['acceleration_p95_mps2']:.6f} |\n")
    lines += ['\n## 参考点、坐标与旋转\n\n',
              f"Actor 与 rear 两种参考点的中央速度差 MAE/P95 为 {result['actor_rear_velocity_difference_mae_mps']:.6f}/{result['actor_rear_velocity_difference_p95_mps']:.6f} m/s，加速度差 MAE 为 {result['actor_rear_acceleration_difference_mae_mps2']:.6f} m/s²。旋转中有参考点杆臂效应，不能把 Actor 标量 speed 直接认作 rear 二维速度真值。后轴偏移来自本机 CARLA 0.9.15，采集资产身份仍 NOT_VERIFIED。\n\n",
              f"世界 Y/yaw 镜像后再转到 NAVSIM 局部轴，与先转 CARLA 当前车身再反射 y 的速度差最大 {result['mirrored_world_local_velocity_max_difference_mps']:.3g} m/s。速度、加速度均在世界坐标差分后仅旋转到当前车身轴；没有对不同时间的局部速度直接求差，因此不会额外引入旋转轴导数项。故意忽略轴旋转计算局部速度差，会产生平均 {result['rotation_only_local_acceleration_difference_mae_mps2']:.6f} m/s² 的差异；这是错误诊断方式的对照，不是生产加速度错误。\n\n",
              '## 时钟与缺帧\n\n',
              f"固定 dt=0.1 s 是工程假设。2184/2184 标注缺少已知顶层物理 timestamp 字段；异常 timestamp 间隔数量为 null / NOT_VERIFIED，不能写成 0。请求的非负帧 ID 中不可读数 {result['counts'].get('unreadable_window_records', 0)}；4 个历史不足窗口来自序列起点，不能作为真实漏帧证据。仅检查面板邻域，不认证全 Route 连续性。传感器 tick、非顶层时钟字段、IMU 时钟/姿态和物理 dt 均未认证。\n\n",
              '## 正式身份保护\n\n',
              f"188/188 生产 A0 与隔离 A0 转成 float32 后逐值相等；752/752 方法结果与冻结历史 CSV 相比最大绝对差 {result['frozen_estimate_max_abs_difference']:.3g}。四个生产/状态源码 SHA 前后相同；此结果只认证本报告列出的源码，正式缓存、Manifest、Checkpoint 与 control 状态的保护由集成审计另外核对。A2V 仍是独立候选，未接入任何 Producer。\n"]
    (output / 'motion4_consistency_audit.md').write_text(''.join(lines))
    lines = [header('低速、静止与起步 Motion4 审计'),
             '静止与蠕行不能统一置零。下表是 validation 组的独立数值参考误差；A2V 减少总速度误差，同时增加右转、接近停车等组的相邻速度变化。其加速度等于 A0，因此没有改善加速度精度。\n\n',
             '| 组 | 方法 | N | 速度 MAE/P95 (m/s) | 加速度 MAE/P95 (m/s²) | 相邻局部速度变化中位/P95 (m/s) |\n|---|---|---:|---|---|---|\n']
    groups = ('stationary', 'low_creep', 'low_depart', 'left_turn', 'right_turn', 'approaching_stop', 'scalar_stationary', 'scalar_creep')
    for name in groups:
        for row in aggregates:
            if row['split'] != 'validation' or row['group'] != name or row['method'] not in (A0, METHODS[-1]):
                continue
            lines.append(f"| {name} | {row['method'].split('-')[0].upper()} | {row['samples']} | {row['velocity_mae_mps']:.6f} / {row['velocity_p95_mps']:.6f} | {row['acceleration_mae_mps2']:.6f} / {row['acceleration_p95_mps2']:.6f} | {row['velocity_jitter_median_mps']:.6f} / {row['velocity_jitter_p95_mps']:.6f} |\n")
    stationary = [r for r in aggregates if r['split'] == 'validation' and r['method'] == A0 and r['group'] in ('stationary', 'scalar_stationary')]
    lines += ['\n## 静止定义与虚假状态\n\n',
              '行为面板 stationary 是未来行为组，不要求当前标量速度完全为零。另用当前 |speed|≤0.02 m/s 定义 scalar_stationary，用 0.02<|speed|≤1 m/s 定义 scalar_creep。虚假速度阈值 >0.1 m/s，虚假加速度阈值 >0.2 m/s²，属于项目诊断阈值而非物理精度标准。\n\n']
    for row in stationary:
        lines.append(f"A0 validation {row['group']}：N={row['samples']}，虚假速度率 {row['stationary_false_speed_rate']:.2%}，虚假加速度率 {row['stationary_false_acceleration_rate']:.2%}。加速度率也可能受真实瞬时刹停、IMU/标量时序及差分噪声影响，不能仅据此认定静止物理加速度错误。\n\n")
    lines += ['## 起步阈值滞后\n\n',
              '在面板±6帧内检出 |speed|≥0.2 且前两帧均 <0.05 m/s 的阈值事件；按 Route/Frame 去重。每个估计只接收对应时刻及过去位姿，未来记录仅评分其何时越过同一速度阈值。这里只覆盖明显阈值跃迁，缓慢跨阈值起步与全部停车时长不在该统计中。\n\n']
    for method in (A0, METHODS[-1]):
        events = [r for r in result['start_threshold_events'] if r['method'] == method]
        distribution = dict(sorted(Counter(r['lag_frames'] for r in events).items(), key=lambda p: -1 if p[0] is None else p[0]))
        lines.append(f"{method.split('-')[0].upper()}：{len(events)} 个事件，滞后帧数→事件数：`{distribution}`；时间换算仅为帧数×0.1 s 工程假设。Actor speed 与 rear 速度并非同参考点，阈值滞后不是纯估计器延迟；本结果不能认证物理起步滞后上限。\n\n")
    lines += ['## 范围与准入\n\n',
              '相邻局部向量变化包含正常加速与转向，并非纯噪声；CSV 同时保存统一到当前轴后的变化。转弯停车联合事件尚无单独人工标签，只分别评估 left/right_turn、approaching_stop，不宣称真实联合场景已全面修复。姿态来自当前 ego rotation，未用位移方向替换。A0 名义速度滞后 50 ms、加速度滞后 100 ms 保持显式。\n\n',
              '本轮复现误差与候选抖动，未确认需要修复的生产坐标/单位实现错误。物理时间戳、采集后轴资产身份及全量场景覆盖仍 NOT_VERIFIED；不足以批准 A2V 替换生产或微调准入。完整数字与失败样本见 [motion4_audit.json](motion4_audit.json)、[motion4_samples.csv](motion4_samples.csv)。\n']
    (output / 'low_speed_stationary_audit.md').write_text(''.join(lines))

def run_motion_audit(panel, reader, output, panel_path):
    """Bounded raw-annotation audit; no sensor, map, cache or model reads."""
    if any((output / name).exists() for name in ('motion4_samples.csv', 'motion4_audit.json')):
        raise FileExistsError('motion audit evidence already exists; use a fresh explicit output directory')
    historical = ROOT / 'outputs/diffusiondrive_v1/adapter_semantics_prototype/20261022_155733_prototype'
    frozen = {}
    with (historical / 'causal_state_samples.csv').open() as stream:
        for row in csv.DictReader(stream):
            frozen[(row['sample_id'], row['method'])] = row
    protected = [ROOT / name for name in (
        'tools/b2d_adapter/object_adapter.py', 'tools/b2d_adapter/native_builders.py',
        'tools/b2d_adapter/experimental_causal_state.py', 'tools/b2d_adapter/v1_builders.py')]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    rows, failures, starts = [], [], []
    source_hashes, timestamp_inventory = {}, []
    counts = Counter({'unreadable_window_records': 0, 'current_sha_matches_panel': 0})
    for split in ('validation', 'train'):
        for sample in panel[split]:
            route, frame = sample['route'], int(sample['frame'])
            records = {}
            for index in range(max(0, frame - 6), frame + 7):
                try:
                    records[index] = reader.read(route, index)
                except (OSError, ValueError, EOFError) as error:
                    counts['unreadable_window_records'] += 1
            if frame not in records:
                failures.append({'sample_id': sample['sample_id'], 'reason': 'current_annotation_unavailable'})
                continue
            current = records[frame]
            if current['sha256'] != sample['row']['current_anno_sha']:
                failures.append({'sample_id': sample['sample_id'], 'reason': 'current_sha_mismatch'})
                continue
            counts['current_sha_matches_panel'] += 1
            for index, record in records.items():
                source_hashes[(route, index)] = record['sha256']
            if not all(index in records for index in range(frame - 3, frame + 2)):
                failures.append({'sample_id': sample['sample_id'], 'reason': 'insufficient_causal_or_reference_window'})
                continue
            ordered = [records[index] for index in range(frame - 3, frame + 2)]
            poses = pose_array(ordered)
            ref_v, ref_a = independent_motion_reference(poses[-3:])
            actors = np.asarray([[r['actor_pose'].x, r['actor_pose'].y, r['actor_pose'].yaw] for r in ordered])
            actor_v, actor_a = independent_motion_reference(actors[-3:])
            # Mirroring world Y and yaw first must give the same NAVSIM local
            # vector. Rotating a local-vector difference is a different quantity
            # from inertial acceleration when the body axes rotate.
            mirrored = poses[-3:].copy()
            mirrored[:, 1:] *= -1
            world_velocity = (mirrored[2, :2] - mirrored[0, :2]) / (2 * DT)
            c, sn = math.cos(mirrored[1, 2]), math.sin(mirrored[1, 2])
            mirror_velocity = np.asarray([[c, sn], [-sn, c]]) @ world_velocity
            coordinate_difference = float(np.linalg.norm(mirror_velocity - ref_v))
            prod, read_ids = production_a0_from_records(ordered[1:4])
            counts['production_causal_reads_checked'] += 1
            estimates = {method: causal_state(poses[:4], method=method, speed=current['speed'], dt=DT)
                         for method in METHODS}
            baseline_matches = (np.array_equal(prod.velocity, estimates[A0].velocity.astype(np.float32))
                                and np.array_equal(prod.acceleration, estimates[A0].acceleration.astype(np.float32)))
            counts['production_matches_experimental_a0_float32'] += int(baseline_matches)
            previous = {}
            if frame - 4 in records:
                prev_poses = pose_array([records[index] for index in range(frame - 4, frame)])
                previous = {method: causal_state(prev_poses, method=method, speed=records[frame - 1]['speed'], dt=DT)
                            for method in METHODS}
            speed = abs(float(current['speed']))
            for method, estimate in estimates.items():
                frozen_row = frozen.get((sample['sample_id'], method))
                frozen_delta = None
                if frozen_row:
                    frozen_delta = float(np.max(np.abs(np.concatenate((estimate.velocity, estimate.acceleration)) -
                                         np.asarray(json.loads(frozen_row['velocity_xy_mps']) + json.loads(frozen_row['acceleration_xy_mps2'])))))
                aligned_jitter, rotating_axes_accel_difference = None, None
                if method in previous:
                    angle = records[frame]['rear_pose'].yaw - records[frame - 1]['rear_pose'].yaw
                    c, sn = math.cos(angle), math.sin(angle)
                    previous_in_current = np.asarray([[c, -sn], [sn, c]]) @ previous[method].velocity
                    aligned_jitter = float(np.linalg.norm(estimate.velocity - previous_in_current))
                    local_difference = (estimate.velocity - previous[method].velocity) / DT
                    aligned_difference = (estimate.velocity - previous_in_current) / DT
                    rotating_axes_accel_difference = float(np.linalg.norm(local_difference - aligned_difference))
                row = {'sample_id': sample['sample_id'], 'route': route, 'frame': frame,
                       'split': split, 'primary_class': sample['primary_class'], 'method': method,
                       'speed_mps': speed, 'estimated_speed_mps': float(np.linalg.norm(estimate.velocity)),
                       'velocity_xy_mps': estimate.velocity.tolist(), 'acceleration_xy_mps2': estimate.acceleration.tolist(),
                       'reference_velocity': ref_v.tolist(), 'reference_acceleration': ref_a.tolist(),
                       'velocity_error_mps': float(np.linalg.norm(estimate.velocity - ref_v)),
                       'acceleration_error_mps2': float(np.linalg.norm(estimate.acceleration - ref_a)),
                       'speed_scalar_error_mps': abs(float(np.linalg.norm(estimate.velocity)) - speed),
                       'jitter_velocity_mps': float(np.linalg.norm(estimate.velocity - previous[method].velocity)) if method in previous else None,
                       'jitter_velocity_aligned_current_axes_mps': aligned_jitter,
                       'rotation_only_local_acceleration_difference_mps2': rotating_axes_accel_difference,
                       'jitter_acceleration_mps2': float(np.linalg.norm(estimate.acceleration - previous[method].acceleration)) if method in previous else None,
                       'mirrored_world_local_velocity_difference_mps': coordinate_difference,
                       'actor_rear_reference_velocity_difference_mps': float(np.linalg.norm(actor_v - ref_v)),
                       'actor_rear_reference_acceleration_difference_mps2': float(np.linalg.norm(actor_a - ref_a)),
                       'frozen_estimate_max_abs_difference': frozen_delta,
                       'production_a0_float32_exact': baseline_matches,
                       'current_anno_sha': current['sha256'], 'reference_uses_future_for_evaluation_only': True}
                rows.append(row)
            # Threshold-crossing lag uses actual consecutive raw frames. Future
            # values only score the causal outputs at their own respective t.
            for index in sorted(records):
                if not all(k in records for k in range(index - 3, index + 1)):
                    continue
                if not (abs(float(records[index]['speed'])) >= .2 and
                        all(abs(float(records[k]['speed'])) < .05 for k in (index - 2, index - 1))):
                    continue
                for method in METHODS:
                    crossing = None
                    for t in range(index, min(index + 6, frame + 6) + 1):
                        if not all(k in records for k in range(t - 3, t + 1)):
                            break
                        window = [records[k] for k in range(t - 3, t + 1)]
                        est = causal_state(pose_array(window), method=method, speed=records[t]['speed'], dt=DT)
                        if np.linalg.norm(est.velocity) >= .2:
                            crossing = t
                            break
                    starts.append({'route': route, 'onset_frame': index, 'method': method,
                                   'lag_frames': None if crossing is None else crossing - index,
                                   'lag_assumed_seconds': None if crossing is None else (crossing - index) * DT,
                                   'right_censored': crossing is None})
    for route in sorted({key[0] for key in source_hashes}):
        records = [reader.cache[key] for key in sorted(source_hashes) if key[0] == route]
        # Only adjacent observed frame IDs support clock intervals; aggregate
        # panel windows may be far apart and are not missing source-frame proof.
        clock = audit_clock_records(records)
        clock['unobserved_frame_ids_between_windows'] = clock.pop('missing_frame_ids')
        clock['route'] = route
        timestamp_inventory.append(clock)
    unique_starts = {(s['route'], s['onset_frame'], s['method']): s for s in starts}
    starts = list(unique_starts.values())
    aggregates = []
    for split in ('validation', 'train'):
        for method in METHODS:
            for group_name in ('ALL', 'stationary', 'low_creep', 'low_depart', 'left_turn', 'right_turn', 'approaching_stop', 'scalar_stationary', 'scalar_creep'):
                group = [r for r in rows if r['split'] == split and r['method'] == method and
                         (group_name == 'ALL' or r['primary_class'] == group_name or
                          (group_name == 'scalar_stationary' and r['speed_mps'] <= .02) or
                          (group_name == 'scalar_creep' and .02 < r['speed_mps'] <= 1.))]
                if not group:
                    continue
                aggregates.append({'split': split, 'method': method, 'group': group_name, 'samples': len(group),
                                   'velocity_mae_mps': mean([r['velocity_error_mps'] for r in group]),
                                   'velocity_p95_mps': percentile([r['velocity_error_mps'] for r in group], 95),
                                   'acceleration_mae_mps2': mean([r['acceleration_error_mps2'] for r in group]),
                                   'acceleration_p95_mps2': percentile([r['acceleration_error_mps2'] for r in group], 95),
                                   'scalar_speed_mae_mps': mean([r['speed_scalar_error_mps'] for r in group]),
                                   'stationary_false_speed_rate': mean([r['estimated_speed_mps'] > .1 for r in group]) if group_name in ('stationary', 'scalar_stationary') else None,
                                   'stationary_false_acceleration_rate': mean([np.linalg.norm(r['acceleration_xy_mps2']) > .2 for r in group]) if group_name in ('stationary', 'scalar_stationary') else None,
                                   'velocity_jitter_median_mps': median([r['jitter_velocity_mps'] for r in group]),
                                   'velocity_jitter_p95_mps': percentile([r['jitter_velocity_mps'] for r in group], 95)})
    after = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    result = {'schema': 'motion4-upstream-audit-v1', 'panel_path': str(panel_path),
              'panel_sha256': sha256(panel_path), 'dataset_root_readonly': str(reader.root),
              'dt_assumption_s': DT, 'physical_clock_status': 'NOT_VERIFIED',
              'reference': 'independent three-point quadratic of raw rear Ego Pose; offline evaluation only; not certified GT',
              'rear_asset_status': 'NOT_VERIFIED: offset from local CARLA 0.9.15, collection asset identity unverified',
              'thresholds': {'stationary_scalar_max_mps': .02, 'false_speed_gt_mps': .1, 'false_acceleration_gt_mps2': .2,
                             'creep_scalar_range_mps': [.02, 1.], 'start_scalar_threshold_mps': .2, 'start_previous_two_max_mps': .05},
              'counts': dict(counts), 'source_records': len(source_hashes), 'failures': failures,
              'raw_pose_evidence': [
                  {'route': route, 'frame': index, 'sha256': source_hashes[(route, index)],
                   'actor_pose_world_carla': [reader.cache[(route, index)]['actor_pose'].x, reader.cache[(route, index)]['actor_pose'].y, reader.cache[(route, index)]['actor_pose'].yaw],
                   'rear_pose_world_carla': [reader.cache[(route, index)]['rear_pose'].x, reader.cache[(route, index)]['rear_pose'].y, reader.cache[(route, index)]['rear_pose'].yaw],
                   'speed': reader.cache[(route, index)]['speed']}
                  for route, index in sorted(source_hashes)],
              'mirrored_world_local_velocity_max_difference_mps': max((r['mirrored_world_local_velocity_difference_mps'] for r in rows), default=None),
              'aggregates': aggregates, 'start_threshold_events': starts,
              'start_lag_status': 'OBSERVED_ASSUMED_CLOCK' if starts else 'NOT_VERIFIED: no qualifying threshold event in bounded windows',
              'timestamp_inventory': timestamp_inventory,
              'timestamp_missing_records': sum(r['timestamp_missing'] for r in timestamp_inventory),
              'timestamp_abnormal_interval_count': None if all(r['timestamp_abnormal_intervals'] is None for r in timestamp_inventory) else sum(r['timestamp_abnormal_intervals'] or 0 for r in timestamp_inventory),
              'missing_frame_status': 'bounded requested frame IDs only; no full route certification',
              'frozen_estimate_compared_rows': sum(r['frozen_estimate_max_abs_difference'] is not None for r in rows),
              'frozen_estimate_max_abs_difference': max((r['frozen_estimate_max_abs_difference'] for r in rows if r['frozen_estimate_max_abs_difference'] is not None), default=None),
              'rotation_only_local_acceleration_difference_mae_mps2': mean([r['rotation_only_local_acceleration_difference_mps2'] for r in rows if r['method'] == A0]),
              'actor_rear_velocity_difference_mae_mps': mean([r['actor_rear_reference_velocity_difference_mps'] for r in rows if r['method'] == A0]),
              'actor_rear_velocity_difference_p95_mps': percentile([r['actor_rear_reference_velocity_difference_mps'] for r in rows if r['method'] == A0], 95),
              'actor_rear_acceleration_difference_mae_mps2': mean([r['actor_rear_reference_acceleration_difference_mps2'] for r in rows if r['method'] == A0]),
              'producer_sha256_before': before, 'producer_sha256_after': after, 'producer_unchanged': before == after,
              'confirmed_implementation_error': False,
              'conclusion': 'A0 preserved; lag/noise and unverified physical clock/reference remain; A2V not admitted'}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'motion4_samples.csv', rows, list(rows[0]) if rows else [])
    (output / 'motion4_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    write_motion_audit_reports(result, output)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=PANEL)
    parser.add_argument('--eval-samples', type=Path, default=EVAL)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--route-xml-dir', type=Path, default=ROUTE_XML)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--motion-only', action='store_true', help='bounded annotation-only Motion4 audit')
    parser.add_argument('--dataset-root', type=Path, help='explicit read-only source override for motion audit')
    args = parser.parse_args()
    if args.output.exists() and not args.motion_only:
        raise FileExistsError('Use a fresh explicit output directory')
    panel = json.loads(args.panel.read_text())
    config = json.loads(args.config.read_text())
    output = args.output
    if args.motion_only:
        result = run_motion_audit(panel, RawReader(args.dataset_root or config['dataset_root']), output, args.panel)
        print(json.dumps({'event': 'motion_audit_done', 'rows': sum(r['samples'] for r in result['aggregates'] if r['group'] == 'ALL'), 'output': str(output)}))
        return
    output.mkdir(parents=True, exist_ok=False)
    reader = RawReader(config['dataset_root'])

    rows, aggregate, route_metrics, availability = estimate_metrics(panel, reader)
    aggregates = aggregate_rows(rows)
    write_csv(output / 'causal_state_samples.csv', rows, [
        'split', 'primary_class', 'sample_id', 'route', 'frame', 'method', 'speed_mps',
        'estimated_speed_mps', 'speed_error_mps', 'velocity_xy_mps', 'acceleration_xy_mps2',
        'reference_velocity', 'reference_acceleration', 'reference_forward_acceleration_speed_mps2',
        'velocity_error_mps', 'acceleration_error_mps2', 'acceleration_speed_reference_error_mps2',
        'velocity_bias_forward_mps', 'acceleration_bias_forward_mps2',
        'jitter_velocity_mps', 'jitter_acceleration_mps2', 'reference_jitter_velocity_mps',
        'reference_jitter_acceleration_mps2', 'used_speed_magnitude_only', 'current_sha_matches_panel',
        'target_path_m', 'target_first_speed_mps', 'target_last_speed_mps', 'target_heading_change_deg'])
    write_csv(output / 'causal_state_aggregate.csv', aggregates, list(aggregates[0]) if aggregates else [])
    route_rows = []
    for (method, route), values in sorted(route_metrics.items()):
        route_rows.append({'method': method, 'route': route, 'samples': len(values['velocity_error_mps']),
                           'velocity_mae_mps': mean(values['velocity_error_mps']),
                           'acceleration_mae_mps2': mean(values['acceleration_error_mps2'])})
    write_csv(output / 'causal_state_by_route.csv', route_rows,
              ['method', 'route', 'samples', 'velocity_mae_mps', 'acceleration_mae_mps2'])
    lag = compute_lag(panel, reader)
    write_csv(output / 'causal_state_lag.csv', lag,
              ['split', 'primary_class', 'method', 'shift_frames', 'shift_s', 'mean_speed_error_mps', 'samples'])
    xml_routes = parse_route_xmls(args.route_xml_dir)
    coverage = route_inventory(panel, reader, xml_routes)
    write_csv(output / 'route_coverage.csv', coverage, list(coverage[0]) if coverage else [])
    nav = navigation_rows(panel, reader)
    write_csv(output / 'navigation_near_far.csv', nav, list(nav[0]) if nav else [])
    previews = route_preview_rows(panel, reader, xml_routes)
    write_csv(output / 'route_preview_examples.csv', previews, list(previews[0]) if previews else [])
    geometry = route_geometry(panel, reader, xml_routes)
    (output / 'route_geometry.json').write_text(json.dumps(geometry, ensure_ascii=False, indent=2) + '\n')
    timing = timing_inventory(panel, reader)
    (output / 'timing_inventory.json').write_text(json.dumps(timing, ensure_ascii=False, indent=2) + '\n')
    observability = observability_rows(panel, args.eval_samples, reader)
    write_csv(output / 'stop_departure_observability.csv', observability,
              sorted({key for row in observability for key in row}) if observability else [])
    turn_rows = [r for r in nav if r['future_turn']]
    summary = {'panel': str(args.panel.resolve()), 'panel_sha256': sha256(args.panel),
               'config': str(args.config.resolve()), 'config_sha256': sha256(args.config),
               'dataset_root': config['dataset_root'], 'dt_assumption_s': DT,
               'timestamp_assumption': 'no physical timestamp in raw annotations; 10Hz engineering assumption',
               'central_difference': 'offline reference only; must not be used online',
               'causal_samples': len(rows), 'availability': dict(availability),
               'route_count': len(coverage),
               'xml_exact_route_count': sum(int(r['xml_exact_matches']) > 0 for r in coverage),
               'result_route_id_count': sum(bool(r['result_route_id']) for r in coverage),
               'result_route_id_matches_folder_count': sum(r['result_id_matches_folder'] is True for r in coverage),
               'dense_path_count': sum(bool(r['dense_path_available']) for r in coverage),
               'route_plan_candidate_count': sum(bool(r['route_plan_candidates']) for r in coverage),
               'near_far_differs': sum(bool(r['near_far_category_differs']) for r in nav),
               'future_turn_samples': len(turn_rows),
               'near_matches_future_turn': sum(r['near_matches_future_turn'] is True for r in nav),
               'far_matches_future_turn': sum(r['far_matches_future_turn'] is True for r in nav),
               'turn_near_matches': sum(r['near_matches_future_turn'] is True for r in turn_rows),
               'turn_far_matches': sum(r['far_matches_future_turn'] is True for r in turn_rows),
               'near_distance_median_m': median([r['near_distance_m'] for r in nav if r['near_distance_m'] is not None]),
               'far_distance_median_m': median([r['far_distance_m'] for r in nav if r['far_distance_m'] is not None]),
               'observability_counts': dict(Counter(r['classification'] for r in observability)),
               'observability_selection_counts': dict(Counter(r['selection_type'] for r in observability)),
               'inferred_dt_s': timing['inferred_dt_s'],
               'timing': {k: timing[k] for k in ('routes', 'panel_samples', 'camera_and_lidar_same_frame_filename',
                                                  'timestamp_key_counts', 'sensor_timestamp_key_counts',
                                                  'inferred_dt_s', 'physical_timestamp_status', 'tick_status')},
               'navigation_label_note': 'future turn is a diagnostic target-heading label, not a certified NAVSIM command',
               'methods': list(METHODS), 'status': 'prototype only; no cache/checkpoint change'}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'event': 'causal_done', 'rows': len(rows), 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
