"""Isolated road-level route reconstruction and 20 m route-preview prototype.

This module consumes the audit's compact HD-map Center export (read-only) and
never touches Feature/Target caches, the current Feature Producer or model.
Level 1 may use the complete expert pose sequence to reconstruct road-level
maneuvers. Level 2 must use only the current rear pose plus the reconstructed
map path; it must not read future ego poses or controls.

The compact map export is derived from the frozen Bench2Drive HD-map audit:
  points: x, y, z, yaw_deg, is_junction
  offsets: N+1 centerline offsets
  center metadata: road/lane/topology_type/topology/left/right
  links: source/target center ids and endpoint gap
"""
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .experimental_route_preview import RoutePolyline, COMMAND_ORDER, world_to_rear_left


@dataclass(frozen=True)
class MapCenter:
    id: int
    road: int
    lane: int
    topology_type: str
    start_offset: int
    end_offset: int
    topology: tuple
    is_junction: bool


@dataclass(frozen=True)
class MapMatch:
    center_id: int
    road: int
    lane: int
    topology_type: str
    is_junction: bool
    point_index: int
    point_xy: tuple
    point_z: float
    point_yaw_deg: float
    lateral_m: float
    heading_error_deg: float
    z_error_m: Optional[float]
    score: float


class CompactMap:
    """Read-only centerline graph built from the audit compact export."""

    def __init__(self, centers_npz, metadata_json, source):
        data = np.load(centers_npz, allow_pickle=False)
        self.points = np.asarray(data['points'], dtype=np.float64)
        offsets = np.asarray(data['offsets'], dtype=np.int64)
        meta = json.loads(Path(metadata_json).read_text())
        raw = meta['centers']
        self.centers = {}
        for index, item in enumerate(raw):
            cid = int(item['id'])
            self.centers[cid] = MapCenter(
                id=cid, road=int(item['road']), lane=int(item['lane']),
                topology_type=str(item['topology_type']),
                start_offset=int(offsets[index]), end_offset=int(offsets[index + 1]),
                topology=tuple(tuple(x) for x in item.get('topology', [])),
                is_junction=bool(item['topology_type'] == 'Junction'
                                 or (len(self.points) and self.points[offsets[index], 4] > .5)),
            )
        self.links = {}
        for link in meta['links']:
            self.links.setdefault(int(link['source']), []).append((int(link['target']), float(link['gap_m'])))
        self.owner = np.repeat([int(item['id']) for item in raw], np.diff(offsets))
        self.tree = cKDTree(self.points[:, :2])
        self.source = source
        self.sha256 = None
        self.npz_path = str(centers_npz)
        self.metadata_path = str(metadata_json)

    def center_points(self, center_id):
        c = self.centers[center_id]
        return self.points[c.start_offset:c.end_offset]

    def candidates(self, x, y, yaw, z=None, k=32, max_lateral=3.0,
                   max_heading_deg=50.0, max_z=1.0):
        distances, indices = self.tree.query([float(x), float(y)], k=min(k, len(self.points)))
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        by_center = {}
        for lateral, point_index in zip(distances, indices):
            center_id = int(self.owner[int(point_index)])
            point = self.points[int(point_index)]
            heading = abs((math.radians(float(point[3])) - float(yaw) + math.pi) % (2 * math.pi) - math.pi)
            z_error = None if z is None else abs(float(point[2]) - float(z))
            if lateral > max_lateral or heading > math.radians(max_heading_deg):
                continue
            if z_error is not None and z_error > max_z:
                continue
            center = self.centers[center_id]
            # A compact center can occur in multiple sampled points; keep the
            # nearest point for this center.
            score = float(lateral + .01 * math.degrees(heading) + .25 * (z_error or 0.0))
            current = by_center.get(center_id)
            if current is None or score < current[0]:
                by_center[center_id] = (score, MapMatch(
                    center_id=center_id, road=center.road, lane=center.lane,
                    topology_type=center.topology_type, is_junction=center.is_junction,
                    point_index=int(point_index), point_xy=(float(point[0]), float(point[1])),
                    point_z=float(point[2]), point_yaw_deg=float(point[3]),
                    lateral_m=float(lateral), heading_error_deg=math.degrees(heading),
                    z_error_m=z_error, score=float(score)))
        return [item[1] for item in sorted(by_center.values(), key=lambda x: x[0])]

    def best_match(self, x, y, yaw, z=None, **kwargs):
        candidates = self.candidates(x, y, yaw, z, **kwargs)
        return candidates[0] if candidates else None

    def center_path(self, start, end, max_hops=8):
        """Shortest topology path, returning center ids, or None."""
        if start == end:
            return [start]
        queue = [(0, start, [start])]
        seen = {}
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node in seen and seen[node] <= cost:
                continue
            seen[node] = cost
            if node == end:
                return path
            if len(path) - 1 >= max_hops:
                continue
            for target, gap in self.links.get(node, []):
                next_cost = cost + 1.0 + gap
                if target not in seen or next_cost < seen[target]:
                    heapq.heappush(queue, (next_cost, target, path + [target]))
        return None


def wrap_deg(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def yaw_from_poses(poses, indexes, radius=3):
    if not indexes:
        return None
    values = []
    for index in indexes:
        lo, hi = max(0, index - radius), min(len(poses), index + radius + 1)
        for a, b in zip(poses[lo:hi - 1], poses[lo + 1:hi]):
            values.append(math.atan2(b[1] - a[1], b[0] - a[0]))
    if not values:
        return None
    return math.atan2(sum(math.sin(v) for v in values), sum(math.cos(v) for v in values))


def maneuver_from_delta(delta_deg):
    value = wrap_deg(delta_deg)
    absolute = abs(value)
    if absolute < 20.0:
        return 'Straight'
    if absolute >= 135.0:
        return 'UTurn'
    # CARLA yaw decreases for a right turn and increases for a left turn in
    # this data/heading convention. This was calibrated against the
    # reconstructed path preview and the labeled turning routes.
    return 'Right' if value > 0 else 'Left'


def match_poses(town, poses, z_values=None):
    """Match every frame; unmatched frames stay None, never silently filled."""
    matches = []
    for index, pose in enumerate(poses):
        z = None if z_values is None else z_values[index]
        matches.append(town.best_match(pose[0], pose[1], pose[2], z=z))
    return matches


def _normal_runs(matches):
    runs = []
    current = None
    for index, match in enumerate(matches):
        if match is None or match.topology_type != 'Normal' or match.is_junction:
            if current is not None:
                runs.append(current)
                current = None
            continue
        key = (match.road, match.lane)
        if current is None or current['key'] != key:
            if current is not None:
                runs.append(current)
            current = {'key': key, 'indices': [], 'matches': []}
        current['indices'].append(index)
        current['matches'].append(match)
    if current is not None:
        runs.append(current)
    return runs


def _clean_points(points):
    clean = []
    for point in points:
        point = [float(point[0]), float(point[1])]
        if not clean or math.hypot(point[0] - clean[-1][0], point[1] - clean[-1][1]) > 1e-6:
            clean.append(point)
    return clean


def _center_overlaps(town, previous_id, current_id, max_gap_m, tree_cache=None):
    """Detect duplicate/overlapping centerline records for the same lane.

    The compact map contains multiple center records with the same normal
    road/lane whose endpoints overlap.  A pure nearest-center match can then
    alternate between those records frame by frame.  Append a centerline only
    when it is a real continuation; never append the same lane segment again.
    """
    if previous_id == current_id:
        return True
    previous_center = town.centers.get(int(previous_id))
    current_center = town.centers.get(int(current_id))
    if previous_center is None or current_center is None:
        return False
    if (previous_center.road, previous_center.lane) != (current_center.road, current_center.lane):
        return False
    previous = town.center_points(int(previous_id))[:, :2]
    current = town.center_points(int(current_id))[:, :2]
    if len(previous) < 2 or len(current) < 2:
        return False
    # Shared first or last endpoint is the common duplicate-record pattern.
    if (np.linalg.norm(previous[0] - current[0]) <= max_gap_m or
            np.linalg.norm(previous[-1] - current[-1]) <= max_gap_m):
        return True
    tree_cache = tree_cache if tree_cache is not None else {}
    previous_tree = tree_cache.get(int(previous_id))
    if previous_tree is None:
        previous_tree = cKDTree(previous)
        tree_cache[int(previous_id)] = previous_tree
    current_tree = tree_cache.get(int(current_id))
    if current_tree is None:
        current_tree = cKDTree(current)
        tree_cache[int(current_id)] = current_tree
    previous_on_current = [float(current_tree.query(point)[0]) for point in (previous[0], previous[-1])]
    current_on_previous = [float(previous_tree.query(point)[0]) for point in (current[0], current[-1])]
    return (max(previous_on_current) <= max_gap_m or
            max(current_on_previous) <= max_gap_m)


def _stable_centers(town, matches, max_gap_m=1.0):
    """Collapse alternating duplicate centers without merging real transitions."""
    centers = []
    tree_cache = {}
    for match in matches:
        center_id = int(match.center_id)
        if not centers:
            centers.append(center_id)
            continue
        if center_id == centers[-1]:
            continue
        if _center_overlaps(town, centers[-1], center_id, max_gap_m, tree_cache):
            continue
        centers.append(center_id)
    return centers


def _points_for_centers(town, centers):
    points = []
    for center_id in centers:
        center_points = town.center_points(center_id)[:, :2]
        if len(center_points) == 0:
            continue
        points.extend(center_points.tolist())
    return _clean_points(points)


def _path_gaps(points):
    return [float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
            for a, b in zip(points, points[1:])]


def _join_gaps(town, centers):
    gaps = []
    for previous, current in zip(centers, centers[1:]):
        left = town.center_points(previous)[:, :2]
        right = town.center_points(current)[:, :2]
        if len(left) and len(right):
            gaps.append(float(np.linalg.norm(right[0] - left[-1])))
    return gaps


def build_segments(town, runs, route_name='', max_gap_m=1.0):
    """Split a route into continuous, topology-supported reference segments.

    A segment never bridges a missing topology link or a centerline gap larger
    than ``max_gap_m``.  The original full-route fields remain available for
    baseline regression; Level 2 may use a segment even when another part of
    the same Route is broken.
    """
    if not runs:
        return []
    segments = []
    current = None
    previous_gap_reason = None
    for index, run in enumerate(runs):
        run_centers = _stable_centers(town, run['matches'], max_gap_m)
        bridge = None
        bridge_reason = None
        if index > 0:
            previous = runs[index - 1]
            if previous['indices'][-1] + 1 < run['indices'][0]:
                bridge_reason = 'unmatched_frames_between_runs'
            bridge = town.center_path(previous['matches'][-1].center_id,
                                      run['matches'][0].center_id, max_hops=10)
            if bridge is None:
                bridge_reason = 'topology_path_missing'
        if current is None:
            current = {'centers': [], 'start_frame': int(run['indices'][0]),
                       'end_frame': int(run['indices'][-1]), 'break_before': previous_gap_reason,
                       'runs': 1}
        candidate_centers = []
        for center_id in current['centers'] + (bridge[1:] if bridge else []) + run_centers:
            if not candidate_centers or candidate_centers[-1] != center_id:
                candidate_centers.append(center_id)
        candidate_points = _points_for_centers(town, candidate_centers)
        candidate_join_gaps = _join_gaps(town, candidate_centers)
        candidate_valid = (bridge is not None and
                           (not candidate_join_gaps or max(candidate_join_gaps) <= max_gap_m))
        if index == 0:
            candidate_valid = True
        if not candidate_valid:
            # Close the previous run before the broken transition.  The
            # detached run becomes a new segment; its break reason is explicit.
            segments.append(current)
            previous_gap_reason = bridge_reason or 'centerline_gap_exceeded'
            current = {'centers': run_centers,
                       'start_frame': int(run['indices'][0]),
                       'end_frame': int(run['indices'][-1]),
                       'break_before': previous_gap_reason, 'runs': 1}
        else:
            current['centers'] = candidate_centers
            current['end_frame'] = int(run['indices'][-1])
            current['runs'] += 1
            if index == 0:
                current['break_before'] = None
    if current is not None:
        segments.append(current)
    cursor = 0.0
    result = []
    for segment_id, segment in enumerate(segments):
        points = _points_for_centers(town, segment['centers'])
        gaps = _path_gaps(points)
        join_gaps = _join_gaps(town, segment['centers'])
        length = float(sum(gaps))
        result.append({
            'segment_id': segment_id,
            'route': route_name,
            'start_frame': int(segment['start_frame']),
            'end_frame': int(segment['end_frame']),
            'run_count': int(segment['runs']),
            'centers': [int(x) for x in segment['centers']],
            'points_xy': points,
            'length_m': length,
            'gaps_m': gaps,
            'join_gaps_m': join_gaps,
            'max_gap_m': max(gaps) if gaps else 0.0,
            'max_join_gap_m': max(join_gaps) if join_gaps else 0.0,
            'break_before': segment['break_before'],
            'break_after': None,
            'navigation_source': 'RECONSTRUCTED_ROUTE',
            'provenance': 'RECONSTRUCTED_ROUTE',
            'trust': 'HIGH' if points and (not join_gaps or max(join_gaps) <= max_gap_m) else 'CONDITIONAL',
            'route_progress_start_m': cursor,
            'route_progress_end_m': cursor + length,
        })
        cursor += length
    for before, after in zip(result, result[1:]):
        before['break_after'] = after['break_before'] or 'segment_boundary'
    return [row for row in result if len(row['points_xy']) >= 2]


def reconstruct(town, poses, z_values=None, route_name=''):
    """Recover ordered junction decisions and a compact reference path."""
    matches = match_poses(town, poses, z_values)
    matched_count = sum(match is not None for match in matches)
    runs = _normal_runs(matches)
    segments = build_segments(town, runs, route_name=route_name)
    events = []
    failures = []
    transitions = []
    junction_transition_count = 0
    for left, right in zip(runs, runs[1:]):
        if left['key'] == right['key']:
            continue
        entry_match, exit_match = left['matches'][-1], right['matches'][0]
        entry_frame, exit_frame = left['indices'][-1], right['indices'][0]
        path = town.center_path(entry_match.center_id, exit_match.center_id, max_hops=10)
        junction_hits = [] if path is None else [
            cid for cid in path[1:-1]
            if town.centers[cid].is_junction or town.centers[cid].topology_type == 'Junction'
        ]
        transition = {
            'entry_frame': int(entry_frame), 'exit_frame': int(exit_frame),
            'entry_road': int(entry_match.road), 'exit_road': int(exit_match.road),
            'entry_center': int(entry_match.center_id), 'exit_center': int(exit_match.center_id),
            'topology_path_found': path is not None,
            'junction_center_count': len(junction_hits),
        }
        if not junction_hits:
            transition['reason'] = 'no_junction_center_in_transition'
            transitions.append(transition)
            # Normal road-boundary transitions and arbitrary nearest-center
            # switches are not navigation decisions.
            continue
        junction_transition_count += 1
        entry_yaw = yaw_from_poses(poses, left['indices'][-4:])
        exit_yaw = yaw_from_poses(poses, right['indices'][:4])
        if entry_yaw is None or exit_yaw is None:
            transition['reason'] = 'missing_heading_for_maneuver'
            transitions.append(transition)
            failures.append({'route': route_name, 'entry_frame': entry_frame, 'exit_frame': exit_frame,
                             'reason': 'missing_heading_for_maneuver',
                             'entry_road': entry_match.road, 'exit_road': exit_match.road})
            continue
        delta = math.degrees((exit_yaw - entry_yaw + math.pi) % (2 * math.pi) - math.pi)
        maneuver = maneuver_from_delta(delta)
        transition.update(delta_heading_deg=delta, maneuver=maneuver, reason='decision')
        transitions.append(transition)
        margin = min(abs(abs(delta) - 20.0), abs(abs(delta) - 135.0))
        match_quality = max(entry_match.lateral_m, exit_match.lateral_m,
                            entry_match.heading_error_deg / 50.0,
                            exit_match.heading_error_deg / 50.0)
        confidence = 'HIGH' if margin >= 15 and match_quality <= .5 else 'MEDIUM' if match_quality <= .8 else 'LOW'
        event = {
            'event_index': len(events), 'entry_frame': int(entry_frame), 'exit_frame': int(exit_frame),
            'entry_road': int(entry_match.road), 'entry_lane': int(entry_match.lane),
            'exit_road': int(exit_match.road), 'exit_lane': int(exit_match.lane),
            'entry_center': int(entry_match.center_id), 'exit_center': int(exit_match.center_id),
            'junction_path': [int(x) for x in (path or [])],
            'junction_center_count': len(junction_hits),
            'junction_xy': [(poses[entry_frame][0] + poses[exit_frame][0]) / 2,
                            (poses[entry_frame][1] + poses[exit_frame][1]) / 2],
            'delta_heading_deg': delta, 'maneuver': maneuver,
            'confidence': confidence, 'topology_supported': path is not None,
            'source': 'RECONSTRUCTED_ROUTE', 'map_source': town.source,
            'entry_lateral_m': entry_match.lateral_m, 'exit_lateral_m': exit_match.lateral_m,
            'entry_heading_error_deg': entry_match.heading_error_deg,
            'exit_heading_error_deg': exit_match.heading_error_deg,
        }
        if path is None:
            event['source'] = 'AMBIGUOUS'
            failures.append({'route': route_name, 'entry_frame': entry_frame, 'exit_frame': exit_frame,
                             'reason': 'topology_path_missing', 'entry_road': entry_match.road,
                             'exit_road': exit_match.road})
        event['segment_id'] = next((segment['segment_id'] for segment in segments
                                    if segment['start_frame'] <= entry_frame <= segment['end_frame']),
                                   None)
        events.append(event)
    # A compact map reference path: use normal-road runs and insert the
    # topology bridge through each junction. This is an offline reconstructed
    # path, not an online expected-route source.
    sequence = []
    for run_index, run in enumerate(runs):
        for center_id in _stable_centers(town, run['matches']):
            if not sequence or sequence[-1] != center_id:
                sequence.append(center_id)
        if run_index + 1 < len(runs):
            bridge = town.center_path(run['matches'][-1].center_id,
                                      runs[run_index + 1]['matches'][0].center_id,
                                      max_hops=10)
            if bridge:
                for center_id in bridge[1:]:
                    if not sequence or sequence[-1] != center_id:
                        sequence.append(center_id)
    path_points = []
    path_gaps = []
    for center_id in sequence:
        center_points = town.center_points(center_id)[:, :2]
        if len(center_points) == 0:
            continue
        if path_points:
            gap = float(np.linalg.norm(center_points[0] - path_points[-1]))
            if gap > .05:
                path_gaps.append(gap)
                path_points.extend([path_points[-1], center_points[0]])
        path_points.extend(center_points.tolist())
    # RoutePolyline rejects duplicate adjacent points; compact center joins
    # frequently repeat an endpoint by a few centimetres.
    clean_points = []
    for point in path_points:
        if not clean_points or np.linalg.norm(np.asarray(point) - np.asarray(clean_points[-1])) > 1e-6:
            clean_points.append([float(point[0]), float(point[1])])
    # A decision is trusted only if the reconstructed path's own 20 m preview
    # agrees with the maneuver inferred from entry/exit poses.
    if len(clean_points) >= 2:
        preview_path = RoutePolyline.from_waypoints(clean_points, source='reconstructed', dense=False)
        for event in events:
            entry = int(event['entry_frame'])
            pose = poses[entry]
            segment = next((item for item in segments
                            if item['segment_id'] == event.get('segment_id')), None)
            event_path = preview_path
            if segment is not None:
                event['segment_max_gap_m'] = segment['max_gap_m']
            preview = local_preview(event_path, pose[:2], pose[2])
            event['entry_path_preview'] = preview.get('category')
            event['entry_path_preview_status'] = preview.get('status')
            event['path_consistent'] = preview.get('category') == event['maneuver']
            if not event['path_consistent']:
                event['source'] = 'AMBIGUOUS'
    match_rate = matched_count / len(matches) if matches else 0.0
    max_path_gap = max(path_gaps, default=0.0)
    no_decision_reason = None
    if not events:
        if match_rate < .8:
            no_decision_reason = 'map_match_incomplete'
        elif max_path_gap > 1.0:
            no_decision_reason = 'route_path_gap'
        elif len(runs) <= 1:
            no_decision_reason = 'single_normal_run_no_junction_transition'
        elif junction_transition_count:
            no_decision_reason = 'junction_transition_without_valid_decision'
        elif transitions:
            no_decision_reason = 'transitions_without_junction_topology'
        else:
            no_decision_reason = 'no_junction_observed'
    diagnostics = {
        'normal_run_count': len(runs),
        'transition_count': len(transitions),
        'junction_transition_count': int(junction_transition_count),
        'unmatched_frame_count': int(len(matches) - matched_count),
        'max_reference_path_gap_m': float(max_path_gap),
        'segment_count': len(segments),
        'no_decision_reason': no_decision_reason,
        'transitions': transitions,
        'heading_span_deg': wrap_deg(poses[-1][2] - poses[0][2]) if len(poses) >= 2 else 0.0,
    }
    return {
        'route': route_name, 'map_source': town.source,
        'matched_frames': matched_count, 'total_frames': len(matches),
        'match_rate': match_rate,
        'events': events, 'failures': failures,
        'reference_path_xy': clean_points,
        'reference_path_gaps_m': path_gaps,
        'reference_path_centers': sequence,
        'segments': segments, 'diagnostics': diagnostics,
        'matches': matches,
        'reference_path_source': 'RECONSTRUCTED_ROUTE',
        'reference_path_online_available': False,
    }


def local_preview(path, current_xy, current_yaw, preview_m=20.0, lateral_threshold_m=2.0):
    """20 m route preview with a separate UTurn decision and V1 compatibility flag."""
    route = path if isinstance(path, RoutePolyline) else RoutePolyline.from_waypoints(
        path, source='reconstructed_map_centerline', dense=False)
    s0, projection, projected, segment = route.project(current_xy)
    target = route.point_at_distance(s0 + preview_m)
    if target is None:
        return {
            'status': 'FAIL', 'reason': 'route_end_before_preview', 's0_m': s0,
            'projection_distance_m': projection, 'preview_m': preview_m,
            'category': 'Unknown', 'uturn': False, 'v1_compatible': False,
        }
    current_point = route.point_at_distance(s0)
    tangent_current = route.point_at_distance(min(route.length_m, s0 + .5))
    target_after = route.point_at_distance(min(route.length_m, s0 + preview_m + .5))
    target_before = route.point_at_distance(max(0.0, s0 + preview_m - .5))
    if current_point is None or tangent_current is None:
        return {'status': 'FAIL', 'reason': 'insufficient_tangent', 's0_m': s0,
                'projection_distance_m': projection, 'preview_m': preview_m,
                'category': 'Unknown', 'uturn': False, 'v1_compatible': False}
    heading_current = math.atan2(tangent_current[1] - current_point[1], tangent_current[0] - current_point[0])
    # Use the target's outgoing tangent, not the chord from the current
    # tangent sample to the target; the chord hides U-turn reversals.
    tangent_target = target_after if target_after is not None else target_before
    if tangent_target is None:
        return {'status': 'FAIL', 'reason': 'insufficient_tangent', 's0_m': s0,
                'projection_distance_m': projection, 'preview_m': preview_m,
                'category': 'Unknown', 'uturn': False, 'v1_compatible': False}
    heading_target = math.atan2(tangent_target[1] - target[1], tangent_target[0] - target[0])
    delta = math.degrees((heading_target - heading_current + math.pi) % (2 * math.pi) - math.pi)
    local = world_to_rear_left(target, current_xy, current_yaw)
    lateral = float(local[1])
    if abs(delta) >= 135.0:
        category, uturn = 'UTurn', True
    elif lateral > lateral_threshold_m:
        category, uturn = 'Left', False
    elif lateral < -lateral_threshold_m:
        category, uturn = 'Right', False
    elif local[0] > 0:
        category, uturn = 'Straight', False
    else:
        category, uturn = 'Unknown', False
    # Current V1 has [Left, Straight, Right, Unknown]; UTurn is not a slot.
    v1 = {'Left': [1, 0, 0, 0], 'Straight': [0, 1, 0, 0],
          'Right': [0, 0, 1, 0], 'UTurn': [0, 0, 0, 1], 'Unknown': [0, 0, 0, 1]}[category]
    experimental = {'Left': [1, 0, 0, 0], 'Straight': [0, 1, 0, 0],
                    'Right': [0, 0, 1, 0], 'UTurn': [0, 0, 0, 1], 'Unknown': [0, 0, 0, 0]}[category]
    return {
        'status': 'PASS' if category != 'Unknown' else 'FAIL',
        'reason': 'uturn_heading' if uturn else 'normal_preview',
        's0_m': s0, 'projection_distance_m': projection, 'projected_xy': projected.tolist(),
        'target_xy': target.tolist(), 'local_xy_m': local.tolist(),
        'lateral_m': lateral, 'heading_delta_deg': delta, 'preview_m': preview_m,
        'category': category, 'uturn': uturn, 'v1_compatible': not uturn,
        'command4_v1_order': v1, 'command4_uturn_order': experimental,
    }
