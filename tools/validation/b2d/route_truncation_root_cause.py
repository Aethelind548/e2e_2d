"""Read-only root-cause diagnosis for frozen Route Preview truncations.

The original route-end classifier is the authority for ``route_end_class``.
This tool only subdivides rows already classified as
``INTERNAL_SEGMENT_TRUNCATION``; it does not reclassify valid frames or real
sequence ends.

For every truncation frame it joins three independent evidence sources:
  * the frozen trial frame and reference segment;
  * future readable ego poses and their map matches;
  * the read-only compact-map successor graph and endpoint geometry.

It never reads or writes the live full-generation target, production caches,
the Feature Producer, a model/checkpoint, or training state.
"""
import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.b2d_adapter.experimental_route_navigation import CompactMap, _clean_points, match_poses  # noqa: E402
from tools.b2d_adapter.experimental_route_preview import RoutePolyline  # noqa: E402
from tools.validation.b2d.route_end_classifier import (  # noqa: E402
    END_FRAME_TOLERANCE, load_jsonl, source_frames,
)
from tools.validation.b2d.route_navigation_preview import read_route, town_of  # noqa: E402
from tools.validation.b2d.route_preview_v11 import COMPACT  # noqa: E402


DEFAULT_CLASSIFICATION = (
    ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/'
    '20261022_route_end_audit/route_end_classification.jsonl'
)
DEFAULT_TRIAL = ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261022_status_upstream/reconstruction_trial'
DEFAULT_SOURCE = Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/v1')
DEFAULT_COMPACT = COMPACT
MAX_JOIN_GAP_M = 1.0
MAP_AMBIGUITY_SCORE_M = 0.2
POSE_JUMP_M = 2.0
YAW_JUMP_DEG = 45.0
SUBTYPES = (
    'RECONSTRUCTION_STOP',
    'TOPOLOGY_LINK_MISSING',
    'MAP_MATCH_AMBIGUOUS',
    'SOURCE_DISCONTINUITY',
    'UNRESOLVED',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _center_point(town, center_id):
    center = town.centers.get(int(center_id))
    if center is None:
        return None
    points = town.center_points(int(center_id))[:, :2]
    return points


def successor_branches(town, start_center_id, max_depth=4):
    """Return distinct Normal successors reachable through junction links.

    A branch is a Normal center, not an arbitrary nearby center.  Traversal
    stops at the first Normal center on each path so that downstream roads do
    not inflate the branch count.  Missing links return an empty list.
    """
    if start_center_id is None or int(start_center_id) not in town.centers:
        return []
    start = int(start_center_id)
    queue = deque([(start, [start], 0.0, 0.0)])
    visited = {(start, 0)}
    found = {}
    while queue:
        current, path, total_gap, max_gap = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for target, link_gap in town.links.get(current, ()):
            target = int(target)
            if target in path or target not in town.centers:
                continue
            center = town.centers[target]
            new_path = path + [target]
            new_total = total_gap + float(link_gap)
            new_max = max(max_gap, float(link_gap))
            state = (target, len(new_path) - 1)
            if state in visited:
                continue
            visited.add(state)
            if center.topology_type == 'Normal':
                found.setdefault(target, {
                    'normal_center_id': target,
                    'road': int(center.road),
                    'lane': int(center.lane),
                    'topology_type': center.topology_type,
                    'path_center_ids': new_path,
                    'link_gap_total_m': float(new_total),
                    'max_link_gap_m': float(new_max),
                })
                continue
            queue.append((target, new_path, new_total, new_max))
    return sorted(found.values(), key=lambda item: (item['link_gap_total_m'],
                                                    item['normal_center_id']))


def branch_extension(town, segment, branch):
    """Append a map branch to a frozen segment without changing its points."""
    points = [list(map(float, point)) for point in segment.get('points_xy', [])]
    if len(points) < 2:
        return None
    path = [int(value) for value in branch['path_center_ids']]
    join_gaps = []
    for center_id in path[1:]:
        center_points = _center_point(town, center_id)
        if center_points is None or len(center_points) == 0:
            return None
        if points:
            join_gaps.append(float(np.linalg.norm(center_points[0] - points[-1])))
        points.extend(center_points.tolist())
    points = _clean_points(points)
    if len(points) < 2:
        return None
    return {
        'points_xy': points,
        'join_gaps_m': join_gaps,
        'max_join_gap_m': max(join_gaps) if join_gaps else 0.0,
        'valid_join': not join_gaps or max(join_gaps) <= MAX_JOIN_GAP_M + 1e-9,
    }


def candidate_map_match(town, pose, z):
    candidates = town.candidates(float(pose[0]), float(pose[1]), float(pose[2]), z=z)
    top = candidates[:5]
    conflict = False
    for left, right in zip(top, top[1:]):
        if left.center_id != right.center_id and right.score - left.score <= MAP_AMBIGUITY_SCORE_M:
            conflict = True
            break
    return {
        'candidates': [{
            'center_id': int(item.center_id), 'road': int(item.road),
            'lane': int(item.lane), 'topology_type': item.topology_type,
            'lateral_m': round(float(item.lateral_m), 6),
            'heading_error_deg': round(float(item.heading_error_deg), 6),
            'score': round(float(item.score), 6),
        } for item in top],
        'conflict': bool(conflict),
        'distinct_center_count': len({item.center_id for item in top}),
    }


def _pose_distance(left, right):
    return float(np.linalg.norm(np.asarray(left[:2], dtype=np.float64) -
                                np.asarray(right[:2], dtype=np.float64)))


def _source_discontinuity(route_state, current_index, branch_ids):
    """Check future readable frames and raw source boundaries for a break."""
    frames = route_state['frames']
    poses = route_state['poses']
    source = route_state['source_frames']
    records = route_state['records']
    current_frame = frames[current_index]
    future_source = [frame for frame in source if frame > current_frame]
    future_readable = list(range(current_index + 1, len(frames)))
    frame_gaps = [b - a for a, b in zip(future_source, future_source[1:]) if b - a > 1]
    missing_source = len(future_source) - len(future_readable)
    teleports = []
    pose_steps = []
    yaw_steps = []
    for index in future_readable:
        frame = frames[index]
        record = records.get(frame)
        if record is not None and record.get('anno', {}).get('teleport_flag'):
            teleports.append(frame)
        if index > current_index:
            pose_steps.append(_pose_distance(poses[index - 1], poses[index]))
            delta = abs((float(poses[index][2]) - float(poses[index - 1][2]) + math.pi)
                        % (2 * math.pi) - math.pi)
            yaw_steps.append(math.degrees(delta))
    max_step = max(pose_steps) if pose_steps else 0.0
    max_yaw = max(yaw_steps) if yaw_steps else 0.0
    reasons = []
    if frame_gaps:
        reasons.append('source_frame_gap')
    if missing_source > 0:
        reasons.append('unreadable_future_frame')
    if teleports:
        reasons.append('teleport_flag')
    if max_step > POSE_JUMP_M:
        reasons.append('pose_jump')
    if max_yaw > YAW_JUMP_DEG:
        reasons.append('yaw_jump')
    return {
        'present': bool(reasons),
        'reasons': reasons,
        'future_source_frame_count': len(future_source),
        'future_readable_frame_count': len(future_readable),
        'source_frame_gaps': frame_gaps,
        'unreadable_future_count': max(0, missing_source),
        'teleport_frames': teleports,
        'max_pose_step_m': round(float(max_step), 6),
        'max_yaw_step_deg': round(float(max_yaw), 6),
        'branch_ids_checked': branch_ids,
    }


def _sample_future_matches(route_state, current_index, branch_ids):
    frames = route_state['frames']
    poses = route_state['poses']
    matches = route_state['matches']
    future = list(range(current_index + 1, len(frames)))
    if not future:
        return [], [], []
    sample_indices = [future[0], future[len(future) // 2], future[-1]]
    sample_indices = list(dict.fromkeys(sample_indices))
    samples = []
    centers = []
    observed = set()
    for index in future:
        match = matches[index]
        if match is not None:
            centers.append(int(match.center_id))
            if int(match.center_id) in branch_ids:
                observed.add(int(match.center_id))
    for index in sample_indices:
        match = matches[index]
        samples.append({
            'frame': int(frames[index]),
            'rear_xy': [round(float(poses[index][0]), 6), round(float(poses[index][1]), 6)],
            'yaw_rad': round(float(poses[index][2]), 6),
            'match_center_id': None if match is None else int(match.center_id),
            'match_road': None if match is None else int(match.road),
            'match_lane': None if match is None else int(match.lane),
            'match_lateral_m': None if match is None else round(float(match.lateral_m), 6),
            'match_heading_error_deg': None if match is None else round(float(match.heading_error_deg), 6),
        })
    return samples, sorted(set(centers)), sorted(observed)


def classify_subtype(evidence):
    """Classify a truncation from map/future evidence, not from its label."""
    if not evidence.get('segment_present') or not evidence.get('current_match_present'):
        return 'UNRESOLVED', 'missing_segment_or_current_map_match'
    if evidence.get('source_discontinuity', {}).get('present'):
        return 'SOURCE_DISCONTINUITY', ','.join(evidence['source_discontinuity']['reasons'])
    if evidence.get('map_candidate_conflict') and not evidence.get('future_observed_branch_ids'):
        return 'MAP_MATCH_AMBIGUOUS', 'multiple_close_map_candidates_without_future_resolution'
    branches = evidence.get('branches') or []
    if not branches:
        if evidence.get('successor_links'):
            return 'TOPOLOGY_LINK_MISSING', 'links_exist_but_no_reachable_normal_branch'
        return 'TOPOLOGY_LINK_MISSING', 'no_topology_successor_link'
    observed = [item for item in branches if item['normal_center_id']
                in set(evidence.get('future_observed_branch_ids') or [])]
    if len(observed) == 1 and observed[0].get('valid_join'):
        return 'RECONSTRUCTION_STOP', 'future_ego_selects_one_branch_but_segment_stopped'
    if len(branches) == 1 and branches[0].get('valid_join'):
        return 'RECONSTRUCTION_STOP', 'unique_topology_branch_not_extended'
    if len(branches) > 1:
        return 'MAP_MATCH_AMBIGUOUS', 'multiple_normal_successor_branches_without_future_resolution'
    if any(not item.get('valid_join') for item in branches):
        return 'TOPOLOGY_LINK_MISSING', 'topology_branch_join_gap_exceeded'
    return 'UNRESOLVED', 'insufficient_future_or_topology_evidence'


def load_route_state(route, source_root, compact_dir, trial_root, cache):
    if route in cache:
        return cache[route]
    town_name = town_of(route)
    town = cache.get('town:' + town_name)
    if town is None:
        npz = compact_dir / f'{town_name}_centers.npz'
        metadata = compact_dir / f'{town_name}_center_metadata.json'
        if not npz.is_file() or not metadata.is_file():
            raise FileNotFoundError(f'compact map unavailable for {town_name}')
        town = CompactMap(npz, metadata, source=f'{town_name}_compact')
        cache['town:' + town_name] = town
    frames, poses, z_values, records = read_route(source_root / route)
    matches = match_poses(town, poses, z_values)
    segments = load_jsonl(trial_root / route / 'reference_segments.jsonl')
    source = source_frames(source_root, route)
    reverse_key = 'reverse:' + town_name
    reverse_links = cache.get(reverse_key)
    if reverse_links is None:
        reverse_links = {}
        for source_center, links in town.links.items():
            for target, gap in links:
                center = town.centers.get(int(source_center))
                if center is not None:
                    reverse_links.setdefault(int(target), []).append({
                        'source_center_id': int(source_center),
                        'gap_m': round(float(gap), 6),
                        'road': int(center.road), 'lane': int(center.lane),
                        'topology_type': center.topology_type,
                    })
        cache[reverse_key] = reverse_links
    cumulative = [0.0]
    for left, right in zip(poses, poses[1:]):
        cumulative.append(cumulative[-1] + _pose_distance(left, right))
    state = {
        'town': town, 'frames': frames, 'poses': poses, 'z_values': z_values,
        'records': records, 'matches': matches, 'segments': segments,
        'source_frames': source, 'cumulative': cumulative,
        'frame_index': {int(frame): index for index, frame in enumerate(frames)},
        'reverse_links': reverse_links,
        'branch_extension_cache': {},
        'segment_path_cache': {},
        'town_name': town_name,
    }
    cache[route] = state
    return state


def build_row(row, state, classification_row):
    route = row['route']
    frame = int(row['frame'])
    town = state['town']
    frames = state['frames']
    poses = state['poses']
    matches = state['matches']
    index = state['frame_index'].get(frame)
    segment = next((item for item in state['segments']
                    if int(item['start_frame']) <= frame <= int(item['end_frame'])), None)
    base = {
        'schema': 'route-truncation-root-cause-v1',
        'route': route,
        'frame': frame,
        'route_end_class': classification_row.get('classification'),
        'original_failure_reason': row.get('failure_reason') or row.get('reason'),
        'original_preview_validity': row.get('preview_validity'),
        'source_last_frame': classification_row.get('source_last_frame'),
        'segment_present': segment is not None,
        'current_match_present': index is not None and matches[index] is not None,
    }
    if index is None or segment is None:
        base.update(truncation_subtype='UNRESOLVED',
                    subtype_reason='missing_frame_or_reference_segment',
                    future_ego_exists=False, branches=[], successor_links=[])
        return base
    current_pose = poses[index]
    match = matches[index]
    segment_end_center = int(segment['centers'][-1]) if segment.get('centers') else None
    branches = successor_branches(town, segment_end_center)
    branch_ids = {int(item['normal_center_id']) for item in branches}
    for item in branches:
        cache_key = (int(segment.get('segment_id')), int(item['normal_center_id']))
        extension = state['branch_extension_cache'].get(cache_key)
        if extension is None:
            extension = branch_extension(town, segment, item)
            state['branch_extension_cache'][cache_key] = extension
        item['valid_join'] = bool(extension and extension['valid_join'])
        item['join_gaps_m'] = [] if extension is None else [
            round(float(value), 6) for value in extension['join_gaps_m']]
        item['max_join_gap_m'] = None if extension is None else round(
            float(extension['max_join_gap_m']), 6)
    successor_links = [{
        'target_center_id': int(target), 'gap_m': round(float(gap), 6),
        'road': int(town.centers[int(target)].road),
        'lane': int(town.centers[int(target)].lane),
        'topology_type': town.centers[int(target)].topology_type,
    } for target, gap in town.links.get(segment_end_center, ())] if segment_end_center is not None else []
    future = [value for value in state['frames'] if value > frame]
    future_indices = [state['frame_index'][value] for value in future
                      if value in state['frame_index']]
    future_samples, future_centers, observed_branch_ids = _sample_future_matches(
        state, index, branch_ids)
    future_observed = [item for item in branches
                       if int(item['normal_center_id']) in set(observed_branch_ids)]
    discontinuity = _source_discontinuity(state, index, sorted(branch_ids))
    map_match = candidate_map_match(town, current_pose, state['z_values'][index])
    segment_id = int(segment.get('segment_id'))
    segment_path = state['segment_path_cache'].get(segment_id)
    if segment_path is None and len(segment.get('points_xy', [])) >= 2:
        segment_path = RoutePolyline(segment['points_xy'], source='frozen_trial_segment', dense=True)
        state['segment_path_cache'][segment_id] = segment_path
    route_progress = None
    projection_distance = None
    remaining = None
    if segment_path is not None:
        route_progress, projection_distance, _, _ = segment_path.project(current_pose[:2])
        remaining = float(segment_path.length_m) - float(route_progress)
    elif row.get('route_progress_m') is not None and segment.get('length_m') is not None:
        route_progress = float(row['route_progress_m'])
        remaining = float(segment['length_m']) - route_progress
    predecessor_links = state['reverse_links'].get(segment_end_center, [])
    observed_future_count = len(observed_branch_ids)
    move = (state['cumulative'][future_indices[-1]] - state['cumulative'][index]
            if future_indices else 0.0)
    displacement = (_pose_distance(current_pose, poses[future_indices[-1]])
                    if future_indices else 0.0)
    evidence = {
        'segment_present': True,
        'current_match_present': True,
        'source_discontinuity': discontinuity,
        'map_candidate_conflict': map_match['conflict'],
        'branches': branches,
        'successor_links': successor_links,
        'future_observed_branch_ids': observed_branch_ids,
    }
    subtype, reason = classify_subtype(evidence)
    base.update({
        'truncation_subtype': subtype,
        'subtype_reason': reason,
        'future_ego_exists': bool(future_indices),
        'future_frame_count': len(future_indices),
        'future_first_frame': int(future[0]) if future else None,
        'future_last_frame': int(future[-1]) if future else None,
        'future_movement_m': round(float(move), 6),
        'future_displacement_m': round(float(displacement), 6),
        'future_map_center_ids': future_centers,
        'future_observed_branch_ids': observed_branch_ids,
        'future_observed_branch_count': observed_future_count,
        'future_pose_samples': future_samples,
        'source_discontinuity': discontinuity,
        'map_match': map_match,
        'segment': {
            'segment_id': segment.get('segment_id'),
            'start_frame': int(segment['start_frame']),
            'end_frame': int(segment['end_frame']),
            'length_m': round(float(segment['length_m']), 6),
            'point_count': len(segment.get('points_xy', [])),
            'start_xy': segment['points_xy'][0],
            'end_xy': segment['points_xy'][-1],
            'end_center_id': segment_end_center,
            'max_join_gap_m': segment.get('max_join_gap_m'),
            'break_before': segment.get('break_before'),
            'break_after': segment.get('break_after'),
        },
        'current_pose': [round(float(value), 6) for value in current_pose],
        'current_match': {
            'center_id': int(match.center_id), 'road': int(match.road),
            'lane': int(match.lane), 'topology_type': match.topology_type,
            'lateral_m': round(float(match.lateral_m), 6),
            'heading_error_deg': round(float(match.heading_error_deg), 6),
        },
        'route_progress_m': None if route_progress is None else round(float(route_progress), 6),
        'projection_distance_m': None if projection_distance is None else round(float(projection_distance), 6),
        'remaining_path_m': None if remaining is None else round(float(remaining), 6),
        'distance_to_segment_end_m': round(float(_pose_distance(current_pose, segment['points_xy'][-1])), 6),
        'end_center_id': segment_end_center,
        'predecessor_links': predecessor_links,
        'successor_links': successor_links,
        'branches': branches,
        'branch_count': len(branches),
        'normal_branch_count': len(branches),
        'source_boundary': {
            'source_first_frame': state['source_frames'][0] if state['source_frames'] else None,
            'source_last_frame': state['source_frames'][-1] if state['source_frames'] else None,
            'source_frame_count': len(state['source_frames']),
            'readable_frame_count': len(frames),
            'unreadable_frame_count': max(0, len(state['source_frames']) - len(frames)),
            'near_source_end': bool(classification_row.get('tail_frames') is not None
                                    and classification_row.get('tail_frames') <= END_FRAME_TOLERANCE),
        },
        'repair_selected_branch_id': (
            branches[0]['normal_center_id']
            if subtype == 'RECONSTRUCTION_STOP' and len(branches) == 1
            else (observed_branch_ids[0]
                  if subtype == 'RECONSTRUCTION_STOP' and observed_future_count == 1
                  else None)),
    })
    base['repair_eligible'] = bool(
        subtype == 'RECONSTRUCTION_STOP'
        and base['repair_selected_branch_id'] is not None
        and any(item['normal_center_id'] == base['repair_selected_branch_id']
                and item.get('valid_join') for item in branches))
    return base


def summarize(rows, classification_rows, trial_root, source_root, compact_dir):
    by_subtype = Counter(row['truncation_subtype'] for row in rows)
    by_route = {}
    for row in rows:
        route = row['route']
        by_route.setdefault(route, {
            'internal_truncation_count': 0, 'subtypes': Counter(),
            'branch_counts': Counter(), 'future_movement_m': [],
            'future_observed_branch_count': Counter(),
        })
        item = by_route[route]
        item['internal_truncation_count'] += 1
        item['subtypes'][row['truncation_subtype']] += 1
        item['branch_counts'][row.get('branch_count')] += 1
        item['future_movement_m'].append(row.get('future_movement_m') or 0.0)
        item['future_observed_branch_count'][row.get('future_observed_branch_count')] += 1
    for item in by_route.values():
        values = item['future_movement_m']
        item['future_movement_m'] = {
            'min': round(min(values), 6) if values else None,
            'max': round(max(values), 6) if values else None,
            'mean': round(sum(values) / len(values), 6) if values else None,
        }
        item['subtypes'] = dict(item['subtypes'])
        item['branch_counts'] = dict(item['branch_counts'])
        item['future_observed_branch_count'] = dict(item['future_observed_branch_count'])
    return {
        'schema': 'route-truncation-root-cause-summary-v1',
        'classification': str(classification_rows['path']),
        'classification_sha256': classification_rows['sha256'],
        'trial_root': str(trial_root.resolve()),
        'source_root': str(source_root),
        'compact_dir': str(compact_dir.resolve()),
        'routes': len(by_route),
        'internal_truncation_frames': len(rows),
        'subtype_counts': {name: by_subtype.get(name, 0) for name in SUBTYPES},
        'route_stats': by_route,
        'all_future_ego_exists': all(row.get('future_ego_exists') for row in rows),
        'all_source_boundaries_present': all(
            row.get('source_boundary', {}).get('source_last_frame') is not None for row in rows),
        'all_branch_counts_positive': all(row.get('branch_count', 0) > 0 for row in rows),
        'rows_with_single_branch': sum(row.get('branch_count') == 1 for row in rows),
        'rows_with_multiple_branches': sum(row.get('branch_count', 0) > 1 for row in rows),
        'rows_with_future_branch_observation': sum(
            row.get('future_observed_branch_count', 0) > 0 for row in rows),
        'rows_with_source_discontinuity': sum(
            bool(row.get('source_discontinuity', {}).get('present')) for row in rows),
        'rows_with_map_candidate_conflict': sum(
            bool(row.get('map_match', {}).get('conflict')) for row in rows),
        'repair_eligible_rows': sum(bool(row.get('repair_eligible')) for row in rows),
        'source_boundary': 'Frozen trial and read-only alternate source only; no active generation or production writes.',
        'subtype_rule': {
            'RECONSTRUCTION_STOP': 'unique topology branch (or one future-observed branch) with valid geometry',
            'TOPOLOGY_LINK_MISSING': 'no reachable normal successor or invalid join geometry',
            'MAP_MATCH_AMBIGUOUS': 'multiple normal successor branches or close map candidates, unresolved by future ego',
            'SOURCE_DISCONTINUITY': 'source frame gap, unreadable future frame, teleport, pose or yaw jump',
            'UNRESOLVED': 'insufficient segment, map, or future evidence',
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification', type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument('--trial-root', type=Path, default=DEFAULT_TRIAL)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--compact-dir', type=Path, default=DEFAULT_COMPACT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    classification = load_jsonl(args.classification)
    internal = [row for row in classification
                if row.get('classification') == 'INTERNAL_SEGMENT_TRUNCATION']
    if not internal:
        raise ValueError('classification contains no INTERNAL_SEGMENT_TRUNCATION rows')
    args.output.mkdir(parents=True)
    cache = {}
    rows = []
    ordered = sorted(internal, key=lambda value: value['route'])
    for route, group in itertools.groupby(ordered, key=lambda value: value['route']):
        group = list(group)
        state = load_route_state(route, args.source_root, args.compact_dir,
                                 args.trial_root, cache)
        for row in group:
            rows.append(build_row(row, state, row))
    summary = summarize(rows, {'path': args.classification.resolve(),
                              'sha256': sha256(args.classification)},
                        args.trial_root, args.source_root, args.compact_dir)
    with (args.output / 'route_truncation_root_cause.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    (args.output / 'route_truncation_root_cause_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'internal_truncation_frames': len(rows),
        'subtype_counts': summary['subtype_counts'],
        'routes': summary['routes'],
        'repair_eligible_rows': summary['repair_eligible_rows'],
        'output': str(args.output),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
