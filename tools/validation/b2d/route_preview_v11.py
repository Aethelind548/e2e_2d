"""Route Preview V1.1 panel repair and contract evidence.

This runner is deliberately separate from the V1.0 reconstruction runner.  It
reads the frozen V1.0 manifest/reference paths and the audit's frozen
per-frame topology candidates; it does not read raw route directories.  That is
necessary because the source mount is currently returning I/O errors.  The
resulting panel evidence is therefore a repair of the existing panel, not a
full-dataset generation claim.

The fallback hierarchy is explicit:
  * full reconstructed route path;
  * a unique local topology candidate segment;
  * an ensemble of candidates only when all candidates agree at the 20 m
    horizon (the paths may diverge beyond 20 m);
  * otherwise the frame remains failed/not verified.
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.b2d_adapter.coordinate_transform import ActorPose, actor_to_reference  # noqa: E402
from tools.b2d_adapter.experimental_route_navigation import CompactMap  # noqa: E402
from tools.b2d_adapter.route_preview_contract import (  # noqa: E402
    COMMAND_ORDER, UNSUPPORTED_UTURN, preview_command4, status8,
)

PANEL = ROOT / 'outputs/diffusiondrive_v1/scratch_attribution/20261022_142557_panel/behavior_panel.json'
BASELINE = ROOT / 'outputs/diffusiondrive_v1/route_preview_reconstruction/20261022_183306_route_preview'
EXAMPLES = ROOT / 'outputs/diffusiondrive_v1/route_preview_data_audit/20261022_readonly/route_preview_examples.json'
ASSOCIATION = ROOT / 'outputs/diffusiondrive_v1/route_preview_data_audit/20261022_readonly/route_source_association.json'
BASELINE_TAG = '20261022_183306_route_preview'
OFFLINE_MAX_PROJECTION_M = 5.0
COMPACT = ROOT / 'outputs/diffusiondrive_v1/route_preview_data_audit/20261022_readonly'
TOWN_RE = re.compile(r'_(Town\d+[A-Za-z]*)_')


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dedupe_points(points):
    out = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        if not out or math.hypot(x - out[-1][0], y - out[-1][1]) > 1e-6:
            out.append((x, y))
    return out


def row_category(row):
    return row.get('preview_status') or ('PASS' if row.get('status') == 'PASS' else 'FAIL')


def baseline_reason(row):
    return row.get('preview_reason') or row.get('reason') or 'baseline_preview_invalid'


def make_preview_row(base, example, panel_row, preview, navigation_source, trust,
                     route_scope, candidate_info=None, provenance=None):
    command = preview.get('command4')
    motion = panel_row.get('status', [None] * 8)[4:8]
    valid = bool(preview.get('preview_valid'))
    status_vector = status8(command, motion) if valid and command is not None else None
    status_command = panel_row.get('status', [None] * 8)[:4]
    command_matches_status = (command is not None and status_command is not None
                              and [float(v) for v in command] == [float(v) for v in status_command])
    candidate_info = candidate_info or {}
    return {
        'sample_id': base.get('sample_id'),
        'split': base.get('split'),
        'route': base['route'],
        'town': base.get('town'),
        'frame': int(base['frame']),
        'primary_class': base.get('primary_class'),
        'baseline_preview_status': row_category(base),
        'baseline_preview_reason': baseline_reason(base),
        'preview_status': preview.get('status'),
        'preview_valid': valid,
        'preview_reason': preview.get('reason'),
        'preview_category': preview.get('category'),
        'command4_json': json.dumps(command) if command is not None else None,
        'status8_json': json.dumps(status_vector) if status_vector is not None else None,
        'status8_valid': status_vector is not None,
        'navigation_source': navigation_source,
        'provenance': (provenance or ('AMBIGUOUS' if preview.get('status') == 'AMBIGUOUS'
                                       else 'RECONSTRUCTED_ROUTE' if valid else 'NOT_VERIFIED')),
        'trust': trust,
        'route_scope': route_scope,
        'route_progress_m': preview.get('route_progress_m'),
        'progress_fraction': preview.get('progress_fraction'),
        'route_length_m': preview.get('route_length_m'),
        'projection_distance_m': preview.get('projection_distance_m'),
        'preview_point_xy_json': json.dumps(preview.get('preview_point_xy')),
        'local_xy_m_json': json.dumps(preview.get('local_xy_m')),
        'lateral_m': preview.get('lateral_m'),
        'longitudinal_m': preview.get('longitudinal_m'),
        'heading_delta_deg': preview.get('heading_delta_deg'),
        'segment_index': preview.get('segment_index'),
        'uturn_status': preview.get('uturn_status'),
        'v1_compatible': preview.get('v1_compatible'),
        'near_category': base.get('near_category'),
        'far_category': base.get('far_category'),
        'status_command4_json': json.dumps(status_command) if status_command is not None else None,
        'command_matches_status4': command_matches_status,
        'candidate_status': candidate_info.get('candidate_status'),
        'candidate_count': candidate_info.get('candidate_count'),
        'candidate_consensus': candidate_info.get('candidate_consensus'),
        'selected_candidate_index': candidate_info.get('selected_candidate_index'),
        'candidate_target_spread_m': candidate_info.get('candidate_target_spread_m'),
        'recovery_reason': candidate_info.get('recovery_reason'),
        'source_pose': 'audit_preview_origin_or_derived_actor_rear',
        'source_pose_certified': False,
    }


def example_rear(record):
    if 'preview_origin' in record:
        rear = record['preview_origin']['rear_xyz']
        return (float(rear[0]), float(rear[1])), math.radians(float(record['actor_yaw_deg']))
    actor = record['actor_xyz']
    yaw = math.radians(float(record['actor_yaw_deg']))
    rear = actor_to_reference(ActorPose(float(actor[0]), float(actor[1]), yaw), (-1.388632877, 0.0))
    return (float(rear.x), float(rear.y)), yaw


def candidate_results(record):
    rear, yaw = example_rear(record)
    results = []
    for index, candidate in enumerate(record.get('candidates') or []):
        points = dedupe_points([(p[0], p[1]) for p in candidate.get('points_xyz', [])])
        if len(points) < 2:
            results.append((index, None, 'candidate_too_short'))
            continue
        preview = preview_command4(points, (rear[0], rear[1]), yaw,
                                   max_projection_m=OFFLINE_MAX_PROJECTION_M)
        results.append((index, preview, preview.get('reason')))
    return results


def recover_candidate(record):
    """Return (preview, info) only when the 20 m horizon is unambiguous."""
    candidates = record.get('candidates') or []
    results = candidate_results(record)
    info = {
        'candidate_status': record.get('candidate_status'),
        'candidate_count': len(candidates),
        'candidate_consensus': False,
        'selected_candidate_index': None,
        'candidate_target_spread_m': None,
    }
    valid = [(index, preview) for index, preview, _ in results
             if preview is not None and preview.get('preview_valid')]
    if not valid:
        info['recovery_reason'] = 'no_valid_candidate_at_20m'
        return None, info
    commands = {tuple(preview.get('command4')) for _, preview in valid}
    targets = [preview.get('preview_point_xy') for _, preview in valid]
    spread = None
    if targets:
        arr = np.asarray(targets, dtype=np.float64)
        spread = float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))
    info['candidate_target_spread_m'] = spread
    if len(valid) == 1:
        info['candidate_consensus'] = True
        info['selected_candidate_index'] = valid[0][0]
        info['recovery_reason'] = 'unique_candidate_at_20m'
        return valid[0][1], info
    if len(commands) == 1 and spread is not None and spread <= .5:
        # The candidates may diverge after 20 m; the 20 m target and command
        # are invariant, so this is not an arbitrary candidate selection.
        info['candidate_consensus'] = True
        info['selected_candidate_index'] = valid[0][0]
        info['recovery_reason'] = 'candidate_consensus_at_20m'
        return valid[0][1], info
    info['recovery_reason'] = 'candidate_disagreement_at_20m'
    return None, info


def town_of(route):
    match = TOWN_RE.search(route)
    return match.group(1) if match else None


def audit_path(path, town, map_cache):
    """Cheap map/path audit for routes with no recovered decision.

    This is evidence for a possible missed event, not a new decision.  It uses
    the frozen reconstructed path only; it cannot certify the original plan.
    """
    if not path or len(path) < 2 or town is None:
        return {'status': 'NOT_VERIFIED', 'reason': 'path_unavailable'}
    if town not in map_cache:
        npz = COMPACT / f'{town}_centers.npz'
        meta = COMPACT / f'{town}_center_metadata.json'
        map_cache[town] = (CompactMap(npz, meta, f'{town}_compact')
                           if npz.is_file() and meta.is_file() else None)
    map_obj = map_cache[town]
    if map_obj is None:
        return {'status': 'NOT_VERIFIED', 'reason': 'compact_map_unavailable'}
    points = np.asarray(path, dtype=np.float64)
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(step)))
    samples = []
    for distance in np.arange(0.0, cumulative[-1] + 1e-9, 1.0):
        index = int(np.searchsorted(cumulative, distance, side='right') - 1)
        index = min(index, len(points) - 2)
        segment = max(step[index], 1e-9)
        t = min(1.0, max(0.0, (distance - cumulative[index]) / segment))
        xy = points[index] + t * (points[index + 1] - points[index])
        j = min(index + 1, len(points) - 1)
        yaw = math.atan2(points[j][1] - points[index][1], points[j][0] - points[index][0])
        match = map_obj.best_match(float(xy[0]), float(xy[1]), yaw)
        samples.append(match)
    matched = [match for match in samples if match is not None]
    junction_hits = sum(bool(match.is_junction or match.topology_type == 'Junction')
                        for match in matched)
    changes = sum(1 for a, b in zip(matched, matched[1:]) if a.center_id != b.center_id)
    turn_values = []
    stride = max(1, int(round(5.0 / max(float(np.median(step)), .1))))
    for i in range(0, max(0, len(points) - stride), stride):
        a = points[i + stride] - points[i]
        if i + 2 * stride < len(points):
            b = points[i + 2 * stride] - points[i + stride]
            turn_values.append(math.degrees((math.atan2(b[1], b[0]) - math.atan2(a[1], a[0]) + math.pi)
                                            % (2 * math.pi) - math.pi))
    peak = max(turn_values, key=abs) if turn_values else 0.0
    if not matched or len(matched) / len(samples) < .8:
        diagnosis = 'map_match_incomplete'
    elif junction_hits:
        diagnosis = ('possible_straight_junction_missed' if abs(peak) < 20 else
                     'possible_uturn_unsupported' if abs(peak) >= 135 else
                     'possible_left_turn_missed' if peak < 0 else
                     'possible_right_turn_missed')
    elif abs(peak) < 20:
        diagnosis = 'no_junction_observed'
    else:
        diagnosis = 'possible_turn_without_junction'
    return {
        'status': 'CANDIDATE_NOT_VERIFIED',
        'reason': diagnosis,
        'sampled_points': len(samples),
        'matched_points': len(matched),
        'match_rate': len(matched) / len(samples) if samples else 0.0,
        'junction_center_hits': int(junction_hits),
        'center_transition_count': int(changes),
        'peak_turn_deg': float(peak),
        'path_length_m': float(cumulative[-1]),
    }


def failure_preview(reason):
    return {
        'status': 'FAIL', 'preview_valid': False, 'reason': reason,
        'category': 'Unknown', 'command4': [0, 0, 0, 1],
        'v1_compatible': False, 'uturn_status': None,
        'preview_point_xy': None, 'local_xy_m': None, 'lateral_m': None,
        'longitudinal_m': None, 'heading_delta_deg': None,
        'route_progress_m': None, 'progress_fraction': None,
        'route_length_m': None, 'projection_distance_m': None,
        'segment_index': None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--examples', type=Path, default=EXAMPLES)
    parser.add_argument('--panel', type=Path, default=PANEL)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    args.output.mkdir(parents=True)
    baseline = jsonl(args.baseline / 'route_navigation_manifest.jsonl')
    baseline_by_route = {row['route']: row for row in baseline}
    path_by_route = {row['route']: dedupe_points(row['points'])
                     for row in jsonl(args.baseline / 'route_reference_paths.jsonl')}
    frame_rows = jsonl(args.baseline / 'frame_route_preview.jsonl')
    panel = json.loads(args.panel.read_text())
    panel_by_key = {}
    for split in ('validation', 'train'):
        for row in panel[split]:
            panel_by_key[(row['route'], int(row['frame']))] = dict(row, split=split)
    example_by_key = {(row['route'], int(row['frame'])): row
                      for row in json.loads(args.examples.read_text())['records']}

    output_frames = []
    segment_rows = []
    recovery_counts = Counter()
    for base in frame_rows:
        key = (base['route'], int(base['frame']))
        panel_row = panel_by_key.get(key)
        example = example_by_key.get(key)
        if panel_row is None or example is None:
            preview = failure_preview('missing_panel_or_pose_evidence')
            output_frames.append(make_preview_row(base, example, panel_row or {},
                                                  preview, 'NOT_VERIFIED', 'NOT_VERIFIED',
                                                  'NONE'))
            recovery_counts['missing_pose_evidence'] += 1
            continue
        rear, yaw = example_rear(example)
        baseline_status = row_category(base)
        preview = None
        navigation_source = 'RECONSTRUCTED_ROUTE'
        trust = 'HIGH'
        route_scope = 'FULL_ROUTE'
        candidate_info = None
        failure_provenance = None
        if baseline_status == 'PASS':
            path = path_by_route.get(base['route'])
            if path and len(path) >= 2:
                preview = preview_command4(path, (rear[0], rear[1]), yaw,
                                           max_projection_m=OFFLINE_MAX_PROJECTION_M)
        if preview is None or not preview.get('preview_valid'):
            preview, candidate_info = recover_candidate(example)
            if preview is not None and preview.get('preview_valid'):
                navigation_source = ('LOCAL_REFERENCE_SEGMENT'
                                     if candidate_info.get('selected_candidate_index') is not None
                                     else 'TOPOLOGY_CANDIDATE')
                trust = 'CONDITIONAL'
                route_scope = 'LOCAL_20M'
                recovery_counts[('recovered', candidate_info.get('recovery_reason'))] += 1
                candidate = example['candidates'][candidate_info['selected_candidate_index']]
                segment_rows.append({
                    'route': base['route'], 'town': town_of(base['route']),
                    'segment_id': None,
                    'start_frame': int(base['frame']), 'end_frame': int(base['frame']),
                    'segment_type': 'LOCAL_REFERENCE_SEGMENT',
                    'navigation_source': navigation_source, 'provenance': 'RECONSTRUCTED_ROUTE',
                    'trust': trust, 'candidate_status': candidate_info.get('candidate_status'),
                    'candidate_count': candidate_info.get('candidate_count'),
                    'candidate_consensus': candidate_info.get('candidate_consensus'),
                    'selected_candidate_index': candidate_info.get('selected_candidate_index'),
                    'candidate_center_ids': candidate.get('center_ids'),
                    'points_xy': [[float(point[0]), float(point[1])]
                                  for point in candidate.get('points_xyz', [])],
                    'map_source': (baseline_by_route.get(base['route']) or {}).get('map_source'),
                    'map_compact_sha256': (baseline_by_route.get(base['route']) or {}).get('map_compact_sha256'),
                    'source_detail': 'causal_topology_candidate_at_current_anchor',
                    'max_join_gap_m': None,
                    'break_before': baseline_reason(base),
                    'break_after': None,
                })
            else:
                reason = (preview.get('reason') if preview is not None
                          else (candidate_info or {}).get('recovery_reason')
                          or base.get('preview_reason') or base.get('reason')
                          or 'no_candidate_available')
                preview = failure_preview(reason)
                failure_provenance = ('AMBIGUOUS' if (candidate_info or {}).get('recovery_reason')
                                      == 'candidate_disagreement_at_20m' else 'NOT_VERIFIED')
                navigation_source = 'NOT_VERIFIED'
                trust = 'NOT_VERIFIED'
                route_scope = 'NONE'
                recovery_counts[('failed', reason)] += 1
        else:
            recovery_counts['baseline_full_route'] += 1
        output_frames.append(make_preview_row(base, example, panel_row, preview,
                                              navigation_source, trust, route_scope,
                                              candidate_info, failure_provenance))

    # Full-route segment rows for every valid baseline path.
    for route, path in path_by_route.items():
        meta = baseline_by_route.get(route, {})
        segment_rows.append({
            'route': route, 'town': meta.get('town'), 'segment_id': 0,
            'start_frame': 0, 'end_frame': int(meta.get('frames', 0)) - 1,
            'segment_type': 'FULL_RECONSTRUCTED_ROUTE',
            'navigation_source': 'RECONSTRUCTED_ROUTE',
            'provenance': 'RECONSTRUCTED_ROUTE', 'trust': 'HIGH',
            'candidate_status': None, 'candidate_count': None,
            'candidate_consensus': None, 'selected_candidate_index': None,
            'candidate_center_ids': None, 'points_xy': path,
            'map_source': meta.get('map_source'),
            'map_compact_sha256': meta.get('map_compact_sha256'),
            'source_detail': 'frozen_v10_full_reconstructed_path',
            'max_join_gap_m': meta.get('reference_path_max_gap_m'),
            'break_before': None, 'break_after': None,
        })

    # Stable segment IDs after all rows are known.
    for index, row in enumerate(segment_rows):
        row['segment_id'] = index

    # Route manifest: keep frozen V1 fields, add V1.1 coverage/diagnosis.
    frames_by_route = defaultdict(list)
    for row in output_frames:
        frames_by_route[row['route']].append(row)
    route_manifest = []
    map_cache = {}
    for route, meta in baseline_by_route.items():
        rows = frames_by_route.get(route, [])
        valid = sum(bool(row['preview_valid']) for row in rows)
        recovered = sum(row['navigation_source'] in ('LOCAL_REFERENCE_SEGMENT',
                                                     'TOPOLOGY_CANDIDATE')
                        for row in rows)
        frame_provenance = Counter(row['provenance'] for row in rows)
        route_provenance = ('AMBIGUOUS' if frame_provenance.get('AMBIGUOUS')
                            else 'RECONSTRUCTED_ROUTE' if frame_provenance.get('RECONSTRUCTED_ROUTE')
                            else 'NOT_VERIFIED')
        decisions = meta.get('decision_count', len(meta.get('decisions', [])))
        path_audit = None
        if decisions:
            no_reason = None
            diagnosis = 'OK'
        elif meta.get('reference_path_max_gap_m', 0) > 1:
            no_reason = 'route_path_gap'
            diagnosis = 'FRAGMENTED_ROUTE'
        elif meta.get('match_rate', 0) < .8:
            no_reason = 'map_match_incomplete'
            diagnosis = 'MAP_MATCH'
        else:
            path_audit = audit_path(path_by_route.get(route), meta.get('town'), map_cache)
            no_reason = path_audit.get('reason')
            diagnosis = path_audit.get('status')
        row = dict(meta)
        row.update({
            'schema': 'route-navigation-v11',
            'v11_segment_count': sum(1 for segment in segment_rows
                                     if segment['route'] == route),
            'v11_preview_valid': valid,
            'v11_preview_total': len(rows),
            'v11_recovered_preview_count': recovered,
            'v11_missing_preview_count': len(rows) - valid,
            'v11_route_provenance': route_provenance,
            'v11_frame_provenance_counts': dict(frame_provenance),
            'v11_no_decision_reason': no_reason,
            'v11_no_decision_diagnosis': diagnosis,
            'v11_no_decision_basis': ('frozen_baseline_manifest_and_cached_candidates'
                                      if not decisions else 'baseline_decision'),
            'v11_path_audit': path_audit,
            'v11_source_boundary': 'NO_RAW_SOURCE_READ; SOURCE_MOUNT_IO_ERROR',
        })
        route_manifest.append(row)

    parquet_rows = []
    for row in output_frames:
        parquet_rows.append({key: value for key, value in row.items()})
    frame_df = pd.DataFrame(parquet_rows)
    frame_df.to_parquet(args.output / 'frame_route_preview.parquet', index=False)
    (args.output / 'frame_route_preview.jsonl').write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in output_frames) + '\n')
    (args.output / 'route_segment_manifest.jsonl').write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in segment_rows) + '\n')
    (args.output / 'route_navigation_manifest.jsonl').write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in route_manifest) + '\n')
    with (args.output / 'route_preview_v11_failures.csv').open('w', newline='') as stream:
        fields = ['route', 'frame', 'baseline_status', 'navigation_source', 'reason', 'category']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in output_frames:
            if not row['preview_valid']:
                writer.writerow({'route': row['route'], 'frame': row['frame'],
                                 'baseline_status': row['baseline_preview_status'],
                                 'navigation_source': row['navigation_source'],
                                 'reason': row['preview_reason'], 'category': row['preview_category']})

    valid_count = sum(bool(row['preview_valid']) for row in output_frames)
    provenance_labels = ('ORIGINAL_ROUTE', 'RECONSTRUCTED_ROUTE', 'AMBIGUOUS', 'NOT_VERIFIED')
    frame_provenance_counts = {label: 0 for label in provenance_labels}
    for row in output_frames:
        frame_provenance_counts[row['provenance']] += 1
    route_provenance_counts = {label: 0 for label in provenance_labels}
    for row in route_manifest:
        route_provenance_counts[row['v11_route_provenance']] += 1
    no_decision_counts = Counter(
        (row.get('v11_no_decision_reason'), row.get('v11_no_decision_diagnosis'))
        for row in route_manifest if not row.get('decision_count'))
    summary = {
        'schema': 'route-preview-v11-panel-summary-v1',
        'baseline': str(args.baseline.resolve()),
        'baseline_tag': BASELINE_TAG,
        'panel': str(args.panel.resolve()),
        'panel_sha256': __import__('hashlib').sha256(args.panel.read_bytes()).hexdigest(),
        'source_data_status': 'SOURCE_MOUNT_IO_ERROR',
        'source_data_read': False,
        'routes': len(route_manifest),
        'panel_frames': len(output_frames),
        'original_route_xml_match_routes': sum(bool(row.get('original_route_xml_match'))
                                               for row in route_manifest),
        'preview_valid': valid_count,
        'preview_failed': len(output_frames) - valid_count,
        'preview_coverage': valid_count / len(output_frames) if output_frames else 0.0,
        'recovered_from_local_segment': sum(row['navigation_source'] in
                                            ('LOCAL_REFERENCE_SEGMENT', 'TOPOLOGY_CANDIDATE')
                                            for row in output_frames),
        'baseline_valid': sum(row['baseline_preview_status'] == 'PASS'
                              for row in output_frames),
        'navigation_source_counts': dict(Counter(row['navigation_source'] for row in output_frames)),
        'provenance_labels': list(provenance_labels),
        'frame_provenance_counts': frame_provenance_counts,
        'route_provenance_counts': route_provenance_counts,
        'preview_status_counts': dict(Counter(row['preview_status'] for row in output_frames)),
        'preview_category_counts': dict(Counter(row['preview_category'] for row in output_frames)),
        'candidate_consensus_frames': sum(bool(row['candidate_consensus']) for row in output_frames),
        'status8_valid': sum(bool(row['status8_valid']) for row in output_frames),
        'command_matches_status4': sum(bool(row['command_matches_status4']) for row in output_frames),
        'route_with_valid_preview': sum(any(row['preview_valid'] for row in frames_by_route[route])
                                        for route in baseline_by_route),
        'no_decision_route_counts': {str(k): v for k, v in no_decision_counts.items()},
        'segment_rows': len(segment_rows),
        'recovery_counts': {str(k): v for k, v in recovery_counts.items()},
        'contract': {
            'command_order': list(COMMAND_ORDER),
            'preview_m': 20.0,
            'lateral_threshold_m': 2.0,
            'frame': 'rear_axle_world_to_rear_left',
            'uturn': UNSUPPORTED_UTURN,
        },
        'source_boundary': ('Panel uses frozen V1 reference paths and frozen causal '
                            'topology-candidate audit; no raw source read. Fallback '
                            'coverage is conditional and is not full-dataset proof.'),
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'event': 'v11_panel_done', 'frames': len(output_frames),
                      'valid': valid_count, 'recovered': summary['recovered_from_local_segment'],
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
