"""Interval-level check that Route Preview continues along the driven road.

The earlier rounds compared commands frame by frame against real session plans,
which is too strict for turn intervals: a 20 m preview can point into a junction
before the ego yaw changes.  This tool instead groups contiguous Left/Right
preview frames into a turn interval and asks whether the *expert trajectory over
that interval* bends to the same side.

It also consolidates the already frozen geometry audit, the 12 geometry-defect
routes and the confirmed session-plan mismatch intervals into one exception
index.  No preview data is regenerated.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from tools.b2d_adapter import FrameReader, RouteIndex
from tools.validation.b2d.route_source_certification import pose_of

VERSION = 'preview-trajectory-consistency-v2'
SIGN = {'Left': 1.0, 'Right': -1.0}
WINDOW_PAD_FRAMES = 5
LOOKAHEAD_FRAMES = 25
MIN_RUN_FRAMES = 3
ROUTE_TURN_DEG = 20.0
EXPERT_TURN_DEG = 15.0
EXPERT_STRAIGHT_DEG = 10.0


def wrap_deg(value):
    return (value + 180.0) % 360.0 - 180.0


def command_index(command4):
    if not command4 or len(command4) < 4 or 1.0 not in command4[:4]:
        return None
    return command4[:4].index(1.0)


def runs_of_turns(records):
    """Contiguous intervals where the route heading turns by >= 20 degrees."""
    runs, current = [], None
    for record in records:
        delta = record.get('heading_delta_deg')
        name = None
        if delta is not None and abs(delta) >= ROUTE_TURN_DEG and record['preview_valid']:
            name = 'Left' if delta > 0 else 'Right'
        if name is None:
            current = None
            continue
        if current is None or current['command'] != name:
            current = {'command': name, 'start': record['frame'], 'end': record['frame'],
                       'frames': 0, 'delta_sum': 0.0}
            runs.append(current)
        current['end'] = record['frame']
        current['frames'] += 1
        current['delta_sum'] += record['heading_delta_deg']
    return [run for run in runs if run['frames'] >= MIN_RUN_FRAMES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-root', type=Path,
                        default=Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/route_preview_v1'))
    parser.add_argument('--source-root', type=Path,
                        default=Path('/data/datasets/Bench2Drive-V0.0.4/v1'))
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--defect-exclusion', type=Path, required=True)
    parser.add_argument('--anomaly-intervals', type=Path, default=None)
    parser.add_argument('--route-step', type=int, default=25,
                        help='deterministic sampling stride over audit routes')
    parser.add_argument('--extra-routes', type=Path, default=None,
                        help='jsonl of routes (e.g. anomaly intervals) always included')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    audit = {row['route']: row for row in
             (json.loads(line) for line in args.audit.open())}
    defects = {row['route'] for row in
               json.loads(args.defect_exclusion.read_text())['defect_routes']}
    anomaly_routes = set()
    anomaly_by_route = {}
    if args.anomaly_intervals and args.anomaly_intervals.is_file():
        for line in args.anomaly_intervals.open():
            if not line.strip():
                continue
            row = json.loads(line)
            anomaly_routes.add(row['route'])
            anomaly_by_route.setdefault(row['route'], []).append(row)
    if args.extra_routes and args.extra_routes.is_file():
        anomaly_routes |= {line.strip() for line in args.extra_routes.read_text().splitlines() if line.strip()}

    ordered = sorted(audit)
    sampled = ordered[::max(1, args.route_step)]
    selected = sorted(set(sampled) | defects | {r for r in anomaly_routes if r in audit})

    exceptions, interval_rows = [], []
    counts = Counter()
    per_route = {}
    for route in selected:
        path = args.preview_root / 'routes' / route / 'frame_route_preview.parquet'
        try:
            table = pq.read_table(path, columns=['frame', 'command4_json', 'preview_valid',
                                                 'preview_point_xy_json',
                                                 'heading_delta_deg']).to_pylist()
            index = RouteIndex.from_dataset_root(args.source_root, route)
            reader = FrameReader(index)
        except Exception as error:  # noqa: BLE001
            exceptions.append({'route': route, 'kind': 'READ_ERROR', 'error': str(error)[:200]})
            continue
        records = []
        invalid_runs = []
        invalid_current = None
        for row in table:
            command4 = json.loads(row['command4_json']) if row.get('command4_json') else None
            record = {'frame': row['frame'], 'command_index': command_index(command4),
                      'preview_valid': bool(row['preview_valid']),
                      'heading_delta_deg': row.get('heading_delta_deg')}
            records.append(record)
            if not record['preview_valid']:
                counts['invalid_preview_frames'] += 1
                if invalid_current is None:
                    invalid_current = {'start': row['frame'], 'end': row['frame'], 'frames': 0}
                    invalid_runs.append(invalid_current)
                invalid_current['end'] = row['frame']
                invalid_current['frames'] += 1
            else:
                invalid_current = None
        pose_cache = {}

        def pose_at(frame):
            if frame not in pose_cache:
                pose_cache[frame] = pose_of(reader, index, frame)
            return pose_cache[frame]

        route_counts = Counter()
        for run in runs_of_turns(records):
            start = max(run['start'] - WINDOW_PAD_FRAMES, records[0]['frame'])
            end = min(run['end'] + LOOKAHEAD_FRAMES, records[-1]['frame'])
            try:
                rear_start, yaw_start = pose_at(start)
                rear_end, yaw_end = pose_at(end)
            except Exception as error:  # noqa: BLE001
                exceptions.append({'route': route, 'kind': 'POSE_READ_ERROR',
                                   'frame': start, 'error': str(error)[:200]})
                continue
            route_delta = run['delta_sum'] / max(1, run['frames'])
            # The shared contract reports positive heading delta for Left, so the
            # expert side uses the same wrapped yaw change without sign flipping.
            expert_left = wrap_deg(math.degrees(yaw_end - yaw_start))
            if abs(expert_left) < EXPERT_STRAIGHT_DEG:
                verdict = 'route_turn_expert_straight'
            elif np.sign(expert_left) == np.sign(route_delta) and abs(expert_left) >= EXPERT_TURN_DEG:
                verdict = 'aligned'
            elif np.sign(expert_left) != np.sign(route_delta) and abs(expert_left) >= EXPERT_TURN_DEG:
                verdict = 'opposite_branch'
            else:
                verdict = 'ambiguous'
            counts[verdict] += 1
            route_counts[verdict] += 1
            row = {'route': route, 'kind': 'TURN_INTERVAL', 'command': run['command'],
                   'start_frame': start, 'end_frame': end, 'run_frames': run['frames'],
                   'route_heading_delta_deg': round(route_delta, 2),
                   'expert_heading_delta_deg': round(expert_left, 2),
                   'verdict': verdict}
            interval_rows.append(row)
            if verdict != 'aligned':
                exceptions.append(row)
        per_route[route] = {'turn_intervals': sum(route_counts.values()),
                            'aligned': route_counts['aligned'],
                            'route_turn_expert_straight': route_counts['route_turn_expert_straight'],
                            'ambiguous': route_counts['ambiguous'],
                            'opposite_branch': route_counts['opposite_branch'],
                            'invalid_preview_frames': sum(r['frames'] for r in invalid_runs),
                            'frames': len(records)}
        for issue in audit[route].get('issues') or []:
            exceptions.append({'route': route, 'kind': 'GEOMETRY_ISSUE', 'issue': issue})
        for invalid in invalid_runs:
            exceptions.append({'route': route, 'kind': 'INVALID_PREVIEW_RUN',
                               'start_frame': invalid['start'], 'end_frame': invalid['end'],
                               'frames': invalid['frames']})
        for anomaly in anomaly_by_route.get(route, []):
            exceptions.append({'route': route, 'kind': 'CONFIRMED_PLAN_INTERVAL',
                               'interval_kind': anomaly.get('kind'),
                               'start_frame': anomaly.get('start_frame'),
                               'end_frame': anomaly.get('end_frame'),
                               'frames': anomaly.get('frames')})

    verdict_keys = ('aligned', 'route_turn_expert_straight', 'ambiguous', 'opposite_branch')
    summary = {
        'version': VERSION,
        'routes_sampled': len(selected),
        'routes_total': len(audit),
        'route_step': args.route_step,
        'turn_intervals': sum(counts[k] for k in verdict_keys),
        'verdicts': {k: counts[k] for k in verdict_keys},
        'invalid_preview_frames_sampled': counts['invalid_preview_frames'],
        'geometry_defect_routes': sorted(defects),
        'confirmed_plan_routes': sorted(r for r in anomaly_routes if r in audit),
        'per_route': per_route,
        'parameters': {'window_pad_frames': WINDOW_PAD_FRAMES,
                       'lookahead_frames': LOOKAHEAD_FRAMES,
                       'min_run_frames': MIN_RUN_FRAMES,
                       'route_turn_deg': ROUTE_TURN_DEG,
                       'expert_turn_deg': EXPERT_TURN_DEG},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'preview_geometry_exception_index.jsonl').open('w') as stream:
        for row in exceptions:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    with (args.output / 'turn_interval_index.jsonl').open('w') as stream:
        for row in interval_rows:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    (args.output / 'preview_consistency_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in ('routes_sampled', 'turn_intervals', 'verdicts',
                                              'invalid_preview_frames_sampled')}, sort_keys=True))


if __name__ == '__main__':
    main()
