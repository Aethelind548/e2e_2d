"""Replay real session plans through the Agent Status builder and the offline rule.

For every frame of every route with a real ``set_global_plan`` capture this tool:

1. builds a fresh ``OnlineStatusBuilder`` from the captured dense world plan;
2. feeds the frame's rear pose three times at 10 Hz so Motion4 is well defined;
3. compares the builder's Navigation4 with the shared offline calculator
   (``route_preview_contract.preview_command4``);
4. compares rejection reasons for invalid previews;
5. records Status8 shape/one-hot/dtype and a determinism re-run.

The comparison deliberately targets *same-source calculator parity*: both sides
use the frozen rule, so 100% agreement is the acceptance bar.  It does not claim
that the historical offline Overlay agrees with the real plan where the route
branches differently.

Read-only: reads session captures, the route preview parquet and poses.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agent.common.types import CoordinateFrame, EgoState, Pose2D, Timestamp
from agent.config.settings import NavigationConfig
from agent.navigation.status_builder import OnlineStatusBuilder
from tools.b2d_adapter import FrameReader, RouteIndex
from tools.validation.b2d.certification_level_audit import load_sessions
from tools.validation.b2d.route_source_certification import pose_of
from tools.b2d_adapter.route_preview_contract import preview_command4

VERSION = 'online-status-parity-v1'
PREVIEW_M = 20.0
LATERAL_M = 2.0
MAX_PROJECTION_M = 5.0


def plan_entries(points):
    from types import SimpleNamespace
    option = SimpleNamespace(value=999)
    return [(SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.0),
                             rotation=SimpleNamespace(yaw=0.0)), option) for x, y in points]


def ego(rear_xy, yaw, t):
    return EgoState(Timestamp(t),
                    Pose2D(rear_xy[0], rear_xy[1], yaw, CoordinateFrame.CARLA_WORLD,
                           'lincoln-10hz-rear-v1'),
                    dynamics_frame=CoordinateFrame.MODEL_REAR,
                    provenance={'geometry_policy': 'lincoln-10hz-rear-v1'})


def run_builder(points, rear_xy, yaw, base_t=100.0):
    builder = OnlineStatusBuilder(NavigationConfig())
    builder.set_global_plan(plan_entries(points))
    status = None
    for index in range(3):
        stamp = Timestamp(base_t + index * 0.1)
        status = builder.update(ego(rear_xy, yaw, stamp.seconds), stamp)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', type=Path, required=True)
    parser.add_argument('--preview-root', type=Path,
                        default=Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/route_preview_v1'))
    parser.add_argument('--source-root', type=Path,
                        default=Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/v1'))
    parser.add_argument('--max-frames-per-route', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    sessions = load_sessions(args.session_dir)
    rows, errors = [], []
    counts = Counter()
    for route, session in sorted(sessions.items()):
        dense = session['points']
        path = args.preview_root / 'routes' / route / 'frame_route_preview.parquet'
        try:
            table = pq.read_table(path, columns=['frame', 'command4_json']).to_pylist()
            index = RouteIndex.from_dataset_root(args.source_root, route)
            reader = FrameReader(index)
        except Exception as error:  # noqa: BLE001
            errors.append({'route': route, 'error': f'{type(error).__name__}: {error}'})
            continue
        stored = {row['frame']: row for row in table}
        frames = sorted(set(index.frame_ids) & set(stored))
        if args.max_frames_per_route:
            frames = frames[:args.max_frames_per_route]
        for frame in frames:
            rear, yaw = pose_of(reader, index, frame)
            offline = preview_command4(dense, rear, yaw, preview_m=PREVIEW_M,
                                       lateral_threshold_m=LATERAL_M,
                                       max_projection_m=MAX_PROJECTION_M)
            status = run_builder(dense, rear, yaw)
            repeat = run_builder(dense, rear, yaw)
            offline_command = offline.get('command4')
            command_ok = status.command4 == (None if offline_command is None
                                             else tuple(float(v) for v in offline_command))
            if offline.get('preview_valid') is True:
                # Valid samples carry no failure reason on either side.
                reason_ok = status.validity == 'VALID'
            else:
                reason_ok = (status.reason or '').split('+')[0] == (offline.get('reason') or '')
            point_ok = (status.preview.get('preview_point_xy') is None
                        and offline.get('preview_point_xy') is None) or (
                status.preview.get('preview_point_xy') is not None
                and offline.get('preview_point_xy') is not None
                and np.allclose(status.preview['preview_point_xy'],
                                offline['preview_point_xy'], atol=1e-12, rtol=0.0))
            status8_ok = (status.status8 is None) or (
                len(status.status8) == 8 and abs(sum(status.status8[:4]) - 1.0) == 0.0)
            deterministic = status.status8 == repeat.status8
            stored_command = (json.loads(stored[frame]['command4_json'])
                              if stored[frame].get('command4_json') else None)
            counts['frames'] += 1
            counts['offline_valid'] += int(offline.get('preview_valid') is True)
            if offline.get('preview_valid') is True:
                counts['valid_checked'] += 1
                counts['command_match'] += int(command_ok)
            else:
                counts['invalid_checked'] += 1
                counts['invalid_reason_match'] += int(reason_ok)
            counts['reason_match'] += int(reason_ok)
            counts['point_match'] += int(point_ok)
            counts['status8_ok'] += int(status8_ok)
            counts['deterministic'] += int(deterministic)
            counts['status8_valid'] += int(status.validity == 'VALID')
            counts['overlay_command_equal'] += int(
                stored_command is not None and status.command4 == tuple(float(v) for v in stored_command))
            if not (command_ok and reason_ok and point_ok and status8_ok and deterministic):
                rows.append({'route': route, 'frame': frame, 'builder_command': status.command4,
                             'offline_command': offline_command,
                             'builder_reason': status.reason, 'offline_reason': offline.get('reason'),
                             'command_ok': command_ok, 'reason_ok': reason_ok,
                             'point_ok': point_ok, 'status8_ok': status8_ok,
                             'deterministic': deterministic})
            if offline.get('reason'):
                counts['reason:' + str(offline['reason'])] += 1

    summary = {
        'version': VERSION,
        'parameters': {'preview_m': PREVIEW_M, 'lateral_threshold_m': LATERAL_M,
                       'max_projection_m': MAX_PROJECTION_M},
        'routes': len(sessions),
        'counts': dict(counts),
        'mismatch_rows': len(rows),
        'errors': errors,
        'parity_rate': (counts['command_match'] / counts['valid_checked']) if counts['valid_checked'] else None,
        'invalid_reason_rate': (counts['invalid_reason_match'] / counts['invalid_checked'])
        if counts['invalid_checked'] else None,
        'overlay_equal_rate': (counts['overlay_command_equal'] / counts['valid_checked'])
        if counts['valid_checked'] else None,
        'status8_valid_rate': (counts['status8_valid'] / counts['frames']) if counts['frames'] else None,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'parity_mismatches.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    (args.output / 'parity_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in ('routes', 'counts', 'mismatch_rows', 'parity_rate',
                                              'status8_valid_rate')}, sort_keys=True))


if __name__ == '__main__':
    main()
