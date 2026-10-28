"""Audit real session routes on every stored frame, not five samples.

The earlier round compared five sampled frames per route.  This tool walks the
whole frame range of the routes that have a real ``set_global_plan`` capture and
recomputes the 20 m preview on the dense captured plan at every frame, using the
same parameters as the offline preview generator (``preview_m=20``,
``lateral_threshold_m=2.0``, ``max_projection_m=5.0``).

It records continuous-window statistics (runs of equal Command4, switch frames,
invalid runs) so coverage of turns and stop/start can be judged from frames
rather than from route names.

Read-only: reads dataset poses, the route preview parquet, the frozen overlay and
the session captures; writes one output directory.
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from tools.b2d_adapter import FrameReader, RouteIndex
from tools.training.diffusiondrive_v1.features_cache import sample_id as make_sample_id
from tools.validation.b2d.certification_level_audit import command_of, load_sessions
from tools.validation.b2d.route_source_certification import (
    dedupe_points,
    pose_of,
    preview_command4,
    project_progress,
)

VERSION = 'session-continuous-audit-v1'
PREVIEW_M = 20.0
LATERAL_THRESHOLD_M = 2.0
MAX_PROJECTION_M = 5.0
WINDOW_SWITCH_MARGIN = 25


def load_overlay_for_routes(path, routes):
    rows = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            route = row['sample_identity']['route']
            if route in routes:
                rows[row['sample_id']] = row
    return rows


def classify(session, frame, rear, yaw, offline_command, offline_motion, overlay_row):
    dense = dedupe_points(session['points'])
    preview = preview_command4(dense, rear, yaw, preview_m=PREVIEW_M,
                               lateral_threshold_m=LATERAL_THRESHOLD_M,
                               max_projection_m=MAX_PROJECTION_M)
    record = {
        'route': session['metadata'].get('dataset_route'),
        'frame': frame,
        'split': None if overlay_row is None else overlay_row['sample_identity']['split'],
        'offline_command': command_of(offline_command),
        'offline_status': 'PASS' if offline_command is not None else 'FAIL',
        'preview_status': preview.get('status'),
        'preview_reason': preview.get('reason'),
        'preview_command': command_of(preview.get('command4')),
        'preview_point_xy': preview.get('preview_point_xy'),
        'lateral_m': preview.get('lateral_m'),
        'longitudinal_m': preview.get('longitudinal_m'),
        'heading_delta_deg': preview.get('heading_delta_deg'),
        'session_capture_sha256': session['sha256'],
        'status_available': overlay_row is not None,
    }
    if preview.get('status') == 'PASS':
        rear_arc, rear_projection = project_progress(dense, rear)
        preview_arc, _ = project_progress(dense, preview['preview_point_xy'])
        record['rear_arc_m'] = round(rear_arc, 3)
        record['rear_projection_m'] = round(rear_projection, 3)
        record['horizon_m'] = round(preview_arc - rear_arc, 3)

    if offline_command is None:
        record['level'] = 'INSUFFICIENT_EVIDENCE'
        record['reason'] = 'OFFLINE_PREVIEW_INVALID'
        return record
    if preview.get('status') != 'PASS':
        record['level'] = 'INSUFFICIENT_EVIDENCE'
        record['reason'] = preview.get('reason') or 'SESSION_PREVIEW_INVALID'
        return record
    if record.get('rear_projection_m', 0.0) > MAX_PROJECTION_M:
        record['level'] = 'INSUFFICIENT_EVIDENCE'
        record['reason'] = 'COORDINATE_PROJECTION_TOO_FAR'
        return record
    if preview.get('command4') != offline_command:
        record['level'] = 'COMMAND_MISMATCH'
        record['reason'] = 'SESSION_COMMAND_DIFFERS_FROM_OFFLINE'
        return record
    record['command_match'] = True
    if overlay_row is None:
        record['level'] = 'COMMAND_MATCH_VERIFIED'
        record['reason'] = 'NO_FROZEN_STATUS_ROW'
        return record
    # Contract rule: the frozen row's Motion4 must be the frozen A0 values and the
    # Status row must not move them.  The preview parquet's recomputed motion is a
    # diagnostic only; it is not the A0 identity contract.
    a0 = list(overlay_row['a0_motion4'])
    old_motion = list(overlay_row['old_status8'][4:])
    new_motion = list(overlay_row['status8'][4:])
    record['a0_motion_identity'] = bool(old_motion == a0 == new_motion)
    if offline_motion is not None:
        record['recomputed_motion_max_abs_delta'] = round(max(
            abs(a - b) for a, b in zip(offline_motion, a0)), 6)
    if not record['a0_motion_identity']:
        record['level'] = 'COMMAND_MATCH_VERIFIED'
        record['reason'] = 'MOTION4_DIFFERS_FROM_FROZEN_A0'
        return record
    record['level'] = 'CERTIFIED'
    record['reason'] = 'SESSION_PLAN_COMMAND_VERIFIED'
    return record


def window_stats(records):
    """Runs of equal preview command, switch frames and invalid runs."""
    ordered = sorted(records, key=lambda r: r['frame'])
    runs = []
    current = None
    for record in ordered:
        key = record['preview_command'] or 'INVALID'
        if current is None or current['command'] != key:
            current = {'command': key, 'start': record['frame'], 'end': record['frame'],
                       'frames': 0}
            runs.append(current)
        current['end'] = record['frame']
        current['frames'] += 1
    switches = [runs[i]['start'] for i in range(1, len(runs))]
    invalid_runs = [run for run in runs if run['command'] == 'INVALID']
    frames_with_switch = 0
    for record in ordered:
        if any(abs(record['frame'] - switch) <= WINDOW_SWITCH_MARGIN for switch in switches):
            frames_with_switch += 1
    return {
        'runs': len(runs),
        'switch_frames': switches[:50],
        'switch_count': len(switches),
        'median_run_frames': statistics.median([run['frames'] for run in runs]) if runs else 0,
        'invalid_runs': len(invalid_runs),
        'invalid_frames': sum(run['frames'] for run in invalid_runs),
        'frames_within_switch_margin': frames_with_switch,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True,
                        help='comparison jsonl providing dataset_id and checked frames')
    parser.add_argument('--preview-root', type=Path,
                        default=Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/route_preview_v1'))
    parser.add_argument('--source-root', type=Path,
                        default=Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/v1'))
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    manifest = json.loads(args.manifest.read_text())
    dataset_id = manifest['dataset_id']
    sessions = load_sessions(args.session_dir)
    overlay = load_overlay_for_routes(args.overlay, set(sessions))
    columns = ['frame', 'preview_valid', 'preview_reason', 'command4_json',
               'motion4_json', 'route_progress_m']

    all_records, errors, per_route = [], [], {}
    for route, session in sorted(sessions.items()):
        path = args.preview_root / 'routes' / route / 'frame_route_preview.parquet'
        try:
            rows = {row['frame']: row for row in pq.read_table(path, columns=columns).to_pylist()}
            index = RouteIndex.from_dataset_root(args.source_root, route)
            reader = FrameReader(index)
        except Exception as error:  # noqa: BLE001
            errors.append({'route': route, 'error': f'{type(error).__name__}: {error}'})
            continue
        frames = sorted(set(index.frame_ids) & set(rows))
        if args.max_frames:
            frames = frames[:args.max_frames]
        records = []
        for frame in frames:
            stored = rows[frame]
            rear, yaw = pose_of(reader, index, frame)
            offline_command = json.loads(stored['command4_json']) if stored.get('command4_json') else None
            offline_motion = json.loads(stored['motion4_json']) if stored.get('motion4_json') else None
            sid = make_sample_id(dataset_id, route, frame)
            record = classify(session, frame, rear, yaw, offline_command, offline_motion,
                              overlay.get(sid))
            record['sample_id'] = sid
            records.append(record)
        per_route[route] = {
            'frames': len(records),
            'levels': dict(Counter(r['level'] for r in records)),
            'reasons': dict(Counter(r['reason'] for r in records)),
            'windows': window_stats(records),
            'classes': dict(Counter(r['offline_command'] for r in records)),
        }
        all_records.extend(records)

    summary = {
        'version': VERSION,
        'frames': len(all_records),
        'levels': dict(Counter(r['level'] for r in all_records)),
        'by_split': {f'{split}:{level}': count for (split, level), count in sorted(
            Counter((r['split'], r['level']) for r in all_records).items(),
            key=lambda item: (str(item[0][0]), item[0][1]))},
        'reasons': dict(Counter(r['reason'] for r in all_records)),
        'classes': dict(Counter(r['offline_command'] for r in all_records)),
        'errors': errors,
        'per_route': per_route,
        'parameters': {'preview_m': PREVIEW_M, 'lateral_threshold_m': LATERAL_THRESHOLD_M,
                       'max_projection_m': MAX_PROJECTION_M},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'session_continuous_rows.jsonl').open('w') as stream:
        for record in all_records:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
    (args.output / 'session_continuous_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in ('frames', 'levels', 'by_split', 'errors')},
                     sort_keys=True))


if __name__ == '__main__':
    main()
