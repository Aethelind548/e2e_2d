"""Offline terminal qualification; physical time stays unknown when not recorded.

This is a gate for the existing trajectory builder, not a second target builder.
Only explicit SI timestamps can certify the stationary duration. The frozen
10 Hz engineering clock is reported separately and never becomes an observation.
"""
import hashlib
import json
import math

from .coordinate_transform import actor_to_reference, wrap_angle
from .errors import AdapterError

POLICY = dict(schema='b2d-terminal-static-padding-v1', tail_frames=40,
              stop_speed_mps=0.1, minimum_stop_s=2.0,
              stop_path_m=0.2, stop_yaw_rad=math.radians(1),
              stop_step_m=0.02)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal_row(raw, geometry):
    rear = actor_to_reference(raw.actor_pose, (geometry.axle_dx_m, geometry.axle_dy_m))
    speed = float(raw.state['speed'])
    stamp = raw.annotation.get('timestamp_s')
    if not math.isfinite(speed) or (stamp is not None and not math.isfinite(float(stamp))):
        raise AdapterError('invalid_terminal_state', raw.ref.token)
    issues = []
    if raw.ego_box.get('type_id') != geometry.axle_vehicle_type:
        issues.append('reference_vehicle_mismatch')
    if max(abs(float(v)) for v in raw.ego_box['rotation'][:2]) > geometry.max_planar_tilt_deg:
        issues.append('nonplanar_pose')
    if raw.state.get('teleport_flag'):
        issues.append('teleport_flag')
    return dict(frame=raw.ref.frame_id, speed_mps=speed,
                rear_x_m=rear.x, rear_y_m=rear.y, rear_yaw_rad=rear.yaw,
                timestamp_s=None if stamp is None else float(stamp), geometry_issues=issues)


def terminal_metrics(rows, source_dt_s):
    """Measure a contiguous observed suffix; never turn frame IDs into timestamps."""
    if not rows or any(b['frame'] != a['frame'] + 1 for a, b in zip(rows, rows[1:])):
        raise AdapterError('terminal_annotation_gap', 'contiguous tail required')
    stamps = [r['timestamp_s'] for r in rows]
    clock = all(t is not None for t in stamps) and len(stamps) > 1
    if clock and any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise AdapterError('invalid_terminal_clock', 'non-increasing timestamp_s')
    if clock and any(abs((b-a)-source_dt_s) > 1e-3 for a, b in zip(stamps, stamps[1:])):
        raise AdapterError('terminal_clock_sampling_mismatch', 'fixed-stride builder requires the configured clock')
    steps = [math.hypot(b['rear_x_m']-a['rear_x_m'], b['rear_y_m']-a['rear_y_m'])
             for a, b in zip(rows, rows[1:])]
    yaws = [abs(wrap_angle(b['rear_yaw_rad']-a['rear_yaw_rad'])) for a, b in zip(rows, rows[1:])]
    stopped = 0
    for r in reversed(rows):
        if abs(r['speed_mps']) > POLICY['stop_speed_mps']:
            break
        stopped += 1
    start = len(rows)-stopped
    stop_steps, stop_yaws = steps[start:], yaws[start:]
    return dict(frame_count=len(rows), observed_span_s=stamps[-1]-stamps[0] if clock else None,
                nominal_span_s=(rows[-1]['frame']-rows[0]['frame'])*source_dt_s,
                physical_clock_verified=clock, path_m=sum(steps), yaw_abs_change_rad=sum(yaws),
                yaw_net_change_rad=wrap_angle(rows[-1]['rear_yaw_rad']-rows[0]['rear_yaw_rad']),
                final_speed_mps=rows[-1]['speed_mps'],
                final_pose_speed_nominal_mps=steps[-1]/source_dt_s if steps else None,
                stop_frames=stopped,
                stop_observed_s=(stamps[-1]-stamps[start]) if clock and stopped else None,
                stop_nominal_s=max(stopped-1, 0)*source_dt_s,
                stop_path_m=sum(stop_steps), stop_yaw_abs_rad=sum(stop_yaws),
                stop_max_step_m=max(stop_steps, default=0.0),
                max_step_m=max(steps, default=0.0), max_yaw_step_rad=max(yaws, default=0.0))


def classify_terminal(record, identity_valid, metrics, error=None):
    """Failure takes precedence over motion; successful status alone is insufficient."""
    if error or not identity_valid:
        return 'E', error or 'result_identity_invalid'
    status = record.get('status', '')
    infractions = record.get('infractions', {})
    reasons = [k for k in ('route_timeout', 'vehicle_blocked', 'route_dev', 'scenario_timeouts') if infractions.get(k)]
    if status not in ('Completed', 'Perfect') and status not in ('', 'Started', 'Running'):
        return 'E', 'status='+status+';'+(','.join(reasons) or 'termination_reason_not_recorded')
    if reasons:
        return 'E', ','.join(reasons)
    if abs(metrics['final_speed_mps']) > POLICY['stop_speed_mps']:
        return 'D', 'terminal_speed_above_stop_threshold'
    complete = status in ('Completed', 'Perfect') and record.get('scores', {}).get('score_route') == 100
    stable = (metrics['stop_observed_s'] is not None and metrics['stop_observed_s'] >= POLICY['minimum_stop_s']
              and metrics['stop_path_m'] <= POLICY['stop_path_m']
              and metrics['stop_yaw_abs_rad'] <= POLICY['stop_yaw_rad']
              and metrics['stop_max_step_m'] <= POLICY['stop_step_m']
              and metrics.get('geometry_valid', True))
    if complete and stable:
        return 'A', 'completed_and_observed_stationary_suffix'
    if complete:
        return 'B', 'stationary_duration_or_geometry_or_physical_clock_insufficient'
    return 'C', 'completion_or_termination_unknown'


def inspect_terminal(reader, geometry):
    """Read result and true annotations; reused by audit and the opt-in builder gate."""
    # Reuse the existing exact directory/result identity parser. This import is
    # confined to offline qualification, never the default frozen target path.
    from tools.validation.b2d.collection_source_recovery import parse_directory, result_identity
    index = reader.index
    path = index.route / 'result.json'
    result = json.loads(path.read_text())
    _, issues = result_identity(parse_directory(index.route.name), result)
    records = result.get('_checkpoint', {}).get('records', [])
    record = records[0] if len(records) == 1 else {}
    ids = index.frame_ids[-POLICY['tail_frames']:]
    rows = [terminal_row(reader.read(index.ref(i)), geometry) for i in ids]
    tail = terminal_metrics(rows, geometry.source_dt_s)
    # Extend backwards only when the entire tail is stopped, to measure the
    # complete stopped suffix rather than claim a 40-frame lower bound is exact.
    extra = []
    if tail['stop_frames'] == len(rows):
        for i in reversed(index.frame_ids[:-len(rows)]):
            row = terminal_row(reader.read(index.ref(i)), geometry)
            if abs(row['speed_mps']) > POLICY['stop_speed_mps']:
                break
            extra.append(row)
    suffix = terminal_metrics(list(reversed(extra))+rows, geometry.source_dt_s)
    for key in ('stop_frames', 'stop_observed_s', 'stop_nominal_s', 'stop_path_m', 'stop_yaw_abs_rad', 'stop_max_step_m'):
        tail[key] = suffix[key]
    error = None
    tail['geometry_valid'] = not any(r['geometry_issues'] for r in rows+extra)
    tail['geometry_issues'] = sorted({issue for r in rows+extra for issue in r['geometry_issues']})
    if 'teleport_flag' in tail['geometry_issues']:
        error = 'teleport_flag'
    if tail['max_step_m']/geometry.source_dt_s > geometry.max_speed_mps or tail['max_yaw_step_rad']/geometry.source_dt_s > geometry.max_yaw_rate_radps:
        error = 'teleport_or_discontinuity'
    category, reason = classify_terminal(record, not issues, tail, error)
    return dict(route=index.route.name, category=category, reason=reason,
                policy=POLICY, metrics=tail, last40=rows, record=record,
                entry_status=result.get('entry_status'), identity_issues=issues,
                result_path=str(path), result_sha256=file_sha(path))


def require_terminal_padding(reader, geometry):
    evidence = inspect_terminal(reader, geometry)
    if evidence['category'] != 'A':
        raise AdapterError('terminal_padding_ineligible', evidence['category']+':'+evidence['reason'])
    return evidence
