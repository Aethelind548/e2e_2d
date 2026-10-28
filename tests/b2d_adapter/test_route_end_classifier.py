from tools.validation.b2d.route_end_classifier import classify_failure
from tools.b2d_adapter.terminal_policy import terminal_metrics, classify_terminal
from tools.b2d_adapter.errors import AdapterError
import pytest


def _terminal_rows(speed=0., timestamps=True, yaw=0.):
    return [dict(frame=i, speed_mps=speed, rear_x_m=0., rear_y_m=0., rear_yaw_rad=yaw,
                 timestamp_s=i*.1 if timestamps else None) for i in range(40)]


def test_terminal_classes_require_success_and_observed_stationarity():
    record = dict(status='Completed', scores={'score_route':100}, infractions={})
    m = terminal_metrics(_terminal_rows(), .1)
    assert classify_terminal(record, True, m)[0]=='A'
    assert m['observed_span_s']==pytest.approx(3.9)
    assert m['stop_observed_s']==pytest.approx(3.9)
    unknown_clock = terminal_metrics(_terminal_rows(timestamps=False), .1)
    assert unknown_clock['observed_span_s'] is None
    assert unknown_clock['nominal_span_s']==pytest.approx(3.9)
    assert classify_terminal(record, True, unknown_clock)[0]=='B'
    assert classify_terminal(dict(status='Started'), True, m)[0]=='C'
    moving=terminal_metrics(_terminal_rows(speed=3.), .1)
    assert classify_terminal(record, True, moving)[0]=='D'
    assert classify_terminal(dict(status='Failed'), True, moving)[0]=='E'
    assert classify_terminal(record, False, m)[0]=='E'
    assert classify_terminal(dict(record, infractions={'route_timeout':['timeout']}), True, m)[0]=='E'


def test_scattered_slow_frames_do_not_become_a_stationary_suffix():
    rows = _terminal_rows(speed=1.)
    for i in range(10):
        rows[i]['speed_mps']=0.
    rows[-1]['speed_mps']=0.
    m=terminal_metrics(rows,.1)
    assert m['stop_frames']==1
    assert m['stop_observed_s']==0.
    assert classify_terminal(dict(status='Perfect',scores={'score_route':100}),True,m)[0]=='B'


def test_signed_raw_speed_uses_magnitude_without_rewriting_observations():
    rows=_terminal_rows(speed=-0.001)
    m=terminal_metrics(rows,.1)
    assert m['stop_frames']==40
    assert m['final_speed_mps']==-0.001
    rows[-1]['speed_mps']=-3.
    m=terminal_metrics(rows,.1)
    assert classify_terminal(dict(status='Completed',scores={'score_route':100}),True,m)[0]=='D'


def test_terminal_geometry_and_yaw_wrap_are_checked():
    rows=_terminal_rows(yaw=3.1415)
    rows[-1]['rear_yaw_rad']=-3.1415
    m=terminal_metrics(rows,.1)
    assert m['yaw_abs_change_rad']<.001
    rows[-1]['rear_x_m']=1.
    m=terminal_metrics(rows,.1)
    assert classify_terminal(dict(status='Completed',scores={'score_route':100}),True,m)[0]=='B'


def test_terminal_gaps_and_nonmonotonic_or_irregular_clock_reject():
    for field,value in [('frame',99),('timestamp_s',0.),('timestamp_s',4.9)]:
        rows=_terminal_rows();rows[-1][field]=value
        with pytest.raises(AdapterError):
            terminal_metrics(rows,.1)


def _row(frame, reason='local_segment_end', validity='LOCAL_SEGMENT_END'):
    return {'frame': frame, 'failure_reason': reason, 'preview_validity': validity}


def test_local_end_at_source_clip_end_is_expected():
    segments = [{'segment_id': 0, 'start_frame': 0, 'end_frame': 100}]
    kind, evidence, _, _, _ = classify_failure(_row(100), segments, 100)
    assert kind == 'EXPECTED_SEQUENCE_END'
    assert 'source_clip_end' in evidence


def test_local_end_far_before_source_clip_end_is_truncation():
    segments = [{'segment_id': 0, 'start_frame': 0, 'end_frame': 100}]
    kind, evidence, _, _, _ = classify_failure(_row(50), segments, 100)
    assert kind == 'INTERNAL_SEGMENT_TRUNCATION'
    assert 'local_path_end' in evidence


def test_boundary_with_following_segment_is_gap():
    segments = [
        {'segment_id': 0, 'start_frame': 0, 'end_frame': 50},
        {'segment_id': 1, 'start_frame': 60, 'end_frame': 100},
    ]
    kind, evidence, _, _, _ = classify_failure(_row(50), segments, 100)
    assert kind == 'INTERNAL_ROUTE_GAP'
    assert 'following_segment' in evidence


def test_ambiguous_and_other_geometry_are_separate():
    segments = [{'segment_id': 0, 'start_frame': 0, 'end_frame': 100}]
    kind, _, _, _, _ = classify_failure(
        _row(50, 'candidate_disagreement_at_20m', 'AMBIGUOUS'), segments, 100)
    assert kind == 'AMBIGUOUS_TOPOLOGY'
    kind, _, _, _, _ = classify_failure(
        _row(50, 'ValueError:teleport_flag', 'INVALID'), segments, 100)
    assert kind == 'OTHER_INVALID_GEOMETRY'
