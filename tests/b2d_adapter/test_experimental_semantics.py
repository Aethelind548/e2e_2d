import json
import math
import numpy as np
import pytest
from tools.b2d_adapter.experimental_causal_state import (
    A0, A1, A2, A2V, causal_state, reference_state,
)
from tools.b2d_adapter.experimental_route_preview import (
    RoutePolyline, bearing_command, preview_command,
)
from tools.b2d_adapter.experimental_route_progress import ReferenceSegment, RouteProgressTracker
from tools.b2d_adapter.experimental_status_builder import build_status, build_overlay_record, write_status_overlay
from tools.validation.b2d.adapter_semantics_prototype import independent_motion_reference, production_a0_from_records, audit_clock_records
from tools.b2d_adapter.coordinate_transform import ActorPose

def test_a1_quadratic_state_is_exact_and_causal():
    dt = .1
    times = np.array([-3, -2, -1, 0], dtype=float) * dt
    a = 2.0
    v0 = 3.0
    x = .5 * a * times ** 2 + v0 * times
    poses = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    s = causal_state(poses, method=A1, dt=dt)
    assert s.velocity[0] == pytest.approx(v0, abs=1e-12)
    assert s.velocity[1] == pytest.approx(0, abs=1e-12)
    assert s.acceleration[0] == pytest.approx(a, abs=1e-12)
    assert s.acceleration[1] == pytest.approx(0, abs=1e-12)


def test_a0_has_expected_midpoint_lag_on_acceleration():
    dt = .1
    times = np.array([-2, -1, 0], dtype=float) * dt
    a = 2.0
    x = .5 * a * times ** 2
    poses = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    s = causal_state(poses, method=A0, dt=dt)
    # Position second difference at t is the acceleration at t-1, not a
    # physically different label; this is the documented 100 ms causal lag.
    assert s.acceleration[0] == pytest.approx(a, abs=1e-12)


def test_a2_uses_speed_only_as_magnitude_and_keeps_direction():
    dt = .1
    times = np.array([-3, -2, -1, 0], dtype=float) * dt
    # CARLA world +y is right; this motion is [forward,left]=[3,4]/5.
    poses = np.column_stack((.6 * times, -.8 * times, np.zeros_like(times)))
    s = causal_state(poses, method=A2, speed=10.0, dt=dt)
    np.testing.assert_allclose(s.velocity, [6.0, 8.0], atol=1e-12)


def test_a2v_keeps_a0_acceleration_but_uses_speed_fused_velocity():
    dt = .1
    times = np.array([-3, -2, -1, 0], dtype=float) * dt
    a = 2.0
    v0 = 3.0
    x = .5 * a * times ** 2 + v0 * times
    poses = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    s = causal_state(poses, method=A2V, speed=v0, dt=dt)
    assert s.velocity[0] == pytest.approx(3.0, abs=1e-12)
    assert s.acceleration[0] == pytest.approx(a, abs=1e-12)


def test_reference_requires_future_pose_and_is_not_used_by_candidates():
    poses = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    ref = reference_state(poses, dt=.1)
    assert ref.velocity[0] == pytest.approx(10.0)
    # Candidate API only accepts the trailing window it is given.
    with pytest.raises(ValueError):
        causal_state(poses[:2], method=A1, dt=.1)


def test_route_preview_straight_left_and_right():
    dt = .1
    straight = RoutePolyline.from_waypoints([[0, 0], [50, 0], [100, 0]], 'synthetic')
    p = preview_command(straight, (0, 0), 0, preview_m=20, lateral_threshold_m=2)
    assert p['category_name'] == 'Straight'

    left = RoutePolyline.from_waypoints(
        [[0, 0], [10, 0], [20, 0], [20, -10], [20, -30]], 'synthetic-left')
    p = preview_command(left, (0, 0), 0, preview_m=25, lateral_threshold_m=2)
    assert p['category_name'] == 'Left'
    assert p['lateral_m'] > 2

    right = RoutePolyline.from_waypoints(
        [[0, 0], [10, 0], [20, 0], [20, 10], [20, 30]], 'synthetic-right')
    p = preview_command(right, (0, 0), 0, preview_m=25, lateral_threshold_m=2)
    assert p['category_name'] == 'Right'
    assert p['lateral_m'] < -2


def test_route_preview_uses_arc_length_and_fails_at_route_end():
    route = RoutePolyline.from_waypoints([[0, 0], [10, 0], [20, 0]], 'short')
    p = preview_command(route, (0, 0), 0, preview_m=15, lateral_threshold_m=2)
    assert p['target_xy'] == pytest.approx([15, 0])
    p = preview_command(route, (0, 0), 0, preview_m=25, lateral_threshold_m=2)
    assert p['category_name'] == 'Unknown'
    assert p['reason'] == 'route_end_before_preview'


def test_bearing_command_sign_and_distance_gate():
    left = bearing_command((5, -3), (0, 0), 0)
    right = bearing_command((5, 3), (0, 0), 0)
    straight = bearing_command((5, .1), (0, 0), 0)
    assert left['category_name'] == 'Left'
    assert right['category_name'] == 'Right'
    assert straight['category_name'] == 'Straight'
    behind = bearing_command((-1, 0), (0, 0), 0)
    assert behind['category_name'] == 'Unknown'





def progress_segment(points, sid=0, offset=0, **kwargs):
    return ReferenceSegment(sid, RoutePolyline(points, 'synthetic'), offset, **kwargs)


def left_progress_tracker():
    return RouteProgressTracker([progress_segment([[0, 0], [10, 0], [10, -50]])])


def test_stationary_jitter_no_branch_jump():
    # Two parallel portions of one expected route, separated by 50 m in arc.
    route = progress_segment([[0, 0], [30, 0], [30, 1], [0, 1], [-30, 1]])
    tracker = RouteProgressTracker([route])
    first = tracker.update([10, 0], 0)
    assert first['route_progress_m'] == pytest.approx(10)
    for y in (.1, -.1, .12, -.08):
        result = tracker.update([10.01, y], 0)
        assert result['route_progress_valid']
        assert result['route_progress_m'] == pytest.approx(10.01)


def test_waiting_left_intent_and_creep_startup():
    tracker = left_progress_tracker()
    for index, x in enumerate([0, .001, -.001, .01, .02, .05, .1, .5, 1]):
        result = tracker.update([x, 0], 0, timestamp=index * .1)
        assert result['preview_valid']
        assert result['category'] == 'Left'
        assert result['route_progress_m'] == pytest.approx(max(0, x))
    assert result['route_progress_m'] == 1


def test_current_body_yaw_controls_local_frame_when_stopped():
    tracker = left_progress_tracker()
    first = tracker.update([10, -2], -math.pi / 4)
    second = tracker.update([10, -2], -math.pi / 2)
    assert first['local_xy_m'][1] > 2
    assert second['local_xy_m'] == pytest.approx([20, 0], abs=1e-12)
    assert second['category'] == 'Straight'


def test_after_turn_command_transitions_without_lag():
    tracker = left_progress_tracker()
    assert tracker.update([0, 0], 0)['category'] == 'Left'
    for x in [1, 4, 8, 10]:
        tracker.update([x, 0], 0)
    result = tracker.update([10, -1], -math.pi / 2)
    assert result['route_progress_m'] == pytest.approx(11)
    assert result['category'] == 'Straight'


def test_reverse_progress_and_relocalization_and_replan():
    tracker = RouteProgressTracker([progress_segment([[0, 0], [100, 0]])])
    assert tracker.update([30, 0], 0)['route_progress_m'] == 30
    result = tracker.update([28, 0], 0)
    assert result['route_progress_m'] == pytest.approx(28)
    assert result['reverse_motion']
    # Self-overlapping route needs a fresh match after relocation, while a
    # route revision cannot silently keep old progress.
    result = tracker.update([70, 0], 0, relocalized=True)
    assert result['recovery_reason'] == 'relocalized'
    assert result['route_progress_m'] == pytest.approx(70)
    assert tracker.update([70, 0], 0, route_revision='v2')['preview_validity'] == 'ROUTE_REVISION_MISMATCH'
    tracker.set_route([progress_segment([[50, 0], [150, 0]])], route_revision='v2')
    result = tracker.update([70, 0], 0, route_revision='v2')
    assert result['recovery_reason'] == 'route_replanned'
    assert result['route_progress_m'] == 20


@pytest.mark.parametrize('end_kind', ['TASK_END', 'LOCAL_SEGMENT_END', 'MAP_MISSING', 'ROUTE_GAP'])
def test_endpoint_kinds_are_explicit_and_not_unknown(end_kind):
    tracker = RouteProgressTracker([progress_segment([[0, 0], [10, 0]], end_kind=end_kind)])
    result = tracker.update([0, 0], 0)
    assert result['preview_validity'] == end_kind
    assert result['failure_reason'] == end_kind.lower()
    assert result['command4'] is None
    assert result['category'] is None


def test_gap_rejected_even_with_graph_link_but_no_missing_geometry():
    first = progress_segment([[0, 0], [10, 0]], directed_next_segment_ids=(1,), end_kind='ROUTE_GAP')
    second = progress_segment([[20, 0], [80, 0]], sid=1, offset=10)
    tracker = RouteProgressTracker([first, second])
    assert tracker.update([0, 0], 0)['preview_validity'] == 'ROUTE_GAP'


def test_certified_directed_continuous_geometry_allows_preview_and_match():
    first = progress_segment([[0, 0], [10, 0]], directed_next_segment_ids=(1,))
    second = progress_segment([[10, 0], [80, 0]], sid=1, offset=10)
    tracker = RouteProgressTracker([first, second])
    assert tracker.update([0, 0], 0)['preview_point_xy'] == pytest.approx([20, 0])
    for x in [4, 8, 10, 11]:
        result = tracker.update([x, 0], 0)
    assert result['reference_segment_id'] == 1
    assert result['route_progress_m'] == 11


def test_ambiguous_crossing_rejected_without_history():
    route = progress_segment([[0, 0], [30, 0], [30, 10], [0, 10], [0, 0], [50, 0]])
    tracker = RouteProgressTracker([route])
    result = tracker.update([10, 0], 0)
    assert result['preview_validity'] == 'AMBIGUOUS_MATCH'
    assert result['command4'] is None


def test_deterministic_and_no_future_input_and_route_is_frozen():
    points = np.array([[0., 0.], [100., 0.]])
    segment = progress_segment(points)
    points[:] = -99
    a = RouteProgressTracker([segment]).update([0, 0], 0)
    b = RouteProgressTracker([segment]).update([0, 0], 0)
    assert a == b
    assert a['preview_point_xy'] == [20, 0]
    assert a['source_status'] == 'NOT_VERIFIED'
    with pytest.raises(TypeError):
        RouteProgressTracker([segment]).update([0, 0], 0, future_pose=[1, 0])


def test_two_meter_threshold_is_exact_and_unsmoothed():
    route = progress_segment([[0, 0], [100, 0]])
    assert RouteProgressTracker([route]).update([0, 2], 0)['category'] == 'Straight'
    assert RouteProgressTracker([route]).update([0, 2.00001], 0)['category'] == 'Left'
    assert RouteProgressTracker([route]).update([0, -2.00001], 0)['category'] == 'Right'


def test_mapping_does_not_promote_reconstruction_trust_or_infer_task_end():
    segment = ReferenceSegment.from_mapping({'segment_id': 4, 'points_xy': [[0, 0], [100, 0]],
                                             'trust': 'HIGH', 'navigation_source': 'RECONSTRUCTED_ROUTE'})
    assert segment.navigation_source == 'RECONSTRUCTED_ROUTE_OFFLINE'
    assert segment.source_status == 'NOT_VERIFIED'
    assert segment.end_kind == 'LOCAL_SEGMENT_END'


def test_invalid_timestamp_and_pose_are_rejected():
    tracker = left_progress_tracker()
    tracker.update([0, 0], 0, timestamp=1)
    assert tracker.update([0, 0], 0, timestamp=.5)['preview_validity'] == 'INVALID_TIME'
    assert tracker.update([0, 0], float('nan'))['preview_validity'] == 'INVALID_POSE'



STATUS_SAMPLE_IDENTITY = {'sample_id':'s1','route':'r1','frame':10}
def status_result(command=(1,0,0,0), motion=(1,2,3,4), nav='VALID', mov='VALID', provenance='NOT_VERIFIED', source='RECONSTRUCTED_ROUTE_OFFLINE', **kwargs):
    return build_status(command,motion,nav,mov,source,STATUS_SAMPLE_IDENTITY,navigation_provenance=provenance,**kwargs)
def status_overlay_record(result=None):
    return build_overlay_record(result or status_result(), [0,1,0,0,1,2,3,4], feature_sha256='a'*64,target_sha256='b'*64)

def test_eight_dimensional_contract_and_determinism():
    a=status_result(provenance='CERTIFIED'); b=status_result(provenance='CERTIFIED')
    assert a.status_validity=='VALID'
    assert a.status8.dtype==np.float32 and a.status8.shape==(8,)
    np.testing.assert_array_equal(a.status8,[1,0,0,0,1,2,3,4])
    assert a.as_dict()==b.as_dict()
    assert not a.status8.flags.writeable

def test_legal_unknown_vs_failure_and_unverified():
    unknown=status_result(command=[0,0,0,1],provenance='CERTIFIED')
    assert unknown.status_validity=='VALID' and unknown.diagnostic_metadata['legal_unknown']
    failed=status_result(command=None,nav='FAILED')
    assert failed.status8 is None and failed.status_validity=='INVALID'
    unverified=status_result(command=[0,0,0,1])
    assert unverified.status8 is not None and unverified.status_validity=='NOT_VERIFIED'
    failed_with_unknown=status_result(command=[0,0,0,1],nav='FAILED')
    assert failed_with_unknown.status8 is None
    assert not failed_with_unknown.diagnostic_metadata['legal_unknown']

def test_near_fallback_cannot_silently_claim_route_certification():
    fallback=status_result(source='NEAR_FALLBACK')
    assert fallback.status_validity=='NOT_VERIFIED' and fallback.fallback_source=='NEAR_FALLBACK'
    with pytest.raises(ValueError):status_result(source='NEAR_FALLBACK',provenance='CERTIFIED')

def test_motion_contract_does_not_override_failure():
    assert status_result(mov='FAILED').status8 is None
    assert status_result(mov='NOT_VERIFIED',provenance='CERTIFIED').status_validity=='NOT_VERIFIED'

@pytest.mark.parametrize('value',[[1,0,0], [0,0,0,0], [1,1,0,0], [float('nan'),0,0,0], [1e40,0,0,0], ['1','0','0','0'], [1+1j,0,0,0]])
def test_bad_commands_rejected(value):
    with pytest.raises(ValueError):status_result(command=value)

@pytest.mark.parametrize('value',[[1,2,3], [1,2,3,float('inf')], [1,2,3,1e40]])
def test_bad_motion_rejected(value):
    with pytest.raises(ValueError):status_result(motion=value)

def test_overlay_identity_sources_and_exclusive_write(tmp_path):
    row=status_overlay_record(); path=tmp_path/'status_overlay_manifest.jsonl'
    summary=write_status_overlay(path,[row]); saved=json.loads(path.read_text())
    assert summary['rows']==1 and not saved['training_admitted']
    assert saved['source_identity']['feature_sha256']=='a'*64
    assert len(saved['generator']['sha256'])==64
    assert saved['source_identity_validity']=='NOT_VERIFIED'
    with pytest.raises(FileExistsError):write_status_overlay(path,[row])
    assert json.loads(path.read_text())==saved

def test_duplicate_id_and_physical_identity_rejected_before_write(tmp_path):
    row=status_overlay_record(); path=tmp_path/'duplicates.jsonl'
    with pytest.raises(ValueError):write_status_overlay(path,[row,row])
    assert not path.exists()
    other=json.loads(json.dumps(row));other['sample_id']='s2';other['sample_identity']['sample_id']='s2';other['sample_identity']['frame']='00010'
    with pytest.raises(ValueError):write_status_overlay(path,[row,other])
    assert not path.exists()

def test_overlay_rejects_forged_valid_and_admission(tmp_path):
    row=status_overlay_record();row['status_validity']='VALID'
    with pytest.raises(ValueError):write_status_overlay(tmp_path/'forged.jsonl',[row])
    row=status_overlay_record();row['training_admitted']=True
    with pytest.raises(ValueError):write_status_overlay(tmp_path/'admitted.jsonl',[row])

def test_failure_overlay_preserves_missing(tmp_path):
    row=build_overlay_record(status_result(command=None,nav='FAILED'), None, feature_sha256=None,target_sha256=None)
    write_status_overlay(tmp_path/'failed.jsonl',[row])
    assert row['status8'] is None and row['old_status8'] is None
    assert 'old_status8' in row['missing_source_evidence']




def records_from_actor(points):
    return [{'frame': i + 10, 'actor_pose': ActorPose(*p), 'sha256': f'sha-{i}',
             'anno': {'bounding_boxes': [{'class':'ego_vehicle', 'type_id':'vehicle.lincoln.mkz_2020'}]}}
            for i,p in enumerate(points)]


def test_independent_reference_current_body_axes_and_acceleration():
    # CARLA world +Y at heading +pi/2 is forward; +X is local left.
    poses = [[0., -0.3, math.pi/2], [0., 0., math.pi/2], [0., .5, math.pi/2]]
    velocity, acceleration = independent_motion_reference(poses)
    np.testing.assert_allclose(velocity, [4,0], atol=1e-12)
    np.testing.assert_allclose(acceleration, [20,0], atol=1e-12)


def test_production_causal_reader_uses_only_trailing_three_and_keeps_float32():
    points = [[0.,0.,0.], [.01,0.,0.], [.03,0.,0.]]
    state, read_ids = production_a0_from_records(records_from_actor(points))
    assert read_ids == [10,11,12]
    assert state.velocity.dtype == np.float32
    np.testing.assert_allclose(state.velocity, [.2,0], atol=1e-7)
    np.testing.assert_allclose(state.acceleration, [1,0], atol=1e-7)


def test_production_guard_rejects_missing_frame_instead_of_fixed_dt_difference():
    records = records_from_actor([[0.,0.,0.], [1.,0.,0.], [3.,0.,0.]])
    records[-1]['frame'] += 1
    with pytest.raises(ValueError, match='nonconsecutive'):
        production_a0_from_records(records)


def test_stationary_rotation_changes_rear_reference_velocity():
    records = records_from_actor([[0.,0.,-.02], [0.,0.,-.01], [0.,0.,0.]])
    state, _ = production_a0_from_records(records)
    # Actor is fixed, rear axle follows a circle under rotation. This difference
    # is physical reference-point kinematics, not zero-speed replacement.
    assert np.linalg.norm(state.velocity) > .1
    assert abs(state.velocity[1]) > .1


def test_timestamp_absence_is_unknown_not_zero_anomaly_count():
    result = audit_clock_records([{'frame': i, 'anno': {}} for i in range(3)])
    assert result['timestamp_missing'] == 3
    assert result['timestamp_abnormal_intervals'] is None
    assert result['physical_interval_status'] == 'NOT_VERIFIED'


def test_timestamp_actual_anomaly_and_missing_frame_counts():
    records = [{'frame': f, 'anno': {'timestamp': t}} for f,t in [(0,0.),(1,.1),(3,.8),(4,.7)]]
    result = audit_clock_records(records)
    assert result['missing_frame_ids'] == 1
    assert result['timestamp_abnormal_intervals'] == 2
    assert result['timestamp_nonpositive_intervals'] == 1


@pytest.mark.parametrize('dt', [0.,-1.,float('nan')])
def test_independent_reference_rejects_invalid_clock(dt):
    with pytest.raises(ValueError):
        independent_motion_reference(np.zeros((3,3)), dt=dt)


def test_crossing_history_preserves_the_selected_route_occurrence():
    tracker = RouteProgressTracker([progress_segment([[-20, 0], [30, 0], [30, 10], [0, 10], [0, 0], [50, 0]])])
    for x in [-10, -5, -1, 0, 1, 2]:
        result = tracker.update([x, .01], 0)
        assert result['route_progress_valid']
        assert result['route_progress_m'] == pytest.approx(x + 20)


def test_reverse_with_opposite_body_heading_keeps_valid_match():
    tracker = RouteProgressTracker([progress_segment([[0, 0], [100, 0]])])
    result = tracker.update([30, 0], math.pi, reverse=True)
    assert result['route_progress_valid']
    assert result['heading_error_deg'] == pytest.approx(180)
    assert result['preview_validity'] == 'TARGET_NOT_FORWARD'
    assert result['command4'] is None


def test_cross_segment_provenance_cannot_be_promoted():
    first = progress_segment([[0, 0], [10, 0]], source_status='VERIFIED', directed_next_segment_ids=(1,))
    second = progress_segment([[10, 0], [80, 0]], sid=1, offset=10, source_status='NOT_VERIFIED')
    result = RouteProgressTracker([first, second]).update([0, 0], 0)
    assert result['preview_valid']
    assert result['source_status'] == 'NOT_VERIFIED'


def test_protected_snapshot_detects_changed_or_missing_source(tmp_path):
    from tools.validation.b2d.status_upstream_audit import sha256, verify_protected_snapshot
    source = tmp_path / 'protected.py'
    source.write_text('frozen = True\n')
    snapshot = {'protected_files': {str(source): {'sha256': sha256(source), 'bytes': source.stat().st_size}}}
    assert verify_protected_snapshot(snapshot) == []
    source.write_text('frozen = False\n')
    assert verify_protected_snapshot(snapshot) == [str(source)]
    source.unlink()
    assert verify_protected_snapshot(snapshot) == [str(source)]


def test_overlay_keeps_exact_frozen_a0_motion_components():
    records = records_from_actor([[0., 0., -.02], [.01, .02, -.01], [.04, .05, 0.]])
    a0, read_ids = production_a0_from_records(records)
    motion = np.concatenate([a0.velocity, a0.acceleration])
    result = build_status([1, 0, 0, 0], motion, 'VALID', 'NOT_VERIFIED',
                          'RECONSTRUCTED_ROUTE_OFFLINE', {'sample_id': 'frozen-a0'})
    np.testing.assert_array_equal(result.status8[4:], motion)
    assert read_ids == [10, 11, 12]
    assert result.status_validity == 'NOT_VERIFIED'


def test_pair_identity_is_bound_to_frozen_index(tmp_path):
    import sqlite3
    from tools.validation.b2d.status_upstream_audit import sha256, verify_panel_pairs
    path = tmp_path / 'pairs.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE pairs(sample_id TEXT, feature_sha TEXT)')
        db.execute('INSERT INTO pairs VALUES (?,?)', ('s1', 'a'*64))
    manifest = {'pair_index': str(path), 'index_sha256': sha256(path)}
    sample = {'sample_id': 's1', 'row': {'sample_id': 's1', 'feature_sha': 'a'*64}}
    verify_panel_pairs([sample], manifest)
    sample['row']['feature_sha'] = 'b'*64
    with pytest.raises(ValueError, match='panel_pair_identity_mismatch'):
        verify_panel_pairs([sample], manifest)


def test_trial_motion_rejects_missing_and_teleported_history_without_zeroing_creep():
    from tools.validation.b2d.status_upstream_audit import trailing_motion4
    annotations = [{'bounding_boxes': [{'class': 'ego_vehicle', 'type_id': 'vehicle.lincoln.mkz_2020',
                                       'rotation': [0, 0, 0]}]} for _ in range(3)]
    poses = [[0., 0., 0.], [.001, 0., 0.], [.002, 0., 0.]]
    motion = trailing_motion4([10, 11, 12], poses, annotations)
    assert motion[0] == pytest.approx(.01)
    with pytest.raises(ValueError, match='nonconsecutive'):
        trailing_motion4([9, 11, 12], poses, annotations)
    annotations[0]['teleport_flag'] = True
    with pytest.raises(ValueError, match='motion_history_pose_invalid'):
        trailing_motion4([10, 11, 12], poses, annotations)
