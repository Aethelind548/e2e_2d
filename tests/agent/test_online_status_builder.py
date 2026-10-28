"""CPU tests for the single online Status8 builder (no model, no CARLA)."""
import math
from types import SimpleNamespace

import pytest

from agent.adapters.online_feature_adapter import OnlineFeatureAdapter
from agent.common.errors import FeatureError, NavigationError
from agent.common.types import CoordinateFrame, EgoState, Pose2D, Timestamp
from agent.config.settings import FeatureConfig, NavigationConfig
from agent.navigation.status_builder import STATUS_POLICY, OnlineStatusBuilder


def world_plan(points, command=None):
    option = command if command is not None else SimpleNamespace(value=999)
    return [(SimpleNamespace(location=SimpleNamespace(x=float(x), y=float(y), z=0.0),
                             rotation=SimpleNamespace(yaw=0.0)), option) for x, y in points]


def gps_plan(count):
    return [({'lat': 0.0, 'lon': 0.0}, None) for _ in range(count)]


def rear_ego(x, y, yaw, t):
    rear = Pose2D(x, y, yaw, CoordinateFrame.CARLA_WORLD, 'lincoln-10hz-rear-v1')
    return EgoState(Timestamp(t), rear, dynamics_frame=CoordinateFrame.MODEL_REAR,
                    provenance={'geometry_policy': 'lincoln-10hz-rear-v1'})


def feed(builder, x, y, yaw=0.0, t0=1.0, count=3, dt=0.1):
    status = None
    for index in range(count):
        stamp = Timestamp(t0 + index*dt)
        status = builder.update(rear_ego(x, y, yaw, stamp.seconds), stamp)
    return status


def test_dense_plan_is_retained_once_and_hashed():
    plan = [(float(i), 0.0) for i in range(100)]
    builder = OnlineStatusBuilder(NavigationConfig())
    builder.set_global_plan(world_plan(plan))
    route = builder.route
    assert route['point_count'] == 100
    assert route['arc_length_m'] == pytest.approx(99.0)
    assert route['generation'] == 1
    assert len(route['sha256']) == 64
    before = route['sha256']
    # Re-issuing the same plan is a new generation but the same content identity.
    builder.set_global_plan(world_plan(plan))
    assert builder.route['sha256'] == before and builder.route['generation'] == 2


def test_only_consecutive_duplicates_are_removed():
    plan = [(0., 0.), (0., 0.), (10., 0.), (10., 0.), (10., 0.), (20., 0.)]
    builder = OnlineStatusBuilder(NavigationConfig())
    builder.set_global_plan(world_plan(plan))
    assert builder.route['points'] == ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))
    assert builder.route['duplicates_removed'] == 3


def test_status8_is_one_hot_plus_a0_motion():
    builder = OnlineStatusBuilder(NavigationConfig())
    builder.set_global_plan(world_plan([(0., 0.), (20., 0.), (40., 0.), (60., 0.)]))
    poses = [(0., 0.), (0.2, 0.), (0.45, 0.)]
    for index, (x, y) in enumerate(poses):
        stamp = Timestamp(1.0 + index*0.1)
        status = builder.update(rear_ego(x, y, 0.0, stamp.seconds), stamp)
    assert status.validity == 'VALID'
    assert len(status.status8) == 8
    assert sum(status.status8[:4]) == 1.0
    assert status.status8[:4] == (0.0, 1.0, 0.0, 0.0)
    assert status.status8[4:] == pytest.approx((2.5, 0.0, 5.0, 0.0), abs=1e-9)


def test_invalid_preview_never_produces_unknown_status():
    builder = OnlineStatusBuilder(NavigationConfig())
    builder.set_global_plan(world_plan([(0., 0.), (10., 0.)]))
    status = feed(builder, 0., 0.)
    assert status.validity == 'INVALID'
    assert status.status8 is None
    assert status.command4 is None
    assert status.reason == 'route_end_before_preview'


def test_uturn_is_explicitly_unsupported():
    builder = OnlineStatusBuilder(NavigationConfig())
    builder.set_global_plan(world_plan([(0., 0.), (20., 0.), (0., 0.)]))
    status = feed(builder, 0., 0.)
    assert status.validity == 'INVALID'
    assert status.status8 is None
    assert 'uturn' in status.reason


def test_motion_history_decimates_20hz_ticks_to_10hz():
    builder = OnlineStatusBuilder(NavigationConfig())
    builder.set_global_plan(world_plan([(0., 0.), (40., 0.), (80., 0.)]))
    validities = []
    for index in range(5):
        stamp = Timestamp(1.0 + index*0.05)
        status = builder.update(rear_ego(0.05*index, 0.0, 0.0, stamp.seconds), stamp)
        validities.append(status.validity)
    assert validities[:3] == ['INVALID', 'INVALID', 'INVALID']
    assert validities[4] == 'VALID'
    assert builder.motion.history_size() == 3


def test_irregular_motion_interval_is_rejected():
    builder = OnlineStatusBuilder(NavigationConfig())
    builder.set_global_plan(world_plan([(0., 0.), (40., 0.), (80., 0.)]))
    for index, t in enumerate((1.0, 1.1, 1.35)):
        status = builder.update(rear_ego(0.1*index, 0.0, 0.0, t), Timestamp(t))
    assert status.validity == 'INVALID'
    assert status.motion_reason == 'MOTION_SAMPLING_INTERVAL'


def test_status_is_deterministic_for_identical_input():
    def run():
        builder = OnlineStatusBuilder(NavigationConfig())
        builder.set_global_plan(world_plan([(0., 0.), (20., -3.), (40., -3.), (60., -3.)]))
        return [builder.update(rear_ego(0.1*i, 0.0, 0.0, 1.0 + 0.1*i),
                               Timestamp(1.0 + 0.1*i)).status8 for i in range(3)]
    first, second = run(), run()
    assert first == second
    assert first[-1] is not None


def test_adapter_builds_float32_status_and_rejects_invalid():
    import torch
    builder = OnlineStatusBuilder(NavigationConfig())
    builder.set_global_plan(world_plan([(0., 0.), (20., 0.), (40., 0.), (60., 0.)]))
    status = feed(builder, 0., 0.)
    adapter = OnlineFeatureAdapter(FeatureConfig())
    from agent.common.types import NavigationState
    nav = NavigationState(status.timestamp, command4=status.command4, policy=STATUS_POLICY,
                          status8=status.status8, status_validity=status.validity,
                          status_reason=status.reason)
    tensor = adapter.build_status(rear_ego(0., 0., 0., 1.2), nav)
    assert tensor.dtype == torch.float32 and tuple(tensor.shape) == (8,)
    invalid = NavigationState(status.timestamp, policy=STATUS_POLICY, status8=None,
                              status_validity='INVALID', status_reason='route_end_before_preview')
    with pytest.raises(FeatureError, match='not valid'):
        adapter.build_status(rear_ego(0., 0., 0., 1.2), invalid)


def test_wrong_policy_configuration_is_rejected():
    with pytest.raises(NavigationError, match='b2d-route-preview-20m-v1'):
        OnlineStatusBuilder(NavigationConfig(status_policy='b2d-near-bearing-v1'))
