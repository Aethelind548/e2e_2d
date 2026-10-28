"""M2/M3: test real parsing/storage; distinguish missing algorithms from passes."""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from agent.common.errors import NavigationError, PendingImplementationError, SensorError
from agent.common.types import EgoState, NavigationState, SensorObservation, Timestamp
from agent.config.settings import NavigationConfig
from agent.navigation.status_builder import COMMAND_ORDER, validate_command4
from agent.navigation.route_manager import RouteManager
from agent.sensors.ego_state import PendingEgoStateEstimator
from agent.sensors.observation_buffer import ObservationBuffer
from agent.sensors.sensor_manager import SensorManager


def test_observation_payload_and_frame_are_preserved(sensor_config, raw_observation):
    manager = SensorManager(sensor_config)
    obs = manager.collect(raw_observation, Timestamp(1.0))
    for name, (frame, payload) in raw_observation.items():
        assert obs.samples[name].frame_id == frame
        assert obs.samples[name].payload is payload
        assert obs.samples[name].timestamp is None
    assert obs.timestamp.seconds == 1.0
    assert manager.buffer.latest is obs


@pytest.mark.parametrize('bad', [{}, {'rgb': (1, None)}, {'rgb': ('1', object())},
                              {'rgb': (-1, object())}, {'rgb': (True, object())},
                              {'rgb': (1,)}, {'rgb': object()}])
def test_missing_or_malformed_envelope_rejected(sensor_config, bad):
    with pytest.raises(SensorError):
        SensorManager(replace(sensor_config, required_ids=())).collect(bad, Timestamp(1.0))


def test_required_sensor_missing_rejected(sensor_config, raw_observation):
    raw_observation.pop('lidar')
    with pytest.raises(SensorError, match='lidar'):
        SensorManager(sensor_config).collect(raw_observation, Timestamp(1.0))


@pytest.mark.parametrize('sensor,payload', [
    ('rgb', np.zeros((6, 8, 3), dtype=np.uint8)),
    ('rgb', np.zeros((7, 8, 4), dtype=np.uint8)),
    ('rgb', np.zeros((6, 8, 4), dtype=np.float32)),
    ('lidar', np.zeros((3, 3), dtype=np.float32)),
    ('lidar', np.full((3, 4), np.nan, dtype=np.float32)),
    ('imu', np.zeros(6)), ('imu', np.full(7, np.inf)),
    ('gnss', np.zeros(2)), ('gnss', np.array([np.nan, 0., 0.])),
    ('speed', {'speed': np.nan}), ('speed', {}), ('speed', {'speed': True})])
def test_known_payload_shape_and_finiteness_rejected(sensor_config, raw_observation, sensor, payload):
    raw_observation[sensor] = (100, payload)
    manager = SensorManager(sensor_config)
    with pytest.raises(SensorError):
        manager.collect(raw_observation, Timestamp(1.0))
    assert manager.buffer.history() == ()


@pytest.mark.parametrize('time', [1.0, 0.5, math.nan, math.inf])
def test_stale_and_nonfinite_observation_clock_rejected(sensor_config, raw_observation, time):
    manager = SensorManager(sensor_config)
    manager.collect(raw_observation, Timestamp(1.0))
    with pytest.raises(SensorError):
        manager.collect(raw_observation, Timestamp(time))
    assert len(manager.buffer.history()) == 1


def test_bounded_history_and_reset(sensor_config, raw_observation):
    manager = SensorManager(sensor_config)
    for i in range(5):
        manager.collect(raw_observation, Timestamp(1.0 + i * 0.1))
    assert [obs.timestamp.seconds for obs in manager.buffer.history()] == pytest.approx([1.2, 1.3, 1.4])
    manager.reset()
    assert manager.buffer.history() == ()
    assert manager.buffer.latest is None


def test_clock_source_change_is_not_resampled(raw_observation, sensor_config):
    manager = SensorManager(sensor_config)
    manager.collect(raw_observation, Timestamp(1.0))
    with pytest.raises(SensorError, match='clock'):
        manager.collect(raw_observation, Timestamp(2.0, 'other_clock'))


def test_unconfigured_alignment_and_estimator_fail_explicitly(sensor_config, raw_observation):
    manager = SensorManager(sensor_config)
    obs = manager.collect(raw_observation, Timestamp(1.0))
    with pytest.raises(PendingImplementationError, match='alignment'):
        manager.validate_alignment(obs)
    with pytest.raises(PendingImplementationError, match='estimation'):
        PendingEgoStateEstimator().estimate(obs, manager.buffer.history())


def test_route_storage_reset_and_missing_route(world_route):
    manager = RouteManager(NavigationConfig())
    with pytest.raises(NavigationError, match='Global plan'):
        manager.update(EgoState(Timestamp(1.0)), Timestamp(1.0))
    gps, world = world_route()
    manager.set_global_plan(gps, world)
    assert manager.route.gps_plan == tuple(gps)
    assert manager.route.world_plan == tuple(world)
    with pytest.raises(NavigationError, match='rear pose'):
        manager.update(EgoState(Timestamp(1.0)), Timestamp(1.0))
    manager.reset()
    assert manager.route is None and manager.builder.route['point_count'] == 0


@pytest.mark.parametrize('command', [(0, 0, 0, 0), (1, 1, 0, 0), (1, 0, 0),
                                  (math.nan, 0, 0, 0), (0.5, 0.5, 0, 0)])
def test_invalid_command4_rejected(command):
    with pytest.raises(NavigationError):
        validate_command4(command)


@pytest.mark.parametrize('index', range(4))
def test_confirmed_model_command_order_and_valid_one_hot(index):
    assert COMMAND_ORDER == ('Left', 'Straight', 'Right', 'Unknown')
    validate_command4(tuple(float(i == index) for i in range(4)))


def test_retired_near_policy_cannot_be_selected():
    with pytest.raises(NavigationError, match='b2d-route-preview-20m-v1'):
        RouteManager(NavigationConfig(status_policy='b2d-near-bearing-v1'))


@pytest.mark.parametrize('right_y,expected', [(-5., (1, 0, 0, 0)), (0., (0, 1, 0, 0)), (5., (0, 0, 1, 0))])
def test_offline_near_bearing_builder_remains_for_frozen_reproduction(right_y, expected):
    """Offline V1 feature builder semantics stay readable; the online Agent no longer uses them."""
    from tools.b2d_adapter.coordinate_transform import ActorPose
    from tools.b2d_adapter.native_navigation import near_route_command
    from tools.b2d_adapter.v1_builders import v1_native_config
    raw = SimpleNamespace(navigation={'x_command_near': 10., 'y_command_near': right_y, 'command_near': 999},
                          actor_pose=ActorPose(0., 0., 0.), ref=SimpleNamespace(frame_id=100),
                          provenance={'source_files': {'anno': 'synthetic-current-only'}})
    command, provenance = near_route_command(raw, v1_native_config())
    assert tuple(command) == expected
    assert provenance['future_gt_used'] is False
    assert provenance['navsim_20m_equivalent'] is False
