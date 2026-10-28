"""Real M2/M3 functionality with CPU GNSS/IMU and independently generated poses."""

from dataclasses import replace
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent.bench2drive_agent import Bench2DriveAgent
from agent.common.errors import ConfigurationError, NavigationError, PendingImplementationError, SensorError, WarmupRequiredError
from agent.common.types import CoordinateFrame, EgoState, NavigationState, Pose2D, RuntimeStatus, SimulationTick, Timestamp
from agent.config import load_config
from agent.config.settings import NavigationConfig
from agent.navigation.route_manager import RouteManager
from agent.sensors.ego_state import CausalGNSSStateEstimator
from agent.sensors.georeference import CarlaGeoReference
from agent.sensors.sensor_manager import SensorManager


def manager_from(config):
    return SensorManager(config.sensors, CausalGNSSStateEstimator(config.ego_state, config.sensors))


def ready(manager, stream, path=None):
    for index in range(5):
        raw, tick = stream(index, path)
        manager.set_simulation_tick(tick)
        if index < 4:
            with pytest.raises(WarmupRequiredError):
                manager.update(raw, tick.timestamp)
        else:
            return manager.update(raw, tick.timestamp)


def ego_at(x=0., y=0., yaw=0., t=1.0):
    actor = Pose2D(x+1.388632877*math.cos(yaw), y+1.388632877*math.sin(yaw), yaw,
                   CoordinateFrame.CARLA_WORLD, 'carla_actor_origin')
    rear = Pose2D(x, y, yaw, CoordinateFrame.CARLA_WORLD, 'lincoln-10hz-rear-v1')
    return EgoState(Timestamp(t), rear, dynamics_frame=CoordinateFrame.MODEL_REAR,
                    provenance={'geometry_policy': 'lincoln-10hz-rear-v1'}, actor_pose=actor,
                    navigation_world_xy_m=(x-.011367123*math.cos(yaw), y-.011367123*math.sin(yaw)))


def advance(manager, poses, t0=1.0, dt=0.1):
    """Feed three 10 Hz rear poses; return the last NavigationState."""
    state = None
    for index, (x, y) in enumerate(poses):
        stamp = Timestamp(t0 + index*dt)
        state = manager.update(ego_at(x, y, 0., stamp.seconds), stamp)
    return state


def test_current_frame_anchor_and_real_history_warmup(m2m3_config, config_file, sensor_stream):
    manager = manager_from(load_config(config_file(m2m3_config)))
    obs, ego = ready(manager, sensor_stream)
    assert obs.timestamp.seconds == pytest.approx(10.2)
    assert all(s.frame_id == 1004 and s.timestamp == obs.timestamp for s in obs.samples.values())
    assert ego.pose.reference == 'lincoln-10hz-rear-v1'
    assert (ego.pose.x_m, ego.pose.y_m) == pytest.approx((.4, 0.), abs=1e-7)
    assert ego.actor_pose.x_m == pytest.approx(.4+1.388632877, abs=1e-7)
    assert ego.velocity_xy_mps == pytest.approx((2., 0.), abs=1e-6)
    assert ego.acceleration_xy_mps2 == pytest.approx((0., 0.), abs=2e-6)
    assert ego.provenance['source_frame_ids'] == [1000, 1002, 1004]
    assert ego.provenance['velocity_lag_s'] == .05
    assert ego.provenance['acceleration_lag_s'] == .1
    assert ego.provenance['imu_acceleration_used'] is False
    manager.reset()
    assert manager.buffer.history() == () and manager._last_tick is None
    raw, tick = sensor_stream(0, frame_base=3, time_base=.05)
    manager.set_simulation_tick(tick)
    with pytest.raises(WarmupRequiredError):
        manager.update(raw, tick.timestamp)


def test_acceleration_and_lateral_velocity_are_position_based(m2m3_config, config_file, sensor_stream):
    manager = manager_from(load_config(config_file(m2m3_config)))
    _, ego = ready(manager, sensor_stream, lambda t: (2*t+1.5*t*t, .4*t, 0.))
    assert ego.velocity_xy_mps == pytest.approx((2.45, -.4), abs=2e-6)
    assert ego.acceleration_xy_mps2 == pytest.approx((3., 0.), abs=3e-6)
    assert ego.speed_mps == 2.  # Measured Actor scalar was not substituted for vx.


@pytest.mark.parametrize('mode', ['straight', 'left', 'right', 'decelerate', 'stop'])
def test_state_matches_existing_candidate_causal_formula(mode, m2m3_config, config_file, sensor_stream):
    from tools.b2d_adapter.coordinate_transform import ActorPose
    from tools.b2d_adapter.object_adapter import candidate_causal_state
    from tools.b2d_adapter.v1_builders import v1_native_config
    def path(t):
        if mode in ('left', 'right'):
            sign = -1 if mode == 'left' else 1
            yaw = sign*.3*t
            return 10*math.sin(.3*t), sign*10*(1-math.cos(.3*t)), yaw
        return (0. if mode == 'stop' else 2*t-.5*t*t if mode == 'decelerate' else 2*t), 0., 0.
    manager = manager_from(load_config(config_file(m2m3_config)))
    _, ego = ready(manager, sensor_stream, path)
    def read(ref):
        x, y, yaw = path(ref.frame_id*.1)
        actor = ActorPose(x+1.388632877*math.cos(yaw), y+1.388632877*math.sin(yaw), yaw)
        return SimpleNamespace(ref=ref, actor_pose=actor, ego_box={'type_id': 'vehicle.lincoln.mkz_2020'},
                               provenance={'source_files': {'anno': 'synthetic-formula-oracle'}})
    reader = SimpleNamespace(index=SimpleNamespace(ref=lambda i: SimpleNamespace(frame_id=i)), read=read)
    expected = candidate_causal_state(reader, SimpleNamespace(frame_id=2), v1_native_config().geometry)
    np.testing.assert_allclose(ego.velocity_xy_mps, expected.velocity, atol=2e-6)
    np.testing.assert_allclose(ego.acceleration_xy_mps2, expected.acceleration, atol=3e-6)


@pytest.mark.parametrize('problem', ['stale_sensor', 'future_sensor', 'wrong_time', 'frame_jump', 'variable_dt', 'repeated_frame', 'missing_tick'])
def test_bad_clock_or_expired_observation_does_not_enter_history(problem, m2m3_config, config_file, sensor_stream):
    manager = manager_from(load_config(config_file(m2m3_config)))
    ready(manager, sensor_stream)
    previous = manager.buffer.history()
    raw, tick = sensor_stream(5)
    if problem in ('stale_sensor', 'future_sensor'):
        raw['rgb'] = (tick.frame_id + (-1 if problem == 'stale_sensor' else 1), raw['rgb'][1])
    elif problem == 'wrong_time':
        tick = replace(tick, timestamp=Timestamp(99.))
    elif problem == 'frame_jump':
        tick = replace(tick, frame_id=2000)
    elif problem == 'variable_dt':
        tick = replace(tick, delta_seconds=.1)
    elif problem == 'repeated_frame':
        tick = replace(tick, frame_id=1004)
    manager.set_simulation_tick(None if problem == 'missing_tick' else tick)
    with pytest.raises(SensorError):
        manager.update(raw, Timestamp(10.25))
    assert manager.buffer.history() == previous


@pytest.mark.parametrize('delta,capacity', [(.03, 9), (.05, 3), (.2, 3)])
def test_unsupported_sampling_and_insufficient_capacity_rejected(delta, capacity, m2m3_config, config_file, sensor_stream):
    m2m3_config['sensors']['buffer_size'] = capacity
    manager = manager_from(load_config(config_file(m2m3_config)))
    raw, tick = sensor_stream(0, delta=delta)
    manager.set_simulation_tick(tick)
    with pytest.raises((SensorError, ConfigurationError)):
        manager.update(raw, tick.timestamp)
    assert not manager.buffer.history()


def test_missing_required_past_sample_is_not_fabricated(m2m3_config, config_file, sensor_stream):
    manager = manager_from(load_config(config_file(m2m3_config)))
    for index in (0, 1, 3, 4):
        raw, tick = sensor_stream(index)
        manager.set_simulation_tick(tick)
        if index < 4:
            with pytest.raises(WarmupRequiredError):
                manager.update(raw, tick.timestamp)
        else:
            with pytest.raises(SensorError, match='sample is missing'):
                manager.update(raw, tick.timestamp)


@pytest.mark.parametrize('field,value', [('lat_ref_deg', None), ('lon_ref_deg', math.nan),
    ('vehicle_type', 'vehicle.other'), ('planar_assumption', False), ('geometry_policy', 'unknown')])
def test_missing_state_basis_rejected(field, value, m2m3_config, config_file):
    m2m3_config['ego_state'][field] = value
    with pytest.raises(ConfigurationError):
        manager_from(load_config(config_file(m2m3_config)))


def test_nonzero_mount_rotation_rejected(m2m3_config, config_file):
    for spec in m2m3_config['sensors']['specifications']:
        if spec['id'] == 'imu':
            spec['yaw'] = 5.
    with pytest.raises(ConfigurationError, match='zero-rotation'):
        manager_from(load_config(config_file(m2m3_config)))


def test_invalid_gnss_domain_and_motion_discontinuity_rejected(m2m3_config, config_file, sensor_stream):
    manager = manager_from(load_config(config_file(m2m3_config)))
    raw, tick = sensor_stream(0)
    raw['gnss'][1][0] = 90.
    manager.set_simulation_tick(tick)
    with pytest.raises(SensorError, match='Mercator'):
        manager.update(raw, tick.timestamp)
    manager.reset()
    for index in range(5):
        raw, tick = sensor_stream(index, lambda t: (1000*t, 0., 0.))
        manager.set_simulation_tick(tick)
        with pytest.raises(WarmupRequiredError if index < 4 else SensorError):
            manager.update(raw, tick.timestamp)


@pytest.mark.parametrize('lateral,expected', [(-5., (1., 0., 0., 0.)), (0., (0., 1., 0., 0.)), (5., (0., 0., 1., 0.))])
def test_route_preview_command_from_dense_plan(lateral, expected, world_route):
    manager = RouteManager(NavigationConfig())
    gps, world = world_route(((0., 0.), (10., lateral), (20., lateral), (30., lateral), (40., lateral)), command=999)
    manager.set_global_plan(gps, world)
    state = advance(manager, [(0., 0.)]*3)
    assert state.status_validity == 'VALID'
    assert state.command4 == expected
    assert state.near_index is None and state.raw_command is None
    assert state.provenance['sparse_plan_used'] is False
    assert state.provenance['road_option_available'] is False
    assert state.provenance['route_source'] == 'dense_set_global_plan_world_coords'
    assert len(state.status8) == 8


@pytest.mark.parametrize('yaw', [0., .7, -1.])
@pytest.mark.parametrize('lateral', [-4., 0., 4.])
def test_online_command_exactly_matches_shared_calculator(yaw, lateral, world_route):
    from tools.b2d_adapter.route_preview_contract import preview_command4
    points = ((0., 0.), (10., lateral), (20., lateral), (30., lateral))
    manager = RouteManager(NavigationConfig())
    manager.set_global_plan(*world_route(points))
    rear = (3*math.cos(yaw) - lateral*math.sin(yaw),
            3*math.sin(yaw) + lateral*math.cos(yaw))
    expected = preview_command4([tuple(p) for p in points], rear, yaw, max_projection_m=5.0)
    state = None
    for index in range(3):
        stamp = Timestamp(1. + index*0.1)
        state = manager.update(ego_at(rear[0], rear[1], yaw, stamp.seconds), stamp)
    assert state.command4 == tuple(expected['command4'])
    assert state.status_validity == 'VALID'


def test_route_switch_is_atomic_and_bumps_generation(world_route):
    manager = RouteManager(NavigationConfig())
    manager.set_global_plan(*world_route(((0., 0.), (10., 0.), (20., 0.), (30., 0.), (40., 0.))))
    assert advance(manager, [(1., 0.)]*3).status_validity == 'VALID'
    old_route, generation = manager.route, manager.generation
    with pytest.raises(NavigationError):
        manager.set_global_plan(*world_route(((0., 0.), (math.nan, 2.))))
    assert manager.route is old_route and manager.generation == generation
    manager.set_global_plan(*world_route(((0., 0.), (10., -5.), (20., -5.), (30., -5.), (40., -5.))))
    assert manager.generation == generation + 1
    assert manager.builder.route['point_count'] == 5
    assert advance(manager, [(0., 0.)]*3, t0=1.3).command4 == (1., 0., 0., 0.)


def test_duplicate_consecutive_points_are_removed_once(world_route):
    manager = RouteManager(NavigationConfig())
    manager.set_global_plan(*world_route(((0., 0.), (0., 0.), (10., 0.), (20., 0.))))
    assert manager.builder.route['point_count'] == 3
    assert manager.builder.route['duplicates_removed'] == 1


@pytest.mark.parametrize('points', [((0., 0.), (0., 0.)), ((0., 0.), (math.inf, 0.))])
def test_invalid_route_geometry_rejected(points, world_route):
    manager = RouteManager(NavigationConfig())
    with pytest.raises(NavigationError):
        manager.set_global_plan(*world_route(points))


@pytest.mark.parametrize('points,reason', [
    (((0., 0.), (10., 0.)), 'route_end_before_preview'),
    (((0., 20.), (60., 20.)), 'projection_distance_exceeded'),
])
def test_invalid_preview_is_explicit_not_silent_unknown(points, reason, world_route):
    manager = RouteManager(NavigationConfig())
    manager.set_global_plan(*world_route(points))
    state = advance(manager, [(0., 0.)]*3)
    assert state.status_validity == 'INVALID'
    assert state.status8 is None
    assert state.command4 is None
    assert state.status_reason == reason
    assert state.provenance['status8_validity'] == 'INVALID'


def test_motion_history_insufficient_is_explicit_then_valid(world_route):
    manager = RouteManager(NavigationConfig())
    manager.set_global_plan(*world_route(((0., 0.), (20., 0.), (40., 0.), (60., 0.))))
    first = manager.update(ego_at(0., 0., 0., 1.0), Timestamp(1.0))
    assert first.status_validity == 'INVALID'
    assert first.status_reason == 'MOTION_HISTORY_INSUFFICIENT'
    assert first.command4 == (0., 1., 0., 0.)
    state = advance(manager, [(0., 0.), (1., 0.), (2., 0.)], t0=1.1)
    assert state.status_validity == 'VALID'


def test_online_motion4_matches_a0_definition(world_route):
    from tools.b2d_adapter.experimental_causal_state import A0, causal_state
    manager = RouteManager(NavigationConfig())
    manager.set_global_plan(*world_route(((0., 0.), (40., 0.), (80., 0.))))
    poses = [(0., 0.), (0.2, 0.), (0.45, 0.)]
    state = advance(manager, poses)
    expected = causal_state([(x, y, 0.) for x, y in poses], method=A0, dt=0.1)
    assert state.motion4 == pytest.approx(tuple(expected.velocity) + tuple(expected.acceleration),
                                          abs=1e-12)
    assert state.status8[4:] == pytest.approx(state.motion4, abs=1e-12)
    assert sum(state.status8[:4]) == 1.0


def test_motion_history_resets_after_a_large_gap(world_route):
    manager = RouteManager(NavigationConfig())
    manager.set_global_plan(*world_route(((0., 0.), (40., 0.), (80., 0.))))
    advance(manager, [(0., 0.), (1., 0.), (2., 0.)])
    state = manager.update(ego_at(3., 0., 0., 5.0), Timestamp(5.0))
    assert state.status_validity == 'INVALID'
    assert 'MOTION_HISTORY_INSUFFICIENT' in state.status_reason


@pytest.mark.parametrize('scenario,lateral,expected', [('straight', 0., (0., 1., 0., 0.)),
    ('left', -5., (1., 0., 0., 0.)), ('right', 5., (0., 0., 1., 0.)),
    ('decelerate', 0., (0., 1., 0., 0.)), ('stop', 0., (0., 1., 0., 0.))])
def test_real_chain_passes_m2_m3_and_reaches_real_m4(scenario, lateral, expected,
        m2m3_config, config_file, world_route, sensor_stream, fake_carla):
    agent = Bench2DriveAgent()
    agent.set_global_plan(*world_route(((0., 0.), (10., lateral), (20., lateral), (30., lateral), (40., lateral))))
    agent.setup(config_file(m2m3_config))
    path = (lambda t: (0., 0., 0.)) if scenario == 'stop' else (lambda t: (2*t-.5*t*t, 0., 0.)) if scenario == 'decelerate' else None
    try:
        for index in range(5):
            raw, tick = sensor_stream(index, path)
            agent.set_simulation_tick(tick)
            if index < 4:
                with pytest.raises(WarmupRequiredError):
                    agent.run_step(raw, tick.timestamp.seconds)
                assert agent.supervisor.state.status == RuntimeStatus.WARMING_UP
                assert agent.supervisor.state.last_error_type is None
            else:
                with pytest.raises(PendingImplementationError, match='Online sensor-to-V1'):
                    agent.run_step(raw, tick.timestamp.seconds)
        state = agent.supervisor.state
        assert state.module_status == {'M2.sensors': 'completed', 'M3.navigation': 'completed', 'M4.features': 'failed'}
        assert state.active_module == 'M4.features' and state.warmup_steps == 4
        assert agent.last_navigation_state.command4 == expected
        assert agent.last_ego_state.timestamp == agent.last_navigation_state.timestamp
        assert agent.last_ego_state.timestamp == agent.sensor_manager.buffer.latest.timestamp
        assert state.completed_steps == 0 and fake_carla.created == []
        assert agent._previous_timestamp.seconds == pytest.approx(10.2)
    finally:
        agent.destroy()
    assert agent.last_ego_state is None and agent.last_navigation_state is None


def test_agent_route_switch_during_warmup_preserves_physical_history(m2m3_config, config_file, world_route, sensor_stream):
    agent = Bench2DriveAgent()
    agent.set_global_plan(*world_route())
    agent.setup(config_file(m2m3_config))
    try:
        for index in range(4):
            raw, tick = sensor_stream(index)
            agent.set_simulation_tick(tick)
            with pytest.raises(WarmupRequiredError):
                agent.run_step(raw, tick.timestamp.seconds)
        agent.set_global_plan(*world_route(((0., 0.), (10., -5.), (20., -5.))))
        assert len(agent.sensor_manager.buffer.history()) == 4
        raw, tick = sensor_stream(4)
        agent.set_simulation_tick(tick)
        with pytest.raises(PendingImplementationError, match='Online sensor-to-V1'):
            agent.run_step(raw, tick.timestamp.seconds)
        assert agent.last_navigation_state.route_generation == 2
        assert agent.last_navigation_state.command4 == (1., 0., 0., 0.)
    finally:
        agent.destroy()


def test_ten_hz_ticks_need_three_actual_observations(m2m3_config, config_file, sensor_stream):
    m2m3_config['sensors']['buffer_size'] = 3
    manager = manager_from(load_config(config_file(m2m3_config)))
    for index in range(3):
        raw, tick = sensor_stream(index, delta=.1)
        manager.set_simulation_tick(tick)
        if index < 2:
            with pytest.raises(WarmupRequiredError):
                manager.update(raw, tick.timestamp)
        else:
            _, ego = manager.update(raw, tick.timestamp)
            assert ego.provenance['source_frame_ids'] == [1000, 1001, 1002]
            assert ego.velocity_xy_mps == pytest.approx((2., 0.), abs=1e-6)


def test_real_m2_m3_30_ready_updates_with_no_padding(m2m3_config, config_file, sensor_stream, world_route):
    config = load_config(config_file(m2m3_config))
    manager = manager_from(config)
    route = RouteManager(config.navigation)
    route.set_global_plan(*world_route())
    ready_updates = 0
    for index in range(34):
        raw, tick = sensor_stream(index)
        manager.set_simulation_tick(tick)
        if index < 4:
            with pytest.raises(WarmupRequiredError):
                manager.update(raw, tick.timestamp)
            continue
        obs, ego = manager.update(raw, tick.timestamp)
        navigation = route.update(ego, obs.timestamp)
        assert navigation.command4 == (0., 1., 0., 0.)
        assert ego.provenance['source_frame_ids'] == [1000+index-4, 1000+index-2, 1000+index]
        ready_updates += 1
    assert ready_updates == 30
    assert len(manager.buffer.history()) == 5


def test_bad_state_or_navigation_basis_is_explicit(world_route):
    route = RouteManager(NavigationConfig())
    route.set_global_plan(*world_route())
    with pytest.raises(NavigationError, match='rear pose'):
        route.update(EgoState(Timestamp(1.)), Timestamp(1.))
    with pytest.raises(NavigationError, match='geometry policy'):
        route.update(replace(ego_at(), provenance={}), Timestamp(1.))
    with pytest.raises(NavigationError, match='b2d-route-preview-20m-v1'):
        RouteManager(NavigationConfig(status_policy='raw-road-option'))


def test_live_wrapper_uses_current_tick_without_actor_motion(monkeypatch, config_file, m2m3_config,
                                                           world_route, sensor_stream):
    """Only external modules are doubles; wrapper and all M2/M3 logic are real."""
    import importlib
    import sys
    from enum import Enum
    from types import ModuleType
    class Track(Enum):
        SENSORS = 'SENSORS'
    class Base:
        def __init__(self, host, port, debug=False):
            self.hero_actor = SimpleNamespace(type_id='vehicle.lincoln.mkz_2020')
        def set_global_plan(self, gps, world):
            pass
    now = {}
    class GameTime:
        @staticmethod
        def get_frame():
            return now['tick'].frame_id
        @staticmethod
        def get_time():
            return now['tick'].timestamp.seconds
    class Provider:
        @staticmethod
        def get_world():
            tick = now['tick']
            return SimpleNamespace(get_snapshot=lambda: SimpleNamespace(
                frame=tick.frame_id, timestamp=SimpleNamespace(delta_seconds=tick.delta_seconds)))
    for name in ('leaderboard', 'leaderboard.autoagents', 'srunner', 'srunner.scenariomanager'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name, attributes in {
        'leaderboard.autoagents.autonomous_agent': {'AutonomousAgent': Base, 'Track': Track},
        'srunner.scenariomanager.timer': {'GameTime': GameTime},
        'srunner.scenariomanager.carla_data_provider': {'CarlaDataProvider': Provider}}.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, 'agent.environment_interface', raising=False)
    core = importlib.import_module('agent.bench2drive_agent')
    agent = None
    try:
        cls = getattr(core, core.get_entry_point())
        agent = cls('localhost', 2000)
        agent.set_global_plan(*world_route())
        agent.setup(config_file(m2m3_config))
        for index in range(5):
            raw, tick = sensor_stream(index)
            now['tick'] = tick
            with pytest.raises(WarmupRequiredError if index < 4 else PendingImplementationError):
                agent.run_step(raw, tick.timestamp.seconds)
        assert agent.supervisor.state.active_module == 'M4.features'
        assert agent.last_ego_state.provenance['source_frame_ids'] == [1000, 1002, 1004]
    finally:
        if agent is not None:
            agent.destroy()
        sys.modules.pop('agent.environment_interface', None)
        core.__dict__.pop('DiffusionDriveLeaderboardAgent', None)


def test_wrong_georeference_is_rejected_against_real_route_pairs(m2m3_config, config_file, world_route):
    agent = Bench2DriveAgent()
    agent.set_global_plan(*world_route())
    m2m3_config['ego_state']['lon_ref_deg'] = 3.
    with pytest.raises(ConfigurationError, match='GPS/world route pairs'):
        agent.setup(config_file(m2m3_config))
    assert agent.supervisor.state.status == RuntimeStatus.FAILED
    agent.destroy()


def test_compass_outside_sensor_domain_is_rejected(m2m3_config, config_file, sensor_stream):
    manager = manager_from(load_config(config_file(m2m3_config)))
    raw, tick = sensor_stream(0)
    raw['imu'][1][6] = 7.
    manager.set_simulation_tick(tick)
    with pytest.raises(SensorError, match='compass'):
        manager.update(raw, tick.timestamp)
