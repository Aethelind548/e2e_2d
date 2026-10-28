"""Agent-level integration of the single online Status interface (no model)."""
import math

import pytest

from agent.bench2drive_agent import Bench2DriveAgent
from agent.common.errors import NavigationError
from agent.common.types import CoordinateFrame, EgoState, Pose2D, RuntimeStatus, Timestamp
from agent.config import load_config
from agent.navigation.route_manager import RouteManager


def rear_ego(x, y, yaw, t):
    return EgoState(Timestamp(t),
                    Pose2D(x, y, yaw, CoordinateFrame.CARLA_WORLD, 'lincoln-10hz-rear-v1'),
                    dynamics_frame=CoordinateFrame.MODEL_REAR,
                    provenance={'geometry_policy': 'lincoln-10hz-rear-v1'})


def gps_from_world(x, y, lat_ref=42., lon_ref=2.):
    """Same oracle as tests/agent/conftest.py (local route_manipulation formula)."""
    radius, scale = 6378137., math.cos(math.radians(lat_ref))
    mx = scale*math.radians(lon_ref)*radius + x
    my = scale*radius*math.log(math.tan(math.radians(90.+lat_ref)/2.)) - y
    return {'lat': math.degrees(2*math.atan(math.exp(my/(radius*scale))))-90.,
            'lon': math.degrees(mx/(radius*scale)), 'z': 0.}


def dense_plan(points, command=999):
    from types import SimpleNamespace
    option = SimpleNamespace(value=command)
    world = [(SimpleNamespace(location=SimpleNamespace(x=float(x), y=float(y), z=0.0),
                              rotation=SimpleNamespace(yaw=0.0)), option) for x, y in points]
    gps = [(gps_from_world(float(x), float(y)), option) for x, y in points]
    return gps, world


def test_agent_receives_and_keeps_the_dense_plan(config_file, m2m3_config):
    agent = Bench2DriveAgent()
    gps, world = dense_plan([(float(i), 0.0) for i in range(120)])
    agent.set_global_plan(gps, world)          # pre-setup call, as the leaderboard does
    agent.setup(config_file(m2m3_config))
    try:
        route = agent.navigation_manager.builder.route
        assert route['point_count'] == 120
        assert route['points'][0] == (0.0, 0.0) and route['points'][-1] == (119.0, 0.0)
        assert route['arc_length_m'] == pytest.approx(119.0)
        assert len(route['sha256']) == 64
        # A 50 m downsample would keep 3-4 points; the builder must not.
        sparse = [route['points'][i] for i in range(0, 120, 50)] + [route['points'][-1]]
        assert len(sparse) <= 4 < route['point_count']
    finally:
        agent.destroy()


def test_multi_frame_status_and_command_switch_match_calculator(config_file, m2m3_config):
    from tools.b2d_adapter.route_preview_contract import preview_command4
    agent = Bench2DriveAgent()
    points = [(float(i), 0.0) for i in range(0, 30)] + \
             [(30.0, -float(i)) for i in range(1, 31)]
    agent.set_global_plan(*dense_plan(points))
    agent.setup(config_file(m2m3_config))
    try:
        manager = agent.navigation_manager
        commands = []
        for index in range(12):
            t = 1.0 + index*0.1
            x = 2.0*index
            state = manager.update(rear_ego(x, 0.0, 0.0, t), Timestamp(t))
            expected = preview_command4([tuple(p) for p in points], (x, 0.0), 0.0,
                                        max_projection_m=5.0)
            if state.status_validity == 'VALID':
                assert state.command4 == tuple(expected['command4'])
                assert len(state.status8) == 8 and sum(state.status8[:4]) == 1.0
                commands.append(state.command4)
        assert commands and len(set(commands)) > 1  # a real switch was observed
    finally:
        agent.destroy()


def test_no_per_frame_replanning(config_file, m2m3_config):
    agent = Bench2DriveAgent()
    agent.set_global_plan(*dense_plan([(float(i), 0.0) for i in range(80)]))
    agent.setup(config_file(m2m3_config))
    try:
        builder = agent.navigation_manager.builder
        sha, generation = builder.route['sha256'], builder.generation
        for index in range(20):
            t = 1.0 + index*0.1
            agent.navigation_manager.update(rear_ego(0.5*index, 0.0, 0.0, t), Timestamp(t))
        assert builder.generation == generation
        assert builder.route['sha256'] == sha
    finally:
        agent.destroy()


@pytest.mark.parametrize('case', ['route_end', 'projection', 'uturn'])
def test_invalid_cases_never_produce_fake_status(case, config_file, m2m3_config):
    agent = Bench2DriveAgent()
    if case == 'route_end':
        agent.set_global_plan(*dense_plan([(0., 0.), (10., 0.)]))
        ego = rear_ego(0., 0., 0., 1.2)
    elif case == 'projection':
        agent.set_global_plan(*dense_plan([(0., 20.), (60., 20.)]))
        ego = rear_ego(0., 0., 0., 1.2)
    else:
        agent.set_global_plan(*dense_plan([(0., 0.), (20., 0.), (0., 0.)]))
        ego = rear_ego(0., 0., 0., 1.2)
    agent.setup(config_file(m2m3_config))
    try:
        state = None
        for index in range(3):
            t = 1.0 + index*0.1
            state = agent.navigation_manager.update(rear_ego(ego.pose.x_m, ego.pose.y_m, 0., t),
                                                    Timestamp(t))
        assert state.status_validity == 'INVALID'
        assert state.status8 is None and state.command4 is None
        assert state.status_reason
    finally:
        agent.destroy()


def test_missing_plan_is_logged_and_agent_fails(config_file, m2m3_config):
    agent = Bench2DriveAgent()
    agent.setup(config_file(m2m3_config))
    try:
        with pytest.raises(NavigationError, match='Global plan'):
            agent.navigation_manager.update(rear_ego(0., 0., 0., 1.0), Timestamp(1.0))
        assert agent.navigation_manager.route is None
    finally:
        agent.destroy()


def test_route_manager_rejects_sparse_only_plan_input(config_file, m2m3_config):
    manager = RouteManager(load_config(config_file(m2m3_config)).navigation)
    with pytest.raises(NavigationError):
        manager.set_global_plan([({'lat': 0., 'lon': 0.}, None)],
                                [({'location': {'x': 0., 'y': 0.}}, None)])
    assert manager.route is None
