from types import SimpleNamespace

import pytest

from agent.common.types import CoordinateFrame, EgoState, Pose2D, Timestamp
from agent.config.settings import NavigationConfig
from agent.navigation.route_manager import RouteManager
from agent.navigation.route_preview import route_preview_from_world_nodes


def test_agent_route_preview_uses_shared_calculator():
    r = route_preview_from_world_nodes(
        [(0, 0), (50, 0)], (0, 0), 0, route_generation=7)
    assert r.source == 'CARLA_SET_GLOBAL_PLAN'
    assert r.route_generation == 7
    assert r.valid is True
    assert r.command4 == [0, 1, 0, 0]
    assert r.preview['route_progress_m'] == pytest.approx(0)


def test_agent_route_preview_uturn_remains_unsupported():
    r = route_preview_from_world_nodes(
        [(0, 0), (20, 0), (0, 0)], (0, 0), 0)
    assert r.valid is False
    assert r.preview['uturn_status'] == 'UNSUPPORTED_UTURN'
    assert r.command4 == [0, 0, 0, 1]


def test_route_manager_preview_uses_dense_global_plan():
    manager = RouteManager(NavigationConfig())
    world = []
    for x in (0, 10, 20, 30, 40):
        world.append((SimpleNamespace(location=SimpleNamespace(x=float(x), y=0.0),
                                      rotation=SimpleNamespace(yaw=0.0)), None))
    gps = [({'lat': 0.0, 'lon': 0.0}, None) for _ in world]
    manager.set_global_plan(gps, world)
    ego = EgoState(timestamp=Timestamp(1.0),
                   pose=Pose2D(0.0, 0.0, 0.0, CoordinateFrame.CARLA_WORLD, 'test_rear'),
                   provenance={'geometry_policy': 'lincoln-10hz-rear-v1'})
    result = manager.preview(ego)
    assert result.valid is True
    assert result.command4 == [0, 1, 0, 0]
    # The production update path now uses the same dense plan.
    assert manager.builder.route['point_count'] == 5
    assert manager.builder.route['policy'] == 'b2d-route-preview-20m-v1'


def test_route_manager_preview_requires_current_world_rear_pose():
    manager = RouteManager(NavigationConfig())
    with pytest.raises(Exception):
        manager.preview(EgoState(timestamp=Timestamp(1.0)))
