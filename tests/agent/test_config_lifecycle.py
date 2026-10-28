"""M1/config tests. External API doubles are confined to the environment boundary."""

import ast
from dataclasses import asdict
from enum import Enum
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

from agent.bench2drive_agent import Bench2DriveAgent
from agent.common.errors import ConfigurationError, LifecycleError, NavigationError, PendingImplementationError, SensorError
from agent.common.types import ControlCommand, RuntimeStatus, Timestamp
from agent.config import AgentConfig, load_config
from agent.planning.diffusiondrive_predictor import DiffusionDrivePredictor

ROOT = Path(__file__).resolve().parents[2]


def test_default_config_matches_json_and_does_not_load_model():
    assert load_config() == load_config(ROOT / "agent/config/default.json") == AgentConfig()
    agent = Bench2DriveAgent()
    agent.setup(None)
    assert isinstance(agent.predictor, DiffusionDrivePredictor)
    assert agent.predictor.model is None
    assert not agent.predictor.initialized
    assert agent.supervisor.state.status == RuntimeStatus.CONFIGURED
    with pytest.raises(PendingImplementationError, match="sensor specifications"):
        agent.sensors()
    agent.destroy()


@pytest.mark.parametrize("data", [[], {"unknown": 1}, {"planner": {"kind": "rl"}},
    {"sensors": {"buffer_size": 0}}, {"sensors": {"buffer_size": True}},
    {"sensors": {"required_ids": ["absent"]}}, {"planner": {"interval_s": float("nan")}},
    {"planner": {"num_poses": 1.5}}, {"planner": {"output_frame": "guessed"}},
    {"sensors": []}, {"planner": []}, {"controller": []}])
def test_invalid_config_is_rejected(config_file, data):
    with pytest.raises(ConfigurationError):
        load_config(config_file(data))


def test_asset_paths_resolve_relative_to_config_not_cwd(config_file):
    path = config_file({"planner": {"checkpoint_path": "assets/model.ckpt"}})
    loaded = load_config(path)
    assert loaded.planner.checkpoint_path == str(Path(path).parent / "assets/model.ckpt")
    assert not Path(loaded.planner.checkpoint_path).exists()


@pytest.mark.parametrize("gps,world", [([], []), ([1], []), ([1], [1, 2])])
def test_invalid_pre_setup_route_rejected(gps, world):
    agent = Bench2DriveAgent()
    with pytest.raises(NavigationError):
        agent.set_global_plan(gps, world)


def test_pre_setup_route_and_defensive_sensor_specification(config_file, sensor_config, world_route):
    agent = Bench2DriveAgent()
    gps, world = world_route()
    agent.set_global_plan(gps, world)
    agent.setup(config_file({"sensors": asdict(sensor_config)}))
    assert agent.navigation_manager.route.gps_plan == tuple(gps)
    assert agent.navigation_manager.route.world_plan == tuple(world)
    specifications = agent.sensors()
    specifications[0]["width"] = 999
    assert agent.sensors()[0]["width"] == 8
    agent.destroy()


def test_failure_destroy_and_new_instance_have_no_prior_state(raw_observation):
    first = Bench2DriveAgent()
    with pytest.raises(LifecycleError):
        first.run_step(raw_observation, 1.0)
    first.setup(None)
    with pytest.raises(SensorError):
        first.run_step({}, 1.0)
    assert first.supervisor.state.status == RuntimeStatus.FAILED
    first.destroy()
    first.destroy()
    assert first.sensor_manager.buffer.history() == ()
    assert first.navigation_manager.route is None
    with pytest.raises(LifecycleError):
        first.setup(None)
    with pytest.raises(LifecycleError):
        first.run_step(raw_observation, 2.0)
    assert first.supervisor.state.status == RuntimeStatus.DESTROYED
    second = Bench2DriveAgent()
    second.setup(None)
    assert second.supervisor.state.status == RuntimeStatus.CONFIGURED
    assert second.supervisor.state.last_error_type is None
    assert second.supervisor.state.timestamp is None
    assert second.supervisor.state.completed_steps == 0
    assert second.sensor_manager.buffer.history() == ()
    assert second._pending_global_route is None
    second.destroy()


def test_partial_setup_failure_can_be_destroyed(config_file):
    agent = Bench2DriveAgent()
    with pytest.raises(ConfigurationError):
        agent.setup(config_file({"runtime": {"log_level": "INVALID"}}))
    assert agent.supervisor.state.status == RuntimeStatus.FAILED
    agent.destroy()
    assert agent.supervisor.state.status == RuntimeStatus.DESTROYED


def test_vehicle_control_boundary_copies_validated_fields(fake_carla):
    command = ControlCommand(Timestamp(1.0), -0.4, 0.0, 0.7, True, True)
    result = Bench2DriveAgent.to_vehicle_control(command)
    assert vars(result) == dict(steer=-0.4, throttle=0.0, brake=0.7,
                                hand_brake=True, reverse=True, manual_gear_shift=False)
    assert len(fake_carla.created) == 1


def test_local_leaderboard_signatures_and_evaluator_order():
    location = os.environ.get('B2D_LEADERBOARD_SOURCE')
    if not location:
        pytest.skip('Set B2D_LEADERBOARD_SOURCE for optional real Leaderboard source parity')
    local_b2d = Path(location)
    source = local_b2d / 'autoagents/autonomous_agent.py'
    assert source.is_file(), f'Explicit Leaderboard source unavailable: {source}'
    tree = ast.parse(source.read_text())
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AutonomousAgent')
    methods = {node.name: node for node in base.body if isinstance(node, ast.FunctionDef)}
    assert [arg.arg for arg in methods['__init__'].args.args] == ['self', 'carla_host', 'carla_port', 'debug']
    for name, names in {'setup': ['self', 'path_to_conf_file'], 'sensors': ['self'],
                        'run_step': ['self', 'input_data', 'timestamp'], 'destroy': ['self'],
                        'set_global_plan': ['self', 'global_plan_gps', 'global_plan_world_coord']}.items():
        assert [arg.arg for arg in methods[name].args.args] == names
        assert list(inspect.signature(getattr(Bench2DriveAgent, name)).parameters) == names
    evaluator = (local_b2d / 'leaderboard_evaluator.py').read_text()
    positions = [evaluator.index(fragment) for fragment in ('agent_class_obj(args.host, args.port, args.debug)',
        'self.agent_instance.set_global_plan(', 'self.agent_instance.setup(', 'self.agent_instance.sensors()')]
    assert positions == sorted(positions)
    assert "args.agent_config = args.agent_config + '+' + save_name" in evaluator


def test_get_entry_point_and_wrapper_with_external_base_double(monkeypatch, config_file, world_route):
    """Tests wrapper code, not a real CARLA/Leaderboard import or server."""
    calls = []

    class Track(Enum):
        SENSORS = 'SENSORS'

    class Base:
        def __init__(self, carla_host, carla_port, debug=False):
            calls.append(('construct', carla_host, carla_port, debug))

        def set_global_plan(self, gps, world):
            calls.append(('route', gps, world))

        @staticmethod
        def get_ros_version():
            return -1

    external = ModuleType('leaderboard.autoagents.autonomous_agent')
    external.AutonomousAgent, external.Track = Base, Track
    for name in ('leaderboard', 'leaderboard.autoagents'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.delitem(sys.modules, 'agent.environment_interface', raising=False)
    core = importlib.import_module('agent.bench2drive_agent')
    try:
        entry = core.get_entry_point()
        wrapper = getattr(core, entry)
        assert issubclass(wrapper, Base)
        assert list(inspect.signature(wrapper).parameters) == ['carla_host', 'carla_port', 'debug']
        agent = wrapper('localhost', 2000, False)
        gps, world = world_route()
        agent.set_global_plan(gps, world)
        path = config_file({})
        agent.setup(path + '+run001')
        assert agent.run_label == 'run001'
        assert agent.track is Track.SENSORS
        assert agent.navigation_manager.route.world_plan == tuple(world)
        assert calls[0] == ('construct', 'localhost', 2000, False)
        assert calls[1][0] == 'route'
        agent.destroy()
    finally:
        sys.modules.pop('agent.environment_interface', None)
        core.__dict__.pop('DiffusionDriveLeaderboardAgent', None)
