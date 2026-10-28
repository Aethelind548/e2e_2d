"""CPU fixtures; synthetic observations never stand in for core business modules."""

import json
import math
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from agent.common.types import CoordinateFrame, ModelFeatures, Timestamp
from agent.config.settings import SensorConfig

ROOT = Path(__file__).resolve().parents[2]


def gps_from_world(x, y, lat_ref=42., lon_ref=2.):
    """Independent CARLA location_to_gps oracle from local route_manipulation."""
    radius, scale = 6378137., math.cos(math.radians(lat_ref))
    mx = scale*math.radians(lon_ref)*radius + x
    my = scale*radius*math.log(math.tan(math.radians(90.+lat_ref)/2.)) - y
    return {'lat': math.degrees(2*math.atan(math.exp(my/(radius*scale))))-90.,
            'lon': math.degrees(mx/(radius*scale)), 'z': 0.}


@pytest.fixture
def world_route():
    def make(points=((0., 0.), (10., 0.), (20., 0.), (30., 0.), (40., 0.)), command=None):
        command = command if command is not None else SimpleNamespace(value=999)
        world = [(SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.),
                                  rotation=SimpleNamespace(yaw=0.)), command) for x, y in points]
        return [(gps_from_world(x, y), command) for x, y in points], world
    return make


@pytest.fixture
def m2m3_config(sensor_config):
    sensors = asdict(sensor_config)
    sensors.update(buffer_size=5, alignment_policy='leaderboard-current-frame-v1')
    for spec in sensors['specifications']:
        if spec['id'] in ('imu', 'gnss'):
            spec.update(x=-1.4, y=0., z=0., roll=0., pitch=0., yaw=0.)
    return {'sensors': sensors,
            'ego_state': {'policy': 'b2d-causal-gnss-v1', 'gnss_id': 'gnss', 'imu_id': 'imu',
                'speed_id': 'speed', 'lat_ref_deg': 42., 'lon_ref_deg': 2.,
                'vehicle_type': 'vehicle.lincoln.mkz_2020', 'geometry_policy': 'lincoln-10hz-rear-v1',
                'planar_assumption': True},
            'navigation': {'status_policy': 'b2d-route-preview-20m-v1'}}


@pytest.fixture
def sensor_stream(raw_observation):
    def make(index, path=None, frame_base=1000, delta=.05, time_base=10.):
        from agent.common.types import SimulationTick
        t = index*delta
        x, y, yaw = path(t) if path is not None else (2.*t, 0., 0.)
        # Rear->GNSS mounting difference, independent of estimator implementation.
        relative_x = -1.4 - (-1.388632877)
        gx, gy = x+relative_x*math.cos(yaw), y+relative_x*math.sin(yaw)
        gps = gps_from_world(gx, gy)
        samples = {name: (frame_base+index, payload.copy() if hasattr(payload, 'copy') else payload)
                   for name, (_, payload) in raw_observation.items()}
        samples['gnss'] = (frame_base+index, np.array([gps['lat'], gps['lon'], 0.]))
        samples['imu'] = (frame_base+index, np.array([99., -88., 9.81, 0., 0., 0., (yaw+math.pi/2.) % (2*math.pi)]))
        return samples, SimulationTick(frame_base+index, Timestamp(time_base+t), delta)
    return make


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def sensor_config():
    return SensorConfig(specifications=(
        {"id": "rgb", "type": "sensor.camera.rgb", "width": 8, "height": 6},
        {"id": "lidar", "type": "sensor.lidar.ray_cast"},
        {"id": "imu", "type": "sensor.other.imu"},
        {"id": "gnss", "type": "sensor.other.gnss"},
        {"id": "speed", "type": "sensor.speedometer"},
    ), required_ids=("rgb", "lidar", "imu", "gnss", "speed"), buffer_size=3)


@pytest.fixture
def raw_observation():
    rng = np.random.default_rng(20261016)
    return {"rgb": (100, rng.integers(0, 256, (6, 8, 4), dtype=np.uint8)),
            "lidar": (100, rng.normal(size=(12, 4)).astype(np.float32)),
            "imu": (100, np.array([0.2, 0.0, 9.81, 0.0, 0.0, 0.0, 0.0])),
            "gnss": (100, np.array([31.0, 121.0, 5.0])),
            "speed": (100, {"speed": 2.0})}


@pytest.fixture
def features():
    generator = torch.Generator(device="cpu").manual_seed(20261016)
    return ModelFeatures(Timestamp(1.0), torch.rand((3, 256, 1024), generator=generator),
                         torch.rand((1, 256, 256), generator=generator),
                         torch.tensor([0, 1, 0, 0, 2.0, 0.1, -0.2, 0.3]),
                         CoordinateFrame.MODEL_REAR)


@pytest.fixture
def config_file(tmp_path):
    def write(data):
        path = tmp_path / "agent.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)
    return write


@pytest.fixture
def fake_carla(monkeypatch):
    """Replace only the external output object, never controllers or the pipeline."""
    import sys
    module = ModuleType("carla")
    module.created = []

    class VehicleControl(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            module.created.append(self)

    module.VehicleControl = VehicleControl
    monkeypatch.setitem(sys.modules, "carla", module)
    return module


def pytest_collection_modifyitems(items):
    """Attach explicit test/source links to the JUnit per-case evidence."""
    groups = {
        'test_m2_m3.py': ['agent/sensors/sensor_manager.py', 'agent/sensors/ego_state.py',
            'agent/sensors/georeference.py', 'agent/navigation/route_manager.py',
            'agent/navigation/status_builder.py', 'agent/bench2drive_agent.py', 'agent/environment_interface.py'],
        'test_config_lifecycle.py': ['agent/bench2drive_agent.py', 'agent/environment_interface.py', 'agent/config/settings.py'],
        'test_sensors_navigation.py': ['agent/sensors/sensor_manager.py', 'agent/sensors/observation_buffer.py',
            'agent/sensors/ego_state.py', 'agent/navigation/route_manager.py', 'agent/navigation/status_builder.py',
            'tools/b2d_adapter/native_navigation.py'],
        'test_features_predictor.py': ['agent/adapters/online_feature_adapter.py', 'agent/planning/mock_predictor.py',
            'agent/planning/diffusiondrive_predictor.py', 'agent/trajectory/trajectory_processor.py',
            'tools/training/diffusiondrive_v1/runtime.py', 'tools/b2d_adapter/camera_adapter.py',
            'repositories/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_config.py'],
        'test_trajectory_control_runtime.py': ['agent/trajectory/trajectory_decoder.py', 'agent/trajectory/trajectory_processor.py',
            'agent/adapters/coordinate_transform.py', 'agent/control/longitudinal_controller.py',
            'agent/control/vehicle_controller.py', 'agent/runtime/safety_supervisor.py', 'agent/runtime/diagnostics.py'],
        'test_mock_integration.py': ['agent/bench2drive_agent.py', 'agent/sensors/sensor_manager.py',
            'agent/navigation/route_manager.py', 'agent/adapters/online_feature_adapter.py',
            'agent/planning/mock_predictor.py', 'agent/trajectory/trajectory_processor.py',
            'agent/control/vehicle_controller.py', 'agent/runtime/safety_supervisor.py'],
    }
    pending = {
        'M4_online_features': 'agent/adapters/online_feature_adapter.py', 'M4_lidar': 'agent/adapters/online_feature_adapter.py',
        'M4_status': 'agent/adapters/online_feature_adapter.py', 'M6_transform': 'agent/trajectory/trajectory_processor.py',
        'M6_resample': 'agent/trajectory/trajectory_processor.py', 'M6_control_point': 'agent/trajectory/trajectory_processor.py',
        'M7_lateral': 'agent/control/lateral_controller.py', 'M7_longitudinal': 'agent/control/longitudinal_controller.py',
        'M8_failure_response': 'agent/runtime/safety_supervisor.py', 'M1_first_interval': 'agent/bench2drive_agent.py',
    }
    for item in items:
        if item.path.name == 'test_pending_capabilities.py':
            sources = [pending[item.callspec.params['capability']]]
        else:
            sources = groups.get(item.path.name)
        if sources is None:
            continue
        item.user_properties.extend([
            ('source_files', json.dumps(sources, ensure_ascii=False)),
            ('test_function', str(item.path.relative_to(ROOT)) + '::' + item.originalname),
            ('level', 'L2' if item.path.name == 'test_mock_integration.py' or item.originalname in (
                'test_real_chain_passes_m2_m3_and_reaches_real_m4',
                'test_agent_route_switch_during_warmup_preserves_physical_history',
                'test_live_wrapper_uses_current_tick_without_actor_motion') else 'L1'),
        ])
