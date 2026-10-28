"""M4/M5 contract tests with real CPU tensors and no neural network construction."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from agent.adapters.online_feature_adapter import OnlineFeatureAdapter
from agent.common.errors import ConfigurationError, FeatureError, PendingImplementationError, PredictionError, TrajectoryError
from agent.common.types import CoordinateFrame, EgoState, NavigationState, Pose2D, Timestamp
from agent.config.settings import FeatureConfig, PlannerConfig
from agent.planning.diffusiondrive_predictor import DiffusionDrivePredictor
from agent.planning.mock_predictor import MockTrajectoryPredictor

ROOT = Path(__file__).resolve().parents[2]


def test_cpu_feature_contract_and_dict(features):
    OnlineFeatureAdapter.validate_features(features)
    assert set(features.as_dict()) == {'camera_feature', 'lidar_feature', 'status_feature'}
    assert all(tensor.device.type == 'cpu' for tensor in features.as_dict().values())
    assert features.as_dict()['status_feature'] is features.status_feature


@pytest.mark.parametrize('name', ['camera_feature', 'lidar_feature', 'status_feature'])
@pytest.mark.parametrize('corruption', ['shape', 'dtype', 'nan', 'inf'])
def test_feature_shape_dtype_and_finiteness(features, name, corruption):
    tensor = getattr(features, name).clone()
    if corruption == 'shape':
        tensor = tensor.unsqueeze(0)
    elif corruption == 'dtype':
        tensor = tensor.to(torch.float64)
    else:
        tensor.flatten()[0] = float(corruption)
    with pytest.raises(FeatureError, match=name):
        OnlineFeatureAdapter.validate_features(replace(features, **{name: tensor}))


@pytest.mark.parametrize('name', ['camera_feature', 'lidar_feature'])
@pytest.mark.parametrize('value', [-0.001, 1.001])
def test_sensor_feature_normalization_range(features, name, value):
    tensor = getattr(features, name).clone()
    tensor.flatten()[0] = value
    with pytest.raises(FeatureError, match=r'\[0,1\]'):
        OnlineFeatureAdapter.validate_features(replace(features, **{name: tensor}))


def test_feature_timestamp_is_finite(features):
    with pytest.raises(FeatureError, match='timestamp'):
        OnlineFeatureAdapter.validate_features(replace(features, timestamp=Timestamp(math.nan)))


def test_feature_device_mismatch_rejected_before_tensor_operations(features):
    lidar = torch.empty_like(features.lidar_feature, device='meta')
    with pytest.raises(FeatureError, match='device'):
        OnlineFeatureAdapter.validate_features(replace(features, lidar_feature=lidar))


def test_feature_frame_is_explicit(features):
    with pytest.raises(FeatureError, match='frame'):
        OnlineFeatureAdapter.validate_features(replace(features, frame=CoordinateFrame.UNKNOWN))


def test_adapter_rejects_misaligned_data_before_pending_path(features):
    from agent.common.types import SensorObservation
    adapter = OnlineFeatureAdapter(FeatureConfig())
    with pytest.raises(FeatureError, match='timestamp'):
        adapter.build_features(SensorObservation(Timestamp(1.0), {}), EgoState(Timestamp(2.0)),
                               NavigationState(Timestamp(1.0)))


@pytest.mark.parametrize('config_name,expected_policy', [
    ('pilot_original.json', 'navsim-pixel-crop-v1'),
    ('pilot_proportional.json', 'b2d-proportional-crop-v1'),
    ('features_remaining_20261015.json', 'b2d-proportional-crop-v1')])
def test_current_training_contract_from_official_v1_config(config_name, expected_policy):
    # Only import the official configuration class; never instantiate a network.
    sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
    try:
        from tools.b2d_adapter.v1_builders import official_v1_module, validate_v1_config
        from tools.training.diffusiondrive_v1.runtime import native_configuration
        training = json.loads((ROOT / 'tools/training/diffusiondrive_v1' / config_name).read_text())
        native = native_configuration(training)
        model = official_v1_module('transfuser_config').TransfuserConfig()
        contract = validate_v1_config(native, model)
        assert contract['tensor_contract']['camera_width'] == 1024
        assert contract['tensor_contract']['camera_height'] == 256
        assert model.trajectory_sampling.num_poses == 8
        assert model.trajectory_sampling.interval_length == 0.5
        assert native.camera_policy == expected_policy
        if expected_policy == 'navsim-pixel-crop-v1':
            registry = json.loads((ROOT / 'tools/training/diffusiondrive_v1/camera_original_compatibility.json').read_text())
            digest = hashlib.sha256(json.dumps(training, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            assert registry['historical_configs'][digest]['path'] == 'tools/training/diffusiondrive_v1/' + config_name
            assert native.camera_policy == registry['historical_configs'][digest]['camera_policy']
        else:
            assert training['camera_policy'] == expected_policy
        assert OnlineFeatureAdapter.V1_SHAPES == {
            'camera_feature': (3, model.camera_height, model.camera_width),
            'lidar_feature': (model.lidar_seq_len, model.lidar_resolution_height, model.lidar_resolution_width),
            'status_feature': (8,)}
    finally:
        sys.path.remove(str(ROOT / 'repositories/DiffusionDrive'))


def test_verified_rgb_channel_order_normalization_and_stitch():
    adapter = OnlineFeatureAdapter(FeatureConfig(camera_policy='b2d-proportional-crop-v1'))
    cameras = {}
    colors = {'CAM_FRONT_LEFT': (255, 0, 0), 'CAM_FRONT': (0, 255, 0), 'CAM_FRONT_RIGHT': (0, 0, 255)}
    for name, rgb in colors.items():
        cameras[name] = np.broadcast_to(np.array(rgb, dtype=np.uint8), (900, 1600, 3)).copy()
    camera = adapter.camera_from_verified_rgb(cameras)
    assert camera.shape == (3, 256, 1024)
    assert camera.dtype == torch.float32 and camera.device.type == 'cpu'
    torch.testing.assert_close(camera[:, 128, 0], torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(camera[:, 128, 512], torch.tensor([0., 1., 0.]))
    torch.testing.assert_close(camera[:, 128, -1], torch.tensor([0., 0., 1.]))


def test_camera_reuses_current_shared_policy_on_identical_rgb_data():
    from types import SimpleNamespace
    from tools.b2d_adapter.camera_adapter import proportional_camera_feature
    rng = np.random.default_rng(20261016)
    cameras = {name: rng.integers(0, 256, (900, 1600, 3), dtype=np.uint8)
               for name in ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT')}
    actual = OnlineFeatureAdapter(FeatureConfig(camera_policy='b2d-proportional-crop-v1')).camera_from_verified_rgb(cameras)
    torch.testing.assert_close(actual, proportional_camera_feature(SimpleNamespace(cameras=cameras)), rtol=0, atol=0)


def test_rgb_invalid_input_not_replaced_by_valid_features():
    from tools.b2d_adapter.errors import AdapterError
    adapter = OnlineFeatureAdapter(FeatureConfig(camera_policy='b2d-proportional-crop-v1'))
    with pytest.raises(AdapterError):
        adapter.camera_from_verified_rgb({'CAM_FRONT': np.zeros((6, 8, 3), dtype=np.uint8)})
    with pytest.raises(FeatureError, match='policy'):
        OnlineFeatureAdapter(FeatureConfig()).camera_from_verified_rgb({})


def test_mock_lifecycle_determinism_protocol_and_origin(features):
    config = PlannerConfig(kind='mock', num_poses=8, interval_s=0.5,
                           output_frame=CoordinateFrame.MODEL_REAR, mock_speed_mps=2.)
    mock = MockTrajectoryPredictor(config)
    with pytest.raises(PredictionError, match='initialized'):
        mock.predict(features)
    mock.initialize()
    origin = Pose2D(5., 10., 0.2, CoordinateFrame.CARLA_WORLD, 'explicit-test-rear')
    features = replace(features, origin=origin)
    first, second = mock.predict(features), mock.predict(features)
    assert first == second
    assert first.timestamp == features.timestamp
    assert first.origin == origin
    assert first.frame == features.frame
    assert first.source == 'mock_constant_speed'
    assert [p.x_m for p in first.points] == pytest.approx(range(1, 9))
    assert [p.time_from_start_s for p in first.points] == pytest.approx([0.5 * i for i in range(1, 9)])
    mock.reset()
    assert mock.predict(features) == first
    mock.close()
    with pytest.raises(PredictionError):
        mock.predict(features)


def test_mock_requires_explicit_configuration():
    with pytest.raises(ConfigurationError):
        MockTrajectoryPredictor(PlannerConfig(kind='mock')).initialize()


def test_real_predictor_is_an_uninitialized_interface(features):
    predictor = DiffusionDrivePredictor(PlannerConfig())
    assert predictor.model is None
    with pytest.raises(PredictionError):
        predictor.predict(features)
    with pytest.raises(PendingImplementationError):
        predictor.initialize()
    predictor.close()


@pytest.mark.parametrize('mode', ['normal', 'empty', 'exception'])
def test_mock_prediction_processor_fault_protocol(features, mode):
    from agent.runtime.diagnostics import get_logger
    from agent.runtime.safety_supervisor import RuntimeSupervisor
    from agent.trajectory.trajectory_processor import TrajectoryProcessor

    class FaultPredictor(MockTrajectoryPredictor):
        def predict(self, supplied_features):
            prediction = super().predict(supplied_features)
            if mode == 'exception':
                raise PredictionError('deterministic unit inference failure')
            return replace(prediction, points=()) if mode == 'empty' else prediction

    predictor = FaultPredictor(PlannerConfig(kind='mock', num_poses=8, interval_s=.5,
                                           output_frame=CoordinateFrame.MODEL_REAR, mock_speed_mps=2.))
    predictor.initialize()
    supervisor = RuntimeSupervisor(get_logger())
    supervisor.configure()
    supervisor.begin_step(features.timestamp)
    processor = TrajectoryProcessor()
    if mode == 'normal':
        reference = processor.process(predictor.predict(features), EgoState(features.timestamp))
        assert len(reference.points) == 8
        assert reference.source == 'mock_constant_speed'
    else:
        error_type = PredictionError if mode == 'exception' else TrajectoryError
        supervisor.enter_module('M5.predictor' if mode == 'exception' else 'M6.trajectory')
        with pytest.raises(error_type) as caught:
            processor.process(predictor.predict(features), EgoState(features.timestamp))
        supervisor.handle_exception(caught.value)
        assert supervisor.state.last_error_type == error_type.__name__
        assert supervisor.state.status.value == 'failed'
        assert processor.last_reference is None
    predictor.close()


def test_mock_prediction_and_reference_30_cpu_steps(features):
    """M5->M6 only; this is explicitly not 30 full Agent control cycles."""
    from agent.trajectory.trajectory_processor import TrajectoryProcessor
    predictor = MockTrajectoryPredictor(PlannerConfig(kind='mock', num_poses=8, interval_s=.5,
        output_frame=CoordinateFrame.MODEL_REAR, mock_speed_mps=2.))
    predictor.initialize()
    processor = TrajectoryProcessor()
    for index in range(30):
        supplied = replace(features, timestamp=Timestamp(1. + .1 * index))
        OnlineFeatureAdapter.validate_features(supplied)
        prediction = predictor.predict(supplied)
        reference = processor.process(prediction, EgoState(supplied.timestamp))
        assert reference.timestamp == supplied.timestamp
        assert len(reference.points) == 8
        assert reference.points[-1].x_m == 8.
    assert processor.last_reference.timestamp.seconds == pytest.approx(3.9)
    predictor.close()
