"""Executable Blocked probes: only the specific unimplemented exception is xfail.

An unexpected exception fails, and implementation returning successfully is a
strict XPASS failure that requires replacing the probe with behavioral checks.
"""

import pytest

from agent.adapters.online_feature_adapter import OnlineFeatureAdapter
from agent.bench2drive_agent import Bench2DriveAgent
from agent.common.errors import PendingImplementationError
from agent.common.types import CoordinateFrame, EgoState, NavigationState, Pose2D, Timestamp
from agent.config.settings import ControllerConfig, FeatureConfig, NavigationConfig
from agent.control.lateral_controller import LateralController
from agent.control.longitudinal_controller import LongitudinalController
from agent.navigation.route_manager import RouteManager
from agent.runtime.diagnostics import get_logger
from agent.runtime.safety_supervisor import RuntimeSupervisor
from agent.sensors.ego_state import PendingEgoStateEstimator
from agent.sensors.sensor_manager import SensorManager
from agent.trajectory.trajectory_decoder import TrajectoryDecoder
from agent.trajectory.trajectory_processor import TrajectoryProcessor


@pytest.mark.xfail(strict=True, raises=PendingImplementationError, reason='BLOCKED: core capability is explicitly unimplemented; see docs/问题记录.md')
@pytest.mark.parametrize('capability', [
    'M4_online_features', 'M4_lidar', 'M6_transform', 'M6_resample',
    'M6_control_point', 'M7_lateral', 'M7_longitudinal', 'M8_failure_response', 'M1_first_interval'])
def test_required_capability_blocked(capability, sensor_config, raw_observation):
    manager = SensorManager(sensor_config)
    observation = manager.collect(raw_observation, Timestamp(1.))
    ego = EgoState(Timestamp(1.), pose=Pose2D(0., 0., 0., CoordinateFrame.CARLA_WORLD, 'explicit-test-rear'),
                   speed_mps=2., velocity_xy_mps=(2., 0.), acceleration_xy_mps2=(.2, 0.),
                   dynamics_frame=CoordinateFrame.MODEL_REAR)
    nav = NavigationState(Timestamp(1.), raw_command=4)
    adapter = OnlineFeatureAdapter(FeatureConfig())
    processor = TrajectoryProcessor()
    prediction = TrajectoryDecoder().decode_poses([(1., 0., 0.), (2., 0., 0.)], [.5, 1.],
                                                Timestamp(1.), CoordinateFrame.MODEL_REAR, 'explicit-test')
    reference = processor.process(prediction, ego)
    calls = {
        'M4_online_features': lambda: adapter.build_features(observation, ego, nav),
        'M4_lidar': lambda: adapter.build_lidar(observation, ego),
        'M6_transform': lambda: processor.transform(reference, CoordinateFrame.CARLA_WORLD),
        'M6_resample': lambda: processor.resample(reference, [.25, .5, .75]),
        'M6_control_point': lambda: processor.select_control_point(reference, ego),
        'M7_lateral': lambda: LateralController(ControllerConfig()).compute(reference, ego, .1),
        'M7_longitudinal': lambda: LongitudinalController(ControllerConfig()).compute(reference, ego, .1),
        'M8_failure_response': lambda: RuntimeSupervisor(get_logger()).fallback_control(Timestamp(1.)),
        'M1_first_interval': lambda: Bench2DriveAgent()._compute_control(reference, ego, None),
    }
    calls[capability]()
