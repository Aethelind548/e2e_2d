"""M6/M7/M8: mathematical contracts, diagnostics and real PID state."""

from dataclasses import replace
import math

import pytest

from agent.adapters.coordinate_transform import carla_world_to_model_xy, model_xy_to_carla_world
from agent.common.errors import ControlError, LifecycleError, PendingImplementationError, PredictionError, SensorError, TrajectoryError
from agent.common.types import CoordinateFrame, ControlCommand, EgoState, Pose2D, RuntimeStatus, Timestamp, TrajectoryPoint, TrajectoryPrediction
from agent.config.settings import ControllerConfig, PIDConfig
from agent.control.longitudinal_controller import PIDController, clamp
from agent.control.vehicle_controller import VehicleController
from agent.runtime.diagnostics import get_logger, runtime_snapshot
from agent.runtime.safety_supervisor import RuntimeSupervisor
from agent.trajectory.trajectory_decoder import TrajectoryDecoder
from agent.trajectory.trajectory_processor import TrajectoryProcessor


@pytest.mark.parametrize('name,poses', [
    ('straight', [(1., 0., 0.), (2., 0., 0.), (3., 0., 0.)]),
    ('left', [(1., .1, .1), (2., .4, .2), (3., .9, .3)]),
    ('right', [(1., -.1, -.1), (2., -.4, -.2), (3., -.9, -.3)]),
    ('decelerate', [(1.5, 0., 0.), (2.5, 0., 0.), (3., 0., 0.)]),
    ('stop', [(0., 0., 0.)] * 3)])
def test_decode_and_reference_preserve_geometry_times_and_speed_evidence(name, poses):
    times = [.5, 1., 1.5]
    prediction = TrajectoryDecoder().decode_poses(poses, times, Timestamp(1.), CoordinateFrame.MODEL_REAR, name)
    processor = TrajectoryProcessor()
    reference = processor.process(prediction, EgoState(Timestamp(1.)))
    assert reference.points == prediction.points
    assert [(p.x_m, p.y_m, p.heading_rad) for p in reference.points] == poses
    assert [p.time_from_start_s for p in reference.points] == times
    assert processor.transform(reference, reference.frame) is reference
    # Geometric finite-difference evidence only; no production target-speed API exists.
    speeds = [math.hypot(b.x_m-a.x_m, b.y_m-a.y_m) / (b.time_from_start_s-a.time_from_start_s)
              for a, b in zip(reference.points, reference.points[1:])]
    if name == 'decelerate':
        assert speeds == pytest.approx([2., 1.])
    elif name == 'stop':
        assert speeds == [0., 0.]
    processor.reset()
    assert processor.last_reference is None


@pytest.mark.parametrize('poses,times', [([], []), ([(0., 0., 0.)], []),
    ([(0., 0.)], [.5]), ([(math.nan, 0., 0.)], [.5]), ([(0., math.inf, 0.)], [.5]),
    ([(0., 0., math.nan)], [.5]), ([(0., 0., 0.)], [0.]),
    ([(0., 0., 0.)], [-.5]), ([(0., 0., 0.)], [math.nan]),
    ([(0., 0., 0.)] * 2, [.5, .5]), ([(0., 0., 0.)] * 2, [1., .5])])
def test_invalid_pose_time_sequences_rejected(poses, times):
    with pytest.raises(TrajectoryError):
        TrajectoryDecoder().decode_poses(poses, times, Timestamp(1.), CoordinateFrame.MODEL_REAR, 'test')


@pytest.mark.parametrize('timestamp', [math.nan, math.inf])
def test_nonfinite_prediction_timestamp_rejected(timestamp):
    with pytest.raises(TrajectoryError):
        TrajectoryDecoder().decode_poses([(1., 0., 0.)], [.5], Timestamp(timestamp), CoordinateFrame.MODEL_REAR, 'test')


@pytest.mark.parametrize('yaw', [0., math.pi/2, -math.pi/2, math.pi])
@pytest.mark.parametrize('local', [(5., 2.), (5., -2.), (0., 0.)])
def test_world_model_transform_roundtrip_and_left_right_sign(yaw, local):
    origin = Pose2D(10., 20., yaw, CoordinateFrame.CARLA_WORLD, 'explicit-test-rear')
    world = model_xy_to_carla_world(local, origin)
    assert carla_world_to_model_xy(world, origin) == pytest.approx(local)
    if yaw == 0.:
        assert world == pytest.approx((10. + local[0], 20. - local[1]))


def test_coordinate_unknown_origin_and_nonfinite_input_rejected():
    origin = Pose2D(0., 0., 0., CoordinateFrame.UNKNOWN, 'unverified')
    with pytest.raises(TrajectoryError):
        model_xy_to_carla_world((1., 0.), origin)
    origin = replace(origin, frame=CoordinateFrame.CARLA_WORLD)
    with pytest.raises(TrajectoryError):
        carla_world_to_model_xy((math.inf, 0.), origin)


def test_pid_acceleration_deceleration_integral_limit_and_reset():
    pid = PIDController(PIDConfig(kp=.5, ki=.1, kd=.2, output_min=-1., output_max=1., integral_limit=.25))
    assert pid.step(1., .1) > 0
    assert pid.step(-1., .1) < 0
    for _ in range(30):
        assert -1. <= pid.step(100., .1) <= 1.
    assert pid.integral == .25
    pid.reset()
    assert pid.integral == 0. and pid.previous_error is None
    assert pid.step(0., .1) == 0.
    pid.update_parameters(PIDConfig(kp=1., ki=0., kd=0., output_min=-2., output_max=2.))
    assert pid.previous_error is None and pid.integral == 0.


@pytest.mark.parametrize('error,dt', [(math.nan, .1), (1., 0.), (1., -.1), (1., math.inf)])
def test_pid_rejects_invalid_inputs_without_polluting_history(error, dt):
    pid = PIDController(PIDConfig(kp=1., ki=0., kd=0., output_min=-1., output_max=1.))
    with pytest.raises(ControlError):
        pid.step(error, dt)
    assert pid.previous_error is None and pid.integral == 0.


@pytest.mark.parametrize('value,low,high', [(math.nan, 0., 1.), (0., 1., 0.), (0., -math.inf, 1.)])
def test_clamp_rejects_invalid_values(value, low, high):
    with pytest.raises(ControlError):
        clamp(value, low, high)


@pytest.mark.parametrize('steer,throttle,brake', [(0., 1., 0.), (-1., 0., 1.), (1., 0., 0.)])
def test_control_command_normalized_boundaries(steer, throttle, brake):
    VehicleController.validate_command(ControlCommand(Timestamp(1.), steer, throttle, brake))


@pytest.mark.parametrize('steer,throttle,brake', [(1.01, 0., 0.), (-1.01, 0., 0.),
    (0., -0.01, 0.), (0., 1.01, 0.), (0., 0., 1.01), (0., 0., -.01),
    (math.nan, 0., 0.), (0., math.inf, 0.), (0., .5, .5)])
def test_invalid_control_and_simultaneous_pedals_rejected(steer, throttle, brake):
    with pytest.raises(ControlError):
        VehicleController.validate_command(ControlCommand(Timestamp(1.), steer, throttle, brake))


@pytest.mark.parametrize('handler,error', [
    ('handle_sensor_failure', SensorError('missing input')),
    ('handle_prediction_failure', PredictionError('inference exception')),
    ('handle_invalid_trajectory', TrajectoryError('invalid points')),
    ('handle_control_failure', ControlError('invalid control'))])
def test_runtime_records_fault_type_module_and_message(handler, error, caplog):
    supervisor = RuntimeSupervisor(get_logger())
    supervisor.configure()
    supervisor.begin_step(Timestamp(1.))
    supervisor.enter_module('tested-module')
    getattr(supervisor, handler)(error)
    snapshot = runtime_snapshot(supervisor.state)
    assert snapshot['status'] == 'failed'
    assert snapshot['modules']['tested-module'] == 'failed'
    assert snapshot['last_error_type'] == type(error).__name__
    assert str(error) in snapshot['last_error_message']
    assert str(error) in caplog.text
    with pytest.raises(LifecycleError):
        supervisor.begin_step(Timestamp(2.))


def test_runtime_30_bookkeeping_steps_and_reset_is_new_instance():
    supervisor = RuntimeSupervisor(get_logger())
    supervisor.configure()
    for index in range(30):
        timestamp = Timestamp(1. + index * .1)
        supervisor.begin_step(timestamp)
        supervisor.enter_module('M7.control')
        supervisor.check_control(ControlCommand(timestamp, 0., .2, 0.))
        supervisor.complete_module('M7.control')
        supervisor.complete_step()
    assert supervisor.state.completed_steps == 30
    assert supervisor.state.status == RuntimeStatus.RUNNING
    supervisor.destroy()
    assert supervisor.state.status == RuntimeStatus.DESTROYED
    assert RuntimeSupervisor(get_logger()).state.timestamp is None


def test_runtime_rejects_stale_clock_and_mismatched_command_time():
    supervisor = RuntimeSupervisor(get_logger())
    supervisor.configure()
    supervisor.begin_step(Timestamp(1.))
    with pytest.raises(LifecycleError):
        supervisor.begin_step(Timestamp(1.))
    with pytest.raises(LifecycleError):
        supervisor.check_control(ControlCommand(Timestamp(2.), 0., 0., 0.))


def test_supervisor_rejects_bad_control_before_environment_object(fake_carla):
    from agent.bench2drive_agent import Bench2DriveAgent
    supervisor = RuntimeSupervisor(get_logger())
    supervisor.configure()
    supervisor.begin_step(Timestamp(1.))
    command = ControlCommand(Timestamp(1.), math.nan, 0., 0.)
    with pytest.raises(ControlError):
        supervisor.check_control(command)
    with pytest.raises(ControlError):
        Bench2DriveAgent.to_vehicle_control(command)
    assert fake_carla.created == []
