"""M1 orchestration and delayed binding to the locally inspected Leaderboard API."""

from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar

from agent.adapters.online_feature_adapter import OnlineFeatureAdapter
from agent.common.errors import LifecycleError, NavigationError, PendingImplementationError, WarmupRequiredError
from agent.common.types import ControlCommand, EgoState, GlobalRoute, NavigationState, ReferenceTrajectory, RuntimeStatus, SimulationTick, Timestamp
from agent.config import AgentConfig, load_config
from agent.config.settings import NavigationConfig
from agent.control.vehicle_controller import VehicleController
from agent.navigation.route_manager import RouteManager
from agent.planning.diffusiondrive_predictor import DiffusionDrivePredictor
from agent.planning.mock_predictor import MockTrajectoryPredictor
from agent.planning.predictor import TrajectoryPredictor
from agent.runtime.diagnostics import get_logger
from agent.runtime.safety_supervisor import RuntimeSupervisor
from agent.sensors.sensor_manager import SensorManager
from agent.sensors.ego_state import CausalGNSSStateEstimator
from agent.trajectory.trajectory_processor import TrajectoryProcessor

StageResult = TypeVar("StageResult")


def get_entry_point() -> str:
    """Expose the real AutonomousAgent subclass only when the runner requests it.

    Importing this scaffold alone does not import CARLA/Leaderboard. The actual
    runner must provide its normal CARLA/Leaderboard/scenario_runner PYTHONPATH.
    """
    from agent.environment_interface import DiffusionDriveLeaderboardAgent
    globals()["DiffusionDriveLeaderboardAgent"] = DiffusionDriveLeaderboardAgent
    return "DiffusionDriveLeaderboardAgent"


class Bench2DriveAgent:
    """Simulator-independent pipeline core, wrapped by environment_interface.

    setup only creates framework objects. Each incomplete online stage fails
    explicitly; no default throttle, synthetic features or automatic mock use.
    """

    def __init__(self):
        self.config: Optional[AgentConfig] = None
        self.sensor_manager: Optional[SensorManager] = None
        self.navigation_manager: Optional[RouteManager] = None
        self.feature_adapter: Optional[OnlineFeatureAdapter] = None
        self.predictor: Optional[TrajectoryPredictor] = None
        self.trajectory_processor: Optional[TrajectoryProcessor] = None
        self.vehicle_controller: Optional[VehicleController] = None
        self.supervisor = RuntimeSupervisor(get_logger())
        self._pending_global_route: Optional[GlobalRoute] = None
        self._previous_timestamp: Optional[Timestamp] = None
        self.last_ego_state: Optional[EgoState] = None
        self.last_navigation_state: Optional[NavigationState] = None

    def set_global_plan(self, global_plan_gps: Sequence[Any], global_plan_world_coord: Sequence[Any]) -> None:
        """Retain original complete plans, including the pre-setup runner call."""
        if self.supervisor.state.status in (RuntimeStatus.FAILED, RuntimeStatus.DESTROYED):
            raise LifecycleError("Cannot update the route of a failed/destroyed Agent")
        route = GlobalRoute(tuple(global_plan_gps), tuple(global_plan_world_coord))
        if not route.gps_plan or not route.world_plan or len(route.gps_plan) != len(route.world_plan):
            raise NavigationError("Nonempty aligned GPS and world routes required")
        if self.navigation_manager is not None:
            # Validate replacement structure before reference checks or mutation.
            RouteManager(self.config.navigation).set_global_plan(route.gps_plan, route.world_plan)
            reference_check = getattr(self.sensor_manager.estimator, 'validate_route_reference', None)
            if reference_check is not None:
                reference_check(route.gps_plan, route.world_plan)
            self.navigation_manager.set_global_plan(route.gps_plan, route.world_plan)
        else:
            RouteManager(NavigationConfig()).set_global_plan(route.gps_plan, route.world_plan)
        self._pending_global_route = route
        self.last_navigation_state = None

    def setup(self, path_to_conf_file: Optional[str]) -> None:
        if self.supervisor.state.status != RuntimeStatus.CREATED:
            raise LifecycleError("setup requires a fresh Agent instance")
        try:
            self.config = load_config(path_to_conf_file)
            self.supervisor.logger = get_logger(self.config.runtime)
            estimator = (CausalGNSSStateEstimator(self.config.ego_state, self.config.sensors)
                         if self.config.ego_state.policy is not None else None)
            self.sensor_manager = SensorManager(self.config.sensors, estimator)
            self.navigation_manager = RouteManager(self.config.navigation)
            self.feature_adapter = OnlineFeatureAdapter(self.config.features)
            predictor_type = (MockTrajectoryPredictor if self.config.planner.kind == "mock"
                              else DiffusionDrivePredictor)
            self.predictor = predictor_type(self.config.planner)
            self.trajectory_processor = TrajectoryProcessor()
            self.vehicle_controller = VehicleController(self.config.controller)
            if self._pending_global_route is not None:
                route = self._pending_global_route
                if estimator is not None:
                    estimator.validate_route_reference(route.gps_plan, route.world_plan)
                self.navigation_manager.set_global_plan(route.gps_plan, route.world_plan)
            # Deliberately no predictor.initialize(): assets/inference are outside V0.1.
            self.supervisor.configure()
        except Exception as exc:
            self.supervisor.handle_exception(exc)
            raise

    def sensors(self) -> list:
        self._require_setup()
        return self.sensor_manager.sensor_specifications()

    def _require_setup(self) -> None:
        if self.config is None or self.supervisor.state.status not in (RuntimeStatus.CONFIGURED, RuntimeStatus.WARMING_UP, RuntimeStatus.RUNNING):
            raise LifecycleError("Agent must be configured and not failed/destroyed")

    def set_simulation_tick(self, tick: SimulationTick) -> None:
        """M1 clock metadata input, supplied by the real wrapper or CPU driver."""
        self._require_setup()
        self.sensor_manager.set_simulation_tick(tick)

    def _stage(self, name: str, function: Callable[..., StageResult], *args) -> StageResult:
        self.supervisor.enter_module(name)
        value = function(*args)
        self.supervisor.complete_module(name)
        return value

    def _compute_control(self, reference: ReferenceTrajectory, ego: EgoState,
                         previous_timestamp: Optional[Timestamp]) -> ControlCommand:
        if previous_timestamp is None:
            # Supported M2 warmup establishes prior ticks. Isolated control calls
            # without one remain unsupported; no default CARLA startup command.
            raise PendingImplementationError("Control requires an actual prior tick; startup control response is pending")
        return self.vehicle_controller.compute(reference, ego, ego.timestamp.seconds - previous_timestamp.seconds)

    def run_step(self, input_data: Mapping[str, Any], timestamp: float):
        """Observation -> navigation -> features -> prediction -> reference -> control."""
        self._require_setup()
        try:
            stamp = Timestamp(timestamp)
            self.supervisor.begin_step(stamp)
            previous_timestamp = self._previous_timestamp
            self._previous_timestamp = stamp
            observation, ego = self._stage("M2.sensors", self.sensor_manager.update, input_data, stamp)
            self.last_ego_state = ego
            navigation = self._stage("M3.navigation", self.navigation_manager.update, ego, stamp)
            self.last_navigation_state = navigation
            features = self._stage("M4.features", self.feature_adapter.build_features, observation, ego, navigation)
            self._stage("M4.validate", self.feature_adapter.validate_features, features)
            prediction = self._stage("M5.predictor", self.predictor.predict, features)
            reference = self._stage("M6.trajectory", self.trajectory_processor.process, prediction, ego)
            command = self._stage("M7.control", self._compute_control, reference, ego, previous_timestamp)
            self._stage("M8.supervisor", self.supervisor.check_control, command)
            vehicle_control = self._stage("M1.output", self.to_vehicle_control, command)
            self.supervisor.complete_step()
            return vehicle_control
        except WarmupRequiredError as exc:
            self.supervisor.record_warmup(exc)
            raise
        except Exception as exc:
            self.supervisor.handle_exception(exc)
            raise

    @staticmethod
    def to_vehicle_control(command: ControlCommand):
        """Only the environment boundary creates CARLA objects, with exact fields."""
        VehicleController.validate_command(command)
        import carla
        return carla.VehicleControl(steer=command.steer, throttle=command.throttle,
                                    brake=command.brake, hand_brake=command.hand_brake,
                                    reverse=command.reverse, manual_gear_shift=False)

    def destroy(self) -> None:
        """Idempotent cleanup, including partially completed setup."""
        if self.supervisor.state.status == RuntimeStatus.DESTROYED:
            return
        failures = []
        for module in (self.sensor_manager, self.navigation_manager, self.feature_adapter,
                       self.trajectory_processor, self.vehicle_controller, self.predictor):
            if module is None:
                continue
            try:
                module.close() if module is self.predictor else module.reset()
            except Exception as exc:
                failures.append(exc)
                self.supervisor.handle_exception(exc)
        self._pending_global_route = None
        self._previous_timestamp = None
        self.last_ego_state = None
        self.last_navigation_state = None
        self.supervisor.destroy()
        if failures:
            raise failures[0]
