"""Runtime bookkeeping and boundary checks, NOT a validated driving safety system."""

import logging
import math

from agent.common.errors import LifecycleError, PendingImplementationError
from agent.common.types import AgentRuntimeState, ControlCommand, RuntimeStatus, Timestamp
from agent.control.vehicle_controller import VehicleController
from .diagnostics import runtime_snapshot


class RuntimeSupervisor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.state = AgentRuntimeState()

    def configure(self) -> None:
        if self.state.status != RuntimeStatus.CREATED:
            raise LifecycleError("A new Agent instance is required for setup")
        self.state.status = RuntimeStatus.CONFIGURED
        self.logger.info("Agent framework configured; online implementation remains pending")

    def begin_step(self, timestamp: Timestamp) -> None:
        if self.state.status not in (RuntimeStatus.CONFIGURED, RuntimeStatus.WARMING_UP, RuntimeStatus.RUNNING):
            raise LifecycleError(f"Cannot step Agent in {self.state.status.value} state")
        if not math.isfinite(timestamp.seconds):
            raise LifecycleError("Nonfinite simulation timestamp")
        if self.state.timestamp is not None:
            previous = self.state.timestamp
            if timestamp.source != previous.source or timestamp.seconds <= previous.seconds:
                raise LifecycleError("Simulation clock must be consistent and strictly increasing")
        self.state.timestamp = timestamp
        self.state.module_status.clear()
        self.state.active_module = None

    def enter_module(self, name: str) -> None:
        self.state.active_module = name
        self.state.module_status[name] = "entered"

    def complete_module(self, name: str) -> None:
        self.state.module_status[name] = "completed"
        self.state.active_module = None
        if name == 'M2.sensors':
            self.state.readiness_message = None

    def record_warmup(self, exc: Exception) -> None:
        """Expected startup has no fabricated state/control and permits next tick."""
        self.state.status = RuntimeStatus.WARMING_UP
        self.state.warmup_steps += 1
        self.state.readiness_message = str(exc)
        if self.state.active_module:
            self.state.module_status[self.state.active_module] = 'warming_up'
        self.logger.info("Agent state warmup: %s", str(exc))

    def check_control(self, command: ControlCommand) -> None:
        VehicleController.validate_command(command)
        if command.timestamp != self.state.timestamp:
            raise LifecycleError("Control command timestamp does not match current Agent step")

    def complete_step(self) -> None:
        self.state.completed_steps += 1
        self.state.status = RuntimeStatus.RUNNING

    def handle_exception(self, exc: Exception) -> None:
        """Record failure; caller re-raises. No implicit braking command."""
        self.state.status = RuntimeStatus.FAILED
        self.state.last_error_type = type(exc).__name__
        self.state.last_error_message = str(exc)
        if self.state.active_module:
            self.state.module_status[self.state.active_module] = "failed"
        self.logger.error("Agent failure: %s", runtime_snapshot(self.state),
                          exc_info=(type(exc), exc, exc.__traceback__))

    def handle_sensor_failure(self, exc: Exception) -> None:
        self.handle_exception(exc)

    def handle_prediction_failure(self, exc: Exception) -> None:
        self.handle_exception(exc)

    def handle_invalid_trajectory(self, exc: Exception) -> None:
        self.handle_exception(exc)

    def handle_control_failure(self, exc: Exception) -> None:
        self.handle_exception(exc)

    def fallback_control(self, timestamp: Timestamp) -> ControlCommand:
        # TODO: define and validate an environment-specific failure response;
        # neither full brake nor zero throttle is claimed to be a safe strategy.
        raise PendingImplementationError("A validated runtime failure-response policy is not available")

    def destroy(self) -> None:
        self.state.status = RuntimeStatus.DESTROYED
        self.state.active_module = None
