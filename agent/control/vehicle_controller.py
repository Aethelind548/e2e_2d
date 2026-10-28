"""Compose independent controllers into the shared ControlCommand."""

import math

from agent.common.errors import ControlError
from agent.common.types import ControlCommand, EgoState, ReferenceTrajectory
from agent.config.settings import ControllerConfig
from .lateral_controller import LateralController
from .longitudinal_controller import LongitudinalController


class VehicleController:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.lateral = LateralController(config)
        self.longitudinal = LongitudinalController(config)

    def update_parameters(self, config: ControllerConfig) -> None:
        self.longitudinal.update_parameters(config)
        self.lateral.update_parameters(config)
        self.config = config

    def compute(self, reference: ReferenceTrajectory, ego: EgoState, dt_s: float) -> ControlCommand:
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ControlError("Controller requires a positive measured simulation interval")
        if reference.timestamp != ego.timestamp:
            raise ControlError("Control reference and ego state timestamps differ")
        steer = self.lateral.compute(reference, ego, dt_s)
        throttle, brake = self.longitudinal.compute(reference, ego, dt_s)
        command = ControlCommand(ego.timestamp, steer, throttle, brake)
        self.validate_command(command)
        return command

    @staticmethod
    def validate_command(command: ControlCommand) -> None:
        if (not math.isfinite(command.timestamp.seconds)
                or not all(math.isfinite(v) for v in (command.steer, command.throttle, command.brake))):
            raise ControlError("Nonfinite control command")
        if not (-1 <= command.steer <= 1 and 0 <= command.throttle <= 1 and 0 <= command.brake <= 1):
            raise ControlError("Control command outside normalized actuator bounds")
        if command.throttle > 0 and command.brake > 0:
            raise ControlError("Throttle and brake must not both be positive")
        if type(command.hand_brake) is not bool or type(command.reverse) is not bool:
            raise ControlError("Control flags must be booleans")

    def reset(self) -> None:
        self.lateral.reset()
        self.longitudinal.reset()
