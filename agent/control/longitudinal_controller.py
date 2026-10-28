"""Scalar PID is independent of CARLA; actuator policy remains pending."""

import math
from typing import Optional, Tuple

from agent.common.errors import ConfigurationError, ControlError, PendingImplementationError
from agent.common.types import EgoState, ReferenceTrajectory
from agent.config.settings import ControllerConfig, PIDConfig


def clamp(value: float, lower: float, upper: float) -> float:
    if not all(math.isfinite(v) for v in (value, lower, upper)) or lower > upper:
        raise ControlError("Finite value and ordered finite clamp bounds required")
    return max(lower, min(upper, value))


class PIDController:
    def __init__(self, config: PIDConfig):
        self.config = config
        self.reset()

    @staticmethod
    def validate_config(config: PIDConfig) -> None:
        values = (config.kp, config.ki, config.kd, config.output_min, config.output_max)
        if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in values):
            raise ConfigurationError("PID gains and output bounds must be explicit finite numbers")
        if config.output_min >= config.output_max:
            raise ConfigurationError("PID output_min must be less than output_max")
        if (config.integral_limit is not None
                and (isinstance(config.integral_limit, bool)
                     or not isinstance(config.integral_limit, (int, float))
                     or not math.isfinite(config.integral_limit) or config.integral_limit < 0)):
            raise ConfigurationError("PID integral_limit must be finite and nonnegative")

    def update_parameters(self, config: PIDConfig) -> None:
        self.validate_config(config)
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error: Optional[float] = None

    def step(self, error: float, dt_s: float) -> float:
        """PID with explicit dt, optional integral cap, and finite output bounds.

        This generic calculation has no vehicle tuning or actuator semantics.
        """
        self.validate_config(self.config)
        if not math.isfinite(error) or not math.isfinite(dt_s) or dt_s <= 0:
            raise ControlError("Finite error and positive finite PID dt required")
        c = self.config
        integral = self.integral + error * dt_s
        if c.integral_limit is not None:
            integral = clamp(integral, -c.integral_limit, c.integral_limit)
        derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt_s
        output = clamp(c.kp * error + c.ki * integral + c.kd * derivative,
                       c.output_min, c.output_max)
        self.integral = integral
        self.previous_error = error
        return output


class LongitudinalController:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.pid = PIDController(config.longitudinal_pid)

    def update_parameters(self, config: ControllerConfig) -> None:
        self.pid.update_parameters(config.longitudinal_pid)
        self.config = config

    def reset(self) -> None:
        self.pid.reset()

    def compute(self, reference: ReferenceTrajectory, ego: EgoState, dt_s: float) -> Tuple[float, float]:
        """Future output: (throttle, brake), each normalized to [0,1]."""
        # TODO: establish target speed from trajectory times and ego reference,
        # signed effort->throttle/brake mapping, stop/reverse and saturation rules.
        raise PendingImplementationError("Vehicle longitudinal speed/actuator policy is pending")
