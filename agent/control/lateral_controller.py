"""Lateral controller boundary; vehicle geometry and steering policy pending."""

from agent.common.errors import PendingImplementationError
from agent.common.types import EgoState, ReferenceTrajectory
from agent.config.settings import ControllerConfig


class LateralController:
    def __init__(self, config: ControllerConfig):
        self.config = config

    def update_parameters(self, config: ControllerConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """No lateral history allocated yet."""

    def compute(self, reference: ReferenceTrajectory, ego: EgoState, dt_s: float) -> float:
        """Future output: CARLA normalized steering, with explicit sign conversion."""
        # TODO: Pure Pursuit requires verified wheelbase, lookahead, rear origin,
        # model left-handed lateral sign and wheel-angle->CARLA steer conversion.
        raise PendingImplementationError("Lateral tracking and steering calibration are pending")
