"""Deterministic fixture for future data-flow work; never an inference fallback."""

import math

from agent.common.errors import ConfigurationError, PredictionError
from agent.common.types import CoordinateFrame, ModelFeatures, TrajectoryPoint, TrajectoryPrediction
from agent.config.settings import PlannerConfig
from .predictor import TrajectoryPredictor


class MockTrajectoryPredictor(TrajectoryPredictor):
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.initialized = False

    def initialize(self) -> None:
        c = self.config
        if (c.kind != "mock" or type(c.num_poses) is not int or c.num_poses <= 0
                or c.interval_s is None or not math.isfinite(c.interval_s) or c.interval_s <= 0
                or c.mock_speed_mps is None or not math.isfinite(c.mock_speed_mps) or c.mock_speed_mps < 0
                or c.output_frame != CoordinateFrame.MODEL_REAR):
            raise ConfigurationError("Mock requires explicit count, positive interval, nonnegative speed and model rear frame")
        self.initialized = True

    def predict(self, features: ModelFeatures) -> TrajectoryPrediction:
        if not self.initialized:
            raise PredictionError("Mock predictor must be explicitly initialized")
        if features.frame != self.config.output_frame:
            raise PredictionError("Mock input/output coordinate frames must agree")
        c = self.config
        points = tuple(TrajectoryPoint(c.mock_speed_mps * i * c.interval_s, 0.0, 0.0, i * c.interval_s)
                       for i in range(1, c.num_poses + 1))
        return TrajectoryPrediction(features.timestamp, points, c.output_frame, "mock_constant_speed", features.origin)

    def reset(self) -> None:
        """Deterministic predictor has no temporal state."""

    def close(self) -> None:
        self.initialized = False
