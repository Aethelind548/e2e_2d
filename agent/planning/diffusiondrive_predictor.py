"""Real predictor scaffold; no checkpoint or GPU work in constructor/setup."""

from typing import Any, Optional

from agent.common.errors import PendingImplementationError, PredictionError
from agent.common.types import ModelFeatures, TrajectoryPrediction
from agent.config.settings import PlannerConfig
from agent.trajectory.trajectory_decoder import TrajectoryDecoder
from .model_loader import V1ModelLoader
from .predictor import TrajectoryPredictor


class DiffusionDrivePredictor(TrajectoryPredictor):
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.loader = V1ModelLoader(config)
        self.decoder = TrajectoryDecoder()
        self.model: Optional[Any] = None
        self.initialized = False

    def initialize(self) -> None:
        # TODO: explicitly build/load with V1ModelLoader after checkpoint/config
        # verification, then establish eval mode, device and inference context.
        raise PendingImplementationError("Real V1 predictor initialization is pending")

    def predict(self, features: ModelFeatures) -> TrajectoryPrediction:
        if not self.initialized or self.model is None:
            raise PredictionError("V1 predictor is not initialized; setup intentionally does not load a model")
        # TODO: validate features, batch once, place tensors on configured device,
        # use eval/inference_mode, call model(features_dict, targets=None), then
        # decode output['trajectory'] with verified timestamps/frame/origin.
        raise PendingImplementationError("V1 forward invocation and output binding are pending")

    def reset(self) -> None:
        """V1 scaffold has no recurrent inference state."""

    def close(self) -> None:
        self.model = None
        self.initialized = False
