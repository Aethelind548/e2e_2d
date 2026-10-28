"""One planner interface for the V1 model and explicit future mock use."""

from abc import ABC, abstractmethod

from agent.common.types import ModelFeatures, TrajectoryPrediction


class TrajectoryPredictor(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Explicit initialization hook, never called by module import."""

    @abstractmethod
    def predict(self, features: ModelFeatures) -> TrajectoryPrediction:
        """Produce timestamped trajectory from one unbatched feature sample."""

    @abstractmethod
    def reset(self) -> None:
        """Reset state between routes."""

    @abstractmethod
    def close(self) -> None:
        """Release owned resources."""
