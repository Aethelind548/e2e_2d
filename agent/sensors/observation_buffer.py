"""Bounded causal observation storage; no resampling or fabricated history."""

from collections import deque
import math
from typing import Optional, Tuple

from agent.common.errors import SensorError
from agent.common.types import SensorObservation


class ObservationBuffer:
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Observation capacity must be a positive integer")
        self._observations = deque(maxlen=capacity)

    def append(self, observation: SensorObservation) -> None:
        stamp = observation.timestamp
        if not math.isfinite(stamp.seconds):
            raise SensorError("Nonfinite observation timestamp")
        if self._observations:
            previous = self._observations[-1].timestamp
            if stamp.source != previous.source or stamp.seconds <= previous.seconds:
                raise SensorError("Observation clock must be consistent and strictly increasing")
        self._observations.append(observation)

    @property
    def latest(self) -> Optional[SensorObservation]:
        return self._observations[-1] if self._observations else None

    def history(self) -> Tuple[SensorObservation, ...]:
        return tuple(self._observations)

    def reset(self) -> None:
        self._observations.clear()
