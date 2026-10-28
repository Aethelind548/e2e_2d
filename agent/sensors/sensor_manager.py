"""Preserve Leaderboard (frame_id, payload) inputs without guessing alignment."""

from copy import deepcopy
from dataclasses import replace
import math
from numbers import Real
from typing import Any, Mapping, Optional, Tuple

from agent.common.errors import ConfigurationError, PendingImplementationError, SensorError
from agent.common.types import EgoState, SensorObservation, SensorSample, SimulationTick, Timestamp
from agent.config.settings import SensorConfig
from .ego_state import EgoStateEstimator, PendingEgoStateEstimator
from .observation_buffer import ObservationBuffer


class SensorManager:
    def __init__(self, config: SensorConfig, estimator: Optional[EgoStateEstimator] = None):
        self.config = config
        self.estimator = estimator if estimator is not None else PendingEgoStateEstimator()
        self.buffer = ObservationBuffer(config.buffer_size)
        self._specifications = {spec["id"]: spec for spec in config.specifications}
        if config.alignment_policy not in (None, 'leaderboard-current-frame-v1'):
            raise ConfigurationError("Unsupported sensor alignment policy")
        self._tick: Optional[SimulationTick] = None
        self._last_tick: Optional[SimulationTick] = None

    def set_simulation_tick(self, tick: SimulationTick) -> None:
        """Supply environment metadata; never infer a clock from sensor payloads."""
        if tick is not None and not isinstance(tick, SimulationTick):
            raise SensorError("Clock metadata must be a SimulationTick")
        self._tick = tick

    def sensor_specifications(self) -> list:
        if not self.config.specifications:
            # TODO: supply verified camera/LiDAR extrinsics, FOV, rates and ids
            # matching the trained B2D contract; no guessed sensor layout.
            raise PendingImplementationError("CARLA sensor specifications have not been verified")
        return deepcopy(list(self.config.specifications))

    def _validate_payload(self, sensor_id: str, payload: Any) -> None:
        """Validate known local Leaderboard layouts without decoding/estimating.

        Unknown/undeclared modalities remain raw. This does not certify clock
        alignment, extrinsics, RGB conversion or a model-compatible ego state.
        """
        specification = self._specifications.get(sensor_id, {})
        kind = specification.get("type")
        if kind == "sensor.speedometer":
            value = payload.get("speed") if isinstance(payload, Mapping) else None
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise SensorError(f"{sensor_id}: finite numeric speed payload required")
            return
        if kind not in ("sensor.camera.rgb", "sensor.lidar.ray_cast", "sensor.other.imu", "sensor.other.gnss"):
            return
        import numpy as np
        if not isinstance(payload, np.ndarray):
            raise SensorError(f"{sensor_id}: NumPy sensor payload required")
        if kind == "sensor.camera.rgb":
            if (payload.ndim != 3 or payload.shape[2] != 4 or payload.dtype != np.uint8
                    or payload.shape[0] <= 0 or payload.shape[1] <= 0
                    or (specification.get("height") is not None and payload.shape[0] != specification["height"])
                    or (specification.get("width") is not None and payload.shape[1] != specification["width"])):
                raise SensorError(f"{sensor_id}: uint8 BGRA image must match declared dimensions")
        elif kind == "sensor.lidar.ray_cast":
            if payload.ndim != 2 or payload.shape[1] != 4 or payload.dtype != np.float32:
                raise SensorError(f"{sensor_id}: float32 Nx4 LiDAR required")
        elif payload.shape != ((7,) if kind == "sensor.other.imu" else (3,)) or payload.dtype.kind != 'f':
            raise SensorError(f"{sensor_id}: floating IMU[7] or GNSS[3] payload required")
        if not np.isfinite(payload).all():
            raise SensorError(f"{sensor_id}: nonfinite sensor payload")

    def collect(self, input_data: Mapping[str, Any], timestamp: Timestamp, store: bool = True) -> SensorObservation:
        """Unpack samples; sensor frame numbers are not interpreted as seconds."""
        if not isinstance(input_data, Mapping) or not input_data:
            raise SensorError("Nonempty Leaderboard observation mapping required")
        missing = set(self.config.required_ids) - set(input_data)
        if missing:
            raise SensorError(f"Missing required sensors: {sorted(missing)}")
        samples = {}
        for sensor_id, item in input_data.items():
            if (not isinstance(sensor_id, str) or not isinstance(item, (tuple, list))
                    or len(item) != 2 or type(item[0]) is not int or item[0] < 0
                    or item[1] is None):
                raise SensorError(f"Expected {sensor_id}: (nonnegative frame_id, payload)")
            self._validate_payload(sensor_id, item[1])
            samples[sensor_id] = SensorSample(sensor_id, item[0], item[1])
        observation = SensorObservation(timestamp, samples)
        if store:
            self.buffer.append(observation)
        return observation

    def validate_alignment(self, observation: SensorObservation) -> SensorObservation:
        """Strict current-frame fixed-step policy from the inspected Leaderboard."""
        if self.config.alignment_policy is None:
            raise PendingImplementationError("Sensor alignment requires explicit leaderboard-current-frame-v1 policy")
        tick = self._tick
        if tick is None:
            raise SensorError("Explicit current SimulationTick is required for alignment")
        if (type(tick.frame_id) is not int or tick.frame_id < 0
                or not isinstance(tick.timestamp, Timestamp)
                or any(isinstance(v, bool) or not isinstance(v, Real)
                       for v in (tick.delta_seconds, tick.timestamp.seconds))
                or not math.isfinite(tick.delta_seconds) or tick.delta_seconds <= 0
                or not math.isfinite(tick.timestamp.seconds) or tick.timestamp.seconds < 0
                or tick.timestamp != observation.timestamp
                or tick.timestamp.source != 'leaderboard_game_time'):
            raise SensorError("Invalid GameTime/frame/delta alignment anchor")
        previous = self._last_tick
        if previous is not None:
            frames = tick.frame_id - previous.frame_id
            elapsed = tick.timestamp.seconds - previous.timestamp.seconds
            if (frames <= 0 or elapsed <= 0
                    or not math.isclose(tick.delta_seconds, previous.delta_seconds, rel_tol=0, abs_tol=1e-9)
                    or not math.isclose(elapsed, frames*tick.delta_seconds, rel_tol=0, abs_tol=1e-7)):
                raise SensorError("Frame/GameTime discontinuity or changing fixed delta")
        for sample in observation.samples.values():
            if sample.frame_id != tick.frame_id:
                raise SensorError(f"{sample.sensor_id}: stale/future sensor frame {sample.frame_id}; expected {tick.frame_id}")
        return replace(observation, samples={key: replace(value, timestamp=tick.timestamp)
                                            for key, value in observation.samples.items()})

    def update(self, input_data: Mapping[str, Any], timestamp: Timestamp
               ) -> Tuple[SensorObservation, EgoState]:
        observation = self.collect(input_data, timestamp, store=False)
        observation = self.validate_alignment(observation)
        interval_check = getattr(self.estimator, 'validate_tick_interval', None)
        if interval_check is not None:
            interval_check(self._tick.delta_seconds, self.config.buffer_size)
        self.buffer.append(observation)
        self._last_tick = self._tick
        ego = self.estimator.estimate(observation, self.buffer.history())
        return observation, ego

    def reset(self) -> None:
        self.buffer.reset()
        self.estimator.reset()
        self._tick = None
        self._last_tick = None
