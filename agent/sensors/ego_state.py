"""EgoState is shared from common.types; no competing local definition."""

from abc import ABC, abstractmethod
import math
from typing import Tuple

from agent.common.errors import ConfigurationError, PendingImplementationError, SensorError, WarmupRequiredError
from agent.common.types import CoordinateFrame, EgoState, Pose2D, SensorObservation
from agent.config.settings import EgoStateConfig, SensorConfig
from .georeference import CarlaGeoReference

TIME_TOLERANCE_S = 1e-7


class EgoStateEstimator(ABC):
    @abstractmethod
    def estimate(self, observation: SensorObservation,
                 history: Tuple[SensorObservation, ...]) -> EgoState:
        """Estimate current causal state with explicit pose reference and clock."""

    def reset(self) -> None:
        """Reset estimator history on route changes or shutdown."""


class PendingEgoStateEstimator(EgoStateEstimator):
    """Unconfigured state source; never chooses implicit physical assumptions."""
    def estimate(self, observation: SensorObservation,
                 history: Tuple[SensorObservation, ...]) -> EgoState:
        raise PendingImplementationError(
            "State estimation requires an explicit supported source and reference policy")


class CausalGNSSStateEstimator(EgoStateEstimator):
    """Engineering 10 Hz rear-position differences, never raw IMU acceleration."""

    def __init__(self, config: EgoStateConfig, sensors: SensorConfig):
        from tools.b2d_adapter.v1_builders import v1_native_config
        self.config = config
        self.native = v1_native_config()
        self.geometry = self.native.geometry
        if (config.policy != "b2d-causal-gnss-v1" or config.geometry_policy != "lincoln-10hz-rear-v1"
                or config.vehicle_type != self.geometry.axle_vehicle_type or config.planar_assumption is not True):
            raise ConfigurationError("Explicit planar b2d-causal-gnss-v1 / Lincoln engineering geometry required")
        self.projection = CarlaGeoReference(config.lat_ref_deg, config.lon_ref_deg)
        specs = {s['id']: s for s in sensors.specifications}
        for sensor_id, kind in ((config.gnss_id, 'sensor.other.gnss'), (config.imu_id, 'sensor.other.imu'),
                                (config.speed_id, 'sensor.speedometer')):
            if (not sensor_id or sensor_id not in sensors.required_ids
                    or specs.get(sensor_id, {}).get('type') != kind):
                raise ConfigurationError(f"Required declared {kind} sensor id missing")
        self.gnss_spec = specs[config.gnss_id]
        for spec in (self.gnss_spec, specs[config.imu_id]):
            for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
                value = spec.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ConfigurationError("Explicit finite GNSS/IMU installation coordinates required")
            if any(spec[key] != 0 for key in ('roll', 'pitch', 'yaw')):
                raise ConfigurationError("Only zero-rotation planar GNSS/IMU installation is supported")
        if sensors.alignment_policy != 'leaderboard-current-frame-v1' or sensors.buffer_size < 3:
            raise ConfigurationError("Causal state requires current-frame alignment and history capacity >=3")
        self.reset()

    def validate_tick_interval(self, delta_seconds: float, capacity: int) -> None:
        stride = self.geometry.source_dt_s / delta_seconds
        if stride < 1 or not math.isclose(stride, round(stride), abs_tol=1e-7, rel_tol=0):
            raise SensorError("10 Hz state samples require an integer simulation-frame stride")
        required = 2 * round(stride) + 1
        if capacity < required:
            raise ConfigurationError(f"State history capacity must be >= {required} for this simulation tick")

    def validate_route_reference(self, gps_plan, world_plan) -> None:
        """Check declared map georeference against supplied route pairs, no fit/fallback."""
        for (gps, _), (transform, _) in zip(gps_plan, world_plan):
            x, y = self.projection.to_world_xy(gps['lat'], gps['lon'])
            if math.hypot(x-transform.location.x, y-transform.location.y) > 1e-3:
                raise ConfigurationError("Configured GNSS georeference disagrees with GPS/world route pairs")

    def _poses(self, observation: SensorObservation):
        from tools.b2d_adapter.coordinate_transform import ActorPose, actor_to_reference, rotate_xy, wrap_angle
        try:
            gnss = observation.samples[self.config.gnss_id]
            imu = observation.samples[self.config.imu_id]
            if gnss.timestamp != observation.timestamp or imu.timestamp != observation.timestamp:
                raise SensorError("State estimation requires aligned current-frame GNSS/IMU")
            x, y = self.projection.to_world_xy(float(gnss.payload[0]), float(gnss.payload[1]))
            compass = float(imu.payload[6])
            if not math.isfinite(compass) or not 0 <= compass <= 2*math.pi:
                raise SensorError("IMU compass must be finite radians in [0, 2pi]")
            yaw = wrap_angle(compass - math.pi / 2.)
            dx, dy = rotate_xy((self.gnss_spec['x'], self.gnss_spec['y']), yaw)
            actor = ActorPose(x-dx, y-dy, yaw)
            rear = actor_to_reference(actor, (self.geometry.axle_dx_m, self.geometry.axle_dy_m))
            return actor, rear, (x, y), gnss.frame_id
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SensorError(f"Invalid state sensor payload: {exc}") from exc

    def estimate(self, observation: SensorObservation, history: Tuple[SensorObservation, ...]) -> EgoState:
        from tools.b2d_adapter.coordinate_transform import rotate_xy, wrap_angle
        current = self._poses(observation)
        now = observation.timestamp.seconds
        if self._start_time is None:
            self._start_time = now
        dt = self.geometry.source_dt_s
        if now - self._start_time < 2*dt - TIME_TOLERANCE_S:
            raise WarmupRequiredError("Causal state needs real samples at t, t-0.1s, t-0.2s")
        selected = []
        for target in (now-2*dt, now-dt, now):
            matches = [obs for obs in history if obs.timestamp.source == observation.timestamp.source
                       and math.isclose(obs.timestamp.seconds, target, abs_tol=TIME_TOLERANCE_S, rel_tol=0)]
            if len(matches) != 1:
                raise SensorError("Required causal state sample is missing; interpolation is unsupported")
            selected.append(self._poses(matches[0]))
        for a, b in zip(selected, selected[1:]):
            if (math.hypot(b[1].x-a[1].x, b[1].y-a[1].y)/dt > self.geometry.max_speed_mps
                    or abs(wrap_angle(b[0].yaw-a[0].yaw))/dt > self.geometry.max_yaw_rate_radps):
                raise SensorError("State position/heading discontinuity exceeds engineering limits")
        p0, p1, p2 = (item[1] for item in selected)
        velocity = ((p2.x-p1.x)/dt, (p2.y-p1.y)/dt)
        acceleration = ((p2.x-2*p1.x+p0.x)/dt**2, (p2.y-2*p1.y+p0.y)/dt**2)
        def local(vector):
            forward, right = rotate_xy(vector, -current[0].yaw)
            return forward, -right
        def pose(value, reference):
            return Pose2D(value.x, value.y, value.yaw, CoordinateFrame.CARLA_WORLD, reference)
        speed_sample = observation.samples[self.config.speed_id]
        if speed_sample.timestamp != observation.timestamp:
            raise SensorError("State speed must have the current aligned timestamp")
        speed = float(speed_sample.payload['speed'])
        if not math.isfinite(speed):
            raise SensorError("Nonfinite speed")
        return EgoState(observation.timestamp, pose(current[1], 'lincoln-10hz-rear-v1'), speed,
                        local(velocity), local(acceleration), CoordinateFrame.MODEL_REAR,
                        {'policy': self.config.policy, 'geometry_policy': self.config.geometry_policy,
                         'causal': True, 'source_frame_ids': [s[3] for s in selected],
                         'sample_interval_s': dt, 'velocity_lag_s': dt/2, 'acceleration_lag_s': dt,
                         'position_source': 'GNSS Mercator minus explicit installation offset',
                         'speed_source': 'Actor forward speedometer; not substituted for vector velocity',
                         'imu_acceleration_used': False, 'certified_ground_truth': False},
                        pose(current[0], 'carla_actor_origin'), current[2])

    def reset(self) -> None:
        self._start_time = None
