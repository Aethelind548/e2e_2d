"""All assumptions are diagnostic; none of these defaults certify V0.0.4."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .errors import AdapterError


# Dataset location, not a sampling/physics certification or automatic scan request.
DEFAULT_B2D_DATA_ROOT = Path("/data/datasets/Bench2Drive-V0.0.4")


@dataclass(frozen=True)
class CandidateConfig:
    source_dt_s: float = 0.1
    target_dt_s: float = 0.5
    history_count: int = 4  # Includes current, not four additional frames.
    scene_future_count: int = 10
    trajectory_count: int = 8
    clock_source: str = "10Hz motion/duration evidence; collection clock unverified"
    axle_dx_m: float = -1.388632877
    axle_dy_m: float = 0.0
    axle_vehicle_type: str = "vehicle.lincoln.mkz_2020"
    axle_source: str = "local stock CARLA 0.9.15; V0.0.4 asset identity unverified"
    # Project diagnostic gates, NOT official dataset accuracy specifications.
    max_planar_tilt_deg: float = 5.0
    max_speed_mps: float = 80.0
    max_yaw_rate_radps: float = 4.0
    lidar_z_shift_m: float = 0.0  # Explicitly unverified height origin.
    lidar_min_xy_m: float = -32.0
    lidar_max_xy_m: float = 32.0
    lidar_pixels_per_m: int = 4
    lidar_split_height_m: float = 0.2
    lidar_max_height_m: float = 100.0
    lidar_count_clip: int = 5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise AdapterError("invalid_config", name + " must be finite")
        for name in ("history_count", "scene_future_count", "trajectory_count",
                     "lidar_pixels_per_m", "lidar_count_clip"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise AdapterError("invalid_config", name + " must be a positive integer")
        if min(self.source_dt_s, self.target_dt_s, self.max_planar_tilt_deg,
               self.max_speed_mps, self.max_yaw_rate_radps) <= 0:
            raise AdapterError("invalid_config", "time intervals and diagnostic gates must be positive")
        if self.scene_future_count < self.trajectory_count:
            raise AdapterError("invalid_config", "Scene must include all trajectory points")
        ratio = self.target_dt_s / self.source_dt_s
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9, rel_tol=0):
            raise AdapterError("unsupported_sampling", "integer stride required; interpolation not implemented")
        bins = (self.lidar_max_xy_m - self.lidar_min_xy_m) * self.lidar_pixels_per_m
        if bins <= 0 or not math.isclose(bins, round(bins), abs_tol=1e-9):
            raise AdapterError("invalid_config", "LiDAR bounds must define integer bins")
        if self.lidar_max_height_m <= self.lidar_split_height_m:
            raise AdapterError("invalid_config", "LiDAR height bounds reversed")
        if not self.clock_source or not self.axle_source or not self.axle_vehicle_type:
            raise AdapterError("invalid_config", "candidate evidence and vehicle type required")

    @property
    def stride(self):
        return int(round(self.target_dt_s / self.source_dt_s))

    def provenance(self):
        values = asdict(self)
        digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return {"schema": "b2d-diagnostic-v1", "candidate_only": True,
                "certified_ground_truth": False, "clock_status": "candidate",
                "reference_status": "candidate", "parameters": values,
                "policy_sha256": digest}
