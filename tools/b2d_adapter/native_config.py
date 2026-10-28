"""Versioned strict/engineering contracts; engineering never toggles certification."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

from .config import CandidateConfig
from .errors import AdapterError


CAMERA_POLICIES = ("navsim-pixel-crop-v1", "b2d-full-view-v1", "b2d-proportional-crop-v1")


@dataclass(frozen=True)
class NativeBuilderConfig:
    camera_policy: str  # Required: do not silently choose a migration policy.
    geometry: CandidateConfig = field(default_factory=CandidateConfig)
    mode: str = "strict"
    navigation_policy: str = "blocked"
    navigation_bearing_threshold_deg: float = 10.0
    navigation_min_distance_m: float = .5
    navigation_max_distance_m: float = 60.0
    lidar_height_policy: str = "actor-zshift-v1"

    def __post_init__(self):
        if not isinstance(self.geometry, CandidateConfig):
            raise AdapterError("invalid_config", "geometry must be CandidateConfig")
        if self.camera_policy not in CAMERA_POLICIES:
            raise AdapterError("invalid_camera_policy", CAMERA_POLICIES)
        c = self.geometry
        if self.mode not in ("strict", "engineering"):
            raise AdapterError("invalid_mode", self.mode)
        if self.navigation_policy not in ("blocked", "b2d-near-bearing-v1") or (self.mode == "strict" and self.navigation_policy != "blocked"):
            raise AdapterError("invalid_navigation_policy", "strict retains scheme C; engineering navigation must be explicit")
        if self.lidar_height_policy != "actor-zshift-v1":
            raise AdapterError("invalid_height_policy", self.lidar_height_policy)
        if not (all(math.isfinite(v) for v in (self.navigation_bearing_threshold_deg, self.navigation_min_distance_m, self.navigation_max_distance_m))
                and 0 < self.navigation_bearing_threshold_deg < 90
                and 0 < self.navigation_min_distance_m < self.navigation_max_distance_m):
            raise AdapterError("invalid_navigation_config", "finite bearing and distance gates required")
        if self.mode == "engineering" and (c.source_dt_s != .1 or c.axle_dx_m != -1.388632877 or c.axle_dy_m != 0
                                            or c.axle_vehicle_type != "vehicle.lincoln.mkz_2020"):
            raise AdapterError("engineering_parameter_mismatch", "this phase fixes 10Hz and Lincoln rear offset (-1.388632877,0)m")
        bins = (c.lidar_max_xy_m-c.lidar_min_xy_m)*c.lidar_pixels_per_m
        if c.trajectory_count != 8 or not math.isclose(c.target_dt_s, .5, rel_tol=0, abs_tol=1e-12) or bins != 256:
            raise AdapterError("native_contract_mismatch", "v0.1 fixes 8x3/.5s trajectory and 256x256 BEV; source dt remains configurable")

    def provenance(self):
        # Keep archived v0.1 strict hashes stable; engineering is a new contract.
        parameters = asdict(self) if self.mode == "engineering" else {"camera_policy": self.camera_policy, "geometry": asdict(self.geometry)}
        schema = "b2d-native-engineering-v0.2" if self.mode == "engineering" else "b2d-native-v0.1"
        digest = hashlib.sha256(json.dumps({"schema": schema, "parameters": parameters}, sort_keys=True).encode()).hexdigest()
        return {"schema": schema, "policy_sha256": digest, "mode": self.mode,
                "parameters": parameters, "geometry_evidence": self.geometry.provenance(),
                "candidate_only": True, "certified_ground_truth": False,
                "engineering_assumptions_accepted": self.mode == "engineering",
                "engineering_parameter_version": "lincoln-10hz-rear-v1" if self.mode == "engineering" else None,
                "engineering_parameter_source": "user-approved integration parameters; historical clock/asset certification deferred",
                "navigation": self.navigation_policy, "height_origin": "Actor z plus explicit engineering shift; not ground height",
                "lidar_height_policy": self.lidar_height_policy}
