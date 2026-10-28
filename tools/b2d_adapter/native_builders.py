"""Versioned B2D-native Builders: strict certification and engineering entries.

No map, labels, PDM, network, or synthesized navigation. Feature reads never
call the existing full-scene object adapter, which includes future supervision.
"""

from dataclasses import dataclass
import numpy as np

from .camera_adapter import candidate_camera_feature, camera_metadata, proportional_camera_feature
from .coordinate_transform import wrap_angle
from .errors import AdapterError
from .lidar_adapter import candidate_lidar_feature
from .lidar_decode import decode_lidar_in_process
from .native_config import NativeBuilderConfig
from .native_navigation import near_route_command
from .object_adapter import candidate_causal_state
from .trajectory_target import CandidateTrajectoryTargetBuilder


@dataclass
class NativeBuildResult:
    tensors: dict
    provenance: dict
    validation: dict


@dataclass
class B2DObservation:
    raw: object
    motion: object
    policy_sha256: str


class B2DEgoStatusBuilder:
    """Engineering status8 uses explicit near-route policy and causal rear motion."""
    def __init__(self, config=None):
        self.config = config

    def diagnose_motion(self, observation):
        state = observation.motion
        if state.velocity is None or state.acceleration is None:
            raise AdapterError("insufficient_past", state.provenance)
        if not all(np.asarray(x).shape == (2,) and np.isfinite(x).all() for x in (state.velocity, state.acceleration)):
            raise AdapterError("invalid_state", "finite 2D candidate velocity/acceleration required")
        return state

    def build(self, observation):
        return self.build_result(observation).tensors["status_feature"]

    def build_result(self, observation):
        state = self.diagnose_motion(observation)
        if self.config is not None and self.config.mode == "engineering" and self.config.navigation_policy != "blocked":
            import torch
            command, navigation = near_route_command(observation.raw, self.config)
            feature = torch.from_numpy(np.concatenate((command, state.velocity, state.acceleration)).astype(np.float32))
            B2DFeatureBuilder._validate_tensor(feature, (8,))
            return NativeBuildResult({"status_feature": feature}, dict(self.config.provenance(),
                                     navigation_source=navigation, motion_source=state.provenance),
                                     {"algorithm": "pass", "engineering": "pass", "formal_semantics": "unverified",
                                      "velocity_lag_s": self.config.geometry.source_dt_s/2,
                                      "acceleration_lag_s": self.config.geometry.source_dt_s})
        # Deliberately no integer mapping, arbitrary one-hot, or Unknown fallback.
        raise AdapterError("unresolved_navigation", {"scheme": "C", "raw_navigation": observation.raw.navigation,
                           "required": "inference-available expected route with certified command mapping",
                           "dynamics": "causal candidate only; clock/reference/filtering unverified"})


class B2DFeatureBuilder:
    def __init__(self, reader, config, lidar_decoder_python=None):
        if not isinstance(config, NativeBuilderConfig):
            raise AdapterError("invalid_config", "explicit NativeBuilderConfig required")
        self.reader, self.config = reader, config
        self.lidar_decoder_python = lidar_decoder_python
        self.status_builder = B2DEgoStatusBuilder(config)

    def _navsim_camera_builder(self):
        # Historical V2 entry stays unchanged; V1 overrides only this factory.
        from navsim.agents.diffusiondrivev2.transfuser_features import TransfuserFeatureBuilder
        from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config import TransfuserConfig
        return TransfuserFeatureBuilder(TransfuserConfig())

    def read_current(self, current):
        c = self.config.geometry
        self.reader.index.check_ref(current)
        # Validate past/current without ever reading future labels.
        if current.frame_id < 2:
            raise AdapterError("insufficient_past", current.token)
        previous = None
        for i in range(current.frame_id-2, current.frame_id+1):
            raw = self.reader.read(self.reader.index.ref(i))
            if raw.ego_box.get("type_id") != c.axle_vehicle_type:
                raise AdapterError("reference_vehicle_mismatch", raw.ref.token)
            if not np.isfinite(raw.ego_box["rotation"]).all():
                raise AdapterError("invalid_pose", raw.ref.token)
            if max(abs(float(v)) for v in raw.ego_box["rotation"][:2]) > c.max_planar_tilt_deg:
                raise AdapterError("nonplanar_pose", raw.ref.token)
            if raw.state.get("teleport_flag", False):
                raise AdapterError("teleport_flag", raw.ref.token)
            if previous is not None:
                p, q = previous.actor_pose, raw.actor_pose
                if np.hypot(q.x-p.x, q.y-p.y)/c.source_dt_s > c.max_speed_mps or abs(wrap_angle(q.yaw-p.yaw))/c.source_dt_s > c.max_yaw_rate_radps:
                    raise AdapterError("teleport_or_discontinuity", raw.ref.token)
            previous = raw
        state = candidate_causal_state(self.reader, current, c)
        raw = self.reader.read(current, cameras=True, lidar=self.lidar_decoder_python is None)
        if self.lidar_decoder_python is not None:
            expected_path = str(self.reader.index.path(current, "lidar"))
            raw.lidar_xyz, raw.lidar_metadata = decode_lidar_in_process(current, self.lidar_decoder_python)
            if raw.lidar_metadata.get("source_files", {}).get("lidar") != expected_path:
                raise AdapterError("decoder_source_mismatch", "worker decoded a different indexed LAZ path")
            raw.provenance["source_files"]["lidar"] = expected_path
        return B2DObservation(raw, state, self.config.provenance()["policy_sha256"])

    def _check_observation(self, observation):
        self.reader.index.check_ref(observation.raw.ref)
        if observation.policy_sha256 != self.config.provenance()["policy_sha256"]:
            raise AdapterError("policy_mismatch", "observation and Builder policies differ")

    def build_camera(self, observation):
        import torch
        self._check_observation(observation)
        raw = observation.raw
        for name in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"):
            image = raw.cameras.get(name)
            if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise AdapterError("invalid_camera", name)
        if self.config.camera_policy == "b2d-proportional-crop-v1":
            feature = proportional_camera_feature(raw)
            implementation = "B2D 23/346 direct crop; LEFT/FRONT/RIGHT stitch; cv2.INTER_LINEAR; ToTensor once"
        elif self.config.camera_policy == "navsim-pixel-crop-v1":
            # Minimal actual camera-only view, not a fake complete Scene/AgentInput.
            from navsim.common.dataclasses import AgentInput, Camera, Cameras
            if any(im.shape[0] <= 56 or im.shape[1] <= 832 for im in raw.cameras.values()):
                raise AdapterError("invalid_camera_crop", "original 28/416 crop would be empty")
            cameras = Cameras(cam_l0=Camera(image=raw.cameras["CAM_FRONT_LEFT"]),
                              cam_f0=Camera(image=raw.cameras["CAM_FRONT"]),
                              cam_r0=Camera(image=raw.cameras["CAM_FRONT_RIGHT"]),
                              cam_l1=Camera(), cam_l2=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera())
            feature = self._navsim_camera_builder()._get_camera_feature(AgentInput([], [cameras], []))
            implementation = "actual TransfuserFeatureBuilder._get_camera_feature; camera-only view"
        else:
            feature = torch.from_numpy(candidate_camera_feature(raw)["candidate_camera_feature"])
            implementation = "existing candidate_camera_feature: full views PIL bilinear, independent 256/512/256 widths"
        self._validate_tensor(feature, (3, 256, 1024), unit_interval=True)
        return NativeBuildResult({"camera_feature": feature}, dict(self.config.provenance(),
                                 source_files=raw.provenance["source_files"], cameras=camera_metadata(raw)),
                                 {"algorithm": "pass", "candidate_semantics": "partial", "formal_semantics": "blocked",
                                  "implementation": implementation, "geometry_equivalence": "unverified"})

    def build_lidar(self, observation):
        import torch
        self._check_observation(observation)
        result = candidate_lidar_feature(observation.raw, self.config.geometry)
        feature = torch.from_numpy(result["candidate_lidar_feature"])
        self._validate_tensor(feature, (1, 256, 256), unit_interval=True)
        return NativeBuildResult({"lidar_feature": feature}, dict(self.config.provenance(),
                                 source_files=observation.raw.provenance["source_files"],
                                 decoder=observation.raw.lidar_metadata),
                                 {"algorithm": "pass", "candidate_semantics": "partial", "formal_semantics": "blocked",
                                  "height_status": result["height_status"], "pixel_axes": result["pixel_axes"],
                                  "height_kept_count": result["height_kept_count"], "in_grid_count": result["in_grid_count"]})

    @staticmethod
    def _validate_tensor(tensor, shape, unit_interval=False):
        import torch
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise AdapterError("invalid_tensor", {"expected": shape, "actual": list(tensor.shape), "dtype": str(tensor.dtype)})
        if unit_interval and not ((tensor >= 0).all() and (tensor <= 1).all()):
            raise AdapterError("invalid_tensor", "sensor values outside [0,1]")

    def compute_features(self, observation):
        return self.build_features(observation).tensors

    def build_features(self, observation):
        self._check_observation(observation)
        # Fail before returning a misleading two-key partial training feature dict.
        status = self.status_builder.build_result(observation)
        camera, lidar = self.build_camera(observation), self.build_lidar(observation)
        return NativeBuildResult(dict(camera.tensors, **lidar.tensors, **status.tensors),
                                 {"policy": self.config.provenance(), "current_token": observation.raw.ref.token,
                                  "camera": camera.provenance, "lidar": lidar.provenance, "status": status.provenance},
                                 {"algorithm": "pass", "engineering": "pass", "formal_semantics": "unverified"})


class B2DTrajectoryTargetBuilder:
    """Complete trajectory-only candidate path; no hidden map/agent/PDM labels."""
    def __init__(self, reader, config):
        if not isinstance(config, NativeBuilderConfig):
            raise AdapterError("invalid_config", "explicit NativeBuilderConfig required")
        self.reader, self.config = reader, config
        self.candidate_builder = CandidateTrajectoryTargetBuilder(reader, config.geometry)

    def build_candidate(self, current, terminal_padding=False):
        import torch
        result = self.candidate_builder.build(current, full_scene=False, target_only=self.config.mode == "engineering",
                                               terminal_padding=terminal_padding)
        tensor = torch.from_numpy(result.candidate_trajectory.copy())
        B2DFeatureBuilder._validate_tensor(tensor, (8, 3))
        return NativeBuildResult({"candidate_trajectory": tensor}, dict(self.config.provenance(),
                                 trajectory_source=result.provenance,
                                 current_token=current.token, future_frame_ids=[r.frame_id for r in result.window.trajectory_future]),
                                 dict(result.validation, algorithm="pass", candidate_semantics="partial",
                                      formal_semantics="blocked", map_dependency=False))

    def compute_targets(self, current):
        return self.build_targets(current).tensors

    def build_targets(self, current, terminal_padding=False):
        result = self.build_candidate(current, terminal_padding=terminal_padding)
        if self.config.mode == "engineering":
            return NativeBuildResult({"trajectory": result.tensors["candidate_trajectory"]},
                                     dict(result.provenance, target_status="engineering supervision; NOT historically certified GT"),
                                     dict(result.validation, engineering="pass", formal_semantics="unverified"))
        raise AdapterError("uncertified_target", {"clock": result.provenance["geometry_evidence"]["clock_status"],
                           "reference": result.provenance["geometry_evidence"]["reference_status"],
                           "candidate_entry": "build_candidate; NOT formal trajectory GT"})
