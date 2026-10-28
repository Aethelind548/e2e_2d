"""Bench2Drive engineering adapter for the unmodified official V1 network.

Partial targets are diagnostic only. Full training requires a real BEV provider;
there is deliberately no background-only substitute or loss-weight override.
"""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from .errors import AdapterError
from .native_builders import B2DFeatureBuilder, B2DTrajectoryTargetBuilder, NativeBuildResult
from .native_config import NativeBuilderConfig
from .native_dataset import NativeSample
from .navsim_bridge import candidate_vehicle_boxes, is_b2d_vehicle


V1_ROOT = Path(__file__).resolve().parents[2] / "repositories/DiffusionDrive"
BEV_CLASSES = ("background", "road", "walkway", "centerline", "static_object", "vehicle", "pedestrian")


def official_v1_module(name):
    module = importlib.import_module("navsim.agents.diffusiondrive." + name)
    expected = V1_ROOT / "navsim/agents/diffusiondrive" / (name + ".py")
    if Path(module.__file__).resolve() != expected.resolve():
        raise AdapterError("wrong_v1_source", {"actual": module.__file__, "expected": str(expected),
                           "remedy": "start a fresh process with repositories/DiffusionDrive first in PYTHONPATH"})
    return module


def v1_native_config(camera_policy="b2d-proportional-crop-v1"):
    # Preserve the existing accepted 10Hz/Lincoln policy and its evidence limits.
    return NativeBuilderConfig(camera_policy, mode="engineering",
                               navigation_policy="b2d-near-bearing-v1")


def validate_v1_config(native_config, model_config):
    if not isinstance(native_config, NativeBuilderConfig) or native_config.mode != "engineering":
        raise AdapterError("invalid_v1_config", "explicit engineering NativeBuilderConfig required")
    if native_config.camera_policy not in ("navsim-pixel-crop-v1", "b2d-proportional-crop-v1"):
        raise AdapterError("invalid_v1_config", "V1 requires the versioned B2D crop or historical NAVSIM crop")
    official = official_v1_module("transfuser_config").TransfuserConfig
    if not isinstance(model_config, official):
        raise AdapterError("invalid_v1_config", "official V1 TransfuserConfig required")
    c = native_config.geometry
    pairs = {
        "lidar_min_x": c.lidar_min_xy_m, "lidar_max_x": c.lidar_max_xy_m,
        "lidar_min_y": c.lidar_min_xy_m, "lidar_max_y": c.lidar_max_xy_m,
        "pixels_per_meter": c.lidar_pixels_per_m, "hist_max_per_pixel": c.lidar_count_clip,
        "lidar_split_height": c.lidar_split_height_m, "max_height_lidar": c.lidar_max_height_m,
        "use_ground_plane": False, "lidar_seq_len": 1, "num_bounding_boxes": 30,
        "camera_width": 1024, "camera_height": 256,
        "bev_pixel_size": .25, "bev_pixel_height": 128, "bev_pixel_width": 256,
        "num_bev_classes": 7,
    }
    mismatches = {key: (getattr(model_config, key), expected) for key, expected in pairs.items()
                  if getattr(model_config, key) != expected}
    if model_config.trajectory_sampling.num_poses != 8 or model_config.trajectory_sampling.interval_length != .5:
        mismatches["trajectory_sampling"] = "8 poses at .5s required"
    if model_config.bev_semantic_classes != official().bev_semantic_classes:
        mismatches["bev_semantic_classes"] = "official seven-class mapping required"
    if mismatches:
        raise AdapterError("v1_contract_mismatch", mismatches)
    payload = {"schema": "b2d-diffusiondrive-v1-full-il-v1", "native_policy": native_config.provenance(),
               "tensor_contract": pairs, "bev_classes": BEV_CLASSES,
               "model_namespace": "navsim.agents.diffusiondrive"}
    payload["contract_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


class DiffusionDriveV1FeatureBuilder(B2DFeatureBuilder):
    def __init__(self, reader, native_config, model_config, lidar_decoder_python=None):
        self.model_config = model_config
        self.contract = validate_v1_config(native_config, model_config)
        super().__init__(reader, native_config, lidar_decoder_python)

    def _navsim_camera_builder(self):
        return official_v1_module("transfuser_features").TransfuserFeatureBuilder(self.model_config)

    def build_features(self, observation):
        result = super().build_features(observation)
        result.provenance["v1_contract"] = self.contract
        return result


class DiffusionDriveV1TargetBuilder:
    """Official vehicle selection/padding plus existing SE2 trajectory labels.

    A BEV provider must implement build_targets(current, reader, native_config,
    model_config) -> NativeBuildResult with one int64 bev_semantic_map tensor.
    Required provenance is checked below. The provider owns real map/actor label
    construction and its validation; B2DOpenDriveBEVBuilder supplies Town maps.
    """
    def __init__(self, reader, native_config, model_config, bev_provider=None):
        self.reader, self.config, self.model_config = reader, native_config, model_config
        self.contract = validate_v1_config(native_config, model_config)
        self.trajectory_builder = B2DTrajectoryTargetBuilder(reader, native_config)
        self.official_builder = official_v1_module("transfuser_features").TransfuserTargetBuilder(model_config)
        self.bev_provider = bev_provider

    def build_agent_targets(self, raw):
        self.reader.index.check_ref(raw.ref)
        if raw.ego_box.get("type_id") != self.config.geometry.axle_vehicle_type:
            raise AdapterError("reference_vehicle_mismatch", raw.ref.token)
        ids = set()
        for box in raw.annotation["bounding_boxes"]:
            if not isinstance(box, dict) or not isinstance(box.get("class"), str):
                raise AdapterError("invalid_annotation", "every bounding box must have a class")
            if box["class"] != "vehicle":
                continue
            if not is_b2d_vehicle(box) or "id" not in box:
                raise AdapterError("invalid_vehicle_box", "vehicle type_id and actor id required")
            actor_id = str(box["id"])
            if actor_id in ids:
                raise AdapterError("invalid_vehicle_box", "duplicate actor id " + actor_id)
            ids.add(actor_id)
            try:
                rotation = np.asarray(box["rotation"], dtype=np.float64)
                if rotation.shape != (3,) or not np.isfinite(rotation).all():
                    raise ValueError("finite rotation3 required")
                if np.max(np.abs(rotation[:2])) > self.config.geometry.max_planar_tilt_deg:
                    raise ValueError("vehicle tilt exceeds planar box policy")
            except (KeyError, TypeError, ValueError) as exc:
                raise AdapterError("invalid_vehicle_box", str(exc)) from exc
        view = candidate_vehicle_boxes(raw, self.config.geometry)
        if not np.isfinite(view.boxes).all():
            raise AdapterError("invalid_vehicle_box", "nonfinite transformed boxes")
        states, labels = self.official_builder._compute_agent_targets(view)
        B2DFeatureBuilder._validate_tensor(states, (30, 5))
        return NativeBuildResult({"agent_states": states, "agent_labels": labels},
                                 dict(view.provenance, current_token=raw.ref.token, v1_contract=self.contract,
                                      valid_count=int(labels.sum()),
                                      selection="official vehicle-only XY bounds, nearest 30, false-slot padding"),
                                 {"algorithm": "pass", "engineering": "pass", "formal_semantics": "unverified"})

    def build_partial_targets(self, current, terminal_padding=False):
        """Explicit diagnostic result, never returned by compute_targets."""
        self.reader.index.check_ref(current)
        agents = self.build_agent_targets(self.reader.read(current))
        trajectory = self.trajectory_builder.build_targets(current, terminal_padding=terminal_padding)
        return NativeBuildResult(dict(trajectory.tensors, **agents.tensors),
                                 {"current_token": current.token, "v1_contract": self.contract,
                                  "trajectory": trajectory.provenance, "agents": agents.provenance,
                                  "complete_training_targets": False, "missing_targets": ["bev_semantic_map"]},
                                 {"engineering": "pass", "full_il": "blocked", "formal_semantics": "unverified"})

    def build_targets(self, current, terminal_padding=False):
        self.reader.index.check_ref(current)
        if self.bev_provider is None:
            raise AdapterError("missing_bev_semantic_map", "real road/walkway/centerline/static/vehicle/pedestrian BEV provider required; full V1 training blocked")
        partial = self.build_partial_targets(current, terminal_padding=terminal_padding)
        bev = self.bev_provider.build_targets(current, self.reader, self.config, self.model_config)
        self._validate_bev(bev, current)
        return NativeBuildResult(dict(partial.tensors, **bev.tensors),
                                 dict(partial.provenance, bev=bev.provenance, complete_training_targets=True,
                                      missing_targets=[]),
                                 {"engineering": "pass", "full_il": "complete_targets", "formal_semantics": "unverified"})

    def _validate_bev(self, result, current):
        import torch
        if not isinstance(result, NativeBuildResult) or set(result.tensors) != {"bev_semantic_map"}:
            raise AdapterError("invalid_bev_target", "NativeBuildResult with exactly bev_semantic_map required")
        t = result.tensors["bev_semantic_map"]
        if not isinstance(t, torch.Tensor) or t.dtype != torch.int64 or tuple(t.shape) != (128, 256):
            raise AdapterError("invalid_bev_target", "int64 128x256 class indices required")
        if not ((t >= 0).all() and (t < 7).all()):
            raise AdapterError("invalid_bev_target", "class indices must be 0..6; unknown labels cannot be background")
        p = result.provenance
        class_names = p.get("class_names")
        map_sha = p.get("map_sha256")
        valid_hash = isinstance(map_sha, str) and len(map_sha) == 64 and all(c in "0123456789abcdef" for c in map_sha)
        if (p.get("current_token") != current.token or p.get("contract_sha256") != self.contract["contract_sha256"]
                or not isinstance(class_names, (tuple, list)) or tuple(class_names) != BEV_CLASSES
                or not p.get("map_source") or not valid_hash
                or p.get("annotation_source") != str(self.reader.index.path(current, "anno"))
                or result.validation.get("real_labels_validated") is not True):
            raise AdapterError("unverified_bev_target", "frame/contract/classes/map hash/annotation source and real_labels_validated required")

    def compute_targets(self, current):
        return self.build_targets(current).tensors


class DiffusionDriveV1Dataset:
    """Two-dictionary dataset for full official IL; missing BEV raises."""
    def __init__(self, reader, frame_refs, native_config, model_config,
                 lidar_decoder_python=None, bev_provider=None):
        self.reader, self.refs = reader, tuple(frame_refs)
        for ref in self.refs:
            reader.index.check_ref(ref)
            reader.index.require(ref)
        self.feature_builder = DiffusionDriveV1FeatureBuilder(reader, native_config, model_config, lidar_decoder_python)
        self.target_builder = DiffusionDriveV1TargetBuilder(reader, native_config, model_config, bev_provider)
        self.contract = self.feature_builder.contract

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, item):
        sample = self.build_sample(item)
        return sample.features, sample.targets

    def build_sample(self, item):
        current = self.refs[item]
        # Refuse incomplete supervision before expensive sensor decoding.
        if self.target_builder.bev_provider is None:
            self.target_builder.compute_targets(current)
        obs = self.feature_builder.read_current(current)
        features = self.feature_builder.build_features(obs)
        targets = self.target_builder.build_targets(current)
        return NativeSample(features.tensors, targets.tensors,
                            {"token": current.token, "contract": self.contract,
                             "features": features.provenance, "targets": targets.provenance,
                             "certified_ground_truth": False})

    def diagnose(self, item):
        current = self.refs[item]
        obs = self.feature_builder.read_current(current)
        features = self.feature_builder.build_features(obs)
        partial = self.target_builder.build_partial_targets(current)
        return NativeSample(features.tensors, partial.tensors,
                            {"token": current.token, "contract": self.contract,
                             "features": features.provenance, "targets": partial.provenance,
                             "complete_training_targets": False, "certified_ground_truth": False})
