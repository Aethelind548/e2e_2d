"""Explicit bounded sample list; strict diagnostics vs engineering two-dict batches."""

from dataclasses import dataclass

from .errors import AdapterError
from .native_builders import B2DFeatureBuilder, B2DTrajectoryTargetBuilder


@dataclass
class NativeDiagnosticSample:
    sensor_features: dict
    candidate_targets: dict
    candidate_motion: object
    metadata: dict
    blockers: dict


@dataclass
class NativeSample:
    features: dict
    targets: dict
    metadata: dict


class B2DDataset:
    def __init__(self, reader, frame_refs, config, lidar_decoder_python=None):
        self.reader, self.refs = reader, tuple(frame_refs)
        for ref in self.refs:
            reader.index.check_ref(ref)
            reader.index.require(ref)
        self.feature_builder = B2DFeatureBuilder(reader, config, lidar_decoder_python)
        self.target_builder = B2DTrajectoryTargetBuilder(reader, config)
        self.config = config

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, item):
        sample = self.build_sample(item)
        return sample.features, sample.targets

    def build_sample(self, item):
        """Traceable result; __getitem__ keeps the model's exact two-dict contract.

        Features are completed before target annotations are read. No future
        cache or full-Scene object is passed to the FeatureBuilder.
        """
        current = self.refs[item]
        observation = self.feature_builder.read_current(current)
        features = self.feature_builder.build_features(observation)
        targets = self.target_builder.build_targets(current)
        return NativeSample(features.tensors, targets.tensors,
                            {"token": current.token, "policy": self.config.provenance(),
                             "features": features.provenance, "targets": targets.provenance,
                             "features_validation": features.validation, "targets_validation": targets.validation,
                             "engineering_sample": self.config.mode == "engineering", "certified_ground_truth": False})

    def diagnose(self, item):
        if self.config.mode != "strict":
            raise AdapterError("invalid_mode", "diagnose is the preserved v0.1 strict path; use build_sample for engineering")
        current = self.refs[item]
        observation = self.feature_builder.read_current(current)
        camera = self.feature_builder.build_camera(observation)
        lidar = self.feature_builder.build_lidar(observation)
        target = self.target_builder.build_candidate(current)
        motion = self.feature_builder.status_builder.diagnose_motion(observation)
        try:
            self.feature_builder.compute_features(observation)
        except AdapterError as exc:
            blocker = {"code": exc.code, "detail": exc.detail}
        else:
            raise AssertionError("v0.1 scheme C must reject complete features")
        return NativeDiagnosticSample(dict(camera.tensors, **lidar.tensors), target.tensors, motion,
                                      {"token": current.token, "policy": self.config.provenance(),
                                       "camera": camera.provenance, "lidar": lidar.provenance,
                                       "target": target.provenance, "target_validation": target.validation,
                                       "training_sample": False},
                                      {"features": blocker, "targets": {"code": "uncertified_target", "clock_and_reference": "candidate"}})


B2DNativeDataset = B2DDataset
