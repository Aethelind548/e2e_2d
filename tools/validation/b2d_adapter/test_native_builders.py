"""New native contracts/negative cases; existing 27+7 tests remain unchanged."""

import copy
from dataclasses import replace
import os
from pathlib import Path
import unittest

from tools.b2d_adapter import CandidateConfig, FrameReader, RouteIndex, AdapterError
from tools.b2d_adapter.native_config import NativeBuilderConfig, CAMERA_POLICIES
from tools.b2d_adapter.native_builders import B2DFeatureBuilder, B2DTrajectoryTargetBuilder
from tools.b2d_adapter.native_dataset import B2DDataset


class NativeConfigTests(unittest.TestCase):
    def test_camera_choice_required_and_unknown_rejected(self):
        with self.assertRaises(TypeError):
            NativeBuilderConfig()
        with self.assertRaises(AdapterError):
            NativeBuilderConfig("automatic")

    def test_version_hash_and_no_certification_override(self):
        configs = [NativeBuilderConfig(policy) for policy in CAMERA_POLICIES]
        self.assertEqual(len({c.provenance()["policy_sha256"] for c in configs}), len(CAMERA_POLICIES))
        self.assertTrue(all(not c.provenance()["certified_ground_truth"] for c in configs))
        with self.assertRaises(TypeError):
            NativeBuilderConfig(CAMERA_POLICIES[0], certified_ground_truth=True)
        with self.assertRaises(AdapterError):
            NativeBuilderConfig(CAMERA_POLICIES[0], geometry=None)

    def test_source_dt_configurable_target_contract_fixed(self):
        config = NativeBuilderConfig(CAMERA_POLICIES[0], replace(CandidateConfig(), source_dt_s=.05))
        self.assertEqual(config.geometry.stride, 10)
        with self.assertRaises(AdapterError):
            NativeBuilderConfig(CAMERA_POLICIES[0], replace(CandidateConfig(), trajectory_count=7))
        with self.assertRaises(AdapterError):
            NativeBuilderConfig(CAMERA_POLICIES[0], replace(CandidateConfig(), target_dt_s=1))


@unittest.skipUnless(os.environ.get("B2D_RUN_NATIVE_TESTS") == "1", "native real tests opt-in: B2D_RUN_NATIVE_TESTS=1")
class RealNativeBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decoder = os.environ.get("B2D_LAZ_DECODER_PYTHON")
        if not cls.decoder:
            raise unittest.SkipTest("explicit independent LAZ decoder required")
        base = Path(os.environ.get("B2D_DATA_ROOT", "/data/datasets/Bench2Drive-V0.0.4"))/"v1"
        cls.routes = [base/"SignalizedJunctionRightTurn_Town12_Route5211_Weather1_02-11-08-13-13",
                      base/"SignalizedJunctionLeftTurn_Town12_Route4013_Weather1_02-13-13-37-52"]
        if not all(p.is_dir() for p in cls.routes):
            raise unittest.SkipTest("R0/R2 unavailable")
        cls.configs = [NativeBuilderConfig(p) for p in CAMERA_POLICIES]
        cls.samples = []
        for route in cls.routes:
            reader = FrameReader(RouteIndex(route))
            cls.samples.append([B2DDataset(reader, [reader.index.ref(80)], c, cls.decoder).diagnose(0) for c in cls.configs])

    def assert_reason(self, reason, call):
        with self.assertRaises(AdapterError) as caught:
            call()
        self.assertEqual(caught.exception.code, reason)

    def test_diagnostic_sample_never_training_tuple_or_gt(self):
        for samples in self.samples:
            for sample in samples:
                self.assertEqual(set(sample.sensor_features), {"camera_feature", "lidar_feature"})
                self.assertEqual(set(sample.candidate_targets), {"candidate_trajectory"})
                self.assertFalse(sample.metadata["training_sample"])
                self.assertFalse(sample.metadata["policy"]["certified_ground_truth"])
                self.assertEqual(sample.blockers["features"]["code"], "unresolved_navigation")
                self.assertEqual(sample.blockers["targets"]["code"], "uncertified_target")

    def test_both_camera_versions_and_target_are_reproducible(self):
        import torch
        for sign, samples in zip((-1, 1), self.samples):
            self.assertFalse(torch.equal(samples[0].sensor_features["camera_feature"], samples[1].sensor_features["camera_feature"]))
            self.assertTrue(torch.equal(samples[0].candidate_targets["candidate_trajectory"], samples[1].candidate_targets["candidate_trajectory"]))
            for sample in samples:
                target = sample.candidate_targets["candidate_trajectory"]
                self.assertEqual(tuple(target.shape), (8, 3))
                self.assertEqual(target.dtype, torch.float32)
                self.assertGreater(sign*float(target[-1, 1]), 0)
                self.assertLess(sample.metadata["target_validation"]["float32_roundtrip_max_m"], 1e-5)
                self.assertFalse(sample.metadata["target_validation"]["map_dependency"])

    def test_strict_builders_and_dataset_reject_real_missing_semantics(self):
        for route in self.routes:
            reader = FrameReader(RouteIndex(route))
            dataset = B2DDataset(reader, [reader.index.ref(80)], self.configs[0], self.decoder)
            self.assertEqual(len(dataset), 1)
            self.assert_reason("unresolved_navigation", lambda: dataset[0])
            self.assert_reason("uncertified_target", lambda: dataset.target_builder.compute_targets(reader.index.ref(80)))

    def test_feature_reader_never_reads_future(self):
        reader = FrameReader(RouteIndex(self.routes[0]))
        original, accessed = reader.read, []
        def observed(ref, *args, **kwargs):
            accessed.append(ref.frame_id)
            return original(ref, *args, **kwargs)
        reader.read = observed
        observation = B2DFeatureBuilder(reader, self.configs[0], self.decoder).read_current(reader.index.ref(80))
        self.assertEqual(set(accessed), {78, 79, 80})
        self.assertEqual(observation.motion.provenance["source_frame_ids"], [78, 79, 80])

    def test_target_boundaries_missing_frame_and_cross_route(self):
        reader = FrameReader(RouteIndex(self.routes[0]))
        builder = B2DTrajectoryTargetBuilder(reader, self.configs[0])
        self.assert_reason("insufficient_past", lambda: builder.build_candidate(reader.index.ref(0)))
        self.assert_reason("insufficient_future", lambda: builder.build_candidate(reader.index.ref(148)))
        self.assert_reason("route_mismatch", lambda: builder.build_candidate(RouteIndex(self.routes[1]).ref(80)))
        # Remove only in-memory index entries, never actual files.
        reader.index.files["anno"].pop(97)
        self.assert_reason("missing_frame", lambda: builder.build_candidate(reader.index.ref(80)))

    def test_invalid_candidate_target_and_current_teleport_rejected(self):
        reader = FrameReader(RouteIndex(self.routes[0]))
        original = reader.read
        def invalid(ref, *args, **kwargs):
            raw = original(ref, *args, **kwargs)
            if ref.frame_id == 97:
                raw.state["teleport_flag"] = True  # In-memory negative target probe only.
            return raw
        reader.read = invalid
        self.assert_reason("teleport_flag", lambda: B2DTrajectoryTargetBuilder(reader, self.configs[0]).build_candidate(reader.index.ref(80)))
        def bad_current(ref, *args, **kwargs):
            raw = original(ref, *args, **kwargs)
            if ref.frame_id == 79:
                raw.state["teleport_flag"] = True
            return raw
        reader.read = bad_current
        self.assert_reason("teleport_flag", lambda: B2DFeatureBuilder(reader, self.configs[0], self.decoder).read_current(reader.index.ref(80)))

    def test_observation_policy_mismatch_and_missing_camera(self):
        reader = FrameReader(RouteIndex(self.routes[0]))
        first, second = [B2DFeatureBuilder(reader, c, self.decoder) for c in self.configs]
        observation = first.read_current(reader.index.ref(80))
        self.assert_reason("policy_mismatch", lambda: second.build_camera(observation))
        missing = copy.deepcopy(observation)
        missing.raw.cameras.pop("CAM_FRONT")
        self.assert_reason("invalid_camera", lambda: first.build_camera(missing))


if __name__ == "__main__":
    unittest.main()
