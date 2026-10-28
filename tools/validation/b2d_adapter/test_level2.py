"""Opt-in actual NAVSIM tests in the model environment, not mocked Builders.

Run B2D_RUN_LEVEL2_TESTS=1 with B2D_LAZ_DECODER_PYTHON pointing at an
existing independent LAZ environment. Native-data control reads ONE trusted log.
"""

from dataclasses import replace
import os
from pathlib import Path
import unittest

import numpy as np

from tools.b2d_adapter import CandidateConfig, FrameReader, RouteIndex, AdapterError
from tools.b2d_adapter.object_adapter import B2DObjectAdapter, candidate_causal_state
from tools.b2d_adapter.navsim_bridge import to_partial_navsim


ROOT = Path("/data/datasets/Bench2Drive-V0.0.4/v1")
ROUTES = ["SignalizedJunctionRightTurn_Town12_Route5211_Weather1_02-11-08-13-13",
          "SignalizedJunctionLeftTurn_Town12_Route4013_Weather1_02-13-13-37-52"]


@unittest.skipUnless(os.environ.get("B2D_RUN_LEVEL2_TESTS") == "1", "actual NAVSIM tests require explicit opt-in")
class ActualBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This historical validator is required only by the explicitly enabled
        # Level2 integration suite. Missing implementation must still fail when
        # enabled, rather than preventing unrelated CPU tests from collecting.
        from importlib import import_module
        validator = import_module('tools.validation.b2d_adapter.validate_level2')
        cls.validate_route = staticmethod(validator.validate_route)
        cls.validate_navsim_reference = staticmethod(validator.validate_navsim_reference)
        cls.decoder = os.environ.get("B2D_LAZ_DECODER_PYTHON")
        if not cls.decoder:
            raise unittest.SkipTest("B2D_LAZ_DECODER_PYTHON missing; do not splice environments")
        cls.root = Path(os.environ.get("B2D_DATA_ROOT", str(ROOT.parent))) / "v1"
        if not all((cls.root / r / "anno").is_dir() for r in ROUTES):
            raise unittest.SkipTest("specified real R0/R2 not mounted")
        cls.reports = [cls.validate_route(cls.root / route, cls.decoder) for route in ROUTES]

    def test_real_object_types_and_window(self):
        for report in self.reports:
            self.assertTrue(all(report["source_type_checks"].values()))
            self.assertEqual(report["history_frame_ids"], [65, 70, 75, 80])
            self.assertEqual(report["scene_future_frame_ids"], list(range(85, 131, 5)))
            self.assertEqual(report["trajectory_frame_ids"], list(range(85, 121, 5)))
            self.assertEqual(report["level2a"], "partial")

    def test_actual_sensor_subpaths_numeric_policy(self):
        for report in self.reports:
            camera, lidar = report["level2b"]["camera"], report["level2b"]["lidar"]
            self.assertEqual(camera["output"]["shape"], [3, 256, 1024])
            self.assertEqual(lidar["output"]["shape"], [1, 256, 256])
            self.assertEqual(camera["same_original_policy_max_error"], 0)
            self.assertEqual(lidar["same_candidate_policy_max_error"], 0)
            self.assertGreater(camera["adapter_vs_original_mae"], 0.01)
            self.assertTrue(report["source_integrity"]["size_mtime_unchanged"])
            self.assertEqual(report["model_environment"]["foreign_site_packages_in_sys_path"], [])

    def test_actual_trajectory_and_real_agent_labels(self):
        for sign, report in zip((-1, 1), self.reports):
            target = report["level2b"]["trajectory"]
            self.assertEqual(target["output"]["shape"], [8, 3])
            self.assertEqual(target["output"]["dtype"], "float64")
            self.assertLess(target["candidate_cast_max_error"], 1e-5)
            self.assertGreater(sign * target["poses"][-1][1], 0)
            self.assertGreater(sign * target["poses"][-1][2], 0)
            agents = report["level2b"]["agent_target"]
            self.assertGreater(agents["valid_count"], 0)
            self.assertEqual(agents["agent_states"]["shape"], [30, 5])
            self.assertEqual(agents["agent_labels"]["dtype"], "bool")

    def test_missing_dependencies_reject_original_complete_paths(self):
        for report in self.reports:
            self.assertEqual(report["level2c"], "blocked")
            self.assertIsNone(report["level2b"]["ego_status"]["command"])
            self.assertIsNone(report["level2b"]["bev_map_target"]["map_api"])
            self.assertEqual(report["actual_failures"]["map_factory"]["type"], "AssertionError")
            self.assertEqual(report["actual_failures"]["complete_feature_builder"]["type"], "TypeError")
            self.assertEqual(report["actual_failures"]["complete_feature_builder"]["locations"][-1]["line"], 47)
            self.assertEqual(report["actual_failures"]["complete_target_builder"]["type"], "AttributeError")
            self.assertEqual(report["actual_failures"]["complete_target_builder"]["locations"][-1]["line"], 231)
            self.assertIn("incomplete_navsim_contract", report["actual_failures"]["strict_contract_gate"]["message"])

    def test_feature_causality_and_pose_state_separation(self):
        for report in self.reports:
            self.assertTrue(all(report["causality"].values()))
            state = report["level2b"]["ego_status"]["state_provenance"]
            self.assertEqual(state["source_frame_ids"], [78, 79, 80])
            self.assertTrue(state["causal"])
            self.assertEqual(state["velocity_lag_s"], .05)
            self.assertFalse(report["training_or_network_forward_run"])
        # Transparent read observer verifies only past/current access by estimator.
        reader = FrameReader(RouteIndex(self.root / ROUTES[0]))
        recorded, original = [], reader.read
        def read(ref, *args, **kwargs):
            recorded.append(ref.frame_id)
            return original(ref, *args, **kwargs)
        reader.read = read
        candidate_causal_state(reader, reader.index.ref(80), CandidateConfig())
        self.assertEqual(recorded, [78, 79, 80])

    def test_calibration_and_candidate_contract_guards(self):
        for report in self.reports:
            for cal in report["calibration"].values():
                self.assertTrue(cal["candidate_only"])
                self.assertLess(cal["orthogonality_max"], 1e-5)
                self.assertAlmostEqual(cal["determinant"], 1, places=5)
                self.assertIn("NOT fabricated zeros", cal["distortion_status"])
        c = CandidateConfig()
        reader = FrameReader(RouteIndex(self.root / ROUTES[0]))
        bundle = B2DObjectAdapter(reader, c, self.decoder).build(reader.index.ref(80))
        with self.assertRaises(AdapterError) as interval:
            to_partial_navsim(bundle, replace(c, target_dt_s=1.0))
        self.assertEqual(interval.exception.code, "navsim_interval_mismatch")
        with self.assertRaises(AdapterError) as policy:
            to_partial_navsim(bundle, replace(c, axle_dx_m=-1.0))
        self.assertEqual(policy.exception.code, "policy_mismatch")
        objects = to_partial_navsim(bundle, c)
        self.assertIsNone(objects.agent_input.cameras[-1].cam_f0.distortion)
        self.assertEqual(objects.agent_input.lidars[-1].lidar_pc.shape[0], 3)
        self.assertFalse(bundle.provenance["certified_ground_truth"])

    def test_native_navsim_reference_full_feature_builder(self):
        root = Path(os.environ.get("NAVSIM_DATA_ROOT", "/run/media/aethelind/ZY-WD/dataset/NAVSIM/dataset"))
        log = root / "navsim_logs/mini/2021.06.09.12.39.51_veh-26_01943_02303.pkl"
        if not log.is_file():
            self.skipTest("one native NAVSIM reference log unavailable")
        report = self.validate_navsim_reference(root, log)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["features"]["status_feature"]["shape"], [8])
        self.assertEqual(report["native_lidar_shape"][0], 6)
        self.assertTrue(report["b2d_navigation_not_filled_from_reference"])


if __name__ == "__main__":
    unittest.main()
