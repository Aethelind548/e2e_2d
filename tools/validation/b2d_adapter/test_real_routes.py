"""Opt-in bounded real integration checks, never synthesize missing dataset data.

Set B2D_RUN_REAL_TESTS=1 to read two explicitly selected Routes. The root is
B2D_DATA_ROOT or DEFAULT_B2D_DATA_ROOT. Missing mounted Routes are skipped with
an explicit reason; skipped tests do not count as real-data acceptance.
"""

import math
import os
from pathlib import Path
import unittest

import numpy as np

from tools.b2d_adapter import (CandidateConfig, FrameReader, RouteIndex, TemporalSampler,
                              CandidateTrajectoryTargetBuilder)
from tools.b2d_adapter.camera_adapter import candidate_camera_feature
from tools.b2d_adapter.config import DEFAULT_B2D_DATA_ROOT
from tools.b2d_adapter.coordinate_transform import target_to_actor_world, wrap_angle
from tools.b2d_adapter.lidar_adapter import candidate_lidar_feature
from tools.b2d_adapter.route_index import CAMERAS


@unittest.skipUnless(os.environ.get("B2D_RUN_REAL_TESTS") == "1", "real tests opt-in: B2D_RUN_REAL_TESTS=1")
class RealRouteTests(unittest.TestCase):
    def validate_route(self, name, turn_sign):
        root = Path(os.environ.get("B2D_DATA_ROOT", str(DEFAULT_B2D_DATA_ROOT)))
        route = root / "v1" / name
        if not (route / "anno").is_dir():
            self.skipTest("dataset/Route not mounted: {}".format(route))
        index = RouteIndex.from_dataset_root(root, "v1/" + name)
        reader = FrameReader(index)
        current = index.ref(80)
        config = CandidateConfig()
        window = TemporalSampler(config).sample(index, current)
        # Fingerprint before any annotation/sensor decode, not only after target construction.
        paths = [index.path(r, "anno") for r in window.continuous]
        paths += [index.path(index.ref(i), modality)
                  for i in range(80, 90) for modality in ("anno", "lidar", *CAMERAS)]
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
        target = CandidateTrajectoryTargetBuilder(reader, config).build(current)
        self.assertEqual(target.candidate_trajectory.shape, (8, 3))
        self.assertEqual(target.candidate_trajectory.dtype, np.float32)
        self.assertTrue(np.isfinite(target.candidate_trajectory).all())
        self.assertTrue(target.provenance["candidate_only"])
        self.assertFalse(target.provenance["certified_ground_truth"])
        self.assertEqual([r.frame_id for r in target.window.history], [65, 70, 75, 80])
        self.assertEqual([r.frame_id for r in target.window.scene_future], list(range(85, 131, 5)))
        self.assertEqual([r.frame_id for r in target.window.trajectory_future], list(range(85, 121, 5)))
        self.assertLess(target.validation["float32_roundtrip_max_m"], 1e-5)
        # Use turn endpoint, NOT a universal instantaneous lateral-sign assertion.
        self.assertGreater(turn_sign * target.candidate_trajectory[-1, 1], 0)
        self.assertGreater(turn_sign * target.candidate_trajectory[-1, 2], 0)

        for i in range(80, 90):
            raw = reader.read(index.ref(i), cameras=True, lidar=True)
            for name in CAMERAS:
                self.assertEqual(raw.cameras[name].shape, (900, 1600, 3))
                self.assertEqual(raw.cameras[name].dtype, np.uint8)
                self.assertEqual(np.asarray(raw.sensors[name]["intrinsic"]).shape, (3, 3))
            self.assertEqual(raw.lidar_xyz.ndim, 2)
            self.assertEqual(raw.lidar_xyz.shape[1], 3)
            self.assertGreater(len(raw.lidar_xyz), 0)
            self.assertTrue(np.isfinite(raw.lidar_xyz).all())
            self.assertFalse(raw.provenance["implicit_transforms"])
            if i == 80:
                origin = raw.actor_pose
                for key, arr in (("camera", candidate_camera_feature(raw)["candidate_camera_feature"]),
                                 ("lidar", candidate_lidar_feature(raw, config)["candidate_lidar_feature"])):
                    self.assertEqual(arr.shape, (3, 256, 1024) if key == "camera" else (1, 256, 256))
                    self.assertEqual(arr.dtype, np.float32)
                    self.assertTrue(np.isfinite(arr).all())
                    self.assertTrue((arr >= 0).all() and (arr <= 1).all())
                self.assertIn("x_target", raw.navigation)
        for point, ref in zip(target.candidate_trajectory, target.window.trajectory_future):
            actual = reader.read(ref).actor_pose
            rebuilt = target_to_actor_world(point, origin, (config.axle_dx_m, config.axle_dy_m))
            self.assertLess(math.hypot(actual.x-rebuilt.x, actual.y-rebuilt.y), 1e-5)
            self.assertLess(abs(wrap_angle(actual.yaw-rebuilt.yaw)), 1e-6)
        after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
        self.assertEqual(before, after)

    def test_r0_right_turn(self):
        self.validate_route("SignalizedJunctionRightTurn_Town12_Route5211_Weather1_02-11-08-13-13", -1)

    def test_r2_left_turn(self):
        self.validate_route("SignalizedJunctionLeftTurn_Town12_Route4013_Weather1_02-13-13-37-52", 1)
