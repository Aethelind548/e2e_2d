"""Small on-disk synthetic fixtures; no mutation of any real dataset."""

from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import laspy

from tools.b2d_adapter import (AdapterError, CandidateConfig, FrameReader, RouteIndex,
                          TemporalSampler, CandidateTrajectoryTargetBuilder)
from tools.b2d_adapter.camera_adapter import candidate_camera_feature, camera_metadata
from tools.b2d_adapter.coordinate_transform import (ActorPose, actor_to_reference, ego_to_world,
                                               world_to_ego, target_relative, target_to_actor_world, wrap_angle)
from tools.b2d_adapter.lidar_adapter import candidate_lidar_feature
from tools.b2d_adapter.route_index import CAMERAS


def fixture_annotation(i, turn=0):
    # Constant-speed straight or constant-radius REAR-AXLE turn, not Actor turn.
    # CARLA +yaw = right. On a turn Actor velocity need not align with body yaw.
    yaw = turn * i * 0.01
    x = i * 0.1 if not turn else 10 * math.sin(yaw) / turn
    y = 0 if not turn else 10 * (1 - math.cos(yaw)) / turn
    c, s = math.cos(yaw), math.sin(yaw)
    if turn:
        # Rear follows the circle tangent; Actor is in front of that rear point.
        x, y = x + 1.388632877*c, y + 1.388632877*s
    return {"bounding_boxes": [{"class": "ego_vehicle", "type_id": "vehicle.lincoln.mkz_2020",
                                "location": [x, y, 0], "rotation": [0, 0, math.degrees(yaw)],
                                "world2ego": [[c, s, 0, -c*x-s*y], [-s, c, 0, s*x-c*y],
                                              [0, 0, 1, 0], [0, 0, 0, 1]]}],
            "x": -9999, "y": -9999, "theta": yaw + math.pi/2, "speed": 1,
            "command_near": 4, "x_command_near": 11, "y_command_near": 12,
            "x_target": 21, "y_target": 22, "next_command": 1,
            "sensors": {name: {"intrinsic": [[10, 0, 4], [0, 10, 3], [0, 0, 1]]} for name in CAMERAS}}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="b2d-adapter-unit-")
        self.addCleanup(self.temp.cleanup)
        self.route = Path(self.temp.name) / "synthetic-route"
        for sub in ("anno", "lidar", "camera/rgb_front_left", "camera/rgb_front", "camera/rgb_front_right"):
            (self.route / sub).mkdir(parents=True, exist_ok=True)
        for i in range(80):
            self.write_anno(i, fixture_annotation(i))
            # Filename placeholders sufficient for index tests; decode test replaces one.
            (self.route / "lidar" / "{:05d}.laz".format(i)).touch()
            for sub in CAMERAS.values():
                (self.route / sub / "{:05d}.jpg".format(i)).touch()
        self.index = RouteIndex(self.route)
        self.reader = FrameReader(self.index)
        self.config = CandidateConfig()

    def write_anno(self, i, anno):
        with gzip.open(self.route / "anno" / "{:05d}.json.gz".format(i), "wt", encoding="utf-8") as stream:
            json.dump(anno, stream)

    def assert_reason(self, reason, fn):
        with self.assertRaises(AdapterError) as caught:
            fn()
        self.assertEqual(caught.exception.code, reason)

    def test_raw_reader_no_implicit_transforms(self):
        frame = self.reader.read(self.index.ref(20))
        self.assertAlmostEqual(frame.actor_pose.x, 2)
        self.assertEqual(frame.state["x"], -9999)
        self.assertEqual(frame.navigation["command_near"], 4)
        self.assertEqual(frame.navigation["x_target"], 21)
        self.assertFalse(frame.provenance["implicit_transforms"])
        self.assertIsNone(frame.provenance["timestamp_s"])
        self.assertEqual(frame.cameras, {})

    def test_exact_scene_vs_target_window(self):
        w = TemporalSampler().sample(self.index, self.index.ref(20))
        self.assertEqual([r.frame_id for r in w.history], [5, 10, 15, 20])
        self.assertEqual([r.frame_id for r in w.scene_future], list(range(25, 71, 5)))
        self.assertEqual([r.frame_id for r in w.trajectory_future], list(range(25, 61, 5)))
        self.assertEqual(len(w.continuous), 66)
        self.assertTrue(w.provenance["candidate_only"])

    def test_boundary_past(self):
        self.assert_reason("insufficient_past", lambda: TemporalSampler().sample(self.index, self.index.ref(5)))

    def test_boundary_future_and_trajectory_only(self):
        self.assert_reason("insufficient_future", lambda: TemporalSampler().sample(self.index, self.index.ref(35)))
        w = TemporalSampler().sample(self.index, self.index.ref(35), full_scene=False)
        self.assertEqual(w.scene_future, ())
        self.assertEqual(w.trajectory_future[-1].frame_id, 75)

    def test_unsampled_missing_annotation(self):
        (self.route / "anno/00023.json.gz").unlink()
        self.assert_reason("missing_frame", lambda: TemporalSampler().sample(self.index, self.index.ref(20)))

    def test_history_missing_modality(self):
        (self.route / "camera/rgb_front/00010.jpg").unlink()
        self.assert_reason("missing_modality", lambda: TemporalSampler().sample(self.index, self.index.ref(20)))

    def test_future_sensor_not_required(self):
        (self.route / "lidar/00025.laz").unlink()
        TemporalSampler().sample(self.index, self.index.ref(20))

    def test_route_identity(self):
        other = self.route.parent / "other-route"
        self.assert_reason("route_mismatch", lambda: self.reader.read(type(self.index.ref(20))(other, 20)))
        self.assert_reason("route_mismatch", lambda: TemporalSampler().sample(self.index, type(self.index.ref(20))(other, 20)))

    def test_explicit_dataset_relative_route(self):
        index = RouteIndex.from_dataset_root(self.route.parent, self.route.name)
        self.assertEqual(index.route, self.index.route)
        self.assert_reason("invalid_route_name", lambda: RouteIndex.from_dataset_root(self.route.parent, "../outside"))
        self.assert_reason("invalid_route_name", lambda: RouteIndex.from_dataset_root(self.route.parent, str(self.route)))

    def test_timestamp_not_silently_ignored(self):
        self.assert_reason("timestamp_sampling_unimplemented", lambda: TemporalSampler().sample(self.index, self.index.ref(20), timestamps={20: 2}))

    def test_config_rejects_invalid_values_and_allows_candidate_stride(self):
        for changes in ({"source_dt_s": 0}, {"axle_dx_m": float("nan")}, {"history_count": 2.5}):
            self.assert_reason("invalid_config", lambda: replace(self.config, **changes))
        self.assert_reason("unsupported_sampling", lambda: replace(self.config, source_dt_s=0.3))
        c = replace(self.config, source_dt_s=0.05)
        self.assertEqual(c.stride, 10)
        self.assertNotEqual(c.provenance()["policy_sha256"], self.config.provenance()["policy_sha256"])

    def test_se2_roundtrip_and_angle_wrap(self):
        max_error = 0
        for yaw in (-math.pi, -1.1, 0, 2.9, math.pi):
            origin = ActorPose(1200, -3000, yaw)
            for xy in ((0, 0), (5, -3), (-25, 30)):
                rebuilt = world_to_ego(ego_to_world(xy, origin), origin)
                max_error = max(max_error, math.dist(xy, rebuilt))
        self.assertLess(max_error, 1e-9)
        start = ActorPose(0, 0, math.radians(179))
        end = ActorPose(1, 0, math.radians(-179))
        self.assertAlmostEqual(target_relative(start, end)[2], math.radians(-2))

    def test_heading_and_turn_signs(self):
        origin = ActorPose(0, 0, 0)
        left, right = ActorPose(8, -4, -0.7), ActorPose(8, 4, 0.7)
        for future, sign in ((left, 1), (right, -1)):
            target = target_relative(origin, future)
            self.assertGreater(sign * target[1], 0)
            self.assertGreater(sign * target[2], 0)

    def test_offset_change_and_inverse(self):
        start, end = ActorPose(1000, 2000, 1), ActorPose(1004, 2007, 1.5)
        rows = []
        for delta in ((0, 0), (-1.388632877, 0), (-2, 0.1)):
            target = target_relative(actor_to_reference(start, delta), actor_to_reference(end, delta))
            rows.append(target)
            inverse = target_to_actor_world(target, start, delta)
            self.assertLess(math.hypot(inverse.x-end.x, inverse.y-end.y), 1e-9)
            self.assertLess(abs(wrap_angle(inverse.yaw-end.yaw)), 1e-9)
        self.assertNotEqual(rows[0][:2], rows[1][:2])

    def test_candidate_target_straight(self):
        result = CandidateTrajectoryTargetBuilder(self.reader).build(self.index.ref(20))
        self.assertEqual(result.candidate_trajectory.shape, (8, 3))
        self.assertEqual(result.candidate_trajectory.dtype, np.float32)
        np.testing.assert_allclose(result.candidate_trajectory[:, 0], np.arange(1, 9)*0.5, atol=1e-6)
        np.testing.assert_allclose(result.candidate_trajectory[:, 1:], 0, atol=1e-6)
        self.assertFalse(result.provenance["certified_ground_truth"])
        self.assertLess(result.validation["double_roundtrip_max_m"], 1e-9)

    def test_candidate_target_left_and_right_and_matrix(self):
        for turn in (-1, 1):
            for i in range(80):
                self.write_anno(i, fixture_annotation(i, turn))
            result = CandidateTrajectoryTargetBuilder(self.reader).build(self.index.ref(20))
            self.assertTrue((-turn * result.candidate_trajectory[:, 1] > 0).all())
            self.assertTrue((-turn * result.candidate_trajectory[:, 2] > 0).all())
            self.assertLess(result.validation["float32_roundtrip_max_m"], 1e-5)
            a, b = self.reader.read(self.index.ref(20)), self.reader.read(self.index.ref(40))
            matrix_delta = np.asarray(a.ego_box["world2ego"]) @ (np.r_[b.ego_box["location"], 1] - np.r_[a.ego_box["location"], 1])
            np.testing.assert_allclose(matrix_delta[:2], world_to_ego((b.actor_pose.x, b.actor_pose.y), a.actor_pose), atol=1e-9)

    def test_candidate_discontinuity_in_unsampled_frame(self):
        a = fixture_annotation(23)
        a["bounding_boxes"][0]["location"][0] = 9999
        self.write_anno(23, a)
        self.assert_reason("teleport_or_discontinuity", lambda: CandidateTrajectoryTargetBuilder(self.reader).build(self.index.ref(20)))

    def test_candidate_vehicle_and_tilt_and_teleport_gates(self):
        for key, value, reason in (("type_id", "vehicle.other", "reference_vehicle_mismatch"),
                                   ("rotation", [10, 0, 0], "nonplanar_pose")):
            a = fixture_annotation(20)
            a["bounding_boxes"][0][key] = value
            self.write_anno(20, a)
            self.assert_reason(reason, lambda: CandidateTrajectoryTargetBuilder(self.reader).build(self.index.ref(20)))
        a = fixture_annotation(20)
        a["teleport_flag"] = True
        self.write_anno(20, a)
        self.assert_reason("teleport_flag", lambda: CandidateTrajectoryTargetBuilder(self.reader).build(self.index.ref(20)))

    def test_invalid_ego_count_and_nonfinite(self):
        a = fixture_annotation(20)
        a["bounding_boxes"] = []
        self.write_anno(20, a)
        self.assert_reason("invalid_ego_count", lambda: self.reader.read(self.index.ref(20)))
        a = fixture_annotation(20)
        a["bounding_boxes"][0]["location"][2] = float("nan")
        self.write_anno(20, a)
        self.assert_reason("invalid_pose", lambda: self.reader.read(self.index.ref(20)))

    def test_duplicate_numeric_filename_id(self):
        (self.route / "anno/20.json.gz").touch()
        self.assert_reason("duplicate_frame_id", lambda: RouteIndex(self.route))

    def test_corrupt_annotation(self):
        with gzip.open(self.route / "anno/00020.json.gz", "wt") as stream:
            stream.write("not JSON")
        self.assert_reason("annotation_decode_failed", lambda: self.reader.read(self.index.ref(20)))

    def test_sampler_requires_current_annotation(self):
        (self.route / "anno/00020.json.gz").unlink()
        self.assert_reason("missing_frame", lambda: TemporalSampler().sample(self.index, self.index.ref(20)))

    def test_camera_jpeg_roundtrip_and_metadata(self):
        for name, sub in CAMERAS.items():
            color = {"CAM_FRONT_LEFT": (255, 0, 0), "CAM_FRONT": (0, 255, 0), "CAM_FRONT_RIGHT": (0, 0, 255)}[name]
            Image.new("RGB", (16, 9), color).save(self.route / sub / "00020.jpg")
        frame = self.reader.read(self.index.ref(20), cameras=True)
        result = candidate_camera_feature(frame)
        arr = result["candidate_camera_feature"]
        self.assertEqual(arr.shape, (3, 256, 1024))
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue((arr >= 0).all() and (arr <= 1).all())
        self.assertEqual(arr[:, 0, 10].argmax(), 0)
        self.assertEqual(arr[:, 0, 500].argmax(), 1)
        self.assertEqual(arr[:, 0, 1000].argmax(), 2)
        self.assertEqual(camera_metadata(frame)["CAM_FRONT"]["shape"], [9, 16, 3])
        self.assertTrue(result["candidate_only"])

    def test_laz_decode_histogram_axes_heights_and_clip(self):
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [0.001, 0.001, 0.001]
        las = laspy.LasData(header)
        points = np.array([[0, 1, 1]]*8 + [[0, 1, 0.2], [0, 1, 100], [32, -32, 1], [-33, 0, 1]])
        las.x, las.y, las.z = points.T
        las.write(self.route / "lidar/00020.laz")  # Real LAZ codec, synthetic content.
        frame = self.reader.read(self.index.ref(20), lidar=True)
        np.testing.assert_allclose(frame.lidar_xyz, points, atol=0.0005)
        original = frame.lidar_xyz.copy()
        cfg = replace(self.config, axle_dx_m=0)
        result = candidate_lidar_feature(frame, cfg)
        arr = result["candidate_lidar_feature"]
        self.assertEqual(arr.shape, (1, 256, 256))
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(arr[0, 128, 124], 1)  # x=0, y_target=-1: clipped 8/5.
        self.assertAlmostEqual(arr[0, 255, 255], 0.2)  # Inclusive final edge.
        self.assertEqual(result["height_kept_count"], 10)
        self.assertEqual(result["in_grid_count"], 9)
        np.testing.assert_array_equal(frame.lidar_xyz, original)  # No input mutation.
        shifted = candidate_lidar_feature(frame, self.config)["candidate_lidar_feature"]
        self.assertEqual(shifted[0, 133, 124], 1)

    def test_corrupt_camera_and_laz_are_explicit(self):
        self.assert_reason("camera_decode_failed", lambda: self.reader.read(self.index.ref(20), cameras=True))
        self.assert_reason("lidar_decode_failed", lambda: self.reader.read(self.index.ref(20), lidar=True))


if __name__ == "__main__":
    unittest.main()
