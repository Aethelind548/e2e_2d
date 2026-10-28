"""Real B2D engineering-chain integration; never runs a network or optimizer."""

import copy
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

from tools.validation.b2d_adapter.validate_native_builders import ROOT
from tools.b2d_adapter import AdapterError, CandidateConfig, FrameReader, RouteIndex
from tools.b2d_adapter.coordinate_transform import ego_to_world, actor_to_reference
from tools.b2d_adapter.native_builders import B2DFeatureBuilder, B2DTrajectoryTargetBuilder
from tools.b2d_adapter.native_config import NativeBuilderConfig, CAMERA_POLICIES
from tools.b2d_adapter.native_dataset import B2DDataset
from tools.b2d_adapter.native_navigation import near_route_command


def engineering_config(camera=CAMERA_POLICIES[1]):
    return NativeBuilderConfig(camera, mode="engineering", navigation_policy="b2d-near-bearing-v1")


class EngineeringConfigTests(unittest.TestCase):
    def test_modes_and_approved_parameters(self):
        c = engineering_config()
        self.assertTrue(c.provenance()["engineering_assumptions_accepted"])
        self.assertFalse(c.provenance()["certified_ground_truth"])
        self.assertNotEqual(c.provenance()["policy_sha256"], NativeBuilderConfig(c.camera_policy).provenance()["policy_sha256"])
        for geometry in [replace(c.geometry, source_dt_s=.05), replace(c.geometry, axle_dx_m=0),
                         replace(c.geometry, axle_vehicle_type="other")]:
            with self.assertRaises(AdapterError):
                replace(c, geometry=geometry)
        with self.assertRaises(AdapterError):
            replace(c, mode="strict")


@unittest.skipUnless(os.environ.get("B2D_RUN_ENGINEERING_TESTS") == "1", "real engineering tests opt-in")
class RealEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decoder = os.environ.get("B2D_LAZ_DECODER_PYTHON")
        if not cls.decoder:
            raise unittest.SkipTest("independent LAZ decoder required")
        base = Path(os.environ.get("B2D_DATA_ROOT", "/data/datasets/Bench2Drive-V0.0.4"))/"v1"
        cls.routes = [base/"SignalizedJunctionRightTurn_Town12_Route5211_Weather1_02-11-08-13-13",
                      base/"SignalizedJunctionLeftTurn_Town12_Route4013_Weather1_02-13-13-37-52"]
        if not all(r.is_dir() for r in cls.routes):
            raise unittest.SkipTest("R0/R2 unavailable")

    def dataset(self, route=0, policy=CAMERA_POLICIES[1], frames=(80, 100)):
        reader = FrameReader(RouteIndex(self.routes[route]))
        return B2DDataset(reader, [reader.index.ref(t) for t in frames], engineering_config(policy), self.decoder)

    def reason(self, code, call):
        with self.assertRaises(AdapterError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def test_complete_dataset_and_batch_both_routes(self):
        import torch
        from torch.utils.data import DataLoader
        for r in range(2):
            d = self.dataset(r)
            f, t = d[0]
            self.assertEqual(set(f), {"camera_feature", "lidar_feature", "status_feature"})
            self.assertEqual(set(t), {"trajectory"})
            batch_f, batch_t = next(iter(DataLoader(d, batch_size=2, num_workers=0)))
            for key, shape in {"camera_feature": (2,3,256,1024), "lidar_feature": (2,1,256,256), "status_feature": (2,8)}.items():
                self.assertEqual(tuple(batch_f[key].shape), shape)
                self.assertEqual(batch_f[key].dtype, torch.float32)
                self.assertTrue(torch.isfinite(batch_f[key]).all())
            self.assertEqual(tuple(batch_t["trajectory"].shape), (2,8,3))
            self.assertEqual(batch_t["trajectory"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(batch_t["trajectory"]).all())
            self.assertTrue((batch_f["status_feature"][:, :4].sum(1) == 1).all())
            metadata = d.build_sample(0).metadata
            self.assertTrue(metadata["engineering_sample"])
            self.assertFalse(metadata["certified_ground_truth"])
            self.assertEqual(metadata["features"]["status"]["motion_source"]["source_frame_ids"], [78,79,80])
            for p in [metadata["features"]["camera"], metadata["features"]["lidar"], metadata["features"]["status"], metadata["targets"]]:
                self.assertEqual(p["policy_sha256"], d.config.provenance()["policy_sha256"])
                self.assertEqual(p["parameters"]["geometry"]["axle_dx_m"], -1.388632877)

    def test_future_perturbation_cannot_change_features(self):
        import torch
        d = self.dataset()
        ref = d.refs[0]
        original, accessed = d.reader.read, []
        first = d.feature_builder.compute_features(d.feature_builder.read_current(ref))
        def poison_future(ref, *args, **kwargs):
            accessed.append(ref.frame_id)
            raw = original(ref, *args, **kwargs)
            if ref.frame_id > 80:
                raw.state["teleport_flag"] = True
                raw.navigation.clear()
            return raw
        d.reader.read = poison_future
        second = d.feature_builder.compute_features(d.feature_builder.read_current(ref))
        self.assertEqual(set(accessed), {78,79,80})
        self.assertTrue(all(torch.equal(first[k], second[k]) for k in first))
        self.reason("teleport_flag", lambda: d.target_builder.compute_targets(ref))

    def test_camera_policies_hash_and_values(self):
        import torch
        a, b = [self.dataset(policy=p, frames=(80,)) for p in CAMERA_POLICIES]
        fa, ta = a[0]
        fb, tb = b[0]
        self.assertNotEqual(a.config.provenance()["policy_sha256"], b.config.provenance()["policy_sha256"])
        self.assertFalse(torch.equal(fa["camera_feature"], fb["camera_feature"]))
        self.assertTrue(torch.equal(fa["lidar_feature"], fb["lidar_feature"]))
        self.assertTrue(torch.equal(fa["status_feature"], fb["status_feature"]))
        self.assertTrue(torch.equal(ta["trajectory"], tb["trajectory"]))

    def test_target_ignores_history_and_sensor_files(self):
        d = self.dataset()
        for key in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "lidar"):
            d.reader.index.files[key].clear()  # in-memory only
        for i in range(80):
            d.reader.index.files["anno"].pop(i, None)
        result = d.target_builder.build_targets(d.refs[0])
        self.assertEqual(tuple(result.tensors["trajectory"].shape), (8,3))
        sources = result.provenance["trajectory_source"]["source_annotations"]
        self.assertEqual(len(sources), 41)
        self.assertTrue(sources[0].endswith("00080.json.gz"))
        fresh = self.dataset()
        self.assertEqual(tuple(fresh.target_builder.compute_targets(fresh.reader.index.ref(0))["trajectory"].shape), (8,3))
        self.reason("insufficient_past", lambda: fresh.feature_builder.read_current(fresh.reader.index.ref(0)))

    def test_real_missing_navigation_and_blocked_policy(self):
        d = self.dataset()
        observation = d.feature_builder.read_current(d.refs[0])
        observation.raw.navigation.pop("x_command_near")
        self.reason("missing_navigation", lambda: d.feature_builder.compute_features(observation))
        blocked = replace(d.config, navigation_policy="blocked")
        builder = B2DFeatureBuilder(d.reader, blocked, self.decoder)
        self.reason("unresolved_navigation", lambda: builder.compute_features(builder.read_current(d.refs[0])))

    def test_navigation_geometry_and_no_integer_fallback(self):
        d = self.dataset()
        raw = d.reader.read(d.refs[0])
        c = d.config.geometry
        origin = actor_to_reference(raw.actor_pose, (c.axle_dx_m, c.axle_dy_m))
        # Synthetic geometry only tests category convention; no fake Dataset inputs.
        for carla_xy, category in [((10,-5),0), ((10,0),1), ((10,5),2)]:
            x,y = ego_to_world(carla_xy, origin)
            probe = copy.deepcopy(raw)
            probe.navigation.update(x_command_near=x, y_command_near=y, command_near=4)
            cmd, p = near_route_command(probe, d.config)
            self.assertEqual(int(cmd.argmax()), category)
            self.assertFalse(p["future_gt_used"])
        for point in [(origin.x,origin.y), (float("nan"),0)]:
            probe.navigation.update(x_command_near=point[0], y_command_near=point[1])
            self.reason("invalid_navigation", lambda: near_route_command(probe,d.config))

    def test_missing_frames_cross_route_and_vehicle_rejected(self):
        d = self.dataset()
        self.reason("route_mismatch", lambda: d.target_builder.compute_targets(RouteIndex(self.routes[1]).ref(80)))
        self.reason("insufficient_future", lambda: d.target_builder.compute_targets(d.reader.index.ref(148)))
        d.reader.index.files["anno"].pop(97)
        self.reason("missing_frame", lambda: d[0])
        fresh = self.dataset()
        original = fresh.reader.read
        def wrong_vehicle(ref,*args,**kwargs):
            raw = original(ref,*args,**kwargs)
            raw.ego_box["type_id"] = "vehicle.other"
            return raw
        fresh.reader.read = wrong_vehicle
        self.reason("reference_vehicle_mismatch",lambda: fresh[0])

    def test_corrupt_files_and_nonfinite_pose_rejected(self):
        d = self.dataset()
        # Corrupt scratch files, never dataset bytes.
        with tempfile.TemporaryDirectory(prefix="b2d-corrupt-") as scratch:
            bad = Path(scratch)/"bad"
            bad.touch()
            d.reader.index.files["anno"][80] = bad
            self.reason("annotation_decode_failed",lambda:d[0])
            fresh = self.dataset()
            fresh.reader.index.files["CAM_FRONT"][80] = bad
            self.reason("camera_decode_failed",lambda:fresh[0])
            fresh = self.dataset()
            fresh.reader.index.files["lidar"].pop(80)
            self.reason("missing_modality",lambda:fresh[0])
        fresh = self.dataset()
        original = fresh.reader.read
        def nonfinite(ref,*args,**kwargs):
            raw = original(ref,*args,**kwargs)
            raw.ego_box["rotation"][0] = float("nan")
            return raw
        fresh.reader.read = nonfinite
        self.reason("invalid_pose",lambda:fresh.target_builder.compute_targets(fresh.refs[0]))

    def test_decoder_source_is_not_an_untracked_substitute(self):
        d = self.dataset()
        with tempfile.TemporaryDirectory(prefix="b2d-source-") as scratch:
            bad = Path(scratch)/"different.laz"
            bad.touch()
            d.reader.index.files["lidar"][80] = bad
            # Worker must not silently replace an indexed path with the original.
            self.reason("decoder_source_mismatch",lambda:d[0])


if __name__ == "__main__":
    unittest.main()
