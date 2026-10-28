"""V1 contract and geometry unit tests; synthetic fixtures are NOT training GT.

Real integration is opt-in and fails (does not skip) when enabled but unavailable.
"""
import copy
from dataclasses import replace
import gzip
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "repositories/DiffusionDrive"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.b2d_adapter import AdapterError, FrameReader, RouteIndex
from tools.b2d_adapter.coordinate_transform import ego_to_world
from tools.b2d_adapter.lidar_adapter import candidate_lidar_xyz
from tools.b2d_adapter.native_builders import B2DObservation, NativeBuildResult
from tools.b2d_adapter.v1_builders import (BEV_CLASSES, DiffusionDriveV1Dataset,
    DiffusionDriveV1FeatureBuilder, DiffusionDriveV1TargetBuilder, official_v1_module,
    validate_v1_config, v1_native_config)


class V1ContractTests(unittest.TestCase):
    def terminal_fixture(self, status='Completed', timestamps=True):
        old = self.reader.index.route
        route = old.with_name('Unit_Town01_Route1_Weather0_02-01-00-00-00')
        old.rename(route)
        for i in range(43):
            p = route/'anno'/('%05d.json.gz'%i)
            with gzip.open(p,'rt') as stream:
                a=json.load(stream)
            a['bounding_boxes'][0]['location']=[i*.1 if i<20 else 2.+(i-20)*.0005,0.,0.]
            a['bounding_boxes'][0]['rotation']=[0.,0.,179.8+i*.01]
            a['speed']=1. if i<20 else .005
            if timestamps:
                a['timestamp_s']=i*.1
            with gzip.open(p,'wt') as stream:
                json.dump(a,stream)
        result={'_checkpoint':{'records':[dict(route_id='RouteScenario_1_rep0',status=status,
                                             scores={'score_route':100},infractions={})]}}
        (route/'result.json').write_text(json.dumps(result))
        self.reader=FrameReader(RouteIndex(route))
        self.builder=DiffusionDriveV1TargetBuilder(self.reader,self.native,self.model)

    def test_terminal_padding_keeps_real_points_and_repeats_true_endpoint(self):
        self.terminal_fixture()
        current=self.reader.index.ref(9)
        self.reason('insufficient_future',lambda:self.builder.trajectory_builder.build_targets(current))
        result=self.builder.trajectory_builder.build_targets(current,terminal_padding=True)
        tensor=result.tensors['trajectory']
        self.assertEqual(tuple(tensor.shape),(8,3));self.assertEqual(tensor.dtype,torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())
        meta=result.provenance['trajectory_source']['terminal_padding']
        self.assertEqual((meta['real_point_count'],meta['padded_point_count']),(6,2))
        self.assertEqual(result.provenance['future_frame_ids'],[14,19,24,29,34,39,42,42])
        self.assertTrue(torch.equal(tensor[6],tensor[7]))
        self.assertFalse(torch.equal(tensor[5],tensor[6]))  # endpoint is not the last stride sample
        # Independent official SE2 conversion after CARLA-world axis reflection.
        from tools.b2d_adapter.coordinate_transform import actor_to_reference
        from nuplan.common.actor_state.state_representation import StateSE2
        from navsim.common.dataclasses import convert_absolute_to_relative_se2_array
        poses=[]
        for f in [9,14,19,24,29,34,39,42,42]:
            rear=actor_to_reference(self.reader.read(self.reader.index.ref(f)).actor_pose,(-1.388632877,0.))
            poses.append([rear.x,-rear.y,-rear.yaw])
        expected=convert_absolute_to_relative_se2_array(StateSE2(*poses[0]),np.array(poses[1:],dtype=np.float64))
        np.testing.assert_allclose(tensor.numpy(),expected,atol=1e-6)
        self.assertLess(float(torch.max(torch.abs(torch.diff(tensor[:,2])))),.01)
        self.assertLess(float(torch.linalg.vector_norm(tensor[6,:2]-tensor[5,:2]))/.5,.1)
        self.assertEqual(set(result.tensors),{'trajectory'})

    def test_terminal_full_window_parity_and_all_missing_boundary(self):
        self.terminal_fixture()
        ref=self.reader.index.ref(2)
        old=self.builder.trajectory_builder.build_targets(ref).tensors['trajectory']
        derived=self.builder.trajectory_builder.build_targets(ref,terminal_padding=True).tensors['trajectory']
        self.assertTrue(torch.equal(old,derived))
        result=self.builder.trajectory_builder.build_targets(self.reader.index.ref(42),terminal_padding=True)
        self.assertTrue(torch.equal(result.tensors['trajectory'],torch.zeros(8,3)))
        meta=result.provenance['trajectory_source']['terminal_padding']
        self.assertEqual((meta['real_point_count'],meta['padded_point_count']),(0,8))

    def test_terminal_failed_and_unknown_clock_cannot_pad(self):
        self.terminal_fixture(status='Failed')
        ref=self.reader.index.ref(40)
        self.reason('terminal_padding_ineligible',lambda:self.builder.trajectory_builder.build_targets(ref,terminal_padding=True))
        p=self.reader.index.route/'result.json'
        a=json.loads(p.read_text());a['_checkpoint']['records'][0]['status']='Completed';p.write_text(json.dumps(a))
        for f in self.reader.index.frame_ids:
            path=self.reader.index.files['anno'][f]
            with gzip.open(path,'rt') as stream:a=json.load(stream)
            del a['timestamp_s']
            with gzip.open(path,'wt') as stream:json.dump(a,stream)
        self.reason('terminal_padding_ineligible',lambda:self.builder.trajectory_builder.build_targets(ref,terminal_padding=True))

    def test_terminal_identity_gap_teleport_and_wrong_scope_reject(self):
        self.terminal_fixture()
        candidate=self.builder.trajectory_builder.candidate_builder
        ref=self.reader.index.ref(40)
        self.reason('invalid_terminal_window',lambda:candidate.build(ref,full_scene=True,terminal_padding=True))
        p=self.reader.index.route/'result.json'
        result=json.loads(p.read_text());result['_checkpoint']['records'][0]['route_id']='RouteScenario_2_rep0';p.write_text(json.dumps(result))
        self.reason('terminal_padding_ineligible',lambda:candidate.build(ref,full_scene=False,target_only=True,terminal_padding=True))
        result['_checkpoint']['records'][0]['route_id']='RouteScenario_1_rep0';p.write_text(json.dumps(result))
        path=self.reader.index.files['anno'][10]
        with gzip.open(path,'rt') as stream:a=json.load(stream)
        a['teleport_flag']=True
        with gzip.open(path,'wt') as stream:json.dump(a,stream)
        self.reason('terminal_padding_ineligible',lambda:candidate.build(ref,full_scene=False,target_only=True,terminal_padding=True))
        path.unlink();self.reader.index.files['anno'].pop(10)
        self.reader.index.frame_ids=tuple(f for f in self.reader.index.frame_ids if f!=10)
        self.reason('terminal_annotation_gap',lambda:candidate.build(ref,full_scene=False,target_only=True,terminal_padding=True))

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="b2d-v1-unit-")
        self.addCleanup(self.scratch.cleanup)
        route = Path(self.scratch.name) / "synthetic-route"
        (route / "anno").mkdir(parents=True)
        # Finite, continuous, constant-speed Actor poses at 10Hz.
        for i in range(43):
            x, yaw = i * .1, 30.
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            anno = {"bounding_boxes": [{"class": "ego_vehicle", "type_id": "vehicle.lincoln.mkz_2020",
                    "location": [x, 0, 0], "rotation": [0, 0, yaw],
                    "world2ego": [[c,s,0,-c*x],[-s,c,0,s*x],[0,0,1,0],[0,0,0,1]]}],
                    "x_command_near": 20., "y_command_near": 0., "speed": 1.}
            with gzip.open(route / "anno" / ("%05d.json.gz" % i), "wt") as stream:
                json.dump(anno, stream)
        self.reader = FrameReader(RouteIndex(route))
        self.ref = self.reader.index.ref(2)
        self.native = v1_native_config()
        self.model = official_v1_module("transfuser_config").TransfuserConfig()
        self.builder = DiffusionDriveV1TargetBuilder(self.reader, self.native, self.model)

    def reason(self, code, fn):
        with self.assertRaises(AdapterError) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)

    def raw_with_vehicles(self, distances=(5., 2.)):
        raw = self.reader.read(self.ref)
        for i, d in enumerate(distances):
            x, y = ego_to_world((d, 1.), raw.actor_pose)
            raw.annotation["bounding_boxes"].append({"class": "vehicle", "type_id": "vehicle.test",
                "id": i, "center": [x,y,0.], "extent": [2.,1.,.75], "rotation": [0.,0.,50.]})
        return raw

    def test_v1_source_and_unchanged_native_policy(self):
        module = official_v1_module("transfuser_features")
        self.assertEqual(Path(module.__file__).resolve(), ROOT / "repositories/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_features.py")
        contract = validate_v1_config(self.native, self.model)
        self.assertEqual(contract["native_policy"]["policy_sha256"], self.native.provenance()["policy_sha256"])

    def test_model_sensor_sampling_and_class_mismatch_rejected(self):
        for field, value in [("lidar_min_x", -31), ("use_ground_plane", True), ("num_bounding_boxes", 29)]:
            altered = copy.copy(self.model)
            setattr(altered, field, value)
            self.reason("v1_contract_mismatch", lambda: validate_v1_config(self.native, altered))
        self.reason("invalid_v1_config", lambda: validate_v1_config(replace(self.native, camera_policy="b2d-full-view-v1"), self.model))

    def test_real_geometry_formula_sorting_dimensions_heading_padding(self):
        result = self.builder.build_agent_targets(self.raw_with_vehicles())
        states, labels = result.tensors["agent_states"], result.tensors["agent_labels"]
        self.assertEqual(labels.dtype, torch.bool)
        self.assertEqual(int(labels.sum()), 2)
        np.testing.assert_allclose(states[:2].numpy(), [[2+1.388632877,-1,math.radians(-20),4,2],
                                                      [5+1.388632877,-1,math.radians(-20),4,2]], atol=1e-6)
        self.assertTrue((states[2:] == 0).all())  # official invalid slots, not fake actors
        self.assertEqual(result.provenance["source_actor_ids"], ["0", "1"])

    def test_outside_bounds_and_nearest_thirty(self):
        raw = self.raw_with_vehicles(tuple(np.linspace(1, 30, 35)) + (100.,))
        result = self.builder.build_agent_targets(raw)
        states = result.tensors["agent_states"]
        self.assertEqual(result.provenance["valid_count"], 30)
        self.assertLess(float(states[-1, 0]), 30.)

    def test_true_empty_scene_has_false_slots(self):
        result = self.builder.build_agent_targets(self.reader.read(self.ref))
        self.assertEqual(int(result.tensors["agent_labels"].sum()), 0)
        self.assertEqual(result.provenance["source_actor_ids"], [])

    def test_malformed_vehicle_is_never_silently_dropped(self):
        for key, value in [("center", [float("nan"),0,0]), ("extent", [-1,1,1]),
                           ("rotation", [0,float("nan"),0]), ("rotation", [20,0,0]), ("type_id", "bad")]:
            raw = self.raw_with_vehicles()
            raw.annotation["bounding_boxes"][1][key] = value
            self.reason("invalid_vehicle_box", lambda: self.builder.build_agent_targets(raw))
        raw = self.raw_with_vehicles()
        raw.annotation["bounding_boxes"][2]["id"] = 0
        self.reason("invalid_vehicle_box", lambda: self.builder.build_agent_targets(raw))

    def test_partial_targets_and_full_dataset_gate(self):
        partial = self.builder.build_partial_targets(self.ref)
        self.assertEqual(set(partial.tensors), {"trajectory", "agent_states", "agent_labels"})
        self.assertEqual(tuple(partial.tensors["trajectory"].shape), (8,3))
        self.assertFalse(partial.provenance["complete_training_targets"])
        self.reason("missing_bev_semantic_map", lambda: self.builder.compute_targets(self.ref))
        dataset = DiffusionDriveV1Dataset(self.reader, [self.ref], self.native, self.model)
        self.reason("missing_bev_semantic_map", lambda: dataset[0])

    def test_feature_history_reads_are_causal(self):
        seen, original = [], self.reader.read
        def read(ref, *args, **kwargs):
            seen.append(ref.frame_id)
            # For this unit probe only, return empty sensor containers; no model runs.
            return original(ref)
        self.reader.read = read
        builder = DiffusionDriveV1FeatureBuilder(self.reader, self.native, self.model)
        observation = builder.read_current(self.ref)
        self.assertEqual(set(seen), {0,1,2})
        self.assertEqual(observation.motion.provenance["source_frame_ids"], [0,1,2])

    def test_camera_lidar_status_parity_with_actual_v1_builder(self):
        # This equality probe targets original NAVSIM pixel crop. New B2D
        # proportional crop intentionally differs; keep the policy explicit.
        self.native = replace(self.native, camera_policy="navsim-pixel-crop-v1")
        raw = self.reader.read(self.ref)
        rng = np.random.default_rng(42)
        raw.cameras = {name: rng.integers(0,256,(900,1600,3),dtype=np.uint8)
                       for name in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")}
        raw.lidar_xyz = np.array([[0,0,0], [3,2,1], [3,2,.2], [32,-2,101]], dtype=np.float64)
        motion = SimpleNamespace(velocity=np.array([1.,.2]), acceleration=np.array([.3,.4]), provenance={})
        obs = B2DObservation(raw, motion, self.native.provenance()["policy_sha256"])
        feature_builder = DiffusionDriveV1FeatureBuilder(self.reader, self.native, self.model)
        actual = feature_builder.compute_features(obs)
        from navsim.common.dataclasses import AgentInput, Camera, Cameras, Lidar, EgoStatus
        cameras = Cameras(cam_l0=Camera(image=raw.cameras["CAM_FRONT_LEFT"]),
            cam_f0=Camera(image=raw.cameras["CAM_FRONT"]), cam_r0=Camera(image=raw.cameras["CAM_FRONT_RIGHT"]),
            cam_l1=Camera(), cam_l2=Camera(), cam_r1=Camera(), cam_r2=Camera(), cam_b0=Camera())
        status = EgoStatus(ego_pose=np.zeros(3), ego_velocity=motion.velocity, ego_acceleration=motion.acceleration,
                          driving_command=actual["status_feature"][:4].numpy(), in_global_frame=False)
        agent_input = AgentInput([status], [cameras], [Lidar(lidar_pc=candidate_lidar_xyz(raw,self.native.geometry).T)])
        expected = feature_builder._navsim_camera_builder().compute_features(agent_input)
        self.assertTrue(all(torch.equal(actual[k], expected[k]) for k in expected))

    def test_bev_validation_rejects_unknown_and_untraceable_labels(self):
        # Unit-level validator probes only; these tensors never train a model.
        result = NativeBuildResult({"bev_semantic_map": torch.full((128,256),7,dtype=torch.int64)}, {}, {})
        self.reason("invalid_bev_target", lambda: self.builder._validate_bev(result,self.ref))
        result.tensors["bev_semantic_map"].fill_(0)
        self.reason("unverified_bev_target", lambda: self.builder._validate_bev(result,self.ref))
        result.tensors["bev_semantic_map"] = result.tensors["bev_semantic_map"].float()
        self.reason("invalid_bev_target", lambda: self.builder._validate_bev(result,self.ref))


@unittest.skipUnless(os.environ.get("B2D_RUN_V1_REAL_TESTS") == "1", "real V1 integration opt-in")
class V1RealTests(unittest.TestCase):
    def test_real_continuous_samples_batch_and_gate(self):
        route = Path(os.environ.get("B2D_V1_ROUTE", "/data/datasets/Bench2Drive-V0.0.4/v1/SignalizedJunctionRightTurn_Town12_Route5211_Weather1_02-11-08-13-13"))
        reader = FrameReader(RouteIndex(route))
        model = official_v1_module("transfuser_config").TransfuserConfig()
        dataset = DiffusionDriveV1Dataset(reader, [reader.index.ref(i) for i in (80,81)], v1_native_config(), model,
                                         os.environ.get("B2D_LAZ_DECODER_PYTHON"))
        samples = [dataset.diagnose(i) for i in range(2)]
        features, targets = next(iter(DataLoader([(s.features,s.targets) for s in samples],batch_size=2)))
        self.assertEqual(tuple(features["camera_feature"].shape), (2,3,256,1024))
        self.assertEqual(tuple(targets["agent_states"].shape), (2,30,5))
        self.assertGreater(int(targets["agent_labels"].sum()), 0)
        with self.assertRaises(AdapterError) as caught:
            dataset[0]
        self.assertEqual(caught.exception.code, "missing_bev_semantic_map")


if __name__ == "__main__":
    unittest.main()
