"""Numerical boundaries for new contracts; real data checks run separately."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import Point,Polygon
from tools.b2d_adapter.config import CandidateConfig
from tools.b2d_adapter.coordinate_transform import ActorPose
from tools.b2d_adapter.errors import AdapterError
from tools.b2d_adapter.v1_geometry import actor_vertices,carla_rotation
from tools.training.diffusiondrive_v1.full_recovery_rules import (
    verified_actor,rear6,world_plane6,plane6_world,RecoveryMap)


class FullGeometryTests(unittest.TestCase):
    def raw(self,rotation):
        return SimpleNamespace(ego_box={'location':[1200.,-500.,10.],'rotation':rotation},
            actor_pose=ActorPose(1200.,-500.,math.radians(rotation[2])))

    def test_gravity6_world_inverse_under_large_pitch_roll(self):
        points=np.array([[1201.,-501.,9.],[1202.,-502.,11.],[1199.,-498.,10.]])
        for rotation in ([0,0,0],[30,45,120],[-15,90,-170],[20,-80,179]):
            raw=self.raw(rotation);local=world_plane6(points,raw,CandidateConfig())
            np.testing.assert_allclose(plane6_world(local,raw,CandidateConfig()),points,atol=1e-10)

    def test_real_3d_rear_differs_from_yaw_flattened_offset(self):
        geometry=CandidateConfig();raw=self.raw([60,0,0])
        result=rear6(raw,geometry)
        self.assertAlmostEqual(result[0]-1200,geometry.axle_dx_m*.5,places=10)
        self.assertAlmostEqual(result[2]-10,geometry.axle_dx_m*math.sqrt(3)/2,places=10)

    def test_projected_rectangle_contains_all_eight_vertices(self):
        raw=self.raw([0,0,17]);box={'id':8,'center':[1210.,-498.,11.],
            'extent':[2.5,1.,.8],'rotation':[20.,75.,50.]}
        box['world_cord']=actor_vertices(box).tolist()
        hull,state,proof=verified_actor(box,raw,CandidateConfig())
        self.assertGreater(proof['detection_enclosure_area_m2'],hull.area)
        self.assertTrue(np.isfinite(state).all());self.assertGreater(min(state[-2:]),0)

    def test_unverified_or_inconsistent_world_corners_are_rejected(self):
        raw=self.raw([0,0,0]);box={'id':8,'center':[1210.,-498.,11.],
            'extent':[2.5,1.,.8],'rotation':[20.,75.,50.]}
        with self.assertRaises(AdapterError):verified_actor(box,raw,CandidateConfig())
        box['world_cord']=(actor_vertices(box)+[.1,0,0]).tolist()
        with self.assertRaises(AdapterError):verified_actor(box,raw,CandidateConfig())

    def test_margin_roi_contains_frozen_roi_and_quantization_edge(self):
        raw=self.raw([0,0,0]);geometry=CandidateConfig();roi=RecoveryMap.roi(raw,geometry)
        rear=np.array([1200+geometry.axle_dx_m,500])
        self.assertTrue(roi.covers(Point(*(rear+[-.25,32.25]))))
        self.assertFalse(roi.covers(Point(*(rear+[-1.,0]))))

    def test_source_triangles_preserve_folded_surface_not_background(self):
        mapped=object.__new__(RecoveryMap);mapped.mesh_cache={};mapped.spacing=.5
        road=ET.fromstring('<road><planView><geometry s="0" x="0" y="0" hdg="0" length="2"><line/></geometry></planView></road>')
        rec={'road':road,'road_id':'x','section':object(),'lane_id':1,
            'roll':np.zeros(3),'s':np.arange(3,dtype=float),'inner_offset':np.zeros(3),'outer_offset':np.ones(3),
            'inner':np.array([[-1,0],[1,1],[-1,2]]),'outer':np.array([[1,0],[-1,1],[1,2]])}
        original=Polygon(np.concatenate((rec['inner'],rec['outer'][::-1])))
        self.assertFalse(original.is_valid)
        pieces,proof=mapped.strip_geometry(rec,0,3)
        self.assertTrue(all(p.is_valid for p in pieces));self.assertEqual(proof['triangles'],4)
        self.assertGreater(proof['union_area_m2'],0)
        self.assertLessEqual(proof['union_area_m2'],proof['triangle_area_sum_m2']+1e-12)

    def test_banked_invalid_surface_is_not_silently_flattened(self):
        mapped=object.__new__(RecoveryMap);mapped.mesh_cache={};mapped.spacing=.5
        rec={'road':ET.fromstring('<road/>'),'road_id':'x','section':object(),'lane_id':1,'roll':np.array([.1,.1])}
        with self.assertRaises(AdapterError):mapped.strip_geometry(rec,0,2)


if __name__=='__main__':unittest.main()
