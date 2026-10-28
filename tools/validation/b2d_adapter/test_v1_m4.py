"""Meaningful geometric boundaries and rule/cache isolation for M4."""
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from tools.b2d_adapter.coordinate_transform import ActorPose
from tools.b2d_adapter.errors import AdapterError
from tools.b2d_adapter.v1_label_rules import label_rules
from tools.b2d_adapter.v1_geometry import (actor_vertices, actor_local_hull, carla_rotation,
    actor_floor_diagnostic, clip_vertical_segment, flat_guardrail_surface)
from tools.b2d_adapter.v1_bev import reference_line, polynomial


class M4Geometry(unittest.TestCase):
    def test_rotation_matches_real_ego_inverse(self):
        # Actual Town07 frame80 inverse rotation, independent annotation evidence.
        observed=np.array([[.1441530436,.9863104224,.0800727680],
                           [-.9895106554,.1444429308,.0021905859],
                           [-.0094053475,-.0795486420,.9967865944]])
        np.testing.assert_allclose(carla_rotation([4.592748642,-.1259157509,81.684875488]).T,observed,atol=1e-7)

    def test_tilted_box_projects_height_and_all_vertices(self):
        box={'center':[0,0,1],'extent':[2,1,.5],'rotation':[30,0,0]}
        raw=SimpleNamespace(actor_pose=ActorPose(0,0,0))
        geometry=SimpleNamespace(axle_dx_m=0,axle_dy_m=0)
        hull,vertices=actor_local_hull(box,raw,geometry)
        self.assertAlmostEqual(hull.bounds[2],2*np.cos(np.pi/6)+.5*np.sin(np.pi/6))
        from shapely.geometry import Point
        self.assertTrue(all(hull.covers(Point(v[:2])) for v in vertices))
        box['extent'][2]=float('nan')
        with self.assertRaises(AdapterError):actor_vertices(box)

    def test_floor_interval_boundaries_and_unknown_support(self):
        rules=replace(label_rules('v1_semantic_enriched'),actor_floor='filter-supported-road')
        lane={'type':'driving','surface_xyz':np.array([[0,0,10],[1,0,10.]])}
        box={'center':[0,0,10],'extent':[2,1,1],'rotation':[0,0,0],'id':1}
        self.assertFalse(actor_floor_diagnostic(box,actor_vertices(box),[lane],rules)['excluded'])
        box['center'][2]=12
        self.assertFalse(actor_floor_diagnostic(box,actor_vertices(box),[lane],rules)['excluded'])
        box['center'][2]=12.01
        self.assertTrue(actor_floor_diagnostic(box,actor_vertices(box),[lane],rules)['excluded'])
        box['center'][2]=4
        self.assertTrue(actor_floor_diagnostic(box,actor_vertices(box),[lane],rules)['excluded'])
        box['center'][2]=16
        self.assertTrue(actor_floor_diagnostic(box,actor_vertices(box),[lane],rules)['excluded'])
        self.assertFalse(actor_floor_diagnostic(box,actor_vertices(box),[],rules)['excluded'])
        other={'type':'driving','surface_xyz':np.array([[0,0,16.]])}
        d=actor_floor_diagnostic(box,actor_vertices(box),[lane,other],rules)
        self.assertEqual(d['status'],'ambiguous_road_height');self.assertFalse(d['excluded'])
        self.assertFalse(actor_floor_diagnostic(box,actor_vertices(box),[lane],label_rules('v1_semantic_enriched'))['excluded'])

    def test_segment_height_clip_does_not_join_layers(self):
        a=np.array([0.,0.,0.,1.]);b=np.array([10.,0.,10.,11.])
        p,q=clip_vertical_segment(a,b,4.,5.)
        np.testing.assert_allclose(p,[3.,0]);np.testing.assert_allclose(q,[5.,0])
        self.assertIsNone(clip_vertical_segment(a,b,12.,13.))

    def test_no_envelope_fill_and_nonflat_insertion_refused(self):
        road=ET.fromstring('<road length="10"><planView><geometry s="0" x="10" y="20" hdg="0" length="10"><line/></geometry></planView><elevationProfile><elevation s="0" a="5" b="0" c="0" d="0"/></elevationProfile></road>')
        obj=ET.fromstring('<object type="barrier" name="GuardRail" s="2" t="3" zOffset="0" hdg="0" width="200"><outline><cornerLocal u="-2" v="0" z="0" height=".9"/><cornerLocal u="2" v="0" z="0" height=".9"/></outline></object>')
        values=flat_guardrail_surface(road,obj,reference_line,polynomial)
        np.testing.assert_allclose(values,[[10,23,5,5.9],[14,23,5,5.9]])
        mixed=ET.SubElement(obj.find('outline'),'cornerRoad',s='0',t='0',dz='0',height='1')
        with self.assertRaises(AdapterError):flat_guardrail_surface(road,obj,reference_line,polynomial)
        obj.find('outline').remove(mixed)
        profile=ET.SubElement(ET.SubElement(road,'lateralProfile'),'shape',s='0',t='-2',a='0',b='0',c='0',d='0')
        flat_guardrail_surface(road,obj,reference_line,polynomial)
        profile.set('a','.1')
        with self.assertRaises(AdapterError):flat_guardrail_surface(road,obj,reference_line,polynomial)
        profile.set('a','0')
        road.find('elevationProfile/elevation').set('b','.1')
        with self.assertRaises(AdapterError):flat_guardrail_surface(road,obj,reference_line,polynomial)

    def test_original_rules_and_versions_fail_closed(self):
        with self.assertRaises(AdapterError):label_rules({'name':'v1_original_7class','actor_floor':'diagnose'})
        with self.assertRaises(AdapterError):label_rules({'name':'v1_semantic_enriched','version':'unknown'})
        self.assertNotEqual(label_rules().sha256,label_rules('v1_semantic_enriched').sha256)


if __name__=='__main__':unittest.main()
