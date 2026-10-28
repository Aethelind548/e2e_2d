"""Independent M5 geometry, interval, floor and cache regression boundaries."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from tools.b2d_adapter.errors import AdapterError
from tools.b2d_adapter.v1_bev import OpenDriveLocalMap
from tools.b2d_adapter.static_layers import MapStaticLayers, ROADMARK_TYPES, load_static_query


XML='''<OpenDRIVE><header revMajor="1" revMinor="4"/>
<road id="1" length="30" junction="-1"><planView><geometry s="0" x="0" y="0" hdg="0" length="30"><line/></geometry></planView>
<elevationProfile><elevation s="0" a="0" b="0.1" c="0" d="0"/></elevationProfile>
<lateralProfile><superelevation s="0" a="0.05" b="0" c="0" d="0"/></lateralProfile>
<lanes><laneOffset s="0" a="0.5" b="0" c="0" d="0"/><laneSection s="0">
<left><lane id="1" type="driving" level="false"><width sOffset="0" a="3" b="0" c="0" d="0"/><roadMark sOffset="0" type="none" laneChange="none"/><userData><vectorLane sOffset="0" travelDir="backward"/></userData></lane></left>
<center><lane id="0" type="none"><roadMark sOffset="0" type="solid solid" width="0.3" color="yellow" laneChange="none"/></lane></center>
<right><lane id="-1" type="driving" level="false"><width sOffset="0" a="3" b="0" c="0" d="0"/><roadMark sOffset="0" type="solid" width="0.15" color="white" laneChange="both"/><roadMark sOffset="10" type="broken" color="white" laneChange="increase"/><userData><vectorLane sOffset="0" travelDir="forward"/></userData></lane>
<lane id="-2" type="driving" level="false"><width sOffset="0" a="3" b="0.02" c="0" d="0"/></lane></right></laneSection></lanes>
<objects><object id="a" type="crosswalk" name="Crosswalk" s="8" t="0" zOffset="0" hdg="0" pitch="0" roll="0" width="8" length="2"><outline>
<cornerLocal u="-1" v="-4" z="0"/><cornerLocal u="1" v="-4" z="0"/><cornerLocal u="1" v="4" z="0"/><cornerLocal u="-1" v="4" z="0"/><cornerLocal u="-1" v="-4" z="0"/>
</outline></object></objects></road></OpenDRIVE>'''


class StaticLayersTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'map.xodr';self.path.write_text(XML)
        self.map=OpenDriveLocalMap(self.path,'TownTest')
        self.layers=MapStaticLayers(self.map,'test-v1',self.root/'cache')
    def tearDown(self):self.temp.cleanup()

    def test_side_geometry_direction_and_shared_boundary(self):
        arrays,m=self.layers.section('1');lanes={v['lane_id']:v for v in m['lanes']}
        self.assertEqual(lanes[-1]['right_boundary_id'],lanes[-2]['left_boundary_id'])
        self.assertEqual(lanes[1]['left_boundary_id'],'1:0:b1')
        self.assertEqual(lanes[-1]['left_boundary_id'],'1:0:b0')
        center=arrays[lanes[-1]['center_array']]
        np.testing.assert_allclose(center[:,1],-1*np.cos(.05))
        np.testing.assert_allclose(center[:,2],.1*arrays['s']-np.sin(.05))
        result=self.layers.query((0,0,0),1.5,road_ids=['1'])
        travel=next(v for v in result['provenance']['lanes'] if v['lane_id']==1)['direction_intervals'][0]
        self.assertEqual(travel['travel_left_boundary_id'],'1:0:b0')
        self.assertFalse(travel['lane_change_decision_normalized'])
        self.assertEqual(len({b['id'] for b in result['provenance']['boundaries']}),4)

    def test_marks_none_unknown_and_real_intervals(self):
        _,m=self.layers.section('1');marks=m['roadmarks']
        pair=[v for v in marks if v['lane_id']==-1]
        self.assertEqual([v['type'] for v in pair],['solid','broken'])
        self.assertEqual([v['s_interval_m'] for v in pair],[[0.,10.],[10.,30.]])
        self.assertIsNone(pair[1]['width_m']);self.assertEqual(pair[1]['laneChange'],'increase')
        self.assertEqual(next(v for v in marks if v['lane_id']==1)['type_id'],ROADMARK_TYPES['none'])
        self.assertEqual(next(v for v in marks if v['lane_id']==-2)['status'],'unknown')
        result=self.layers.query((0,0,0),1.5,road_ids=['1']);types=result['masks']['roadmark_type_mask']
        self.assertEqual(types[20,118],ROADMARK_TYPES['solid'])
        self.assertEqual(types[60,118],ROADMARK_TYPES['broken'])
        self.assertTrue((result['masks']['roadmark_unknown_mask']>0).any())

    def test_crosswalk_is_separate_and_height_filtered(self):
        result=self.layers.query((0,0,0),1.5,road_ids=['1'])
        self.assertGreater(result['masks']['crosswalk_mask'].sum(),0)
        self.assertNotIn('bev_semantic_map',result['masks'])
        upper=self.layers.query((0,0,0),12,road_ids=['1'])
        self.assertEqual(upper['masks']['crosswalk_mask'].sum(),0)
        self.assertEqual(upper['masks']['lane_boundary_mask'].sum(),0)

    def test_bad_crosswalk_never_falls_back_to_box(self):
        road=self.map.roads['1'];obj=road.find('objects/object')
        corners=obj.findall('outline/cornerLocal');corners[1].set('u','1');corners[1].set('v','4');corners[2].set('u','1');corners[2].set('v','-4')
        result=self.layers.query((0,0,0),1.5,road_ids=['1'])
        self.assertEqual(result['masks']['crosswalk_mask'].sum(),0)
        self.assertEqual(result['masks']['crosswalk_validity_mask'].sum(),0)
        self.assertTrue(result['provenance']['unsupported'])

    def test_cache_roundtrip_identity_and_checksum(self):
        first=self.layers.query((0,0,0),1.5,road_ids=['1'])
        again=MapStaticLayers(self.map,'test-v1',self.root/'cache');second=again.query((0,0,0),1.5,road_ids=['1'])
        for key in first['masks']:np.testing.assert_array_equal(first['masks'][key],second['masks'][key])
        self.assertEqual(again.cache_stats['section_hits'],1);self.assertEqual(again.cache_stats['object_hits'],1)
        saved=self.layers.save_query(first,self.root/'local')
        independent=load_static_query(saved,self.map.sha256,self.layers.cache_key)
        np.testing.assert_array_equal(independent['masks']['crosswalk_mask'],first['masks']['crosswalk_mask'])
        with self.assertRaises(AdapterError):load_static_query(saved,expected_map_sha256='other-version')
        path=next((again.cache_dir/'sections').glob('*.npz'))
        with np.load(path,allow_pickle=False) as archive:contents={k:archive[k].copy() for k in archive.files}
        contents['boundary_0_xyz'][0,0]+=1
        np.savez_compressed(path,**contents)
        with self.assertRaises(AdapterError):MapStaticLayers(self.map,'test-v1',self.root/'cache').section('1')
        with self.assertRaises(AdapterError):MapStaticLayers(self.map,'test-v1',expected_sha256='incorrect')
        self.assertNotEqual(self.layers.cache_key,MapStaticLayers(self.map,'test-v2').cache_key)

    def test_unsupported_height_owner_cannot_leak_visible_boundary(self):
        import xml.etree.ElementTree as ET
        lane=self.map.roads['1'].find('lanes/laneSection/left/lane')
        ET.SubElement(lane,'height',sOffset='0',inner='.15',outer='.15')
        # Center is above this height band while its inner boundary is visible.
        # The unsupported owner must still block boundary/RoadMark rasterization.
        with self.assertRaises(AdapterError):self.layers.query((0,0,0),-2.95,road_ids=['1'])

    def test_coincident_xy_roads_have_separate_floors(self):
        import copy
        road=copy.deepcopy(self.map.roads['1']);road.set('id','2')
        road.find('elevationProfile/elevation').set('a','8')
        road.find('objects/object').set('id','b')
        self.map.roads['2']=road
        original_bounds=self.map.bounds[0]
        self.map.bounds.append((road,original_bounds[1],original_bounds[2]))
        low=self.layers.query((0,0,0),1.5)
        high=self.layers.query((0,0,0),9.5)
        self.assertEqual({l['road_id'] for l in low['provenance']['lanes']},{'1'})
        self.assertEqual({l['road_id'] for l in high['provenance']['lanes']},{'2'})
        self.assertEqual({o['object_id'] for o in low['provenance']['crosswalks']},{'a'})
        self.assertEqual({o['object_id'] for o in high['provenance']['crosswalks']},{'b'})

    def test_discontinuous_width_has_no_invented_connector(self):
        import xml.etree.ElementTree as ET
        lane=self.map.roads['1'].find('lanes/laneSection/right/lane')
        ET.SubElement(lane,'width',sOffset='5',a='6',b='0',c='0',d='0')
        _,metadata=self.layers.section('1')
        self.assertTrue(metadata['discontinuities'])
        result=self.layers.query((0,0,0),1.5,road_ids=['1'])
        self.assertEqual(result['masks']['lane_boundary_mask'][20,114],0)
        self.assertEqual(result['masks']['lane_boundary_validity_mask'][20,114],0)


if __name__=='__main__':unittest.main()
