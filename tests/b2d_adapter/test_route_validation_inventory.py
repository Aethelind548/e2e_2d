"""Route candidate decisions never replace strict per-frame session evidence."""
import copy
import hashlib
import json
import math

import numpy as np
import pytest

from tools.b2d_adapter.experimental_route_preview import RoutePolyline
from tools.b2d_adapter.route_preview_contract import preview_command4
from tools.training.diffusiondrive_v1.certified_overlay_v4 import CHECKS
from tools.validation.b2d.route_validation_inventory import (
    CachedRoute, RULE, candidate_accepted, finalize_inventory, intervals, load_xml_candidates, navigation_target,
    projection_metrics, route_pass, route_states, strict_frame_checks,
)


def test_evidence_checksum_tracks_bytes_and_propagates_missing_file(tmp_path):
    from tools.validation.b2d.route_validation_inventory import sha
    path = tmp_path / 'evidence.json'
    path.write_bytes(b'{"identity": "original"}\n')
    original = sha(path)
    assert original == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b'{"identity": "changed"}\n')
    assert sha(path) != original
    path.unlink()
    with pytest.raises(FileNotFoundError):
        sha(path)


def strict_case():
    row = dict(command_match=True, geometry_matched=True, geometry_jump=False,
               session_command_match=True, session_evidence_sha256='session',
               session_planner_evidence_sha256='planner')
    pair = dict(feature_sha='f', target_sha='t')
    frozen = dict(status8=[0.,1.,0.,0.,1.,2.,3.,4.], a0_motion4=[1.,2.,3.,4.], motion_source='A0',
        source_feature_identity=dict(pair_index_sha256='idx',feature_sha256='f',target_sha256='t'))
    previous = dict(contract_checks=dict.fromkeys(CHECKS,True),frozen_overlay_sha256='overlay',
                    evidence=dict(session_capture_sha256='session',planner_capture_sha256='planner'))
    return row,frozen,pair,previous


def test_route_pass_with_isolated_mismatch_does_not_certify_that_frame():
    rows=[dict(route='r',frame=i,sample_id=str(i),comparable=True,geometry_matched=True,
               command_match=i!=50,command_mismatch=i==50,geometry_jump=False) for i in range(100)]
    assert route_pass(rows,True)[0]
    row,frozen,pair,previous=strict_case();row['command_match']=False
    assert not strict_frame_checks(row,frozen,pair,previous,True,'overlay','idx')[0]


@pytest.mark.parametrize('flag',['left_right_reversal','geometry_jump'])
def test_route_candidate_rejects_reversal_and_jump_but_keeps_other_rows(flag):
    rows=[dict(route='r',frame=i,sample_id=str(i),comparable=True,geometry_matched=True,
               command_match=True) for i in range(30)]
    rows[15][flag]=True
    assert not route_pass(rows,True)[0]
    assert len(rows)==30


def test_intervals_break_at_missing_numeric_frame_instead_of_row_position():
    rows=[dict(route='r',frame=i,sample_id=str(i),bad=True) for i in [1,2,10,11,12]]
    result=intervals(rows,lambda r:r['bad'],'COMMAND_MISMATCH')
    assert [(r['start_frame'],r['end_frame'],r['frames']) for r in result]==[(1,2,2),(10,12,3)]


def test_local_xml_never_substitutes_for_a_real_session():
    row,frozen,pair,previous=strict_case()
    assert not strict_frame_checks(row,frozen,pair,previous,False,'overlay','idx')[0]
    assert strict_frame_checks(row,frozen,pair,previous,True,'overlay','idx')[0]


def test_status_is_authoritative_and_missing_is_distinct_from_unknown():
    status=[1.,0.,0.,0.,1.,2.,3.,4.]
    command,agreement,target=navigation_target(status,[0.,0.,1.,0.])
    assert command==status[:4] and agreement is False and target=='FROZEN_STATUS_NAVIGATION4'
    assert navigation_target(None,None)==(None,None,'PREVIEW_ONLY_STATUS_MISSING')
    unknown=[0.,0.,0.,1.,1.,2.,3.,4.]
    assert navigation_target(unknown,None)==(unknown[:4],False,'FROZEN_STATUS_NAVIGATION4')


def test_invalid_frozen_status_is_rejected_instead_of_using_preview():
    with pytest.raises(ValueError,match='invalid_frozen_status8'):
        navigation_target([0.,1.,float('nan'),0.,0.,0.,0.,0.],[0.,1.,0.,0.])


def test_route_states_preserve_mismatch_without_frame_promotion():
    row=dict(source_status='ROUTE_SOURCE_VERIFIED',plan_status='PASS',
             comparable_frames=100,effective_coverage=.9,geometry_matched_frames=99,
             command_agreement_rate=.99,command_mismatch_frames=1)
    states=route_states(row)
    assert set(states)=={'ROUTE_SOURCE_VERIFIED','ROUTE_GEOMETRY_MATCHED',
                         'ROUTE_COMMAND_MATCHED','COMMAND_MISMATCH'}
    assert 'FRAME_CERTIFIED' not in states
    row['source_status']='SOURCE_NOT_VERIFIED'
    assert 'ROUTE_SOURCE_VERIFIED' not in route_states(row)
    assert 'SOURCE_NOT_VERIFIED' in route_states(row)


def test_equal_raw_result_xml_id_is_only_a_recovery_lead(tmp_path):
    output=tmp_path/'audit';output.mkdir()
    raw=tmp_path/'raw';folder=raw/'r';folder.mkdir(parents=True)
    (folder/'result.json').write_text(json.dumps({'_checkpoint':{'records':[{'route_id':'RouteScenario_42_rep0'}]}}))
    row=dict(route='r',town='Town01',source_status='SOURCE_NOT_VERIFIED',
             plan_status='MISSING_OR_FAILED',candidate_xml_count=1,association_basis='NO_RELIABLE_XML',
             comparable_frames=0,effective_coverage=0,geometry_matched_frames=0,
             command_agreement_rate=0,command_mismatch_frames=0,left_right_reversals=0)
    (output/'route_validation_index.jsonl').write_text(json.dumps(row)+'\n')
    (output/'xml_candidates.json').write_text(json.dumps({'k':{'town':'Town01','xml_evidence':[{'xml_id':'42'}]}}))
    (output/'route_validation_summary.json').write_text(json.dumps({'status_formable':0,'strict_frame_certified':0}))
    finalize_inventory(output,raw)
    recovery=json.loads((output/'source_recovery_inventory.json').read_text())['records'][0]
    assert len(recovery['exact_id_leads'])==1 and not recovery['automatic_promotion']
    certified=json.loads((output/'route_certification_manifest.jsonl').read_text())
    assert certified['route_states']==['SOURCE_NOT_VERIFIED']
    assert not certified['frame_promotion_by_route']


@pytest.mark.parametrize('condition',['command_match','session_source','motion_a0','frozen_identity'])
def test_each_frozen_strict_condition_is_still_required(condition):
    row,frozen,pair,previous=strict_case();previous['contract_checks'][condition]=False
    assert not strict_frame_checks(row,frozen,pair,previous,True,'overlay','idx')[0]


def test_session_and_overlay_sha_changes_reject_certification():
    row,frozen,pair,previous=strict_case()
    previous['evidence']['session_capture_sha256']='other'
    assert not strict_frame_checks(row,frozen,pair,previous,True,'overlay','idx')[0]
    previous['evidence']['session_capture_sha256']='session'
    assert not strict_frame_checks(row,frozen,pair,previous,True,'changed','idx')[0]


def test_frozen_a0_cannot_be_changed_or_replaced_with_a2v():
    row,frozen,pair,previous=strict_case();frozen['status8'][4]=10.
    assert not strict_frame_checks(row,frozen,pair,previous,True,'overlay','idx')[0]
    frozen['status8'][4]=1.;frozen['motion_source']='A2V'
    assert not strict_frame_checks(row,frozen,pair,previous,True,'overlay','idx')[0]


def test_cached_projection_and_preview_match_the_shared_calculator():
    points=[[0.,0.],[10.,0.],[20.,-10.],[40.,-10.]]
    original=RoutePolyline(points,'test');fast=CachedRoute(points,'test')
    for xy in [[0.,0.],[5.,.2],[11.,-1.],[22.,-9.]]:
        a=preview_command4(original,xy,-.1,**RULE)
        b=preview_command4(fast,xy,-.1,**RULE)
        assert a['command4']==b['command4']
        assert a['preview_point_xy']==pytest.approx(b['preview_point_xy'])


def test_exact_projection_and_lateral_boundaries_are_unchanged():
    route=CachedRoute([[0,0],[100,0]],'test')
    assert preview_command4(route,[0,2],0,**RULE)['category']=='Straight'
    assert preview_command4(route,[0,2.00001],0,**RULE)['category']=='Left'
    assert preview_command4(route,[0,5],0,**RULE)['preview_valid']
    assert not preview_command4(route,[0,5.00001],0,**RULE)['preview_valid']


def test_xml_geometries_are_deduplicated_but_all_source_shas_retained(tmp_path):
    text='<routes><route town="Town01" id="1"><waypoints><position x="0" y="0"/><position x="50" y="0"/></waypoints></route></routes>'
    (tmp_path/'a.xml').write_text(text);(tmp_path/'b.xml').write_text(text.replace('id="1"','id="2"'))
    candidates,errors=load_xml_candidates([tmp_path])
    assert not errors and len(candidates)==1
    assert len(next(iter(candidates.values()))['xml_evidence'])==2


def test_geometry_association_checks_actual_body_heading():
    points=[[0,0],[100,0]]
    assert candidate_accepted(projection_metrics(points,[[10,0,0],[20,0,0]]))
    assert not candidate_accepted(projection_metrics(points,[[10,0,math.pi],[20,0,math.pi]]))
