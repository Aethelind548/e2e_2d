"""Metadata identity is not an original collection XML binding."""
import hashlib
from pathlib import Path

import pytest

from tools.validation.b2d.collection_source_recovery import (
    local_xml_catalog, make_record, parse_directory, result_identity, unique,
)


def test_result_index_cannot_be_used_as_xml_id():
    directory=parse_directory('Accident_Town12_Route30356_Weather10_02-10-03-31-29')
    ids,issues=result_identity(directory,{'_checkpoint':{'records':[{'index':1,'route_id':'RouteScenario_30356_rep0'}]}})
    assert not issues and ids==[{'xml_id':'30356','repetition':'0'}]
    assert directory['timestamp']=='02-10-03-31-29'


def test_multiple_result_records_are_preserved_and_do_not_pick_first():
    ids,issues=result_identity({'route_id':'42'},{'_checkpoint':{'records':[
        {'route_id':'RouteScenario_42_rep0'},{'route_id':'RouteScenario_43_rep0'}]}})
    assert len(ids)==2
    assert set(issues)=={'EXPECTED_ONE_RESULT_RECORD','DIRECTORY_RESULT_ROUTE_ID_CONFLICT'}


@pytest.mark.parametrize('route_id,expected',[
    ('RouteScenario_7_rep0','DIRECTORY_RESULT_ROUTE_ID_CONFLICT'),
    ('unparseable','RESULT_ROUTE_ID_INVALID')])
def test_result_identity_conflicts_are_explicit(route_id,expected):
    _,issues=result_identity({'route_id':'42'},{'_checkpoint':{'records':[{'route_id':route_id}]}})
    assert expected in issues


@pytest.mark.parametrize('basis',['AMBIGUOUS_LOCAL_XML_CANDIDATES','UNIQUE_LOCAL_XML_GEOMETRY_PROXY'])
def test_numeric_id_and_geometric_proxy_never_prove_original_collection(tmp_path,basis):
    path=tmp_path/'result.json';path.write_text('{}')
    prior=dict(route='r',accepted_candidates=[{'key':'a','coverage':1.},{'key':'b','coverage':.9}],
        selected_key='a' if basis.startswith('UNIQUE') else None,association_basis=basis,
        source_status='ROUTE_SOURCE_VERIFIED',evidence_grade='LOCAL_XML_PROXY',frozen_status_rows=5)
    catalog={k:dict(town='Town12',xml_sources=[dict(path=str(tmp_path/(k+'.xml')),
        sha256='sha'+k,xml_id='42',release_scope='OLDER_EVALUATION_220')]) for k in ['a','b']}
    directory=dict(route_id='42',town='Town12',scenario='Accident',weather_id='1',timestamp='02-10-03-31-29')
    r=make_record(prior,{'exact_id_leads':[]},directory,
        {'_checkpoint':{'records':[{'route_id':'RouteScenario_42_rep0','index':1}]}},path,[],catalog,True)
    assert len(r['candidate_xml_sources'])==2
    assert not r['original_source_recovered'] and not r['ambiguity_resolved']
    assert r['original_route_definition']['anchors_xyz'] is None
    assert not r['training_admitted'] and r['frame_certified_promoted']==0
    if basis.startswith('AMBIGUOUS'):
        assert r['source_status']=='AMBIGUOUS_SOURCE'


def test_xml_file_sha_drift_prevents_using_old_evidence(tmp_path):
    path=tmp_path/'routes.xml';path.write_text('<routes/>')
    candidates={'k':{'town':'Town12','xyz':[[0.,0.,0.],[1.,0.,0.]],
        'xml_evidence':[{'path':str(path),'sha256':'old','xml_id':'42'}]}}
    with pytest.raises(ValueError,match='xml_evidence_sha_changed'):
        local_xml_catalog(candidates)


def test_xml_anchor_change_is_not_hidden_by_equal_id(tmp_path):
    path=tmp_path/'routes.xml'
    path.write_text('<routes><route town="Town12" id="42"><waypoints><position x="0" y="0"/><position x="99" y="0"/></waypoints></route></routes>')
    candidates={'k':{'town':'Town12','xyz':[[0.,0.,0.],[1.,0.,0.]],
        'xml_evidence':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'xml_id':'42'}]}}
    with pytest.raises(ValueError,match='candidate_xml_anchor_identity_changed'):
        local_xml_catalog(candidates)


def test_duplicate_route_identity_is_rejected():
    with pytest.raises(ValueError,match='duplicate_route'):
        unique([{'route':'r'},{'route':'r'}],'route')


def test_incomplete_directory_identity_is_not_guessed():
    with pytest.raises(ValueError,match='not_parseable'):
        parse_directory('Accident_Town12_Route42_Weather1')
