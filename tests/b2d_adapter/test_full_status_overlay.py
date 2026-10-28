"""CPU regressions for the independent full-pair overlay, version 2."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from tools.validation.b2d.full_status_overlay import compose, extract_status, check_row


@pytest.fixture
def pair():
    return {'sample_id':'a'*64,'route':'route-a','frame':12,'split':'train','feature_sha':'b'*64,'target_sha':'c'*64}


@pytest.fixture
def feature_file(tmp_path,pair):
    sources=[{'path':'route-a/anno/00012.json.gz','sha256':'d'*64}]
    path=tmp_path/'feature.pt'
    payload={'schema':'b2d-v1-features-only-v1','sample_id':pair['sample_id'],'route':pair['route'],'frame':12,'dataset_id':'b2d','producer_sha256':'p'*64,'sources':sources,'features':{'status_feature':torch.tensor([0,1,0,0,1,2,3,4],dtype=torch.float32)}}
    torch.save(payload,path)
    feature={**{k:pair[k] for k in ('sample_id','route','frame')},'sha256':pair['feature_sha'],'state':'complete','source_json':json.dumps(sources),'bytes':path.stat().st_size}
    return path,payload,feature


def test_mmap_actual_old_status(feature_file,pair):
    path,_,feature=feature_file
    status=extract_status(path,feature,pair,'p'*64,'b2d')
    assert status.dtype==np.float32
    np.testing.assert_array_equal(status,[0,1,0,0,1,2,3,4])


@pytest.mark.parametrize('key,value',[('sample_id','wrong'),('route','other'),('frame',13),('producer_sha256','x'*64),('dataset_id','other')])
def test_mmap_rejects_embedded_identity(feature_file,pair,key,value):
    path,payload,feature=feature_file
    payload[key]=value;torch.save(payload,path);feature['bytes']=path.stat().st_size
    with pytest.raises(ValueError,match='embedded_identity'):
        extract_status(path,feature,pair,'p'*64,'b2d')


@pytest.mark.parametrize('status',[torch.zeros(8,dtype=torch.float64),torch.zeros(9),torch.tensor([0,1,0,0,float('nan'),0,0,0])])
def test_mmap_rejects_dtype_shape_nonfinite(feature_file,pair,status):
    path,payload,feature=feature_file
    payload['features']['status_feature']=status;torch.save(payload,path);feature['bytes']=path.stat().st_size
    with pytest.raises(ValueError,match='shape_dtype_nonfinite'):
        extract_status(path,feature,pair,'p'*64,'b2d')


def test_mmap_rejects_frozen_sha_disagreement(feature_file,pair):
    path,_,feature=feature_file;feature['sha256']='different'
    with pytest.raises(ValueError,match='index_pair_mismatch'):
        extract_status(path,feature,pair,'p'*64,'b2d')


def test_mmap_rejects_legacy_serialization_without_guessing(feature_file,pair):
    path,payload,feature=feature_file
    torch.save(payload,path,_use_new_zipfile_serialization=False);feature['bytes']=path.stat().st_size
    with pytest.raises(RuntimeError,match='mmap'):
        extract_status(path,feature,pair,'p'*64,'b2d')


def test_offline_geometry_never_certifies_and_motion_uses_old(pair):
    old=np.array([0,1,0,0,1,2,3,4],np.float32)
    preview={'route':pair['route'],'frame':12,'preview_valid':True,'command4_json':'[1,0,0,0]','navigation_source':'RECONSTRUCTED_ROUTE','motion4_json':'[99,99,99,99]'}
    row=compose(pair,preview,old,{'feature_sha256':pair['feature_sha'],'target_sha256':pair['target_sha']})
    assert row['status8']==[1,0,0,0,1,2,3,4]
    assert row['status_validity']=='NOT_VERIFIED'
    assert row['training_admitted'] is False
    assert row['navigation_source']['certification']=='NOT_VERIFIED'
    assert all(check_row(row,pair).values())


def test_preview_failure_does_not_become_legal_unknown_or_near_fallback(pair):
    row=compose(pair,{'route':pair['route'],'frame':12,'preview_valid':False,'preview_reason':'local_segment_end'},np.array([0,1,0,0,1,2,3,4],np.float32),{})
    assert row['status8'] is None
    assert row['status_validity']=='INVALID'
    assert row['preview_validity']=='INVALID'
    assert row['fallback_source'] is None


def test_missing_frame_identity_kept_and_not_guessed(pair):
    row=compose(pair,None,None,{},'missing')
    assert row['sample_id']==pair['sample_id']
    assert row['preview_validity']=='MISSING'
    assert row['status8'] is None
    assert 'ROUTE_FRAME_NOT_IN_FULL_PREVIEW' in row['failure_reason']


def test_legal_unknown_distinct_from_invalid(pair):
    row=compose(pair,{'route':pair['route'],'frame':12,'preview_valid':True,'command4_json':'[0,0,0,1]'},np.array([0,1,0,0,1,2,3,4],np.float32),{})
    assert row['status8'][:4]==[0,0,0,1]
    assert row['status_validity']=='NOT_VERIFIED'
    assert row['diagnostic_metadata']['legal_unknown'] is True
