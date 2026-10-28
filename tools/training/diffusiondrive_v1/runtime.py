"""Shared V1 B2D dataset and initialization; never uses NAVSIM SceneLoader."""
import hashlib
import json
from pathlib import Path

from tools.b2d_adapter import FrameReader, RouteIndex, AdapterError
from tools.b2d_adapter.v1_builders import DiffusionDriveV1Dataset, official_v1_module, v1_native_config
from tools.b2d_adapter.v1_bev import B2DOpenDriveBEVBuilder

ROOT=Path(__file__).resolve().parents[3]


def native_configuration(config):
    """New B2D defaults; only exact frozen historical configs infer old crop."""
    policy = config.get('camera_policy')
    if policy is None:
        key = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        registry = ROOT/'tools/training/diffusiondrive_v1/camera_original_compatibility.json'
        archived = json.loads(registry.read_text())['historical_configs'] if registry.exists() else {}
        policy = archived.get(key, {}).get('camera_policy', 'b2d-proportional-crop-v1')
    return v1_native_config(policy)


def configuration(path):
    config=json.loads(Path(path).read_text())
    if not 1<=config['sample_count']<=8:
        raise ValueError('bounded V1 entry permits 1..8 frames per route')
    return config


def model_configuration(config):
    import numpy as np
    model=official_v1_module('transfuser_config').TransfuserConfig()
    anchor=(ROOT/config['plan_anchor_path']).resolve()
    a=np.load(anchor,allow_pickle=False)
    if a.shape!=(20,8,2) or not np.isfinite(a).all() or hashlib.sha256(anchor.read_bytes()).hexdigest()!=config['expected_anchor_sha256']:
        raise AdapterError('invalid_anchor',str(anchor))
    model.plan_anchor_path=str(anchor)
    if config.get('bkb_path'):
        model.bkb_path=str((ROOT/config['bkb_path']).resolve())
    return model


def datasets(config,model):
    provider=bev_provider(config)
    native=native_configuration(config);result=[]
    for route in config.get('routes',[config['route']]):
        reader=FrameReader(RouteIndex(route))
        refs=[reader.index.ref(config['start_frame']+i) for i in range(config['sample_count'])]
        result.append(DiffusionDriveV1Dataset(reader,refs,native,model,config['lidar_decoder_python'],provider))
    return result


def bev_provider(config):
    paths={}
    for town,spec in config['map_paths'].items():
        if isinstance(spec,str):paths[town]=str((ROOT/spec).resolve())
        else:paths[town]=dict(spec,path=str((ROOT/spec['path']).resolve()))
    cache=ROOT/config['bev_cache_dir'] if config.get('bev_cache_dir') else None
    return B2DOpenDriveBEVBuilder(paths,cache_dir=cache,label_config=config.get("label_rules"))


def route_group(route):
    import re
    match=re.search(r'(Town\d+(?:HD)?)_Route(\d+)(?:_|$)',Path(route).name)
    if not match:raise AdapterError('unknown_route_group',str(route))
    return match.group(1)+':Route'+match.group(2)


def fit_configuration(path):
    config=json.loads(Path(path).read_text())
    groups={}
    tokens=set()
    for split in ('train','validation'):
        entries=config['data'][split]
        if not entries:raise AdapterError('empty_split',split)
        groups[split]={route_group(e['route']) for e in entries}
        for entry in entries:
            if not entry['frames'] or any(type(i)is not int or i<2 for i in entry['frames']):
                raise AdapterError('invalid_frame_ids',entry)
            for i in entry['frames']:
                token=str((ROOT/entry['route']).resolve())+'::'+str(i)
                if token in tokens:raise AdapterError('duplicate_sample',token)
                tokens.add(token)
    if groups['train']&groups['validation']:
        raise AdapterError('route_split_leakage',sorted(groups['train']&groups['validation']))
    if config['training']['max_steps']<=0 or config['batch_size']<=0:
        raise AdapterError('invalid_training_config','positive steps and batch size required')
    return config,groups


def split_datasets(config,model,provider=None):
    from torch.utils.data import ConcatDataset
    provider=provider or bev_provider(config)
    native=native_configuration(config);result={}
    for split,entries in config['data'].items():
        routes=[]
        for entry in entries:
            reader=FrameReader(RouteIndex(ROOT/entry['route']))
            refs=[reader.index.ref(i) for i in entry['frames']]
            routes.append(DiffusionDriveV1Dataset(reader,refs,native,model,config['lidar_decoder_python'],provider))
        result[split]=ConcatDataset(routes)
    return result,provider


def initialization_manifest(config,model):
    import torch
    path=Path(model.bkb_path)
    if not path.is_file():
        raise AdapterError('missing_image_pretrained',str(path))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=config['expected_image_sha256'] or not sha.startswith('46f8f793'):
        raise AdapterError('invalid_image_pretrained',sha)
    weights=torch.load(path,map_location='cpu',weights_only=True)
    if tuple(weights['conv1.weight'].shape)!=(64,3,7,7) or tuple(weights['fc.weight'].shape)!=(1000,512):
        raise AdapterError('invalid_image_pretrained','expected complete timm ResNet34 A1 ImageNet state_dict')
    return {'image_pretrained':{'path':str(path),'sha256':sha,'source':config['image_pretrained_url']},
            'anchor':{'path':model.plan_anchor_path,'sha256':config['expected_anchor_sha256']}}
