"""Explicit paired Feature/Target regeneration for the NEW 6DoF rear contract.

Never combine these Targets with old Feature tensors. Default is a two-sample
pilot; full regeneration requires --all plus a matching real-forward gate.
This optional branch is separate from certified compatible/projected queues.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))

from tools.b2d_adapter import FrameReader,RouteIndex,AdapterError
from tools.b2d_adapter.coordinate_transform import ActorPose,wrap_angle
from tools.b2d_adapter.camera_adapter import proportional_camera_feature
from tools.b2d_adapter.lidar_decode import decode_lidar_in_process
from tools.b2d_adapter.native_builders import NativeBuildResult
from tools.b2d_adapter.v1_geometry import carla_rotation
from tools.training.diffusiondrive_v1.full_recovery_rules import (rear6,world_plane6,plane6_world,
    trajectory_diagnosis,RecoveryBuilder,RecoveryProvider,GEOMETRY_CONTRACT)
from tools.training.diffusiondrive_v1.features_cache import (validate_features,file_digest,digest,durable_json,
    StoragePause,directory_sync,sample_id)
from tools.training.diffusiondrive_v1.target_cache import validate_targets,storage_check
from tools.training.diffusiondrive_v1.target_retry_audit import readonly

CONTRACT=dict(GEOMETRY_CONTRACT,version='b2d-v1-gravity6-paired-v2',
    reference='world XYZ rear = actor location + full CARLA rotation @ (-1.388632877,0,0); gravity-aligned yaw frame',
    lidar='raw Actor 3D XYZ -> world -> gravity frame; histogram only after transformation',
    status='same current near-bearing policy at true rear6, trailing rear6 position differences',
    feature_regeneration_required=True,old_cached_features_compatible=False)


class ReferenceReader:
    def __init__(self,reader,geometry):self.reader,self.geometry,self.index=reader,geometry,reader.index
    def read(self,ref,*args,**kwargs):
        actual=self.reader.read(ref,*args,**kwargs);raw=copy.copy(actual)
        rear=rear6(actual,self.geometry);yaw=actual.actor_pose.yaw
        # A derived view makes the frozen world->rear functions land exactly at
        # true rear6. Actual annotation, full rotation and world2ego stay intact.
        dx,dy=self.geometry.axle_dx_m,self.geometry.axle_dy_m
        c,s=math.cos(yaw),math.sin(yaw)
        raw.actor_pose=ActorPose(float(rear[0]-c*dx+s*dy),float(rear[1]-s*dx-c*dy),yaw)
        return raw


class GravityTrajectory:
    def __init__(self,reader,native):self.reader,self.native=reader,native
    def build_targets(self,ref):
        errors,metrics=trajectory_diagnosis(self.reader,ref,self.native.geometry)
        hard=[e for e in errors if e['code']!='nonplanar_pose']
        if hard:raise AdapterError(hard[0]['code'],hard)
        origin=self.reader.read(ref);values=[];residual=[]
        if np.linalg.norm(carla_rotation(origin.ego_box['rotation'])[:2,0])<1e-6:
            raise AdapterError('degenerate_ego_heading','near-vertical body longitudinal axis')
        for fid in range(ref.frame_id+5,ref.frame_id+41,5):
            future=self.reader.read(self.reader.index.ref(fid));point=rear6(future,self.native.geometry)
            local=world_plane6(point,origin,self.native.geometry)
            residual.append(float(np.max(np.abs(plane6_world(local,origin,self.native.geometry)-point))))
            values.append([local[0],local[1],wrap_angle(origin.actor_pose.yaw-future.actor_pose.yaw)])
        return NativeBuildResult({'trajectory':torch.tensor(values,dtype=torch.float32)},
            {'geometry_contract':CONTRACT,'future_frame_ids':list(range(ref.frame_id+5,ref.frame_id+41,5)),
             'world_inverse_max_m':max(residual),'current_token':ref.token},dict(metrics,engineering='pass'))


def features6(reader,ref,native,decoder):
    history=[reader.read(reader.index.ref(i)) for i in range(ref.frame_id-2,ref.frame_id+1)]
    if any(f.state.get('teleport_flag',False) for f in history):raise AdapterError('teleport_flag','history flag semantics unresolved')
    if any(f.ego_box.get('type_id')!=native.geometry.axle_vehicle_type for f in history):raise AdapterError('reference_vehicle_mismatch',ref.token)
    raw=reader.read(ref,cameras=True,lidar=False)
    raw.lidar_xyz,meta=decode_lidar_in_process(ref,decoder)
    world=raw.lidar_xyz@carla_rotation(raw.ego_box['rotation']).T+np.asarray(raw.ego_box['location'])
    xyz=world_plane6(world,raw,native.geometry)
    keep=(xyz[:,2]>native.geometry.lidar_split_height_m)&(xyz[:,2]<native.geometry.lidar_max_height_m)
    edges=np.linspace(-32.,32.,257);hist=np.histogramdd(xyz[keep,:2],bins=(edges,edges))[0]
    lidar=torch.from_numpy((np.minimum(hist,5)/5).astype(np.float32)[None])
    rear_points=np.array([rear6(f,native.geometry) for f in history]);dt=native.geometry.source_dt_s
    v=(rear_points[-1,:2]-rear_points[-2,:2])/dt;a=(rear_points[-1,:2]-2*rear_points[-2,:2]+rear_points[-3,:2])/dt**2
    if np.linalg.norm(v)>native.geometry.max_speed_mps:raise AdapterError('teleport_or_discontinuity','history rear6 speed')
    yaw=raw.actor_pose.yaw;c,s=math.cos(yaw),math.sin(yaw);basis=np.array([[c,s],[s,-c]])
    near=np.array([float(raw.navigation[k]) for k in ('x_command_near','y_command_near')])
    local=(near-rear6(raw,native.geometry)[:2])@basis
    distance=float(np.linalg.norm(local));bearing=math.degrees(math.atan2(local[1],local[0]))
    if not np.isfinite(local).all() or local[0]<=0 or not native.navigation_min_distance_m<=distance<=native.navigation_max_distance_m:
        raise AdapterError('invalid_navigation','true rear6 near node outside frozen admissible policy')
    category=0 if bearing>native.navigation_bearing_threshold_deg else 2 if bearing<-native.navigation_bearing_threshold_deg else 1
    status=torch.from_numpy(np.r_[np.eye(4)[category],v@basis,a@basis].astype(np.float32))
    result={'camera_feature':proportional_camera_feature(raw),'lidar_feature':lidar,'status_feature':status}
    validate_features(result);return result,{'geometry_contract':CONTRACT,'decoder':meta,'history_frames':[f.ref.frame_id for f in history]}


def build_pair(config,row,native,model):
    reader=FrameReader(RouteIndex.from_dataset_root(config['dataset_root'],row['route']));ref=reader.index.ref(row['frame'])
    if ref.frame_id<2:raise AdapterError('insufficient_past',ref.token)
    actual=reader.read(ref);derived=ReferenceReader(reader,native.geometry)
    provider=RecoveryProvider(config,'projected')
    builder=RecoveryBuilder(derived,native,model,provider,policy='projected')
    builder.trajectory_builder=GravityTrajectory(reader,native)
    built=builder.build_targets(ref);targets=built.tensors;validate_targets(targets)
    features,proof=features6(reader,ref,native,config['lidar_decoder_python'])
    return features,targets,{'features':proof,'targets':CONTRACT,
        'source_map':{k:built.provenance['bev'][k] for k in ['map_source','map_sha256','map_version','map_town']},
        'target_provenance':built.provenance,'true_rear_world_xyz':rear6(actual,native.geometry).tolist()}


class Gravity6Dataset:
    def __init__(self,manifest):self.manifest=manifest;self.rows=manifest['rows']
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];path=Path(self.manifest['root'])/row['relative_path']
        if file_digest(path)!=row['sha256']:raise AdapterError('gravity6_cache_checksum',row['sample_id'])
        value=torch.load(path,map_location='cpu',weights_only=True)
        if value['sample_id']!=row['sample_id'] or value['producer_sha256']!=self.manifest['producer_sha256'] or value['contract']!=CONTRACT:
            raise AdapterError('gravity6_contract_mismatch',row['sample_id'])
        validate_features(value['features']);validate_targets(value['targets']);return value['features'],value['targets']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--limit',type=int,default=2);p.add_argument('--all',action='store_true');args=p.parse_args()
    from tools.training.diffusiondrive_v1.runtime import model_configuration,native_configuration
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    native,model=native_configuration(cfg),model_configuration(cfg);torch.set_num_threads(1)
    from tools.training.diffusiondrive_v1.features_cache import feature_producer
    from tools.training.diffusiondrive_v1.target_cache import producer_identity
    state=args.state.resolve();out=state/'gravity6';out.mkdir(parents=True,exist_ok=True)
    producer={'contract':CONTRACT,'implementation_sha256':file_digest(Path(__file__)),
        'rules_sha256':file_digest(Path(__file__).with_name('full_recovery_rules.py')),'dataset_id':cfg['dataset_id'],
        'base_feature_producer_sha256':digest(feature_producer(cfg)),
        'base_target_producer_sha256':digest(producer_identity(cfg,model,native)),
        'maps':{town:{'sha256':spec['sha256'],'version':spec['carla_version']} for town,spec in cfg['map_paths'].items()}};sha=digest(producer)
    if args.all:
        gate=out/'certified.json'
        if not gate.exists() or json.loads(gate.read_text()).get('producer_sha256')!=sha:raise StoragePause('Certify real regenerated Feature/Target batch before full gravity6 regeneration')
    if not args.all and not 1<=args.limit<=8:p.error('Use a bounded <=8 pilot, or explicit certified --all')
    with readonly(state/'recovery.sqlite') as db:
        rows=[dict(r) for r in db.execute("SELECT * FROM tasks WHERE status='needs_feature_contract' AND frame>=2 ORDER BY route,frame")]
    root=Path(cfg['target_root']['path']).parent/'diffusiondrive_v1_targets_recovery_20261017'/'gravity6_pairs'
    spec=dict(cfg['target_root'],path=str(root),budget_gib=32)
    storage_check(spec,True);root.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema':'b2d-v1-gravity6-paired-v1','producer':producer,'producer_sha256':sha,'root':str(root),'rows':[]}
    if manifest['producer_sha256']!=sha:raise StoragePause('Gravity6 state producer changed')
    complete={r['sample_id'] for r in manifest['rows']};failures=[]
    import fcntl
    with (out/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        used=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in root.rglob('*') if p.is_file())
        for row in rows:
            if row['sample_id'] in complete:continue
            if (out/'stop.request').exists():break
            if not args.all and (len(manifest['rows'])>=args.limit or len(failures)>=16):break
            sid=row['sample_id'];source=json.loads(row['diagnosis']);sources=list(source['sources'])
            # Add actual causal history plus raw cameras/LAZ; SHA before and after regenerating.
            ix=RouteIndex.from_dataset_root(cfg['dataset_root'],row['route']);ref=ix.ref(row['frame'])
            extra=[ix.path(ix.ref(fid),'anno') for fid in range(row['frame']-2,row['frame'])]
            extra+=[ix.path(ref,key) for key in ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','lidar']]
            sources.extend({'path':str(path.relative_to(Path(cfg['dataset_root']))),'sha256':file_digest(path),'bytes':path.stat().st_size} for path in extra)
            for item in sources:
                if file_digest(Path(cfg['dataset_root'])/item['path'])!=item['sha256']:raise StoragePause('Gravity6 original source SHA changed')
            try:features,targets,proof=build_pair(cfg,row,native,model)
            except AdapterError as exc:
                failures.append({'sample_id':sid,'error_code':exc.code,'error':str(exc)});continue
            relative=str(Path(sha)/sid[:2]/(sid+'.pt'));dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            payload={'schema':manifest['schema'],'sample_id':sid,'producer_sha256':sha,'contract':CONTRACT,'route':row['route'],'frame':row['frame'],
                'features':features,'targets':targets,'sources':sources,'proof':proof}
            if dest.exists():
                old=torch.load(dest,map_location='cpu',weights_only=True)
                if old['sample_id']!=sid or old['producer_sha256']!=sha or any(not torch.equal(targets[k],old['targets'][k]) for k in targets):raise StoragePause('Existing unknown/different gravity6 payload')
            else:
                storage_check(spec,True)
                if used+8*1024**2>spec['budget_gib']*1024**3:raise StoragePause('Gravity6 quota threshold')
                if __import__('shutil').disk_usage(root).free<spec['minimum_free_gib']*1024**3+8*1024**2:raise StoragePause('Gravity6 free space threshold')
                temp=dest.with_suffix('.partial')
                with temp.open('wb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
                os.replace(temp,dest);directory_sync(dest.parent)
                used+=max(dest.stat().st_size,dest.stat().st_blocks*512)
            for item in sources:
                if file_digest(Path(cfg['dataset_root'])/item['path'])!=item['sha256']:raise StoragePause('Gravity6 source changed while constructing')
            entry={'sample_id':sid,'route':row['route'],'frame':row['frame'],'relative_path':relative,'sha256':file_digest(dest)}
            manifest['rows'].append(entry);complete.add(sid);durable_json(manifest_path,manifest)
            restored=Gravity6Dataset(manifest)[len(manifest['rows'])-1]
            if any(not torch.equal(v,restored[0][k]) for k,v in features.items()) or any(not torch.equal(v,restored[1][k]) for k,v in targets.items()):raise StoragePause('Gravity6 pair readback mismatch')
            print(json.dumps({'gravity6_complete':sid,'count':len(manifest['rows'])}),flush=True)
    durable_json(out/'pilot_failures.json',failures)


if __name__=='__main__':main()
