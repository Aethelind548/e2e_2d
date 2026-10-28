"""Certify bounded real recovery pilots before full profile apply is unlocked."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def forward(pairs,config,model):
    import torch
    from torch.utils.data import default_collate
    from tools.b2d_adapter.v1_builders import official_v1_module
    features,targets=default_collate(pairs[:2]);torch.set_num_threads(2);torch.manual_seed(20261017)
    agent=official_v1_module('transfuser_agent').TransfuserAgent(model,config['learning_rate'],checkpoint_path=None).train()
    with torch.no_grad():
        prediction=agent.forward(features,targets=targets);loss=agent.compute_loss(features,targets,prediction)
    assert all(torch.isfinite(v).all() for v in prediction.values() if isinstance(v,torch.Tensor))
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in loss.values() if v is not None)
    return {'batch_size':2,'device':'cpu','optimizer_steps':0,'all_predictions_and_losses_finite':True,
        'prediction_shapes':{k:list(v.shape) for k,v in prediction.items() if isinstance(v,torch.Tensor)},
        'loss':{k:float(v) for k,v in loss.items() if v is not None}}


def main():
    import torch
    from tools.training.diffusiondrive_v1.target_retry_audit import readonly,full_identity
    from tools.training.diffusiondrive_v1.features_cache import digest,durable_json
    from tools.training.diffusiondrive_v1.runtime import model_configuration,native_configuration
    from tools.training.diffusiondrive_v1.feature_jobs import config_and_state
    from tools.training.diffusiondrive_v1.paired_dataset import PairedDataset,pair_checks
    from tools.training.diffusiondrive_v1.full_recovery_rules import RecoveryBuilder,RecoveryProvider,MemoReader
    from tools.training.diffusiondrive_v1.target_jobs import AnnotationIndex
    from tools.validation.b2d_adapter.audit_v1_target_rejections import trajectory_checks
    from tools.training.diffusiondrive_v1.target_storage import load_target_row
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--profile',choices=['compatible','projected','gravity6'],required=True)
    args=p.parse_args();state=args.state.resolve();out=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/full_target_recovery'
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text());model,native=model_configuration(cfg),native_configuration(cfg)
    torch.set_num_threads(1)
    if args.profile=='gravity6':
        from tools.training.diffusiondrive_v1.gravity6_recovery import Gravity6Dataset
        from tools.training.diffusiondrive_v1.full_recovery_rules import rear6
        manifest=json.loads((state/'gravity6/manifest.json').read_text());dataset=Gravity6Dataset(manifest)
        assert len(dataset)>=2
        pairs=[dataset[i] for i in range(len(dataset))]
        cases=[]
        for row,(_,targets) in zip(manifest['rows'],pairs):
            reader=MemoReader(AnnotationIndex.from_dataset_root(cfg['dataset_root'],row['route']))
            raw=reader.read(reader.index.ref(row['frame']));rear=rear6(raw,native.geometry)
            c,s=math.cos(raw.actor_pose.yaw),math.sin(raw.actor_pose.yaw);residual=[]
            for pose,fid in zip(targets['trajectory'].numpy(),range(row['frame']+5,row['frame']+41,5)):
                true=rear6(reader.read(reader.index.ref(fid)),native.geometry)
                recovered=[rear[0]+c*pose[0]+s*pose[1],rear[1]+s*pose[0]-c*pose[1]]
                residual.append(math.hypot(recovered[0]-true[0],recovered[1]-true[1]))
            assert max(residual)<1e-4
            cases.append(dict(row,current_rotation_deg=raw.ego_box['rotation'],float32_future_rear_xy_inverse_max_m=max(residual)))
        result={'producer_sha256':manifest['producer_sha256'],'profile':'gravity6','real_dataset_reads':len(pairs),
            'feature_regenerated':True,'old_feature_reuse':False,'source_cases':cases,
            'forward':forward(pairs,cfg,model),'full_regeneration_started':False}
        durable_json(out/'certified_gravity6.json',result)
        # Deliberately no bulk gate: capacity has to be explicitly evaluated first.
        print(json.dumps(result));return
    profile=args.profile
    with readonly(state/profile/'index.sqlite') as db:
        meta={r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
        rows=[dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY route,frame")]
    producer=meta['producer'];assert digest(producer)==digest(full_identity(cfg,profile))
    collection=json.loads((ROOT/cfg['features_collection']).read_text());feature_rows={}
    for entry in collection['jobs']:
        fcfg,folder=config_and_state(ROOT/entry['config'])
        with readonly(folder/'index.sqlite') as db:
            for row in rows:
                feature=db.execute("SELECT * FROM samples WHERE state='complete' AND sample_id=?",(row['sample_id'],)).fetchone()
                if feature:feature_rows[row['sample_id']]={'row':dict(feature),'dataset_id':cfg['dataset_id'],
                    'config':entry['config'],'job_id':entry['job_id'],'producer_sha256':collection['producer_sha256']}
    paired=[];world=[]
    for row in rows:
        reader=MemoReader(AnnotationIndex.from_dataset_root(cfg['dataset_root'],row['route']))
        payload=load_target_row(row,meta['target_root'],digest(producer));world.append(trajectory_checks(payload,reader,native))
        if row['sample_id'] not in feature_rows:continue
        f=feature_rows[row['sample_id']];pair_checks(f,row,{'route':row['route'],'frame':row['frame']},native,producer)
        paired.append({'sample_id':row['sample_id'],'route':row['route'],'frame':row['frame'],'dataset_id':cfg['dataset_id'],
            'feature_config':f['config'],'feature_root_id':f['job_id']+':'+f['row']['root_id'],'target_root_id':meta['target_root']['root_id'],
            'feature_producer_sha':f['producer_sha256'],'target_producer_sha':digest(producer),'feature_row':f['row'],'target_row':row})
    config=dict(cfg,target_state_dir=str(state/profile),target_root=meta['target_root'])
    dataset=PairedDataset(paired,config,producer);assert len(dataset)>=2
    pairs=[dataset[i] for i in range(len(dataset))]
    result={'producer_sha256':digest(producer),'profile':profile,'complete_pilot_rows':len(rows),'paired_reads':len(pairs),
        'world_inverse_max_m':max(r['actor_inverse_max_m'] for r in world),'forward':forward(pairs,cfg,model)}
    if profile=='projected':
        # Every historical police frame, including the 37 formerly relevant ones,
        # must now produce complete Targets under the explicit new geometry.
        with readonly(ROOT/'outputs/diffusiondrive_v1/targets_recovery_irrelevant_v1/index.sqlite') as db:
            cases=[dict(r) for r in db.execute('SELECT * FROM samples ORDER BY frame')]
        provider=RecoveryProvider(cfg,'projected');reader=MemoReader(AnnotationIndex.from_dataset_root(cfg['dataset_root'],cases[0]['route']))
        builder=RecoveryBuilder(reader,native,model,provider,policy='projected');cases_report=[]
        for case in cases:
            value=builder.build_targets(reader.index.ref(case['frame']))
            proof=value.provenance['agents']['projected_actors']
            assert all(p['world_corner_residual_m']<=.01 for p in proof)
            cases_report.append({'frame':case['frame'],'status':'complete_new_geometry',
                'actors':proof,'agent_count':int(value.tensors['agent_labels'].sum()),
                'bev_pixels':torch.bincount(value.tensors['bev_semantic_map'].flatten(),minlength=7).tolist()})
        assert len(cases_report)==63;result['town13_projected_regression']=cases_report
    durable_json(out/('certified_'+profile+'.json'),result)
    durable_json(state/('certified_'+profile+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='town13_projected_regression'}))


if __name__=='__main__':main()
