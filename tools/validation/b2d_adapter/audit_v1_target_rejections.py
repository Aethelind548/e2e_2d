"""Bounded real-label audit, independent head regression and cache-only input check.

Outputs live in a new audit directory. All existing indexes are read-only;
no running service or existing tensor is altered. No synthetic future labels.
"""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')

from tools.training.diffusiondrive_v1.features_cache import digest,durable_json,file_digest
from tools.training.diffusiondrive_v1.target_retry_audit import readonly,component_checks

OUT = ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/target_recovery_audit'


def inventory():
    files = sorted((ROOT/'tools/b2d_adapter').glob('*.py'))
    files += [ROOT/'tools/training/diffusiondrive_v1'/n for n in
        ('target_cache.py','target_jobs.py','target_storage.py','target_service.py',
         'paired_dataset.py','prepared.py','prepare.py','runtime.py','target_recovery.py',
         'features_cache.py')]
    files += [ROOT/'repositories/DiffusionDrive/navsim/agents/diffusiondrive'/n for n in
        ('transfuser_features.py','transfuser_loss.py')]
    items=[]
    for path in files:
        source=path.read_text();tree=ast.parse(source);parents={}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node): parents[child]=node
        for node in ast.walk(tree):
            kind=None
            if isinstance(node,ast.Raise):kind='raise'
            elif isinstance(node,ast.Assert):kind='assert'
            elif isinstance(node,ast.ExceptHandler):kind='except'
            elif isinstance(node,ast.Continue):kind='continue'
            elif isinstance(node,ast.Return) and (node.value is None or isinstance(node.value,ast.Constant) and node.value.value is None):kind='return None'
            if kind is None:continue
            parent=parents.get(node);conditions=[];function='<module>'
            while parent is not None:
                if isinstance(parent,ast.If):conditions.append(ast.unparse(parent.test))
                if isinstance(parent,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    function=parent.name;break
                parent=parents.get(parent)
            code=None
            if isinstance(node,ast.Raise) and isinstance(node.exc,ast.Call):
                call=node.exc
                if call.args and isinstance(call.args[0],ast.Constant):code=call.args[0].value
            text=ast.get_source_segment(source,node) or kind
            if kind=='except':text='except '+(ast.unparse(node.type) if node.type else 'BaseException')+': '+ '; '.join(ast.unparse(b) for b in node.body)
            items.append({'file':str(path.relative_to(ROOT)),'line':node.lineno,'function':function,
                          'kind':kind,'code_or_message':code,'conditions':list(reversed(conditions)),
                          'source':text})
    items.sort(key=lambda v:(v['file'],v['line']))
    durable_json(OUT/'source_paths.json',{'files':{str(p.relative_to(ROOT)):file_digest(p) for p in files},
        'paths':items,'scope':'static inventory; reachability and frame impact are described separately in reject_reason_audit.md'})
    lines=['# 源码分支清单（自动提取）','',
        '包含全部显式 raise、except、continue、return None。continue/return None 不等于整帧拒绝；运行路径及影响见主审计。条件是语法祖先，不替代运行测试。','',
        '| 源码位置 / 函数 | 分支 | 条件 | 实际语句 |','|---|---|---|---|']
    for row in items:
        def esc(x):return str(x).replace('|','\\|').replace('\n','<br>')
        link=f"[{row['file']}:{row['line']}]({ROOT/row['file']}:{row['line']})"
        lines.append('| '+link+' / '+row['function']+' | '+row['kind']+' | '+esc(' AND '.join(row['conditions']))+' | '+esc(row['source'])+' |')
    (OUT/'source_paths.md').write_text('\n'.join(lines)+'\n')
    return len(items)


def settings():
    from tools.training.diffusiondrive_v1.runtime import model_configuration,native_configuration,bev_provider
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    return cfg,model_configuration(cfg),native_configuration(cfg),bev_provider(dict(cfg,bev_cache_dir=None))


def snapshot():
    cfg,model,native,provider=settings()
    from tools.training.diffusiondrive_v1.target_cache import producer_identity,current_annotation_source
    from tools.training.diffusiondrive_v1.feature_jobs import config_and_state
    with readonly(ROOT/cfg['target_state_dir']/'index.sqlite') as db:
        pinned=json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
        assert digest(pinned)==digest(producer_identity(cfg,model,native))
        stats=dict(db.execute('SELECT state,COUNT(*) FROM samples GROUP BY state'))
        codes=dict(db.execute("SELECT error_code,COUNT(*) FROM samples WHERE state != 'complete' GROUP BY error_code"))
        good={r['sample_id']:(r['current_anno_sha256'],r['route'],r['frame']) for r in db.execute("SELECT sample_id,current_anno_sha256,route,frame FROM samples WHERE state='complete'")}
        selected=[]
        for town in sorted(cfg['map_paths']):
            row=db.execute("SELECT * FROM samples WHERE state='complete' AND route LIKE ? ORDER BY ABS(frame-80),route LIMIT 1",('%_'+town+'_%',)).fetchone()
            selected.append(dict(row))
        for term in ('BlockedIntersection','PedestrianCrossing','ControlLoss','ParkingExit'):
            row=db.execute("SELECT * FROM samples WHERE state='complete' AND route LIKE ? ORDER BY ABS(frame-80),route LIMIT 1",(term+'_%',)).fetchone()
            if row and row['sample_id'] not in {r['sample_id'] for r in selected}:selected.append(dict(row))
    feature_ids=set();pair_ids=set();mismatches=[];features={};jobs=[]
    for entry in json.loads((ROOT/cfg['features_collection']).read_text())['jobs']:
        fcfg,state=config_and_state(ROOT/entry['config'])
        with readonly(state/'index.sqlite') as db:
            count=0
            for record in db.execute("SELECT * FROM samples WHERE state='complete'"):
                r=dict(record);sid=r['sample_id'];count+=1
                if sid in feature_ids:raise AssertionError('Duplicate feature sample')
                feature_ids.add(sid)
                if sid in good:
                    t=good[sid];source=current_annotation_source(json.loads(r['source_json']),r['route'],r['frame'])
                    if source['sha256']!=t[0] or (r['route'],r['frame'])!=t[1:]:mismatches.append(sid)
                    else:pair_ids.add(sid)
                if sid in {v['sample_id'] for v in selected}:
                    features[sid]={'job_id':entry['job_id'],'config':entry['config'],'row':r,'dataset_id':cfg['dataset_id'],
                        'producer_sha256':json.loads((ROOT/cfg['features_collection']).read_text())['producer_sha256']}
            jobs.append({'job':entry['job_id'],'complete':count})
    result={'target_states':stats,'rejections':codes,'target_producer_sha256':digest(pinned),
        'feature_jobs':jobs,'features_complete':len(feature_ids),'valid_intersection':len(pair_ids),
        'target_only':len(good.keys()-feature_ids),'feature_only':len(feature_ids-good.keys()),
        'pair_identity_mismatches':mismatches,'snapshot_note':'Features may continue after each read-only transaction; counts are not a cross-job wall-clock lock',
        'selected_samples':[{'sample_id':r['sample_id'],'route':r['route'],'frame':r['frame']} for r in selected]}
    durable_json(OUT/'snapshot.json',result)
    return cfg,model,native,provider,pinned,selected,features


def trajectory_checks(payload,reader,native):
    import numpy as np
    from tools.b2d_adapter.coordinate_transform import target_to_actor_world,wrap_angle
    from tools.training.diffusiondrive_v1.target_cache import reference_pose
    cur=reader.read(reader.index.ref(payload['frame']))
    delta=(native.geometry.axle_dx_m,native.geometry.axle_dy_m)
    assert digest(reference_pose(cur,native))==digest(payload['reference_pose'])
    ids=[payload['frame']+5*n for n in range(1,9)]
    assert ids==payload['future_frame_ids']
    errors=[];yaw=[];independent=[];pos=payload['targets']['trajectory'].numpy()
    # Independent rotation/reflection formula (not the generator's forward transform).
    c,s=math.cos(cur.actor_pose.yaw),math.sin(cur.actor_pose.yaw)
    origin=np.array([cur.actor_pose.x,cur.actor_pose.y])+np.array([[c,-s],[s,c]])@np.array(delta)
    for p,frame in zip(pos,ids):
        true=reader.read(reader.index.ref(frame)).actor_pose
        reconstructed=target_to_actor_world(p,cur.actor_pose,delta)
        errors.append(math.hypot(reconstructed.x-true.x,reconstructed.y-true.y))
        yaw.append(abs(wrap_angle(reconstructed.yaw-true.yaw)))
        fc,fs=math.cos(true.yaw),math.sin(true.yaw)
        rear=np.array([true.x,true.y])+np.array([[fc,-fs],[fs,fc]])@np.array(delta)
        displacement=rear-origin
        target=np.array([c*displacement[0]+s*displacement[1],s*displacement[0]-c*displacement[1],wrap_angle(cur.actor_pose.yaw-true.yaw)])
        independent.append(float(np.max(np.abs(target-p))))
    assert max(errors)<1e-4 and max(yaw)<1e-5 and max(independent)<1e-4
    return {'actor_inverse_max_m':max(errors),'yaw_inverse_max_rad':max(yaw),
            'independent_target_max_absolute_error':max(independent),'future_frame_ids':ids,
            'origin_sha_match':True,'physical_clock':'10Hz engineering assumption; historical clock not authenticated'}


def visual_samples():
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from tools.b2d_adapter import FrameReader
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1TargetBuilder,BEV_CLASSES
    from tools.training.diffusiondrive_v1.target_storage import load_target_row
    from tools.training.diffusiondrive_v1.target_cache import source_identity
    from tools.training.diffusiondrive_v1.paired_dataset import pair_checks,PairedDataset
    cfg,model,native,provider,producer,selected,features=snapshot()
    torch.set_num_threads(1);records=[];pair_rows=[];numeric_dir=OUT/'numeric_samples';numeric_dir.mkdir(exist_ok=True)
    colors=['#ededed','#666666','#65c9a5','#ffcc44','#aa77bb','#4477dd','#ee5566']
    for row in selected:
        sid=row['sample_id'];reader=FrameReader(__import__('tools.training.diffusiondrive_v1.target_jobs',fromlist=['AnnotationIndex']).AnnotationIndex.from_dataset_root(cfg['dataset_root'],row['route']))
        payload=load_target_row(row,cfg['target_root'],digest(producer))
        checks=trajectory_checks(payload,reader,native)
        sources,mapped,future=source_identity(reader,reader.index.ref(row['frame']),native,provider,Path(cfg['dataset_root']).resolve())
        assert digest(sources)==digest(payload['sources']) and digest(mapped)==digest(payload['map'])
        builder=DiffusionDriveV1TargetBuilder(reader,native,model,provider)
        fresh=builder.compute_targets(reader.index.ref(row['frame']))
        assert all(torch.equal(fresh[k],payload['targets'][k]) for k in fresh)
        target=payload['targets'];np.savez_compressed(numeric_dir/(sid+'.npz'),**{k:v.numpy() for k,v in target.items()})
        fig,axes=plt.subplots(1,3,figsize=(15,6),constrained_layout=True)
        # Decode this same frame's real front JPG directly; no new Feature cache.
        route=Path(cfg['dataset_root'])/row['route']
        camera=next((route/'camera/rgb_front').glob(f"{row['frame']:05d}.jpg"),None)
        if camera is None:
            from tools.b2d_adapter import RouteIndex
            ix=RouteIndex.from_dataset_root(cfg['dataset_root'],row['route']);camera=ix.path(ix.ref(row['frame']),'CAM_FRONT')
        with Image.open(camera) as im:axes[0].imshow(im)
        axes[0].set_title('Real CAM_FRONT');axes[0].axis('off')
        bev=target['bev_semantic_map'].numpy()
        # rot90(mask)[::-1] is a transpose: row x and column left-y both increase.
        im=axes[1].imshow(bev.T,origin='lower',extent=(0,32,-32,32),interpolation='nearest',aspect='equal',
            cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,7.5),7))
        p=target['trajectory'].numpy();axes[1].plot(p[:,0],p[:,1],'w.-',label='true future rear axle')
        axes[1].set_title('Stored BEV + future trajectory');axes[1].set_xlabel('forward x (m)');axes[1].set_ylabel('left y (m)')
        fig.colorbar(im,ax=axes[1],ticks=range(7),shrink=.7)
        from tools.b2d_adapter.navsim_bridge import candidate_vehicle_boxes
        from tools.b2d_adapter.v1_geometry import actor_local_hull
        raw=reader.read(reader.index.ref(row['frame']))
        view=candidate_vehicle_boxes(raw,native.geometry)
        axes[2].plot(p[:,0],p[:,1],'k.-',label='future rear axle')
        hull_errors=[]
        for box in raw.annotation['bounding_boxes']:
            if box.get('class')!='vehicle':continue
            hull,_=actor_local_hull(box,raw,native.geometry);x,y=hull.exterior.xy
            axes[2].plot(x,y,color='#bbbbbb',lw=.5)
        for state in target['agent_states'][target['agent_labels']].numpy():
            x,y,heading,length,width=state
            cs,sn=math.cos(heading),math.sin(heading)
            poly=np.array([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2],[-length/2,-width/2]])@np.array([[cs,sn],[-sn,cs]])+np.array([x,y])
            axes[2].plot(poly[:,0],poly[:,1],color='#4477dd',lw=1)
        axes[2].set_xlim(-32,48);axes[2].set_ylim(-40,40);axes[2].set_aspect('equal');axes[2].grid(alpha=.3)
        axes[2].set_title('Detection boxes / real 3D XY hulls');axes[2].set_xlabel('forward x (m)');axes[2].set_ylabel('left y (m)')
        axes[2].legend(fontsize=7)
        fig.suptitle(row['route'].split('_Weather')[0]+f" frame {row['frame']}",fontsize=11)
        image=OUT/(sid+'.png');fig.savefig(image,dpi=115);plt.close(fig)
        record={'sample_id':sid,'route':row['route'],'frame':row['frame'],'image':image.name,
            'trajectory':checks,'fresh_all_four_targets_bit_equal':True,'source_sha_match':True,
            'bev_pixels':torch.bincount(target['bev_semantic_map'].flatten(),minlength=7).tolist(),
            'detection_count':int(target['agent_labels'].sum()),'ego_pitch_roll_yaw_deg':raw.ego_box['rotation'],
            'camera_source':str(camera),'map_coverage':payload['provenance']['bev']['coverage']}
        records.append(record)
        if sid in features:
            f=features[sid];pair_checks(f,row,{'route':row['route'],'frame':row['frame']},native,producer)
            pair_rows.append({'sample_id':sid,'dataset_id':cfg['dataset_id'],'route':row['route'],'frame':row['frame'],
                'feature_config':f['config'],'feature_root_id':f['job_id']+':'+f['row']['root_id'],
                'target_root_id':cfg['target_root']['root_id'],'feature_producer_sha':f['producer_sha256'],
                'target_producer_sha':digest(producer),'feature_row':f['row'],'target_row':row})
        print(json.dumps({'visual_checked':row['route'],'frame':row['frame']}),flush=True)
    dataset=PairedDataset(pair_rows,cfg,producer)
    pairs=[dataset[i] for i in range(len(dataset))]
    durable_json(OUT/'sample_validation.json',{'samples':records,'paired_dataset_read_count':len(pairs),'classes':BEV_CLASSES,
        'status':'numeric checks passed; visual conclusions recorded after image inspection','scope':'bounded stratified sample, not a full semantic certification'})
    return pairs,cfg,model


def regression():
    import numpy as np
    import torch
    from tools.b2d_adapter import FrameReader
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1TargetBuilder
    from tools.training.diffusiondrive_v1.target_recovery import IrrelevantActorRecoveryBuilder,irrelevant_tilt
    from tools.training.diffusiondrive_v1.target_jobs import AnnotationIndex
    from tools.training.diffusiondrive_v1.target_storage import load_target_row
    cfg,model,native,provider=settings();torch.set_num_threads(1)
    recovery_cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_recovery_irrelevant_v1.json').read_text())
    with readonly(ROOT/recovery_cfg['target_state_dir']/'index.sqlite') as db:
        rows=[dict(r) for r in db.execute('SELECT * FROM samples ORDER BY route,frame')]
        recovered_producer=json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
    result=[];first=rows[0]
    reader=FrameReader(AnnotationIndex.from_dataset_root(cfg['dataset_root'],first['route']))
    base=DiffusionDriveV1TargetBuilder(reader,native,model,provider)
    alternative=IrrelevantActorRecoveryBuilder(reader,native,model,provider)
    for row in rows:
        ref=reader.index.ref(row['frame']);raw=reader.read(ref)
        actor_records=[]
        for box in raw.annotation['bounding_boxes']:
            if box.get('class')!='vehicle':continue
            tilt=float(np.max(np.abs(box['rotation'][:2])))
            evidence=irrelevant_tilt(box,raw,native.geometry,base.official_builder)
            actor_records.append({'actor_id':str(box['id']),'type_id':box.get('type_id'),'center':box.get('center'),
                'extent':box.get('extent'),'rotation':box.get('rotation'),'tilt_deg':tilt,
                'target_evidence':evidence,
                'detection_geometry_state':'tilt_rejected' if tilt>5 else 'planar_geometry_valid',
                'trajectory_dependency':'none; only ego future poses feed trajectory',
                'bev_numeric_geometry_state':'finite official planar and real 3D projection',
                'proof':evidence if tilt>5 else None})
        before,original=component_checks(base,ref)
        after,candidate=component_checks(alternative,ref)
        for key in original.keys() & candidate.keys():assert torch.equal(original[key],candidate[key])
        cached_equal=None
        if row['state']=='complete':
            cached=load_target_row(row,recovery_cfg['target_root'],digest(recovered_producer))
            assert after['full_training_eligible']
            assert all(torch.equal(candidate[k],cached['targets'][k]) for k in candidate)
            trajectory_checks(cached,reader,native);cached_equal=True
        rec={'sample_id':row['sample_id'],'route':row['route'],'frame':row['frame'],
            'prior_recovery_state':row['state'],'actors':actor_records,'baseline_components':before,
            'recovery_components':after,'unchanged_valid_targets':sorted(original.keys() & candidate.keys()),
            'existing_cache_bit_equal':cached_equal,'full_training_eligible':after['full_training_eligible']}
        result.append(rec)
        print(json.dumps({'regression_frame':row['frame'],'eligible':rec['full_training_eligible']}),flush=True)
    count=sum(r['full_training_eligible'] for r in result)
    assert len(result)==63 and count==26
    durable_json(OUT/'town13_per_frame_actor_target.json',{'frames':result,'total':63,'eligible':count,
        'still_rejected':63-count,'note':'Remaining heads can be numerically valid while the relevant tilted actor remains semantically unresolved; do not train partial records'})


def interface(pairs,cfg,model,*,output_path=None):
    """Check the original network, optionally writing a distinct new report.

    Explicit destinations must be new so recovery checks cannot replace frozen
    baseline evidence. The default retains the historical caller contract.
    """
    destination=Path(output_path) if output_path is not None else OUT/'interface_validation.json'
    if output_path is not None and destination.exists():
        raise FileExistsError(destination)
    import torch
    from torch.utils.data import default_collate
    from tools.b2d_adapter.v1_builders import official_v1_module
    torch.set_num_threads(2);torch.manual_seed(20261017)
    features,targets=default_collate([pairs[0],pairs[1]])
    # No checkpoint metric interpretation; only original-network compatibility.
    agent=official_v1_module('transfuser_agent').TransfuserAgent(model,cfg['learning_rate'],checkpoint_path=None)
    agent.train()
    with torch.no_grad():
        predictions=agent.forward(features,targets=targets)
        losses=agent.compute_loss(features,targets,predictions)
    assert all(torch.isfinite(v).all() for v in predictions.values() if isinstance(v,torch.Tensor))
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values() if v is not None)
    durable_json(destination,{'status':'pass','batch_size':2,'device':'cpu',
        'optimizer_steps':0,'checkpoint_used':False,'training_branch':True,'prediction_shapes':{k:list(v.shape) for k,v in predictions.items() if isinstance(v,torch.Tensor)},
        'feature_shapes':{k:list(v.shape) for k,v in features.items()},'target_shapes':{k:list(v.shape) for k,v in targets.items()},
        'weighted_official_losses':{k:float(v) for k,v in losses.items() if v is not None}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['inventory','samples','regression','all'],default='all');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.phase in ('all','inventory'):print(json.dumps({'source_paths':inventory()}),flush=True)
    if args.phase in ('all','regression'):regression()
    if args.phase in ('all','samples'):
        pairs,cfg,model=visual_samples();interface(pairs,cfg,model)


if __name__=='__main__':main()
