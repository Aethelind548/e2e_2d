"""Bounded real reads and one optimizer step through the explicit paired reader.

No tensor caches, raw data, model sources or historical weights are written.
"""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def main():
    import torch
    from torch.utils.data import default_collate
    from tools.b2d_adapter import AdapterError
    from tools.b2d_adapter.v1_builders import official_v1_module
    from tools.training.diffusiondrive_v1.features_cache import file_digest,durable_json
    from tools.training.diffusiondrive_v1.runtime import model_configuration,fit_configuration
    from tools.training.diffusiondrive_v1.paired_dataset import paired_datasets
    from tools.training.diffusiondrive_v1.recovery_paired import readonly
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--backward',action='store_true')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'validation.json').exists():raise FileExistsError('Preserve prior acceptance output')
    manifest=json.loads(args.manifest.read_text());cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    cfg['allowed_recovery_producers']=[v['producer_sha256'] for v in manifest['target_sources'].values()]
    cfg['approved_new_geometry_contracts']=['b2d-v1-projected-geometry-v1']
    cfg['paired_manifest']=str(args.manifest.resolve());cfg.pop('prepared_manifest',None)
    cfg['data']={'train':[],'validation':[]};cfg['training']=dict(cfg['training'],num_workers=0,max_steps=1)
    cfg['threads']=2;cfg['output_dir']=str(out/'fit_input_check');cfg['tensor_memory_items']=2
    with readonly(manifest['pair_index']) as db:
        route=None;entry=None
        for row in db.execute('SELECT split,route,frame FROM pairs ORDER BY split,route,frame'):
            key=(row['split'],row['route'])
            if key!=route:
                entry={'route':str(Path(cfg['dataset_root'])/row['route']),'frames':[]}
                cfg['data'][row['split']].append(entry);route=key
            entry['frames'].append(row['frame'])
    config_path=out/'fit_configuration.json';durable_json(config_path,cfg)
    cfg,_=fit_configuration(config_path);model=model_configuration(cfg)
    torch.set_num_threads(2);torch.manual_seed(20261019)
    datasets,identity=paired_datasets(args.manifest,cfg,model)
    selected=[];by_source={}
    for split,dataset in datasets.items():
        for i,row in enumerate(dataset.rows):
            source=row['target_source']
            if len(by_source.get(source,[]))>=2 and (split!='validation' or any(r['split']=='validation' for r in selected)):continue
            pair=dataset[i]
            tsource=manifest['target_sources'][source];fsource=manifest['feature_sources'][row['feature_source']]
            with readonly(Path(tsource['state'])/'index.sqlite') as db:t=dict(db.execute('select * from samples where sample_id=?',(row['sample_id'],)).fetchone())
            with readonly(Path(fsource['state'])/'index.sqlite') as db:f=dict(db.execute('select * from samples where sample_id=?',(row['sample_id'],)).fetchone())
            all_sources={v['path']:v for v in json.loads(t['sources_json'])+json.loads(f['source_json'])}
            for item in all_sources.values():
                if file_digest(Path(cfg['dataset_root'])/item['path'])!=item['sha256']:
                    raise AdapterError('pair_source_mismatch',item['path'])
            town=next(k for k in cfg['map_paths'] if ('_'+k+'_') in row['route'])
            spec=cfg['map_paths'][town]
            assert file_digest(spec['path'])==spec['sha256']
            record={k:row[k] for k in ['sample_id','target_source','feature_source','route','frame','split']}
            record.update(raw_sources_sha_verified=len(all_sources),map_sha_verified=True,
                features={k:{'shape':list(v.shape),'dtype':str(v.dtype),'min':float(v.min()),'max':float(v.max())} for k,v in pair[0].items()},
                targets={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in pair[1].items()})
            selected.append(record)
            if split=='train':by_source.setdefault(source,[]).append(pair)
    assert set(by_source)==set(manifest['target_sources'])
    # Meaningful contract refusal checks; only copied metadata is changed.
    negatives=[]
    for kind in ['producer_whitelist','geometry_opt_in','index_sha']:
        altered=copy.deepcopy(cfg);path=args.manifest
        if kind=='producer_whitelist':altered['allowed_recovery_producers']=[]
        elif kind=='geometry_opt_in':altered['approved_new_geometry_contracts']=[]
        else:
            body=copy.deepcopy(manifest);body['index_sha256']='0'*64
            path=out/'invalid_index_manifest.json';durable_json(path,body)
        try:paired_datasets(path,altered,model)
        except AdapterError as exc:negatives.append({'case':kind,'rejected':True,'code':exc.code})
        else:raise AssertionError('Contract mutation accepted: '+kind)
    device=args.device
    if device=='cuda':assert torch.cuda.is_available()
    agent=official_v1_module('transfuser_agent').TransfuserAgent(model,cfg['learning_rate'],checkpoint_path=None).to(device).train()
    forwards=[];backward=None
    for source in ['compatible','projected']:
        features,targets=default_collate(by_source[source])
        features={k:v.to(device) for k,v in features.items()};targets={k:v.to(device) for k,v in targets.items()}
        optimize=args.backward and source=='projected'
        optimizer=agent.get_optimizers()['optimizer'] if optimize else None
        if optimizer:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimize):
            prediction=agent.forward(features,targets=targets);losses=agent.compute_loss(features,targets,prediction)
        assert all(torch.isfinite(v).all() for v in prediction.values() if isinstance(v,torch.Tensor))
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values() if v is not None)
        forwards.append({'source':source,'batch_size':len(targets['trajectory']),'device':device,
            'losses':{k:float(torch.as_tensor(v).detach().cpu()) for k,v in losses.items() if v is not None},
            'prediction_shapes':{k:list(v.shape) for k,v in prediction.items() if isinstance(v,torch.Tensor)}})
        if optimize:
            losses['loss'].backward();norms={};tracked=None;before=None
            for name,param in agent.named_parameters():
                if param.grad is None:continue
                assert torch.isfinite(param.grad).all(),name
                norm=float(param.grad.detach().float().norm())
                group='planning' if 'diffusion' in name or 'trajectory' in name else 'detection' if 'agent' in name else 'bev' if 'bev' in name else 'backbone_other'
                norms[group]=norms.get(group,0.)+norm**2
                if tracked is None and norm>0:tracked=(name,param);before=param.detach().clone()
            assert tracked is not None
            optimizer.step()
            assert all(torch.isfinite(v).all() for v in agent.parameters())
            delta=float((tracked[1].detach()-before).abs().max())
            assert delta>0
            backward={'optimizer_steps':1,'optimizer_class':type(optimizer).__name__,'finite_gradients':True,
                'gradient_l2_by_group':{k:v**.5 for k,v in norms.items()},'tracked_parameter':tracked[0],
                'max_abs_parameter_change':delta,'checkpoint_saved':False}
        del prediction,losses
    result={'dataset_identity':identity,'manifest':str(args.manifest.resolve()),'gravity6_included':manifest.get('gravity6_included'),
        'selected_reads':selected,'negative_contract_checks':negatives,'forwards':forwards,'backward':backward,
        'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
        'scope':'Bounded cache and engineering acceptance; no physical truth or trained performance certification.'}
    durable_json(out/'validation.json',result)
    print(json.dumps({'status':'passed','reads':len(selected),'forward_batches':len(forwards),'optimizer_steps':backward['optimizer_steps'] if backward else 0}))


if __name__=='__main__':main()
