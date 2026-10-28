"""Runtime and frozen-source audit, with no changes to official model or Loss."""
import argparse
import ast
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
REPO=ROOT/'repositories/DiffusionDrive'
COMMIT='9b52ed0ec06b073d82d6f392ab084c7b301c8681'
sys.path.insert(0,str(REPO))


def frozen_sources():
    paths=subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only',COMMIT,'navsim/agents/diffusiondrive','navsim/planning/training/agent_lightning_module.py'],text=True).splitlines()
    result={}
    for rel in paths:
        if not rel.endswith('.py'):continue
        frozen=subprocess.check_output(['git','-C',str(REPO),'show',COMMIT+':'+rel]);current=(REPO/rel).read_bytes()
        result[rel]={'frozen_sha256':hashlib.sha256(frozen).hexdigest(),'current_sha256':hashlib.sha256(current).hexdigest(),'identical':frozen==current}
        if frozen!=current:
            if rel!='navsim/agents/diffusiondrive/modules/scheduler.py':raise RuntimeError('Unexpected official source change: '+rel)
            # Whitelist precisely the constructor API compatibility patch.
            old=ast.parse(frozen);new=ast.parse(current)
            def normalized(tree):
                for node in ast.walk(tree):
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='__init__' and node.args and isinstance(node.args[-1],ast.Name) and node.args[-1].id=='verbose':node.args.pop()
                return ast.dump(tree,include_attributes=False)
            if normalized(old)!=normalized(new):raise RuntimeError('Scheduler math differs from frozen reference')
            result[rel]['compatibility']='Only removed obsolete verbose positional argument to PyTorch _LRScheduler.__init__; AST otherwise equal.'
    return result


def verify_pins(config):
    pins=config.get('official_source_sha256')
    if not pins:return
    for rel,sha in pins.items():
        if hashlib.sha256((REPO/rel).read_bytes()).hexdigest()!=sha:raise RuntimeError('Pinned official V1 source changed: '+rel)


def object_info(obj):
    target=obj if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isclass(obj) else type(obj)
    path=Path(inspect.getfile(target)).resolve()
    return {'type':target.__module__+'.'+target.__qualname__,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from tools.training.diffusiondrive_v1.runtime import model_configuration
    from tools.b2d_adapter.v1_builders import official_v1_module
    from tools.training.diffusiondrive_v1.recovery_paired import RecoveryPairedDataset
    from navsim.planning.training.agent_lightning_module import AgentLightningModule
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    sources=frozen_sources();cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text());model_cfg=model_configuration(cfg)
    torch.set_num_threads(2);torch.manual_seed(20261019)
    agent=official_v1_module('transfuser_agent').TransfuserAgent(model_cfg,cfg['learning_rate'],checkpoint_path=None)
    network=agent._transfuser_model;wrapper=AgentLightningModule(agent)
    weights=torch.load(model_cfg.bkb_path,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,weights[k]) for k,v in network._backbone.image_encoder.state_dict().items())
    old_manifest=json.loads((ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/full_target_recovery/acceptance_20261019/multi_producer_manifest.json').read_text())
    data=RecoveryPairedDataset(old_manifest,cfg,'train');samples=[i for i,r in enumerate(data.rows) if r['target_source']=='baseline'][:2]
    features,targets=default_collate([data[i] for i in samples]);calls=set()
    def profile(frame,event,arg):
        if event!='call':return
        file=frame.f_code.co_filename
        if str(REPO) in file or 'diffusiondrivev2' in file.lower() or 'bndp' in file.lower():calls.add((file,frame.f_code.co_name))
    sys.setprofile(profile)
    try:
        with torch.no_grad():prediction=wrapper.agent.forward(features,targets);loss=wrapper.agent.compute_loss(features,targets,prediction)
        optimization=wrapper.configure_optimizers()
    finally:sys.setprofile(None)
    optimizer=optimization['optimizer'];scheduler=optimization['lr_scheduler'];scheduler=scheduler['scheduler'] if isinstance(scheduler,dict) else scheduler
    names={id(p):n for n,p in agent.named_parameters()}
    groups=[]
    for i,group in enumerate(optimizer.param_groups):
        param_names=[names[id(p)] for p in group['params']]
        groups.append({'index':i,'parameters':len(param_names),'numel':sum(p.numel() for p in group['params']),
            'lr':group['lr'],'initial_lr':group.get('initial_lr'),'weight_decay':group['weight_decay'],'examples':param_names[:5],
            'all_image_encoder':all('image_encoder' in name for name in param_names),'lr_scale':group.get('lr_scale')})
    anchor=network._trajectory_head.plan_anchor;a=np.load(model_cfg.plan_anchor_path,allow_pickle=False)
    assert np.array_equal(anchor.detach().numpy(),a.astype(np.float32)) and not anchor.requires_grad
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in loss.values() if v is not None)
    default_cfg=official_v1_module('transfuser_config').TransfuserConfig()
    changes={f.name:{'default':str(getattr(default_cfg,f.name)),'runtime':str(getattr(model_cfg,f.name))} for f in dataclasses.fields(model_cfg) if str(getattr(default_cfg,f.name))!=str(getattr(model_cfg,f.name))}
    assert set(changes)<={'plan_anchor_path','bkb_path'}
    from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
    dummy=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([dummy],lr=cfg['learning_rate'])
    sched=WarmupCosLR(opt,min_lr=1e-6,lr=cfg['learning_rate'],epochs=100,warmup_epochs=3);trace=[]
    for epoch in range(6):
        trace.append({'scheduler_last_epoch':sched.last_epoch,'lr':opt.param_groups[0]['lr']});opt.step();sched.step()
    result={'reference_commit':COMMIT,'sources':sources,'runtime':{
        'agent':object_info(agent),'network':object_info(network),'backbone':object_info(network._backbone),
        'planning_head':object_info(network._trajectory_head),'forward':object_info(agent.forward),
        'loss':object_info(official_v1_module('transfuser_agent').transfuser_loss),'lightning':object_info(wrapper),
        'configure_optimizers':object_info(wrapper.configure_optimizers),'get_optimizers':object_info(agent.get_optimizers),
        'optimizer':object_info(optimizer),'scheduler':object_info(scheduler)},
        'parameter_count':sum(p.numel() for p in network.parameters()),'trainable_parameter_count':sum(p.numel() for p in network.parameters() if p.requires_grad),
        'parameter_shapes_sha256':hashlib.sha256(json.dumps([(n,list(p.shape),p.requires_grad) for n,p in network.named_parameters()]).encode()).hexdigest(),
        'config_changes':changes,'optimizer_groups':groups,'scheduler_state':scheduler.state_dict(),
        'scheduler_epoch_trace':trace,'scheduler_lightning_default_interval':'epoch (official optimizer return has no interval override)',
        'backbone_initialization':{'path':model_cfg.bkb_path,'sha256':hashlib.sha256(Path(model_cfg.bkb_path).read_bytes()).hexdigest(),'all_image_encoder_state_equal':True},
        'anchor':{'path':model_cfg.plan_anchor_path,'shape':list(a.shape),'file_sha256':hashlib.sha256(Path(model_cfg.plan_anchor_path).read_bytes()).hexdigest(),
            'array_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'source_dtype':str(a.dtype),
            'runtime_dtype':str(anchor.dtype),'runtime_array_sha256':hashlib.sha256(anchor.detach().numpy().tobytes()).hexdigest(),
            'official_float32_cast_max_abs_error':float(np.max(np.abs(a.astype(np.float32).astype(a.dtype)-a))),
            'requires_grad':anchor.requires_grad,'exact_after_official_float32_cast':True},
        'observed_calls':[{'file':f,'function':n} for f,n in sorted(calls)],
        'foreign_model_calls':[{'file':f,'function':n} for f,n in sorted(calls) if 'diffusiondrivev2' in f.lower() or 'bndp' in f.lower()],
        'real_forward_loss_finite':True,'optimizer_updates':0}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'parameters':result['parameter_count'],'groups':groups,'foreign_model_calls':result['foreign_model_calls'],'anchor':result['anchor']}))


if __name__=='__main__':main()
