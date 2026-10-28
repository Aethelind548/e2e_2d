"""Critical warm-start, accumulation, committed-resume and alarm contracts."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'repositories/DiffusionDrive'))

from tools.training.diffusiondrive_v1.evaluate_checkpoint import digest
from tools.training.diffusiondrive_v1.lr_experiment import accumulation_contract, warm_start, check_gradients, HealthStop
from tools.training.diffusiondrive_v1.lr_experiment import training_budget_contract, prune_health_checkpoints
from tools.training.diffusiondrive_v1.sampling import CommittedSampler
from tools.training.diffusiondrive_v1.training_progress import TrainingProgress, normalize_committed_progress
from tools.training.diffusiondrive_v1.training_health import alarms


def test_accumulation_is_explicit_new_contract():
    assert accumulation_contract({'batch_size':8,'training':{}}) is None
    cfg={'batch_size':8,'training':{'accumulate_grad_batches':8}}
    assert accumulation_contract(cfg)['effective_batch_size']==64
    for invalid in [0,-1,True,1.5]:
        cfg['training']['accumulate_grad_batches']=invalid
        with pytest.raises(ValueError):accumulation_contract(cfg)


def test_formal_budget_cannot_be_overridden():
    training={'freeze_budget':True,'max_steps':329400,'max_epochs':100,'validation_every_epochs':1,
              'checkpoint_every_steps':250,'checkpoint_monitor':'val/ade_m','health_checkpoint_keep':4}
    assert training_budget_contract({'training':training},329400)['max_epochs']==100
    with pytest.raises(ValueError):training_budget_contract({'training':training},2000)
    assert training_budget_contract({'training':{}},2) is None


def test_health_retention_only_removes_own_old_numbered_diagnostics(tmp_path):
    for step in [250,500,750,1000,1250]:
        (tmp_path/f'diagnostic-step{step:09d}.ckpt').write_text(str(step))
    preserved=['initial-step000000000.ckpt','restored-step000001250.ckpt','best.ckpt','diagnostic-stepbad.ckpt','diagnostic-step000001500.ckpt.partial']
    for name in preserved:(tmp_path/name).write_text('preserve')
    foreign=tmp_path/'foreign';foreign.write_text('outside diagnostic policy')
    (tmp_path/'diagnostic-step000001750.ckpt').symlink_to(foreign)
    removed=prune_health_checkpoints(tmp_path,4)
    assert [Path(p).name for p in removed]==['diagnostic-step000000250.ckpt']
    assert all((tmp_path/name).read_text()=='preserve' for name in preserved)
    assert foreign.read_text()=='outside diagnostic policy'
    assert (tmp_path/'diagnostic-step000001750.ckpt').is_symlink()
    assert prune_health_checkpoints(tmp_path,None)==[]
    with pytest.raises(ValueError):prune_health_checkpoints(tmp_path,0)


def test_warm_start_loads_buffers_but_not_optimization(tmp_path):
    agent=torch.nn.Linear(2,1);agent.register_buffer('marker',torch.tensor([7.]))
    target={k:torch.ones_like(v)*3 for k,v in agent.state_dict().items()}
    ck={'state_dict':{'agent.'+k:v for k,v in target.items()},'global_step':26347,'epoch':0,
        'b2d_v1_fit':{'semantic':{},'contract_sha256':digest({})},'optimizer_states':[{'state':{'old':999}}]}
    p=tmp_path/'parent.ckpt';torch.save(ck,p)
    result=warm_start(agent,p)
    assert all(torch.equal(agent.state_dict()[k],v) for k,v in target.items())
    assert result['global_step']==26347 and result['new_global_step']==0
    assert result['optimizer_restored'] is False and result['sampler_restored'] is False
    del ck['state_dict']['agent.bias'];torch.save(ck,tmp_path/'missing.ckpt')
    with pytest.raises(RuntimeError):warm_start(agent,tmp_path/'missing.ckpt')


def test_accumulation_boundary_preserves_prefetch_order():
    s=CommittedSampler(200,137);it=iter(s)
    dispatched=[next(it) for _ in range(160)]
    assert s.cursor==0
    assert s.acknowledge(64)==dispatched[:64]
    saved=s.state_dict();reloaded=CommittedSampler(200,999);reloaded.load_state_dict(saved)
    resumed=iter(reloaded)
    assert [next(resumed) for _ in range(64)]==dispatched[64:128]
    assert saved['cursor']==64


def test_progress_counts_updates_and_actual_samples(tmp_path):
    cfg={'training':{'accumulate_grad_batches':8,'validation_every_epochs':1}}
    p=TrainingProgress(tmp_path,2000,210776,8,cfg);p.start(0);p.commit(64,8)
    result=p.update(1,0,force=True)
    assert result['epoch_steps']==3294 and result['micro_batches_per_epoch']==26347
    assert result['consumed_samples']==64 and result['consumed_micro_batches']==8
    restored=TrainingProgress(tmp_path,2000,210776,8,cfg);restored.start(1,p.state_dict())
    assert restored.consumed_samples==64


def test_accumulation_periodic_resume_bookkeeping():
    p={'ready':16,'started':16,'processed':16,'completed':15}
    ck={'global_step':2,'b2d_v1_fit':{'semantic':{'accumulation':{'micro_batches':8}},
        'accumulation_state':{'pending_micro_batches':0,'last_committed_optimizer_step':2}},
        'loops':{'fit_loop':{'epoch_loop.batch_progress':{'total':dict(p),'current':dict(p)},
            'epoch_loop.automatic_optimization.optim_progress':{'optimizer':{'step':{'total':{'completed':2},'current':{'completed':2}}}}}}}
    assert normalize_committed_progress(ck)
    assert ck['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed']==16
    ck['b2d_v1_fit']['accumulation_state']['pending_micro_batches']=1
    with pytest.raises(RuntimeError):normalize_committed_progress(ck)


def test_nonfinite_gradients_stop_before_update():
    p=torch.nn.Parameter(torch.ones(2));p.grad=torch.tensor([0.,float('nan')])
    with pytest.raises(HealthStop):check_gradients([p])
    p.grad=torch.zeros(2);check_gradients([p])


def test_native_warmup_overwrites_paramwise_lr_without_lr_scale():
    import importlib
    from tools.b2d_adapter.v1_builders import V1_ROOT
    module=importlib.import_module('navsim.agents.diffusiondrive.modules.scheduler')
    assert Path(module.__file__).resolve()==(V1_ROOT/'navsim/agents/diffusiondrive/modules/scheduler.py').resolve()
    scheduler_type=module.WarmupCosLR
    params=[torch.nn.Parameter(torch.ones(1)),torch.nn.Parameter(torch.ones(1))]
    optimizer=torch.optim.AdamW([{'params':[params[0]],'lr':1e-4},{'params':[params[1]],'lr':5e-5}],weight_decay=1e-4)
    scheduler=scheduler_type(optimizer,min_lr=1e-6,lr=1e-4,warmup_epochs=3,epochs=100)
    assert [g['lr'] for g in optimizer.param_groups]==pytest.approx([1e-4/3]*2)
    for expected in [2e-4/3,1e-4,1e-4]:
        optimizer.step();scheduler.step()
        assert [g['lr'] for g in optimizer.param_groups]==pytest.approx([expected]*2)


def test_old_resume_cannot_silently_change_learning_rate_or_accumulation():
    from tools.training.diffusiondrive_v1.bf16_policy import validate_resume
    old={'lr':6e-4,'precision':'32-true'}
    parent={'b2d_v1_fit':{'semantic':old,'contract_sha256':digest(old)}}
    with pytest.raises(ValueError):validate_resume(parent,dict(old,lr=1e-4,accumulation={'micro_batches':8}))


def test_checkpoint_boundary_transition_only_allows_fit_source_change():
    from tools.training.diffusiondrive_v1.bf16_policy import POLICY, source_sha256, validate_resume
    old={'lr':1e-4,'precision':'bf16-mixed','precision_policy':POLICY,
         'precision_policy_source_sha256':'old-policy',
         'execution_source_sha256':{'tools/training/diffusiondrive_v1/fit.py':'old',
                                    'tools/training/diffusiondrive_v1/runtime.py':'same'}}
    new=copy.deepcopy(old);new['execution_source_sha256']['tools/training/diffusiondrive_v1/fit.py']='new'
    new['precision_policy_source_sha256']=source_sha256()
    parent={'b2d_v1_fit':{'semantic':old,'contract_sha256':digest(old)},'global_step':2250}
    transition=validate_resume(parent,new,False,True)
    assert transition['schema']=='explicit-checkpoint-boundary-fix-v1'
    assert transition['changed_sources']==['tools/training/diffusiondrive_v1/fit.py']
    assert transition['precision_policy_source_changed']
    with pytest.raises(ValueError):validate_resume(parent,new,False,False)
    extra=copy.deepcopy(new);extra['execution_source_sha256']['tools/training/diffusiondrive_v1/runtime.py']='changed'
    with pytest.raises(ValueError):validate_resume(parent,extra,False,True)
    numeric=copy.deepcopy(new);numeric['lr']=6e-4
    with pytest.raises(ValueError):validate_resume(parent,numeric,False,True)
    policy_only=copy.deepcopy(old);policy_only['precision_policy_source_sha256']=source_sha256()
    with pytest.raises(ValueError):validate_resume(parent,policy_only,False,True)


def test_preserved_checkpoint_defers_batch_save_at_restored_step(tmp_path,monkeypatch):
    from pytorch_lightning.callbacks import ModelCheckpoint
    from tools.training.diffusiondrive_v1.fit import PreservedCheckpoint
    calls=[]
    monkeypatch.setattr(ModelCheckpoint,'on_train_batch_end',
                        lambda self,trainer,*args,**kwargs:calls.append(trainer.global_step))
    callback=PreservedCheckpoint(dirpath=tmp_path,filename='checkpoint',every_n_train_steps=250,save_top_k=1)
    callback._resume_step=2250
    trainer=type('Trainer',(),{'global_step':2250})()
    callback.on_train_batch_end(trainer,None,None,None,0)
    assert calls==[]
    trainer.global_step=2251
    callback.on_train_batch_end(trainer,None,None,None,0)
    assert calls==[2251]
    fresh=PreservedCheckpoint(dirpath=tmp_path,filename='fresh',every_n_train_steps=250,save_top_k=1)
    fresh_trainer=type('Trainer',(),{'global_step':0})()
    fresh.on_train_batch_end(fresh_trainer,None,None,None,0)
    assert calls==[2251]
    fresh_trainer.global_step=1
    fresh.on_train_batch_end(fresh_trainer,None,None,None,0)
    assert calls==[2251,1]


def test_engineering_alarm_requires_sustained_validation_degradation():
    b={'validation':{'query_std':.3,'qk_std':6.,'attention_entropy':.5,'ade_m':2.,'fde_m':4.},'warnings':[]}
    c=copy.deepcopy(b);c['validation']['ade_m']=4
    warnings,stop=alarms(c,b,b);assert not stop
    c['warnings']=warnings
    assert alarms(c,b,c)[1]
    d=copy.deepcopy(b);d['validation']['query_std']=1e-8
    assert alarms(d,b,b)[1]


def test_real_lightning_accumulation_and_epoch_tail(tmp_path):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader,TensorDataset
    class Toy(pl.LightningModule):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(0.))
        def training_step(self,batch,batch_idx):return self.w*batch[0].mean()
        def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=.1)
    class Observe(pl.Callback):
        def __init__(self):self.rows=[]
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            self.rows.append((trainer.global_step,float(module.w.detach())))
    m=Toy();o=Observe()
    t=pl.Trainer(accelerator='cpu',devices=1,max_epochs=1,accumulate_grad_batches=8,logger=False,
        enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,callbacks=[o],default_root_dir=tmp_path)
    t.fit(m,DataLoader(TensorDataset(torch.ones(10)),batch_size=1))
    assert [step for step,_ in o.rows]==[0]*7+[1]*2+[2]
    assert [w for _,w in o.rows[:7]]==[0.]*7
    assert float(m.w.detach())==pytest.approx(-.125,abs=1e-6)  # tail is 2/8, not renormalized to 2/2.


def test_health_panel_preserves_rng_gradients_buffers_and_accepts_numeric_loss(tmp_path):
    import hashlib
    import random
    import numpy as np
    from tools.training.diffusiondrive_v1.training_health import HealthMonitor
    class FixtureAgent(torch.nn.Module):
        def __init__(self):
            super().__init__();self._transfuser_model=torch.nn.Module()
            self._transfuser_model._tf_decoder=torch.nn.TransformerDecoder(
                torch.nn.TransformerDecoderLayer(8,2,dim_feedforward=16,dropout=0,batch_first=True),3)
            self.query=torch.nn.Parameter(torch.randn(1,31,8));self.bn=torch.nn.BatchNorm1d(8)
        def forward(self,f):
            memory=self.bn(f['status_feature']).unsqueeze(1).expand(-1,5,-1)
            out=self._transfuser_model._tf_decoder(self.query,memory)
            trajectory=out[:,:3].transpose(1,2)+torch.randn(1,8,3)*.01
            boxes=out[:,1:,:5].clone();boxes[...,3:5]=boxes[...,3:5].abs()+1
            return {'trajectory':trajectory,'agent_states':boxes,'agent_labels':out[:,1:,0]}
        def compute_loss(self,f,t,p):return {'loss':p['trajectory'].abs().mean(),'diffusion_loss':0.0}
    class FixtureData:
        def __init__(self,rows):self.rows=rows
        def __getitem__(self,index):
            return {'status_feature':torch.zeros(8)}, {'trajectory':torch.zeros(8,3),
                'agent_states':torch.zeros(30,5),'agent_labels':torch.zeros(30,dtype=torch.bool)}
    source=tmp_path/'panel.jsonl';rows=[]
    for split in ['train','validation']:
        for group in ['stationary','low_creep','low_depart','moving']:
            for i in range(8):
                rows.append({'attempt':2,'condition':'best','sample_id':f'{split}-{group}-{i}',
                    'split':split,'group':group,'feature_sha':'f'*64,'target_sha':'a'*64,'seed':137+i})
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    datasets={s:FixtureData([r for r in rows if r['split']==s]) for s in ['train','validation']}
    agent=FixtureAgent().train()
    for p in agent.parameters():p.grad=torch.ones_like(p)
    gradients=[p.grad.clone() for p in agent.parameters()];buffers={k:v.clone() for k,v in agent.named_buffers()}
    random.seed(991);np.random.seed(991);torch.manual_seed(991)
    tr=torch.get_rng_state().clone();nr=np.random.get_state();pr=random.getstate()
    monitor=HealthMonitor(agent,datasets,{'health_monitor':{'every_optimizer_steps':250,
        'panel_source':str(source),'panel_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}},tmp_path/'health')
    result=monitor.run(0)
    assert result['samples']==64 and not result['stop'] and agent.training and agent.bn.training
    assert torch.equal(torch.get_rng_state(),tr) and random.getstate()==pr
    after=np.random.get_state();assert nr[0]==after[0] and np.array_equal(nr[1],after[1]) and nr[2:]==after[2:]
    assert all(torch.equal(p.grad,g) for p,g in zip(agent.parameters(),gradients))
    assert all(torch.equal(v,buffers[k]) for k,v in agent.named_buffers())
