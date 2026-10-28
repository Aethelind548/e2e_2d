"""V1 complete IL with grouped splits, validation, best/last weights and resume.

Explicit configuration controls the training horizon; the supplied M1 config is
a 16-step pilot. It consumes the full Adapter directly, never NAVSIM SceneLoader.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')

from pytorch_lightning.callbacks import ModelCheckpoint


class PreservedCheckpoint(ModelCheckpoint):
    def _generate_state_key(self,**kwargs):
        # Compatible with checkpoints written by the original ModelCheckpoint.
        suffix=repr(kwargs)
        candidates=getattr(self,'_saved_state_keys',())
        return next((key for key in candidates if key.endswith(suffix) and 'Checkpoint' in key),'ModelCheckpoint'+suffix)

    def load_state_dict(self,state):
        # Keep previous scores across directories, without deleting the parent run.
        super().load_state_dict(dict(state,dirpath=self.dirpath,last_model_path=''))

    def _remove_checkpoint(self,trainer,filepath):
        try:Path(filepath).resolve().relative_to(Path(self.dirpath).resolve())
        except ValueError:return
        super()._remove_checkpoint(trainer,filepath)

    def on_train_epoch_end(self,trainer,pl_module):
        # Lightning can finish the saved epoch's validation before taking a new step.
        # Its partially restored log accumulator is not a fresh selection panel.
        if 0<trainer.global_step<=getattr(self,'_resume_step',0):return
        super().on_train_epoch_end(trainer,pl_module)

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        # On a mid-epoch resume, global_step still equals the restored step
        # during the first accumulation window. Saving there would violate the
        # optimizer-boundary contract and raise in Evidence.on_save_checkpoint.
        if trainer.global_step<=getattr(self,'_resume_step',0):return
        super().on_train_batch_end(trainer,pl_module,outputs,batch,batch_idx)


def main():
    import numpy as np
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader,Dataset
    from pytorch_lightning.callbacks import ModelCheckpoint
    from navsim.planning.training.agent_lightning_module import AgentLightningModule
    from tools.b2d_adapter.v1_builders import official_v1_module,validate_v1_config,v1_native_config
    from tools.training.diffusiondrive_v1.runtime import fit_configuration,model_configuration,split_datasets,initialization_manifest,native_configuration
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('pilot_proportional.json'))
    parser.add_argument('--max-steps',type=int,help='Total optimizer step horizon, including resumed steps')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--prepared-manifest',type=Path,help='Explicit accepted manifest from prepare.py; workers never read raw routes')
    parser.add_argument('--paired-manifest',type=Path,help='Independent Features/Targets cache intersection manifest')
    parser.add_argument('--check-inputs',action='store_true',help='One real forward/full loss; no optimizer or Trainer.fit')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--num-workers',type=int,help='DataLoader execution override; numerical/data contract unchanged')
    parser.add_argument('--prefetch-factor',type=int,help='Batches prefetched by each worker')
    parser.add_argument('--mount-check-mode',choices=['findmnt','procfs'],help='Equivalent cache-read mount inspection backend')
    parser.add_argument('--cache-read-mode',choices=['audited','cache-only'],help='Read accepted cached tensors without live raw-data re-audit')
    parser.add_argument('--cache-file-read-mode',choices=['legacy','verified-single-pass'],help='Reuse the SHA-verified artifact bytes for torch.load; numerical contract unchanged')
    parser.add_argument('--allow-fp32-to-bf16',action='store_true',help='Explicit precision-only checkpoint contract transition; all other semantic fields must match')
    parser.add_argument('--allow-checkpoint-boundary-fix',action='store_true',help='One-time transition for checkpoints whose only semantic change is the fit.py accumulation-boundary source hash')
    parser.add_argument("--init-checkpoint",type=Path,default=None,)
    
    parser.add_argument('--pause-after-micro-batches',type=int,help='Smoke-only cooperative pause request; waits for optimizer boundary')
    parser.add_argument('--pause-after-steps',type=int,help='Cooperative pause after this many new optimizer steps')
    args=parser.parse_args();config,groups=fit_configuration(args.config)
    execution_overrides={}
    for key in ('num_workers','prefetch_factor'):
        value=getattr(args,key)
        if value is not None:
            if value<0 or (key=='prefetch_factor' and value==0):raise ValueError('Invalid '+key)
            config['training'][key]=value;execution_overrides[key]=value
    if args.mount_check_mode:
        config['mount_check_mode']=args.mount_check_mode
        execution_overrides['mount_check_mode']=args.mount_check_mode
    if args.cache_read_mode:
        config['cache_read_mode']=args.cache_read_mode
        execution_overrides['cache_read_mode']=args.cache_read_mode
    if args.cache_file_read_mode:
        config['cache_file_read_mode']=args.cache_file_read_mode
        execution_overrides['cache_file_read_mode']=args.cache_file_read_mode
    horizon=args.max_steps or config['training']['max_steps']
    if horizon<=0:raise ValueError('positive max steps required')
    workers=config['training']['num_workers']
    prepared_manifest=args.prepared_manifest or config.get('prepared_manifest')
    paired_manifest=args.paired_manifest or config.get('paired_manifest')
    if paired_manifest and prepared_manifest:
        raise ValueError('Select prepared or paired cache mode, not both')
    if config.get('cache_file_read_mode','legacy')!='legacy' and not (paired_manifest and config.get('cache_read_mode')=='cache-only'):
        raise ValueError('Single-pass artifact reading requires paired cache-only mode')
    if workers<0 or (workers and not (prepared_manifest or paired_manifest)):
        raise ValueError('multi-worker loading requires deterministic prepared tensors and nonnegative num_workers')
    output=args.output_dir or ROOT/config['output_dir']
    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError('--init-checkpoint and --resume are mutually exclusive')
    if args.allow_fp32_to_bf16 and args.allow_checkpoint_boundary_fix:
        raise ValueError('Precision and checkpoint-boundary transitions cannot be combined')
    if args.allow_checkpoint_boundary_fix and args.resume is None:
        raise ValueError('Checkpoint-boundary transition requires --resume')
    from tools.training.diffusiondrive_v1.lr_experiment import accumulation_contract, training_budget_contract, HealthStop
    accumulation = accumulation_contract(config)
    budget_contract = training_budget_contract(config,horizon)
    if any(value is not None and value<=0 for value in (args.pause_after_micro_batches,args.pause_after_steps)):
        raise ValueError('Cooperative pause limits must be positive')
    if config.get('execution_source_sha256'):
        for relative,expected in config['execution_source_sha256'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError('Experiment execution source SHA mismatch: '+relative)
    if config.get('warm_start_source') and not (args.init_checkpoint or args.resume):
        raise ValueError('Weight-only experiment requires --init-checkpoint or its own --resume')
    def output_storage_check():
        if config.get('output_storage'):
            from tools.training.diffusiondrive_v1.target_cache import storage_check
            spec=dict(config['output_storage'],path=str(output))
            free=storage_check(spec,True)
            if free<spec['minimum_free_gib']*1024**3:raise RuntimeError('Checkpoint output below free-space reserve')
    output_storage_check();output.mkdir(parents=True,exist_ok=True)
    if (output/'fit_result.json').exists():raise FileExistsError('use a fresh output directory; prior run results are preserved')
    report={'status':'running','max_steps':horizon,'steps':[],'validation':[],
            'execution_overrides':execution_overrides,
            'route_groups':{k:sorted(v) for k,v in groups.items()}}
    (output/'configuration.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    try:
        from tools.training.diffusiondrive_v1.official_v1_audit import verify_pins
        verify_pins(config)
        torch.set_num_threads(config.get('threads',4));pl.seed_everything(config['seed'],workers=True)
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU nodes must be exposed to this process')
        report['runtime']={'torch':torch.__version__,'device':args.device,'threads':torch.get_num_threads()}
        if args.device=='cuda':
            properties=torch.cuda.get_device_properties(0)
            report['runtime'].update(gpu=properties.name,gpu_total_bytes=properties.total_memory,cuda=torch.version.cuda)
        bf16=config['training']['precision']=='bf16-mixed'
        if bf16:
            from tools.training.diffusiondrive_v1.bf16_policy import POLICY,source_sha256
            if config.get('precision_policy')!=POLICY:raise ValueError('BF16 requires the explicit FP32 loss/evaluation policy')
            if config.get('precision_policy_source_sha256')!=source_sha256():raise ValueError('Precision policy source SHA mismatch')
            if args.device!='cuda' or not torch.cuda.is_bf16_supported():raise RuntimeError('Native CUDA BF16 support required')
        elif config.get('precision_policy') or args.allow_fp32_to_bf16:
            raise ValueError('Precision policy/transition requires bf16-mixed')
        model_config=model_configuration(config);initialization=initialization_manifest(config,model_config)
        from tools.training.diffusiondrive_v1.sampling import CommittedSampler
        from tools.training.diffusiondrive_v1.prepared import prepared_datasets
        prepared_identity=None
        if paired_manifest:
            from tools.training.diffusiondrive_v1.paired_dataset import paired_datasets
            raw_data,prepared_identity=paired_datasets(ROOT/paired_manifest,config,model_config)
            provider=None
            report['paired_data']=prepared_identity
            report['cache_reader']={'mode':config.get('cache_read_mode','audited'),
                'file_read_mode':config.get('cache_file_read_mode','legacy'),
                'mount_check_mode':config.get('mount_check_mode','procfs' if config.get('cache_read_mode')=='cache-only' else 'findmnt'),
                'live_raw_reaudit':config.get('cache_read_mode','audited')!='cache-only'}
        elif prepared_manifest:
            raw_data,prepared_identity=prepared_datasets(ROOT/prepared_manifest,config,model_config)
            provider=None
            report['prepared_data']=prepared_identity
        else:
            raw_data,provider=split_datasets(config,model_config)
        semantic={'data':config['data'],'map_paths':config['map_paths'],
                  'bev_producer_sha256':provider.producer_sha256 if provider else None,
                  'prepared_data':prepared_identity,'sampler_schema':CommittedSampler.schema,
                  'adapter':validate_v1_config(native_configuration(config),model_config),'initialization':initialization,
                  'seed':config['seed'],'batch_size':config['batch_size'],'lr':config['learning_rate'],
                  'precision':config['training']['precision'],'evaluation_noise_seed':config['evaluation_noise_seed']}
        if accumulation:semantic['accumulation']=accumulation
        if budget_contract:semantic['training_budget']=budget_contract
        if config.get('execution_source_sha256'):semantic['execution_source_sha256']=config['execution_source_sha256']
        if config.get('warm_start_source'):semantic['warm_start_source']=config['warm_start_source']
        if config.get('health_monitor'):semantic['health_monitor']=config['health_monitor']
        if bf16:
            semantic['precision_policy']=config['precision_policy']
            semantic['precision_policy_source_sha256']=config['precision_policy_source_sha256']
        if paired_manifest:
            semantic['data_mode']='independent_features_targets'
        if config.get('official_source_sha256'):
            semantic['official_source_sha256']=config['official_source_sha256']
            semantic['official_reference_commit']=config['official_reference_commit']
        if config.get('reference_eval_batches'):
            semantic['reference_eval_batches']=config['reference_eval_batches']
        from tools.b2d_adapter.v1_label_rules import label_rules
        rules=label_rules(config.get('label_rules'))
        if rules.enriched:
            # Raw Dataset resume must distinguish label rules as strictly as prepared manifests.
            semantic['label_rules']=rules.identity()
            semantic['bev_geometry_sha256']=hashlib.sha256((ROOT/'tools/b2d_adapter/v1_geometry.py').read_bytes()).hexdigest()
        contract_hash=hashlib.sha256(json.dumps(semantic,sort_keys=True).encode()).hexdigest()
        report['contract_sha256']=contract_hash;report['initialization']=initialization

        class TensorMemory(Dataset):
            # Bounded pilot cache; large datasets keep at most the explicit configured count.
            def __init__(self,source):
                from collections import OrderedDict
                self.source=source;self.cache=OrderedDict();self.capacity=config.get('tensor_memory_items',8)
                self.tokens=[]
                if hasattr(source,'tokens'):self.tokens=list(source.tokens)
                else:
                    for dataset in source.datasets:self.tokens.extend(ref.token for ref in dataset.refs)
            def __len__(self):return len(self.source)
            def __getitem__(self,i):
                if i in self.cache:
                    self.cache.move_to_end(i);return self.cache[i]
                value=self.source[i];self.cache[i]=value
                while len(self.cache)>self.capacity:self.cache.popitem(last=False)
                return value

        train_data=TensorMemory(raw_data['train']);val_data=TensorMemory(raw_data['validation'])
        sampler=CommittedSampler(len(train_data),config['seed'])
        from tools.training.diffusiondrive_v1.training_progress import TrainingProgress,normalize_committed_progress
        progress=TrainingProgress(output,horizon,len(train_data),config['batch_size'],config)
        loader_generator=torch.Generator().manual_seed(config['seed']+1)
        # Spawn avoids forking an initialized CUDA context. Precomputed items are deterministic.
        loader_options={'num_workers':workers,'pin_memory':args.device=='cuda'}
        if workers:loader_options.update(multiprocessing_context='spawn',prefetch_factor=config['training'].get('prefetch_factor',2),
                                        persistent_workers=config['training'].get('persistent_workers',False))
        # Keep the picklable prepared dataset directly in workers; memory LRU is local-only.
        training_source=raw_data['train'] if workers else train_data
        validation_source=raw_data['validation'] if workers else val_data
        train_loader=DataLoader(training_source,batch_size=config['batch_size'],sampler=sampler,**loader_options,generator=loader_generator)
        val_loader=DataLoader(validation_source,batch_size=config['batch_size'],shuffle=False,**loader_options,
                              generator=torch.Generator().manual_seed(config['seed']+2))
        agent=official_v1_module('transfuser_agent').TransfuserAgent(model_config,config['learning_rate'],checkpoint_path=None)
        if args.init_checkpoint:
            from tools.training.diffusiondrive_v1.lr_experiment import warm_start
            report['warm_start']=warm_start(agent,args.init_checkpoint)
            expected=config.get('warm_start_source')
            if expected and any(report['warm_start'].get(k)!=expected.get(k) for k in ('path','sha256','global_step')):
                raise ValueError('Warm-start source differs from frozen experiment configuration')
            print(json.dumps({'warm_start':report['warm_start']}),flush=True)
        if bf16:
            from tools.training.diffusiondrive_v1.bf16_policy import install_fp32_losses
            install_fp32_losses(agent)
            report['precision_policy']={'name':config['precision_policy'],'parameters':'float32',
                'optimizer_state':'float32','internal_and_external_loss':'float32','validation':'float32',
                'bf16_scope':'backbone Conv2d only; normalization, fusion, all heads and geometry remain FP32',
                'source_sha256':config['precision_policy_source_sha256']}
        if args.check_inputs:
            if args.resume:raise ValueError('Input-only acceptance does not restore training state')
            features,targets=next(iter(train_loader))
            features={k:v.to(args.device) for k,v in features.items()}
            targets={k:v.to(args.device) for k,v in targets.items()}
            agent.to(args.device).train()
            with torch.no_grad(),torch.autocast(args.device,dtype=torch.bfloat16,enabled=bf16):
                predictions=agent.forward(features,targets=targets)
                losses=agent.compute_loss(features,targets,predictions)
            if not all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values() if v is not None):
                raise RuntimeError('Nonfinite paired input acceptance loss')
            if not all(torch.isfinite(v).all() for v in predictions.values() if isinstance(v,torch.Tensor)):
                raise RuntimeError('Nonfinite input acceptance prediction')
            report.update(status='input_check_pass',optimizer_steps=0,batch_size=len(features['status_feature']),
                loss_values={k:float(v) for k,v in losses.items() if v is not None},
                prediction_shapes={k:list(v.shape) for k,v in predictions.items() if isinstance(v,torch.Tensor)},
                feature_shapes={k:list(v.shape) for k,v in features.items()},target_shapes={k:list(v.shape) for k,v in targets.items()})
            (output/'fit_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'status':report['status'],'optimizer_steps':0,'batch_size':report['batch_size'],'loss_values':report['loss_values']},ensure_ascii=False))
            return
        class B2DLightningModule(AgentLightningModule):
            def _step(self,batch,logging_prefix):
                self._actual_batch_size=len(batch[0]['status_feature'])
                if logging_prefix=='val':
                    # Observe trajectory error on the same official evaluation
                    # forward; no extra sampling pass or modified loss.
                    features,targets=batch
                    prediction=self.agent.forward(features,targets)
                    loss_dict=self.agent.compute_loss(features,targets,prediction)
                    for key,value in loss_dict.items():
                        if value is not None:
                            self.log(f'val/{key}',value,on_step=True,on_epoch=True,
                                     prog_bar=True,sync_dist=True,batch_size=self._actual_batch_size)
                    error=torch.linalg.norm(prediction['trajectory'][...,:2]-targets['trajectory'][...,:2],dim=-1)
                    self.log('val/ade_m',error.mean(),on_step=False,on_epoch=True,
                             sync_dist=True,batch_size=self._actual_batch_size)
                    self.log('val/fde_m',error[:,-1].mean(),on_step=False,on_epoch=True,
                             sync_dist=True,batch_size=self._actual_batch_size)
                    return loss_dict['loss']
                return super()._step(batch,logging_prefix)
            def log(self,*args,**kwargs):
                # The original wrapper counts dict keys, which biases partial-batch metrics.
                if 'batch_size' in kwargs:kwargs['batch_size']=self._actual_batch_size
                return super().log(*args,**kwargs)

        module=B2DLightningModule(agent)
        resumed=None
        transition=None
        if args.resume:
            resumed=torch.load(args.resume,map_location='cpu',weights_only=False)
            from tools.training.diffusiondrive_v1.bf16_policy import validate_resume
            transition=validate_resume(resumed,semantic,args.allow_fp32_to_bf16,args.allow_checkpoint_boundary_fix)
            if transition:
                from tools.training.diffusiondrive_v1.features_cache import file_digest
                transition.update(parent_checkpoint=str(args.resume.resolve()),parent_checkpoint_sha256=file_digest(args.resume))
                if transition.get('schema')=='explicit-checkpoint-boundary-fix-v1':
                    report['checkpoint_boundary_fix']=transition
                else:
                    report['precision_transition']=transition
            if horizon<=resumed['global_step']:raise ValueError('resume horizon must exceed checkpoint global_step')
            module.load_state_dict(resumed['state_dict'],strict=True)
        elif args.allow_fp32_to_bf16:
            raise ValueError('Precision transition requires a parent checkpoint')
        elif not args.init_checkpoint:
            weights=torch.load(model_config.bkb_path,map_location='cpu',weights_only=True)
            if not all(torch.equal(v,weights[k]) for k,v in agent._transfuser_model._backbone.image_encoder.state_dict().items()):
                raise RuntimeError('official image initialization mismatch')

        def evaluate(loader):
            device=torch.device(args.device);module.to(device).eval();totals=[];ades=[];fdes=[];sizes=[];components=[]
            # Fixed panel for before/after learning signal; neither consumes training RNG.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device=='cuda' else []),torch.no_grad():
                for batch_idx,(features,targets) in enumerate(loader):
                    if config.get('reference_eval_batches') and batch_idx>=config['reference_eval_batches']:break
                    torch.manual_seed(config['evaluation_noise_seed']+batch_idx)
                    f={k:v.to(device) for k,v in features.items()};t={k:v.to(device) for k,v in targets.items()}
                    prediction=agent.forward(f);loss=agent.compute_loss(f,t,prediction)
                    if not torch.isfinite(loss['loss']):raise RuntimeError('nonfinite fixed evaluation loss')
                    error=torch.linalg.norm(prediction['trajectory'][...,:2]-t['trajectory'][...,:2],dim=-1)
                    totals.append(float(loss['loss']));ades.append(float(error.mean()));fdes.append(float(error[:,-1].mean()))
                    components.append({key:float(value) for key,value in loss.items() if value is not None})
                    sizes.append(len(features['status_feature']))
            return {'official_eval_loss':float(np.average(totals,weights=sizes)),
                    'ade_m':float(np.average(ades,weights=sizes)),'fde_m':float(np.average(fdes,weights=sizes)),
                    'weighted_official_losses':{key:float(np.average([row[key] for row in components],weights=sizes)) for key in components[0]},
                    'batches':len(totals),'samples':sum(sizes)}

        report['before']={'train_reference':evaluate(DataLoader(train_data,batch_size=config['batch_size'],shuffle=False)),
                          'held_out_validation':evaluate(val_loader)}
        module.train()

        health=None
        if config.get('health_monitor'):
            from tools.training.diffusiondrive_v1.training_health import HealthMonitor
            health=HealthMonitor(agent,raw_data,config,output/'health')

        class Evidence(pl.Callback):
            def __init__(self):
                self.before=None;self.gradients=None;self.val_rng=None;self.resume_state=None
                self.val_position=0;self.batch_started=None;self.expected_resume_indices=None
                self.observe=True;self.total_samples=0;self.total_batches=0
                self.total_updates=0;self.pending_micro=0;self.pending_samples=0;self.window_indices=[]
                self.did_update=False;self.loss_sums={};self.window_started=None;self.micro_unchanged=0
            def state_key(self):return 'b2d_v1_fit_state'
            def on_save_checkpoint(self,trainer,module,checkpoint):
                output_storage_check()
                if self.pending_micro:raise RuntimeError('Cannot checkpoint uncommitted accumulated gradients')
                np_state=np.random.get_state()
                checkpoint['b2d_v1_fit']={'contract_sha256':contract_hash,'semantic':semantic,'config':config,
                    'sampler':sampler.state_dict(),'loader_generator':loader_generator.get_state(),
                    'python_rng':random.getstate(),'numpy_rng':{'kind':np_state[0],'keys':np_state[1].tolist(),'pos':np_state[2],
                        'has_gauss':np_state[3],'cached_gaussian':np_state[4]},
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device=='cuda' else []}
                checkpoint['b2d_v1_fit']['progress']=progress.state_dict()
                checkpoint['b2d_v1_fit']['accumulation_state']={'pending_micro_batches':0,'last_committed_optimizer_step':trainer.global_step}
                warm=report.get('warm_start') or (resumed or {}).get('b2d_v1_fit',{}).get('warm_start')
                if warm:checkpoint['b2d_v1_fit']['warm_start']=warm
                if health:checkpoint['b2d_v1_fit']['health']=health.state_dict()
                lineage=(transition or (resumed or {}).get('b2d_v1_fit',{}).get('checkpoint_boundary_fix')
                         or (resumed or {}).get('b2d_v1_fit',{}).get('precision_transition'))
                if lineage:
                    key=('checkpoint_boundary_fix' if lineage.get('schema')=='explicit-checkpoint-boundary-fix-v1'
                         else 'precision_transition')
                    checkpoint['b2d_v1_fit'][key]=lineage
            def on_load_checkpoint(self,trainer,module,checkpoint):
                report['normalized_checkpoint_batch_progress']=normalize_committed_progress(checkpoint)
                s=checkpoint['b2d_v1_fit']
                from tools.training.diffusiondrive_v1.bf16_policy import validate_resume
                validate_resume(checkpoint,semantic,args.allow_fp32_to_bf16,args.allow_checkpoint_boundary_fix)
                if s['sampler']['order']:
                    sampler.load_state_dict(s['sampler'])
                elif checkpoint['global_step']==0 and s['sampler']['cursor']==0 and s['sampler']['size']==len(train_data):
                    sampler.generator.set_state(s['sampler']['generator'])
                else:raise ValueError('Invalid empty sampler order')
                if accumulation and s.get('accumulation_state',{}).get('pending_micro_batches')!=0:
                    raise ValueError('Accumulation checkpoint is not at an optimizer boundary')
                if health and s.get('health'):health.load_state_dict(s['health'])
                loader_generator.set_state(s['loader_generator'])
                random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['kind'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
                torch.set_rng_state(s['torch_rng']);
                if s['cuda_rng']:torch.cuda.set_rng_state_all(s['cuda_rng'])
                self.resume_state=s
            def on_train_start(self,trainer,module):
                import inspect
                opt=trainer.optimizers[0];sched=trainer.lr_scheduler_configs[0]
                report['optimization_runtime']={'optimizer_type':type(opt).__module__+'.'+type(opt).__qualname__,
                    'precision_plugin':type(trainer.precision_plugin).__module__+'.'+type(trainer.precision_plugin).__qualname__,
                    'model_type':type(agent._transfuser_model).__module__+'.'+type(agent._transfuser_model).__qualname__,
                    'network_parameter_count':sum(p.numel() for p in agent._transfuser_model.parameters()),
                    'official_configure_optimizers_inherited':type(module).configure_optimizers is AgentLightningModule.configure_optimizers,
                    'official_training_step_inherited':type(module).training_step is AgentLightningModule.training_step,
                    'configure_optimizers_source':inspect.getfile(type(module).configure_optimizers),
                    'agent_forward_source':inspect.getfile(agent.forward),
                    'scheduler_type':type(sched.scheduler).__module__+'.'+type(sched.scheduler).__qualname__,
                    'scheduler_source':inspect.getfile(type(sched.scheduler)),'interval':sched.interval,'frequency':sched.frequency,
                    'groups':[{'lr':g['lr'],'initial_lr':g.get('initial_lr'),'weight_decay':g['weight_decay'],
                               'parameter_tensors':len(g['params'])} for g in opt.param_groups]}
                if resumed:
                    opt=trainer.optimizers[0];saved_opt=resumed['optimizer_states'][0]
                    actual=[state for state in opt.state.values() if 'exp_avg' in state]
                    expected=[state for state in saved_opt['state'].values() if 'exp_avg' in state]
                    assert len(actual)==len(expected)
                    if not actual and resumed['global_step']!=0:raise RuntimeError('Missing optimizer moments')
                    if actual:
                        assert float(actual[0]['step'])==float(expected[0]['step'])
                        torch.testing.assert_close(actual[0]['exp_avg'].cpu(),expected[0]['exp_avg'],rtol=0,atol=0)
                    if config.get('strict_resume_audit'):
                        for a,e in zip(actual,expected):
                            for key in e:
                                if isinstance(e[key],torch.Tensor):torch.testing.assert_close(a[key].cpu(),e[key].cpu(),rtol=0,atol=0)
                                else:assert a[key]==e[key]
                        assert opt.state_dict()['param_groups']==saved_opt['param_groups']
                        assert trainer.lr_scheduler_configs[0].scheduler.state_dict()==resumed['lr_schedulers'][0]
                    assert trainer.global_step==resumed['global_step']
                    assert trainer.lr_scheduler_configs[0].scheduler.last_epoch==resumed['lr_schedulers'][0]['last_epoch']
                    # Initialization/worker startup may consume incidental RNG; start
                    # optimization at the saved stream, after Trainer setup finishes.
                    s=self.resume_state;random.setstate(s['python_rng'])
                    n=s['numpy_rng'];np.random.set_state((n['kind'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
                    torch.set_rng_state(s['torch_rng'])
                    if s['cuda_rng']:torch.cuda.set_rng_state_all(s['cuda_rng'])
                    assert torch.equal(torch.get_rng_state(),s['torch_rng'])
                    if s['cuda_rng']:assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),s['cuda_rng']))
                    reference=CommittedSampler(len(train_data),config['seed'])
                    if s['sampler']['order']:reference.load_state_dict(s['sampler'])
                    iterator=iter(reference)
                    self.expected_resume_indices=[next(iterator) for _ in range(min(config['batch_size'],len(train_data)))]
                    report['resume']={'path':str(args.resume),'restored_global_step':trainer.global_step,
                        'optimizer_moments_equal':True,'optimizer_state_entries':len(actual),'scheduler_restored':True,
                        'sampler_restored':self.resume_state is not None,'rng_restored':self.resume_state is not None,
                        'restored_cursor':self.resume_state['sampler']['cursor']}
                    report['resume']['all_optimizer_states_checked']=bool(config.get('strict_resume_audit'))
                    report['resume']['restored_epoch']=trainer.current_epoch
                    assert trainer.current_epoch==resumed['epoch']
                if not resumed:
                    if trainer.global_step!=0 or opt.state:raise RuntimeError('Warm start inherited optimization state')
                    report['fresh_optimizer']={'global_step':0,'optimizer_states':len(opt.state),
                        'scheduler_last_epoch':sched.scheduler.last_epoch,'lrs':[g['lr'] for g in opt.param_groups]}
                report['accumulation']={'micro_batch':config['batch_size'],'factor':trainer.accumulate_grad_batches,
                    'effective_batch_size':config['batch_size']*trainer.accumulate_grad_batches,
                    'tail_rule':'Native Lightning flush; all micro-losses including epoch tail divided by accumulation factor',
                    'loss_equivalence':'BN stays micro-batch8; mean of micro-batch object-normalized losses differs from physical batch64'}
                progress.start(trainer.global_step,resumed['b2d_v1_fit'].get('progress') if resumed else None)
                if health:
                    result=health.run(trainer.global_step)
                    if result['stop']:raise HealthStop('Initial health alarm: '+','.join(result['warnings']))
                    self.save_health_checkpoint(trainer,'initial' if not resumed else 'restored')
                progress.update(trainer.global_step,trainer.current_epoch,force=True)
            def on_train_epoch_start(self,trainer,module):
                output_storage_check()
                report.setdefault('epoch_schedule',[]).append({'epoch':trainer.current_epoch,'global_step':trainer.global_step,
                    'scheduler_last_epoch':trainer.lr_scheduler_configs[0].scheduler.last_epoch,
                    'lrs':[g['lr'] for g in trainer.optimizers[0].param_groups]})
            def on_train_epoch_end(self,trainer,module):
                if resumed and trainer.fit_loop.restarting and trainer.global_step==resumed['global_step']:
                    # An epoch-end checkpoint may replay only the parent's end
                    # hooks, with no new batches. Its epoch scheduler state is
                    # already saved; do not let the parent step it a second time.
                    trainer.fit_loop._restarting=False
                    report['resume']['parent_restart_cleared_on_epoch_end_replay']=trainer.current_epoch
                if report.get('pause_requested'):
                    output_storage_check()
                    from tools.training.diffusiondrive_v1.features_cache import directory_sync
                    path=output/f'pause-step{trainer.global_step:09d}.ckpt'
                    temp=path.with_suffix('.ckpt.partial')
                    if path.exists():raise FileExistsError(path)
                    trainer.save_checkpoint(temp)
                    os.replace(temp,path);directory_sync(output)
                    report['pause_checkpoint']=str(path)
            def save_health_checkpoint(self,trainer,label):
                output_storage_check()
                from tools.training.diffusiondrive_v1.features_cache import directory_sync
                path=output/'health'/f'{label}-step{trainer.global_step:09d}.ckpt'
                if path.exists():raise FileExistsError(path)
                path.parent.mkdir(exist_ok=True)
                temp=path.with_suffix('.ckpt.partial');trainer.save_checkpoint(temp)
                os.replace(temp,path);directory_sync(path.parent)
                if health:health.last_checkpoint=str(path)
                report['latest_health_checkpoint']=str(path)
                from tools.training.diffusiondrive_v1.lr_experiment import prune_health_checkpoints
                removed=prune_health_checkpoints(path.parent,config['training'].get('health_checkpoint_keep'))
                if removed:report.setdefault('retired_health_checkpoints',[]).extend(removed)
            def on_train_batch_start(self,trainer,module,batch,batch_idx):
                self.did_update=False
                if self.pending_micro==0:
                    self.observe=trainer.global_step<2 or trainer.global_step%config.get('evidence_every_steps',1)==0 or bf16
                    self.window_started=time.monotonic();self.loss_sums={};self.window_indices=[]
                    self.before=dict(agent._transfuser_model.named_parameters())['_trajectory_head.plan_anchor_encoder.0.weight'].detach().cpu().clone()
            def on_before_optimizer_step(self,trainer,module,optimizer):
                from tools.training.diffusiondrive_v1.lr_experiment import check_gradients
                params=dict(agent._transfuser_model.named_parameters());check_gradients(params.values())
                self.did_update=True;self.gradients={}
                prefixes=('_backbone.image_encoder','_backbone.lidar_encoder','_tf_decoder','_agent_head','_bev_semantic_head','_trajectory_head')
                for prefix in prefixes:
                    gradients=[p.grad for k,p in params.items() if k.startswith(prefix) and p.grad is not None]
                    if gradients:self.gradients[prefix]=float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt())
            def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
                size=len(batch[0]['status_feature']);self.pending_samples+=size;self.pending_micro+=1;self.total_batches+=1
                losses={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if k.startswith('train/') and k.endswith('_step')}
                for key,value in losses.items():self.loss_sums[key]=self.loss_sums.get(key,0.)+value*size
                if args.pause_after_micro_batches and self.total_batches>=args.pause_after_micro_batches:
                    report['pause_request_micro_batch']=args.pause_after_micro_batches
                if not self.did_update:
                    p=dict(agent._transfuser_model.named_parameters())['_trajectory_head.plan_anchor_encoder.0.weight']
                    if not torch.equal(p.detach().cpu(),self.before):raise RuntimeError('Parameters changed without optimizer step')
                    self.micro_unchanged+=1
                    if (output/'pause.request').is_file():report.setdefault('pause_request_micro_batch',self.total_batches)
                    return
                indices=sampler.acknowledge(self.pending_samples)
                if self.expected_resume_indices is not None:
                    if indices[:len(self.expected_resume_indices)]!=self.expected_resume_indices:raise RuntimeError('resume skipped or replayed committed samples')
                    report['resume']['first_batch_tokens']=[train_data.tokens[i] for i in indices[:len(self.expected_resume_indices)]]
                    report['resume']['committed_prefetch_cursor_checked']=True;self.expected_resume_indices=None
                self.total_samples+=len(indices);self.total_updates+=1
                progress.commit(len(indices),self.pending_micro)
                report.update(total_optimized_samples=self.total_samples,total_optimizer_batches=self.total_updates,
                    total_micro_batches=self.total_batches,unchanged_nonupdate_micro_batches=self.micro_unchanged)
                count=self.pending_micro;self.pending_micro=0;self.pending_samples=0
                p=dict(agent._transfuser_model.named_parameters())['_trajectory_head.plan_anchor_encoder.0.weight']
                delta=float((p.detach().cpu()-self.before).abs().max())
                if not delta>0:raise RuntimeError('optimizer did not update planning parameters')
                record={'global_step':trainer.global_step,'epoch':trainer.current_epoch,
                    'sample_tokens':[train_data.tokens[i] for i in indices],'micro_batches':count,'samples':len(indices),
                    'losses':{k:v/len(indices) for k,v in self.loss_sums.items()},'gradient_norms':self.gradients,
                    'planning_max_abs_update':delta,'learning_rates':[g['lr'] for g in trainer.optimizers[0].param_groups],
                    'batch_seconds':time.monotonic()-self.window_started}
                report['steps'].append(record)
                if config.get('step_report_limit') and len(report['steps'])>config['step_report_limit']:report['steps']=report['steps'][-config['step_report_limit']:]
                with (output/'step_evidence.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if resumed and trainer.global_step>resumed['global_step'] and trainer.fit_loop.restarting:
                    trainer.fit_loop._restarting=False;report['resume']['parent_restart_cleared_after_real_step']=trainer.global_step
                if health and (trainer.global_step%health.interval==0 or trainer.global_step>=horizon):
                    result=health.run(trainer.global_step);self.save_health_checkpoint(trainer,'diagnostic')
                    if result['warnings']:print(json.dumps({'health_warning':result['warnings'],'step':trainer.global_step}),flush=True)
                    if result['stop']:report['health_stop']=result['warnings'];report['pause_requested']=True
                if ((output/'pause.request').is_file() or report.get('pause_request_micro_batch')
                    or (args.pause_after_steps and self.total_updates>=args.pause_after_steps) or report.get('health_stop')):
                    report['pause_requested']=True;trainer.limit_val_batches=0;trainer.should_stop=True
                progress.update(trainer.global_step,trainer.current_epoch)
                (output/'fit_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            def on_validation_start(self,trainer,module):
                progress.validation_start(len(val_loader))
                self.val_rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all() if args.device=='cuda' else [])
                self.val_position=0
            def on_validation_batch_start(self,trainer,module,batch,batch_idx,dataloader_idx=0):
                # Restored Lightning loop batch_idx can carry the previous epoch's offset.
                torch.manual_seed(config['evaluation_noise_seed']+self.val_position)
                self.val_position+=1
            def on_validation_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):
                progress.update(trainer.global_step,trainer.current_epoch,phase='validating',validation_batch=self.val_position)
            def on_validation_end(self,trainer,module):
                progress.validation_end(replay=bool(resumed and trainer.global_step<=resumed['global_step']))
                metrics={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if k.startswith('val/') and not k.endswith('_step')}
                record={'global_step':trainer.global_step,'metrics':metrics}
                if resumed and trainer.global_step<=resumed['global_step']:
                    report.setdefault('resume_validation_replay',[]).append(dict(record,used_for_selection=False))
                else:report['validation'].append(record)
                torch.set_rng_state(self.val_rng[0])
                if self.val_rng[1]:torch.cuda.set_rng_state_all(self.val_rng[1])

        validation_steps=config['training'].get('validation_every_steps')
        if validation_steps is not None and (type(validation_steps)is not int or validation_steps<=0):
            raise ValueError('validation_every_steps must be a positive integer')
        selected=PreservedCheckpoint(dirpath=output/'checkpoints',filename='best-epoch{epoch:03d}-step{step:06d}',auto_insert_metric_name=False,
                                    monitor=config['training'].get('checkpoint_monitor','val/loss'),mode='min',save_top_k=1,save_last=True,save_on_train_epoch_end=validation_steps is None)
        periodic=PreservedCheckpoint(dirpath=output/'periodic',filename='step{step:06d}',auto_insert_metric_name=False,
                                     every_n_train_steps=config['training']['checkpoint_every_steps'],save_top_k=1)
        if resumed:
            selected._saved_state_keys=tuple(resumed.get('callbacks',{}))
            periodic._saved_state_keys=tuple(resumed.get('callbacks',{}))
            selected._resume_step=resumed['global_step'];periodic._resume_step=resumed['global_step']
        validation_options={'check_val_every_n_epoch':config['training']['validation_every_epochs']}
        if validation_steps is not None:
            validation_options={'check_val_every_n_epoch':None,'val_check_interval':validation_steps}
        logger=False
        if config.get('csv_logger'):
            from pytorch_lightning.loggers import CSVLogger
            logger=CSVLogger(save_dir=str(output),name='metrics',version=0,flush_logs_every_n_steps=100)
        precision_plugins=None
        if bf16:
            from tools.training.diffusiondrive_v1.bf16_policy import BF16TrainingFP32Evaluation
            precision_plugins=[BF16TrainingFP32Evaluation()]
        trainer=pl.Trainer(accelerator='gpu' if args.device=='cuda' else 'cpu',devices=1,max_steps=horizon,
            accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
            max_epochs=config['training']['max_epochs'],precision=None if bf16 else config['training']['precision'],logger=logger,
            enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
            callbacks=[Evidence(),selected,periodic],**validation_options,
            plugins=precision_plugins,
            default_root_dir=str(output),log_every_n_steps=config.get('log_every_n_steps',1))
        started=time.monotonic();trainer.fit(module,train_dataloaders=train_loader,val_dataloaders=val_loader,ckpt_path=args.resume)
        fit_seconds=time.monotonic()-started
        if report.get('pause_checkpoint'):
            saved=torch.load(report['pause_checkpoint'],map_location='cpu',weights_only=False)
            assert saved['global_step']==trainer.global_step and saved['b2d_v1_fit']['contract_sha256']==contract_hash
            report.update(status='paused',global_step=trainer.global_step,epoch=trainer.current_epoch,
                          last_checkpoint=report['pause_checkpoint'],fit_wall_seconds=fit_seconds)
            progress.update(trainer.global_step,trainer.current_epoch,phase='paused',force=True)
            print(json.dumps({'status':'paused','global_step':trainer.global_step,'checkpoint':report['pause_checkpoint']}),flush=True)
            return
        if trainer.global_step!=horizon:raise RuntimeError('training stopped before requested horizon')
        # Native epoch-end save preserves Lightning's loop phase for exact continuation.
        # Overwriting it after fit/teardown can cause an already completed epoch to replay.
        final=Path(selected.last_model_path or periodic.best_model_path)
        if not final.is_file():raise RuntimeError('no native resume checkpoint was saved')
        report['after']={'train_reference':evaluate(DataLoader(train_data,batch_size=config['batch_size'],shuffle=False)),
                         'held_out_validation':evaluate(val_loader)}
        # Strict reload and target-free inference use the actual saved checkpoint.
        checkpoint=torch.load(final,map_location='cpu',weights_only=False)
        if checkpoint['global_step']!=trainer.global_step:
            # A periodic checkpoint may be newer when max_steps ends between validation epochs.
            candidate=Path(periodic.best_model_path)
            checkpoint=torch.load(candidate,map_location='cpu',weights_only=False)
            if checkpoint['global_step']!=trainer.global_step:raise RuntimeError('latest checkpoint does not cover all optimizer steps')
            final=candidate
        module.load_state_dict(checkpoint['state_dict'],strict=True)
        del checkpoint
        module.to(torch.device(args.device)).eval()
        with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device=='cuda' else []):
            torch.manual_seed(config['evaluation_noise_seed'])
            features,_=val_data[0]
            prediction=agent.forward({key:value.unsqueeze(0).to(args.device) for key,value in features.items()})
            trajectory=prediction['trajectory']
            if trajectory.shape!=(1,8,3) or not torch.isfinite(trajectory).all():raise RuntimeError('invalid checkpoint trajectory inference')
            report['checkpoint_inference']={'strict_reload':True,'targets_supplied':False,'trajectory_shape':list(trajectory.shape),'finite':True}
        batch_seconds=sum(step['batch_seconds'] for step in report['steps'])
        report['throughput']={'optimizer_batches_per_second':len(report['steps'])/batch_seconds,
                              'samples_per_second':sum(len(step['sample_tokens']) for step in report['steps'])/batch_seconds,
                              'scope':'forward/backward/optimizer; excludes decoding wait and validation',
                              'num_workers':workers,
                              'fit_wall_seconds':fit_seconds,
                              'fit_wall_samples_per_second':report['total_optimized_samples']/fit_seconds,
                              'fit_wall_scope':'Trainer.fit including loader wait, epoch validation and checkpoints; excludes fixed before/after panels'}
        report.update(status='pass',global_step=trainer.global_step,epoch=trainer.current_epoch,
            last_checkpoint=str(final),best_checkpoint=selected.best_model_path,
            best_validation_loss=float(selected.best_model_score) if selected.best_model_score is not None else None,
            training_and_evaluation_seconds=time.monotonic()-started,bev_cache_stats=provider.cache_stats if provider else None,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
        progress.update(trainer.global_step,trainer.current_epoch,phase='complete',force=True)
    except (HealthStop,FloatingPointError) as exc:
        resume_from=(report.get('latest_health_checkpoint') or report.get('pause_checkpoint')
                     or (str(args.resume) if args.resume else None))
        report.update(status='safely_stopped_before_invalid_update',error_type=type(exc).__name__,error=str(exc),
                      resume_from=resume_from,current_state_not_saved=True)
        print(json.dumps({'status':report['status'],'error':str(exc),'resume_from':resume_from}),flush=True)
        return
    except Exception as exc:
        resume_from=(report.get('latest_health_checkpoint') or report.get('pause_checkpoint')
                     or (str(args.resume) if args.resume else None))
        report.update(status='failed_recoverable' if resume_from else 'failed',
                      error_type=type(exc).__name__,error=str(exc),
                      resume_from=resume_from,current_state_not_saved=True)
        print(json.dumps({'status':report['status'],'error':str(exc),'resume_from':resume_from}),flush=True)
        raise
    finally:
        (output/'fit_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'global_step':report['global_step'],'before':report['before'],'after':report['after'],
                      'last_checkpoint':report['last_checkpoint'],'best_checkpoint':report['best_checkpoint']},ensure_ascii=False))


if __name__=='__main__':main()
