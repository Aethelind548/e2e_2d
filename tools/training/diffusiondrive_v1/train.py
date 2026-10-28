"""Bounded complete official V1 IL on real B2D; 3..5 steps, then strict reload/infer."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')


def main():
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader,ConcatDataset
    from navsim.planning.training.agent_lightning_module import AgentLightningModule
    from tools.b2d_adapter.v1_builders import official_v1_module,validate_v1_config,v1_native_config
    from tools.training.diffusiondrive_v1.runtime import configuration,model_configuration,datasets,initialization_manifest,native_configuration
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('baseline.json'))
    parser.add_argument('--steps',type=int,default=3,choices=(3,4,5))
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--label-bundle',type=Path,help='Reuse tensors from the real label check; otherwise load the full B2D Dataset')
    parser.add_argument('--verify-checkpoint',action='store_true',help='Finish strict reload/inference on an existing smoke checkpoint without any training steps')
    args=parser.parse_args();config=configuration(args.config)
    output=ROOT/config['training_output_dir'];output.mkdir(parents=True,exist_ok=True)
    result_path=output/'training_result.json'
    report=json.loads(result_path.read_text()) if args.verify_checkpoint else {'status':'running','requested_steps':args.steps,'steps':[],'checkpoint_reload':'not_run','target_free_inference':'not_run'}
    try:
        torch.set_num_threads(config.get('threads',4));pl.seed_everything(config.get('seed',20261013),workers=True)
        if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; run in an environment exposing the real GPU devices')
        model_config=model_configuration(config);report['initialization']=initialization_manifest(config,model_config)
        if args.label_bundle:
            bundle=torch.load(args.label_bundle,map_location='cpu',weights_only=False)
            if bundle['config']!=config: raise ValueError('label bundle config does not match training config')
            # Trust only this project artifact; required real provenance is retained in checkpoint.
            if bundle['model_contract']!=validate_v1_config(native_configuration(config),model_config):
                raise ValueError('label bundle Adapter contract mismatch')
            data=bundle['samples'];provenance=bundle['metadata']
            if not all(s['targets']['complete_training_targets'] and s['targets']['bev']['map_sha256'] for s in provenance):
                raise ValueError('label bundle lacks complete real target provenance')
        else:
            data=ConcatDataset(datasets(config,model_config));provenance={'source':'live full B2D Adapter Dataset','config':config}
        loader=DataLoader(data,batch_size=config['batch_size'],shuffle=False,num_workers=0)
        report['sample_count']=len(data);report['device']=args.device
        agent_class=official_v1_module('transfuser_agent').TransfuserAgent
        if args.verify_checkpoint:
            checkpoint=Path(report['checkpoint'])
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report['checkpoint_sha256']:
                raise RuntimeError('saved smoke checkpoint hash changed')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            stored_initialization=saved['b2d_v1_provenance']['initialization']
            if stored_initialization['image_pretrained']['sha256']!=report['initialization']['image_pretrained']['sha256']:
                raise RuntimeError('checkpoint initialization provenance mismatch')
            report['initialization']=stored_initialization
            reloaded=agent_class(model_config,config['learning_rate'],checkpoint_path=str(checkpoint))
            reloaded.initialize()
            expected_state={k.removeprefix('agent.'):v for k,v in saved['state_dict'].items()}
            if set(expected_state)!=set(reloaded.state_dict()) or not all(torch.equal(v,reloaded.state_dict()[k].cpu()) for k,v in expected_state.items()):
                raise RuntimeError('strict checkpoint state_dict mismatch')
            reloaded.to(args.device).eval()
            features,_=next(iter(loader));features={k:v.to(args.device) for k,v in features.items()}
            with torch.no_grad():
                torch.manual_seed(137);prediction=reloaded.forward(features)['trajectory']
            if prediction.shape!=(config['batch_size'],8,3) or not torch.isfinite(prediction).all():
                raise RuntimeError('invalid target-free trajectory')
            torch.save(prediction.cpu(),output/'target_free_trajectory.pt')
            report.update(status='pass',checkpoint_reload='strict_pass',target_free_inference='finite_8x3_after_strict_reload',trajectory=prediction.cpu().tolist(),verification_only=True)
            report.pop('error',None);report.pop('error_type',None)
            print(json.dumps({'status':'pass','completed_training_steps':len(report['steps']),'additional_training_steps':0,'target_free_inference':report['target_free_inference']}))
            return
        agent=agent_class(model_config,config['learning_rate'],checkpoint_path=None)
        # Check actual ImageNet initialization against the complete official ResNet state.
        reference=torch.load(model_config.bkb_path,map_location='cpu',weights_only=True)
        actual=agent._transfuser_model._backbone.image_encoder.state_dict()
        matches={k:torch.equal(v.cpu(),reference[k]) for k,v in actual.items()}
        if not all(matches.values()): raise RuntimeError('image backbone did not load the configured official pretrained weights')
        report['initialization']['image_pretrained']['matching_encoder_tensors']=len(matches)
        module=AgentLightningModule(agent)

        class Evidence(pl.Callback):
            def __init__(self): self.before={};self.pending=None
            def on_before_optimizer_step(self,trainer,module,optimizer):
                params=dict(module.agent._transfuser_model.named_parameters())
                groups={'camera':'_backbone.image_encoder','lidar':'_backbone.lidar_encoder',
                        'detection':'_agent_head','bev':'_bev_semantic_head','trajectory':'_trajectory_head'}
                grads={};self.before={}
                for label,prefix in groups.items():
                    selected=[(k,p) for k,p in params.items() if k.startswith(prefix) and p.requires_grad and p.grad is not None]
                    if not selected: raise RuntimeError('no gradient for '+label)
                    if not all(torch.isfinite(p.grad).all() for _,p in selected): raise RuntimeError('nonfinite gradient for '+label)
                    norm=float(torch.sqrt(sum(p.grad.float().square().sum() for _,p in selected)))
                    if norm<=0: raise RuntimeError('zero gradient for '+label)
                    grads[label]={'norm':norm,'parameter_count_with_grad':len(selected)}
                    k,p=next((k,p) for k,p in selected if p.grad.abs().max()>0)
                    self.before[label]=(k,p.detach().cpu().clone())
                losses={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if k.startswith('train/') and isinstance(v,torch.Tensor)}
                required=('loss','trajectory_loss','agent_class_loss','agent_box_loss','bev_semantic_loss')
                for name in required:
                    value=losses.get('train/'+name+'_step',losses.get('train/'+name))
                    if value is None or not math.isfinite(value): raise RuntimeError('missing/nonfinite original loss '+name)
                self.pending={'step':int(trainer.global_step)+1,'losses':losses,'gradient_groups':grads,
                              'optimizer_lrs':[g['lr'] for g in optimizer.param_groups]}
            def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
                params=dict(module.agent._transfuser_model.named_parameters());updates={}
                for label,(name,before) in self.before.items():
                    delta=float((params[name].detach().cpu()-before).abs().max())
                    if delta<=0: raise RuntimeError('parameter did not update: '+label)
                    updates[label]={'parameter':name,'max_abs_update':delta}
                self.pending['parameter_updates']=updates;report['steps'].append(self.pending)
                (output/'training_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

        trainer=pl.Trainer(accelerator='gpu' if args.device=='cuda' else 'cpu',devices=1,
            max_steps=args.steps,max_epochs=100,precision='32-true',logger=False,enable_checkpointing=False,
            enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,
            limit_val_batches=0,callbacks=[Evidence()],default_root_dir=str(output),log_every_n_steps=1)
        started=time.monotonic();trainer.fit(module,train_dataloaders=loader)
        report['training_seconds']=time.monotonic()-started
        report['optimizer']=type(trainer.optimizers[0]).__name__
        report['scheduler']=type(trainer.lr_scheduler_configs[0].scheduler).__name__
        if args.device=='cuda': report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        checkpoint=output/'smoke.ckpt';trainer.save_checkpoint(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        saved['b2d_v1_provenance']={'config':config,'initialization':report['initialization'],'data':provenance}
        torch.save(saved,checkpoint)
        if len(report['steps'])!=args.steps: raise RuntimeError('wrong number of completed optimizer steps')
        # Constructor initializes the original model; initialize() strictly loads Lightning keys.
        reloaded=agent_class(model_config,config['learning_rate'],checkpoint_path=str(checkpoint));reloaded.initialize()
        state_equal=all(torch.equal(v.detach().cpu(),reloaded.state_dict()[k].detach().cpu()) for k,v in agent.state_dict().items())
        if not state_equal: raise RuntimeError('checkpoint state_dict mismatch')
        report['checkpoint_reload']='strict_pass';report['checkpoint']=str(checkpoint)
        report['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        features,_=next(iter(loader));features={k:v.to(args.device) for k,v in features.items()}
        agent.to(args.device).eval();reloaded.to(args.device).eval()
        with torch.no_grad():
            torch.manual_seed(137);expected=agent.forward(features)['trajectory']
            torch.manual_seed(137);prediction=reloaded.forward(features)['trajectory']
        if prediction.shape!=(config['batch_size'],8,3) or not torch.isfinite(prediction).all(): raise RuntimeError('invalid target-free trajectory')
        torch.testing.assert_close(prediction,expected,rtol=0,atol=0)
        torch.save(prediction.cpu(),output/'target_free_trajectory.pt')
        report.update(status='pass',target_free_inference='finite_8x3_and_reload_equal',trajectory=prediction.cpu().tolist())
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        (output/'training_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'steps':len(report['steps']),'checkpoint':report['checkpoint'],
                      'target_free_inference':report['target_free_inference']},ensure_ascii=False))


if __name__=='__main__': main()
