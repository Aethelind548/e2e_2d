"""Small real-cache FP32/BF16 comparison from the same full training checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))


def main():
    import torch
    from torch.utils.data import default_collate
    from tools.training.diffusiondrive_v1.runtime import fit_configuration, model_configuration
    from tools.training.diffusiondrive_v1.paired_dataset import paired_datasets
    from tools.b2d_adapter.v1_builders import official_v1_module
    from navsim.planning.training.agent_lightning_module import AgentLightningModule
    from tools.training.diffusiondrive_v1.bf16_policy import (
        POLICY, install_fp32_losses, fp32_call, validate_resume, semantic_digest,
        BF16TrainingFP32Evaluation,source_sha256)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result = {'status': 'running', 'checkpoint': str(args.checkpoint)}
    try:
        assert torch.cuda.is_bf16_supported()
        torch.set_num_threads(4)
        cfg, _ = fit_configuration(args.config)
        cfg.update(cache_read_mode='cache-only', mount_check_mode='procfs')
        model_cfg = model_configuration(cfg)
        ds, identity = paired_datasets(cfg['paired_manifest'], cfg, model_cfg)
        # Real stratified batches, held on GPU to isolate compute throughput.
        indices = []; towns = set()
        for i, row in enumerate(ds['train'].rows):
            town = row['route'].split('_Town')[1].split('_')[0]
            if town not in towns:indices.append(i);towns.add(town)
        while len(indices) < 16:
            i = (len(indices) * 7919) % len(ds['train'])
            if i not in indices:indices.append(i)
        indices = indices[:16]
        batches = []
        for start in (0, 8):
            f, t = default_collate([ds['train'][i] for i in indices[start:start+8]])
            batches.append(({k:v.cuda() for k,v in f.items()}, {k:v.cuda() for k,v in t.items()}))
        result.update(precision_policy=POLICY,dataset_identity=identity, sample_ids=[ds['train'].rows[i]['sample_id'] for i in indices],
                      gpu=torch.cuda.get_device_name(), batch_size=8)
        parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        old = parent['b2d_v1_fit']['semantic']
        new = dict(old, precision='bf16-mixed', precision_policy=POLICY,precision_policy_source_sha256=source_sha256())
        assert validate_resume(parent, new, True)
        assert validate_resume(parent, old) is None
        rejected = []
        for name, sem, allow in [('implicit_precision', new, False),
            ('batch_change', dict(new, batch_size=16), True),
            ('data_change', dict(new, prepared_data={}), True),
            ('reverse_transition', dict(old, precision='16-mixed'), True)]:
            try:validate_resume(parent, sem, allow)
            except ValueError:rejected.append(name)
            else:raise AssertionError(name)
        child = copy.copy(parent)
        child['b2d_v1_fit'] = dict(parent['b2d_v1_fit'], semantic=new, contract_sha256=semantic_digest(new))
        assert validate_resume(child, new) is None
        # Casts happen before reduction, keep int/bool labels and autograd alive.
        x = torch.ones(4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        def loss_probe(x, labels):
            assert x.dtype == torch.float32 and labels.dtype == torch.int64
            assert not torch.is_autocast_enabled('cuda')
            return x.square().sum()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fp32_call(loss_probe)(x, torch.ones(4, device='cuda', dtype=torch.int64)).backward()
        assert torch.equal(x.grad, torch.full_like(x, 2))
        plugin = BF16TrainingFP32Evaluation()
        assert plugin.scaler is None
        with plugin.train_step_context():assert torch.is_autocast_enabled('cuda')
        with plugin.val_step_context():assert not torch.is_autocast_enabled('cuda')
        agent = official_v1_module('transfuser_agent').TransfuserAgent(model_cfg, cfg['learning_rate'], checkpoint_path=None)
        module = AgentLightningModule(agent).cuda()
        snapshots = {}; records = {}
        for mode in ['fp32', 'bf16']:
            module.load_state_dict(parent['state_dict'], strict=True)
            opt = module.configure_optimizers()['optimizer']
            opt.load_state_dict(parent['optimizer_states'][0])
            if mode == 'bf16':install_fp32_losses(agent)
            module.eval();torch.manual_seed(137)
            with torch.no_grad(),plugin.val_step_context():
                initial_eval=agent.forward(*batches[0])['trajectory'].detach().cpu().clone()
            torch.manual_seed(20261020);torch.cuda.manual_seed_all(20261020)
            torch.cuda.reset_peak_memory_stats();module.train()
            losses = []; seconds = []; initial_grads = {}
            for step in range(20):
                f,t = batches[step % 2];opt.zero_grad(set_to_none=True)
                modal=[]
                def capture_modes(computer, inputs):
                    reg,cls,target,anchor=inputs
                    gt_mode=torch.linalg.norm(target['trajectory'][:,None,:,:2]-anchor,dim=-1).mean(-1).argmin(-1)
                    modal.append((reg.detach().float().cpu(),cls.detach().float().cpu(),gt_mode.cpu()))
                hook=agent._transfuser_model._trajectory_head.loss_computer.register_forward_pre_hook(capture_modes) if step==0 else None
                torch.cuda.synchronize();start=time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mode=='bf16'):
                    prediction=agent.forward(f,t);ld=agent.compute_loss(f,t,prediction)
                assert all(torch.isfinite(v).all() for v in ld.values() if isinstance(v,torch.Tensor))
                ld['loss'].backward()
                if step == 0:
                    initial_grads={k:v.grad.detach().cpu().clone() for k,v in module.named_parameters() if v.grad is not None}
                    first_loss={k:float(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in ld.items() if v is not None}
                    first_traj=prediction['trajectory'].detach().float().cpu()
                    first_modal=modal
                    hook.remove()
                opt.step();torch.cuda.synchronize();seconds.append(time.monotonic()-start)
                assert all(p.dtype==torch.float32 and (p.grad is None or (p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all())) for p in module.parameters())
                losses.append(float(ld['loss'].detach()))
            assert all(v.dtype==torch.float32 for s in opt.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq'))
            before_eval={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
            module.eval();torch.manual_seed(137)
            with torch.no_grad(),plugin.val_step_context():
                ev=agent.forward(*batches[0]);el=agent.compute_loss(*batches[0],ev)
            assert torch.isfinite(el['loss']) and ev['trajectory'].dtype==torch.float32
            assert all(torch.equal(v.cpu(),before_eval[k]) for k,v in module.state_dict().items())
            records[mode]={'first_losses':first_loss,'losses':losses,'steps':20,
                'median_compute_seconds':statistics.median(seconds[4:]),
                'compute_samples_per_second':8/statistics.median(seconds[4:]),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'fp32_validation_loss':float(el['loss']), 'parameter_and_adam_moments_fp32':True}
            snapshots[mode]=(first_traj,initial_grads,first_modal,initial_eval)
            del before_eval,opt
            print(json.dumps({mode:records[mode]}),flush=True)
        a,b=snapshots['fp32'][1],snapshots['bf16'][1]
        dot=sum((a[k].double()*b[k].double()).sum().item() for k in a)
        aa=sum(v.double().square().sum().item() for v in a.values());bb=sum(v.double().square().sum().item() for v in b.values())
        cosine=dot/(aa*bb)**0.5;norm_relative=abs((bb/aa)**0.5-1)
        loss_relative={k:abs(records['bf16']['first_losses'][k]-v)/max(abs(v),1e-3) for k,v in records['fp32']['first_losses'].items()}
        traj_error=float(torch.linalg.norm(snapshots['bf16'][0][...,:2]-snapshots['fp32'][0][...,:2],dim=-1).max())
        modal_errors=[];supervised_modes_equal=True
        for (ar,ac,ag),(br,bc,bg) in zip(snapshots['fp32'][2],snapshots['bf16'][2]):
            modal_errors.append(float(torch.linalg.norm(ar[...,:2]-br[...,:2],dim=-1).max()))
            supervised_modes_equal &= torch.equal(ag,bg)
        _,ac,_=snapshots['fp32'][2][-1];_,bc,_=snapshots['bf16'][2][-1]
        aw=ac.argmax(-1);bw=bc.argmax(-1);flips=aw!=bw
        score_error=(ac-bc).abs().max(-1).values
        margins=ac.gather(1,aw[:,None]).squeeze(1)-ac.gather(1,bw[:,None]).squeeze(1)
        flips_within_rounding=bool((margins[flips]<=2*score_error[flips]+1e-7).all())
        evaluation_identical=torch.equal(snapshots['fp32'][3],snapshots['bf16'][3])
        result.update(records=records, gradient_cosine=cosine, gradient_norm_relative_difference=norm_relative,
            first_loss_relative_differences=loss_relative, max_trajectory_xy_difference_m=traj_error,
            mode_aligned_max_xy_differences_m=modal_errors, supervised_anchor_modes_equal=supervised_modes_equal,
            training_top1_mode_flips=int(flips.sum()), fp32_top1_modes=aw.tolist(),bf16_top1_modes=bw.tolist(),
            logit_max_differences=score_error.tolist(), flipped_mode_fp32_score_margins=margins[flips].tolist(),
            flips_within_logit_difference_bound=flips_within_rounding,
            initial_fp32_inference_bitwise_equal=evaluation_identical,
            rejected_contract_changes=rejected, fp32_validation_context_verified=True,
            compute_speed_ratio=records['fp32']['median_compute_seconds']/records['bf16']['median_compute_seconds'],
            limits='16 real training frames, 20 updates per branch, identical initial checkpoint and RNG; GPU-resident compute timing excludes I/O; not convergence or physical-truth certification.')
        gates={'gradient_cosine_ge_0_98':cosine>=0.98,'gradient_norm_relative_le_0_15':norm_relative<=0.15,
            'loss_relative_le_0_05':max(loss_relative.values())<=0.05,
            'mode_aligned_xy_le_0_15m':max(modal_errors)<=0.15,
            'supervised_modes_unchanged':supervised_modes_equal,'initial_fp32_eval_identical':evaluation_identical,
            'training_top1_flip_explained':traj_error<=0.15 or (int(flips.sum())>0 and flips_within_rounding)}
        result['acceptance_gates']=gates
        assert all(gates.values()), gates
        result['status']='passed'
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
