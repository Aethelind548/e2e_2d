"""Summarize recorded pilot evidence; no training or raw-data reads."""
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/training_preparation'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    pilot = read(ROOT/'outputs/diffusiondrive_v1/pilot_original_fit/fit_result.json')
    resume = read(ROOT/'outputs/diffusiondrive_v1/pilot_original_resume/fit_result.json')
    assert pilot['status'] == resume['status'] == 'pass'
    assert pilot['global_step'] == 256 and resume['global_step'] == 320
    assert all(resume['resume'][key] for key in ('optimizer_moments_equal','scheduler_restored',
                                              'sampler_restored','rng_restored','committed_prefetch_cursor_checked'))
    steps = pilot['steps'] + resume['steps']
    assert [s['global_step'] for s in steps] == list(range(1,321))
    assert all(s['planning_max_abs_update']>0 for s in steps)
    assert all(np.isfinite(value) for s in steps for value in s['losses'].values())
    assert all(np.isfinite(value) for s in steps for value in s['gradient_norms'].values())
    panels = pilot['validation'] + resume['validation']
    preparation = read(ROOT/'outputs/diffusiondrive_v1/pilot_original_prepared/preparation_result.json')
    benchmark = read(OUT/'loader_benchmark.json')
    config = read(ROOT/'tools/training/diffusiondrive_v1/pilot_original.json')
    before = pilot['before']['held_out_validation']
    after = resume['after']['held_out_validation']
    blocks = [steps[:32], steps[-32:]]
    loss_means = [{key:statistics.mean(s['losses'][key] for s in block) for key in block[0]['losses']} for block in blocks]
    journal = [json.loads(line) for line in (ROOT/'outputs/diffusiondrive_v1/pilot_original_prepared/attempts.jsonl').read_text().splitlines()]
    coverage = {}
    for split in ('train','validation'):
        rows = [r for r in journal if r['split']==split and r['status']=='accepted']
        coverage[split] = {'class_present_frames': [sum(r['class_pixels'][i]>0 for r in rows) for i in range(7)],
                           'class_pixels': [sum(r['class_pixels'][i] for r in rows) for i in range(7)]}
    # Verify all original repository files and preserved V1 historical files.
    snapshot = read(OUT/'protected_before.json')
    planned = read(OUT/'deletion_plan.json')['files']
    bytecode = {r['path'] for r in planned if r['category']=='generated_bytecode'}
    intentional_changes = {'tools/training/diffusiondrive_v1/fit.py', 'tools/b2d_adapter/README.md'}
    mismatches = [rel for rel,h in snapshot.items() if rel not in bytecode|intentional_changes
                  and (not (ROOT/rel).is_file() or sha(ROOT/rel)!=h)]
    assert not mismatches, mismatches
    checkpoints = {}
    for run, report in (('pilot',pilot),('resume',resume)):
        checkpoints[run] = {key:{'path':report[key], 'sha256':sha(Path(report[key]))}
                            for key in ('best_checkpoint','last_checkpoint')}
    summary = {'status':'pass', 'optimizer_steps':320, 'pilot_steps':256, 'resume_steps':64,
               'initialization':'official ImageNet+anchor, no historical planning checkpoint',
               'data':{'accepted':preparation['accepted'],'rejected':preparation['failed_count'],
                       'manifest':config['prepared_manifest'], 'town_and_route_split_disjoint':True,
                       'coverage':coverage},
               'before_held_out':before, 'after_256_held_out':pilot['after']['held_out_validation'],
               'after_320_held_out':after, 'validation_panels':panels,
               'train_loss_first_last_32_step_means':loss_means,
               'train_reference_before':pilot['before']['train_reference'],
               'train_reference_after_256':pilot['after']['train_reference'],
               'train_reference_after_320':resume['after']['train_reference'],
               'throughput':{'pilot':pilot['throughput'],'resume':resume['throughput']},
               'peak_cuda_bytes':max(pilot['peak_cuda_bytes'],resume['peak_cuda_bytes']),
               'runtime':pilot['runtime'], 'resume':resume['resume'], 'checkpoints':checkpoints,
               'original_repositories_and_historical_v1_sha_mismatches':mismatches,
               'intentional_current_file_changes':sorted(intentional_changes),
               'training_loss_reporting':'official returned weighted components; training multi-stage trajectory loss differs from target-free evaluation L1',
               'conclusion':'engineering pilot passed; limited grouped validation improved; convergence and closed-loop performance remain unverified'}
    summary['official_loss_weights']={'trajectory_weight':12.0,'diff_loss_weight':20.0,
                                     'agent_class_weight':10.0,'agent_box_weight':1.0,'bev_semantic_weight':14.0}
    summary['bev_cross_entropy_before_after']=[before['weighted_official_losses']['bev_semantic_loss']/14,
                                              after['weighted_official_losses']['bev_semantic_loss']/14]
    (OUT/'pilot_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    proposal = {
        'schema':'v1-formal-training-proposal-v1','status':'pilot_validated_recipe_data_scale_not_final',
        'execute_training':False,'validated_configuration':'tools/training/diffusiondrive_v1/pilot_original.json',
        'model_repository':'repositories/DiffusionDrive','label_rules':'v1_original_7class',
        'class_ids':['Background','Road','Walkway','Centerline','Static','Vehicle','Pedestrian'],
        'initialization':pilot['initialization'],
        'optimizer':{'type':'official AdamW','learning_rate':config['learning_rate'],'weight_decay':'official config'},
        'scheduler':{'type':'official WarmupCosLR','warmup_epochs':3,'epochs':100,'min_lr':1e-6},
        'batch_size':2,'precision':'32-true','num_workers':2,'prefetch_factor':2,
        'seed':config['seed'],'evaluation_noise_seed':config['evaluation_noise_seed'],
        'validation':{'mode':'fixed noise target-free trajectory ADE/FDE plus official losses','best_monitor':'val/loss',
                      'pilot_interval_steps':64,'formal_interval':'once per epoch, with periodic checkpoint every 500 steps'},
        'data_expansion':{'next_target_accepted_windows':[2000,5000],
                          'source':'explicit current Route Manifest; freeze Town/Route grouping before preparing',
                          'keep_validation_towns':['Town07','Town13','Town15'],
                          'correlation':'overlapping windows are not independent samples; never split frames of one Route across train/validation',
                          'reject_policy':'record errors; no background fallback or training-time skipping'},
        'storage':{'pilot_mode':'complete prepared tensors + independent original BEV cache',
                   'measured_tensor_bytes_per_sample':benchmark['per_sample_tensor_bytes'],
                   'estimated_2000_to_5000_frame_tensor_gib':[benchmark['per_sample_tensor_bytes']*n/1024**3 for n in (2000,5000)],
                   'full_dataset_mode':'undecided; raw worker=0 is measured slow, full tensor duplication requires capacity assessment',
                   'default_full_dataset_precompute':False},
        'pilot_weights':'diagnostic baseline checkpoints, not converged formal weights',
        'resume_policy':'exact dataset/map/producer/initialization/optimizer contract required; changing dataset means a new run, not --resume',
        'remaining_issues':['Town02 road16 lane-3 sidewalk self-intersection rejected in four newly selected windows.',
                            'Held-out pilot BEV labels contain zero pedestrian-positive frames; add a verified pedestrian Route in held-out Towns before formal semantic evaluation.',
                            '320 steps cover fewer than two training epochs; the original three-epoch LR warmup is not completed.',
                            'Training trajectory loss trend needs larger diverse windows and more epochs; no convergence claim.',
                            'Host CUDA works despite NVML mismatch; sandbox has no CUDA/IPC. GPU runtime visibility must be checked per launch.',
                            'CARLA Agent/closed-loop metrics and full-dataset tensor service are not implemented in this task.']}
    (ROOT/'tools/training/diffusiondrive_v1/formal_training_proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n')

    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    fig, axes=plt.subplots(2,2,figsize=(15,9),constrained_layout=True)
    x=np.array([r['global_step'] for r in steps])
    def smooth(key):
        y=np.array([s['losses'][key] for s in steps])
        return np.array([np.mean(y[max(0,i-15):i+1]) for i in range(len(y))])
    axes[0,0].plot(x,[s['losses']['train/loss_step'] for s in steps],alpha=.2,color='#356fc0')
    axes[0,0].plot(x,smooth('train/loss_step'),color='#356fc0',label='16 步移动平均')
    axes[0,0].set_title('原版训练总损失（训练分支）');axes[0,0].legend()
    for key,label in (('trajectory_loss','轨迹'),('bev_semantic_loss','BEV CE'),
                      ('agent_class_loss','检测分类'),('agent_box_loss','检测框')):
        axes[0,1].plot(x,smooth('train/'+key+'_step'),label=label)
    axes[0,1].set_title('原版加权损失分量 · 16 步移动平均');axes[0,1].legend()
    vx=[0]+[r['global_step'] for r in panels]
    axes[1,0].plot(vx,[before['official_eval_loss']]+[r['metrics']['val/loss'] for r in panels],marker='o')
    axes[1,0].set_title('独立验证总损失（推理分支）')
    axes[1,1].plot(vx,[before['ade_m']]+[r['metrics']['val/ade_m'] for r in panels],marker='o',label='ADE / m')
    axes[1,1].plot(vx,[before['fde_m']]+[r['metrics']['val/fde_m'] for r in panels],marker='o',label='FDE / m')
    axes[1,1].set_title('独立验证轨迹误差 · 100 帧固定噪声面板');axes[1,1].legend()
    for ax in axes.flat:
        ax.axvline(256,color='#8f669f',ls='--',lw=1,label='恢复点')
        ax.set_xlabel('累计 optimizer step');ax.grid(alpha=.2)
    fig.suptitle('原版 V1 Pilot：380 训练窗口 / 100 验证窗口 · Town 与 Route 隔离\n256 步 + 64 步恢复；训练与推理损失分支不能直接等同',fontsize=16)
    fig.savefig(OUT/'pilot_trends.png',dpi=180)
    plt.close(fig)
    print(json.dumps({'status':'pass','before_ade':before['ade_m'],'after_ade':after['ade_m'],
                      'before_fde':before['fde_m'],'after_fde':after['fde_m'],
                      'train_loss_means':[r['train/loss_step'] for r in loss_means],
                      'class_presence':{k:v['class_present_frames'] for k,v in coverage.items()},
                      'protected_mismatches':mismatches},ensure_ascii=False))


if __name__=='__main__':
    main()
