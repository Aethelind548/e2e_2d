"""Only a 12-step run and resume to 32 on a 16/8-example real Baseline subset."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]


def execute(command,output):
    import psutil
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',MPLCONFIGDIR='/tmp/e2e_v1_mpl')
    start=time.monotonic();records=[];previous={}
    with output.with_suffix('.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            now=time.monotonic();rss=0;read_bytes=0;write_bytes=0;cpu_seconds=0
            try:procs=[psutil.Process(process.pid)]+psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:procs=[]
            for p in procs:
                try:
                    rss+=p.memory_info().rss;cpu=p.cpu_times();cpu_seconds+=cpu.user+cpu.system
                    io=p.io_counters();read_bytes+=io.read_bytes;write_bytes+=io.write_bytes
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True)
            gpu_values=[float(v.strip()) for v in gpu.stdout.strip().split(',')] if gpu.returncode==0 else []
            records.append({'seconds':now-start,'tree_rss_bytes':rss,'tree_cpu_seconds':cpu_seconds,'tree_read_bytes':read_bytes,'tree_write_bytes':write_bytes,
                'gpu_used_mib':gpu_values[0] if gpu_values else None,'gpu_utilization':gpu_values[1] if gpu_values else None,
                'evidence_system_free_bytes':__import__('shutil').disk_usage(output.parent).free,
                'pssd_free_bytes':__import__('shutil').disk_usage('/run/media/aethelind/PSSD').free,
                'device_io':{k:{'read_bytes':v.read_bytes,'write_bytes':v.write_bytes} for k,v in psutil.disk_io_counters(perdisk=True).items() if k in ('nvme0n1','sda')},
                'swap_used_bytes':psutil.swap_memory().used})
            try:process.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
    result={'command':command,'returncode':process.returncode,'seconds':time.monotonic()-start,'records':records,
        'peak_tree_rss_bytes':max((r['tree_rss_bytes'] for r in records),default=0),
        'peak_gpu_used_mib':max((r['gpu_used_mib'] or 0 for r in records),default=0)}
    output.write_text(json.dumps(result,indent=2)+'\n')
    if process.returncode:raise RuntimeError('Preflight command failed; inspect '+str(output.with_suffix('.log')))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--resume-only',action='store_true',help='Reuse passed loader/first_12 evidence; test resume in a fresh corrected directory')
    args=p.parse_args()
    ev=args.evidence.resolve();cfg=json.loads(args.config.read_text());base=Path(cfg['output_dir']);base.mkdir(parents=True,exist_ok=True)
    py=sys.executable
    if not args.resume_only:
        execute([py,'-u','-m','tools.validation.b2d_adapter.baseline_loader_check','--config',str(ROOT/'tools/training/diffusiondrive_v1/baseline_original_7class_20261019.json'),'--output',str(ev/'loader_validation.json')],ev/'loader_resources.json')
    else:
        assert json.loads((ev/'loader_validation.json').read_text())['worker_outputs_identical']
    first=base/'first_12';second=base/('resume_32_corrected' if args.resume_only else 'resume_32')
    common=[py,'-u','-m','tools.training.diffusiondrive_v1.fit','--config',str(args.config.resolve()),'--device','cuda']
    if not args.resume_only:execute(common+['--max-steps','12','--output-dir',str(first)],ev/'first_12_resources.json')
    before=json.loads((first/'fit_result.json').read_text());checkpoint=Path(before['last_checkpoint']);assert before['status']=='pass' and before['global_step']==12
    execute(common+['--max-steps','32','--resume',str(checkpoint),'--output-dir',str(second)],ev/('resume_32_corrected_resources.json' if args.resume_only else 'resume_32_resources.json'))
    after=json.loads((second/'fit_result.json').read_text());assert after['status']=='pass' and after['global_step']==32
    for record in before['epoch_schedule']+after['epoch_schedule']:
        assert record['scheduler_last_epoch']==record['epoch'], 'Scheduler advanced more than once per completed epoch: '+str(record)
    result={'first':{'global_step':before['global_step'],'epoch':before['epoch'],'checkpoint':str(checkpoint)},
        'resumed':{'global_step':after['global_step'],'epoch':after['epoch'],'checkpoint':after['last_checkpoint'],'restored':after['resume']},
        'epoch_schedule':before['epoch_schedule']+after['epoch_schedule'],'epoch_scheduler_alignment_checked':True,'long_training_started':False,
        'limits':'Bounded 16 train / 8 validation real-example epochs. No complete epoch over all 218622 pairs.'}
    (ev/'preflight_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
