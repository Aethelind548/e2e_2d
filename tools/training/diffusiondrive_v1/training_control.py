"""Resume/pause/status/logs for the frozen Baseline experiment."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from tools.training.diffusiondrive_v1.features_cache import active_mount, durable_json, file_digest

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight'
STATE = EVIDENCE / 'training_control_state.json'


def live(pid):
    try:
        args = Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
        return b'tools.training.diffusiondrive_v1.fit' in args
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def mount_check(config, output):
    spec = dict(config['output_storage'], path=str(output), mountpoint='/run/media/aethelind/PSSD')
    info = active_mount(spec, writing=True)
    if info['free_bytes'] < spec['minimum_free_gib'] * 1024**3:
        raise RuntimeError('Output disk is below the free-space reserve')


def refresh():
    state = json.loads(STATE.read_text())
    output = Path(state['output'])
    config = json.loads(Path(state['config']).read_text())
    # Never read/write an empty replacement mountpoint after disconnect.
    active_mount(dict(config['output_storage'], path=str(output), mountpoint='/run/media/aethelind/PSSD'))
    result = output / 'fit_result.json'
    if not live(state['pid']) and result.exists():
        report = json.loads(result.read_text())
        if report['status'] == 'paused' and state['state'] != 'paused':
            checkpoint = Path(report['last_checkpoint'])
            state.update(state='paused', checkpoint=str(checkpoint), checkpoint_step=report['global_step'],
                         checkpoint_sha256=file_digest(checkpoint), stopped_at=time.time())
            durable_json(STATE, state)
        elif state['state'] not in ('paused', 'pausing_wait_checkpoint', 'stopping'):
            state.update(state='complete' if report['status']=='pass' else 'stopped_unexpectedly')
            durable_json(STATE, state)
    progress_path = output / 'training_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    return state, config, progress


def duration(seconds):
    if seconds is None:
        return '尚无估计'
    minutes = int(max(0, seconds) / 60)
    return f'{minutes//1440}天{minutes%1440//60}小时{minutes%60}分'


def status():
    state, config, progress = refresh()
    if not progress:
        launch_path=EVIDENCE/'formal_training_launch.json'
        launch=json.loads(launch_path.read_text()) if launch_path.exists() else state
        started=launch.get('started_at',time.time())
        if isinstance(started,str):started=datetime.datetime.fromisoformat(started).timestamp()
        step=state.get('checkpoint_step',state.get('resume_step',0))
        elapsed=state.get('signal_sent_at',time.time())-started
        # A resumed process still initializing has made no new measured steps.
        rate=step/elapsed if step and elapsed>0 and not live(state['pid']) and launch_path.exists() else None
        progress = dict(global_step=step, max_steps=config['training']['max_steps'], steps_per_second=rate,
                        train_only_remaining_seconds=(config['training']['max_steps']-step)/rate if rate else None,
                        estimated_remaining_seconds=None)
    result = dict(state=state['state'], process_running=live(state['pid']), pid=state['pid'],
                  output=state['output'], checkpoint=state.get('checkpoint'), progress=progress)
    if result['process_running'] and progress.get('estimated_remaining_seconds') is not None:
        result['estimated_finish_local']=(datetime.datetime.now(ZoneInfo('Asia/Shanghai'))+
            datetime.timedelta(seconds=progress['estimated_remaining_seconds'])).isoformat()
    durable_json(EVIDENCE/'training_live_status.json', result)
    rate = progress.get('steps_per_second')
    speed=f'{rate:.2f} step/s' if rate else '采样中'
    print(f"状态={result['state']} 阶段={progress.get('state',result['state'])} "
          f"进程运行={result['process_running']} PID={result['pid']} "
          f"step={progress.get('global_step')}/{progress.get('max_steps')} 速度={speed}",flush=True)
    print('剩余训练时间估计：'+duration(progress.get('train_only_remaining_seconds'))+
          '；含验证：'+duration(progress.get('estimated_remaining_seconds')), flush=True)
    if result.get('estimated_finish_local'):
        print('预计完成时间（北京时间）：'+result['estimated_finish_local'],flush=True)
    print('估计为按近期速度需要的运行时长，不包含暂停。未实测验证时，训练估计不含验证开销。', flush=True)


def resume(dry_run=False,num_workers=None,prefetch_factor=None,mount_check_mode=None,cache_read_mode=None,allow_fp32_to_bf16=False,cache_file_read_mode=None):
    state, config, _ = refresh()
    if state['state'] != 'paused' or live(state['pid']):
        raise RuntimeError('Resume requires a fully paused process and verified checkpoint')
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit() and live(entry.name):
            raise RuntimeError('Another fit process is already running: '+entry.name)
    frozen = json.loads((EVIDENCE/'baseline_frozen_identity.json').read_text())
    for name in ('config', 'manifest', 'pair_index'):
        if file_digest(frozen[name]) != frozen[name+'_sha256']:
            raise RuntimeError('Frozen input changed: '+name)
    from tools.training.diffusiondrive_v1.official_v1_audit import verify_pins
    from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
    verify_pins(config)
    manifest = json.loads(Path(config['paired_manifest']).read_text())
    for source in manifest['feature_sources'].values():
        for spec in effective_roots(manifest, source).values():
            active_mount(spec)
    from tools.training.diffusiondrive_v1.target_cache import storage_check
    for source in manifest['target_sources'].values():
        storage_check(source['target_root'])
    checkpoint = Path(state['checkpoint'])
    if not checkpoint.is_file() or file_digest(checkpoint) != state['checkpoint_sha256']:
        raise RuntimeError('Paused checkpoint missing or SHA changed')
    stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    output = Path(config['output_dir']).with_name(Path(config['output_dir']).name+'_resume_'+stamp)
    mount_check(config, output)
    command = [sys.executable, '-u', '-m', 'tools.training.diffusiondrive_v1.fit', '--config', state['config'],
               '--device', 'cuda', '--resume', str(checkpoint), '--output-dir', str(output)]
    if allow_fp32_to_bf16:command.append('--allow-fp32-to-bf16')
    overrides=dict(state.get('execution_overrides',{}))
    for key,value in (('num_workers',num_workers),('prefetch_factor',prefetch_factor)):
        if value is not None:
            if value<0 or (key=='prefetch_factor' and value==0):raise ValueError('Invalid '+key)
            overrides[key]=value
    if mount_check_mode is not None:overrides['mount_check_mode']=mount_check_mode
    if cache_read_mode is not None:overrides['cache_read_mode']=cache_read_mode
    if cache_file_read_mode is not None:overrides['cache_file_read_mode']=cache_file_read_mode
    for key,value in overrides.items():command.extend(['--'+key.replace('_','-'),str(value)])
    if dry_run:
        result=dict(status='passed',process_started=False,command=command,checkpoint_sha256=state['checkpoint_sha256'])
        durable_json(EVIDENCE/'resume_precheck.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    output.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', MPLCONFIGDIR='/tmp/e2e_v1_mpl')
    with (output/'training.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(state='running', pid=process.pid, output=str(output), config=state['config'],
                  execution_overrides=overrides,
                  resume_from=str(checkpoint), resume_step=state['checkpoint_step'],
                  command=command, started_at=time.time())
    durable_json(STATE, record)
    durable_json(output/'launch.json', record)
    monitor = [sys.executable, '-u', '-m', 'tools.training.diffusiondrive_v1.training_monitor',
               '--pid', str(process.pid), '--config', state['config'], '--output-dir', str(output)]
    with (output/'resource_monitor.log').open('x') as log:
        watcher = subprocess.Popen(monitor, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record['monitor_pid'] = watcher.pid
    durable_json(STATE, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main():
    global STATE,EVIDENCE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['resume', 'pause', 'status', 'logs'])
    p.add_argument('--watch', type=int, default=0, help='Status refresh seconds, at least 5')
    p.add_argument('--lines', type=int, default=30)
    p.add_argument('--dry-run',action='store_true',help='Validate resume inputs without starting training')
    p.add_argument('--num-workers',type=int,help='Resume with a different number of deterministic cache workers')
    p.add_argument('--prefetch-factor',type=int,help='Resume with a different per-worker prefetch depth')
    p.add_argument('--mount-check-mode',choices=['findmnt','procfs'],help='Cache-reader mount inspection backend')
    p.add_argument('--cache-read-mode',choices=['audited','cache-only'],help='Accepted cache-only tensor reader')
    p.add_argument('--cache-file-read-mode',choices=['legacy','verified-single-pass'],help='Deserialize the exact SHA-verified bytes in memory')
    p.add_argument('--state-file',type=Path,help='Explicit experiment control file; default remains the historical Baseline profile')
    p.add_argument('--allow-fp32-to-bf16',action='store_true',help='First launch of a separately frozen BF16 profile from an FP32 checkpoint only')
    args = p.parse_args()
    if args.state_file is not None:
        STATE=args.state_file.resolve();EVIDENCE=STATE.parent
    if args.action == 'status':
        while True:
            status()
            if not args.watch: return
            time.sleep(max(5,args.watch))
    elif args.action == 'logs':
        state, _, _ = refresh()
        with (Path(state['output'])/'training.log').open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-32768))
            print('\n'.join(f.read().decode(errors='replace').splitlines()[-args.lines:]))
    else:
        with STATE.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.action == 'resume':
                resume(args.dry_run,args.num_workers,args.prefetch_factor,args.mount_check_mode,args.cache_read_mode,args.allow_fp32_to_bf16,args.cache_file_read_mode)
            else:
                state, config, progress = refresh()
                if not live(state['pid']):
                    print('训练进程已停止。'); return
                if not progress.get('cooperative_pause_supported'):
                    raise RuntimeError('This legacy process has no cooperative pause hook; use the pending checkpoint-safe stop.')
                mount_check(config, Path(state['output']))
                (Path(state['output'])/'pause.request').touch(exist_ok=True)
                state['state']='pausing';durable_json(STATE,state)
                print('已请求暂停；当前批次完成后保存完整 Checkpoint 并退出。')


if __name__ == '__main__':
    main()
