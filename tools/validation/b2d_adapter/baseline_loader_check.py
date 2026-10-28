"""Bounded real Baseline reads with spawn workers, including relocated Features."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def main():
    import torch
    from torch.utils.data import DataLoader,Subset
    from tools.training.diffusiondrive_v1.runtime import model_configuration,fit_configuration
    from tools.training.diffusiondrive_v1.paired_dataset import paired_datasets
    from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
    from tools.training.diffusiondrive_v1.recovery_paired import readonly
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',nargs='+',type=int,default=[0,2])
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--prefetch-factor',type=int,default=2)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--warmup-batches',type=int,default=4)
    p.add_argument('--mount-check-mode',choices=['findmnt','procfs'],default='findmnt')
    p.add_argument('--cache-read-mode',choices=['audited','cache-only'],default='audited')
    args=p.parse_args();cfg,_=fit_configuration(args.config);torch.set_num_threads(1)
    cfg['mount_check_mode']=args.mount_check_mode
    cfg['cache_read_mode']=args.cache_read_mode
    if args.output.exists():raise FileExistsError('Use a fresh evidence path')
    if args.samples<2 or args.repeat<1 or min(args.workers)<0 or args.prefetch_factor<1 or args.warmup_batches<0:raise ValueError('Invalid benchmark parameters')
    data,identity=paired_datasets(cfg['paired_manifest'],cfg,model_configuration(cfg))
    generator=torch.Generator().manual_seed(20261019);indices=torch.randperm(len(data['train']),generator=generator)[:args.samples].tolist()
    records=[]
    cases=[(r,w) for r in range(args.repeat) for w in (args.workers if r%2==0 else list(reversed(args.workers)))]
    for repetition,workers in cases:
        options={'num_workers':workers,'batch_size':2,'shuffle':False}
        if workers:options.update(multiprocessing_context='spawn',prefetch_factor=args.prefetch_factor,persistent_workers=False)
        start=time.monotonic();first=None;count=0;digests=[];steady_start=None;steady_count=0
        import hashlib
        for batch_idx,(features,targets) in enumerate(DataLoader(Subset(data['train'],indices),**options)):
            if first is None:first=time.monotonic()-start
            b=len(targets['trajectory'])
            assert features['camera_feature'].shape==(b,3,256,1024)
            assert targets['bev_semantic_map'].shape==(b,128,256)
            count+=len(targets['trajectory']);digests.append(hashlib.sha256(targets['trajectory'].numpy().tobytes()).hexdigest())
            if batch_idx+1==args.warmup_batches:steady_start=time.monotonic();steady_count=count
            last_batch=time.monotonic()
        elapsed=time.monotonic()-start
        record={'workers':workers,'repeat':repetition,'prefetch_factor':args.prefetch_factor,'samples':count,'seconds':elapsed,'first_batch_seconds':first,'samples_per_second':count/elapsed,
                'steady_samples_per_second':(count-steady_count)/(last_batch-steady_start) if steady_start and count>steady_count else None,'trajectory_batch_sha':digests}
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='trajectory_batch_sha'}),flush=True)
    assert all(r['trajectory_batch_sha']==records[0]['trajectory_batch_sha'] for r in records)
    # Show a real system-origin Feature now resolves to PSSD and pairs correctly.
    manifest=json.loads(Path(cfg['paired_manifest']).read_text());source=manifest['feature_sources']['remaining_20261015']
    with readonly(Path(source['state'])/'index.sqlite') as db:
        moved={r[0] for r in db.execute("select sample_id from samples where state='complete' and root_id='cache_system'")}
    i=next(i for i,r in enumerate(data['train'].rows) if r['sample_id'] in moved)
    data['train'][i];row=data['train'].rows[i]
    roots=effective_roots(manifest,source)
    assert roots['cache_system']['uuid']=='F8E4AE34E4ADF552'
    result={'dataset':identity,'cache_read_mode':args.cache_read_mode,'mount_check_mode':args.mount_check_mode,'loader_runs':records,'worker_outputs_identical':True,'relocated_real_pair':dict(row,physical_feature_root=roots['cache_system']),
        'limits':f'{len(indices)} unique randomly chosen real samples per case; filesystem/page caches are uncontrolled; not a GPU/end-to-end or full-scale cold-cache benchmark.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'runs':[{k:v for k,v in r.items() if k!='trajectory_batch_sha'} for r in records],'relocated_sample_id':row['sample_id']}))


if __name__=='__main__':main()
