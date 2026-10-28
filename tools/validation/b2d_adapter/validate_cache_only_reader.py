"""Real tensor parity, raw-data isolation and spawn-worker checks for cache-only."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def fingerprint(pair):
    digest=hashlib.sha256()
    for group in pair:
        for name,tensor in sorted(group.items()):
            digest.update(name.encode());digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def no_raw_reads(root):
    import builtins
    import os
    from tools.b2d_adapter import FrameReader,RouteIndex
    root=str(Path(root).resolve())+'/'
    def guarded(function):
        def call(path,*a,**kw):
            if isinstance(path,(str,Path)) and str(path).startswith(root):
                raise AssertionError('Attempted raw data access: '+str(path))
            return function(path,*a,**kw)
        return call
    with patch.object(FrameReader,'read',side_effect=AssertionError('FrameReader called')), \
         patch.object(RouteIndex,'from_dataset_root',side_effect=AssertionError('RouteIndex called')), \
         patch.object(builtins,'open',guarded(builtins.open)), \
         patch.object(io,'open',guarded(io.open)), \
         patch.object(os,'scandir',guarded(os.scandir)):
        yield


class GuardedSubset:
    def __init__(self,data,indices):self.data,self.indices=data,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        with no_raw_reads(self.data.config['dataset_root']):
            pair=self.data[self.indices[index]]
        assert self.data._pid==__import__('os').getpid()
        assert all(db.execute('PRAGMA query_only').fetchone()[0]==1 for db in self.data._connections.values())
        return pair


def main():
    import torch
    from torch.utils.data import DataLoader
    from tools.training.diffusiondrive_v1.paired_dataset import paired_datasets
    from tools.training.diffusiondrive_v1.recovery_paired import RecoveryPairedDataset
    from tools.training.diffusiondrive_v1.runtime import fit_configuration,model_configuration
    from tools.training.diffusiondrive_v1.cache_only_reader import CacheOnlyPairedDataset
    from tools.training.diffusiondrive_v1 import features_cache as fc
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cfg,_=fit_configuration(args.config);cfg.update(cache_read_mode='cache-only',mount_check_mode='procfs')
    data,identity=paired_datasets(cfg['paired_manifest'],cfg,model_configuration(cfg))
    assert all(isinstance(d,CacheOnlyPairedDataset) for d in data.values())
    selected={};counts={};references={};samples=[]
    for split,limit in [('train',16),('validation',4)]:
        ds=data[split];ids=[];towns=set()
        for i,row in enumerate(ds.rows):
            town=re.search(r'_(Town\d+(?:HD)?)_',row['route']).group(1)
            if town not in towns:ids.append(i);towns.add(town)
            if len(ids)>=limit:break
        candidates=torch.randperm(len(ds),generator=torch.Generator().manual_seed(20261020)).tolist()
        for i in candidates:
            if len(ids)>=limit:break
            if i not in ids:ids.append(i)
        selected[split]=ids
        for i in ids:
            original=RecoveryPairedDataset.__getitem__(ds,i)
            with no_raw_reads(cfg['dataset_root']):cached=ds[i]
            assert fingerprint(original)==fingerprint(cached)
            for a,b in zip(original,cached):
                assert a.keys()==b.keys()
                for key in a:assert torch.equal(a[key],b[key]) and a[key].dtype==b[key].dtype
            references[(split,i)]=fingerprint(cached)
            samples.append({'split':split,'sample_id':ds.rows[i]['sample_id'],'route':ds.rows[i]['route'],'frame':ds.rows[i]['frame']})
        counts[split]=len(ids)
    # Deliberate metadata mismatch never silently becomes background/defaults.
    ds=data['train'];i=selected['train'][0];saved=dict(ds.rows[i]);rejections=[]
    try:
        ds.rows[i]['target_sha']='0'*64
        with no_raw_reads(cfg['dataset_root']):
            try:ds[i]
            except Exception as exc:
                assert getattr(exc,'code',None)=='recovery_pair_index_mismatch';rejections.append('target_sha_mismatch')
            else:raise AssertionError('Bad metadata accepted')
    finally:ds.rows[i]=saved
    # Source index connections are retained, rather than reopened per sample.
    handles={k:id(v) for k,v in ds._connections.items()}
    with no_raw_reads(cfg['dataset_root']):ds[i]
    assert handles=={k:id(v) for k,v in ds._connections.items()}
    worker_records=[]
    for workers in [0,2]:
        hashes=[];start=time.monotonic()
        for features,targets in DataLoader(GuardedSubset(ds,selected['train']),batch_size=cfg['batch_size'],
                num_workers=workers,shuffle=False,**({'multiprocessing_context':'spawn','prefetch_factor':2} if workers else {})):
            for j in range(len(targets['trajectory'])):
                hashes.append(fingerprint(({k:v[j] for k,v in features.items()},{k:v[j] for k,v in targets.items()})))
        assert hashes==[references[('train',i)] for i in selected['train']]
        worker_records.append({'workers':workers,'samples':len(hashes),'seconds':time.monotonic()-start})
    # Crossover timing, same actual cached inputs. This is not a GPU benchmark.
    runs=[]
    for mode in ['audited','cache-only','cache-only','audited']:
        start=time.monotonic()
        for i in selected['train']:
            pair=ds[i] if mode=='cache-only' else RecoveryPairedDataset.__getitem__(ds,i)
            assert fingerprint(pair)==references[('train',i)]
        seconds=time.monotonic()-start
        runs.append({'mode':mode,'samples':len(selected['train']),'seconds':seconds,'samples_per_second':len(selected['train'])/seconds})
        print(json.dumps(runs[-1]),flush=True)
    result={'status':'passed','dataset_identity':identity,'selected_counts':counts,'samples':samples,
            'all_features_targets_bitwise_equal':True,'raw_access_blocked_in_main_and_workers':True,
            'persistent_readonly_connections_verified':True,'worker_runs':worker_records,'rejected':rejections,'runs':runs,
            'policy':'Cache SHA/producer/recorded lineage/reference remain checked; live raw file re-audit belongs to audited mode.',
            'limits':'Small stratified real-cache check while another training process may be active; no end-to-end GPU speed claim.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for ds in data.values():ds.close()
    print('cache-only reader parity, isolation and spawn workers passed',flush=True)


if __name__=='__main__':main()
