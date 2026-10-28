"""Targeted real-cache parity, one-open, corruption and spawn-worker checks."""
import argparse
import builtins
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch


@contextmanager
def artifact_opens():
    opened = []
    def observe(function):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).suffix == '.pt':
                opened.append(str(path))
            return function(path, *args, **kwargs)
        return call
    with patch.object(io, 'open', observe(io.open)), patch.object(builtins, 'open', observe(builtins.open)):
        yield opened


class CheckedSubset:
    def __init__(self, data, indices):self.data, self.indices = data, indices
    def __len__(self):return len(self.indices)
    def __getitem__(self, i):
        from tools.validation.b2d_adapter.validate_cache_only_reader import no_raw_reads
        with artifact_opens() as opened, no_raw_reads(self.data.config['dataset_root']):
            pair = self.data[self.indices[i]]
        assert len(opened) == 2 and len(set(opened)) == 2, opened
        return pair


def main():
    import torch
    from torch.utils.data import DataLoader
    from tools.training.diffusiondrive_v1.runtime import fit_configuration,model_configuration
    from tools.training.diffusiondrive_v1.paired_dataset import paired_datasets
    from tools.training.diffusiondrive_v1 import features_cache as fc, target_cache as tc, target_storage as ts
    from tools.training.diffusiondrive_v1.single_pass_cache_reader import single_pass_scope, VerifiedBufferReader
    from tools.training.diffusiondrive_v1.procfs_mount_reader import mount_scope
    from tools.validation.b2d_adapter.validate_cache_only_reader import fingerprint,no_raw_reads
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result={'status':'running'}
    try:
        torch.set_num_threads(1)
        cfg,_=fit_configuration(args.config)
        cfg.update(cache_read_mode='cache-only',mount_check_mode='procfs',cache_file_read_mode='verified-single-pass')
        data,identity=paired_datasets(cfg['paired_manifest'],cfg,model_configuration(cfg))
        original=(fc.file_digest,tc.file_digest,torch.load,ts.RAW_LOAD)
        def restored():assert original==(fc.file_digest,tc.file_digest,torch.load,ts.RAW_LOAD)
        counts={};samples=[];reference={};selection={}
        for split,limit in [('train',16),('validation',4)]:
            ds=data[split];ids=[];towns=set()
            for i,row in enumerate(ds.rows):
                town=row['route'].split('_Town')[1].split('_')[0]
                if town not in towns:ids.append(i);towns.add(town)
                if len(ids)>=limit:break
            for i in torch.randperm(len(ds),generator=torch.Generator().manual_seed(20261020)).tolist():
                if len(ids)>=limit:break
                if i not in ids:ids.append(i)
            selection[split]=ids
            for i in ids:
                ds.file_read_mode='legacy'
                with artifact_opens() as old_opens, no_raw_reads(cfg['dataset_root']):a=ds[i]
                ds.file_read_mode='verified-single-pass'
                with artifact_opens() as new_opens, no_raw_reads(cfg['dataset_root']):b=ds[i]
                assert len(old_opens)==4 and len(new_opens)==2, (old_opens,new_opens)
                assert sorted(set(old_opens))==sorted(new_opens)
                for x,y in zip(a,b):
                    assert x.keys()==y.keys()
                    for key in x:assert torch.equal(x[key],y[key]) and x[key].dtype==y[key].dtype
                assert fingerprint(a)==fingerprint(b)
                reference[(split,i)]=fingerprint(b)
                samples.append(dict(split=split,sample_id=ds.rows[i]['sample_id'],route=ds.rows[i]['route'],frame=ds.rows[i]['frame'],legacy_opens=len(old_opens),single_pass_opens=len(new_opens)))
                restored()
            counts[split]=len(ids)
        rejected=[];ds=data['train'];pair=ds.rows[selection['train'][0]]
        fs=ds.manifest['feature_sources'][pair['feature_source']];tsrc=ds.manifest['target_sources'][pair['target_source']]
        fr=dict(ds._row(fs,pair['sample_id']));tr=dict(ds._row(tsrc,pair['sample_id'],target=True))
        roots=ds._roots[pair['feature_source']]
        def reject(name,call,expected):
            reader=None
            try:
                with mount_scope(),single_pass_scope() as reader:call()
            except Exception as exc:
                assert isinstance(exc,expected),repr(exc)
                rejected.append(dict(name=name,error_type=type(exc).__name__,code=getattr(exc,'code',None),stats=dict(reader.stats)))
            else:raise AssertionError(name+' accepted')
            restored()
        reject('feature_bad_sha',lambda:fc.load_feature_row(dict(fr,sha256='0'*64),roots,fs['producer_sha256']),fc.AdapterError)
        reject('feature_bad_size',lambda:fc.load_feature_row(dict(fr,bytes=fr['bytes']+1),roots,fs['producer_sha256']),fc.AdapterError)
        reject('feature_bad_producer',lambda:fc.load_feature_row(fr,roots,'0'*64),fc.StoragePause)
        reject('target_bad_sha',lambda:ts.load_target_row(dict(tr,sha256='0'*64),tsrc['target_root'],tsrc['producer_sha256']),tc.AdapterError)
        reject('target_bad_map_contract',lambda:ts.load_target_row(dict(tr,map_json='{}'),tsrc['target_root'],tsrc['producer_sha256']),tc.AdapterError)
        with tempfile.TemporaryDirectory(prefix='v1_single_pass_') as tmp:
            path=Path(tmp)/'fixture.pt';replacement=Path(tmp)/'replacement.pt'
            torch.save({'x':torch.arange(4)},path)
            replacement.write_bytes(path.read_bytes())
            reader=VerifiedBufferReader(torch.load);reader.digest(path);replacement.replace(path)
            try:reader.load(path,map_location='cpu',weights_only=True)
            except fc.StoragePause:rejected.append({'name':'replacement_after_sha','same_file_bytes':True})
            else:raise AssertionError('Replaced file accepted')
            finally:reader.close()
            with patch.object(Path,'open',side_effect=OSError('injected I/O error')):
                try:VerifiedBufferReader(torch.load).digest(path)
                except fc.StoragePause:rejected.append({'name':'read_io_failure'})
                else:raise AssertionError('Read error ignored')
        worker_runs=[]
        for workers in [0,2]:
            fingerprints=[];start=time.monotonic()
            for f,t in DataLoader(CheckedSubset(ds,selection['train']),batch_size=8,num_workers=workers,
                    **({'multiprocessing_context':'spawn','prefetch_factor':2} if workers else {})):
                fingerprints += [fingerprint(({k:v[j] for k,v in f.items()},{k:v[j] for k,v in t.items()})) for j in range(len(t['trajectory']))]
            assert fingerprints==[reference[('train',i)] for i in selection['train']]
            worker_runs.append(dict(workers=workers,samples=len(fingerprints),seconds=time.monotonic()-start))
            restored()
        runs=[]
        for mode in ['legacy','verified-single-pass','verified-single-pass','legacy']:
            ds.file_read_mode=mode;start=time.monotonic()
            for i in selection['train']:assert fingerprint(ds[i])==reference[('train',i)]
            seconds=time.monotonic()-start
            runs.append(dict(mode=mode,samples=len(selection['train']),seconds=seconds,samples_per_second=len(selection['train'])/seconds))
        result.update(status='passed',dataset_identity=identity,selected_counts=counts,samples=samples,
            bitwise_equal=True,all_frozen_loaders_restored=True,worker_runs=worker_runs,rejected=rejected,runs=runs,
            policy='Every artifact SHA/size and all frozen producer/lineage/reference/map/dtype/shape/value checks retained; numeric source files unchanged.',
            limits='Small real-cache parity and warm-page crossover while GPU training may run; not an end-to-end training speed benchmark.')
        for ds in data.values():ds.close()
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','selected_counts','worker_runs','runs']},indent=2),flush=True)


if __name__=='__main__':main()
