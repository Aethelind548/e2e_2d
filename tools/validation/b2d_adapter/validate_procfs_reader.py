"""Bounded real-cache parity and disk-identity rejection checks; no GPU training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def main():
    import torch
    from tools.training.diffusiondrive_v1 import features_cache as fc, target_cache as tc, system_feature_relocation as relocation
    from tools.training.diffusiondrive_v1.procfs_mount_reader import active_mount,storage_check,mount_scope,ProcfsMountPairedDataset
    from tools.training.diffusiondrive_v1.recovery_paired import RecoveryPairedDataset
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cfg=json.loads(args.config.read_text());manifest=json.loads(Path(cfg['paired_manifest']).read_text())
    checks=[]
    for source in manifest['feature_sources'].values():
        for spec in relocation.effective_roots(manifest,source).values():
            old=fc.active_mount(spec);new=active_mount(spec)
            assert all(old[k]==new[k] for k in ['target','uuid','fstype','maj:min'])
            checks.append({'uuid':new['uuid'],'mount':new['target']})
    for source in manifest['target_sources'].values():
        assert abs(tc.storage_check(source['target_root'])-storage_check(source['target_root']))<1024**3
    spec=relocation.effective_roots(manifest,manifest['feature_sources']['remaining_20261015'])['cache_b']
    rejected=[]
    for name,wrong in [('wrong_uuid',dict(spec,uuid='48EA01DBEA01C5E2')),('wrong_fstype',dict(spec,fstype='ext4')),
                       ('missing_mount',dict(spec,mountpoint='/tmp',path='/tmp/missing_feature_root'))]:
        try:active_mount(wrong)
        except fc.StoragePause:rejected.append(name)
        else:raise AssertionError('Unsafe mount accepted: '+name)
    # Read-only metadata injection tests the parser, without remounting a disk.
    actual_text=Path('/proc/self/mountinfo').read_text();lines=[]
    for line in actual_text.splitlines():
        before,after=line.split(' - ',1);fields=before.split()
        if fields[4]==spec['mountpoint']:
            fields[5]=fields[5].replace('rw','ro');before=' '.join(fields)
        lines.append(before+' - '+after)
    original_read=Path.read_text
    def read_text(path,*a,**kw):
        return '\n'.join(lines) if str(path)=='/proc/self/mountinfo' else original_read(path,*a,**kw)
    with patch.object(Path,'read_text',read_text):
        try:active_mount(spec,writing=True)
        except fc.StoragePause:rejected.append('read_only_write')
        else:raise AssertionError('Read-only write accepted')
    originals=(fc.active_mount,tc.storage_check,relocation.active_mount)
    try:
        with mount_scope():raise RuntimeError('fixture')
    except RuntimeError:pass
    assert (fc.active_mount,tc.storage_check,relocation.active_mount)==originals
    old=RecoveryPairedDataset(manifest,cfg,'train');new=ProcfsMountPairedDataset(manifest,cfg,'train')
    ids=torch.randperm(len(old),generator=torch.Generator().manual_seed(20261019))[:20].tolist()
    runs=[];fingerprints=[]
    for name,reader in [('findmnt',old),('procfs',new),('procfs',new),('findmnt',old)]:
        hashes=[];start=time.monotonic()
        for i in ids:
            pair=reader[i];digest=hashlib.sha256()
            for group in pair:
                for key,tensor in sorted(group.items()):
                    digest.update(key.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
            hashes.append(digest.hexdigest())
        seconds=time.monotonic()-start
        fingerprints.append(hashes);runs.append({'mode':name,'samples':len(ids),'seconds':seconds,'samples_per_second':len(ids)/seconds})
        print(json.dumps(runs[-1]),flush=True)
    assert all(hashes==fingerprints[0] for hashes in fingerprints)
    # A corrupted cache must still fail under the new guard; reuse a row copy
    # with a deliberately wrong SHA so no physical file is changed.
    import sqlite3
    row=old.rows[ids[0]];source=manifest['feature_sources'][row['feature_source']]
    with sqlite3.connect(f"file:{Path(source['state'])/'index.sqlite'}?mode=ro",uri=True) as db:
        db.row_factory=sqlite3.Row;cached=dict(db.execute('select * from samples where sample_id=?',(row['sample_id'],)).fetchone())
    cached['sha256']='0'*64
    with mount_scope():
        try:fc.load_feature_row(cached,relocation.effective_roots(manifest,source),source['producer_sha256'])
        except Exception as exc:
            assert getattr(exc,'code',None)=='invalid_feature_cache',repr(exc)
            rejected.append('cache_sha_mismatch')
        else:raise AssertionError('Checksum mismatch accepted')
    assert (fc.active_mount,tc.storage_check,relocation.active_mount)==originals
    result={'status':'passed','mount_identity_parity':checks,'rejected_cases':rejected,'all_tensor_bytes_equal':True,
            'runs':runs,'sample_ids':[old.rows[i]['sample_id'] for i in ids],
            'limits':'20 real examples, crossover order; current formal training may contend for I/O; no GPU or full training speed claim.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print('All kernel mount guard and real tensor parity checks passed',flush=True)


if __name__=='__main__':main()
