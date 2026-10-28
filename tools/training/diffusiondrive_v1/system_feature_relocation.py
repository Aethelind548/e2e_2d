"""Copy only indexed system-root Features with rsync, verify, never delete.

Publishes an immutable read-only location overlay. Existing configs, manifests,
SQLite rows, producers, root markers and source files are not changed.
"""
import argparse
from contextlib import ExitStack
import csv
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

from tools.training.diffusiondrive_v1.features_cache import active_mount,digest,file_digest,durable_json,safe_path
from tools.training.diffusiondrive_v1.feature_jobs import config_and_state

ROOT=Path(__file__).resolve().parents[3]
SCHEMA='b2d-v1-verified-feature-relocation-v1'


def effective_roots(manifest,source):
    roots=dict(source['roots']);registration=manifest.get('feature_storage_registration')
    if not registration:return roots
    path=Path(registration['path'])
    if file_digest(path)!=registration['sha256']:raise RuntimeError('Feature location registration SHA changed')
    reg=json.loads(path.read_text())
    if reg['schema']!=SCHEMA or not reg['verified'] or reg['producer_sha256']!=source['producer_sha256']:
        raise RuntimeError('Unverified/different Feature relocation producer')
    entry=reg['jobs'].get(str(Path(source['state']).resolve()))
    if entry is None:return roots
    with sqlite3.connect(f"file:{Path(source['state'])/'index.sqlite'}?mode=ro",uri=True) as db:
        meta={k:json.loads(v) for k,v in db.execute("select key,value from meta where key in ('index_id','producer')")}
    if meta['index_id']!=entry['index_id'] or digest(meta['producer'])!=source['producer_sha256']:
        raise RuntimeError('Relocated source index identity changed')
    for rid,item in entry['relocations'].items():
        if roots[rid]!=item['from']:raise RuntimeError('Relocation original root differs')
        active_mount(item['to'])
        if file_digest(Path(item['to']['path'])/'cache_root.json')!=item['destination_owner_sha256']:
            raise RuntimeError('Relocation destination owner changed')
        roots[rid]=item['to']
    return roots


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    args=p.parse_args();state=args.state.resolve();state.mkdir(parents=True,exist_ok=True)
    if (state/'storage_registration.json').exists():raise FileExistsError('Verified relocation already published')
    collection=json.loads((ROOT/'tools/training/diffusiondrive_v1/features_jobs.json').read_text())
    jobs=[];all_roots=[]
    with ExitStack() as stack:
        for item in collection['jobs']:
            cfg,folder=config_and_state(ROOT/item['config'])
            lock=stack.enter_context((folder/'writer.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            db=sqlite3.connect(f'file:{folder}/index.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
            meta={r['key']:json.loads(r['value']) for r in db.execute('select * from meta')}
            assert digest(meta['producer'])==collection['producer_sha256']
            for rid,spec in cfg['cache_roots'].items():
                count,size=db.execute("select count(*),coalesce(sum(bytes),0) from samples where state='complete' and root_id=?",(rid,)).fetchone()
                mount=active_mount(spec)
                all_roots.append({'job':item['job_id'],'index':str(folder/'index.sqlite'),'root_id':rid,'spec':spec,'count':count,'bytes':size,'mount':mount})
                if spec['uuid']!='932410d9-a546-40dd-aeda-09e390df924b':continue
                rows=[dict(r) for r in db.execute("select * from samples where state='complete' and root_id=? order by relative_path",(rid,))]
                if rows:jobs.append({'name':item['job_id'],'state':str(folder),'index_id':meta['index_id'],'root_id':rid,'source':spec,'rows':rows})
            db.close()
        assert jobs,'No indexed system Features found'
        shared=json.loads((ROOT/'tools/training/diffusiondrive_v1/features_shared_storage.json').read_text())
        dest=dict(shared['roots']['cache_b']['spec']);info=active_mount(dest,writing=True)
        assert dest['uuid']=='F8E4AE34E4ADF552' and dest['fstype']=='ntfs3'
        locks=set()
        for path in [Path(dest['path'])]+[Path(j['source']['path']) for j in jobs]:
            if str(path) in locks:continue
            stream=stack.enter_context((path/'.writer.lock').open('a'));fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.add(str(path))
        total=sum(r['bytes'] for j in jobs for r in j['rows'])
        if info['free_bytes']<total+100*1024**3:raise RuntimeError('PSSD reserve would fall below 100 GiB')
        owner_sha=file_digest(Path(dest['path'])/'cache_root.json')
        inventory={'roots':all_roots,'selected_count':sum(len(j['rows']) for j in jobs),'selected_bytes':total,'destination':dest,'sources_preserved':True}
        durable_json(state/'inventory.json',inventory)
        journal=sqlite3.connect(state/'verification.sqlite');journal.execute('pragma synchronous=FULL')
        journal.execute('create table if not exists files(sample_id text primary key,source text,destination text,sha256 text,bytes integer,verified integer default 0)')
        started=time.monotonic();registrations={}
        for job in jobs:
            source=job['source'];flist=state/(job['name']+'_files.txt')
            with flist.open('w') as f:
                for row in job['rows']:
                    src=safe_path(source,row['relative_path']);dst=safe_path(dest,row['relative_path'])
                    assert src.is_file() and not src.is_symlink() and src.stat().st_size==row['bytes']
                    if dst.exists() and (dst.is_symlink() or dst.stat().st_size!=row['bytes'] or file_digest(dst)!=row['sha256']):
                        raise RuntimeError('Refuse to overwrite unknown destination: '+str(dst))
                    assert '\n' not in row['relative_path']
                    f.write(row['relative_path']+'\n')
                    journal.execute('insert or ignore into files values(?,?,?,?,?,0)',(row['sample_id'],str(src),str(dst),row['sha256'],row['bytes']))
            journal.commit()
            active_mount(source);active_mount(dest,writing=True)
            with (state/'rsync.log').open('a') as log:
                subprocess.run(['rsync','-rt','--ignore-existing','--no-perms','--no-owner','--no-group','--partial-dir=.rsync-partial-system-v1',
                    '--files-from='+str(flist),'--info=progress2',source['path']+'/',dest['path']+'/'],stdout=log,stderr=subprocess.STDOUT,check=True)
            copied_seconds=time.monotonic()-started
            durable_json(state/'progress.json',{'phase':'verify','copied_count':len(job['rows']),'copied_seconds':copied_seconds})
            print(json.dumps({'phase':'copy_complete','files':len(job['rows']),'seconds':copied_seconds}),flush=True)
            for i,row in enumerate(job['rows']):
                if (state/'stop.request').exists():raise RuntimeError('Paused at verification boundary; source files retained')
                if i%200==0:active_mount(source);active_mount(dest,writing=True)
                src=safe_path(source,row['relative_path']);dst=safe_path(dest,row['relative_path']);before=src.stat()
                if file_digest(src)!=row['sha256'] or file_digest(dst)!=row['sha256'] or dst.stat().st_size!=row['bytes']:
                    raise RuntimeError('Content verification failed: '+row['sample_id'])
                after=src.stat()
                assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
                journal.execute('update files set verified=1 where sample_id=?',(row['sample_id'],))
                if (i+1)%500==0:
                    journal.commit();durable_json(state/'progress.json',{'phase':'verify','verified':i+1,'total':len(job['rows']),'seconds':time.monotonic()-started})
                    print(json.dumps({'phase':'verify','verified':i+1,'total':len(job['rows'])}),flush=True)
            journal.commit()
            registrations[job['state']]={'index_id':job['index_id'],'selected_rows_sha256':digest([(r['sample_id'],r['relative_path'],r['sha256'],r['bytes']) for r in job['rows']]),
                'count':len(job['rows']),'relocations':{job['root_id']:{'from':source,'to':dest,'destination_owner_sha256':owner_sha}}}
        assert journal.execute('select count(*) from files where verified=0').fetchone()[0]==0
        with (state/'old_cache_cleanup_candidates.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['sample_id','source','verified_destination','sha256','bytes'])
            writer.writerows(journal.execute('select sample_id,source,destination,sha256,bytes from files order by source'))
        journal.close()
        durable_json(state/'storage_registration.json',{'schema':SCHEMA,'verified':True,'producer_sha256':collection['producer_sha256'],
            'jobs':registrations,'verified_files':inventory['selected_count'],'verified_bytes':total,'source_files_deleted':0,
            'copied_seconds':copied_seconds,'total_seconds':time.monotonic()-started,'verification':'Every source and destination SHA256 equals frozen SQLite SHA; every size matches.'})
        durable_json(state/'progress.json',{'phase':'complete','verified':inventory['selected_count'],'source_files_deleted':0,'seconds':time.monotonic()-started})
        print(json.dumps({'phase':'complete','files':inventory['selected_count'],'source_files_deleted':0}),flush=True)


if __name__=='__main__':main()
