"""Flatten in-progress destination copies; original external caches stay untouched."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import time

from tools.training.diffusiondrive_v1 import feature_migration as m
from tools.training.diffusiondrive_v1 import feature_shared_storage as shared


def move_verified_copy(old, new, row):
    """Same-volume rename, recoverable after rename and before journal update."""
    if new.exists():
        m.verify_file(new,row)
        if old.exists():
            before=m.stamp(old); m.verify_file(old,row)
            if m.stamp(old)!=before: raise m.StoragePause('Old destination copy changed')
            old.unlink(); m.directory_sync(old.parent)
        return
    before=m.stamp(old); m.verify_file(old,row)
    if m.stamp(old)!=before: raise m.StoragePause('Old destination copy changed')
    new.parent.mkdir(parents=True,exist_ok=True)
    if old.stat().st_dev != new.parent.stat().st_dev:
        raise m.StoragePause('Flatten requires a same-volume rename')
    os.replace(old,new)
    m.directory_sync(new.parent); m.directory_sync(old.parent)


def prune_empty_namespace(path, expected_owner):
    if not path.exists(): return
    allowed={path/'cache_root.json',path/'.writer.lock'}
    for current,dirs,files in os.walk(path,topdown=False,followlinks=False):
        current=Path(current)
        for filename in files:
            p=current/filename
            if p not in allowed:
                raise m.StoragePause('Untracked file retained in old namespace: '+str(p))
        for dirname in dirs:
            p=current/dirname
            if p.is_symlink(): raise m.StoragePause('Unexpected symlink in old namespace')
            p.rmdir()
    marker=path/'cache_root.json'
    if marker.exists() and m.read_json(marker)!=expected_owner:
        raise m.StoragePause('Old namespace owner changed')
    for p in allowed: p.unlink(missing_ok=True)
    path.rmdir(); m.directory_sync(path.parent)


def make_flat_plan(old):
    plan=json.loads(json.dumps(old)); plan['flat_layout']=True
    plan['layout_version']='shared-owner-flat-v1'
    registry={'schema':shared.REGISTRY_SCHEMA,'dataset_id':plan['jobs'][0]['dataset_id'],
              'producer_sha256':plan['jobs'][0]['producer_sha256'],'roots':{}}
    for root_id,disk in plan['disks'].items():
        spec=dict(plan['jobs'][0]['destinations'][root_id])
        spec['path']=str(Path(spec['path']).parent)
        spec['budget_gib']=disk['budget_gib']
        members=[{'index_id':j['index_id'],'state':j['state'],'job':j['name']} for j in plan['jobs']]
        sid=m.digest({'schema':shared.REGISTRY_SCHEMA,'dataset_id':registry['dataset_id'],
                      'producer_sha256':registry['producer_sha256'],'root_id':root_id,'members':members})[:32]
        registry['roots'][root_id]={'spec':spec,'storage_owner_id':sid,'members':members}
        for job in plan['jobs']:
            if (job['dataset_id']!=registry['dataset_id'] or job['producer_sha256']!=registry['producer_sha256']):
                raise m.StoragePause('Cannot share roots across different datasets/producers')
            job['destinations'][root_id]=dict(spec)
            alias_root=str(Path(job['aliases'][root_id]['path']).parent)
            job['aliases'][root_id]=dict(spec,path=alias_root,mountpoint=alias_root)
            # UUID-gated cache roots use physical paths beneath each mounted
            # volume. The bind aliases expose the exact same directories.
            job['after']['cache_roots'][root_id]=dict(spec)
            job['after']['shared_feature_roots']=str(shared.REGISTRY.relative_to(m.ROOT))
    return plan,registry


def flatten():
    with m.locked(m.STATE/'launch.lock'),m.locked(m.STATE/'worker.lock'),ExitStack() as locks:
        if (m.STATE/'cutover.json').exists():
            raise m.StoragePause('Flatten only before the original migration cutover')
        backup=m.STATE/'plan_before_flatten.json'
        current=m.read_json(m.STATE/'plan.json')
        if current.get('flat_layout') and not (m.STATE/'flatten_in_progress.json').exists():
            print('Destination layout already flat',flush=True); return
        if not backup.exists(): m.durable_json(backup,current)
        old=m.read_json(backup)
        plan,registry=make_flat_plan(old)
        m.durable_json(m.STATE/'flatten_in_progress.json',{'started':time.time(),'backup':str(backup)})
        (m.STATE/'stop.request').unlink(missing_ok=True)
        def request_stop(*_): m.durable_json(m.STATE/'stop.request',{'requested':time.time()})
        previous={sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
        db=m.connection()
        try:
            if db.execute("SELECT COUNT(*) FROM files WHERE phase='deleted'").fetchone()[0]:
                raise m.StoragePause('Original cleanup already started; retain current layout')
            if db.execute('SELECT 1 FROM files GROUP BY root_id,relative_path HAVING COUNT(*)>1 LIMIT 1').fetchone():
                raise m.StoragePause('Duplicate sample path across source indexes')
            acquired=set()
            for job in old['jobs']:
                m.validate_job(job)
                for spec in job['destinations'].values():
                    path=Path(spec['path'])
                    if path.is_dir():
                        locks.enter_context(m.locked(path/'.writer.lock'))
            for root_id,entry in registry['roots'].items():
                spec=entry['spec']; m.active_mount(spec,writing=True)
                path=Path(spec['path']); marker_path=path/'cache_root.json'
                expected=shared.marker(registry,root_id)
                if marker_path.exists() and m.read_json(marker_path)!=expected:
                    raise m.StoragePause('Refuse to replace a foreign shared storage owner')
                lock_path=path/'.writer.lock'
                if lock_path not in acquired:
                    locks.enter_context(m.locked(lock_path)); acquired.add(lock_path)
                if not marker_path.exists(): m.durable_json(marker_path,expected)
            if shared.REGISTRY.exists() and m.read_json(shared.REGISTRY)!=registry:
                raise m.StoragePause('Shared registry already has another identity')
            m.durable_json(shared.REGISTRY,registry)
            old_jobs={j['name']:j for j in old['jobs']}; new_jobs={j['name']:j for j in plan['jobs']}
            rows=db.execute("SELECT * FROM files WHERE phase='copied' ORDER BY job,root_id,sample_id").fetchall()
            m.progress(db,'flattening',flattened_files=0,total_to_flatten=len(rows))
            for count,row in enumerate(rows,1):
                if m.stopped(): raise m.StoragePause('Flatten stopped; rerun flatten to resume')
                old_spec=old_jobs[row['job']]['destinations'][row['root_id']]
                new_spec=new_jobs[row['job']]['destinations'][row['root_id']]
                if count%250==1:
                    m.active_mount(new_spec,writing=True)
                move_verified_copy(m.safe_path(old_spec,row['relative_path']),m.safe_path(new_spec,row['relative_path']),row)
                if count%250==0:
                    m.progress(db,'flattening',flattened_files=count,total_to_flatten=len(rows))
            # Numeric sample paths and source stamps stay unchanged. Only the
            # destination-root mapping changes after every placed copy is valid.
            m.durable_json(m.STATE/'plan.json',plan)
            for job in old['jobs']:
                for root_id,spec in job['destinations'].items():
                    prune_empty_namespace(Path(spec['path']),m.owner(job,root_id))
            m.validate_aliases(plan)
            (m.STATE/'flatten_in_progress.json').unlink()
            m.directory_sync(m.STATE)
            m.progress(db,'flat_layout_ready',flattened_files=len(rows))
        except BaseException as exc:
            m.progress(db,'flatten_paused',error=str(exc))
            raise
        finally:
            db.close()
            for sig,handler in previous.items(): signal.signal(sig,handler)
