"""Isolated, explicit recovery queue; never edits a baseline index or producer.

Only the already verified irrelevant-tilted-vehicle rule is executable. Other
reasons are inventoried for review, not accepted by weakening their gates.
Partial components are diagnostic artifacts, never four-task training samples.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1 import target_cache as core, target_jobs as jobs
from tools.training.diffusiondrive_v1.features_cache import digest, durable_json, file_digest, pin, StoragePause
from tools.training.diffusiondrive_v1.target_recovery import IrrelevantActorRecoveryBuilder, RULE
from tools.training.diffusiondrive_v1.target_storage import codec_scope, load_target_row

EXECUTABLE = {'invalid_vehicle_box'}

# The original API above/below remains available for historical diagnostics.
# Full recovery uses NEW state/roots and producer identities; old completed
# artifacts are read with their pinned producer, never re-identified as current.
FULL_VERSION = 'b2d-v1-full-target-recovery-v1'


def full_identity(config,profile):
    from tools.training.diffusiondrive_v1.runtime import model_configuration,native_configuration
    from tools.training.diffusiondrive_v1.full_recovery_rules import GEOMETRY_CONTRACT
    base=core.producer_identity(config,model_configuration(config),native_configuration(config))
    result=dict(base,full_recovery={'version':FULL_VERSION,'profile':profile,
        'implementation_sha256':file_digest(Path(__file__)),
        'rules_sha256':file_digest(Path(__file__).with_name('full_recovery_rules.py')),
        'legacy_rule_sha256':file_digest(Path(__file__).with_name('target_recovery.py')),
        'baseline_producer_sha256':digest(base),'missing_future_recovery':False})
    if profile=='projected':
        result['geometry_contract']=GEOMETRY_CONTRACT
        result['contract']=dict(base['contract'],schema='b2d-v1-projected-target-contract-v1',
            projected_geometry=GEOMETRY_CONTRACT)
    return result


def full_db(state):
    state=Path(state).resolve();state.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(state/'recovery.sqlite',timeout=30);db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tasks(sample_id TEXT PRIMARY KEY,route TEXT NOT NULL,frame INTEGER NOT NULL,
      original_code TEXT NOT NULL,baseline_row TEXT NOT NULL,status TEXT NOT NULL,
      diagnosis TEXT,profile TEXT,attempts INTEGER NOT NULL DEFAULT 0,last_rule TEXT,updated REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS recovery_queue ON tasks(status,route,frame);
    CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,sample_id TEXT,phase TEXT,rule_id TEXT,
      outcome TEXT,record TEXT,created REAL,UNIQUE(sample_id,phase,rule_id));''')
    return db


def prior_recoveries(config):
    paths=[ROOT/'outputs/diffusiondrive_v1/targets_recovery_irrelevant_v1',ROOT/'outputs/diffusiondrive_v1/target_retry_audit_20261017']
    result={};snapshots=[]
    for state in paths:
        with readonly(state/'index.sqlite') as db:
            meta={r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
            rows=[dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY sample_id")]
        for row in rows:
            if row['sample_id'] in result:raise StoragePause('Duplicate historical recovered sample_id')
            result[row['sample_id']]={'row':row,'spec':meta['target_root'],'producer':meta['producer'],'state':str(state)}
        snapshots.append({'state':str(state),'complete_rows_sha256':digest(rows),'count':len(rows),
            'producer_sha256':digest(meta['producer'])})
    return result,snapshots


def full_plan(config,state,cache_root):
    import fcntl
    state=Path(state).resolve();cache_root=Path(cache_root).resolve()
    if state==(ROOT/config['target_state_dir']).resolve() or cache_root==Path(config['target_root']['path']).resolve():
        raise StoragePause('Full recovery cannot own baseline state/cache')
    prior,snapshots=prior_recoveries(config);base,rows,stats=baseline_rows(config)
    if digest(base)!=full_identity(config,'compatible')['full_recovery']['baseline_producer_sha256']:
        raise StoragePause('Frozen baseline numeric implementation changed')
    state.mkdir(parents=True,exist_ok=True)
    with (state/'retry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);db=full_db(state)
        try:
            pin(db,'version',FULL_VERSION);pin(db,'baseline_producer',base);pin(db,'candidate_rows_sha256',digest(rows))
            pin(db,'prior_recoveries',snapshots);pin(db,'cache_root',str(cache_root));pin(db,'dataset_id',config['dataset_id'])
            for profile in ['compatible','projected']:pin(db,'producer_'+profile,full_identity(config,profile))
            for row in rows:
                status='historical_complete' if row['sample_id'] in prior else 'pending_diagnosis'
                db.execute('INSERT OR IGNORE INTO tasks(sample_id,route,frame,original_code,baseline_row,status,updated) VALUES(?,?,?,?,?,?,?)',
                    (row['sample_id'],row['route'],row['frame'],row['error_code'],json.dumps(row),status,time.time()))
            db.commit()
            durable_json(state/'configuration.json',dict(config,full_recovery_state=str(state),full_recovery_cache=str(cache_root)))
            durable_json(state/'plan.json',{'version':FULL_VERSION,'original_nonfuture':len(rows),'prior_complete':len(prior),
                'remaining':len(rows)-len(prior),'baseline_stats':stats,'cache_root':str(cache_root),
                'profiles':{p:digest(full_identity(config,p)) for p in ['compatible','projected']}})
            return dict(db.execute('SELECT status,COUNT(*) FROM tasks GROUP BY status'))
        finally:db.close()


def full_selection(args,sql='1=1'):
    clauses=[sql];params=[]
    for name,column in [('town','route'),('route','route'),('error','original_code')]:
        value=getattr(args,name,None)
        if value:
            clauses.append(column+(' LIKE ?' if name in ('town','route') else '=?'))
            params.append('%_'+value+'_%' if name=='town' else '%'+value+'%' if name=='route' else value)
    return ' AND '.join(clauses),params


def full_source_scope():
    """Memoize verified compressed-byte SHA with stat/ctime change detection.

    The scope affects this one recovery process only. The frozen source_identity
    still records each actual annotation hash and rechecks all source stats.
    """
    from contextlib import contextmanager
    from collections import OrderedDict
    @contextmanager
    def scope():
        original=core.file_digest;memo=OrderedDict()
        def checked(path):
            path=Path(path);s=path.stat();finger=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            key=str(path)
            if key in memo:
                prior,sha=memo[key]
                if prior!=finger:raise StoragePause('Previously verified source changed: '+key)
                memo.move_to_end(key);return sha
            sha=original(path);after=path.stat()
            if finger!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise StoragePause('Source changed during SHA read')
            memo[key]=(finger,sha)
            while len(memo)>10000:memo.popitem(last=False)
            return sha
        core.file_digest=checked
        try:yield
        finally:core.file_digest=original
    return scope()


def full_diagnose_one(config,row,reader,native,model,provider):
    from tools.b2d_adapter.v1_builders import official_v1_module
    from tools.b2d_adapter.v1_bev import bev_annotations
    from tools.training.diffusiondrive_v1.full_recovery_rules import trajectory_diagnosis,detection_diagnosis,ego6_recipe
    from tools.training.diffusiondrive_v1.features_cache import sample_id
    ref=reader.index.ref(row['frame'])
    record={'sample_id':row['sample_id'],'route':row['route'],'frame':row['frame'],
        'trajectory_errors':[],'detection_errors':[],'bev_errors':[],'source_errors':[],'recovery_options':{}}
    try:
        if row['sample_id']!=sample_id(config['dataset_id'],row['route'],row['frame']):raise AdapterError('sample_id_mismatch',row['sample_id'])
        sources,mapped,future=core.source_identity(reader,ref,native,provider,Path(config['dataset_root']).resolve())
        raw=reader.read(ref);reference=core.reference_pose(raw,native)
        record.update(sources=sources,map=mapped,reference_pose=reference,future_frame_ids=future)
    except AdapterError as exc:
        jobs.failure_state(exc);record['source_errors'].append({'code':exc.code,'detail':str(exc)});return record,'blocked_source'
    record['trajectory_errors'],record['trajectory_metrics']=trajectory_diagnosis(reader,ref,native.geometry)
    official=official_v1_module('transfuser_features').TransfuserTargetBuilder(model)
    record['detection_errors'],record['actors']=detection_diagnosis(raw,native,official)
    map_data,_,_=provider.get_map(ref)
    _,coverage,map_errors=map_data.query(raw,native.geometry,reader,'strict')
    record['bev_errors'].extend(map_errors);record['strict_map_coverage']=coverage
    # Diagnose every actor independently, including nonvehicle BEV-only classes.
    for actor in raw.annotation['bounding_boxes']:
        isolated=copy.copy(raw);isolated.annotation=dict(raw.annotation,bounding_boxes=[actor])
        try:bev_annotations(isolated,native.geometry)
        except AdapterError as exc:record['bev_errors'].append({'code':exc.code,'detail':{'actor':str(actor.get('id')),'error':str(exc)}})
    for policy in ['compatible','projected']:
        _,coverage,remaining=map_data.query(raw,native.geometry,reader,policy)
        record['recovery_options'][policy]={'map_errors':remaining,'map_events':coverage['recovery_events']}
    if record['trajectory_metrics']['nonplanar_frames']:
        record['ego6_proposal']=ego6_recipe(reader,ref,native.geometry)
    if record['source_errors']:return record,'blocked_source'
    # Flag semantics take precedence over reference regeneration, preserving both errors.
    if any(e['code']=='teleport_flag' for e in record['trajectory_errors']):return record,'blocked_source_semantics'
    if any(e['code']=='nonplanar_pose' for e in record['trajectory_errors']):return record,'needs_feature_contract'
    if record['trajectory_errors']:return record,'blocked_trajectory'
    all_irrelevant=all(a.get('irrelevance',{}).get('safely_irrelevant',False) for a in record['actors'])
    non_tilt=[e for e in record['detection_errors'] if not isinstance(e['detail'],dict) or e['detail'].get('reason')!='tilt']
    if not non_tilt and all_irrelevant and not record['recovery_options']['compatible']['map_errors']:
        return record,'compatible_candidate'
    projection_errors=[a.get('projected_error') for a in record['actors'] if a.get('projected_error')]
    if not non_tilt and not projection_errors and not record['recovery_options']['projected']['map_errors']:
        return record,'projected_candidate'
    record['remaining_geometry_blockers']=non_tilt+projection_errors+record['recovery_options']['projected']['map_errors']
    return record,'blocked_geometry'


def full_run(config,state,args,phase):
    import copy
    import fcntl
    import signal
    import torch
    from tools.training.diffusiondrive_v1.runtime import model_configuration,native_configuration
    from tools.training.diffusiondrive_v1.full_recovery_rules import MemoReader,RecoveryProvider,RecoveryBuilder
    state=Path(state).resolve();model,native=model_configuration(config),native_configuration(config)
    torch.set_num_threads(1);stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    selector,params=full_selection(args,'diagnosis IS NULL' if phase=='diagnose' else "status IN ('compatible_candidate','projected_candidate','retryable')")
    if phase!='diagnose' and args.profile:selector+=' AND profile=?';params.append(args.profile)
    with (state/'retry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);db=full_db(state);writers={}
        try:
            producers={p:full_identity(config,p) for p in ['compatible','projected']}
            for p,producer in producers.items():pin(db,'producer_'+p,producer)
            cache=Path(json.loads(db.execute("SELECT value FROM meta WHERE key='cache_root'").fetchone()[0]))
            rows=list(db.execute('SELECT * FROM tasks WHERE '+selector+' ORDER BY substr(route,instr(route,\'_Town\')),route,frame',params))
            # Keep Town maps warm across scenario names; no raw dataset rescan.
            import re
            rows.sort(key=lambda r:(re.search(r'_(Town\d+(?:HD)?)_',r['route']).group(1),r['route'],r['frame']))
            if args.limit is not None:rows=rows[:args.limit]
            providers={p:RecoveryProvider(config,p) for p in ['compatible','projected']}
            current_route=None;count=0;started=time.monotonic()
            with full_source_scope():
                for task in rows:
                    if stopped or (state/'stop.request').exists():break
                    row=json.loads(task['baseline_row']);sid=row['sample_id']
                    if current_route!=row['route']:
                        reader=MemoReader(jobs.AnnotationIndex.from_dataset_root(config['dataset_root'],row['route']));current_route=row['route']
                    record=None
                    try:
                        if phase=='diagnose':
                            record,status=full_diagnose_one(config,row,reader,native,model,providers['compatible'])
                            historical=task['status']=='historical_complete'
                            if historical:status='historical_complete'
                            profile='compatible' if status=='compatible_candidate' else 'projected' if status=='projected_candidate' else None
                            db.execute('UPDATE tasks SET diagnosis=?,status=?,profile=?,updated=? WHERE sample_id=?',
                                (json.dumps(record),status,profile,time.time(),sid))
                        else:
                            record=json.loads(task['diagnosis']);profile=task['profile'];provider=providers[profile]
                            rule_id=digest(producers[profile])
                            if phase=='apply' and not args.pilot:
                                gate=state/('certified_'+profile+'.json')
                                if not gate.exists() or json.loads(gate.read_text()).get('producer_sha256')!=rule_id:
                                    raise StoragePause('Run real paired geometry/forward pilot certification before full '+profile+' apply')
                            ref=reader.index.ref(row['frame']);builder=RecoveryBuilder(reader,native,model,provider,policy=profile)
                            sources,mapped,future=core.source_identity(reader,ref,native,provider,Path(config['dataset_root']).resolve())
                            if digest(sources)!=digest(record['sources']) or digest(mapped)!=digest(record['map']) or digest(core.reference_pose(reader.read(ref),native))!=digest(record['reference_pose']):
                                raise StoragePause('Source/reference changed since independent diagnosis: '+sid)
                            checks,tensors=component_checks(builder,ref)
                            if not checks['full_training_eligible']:
                                record['secondary_errors']=checks;status='blocked_secondary'
                            else:
                                # All exact target poses must invert into the true future world actor pose.
                                from tools.b2d_adapter.coordinate_transform import target_to_actor_world,wrap_angle
                                residual=[];heading=[]
                                origin=reader.read(ref).actor_pose
                                for value,fid in zip(tensors['trajectory'].numpy(),future):
                                    restored=target_to_actor_world(value,origin,(native.geometry.axle_dx_m,native.geometry.axle_dy_m))
                                    true=reader.read(reader.index.ref(fid)).actor_pose
                                    residual.append(math.hypot(restored.x-true.x,restored.y-true.y));heading.append(abs(wrap_angle(restored.yaw-true.yaw)))
                                if max(residual)>1e-4 or max(heading)>1e-5:raise StoragePause('Real trajectory world inverse mismatch')
                                record['validation']={'actor_inverse_max_m':max(residual),'heading_inverse_max_rad':max(heading),
                                    'head_checks':checks,'validated_profile':profile}
                                status=task['status']
                                if phase=='apply':
                                    if profile not in writers:
                                        spec=dict(config['target_root'],path=str(cache/profile),root_id='full_recovery_'+profile,budget_gib=4)
                                        core.storage_check(spec,True);Path(spec['path']).mkdir(parents=True,exist_ok=True)
                                        writer=jobs.open_index(state/profile/'index.sqlite');pin(writer,'producer',producers[profile]);pin(writer,'dataset_id',config['dataset_id']);pin(writer,'target_root',spec)
                                        marker=Path(spec['path'])/'target_root.json';identity={'schema':core.SCHEMA,'state_id':str(state/profile),'root_id':spec['root_id']}
                                        if marker.exists():
                                            if json.loads(marker.read_text())!=identity:raise StoragePause('Recovery output root ownership mismatch')
                                        else:
                                            if any(Path(spec['path']).iterdir()):raise StoragePause('Unmarked nonempty recovery output root')
                                            durable_json(marker,identity)
                                        used=writer.execute("SELECT COALESCE(SUM(bytes),0) FROM samples WHERE state='complete'").fetchone()[0]
                                        writers[profile]=(writer,spec,used)
                                    writer,spec,used=writers[profile]
                                    cached=writer.execute('SELECT * FROM samples WHERE sample_id=?',(sid,)).fetchone()
                                    if cached:
                                        payload=load_target_row(dict(cached),spec,rule_id)
                                    else:
                                        output=dict(config,target_root=spec)
                                        with codec_scope(write=True):cached,pixels,size=jobs.write_target(output,row,producers[profile],reader,builder,native,provider,used)
                                        payload=load_target_row(dict(cached),spec,rule_id)
                                        if any(not torch.equal(v,payload['targets'][k]) for k,v in tensors.items()):raise StoragePause('Head/full cache readback differs')
                                        cols=list(cached);writer.execute('INSERT INTO samples('+','.join(cols)+') VALUES('+','.join('?' for _ in cols)+')',list(cached.values()))
                                        writer.execute('INSERT OR REPLACE INTO bev_counts VALUES(?,?)',(sid,json.dumps(pixels)));writer.commit()
                                        writers[profile]=(writer,spec,used+size)
                                    status='complete_'+profile
                            if digest(core.source_identity(reader,ref,native,provider,Path(config['dataset_root']).resolve()))!=digest((sources,mapped,future)):
                                raise StoragePause('Source changed during recovery')
                            db.execute('UPDATE tasks SET diagnosis=?,status=?,attempts=attempts+1,last_rule=?,updated=? WHERE sample_id=?',
                                (json.dumps(record),status,rule_id,time.time(),sid))
                            db.execute('INSERT OR REPLACE INTO attempts(sample_id,phase,rule_id,outcome,record,created) VALUES(?,?,?,?,?,?)',
                                (sid,phase,rule_id,status,json.dumps(record.get('validation',record.get('secondary_errors',{}))),time.time()))
                        db.commit();count+=1
                        with (state/'operations.jsonl').open('a') as journal:journal.write(json.dumps({'sample_id':sid,'phase':phase,'state':status,'time':time.time()})+'\n')
                        if count%50==0 or args.limit is not None:
                            stats=dict(db.execute('SELECT status,COUNT(*) FROM tasks GROUP BY status'))
                            durable_json(state/'progress.json',{'pid':os.getpid(),'phase':phase,'processed':count,'seconds':time.monotonic()-started,'states':stats,'last_sample_id':sid})
                            print(json.dumps({'phase':phase,'processed':count,'states':stats}),flush=True)
                    except (AdapterError,OSError) as exc:
                        classification=jobs.failure_state(exc)
                        if classification!='failed_retryable':raise
                        db.execute("UPDATE tasks SET status='retryable',updated=? WHERE sample_id=?",(time.time(),sid));db.commit()
                        with (state/'operations.jsonl').open('a') as journal:journal.write(json.dumps({'sample_id':sid,'phase':phase,'state':'retryable','error':str(exc),'time':time.time()})+'\n')
            stats=dict(db.execute('SELECT status,COUNT(*) FROM tasks GROUP BY status'))
            durable_json(state/'progress.json',{'pid':os.getpid(),'phase':phase,'processed':count,'finished':not stopped,
                'seconds':time.monotonic()-started,'states':stats});return stats
        finally:
            for writer,_,_ in writers.values():writer.close()
            db.close()


def full_export(config,state,output):
    import csv
    from collections import Counter
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    db=full_db(state)
    try:
        rows=list(db.execute('SELECT * FROM tasks ORDER BY route,frame'))
        combinations=Counter();by_original={};profiles=Counter();diagnosed=[]
        for row in rows:
            record=json.loads(row['diagnosis']) if row['diagnosis'] else {'sample_id':row['sample_id'],'source_errors':[{'code':'pending_diagnosis','detail':'not yet examined'}]}
            record.update(final_state=row['status'],original_error_code=row['original_code'],attempts=row['attempts'])
            codes=sorted({e['code'] for key in ['trajectory_errors','detection_errors','bev_errors','source_errors'] for e in record.get(key,[])})
            combinations[tuple(codes)]+=1;profiles[row['status']]+=1
            by_original.setdefault(row['original_code'],Counter())[row['status']]+=1
            diagnosed.append(record)
        durable_json(output/'recovery_full_diagnosis.json',{'version':FULL_VERSION,'count':len(rows),'states':dict(profiles),'samples':diagnosed})
        with (output/'recovery_candidate_manifest.csv').open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['sample_id','route','frame','original_code','status','profile','attempts','trajectory_errors','detection_errors','bev_errors','source_errors'])
            for row,rec in zip(rows,diagnosed):writer.writerow([row[k] for k in ['sample_id','route','frame','original_code','status','profile','attempts']]+[','.join(sorted({e['code'] for e in rec.get(key,[])})) for key in ['trajectory_errors','detection_errors','bev_errors','source_errors']])
        lines=['# 全量多重错误分布','','每帧只贡献一行最终状态；下表错误组合按完整集合互斥计数。','',
            '| 完整错误集合 | 帧数 |','|---|---:|']
        for combo,n in combinations.most_common():lines.append('| '+(' + '.join(combo) or 'none')+' | '+str(n)+' |')
        lines+=['','## 最终状态','', '| 状态 | 帧数 |','|---|---:|']+[f'| {k} | {v} |' for k,v in profiles.items()]
        (output/'multi_error_distribution.md').write_text('\n'.join(lines)+'\n')
        summary={'original_nonfuture':len(rows),'historical_complete':profiles['historical_complete'],
            'new_complete':profiles['complete_compatible']+profiles['complete_projected'],
            'states':dict(profiles),'by_original':{k:dict(v) for k,v in by_original.items()},'diagnosed':sum(r['diagnosis'] is not None for r in rows)}
        durable_json(output/'summary.json',summary);return summary
    finally:db.close()


def readonly(path):
    db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('BEGIN')
    return db


def component_checks(builder, ref):
    """Each head runs independently; no fabricated tensors on failure.

    Capture only known data errors. Device/storage faults propagate to the
    caller rather than being disguised as successful partial supervision.
    """
    checks, tensors = {}, {}
    calls = {
        'trajectory': lambda: builder.trajectory_builder.build_targets(ref),
        'detection': lambda: builder.build_agent_targets(builder.reader.read(ref)),
        'bev': lambda: builder.bev_provider.build_targets(ref, builder.reader, builder.config, builder.model_config),
    }
    for name, call in calls.items():
        try:
            result = call()
            if name == 'bev': builder._validate_bev(result, ref)
            checks[name] = {'state': 'valid', 'keys': list(result.tensors), 'provenance': result.provenance,
                            'validation': result.validation}
            tensors.update(result.tensors)
        except AdapterError as exc:
            jobs.failure_state(exc)  # Propagate storage/dependency faults as a safe pause.
            checks[name] = {'state': 'rejected', 'error_code': exc.code, 'error': str(exc)}
    complete = all(v['state'] == 'valid' for v in checks.values())
    if complete: core.validate_targets(tensors)
    return {'components': checks, 'full_training_eligible': complete}, tensors


def baseline_rows(config):
    state = ROOT / config['target_state_dir']
    with readonly(state / 'index.sqlite') as db:
        producer = json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
        rows = [dict(r) for r in db.execute("SELECT * FROM samples WHERE state IN ('ineligible','failed_retryable') AND COALESCE(error_code,'') != 'insufficient_future' ORDER BY route,frame")]
        stats = dict(db.execute('SELECT state,COUNT(*) FROM samples GROUP BY state'))
    return producer, rows, stats


def recovery_identity(config):
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration
    base = core.producer_identity(config, model_configuration(config), native_configuration(config))
    return dict(base, incremental_recovery={
        'rule': RULE, 'implementation_sha256': file_digest(Path(__file__)),
        'rule_module_sha256': file_digest(Path(__file__).with_name('target_recovery.py')),
        'baseline_producer_sha256': digest(base), 'missing_future_recovery': False,
        'partial_targets_trainable': False, 'ignore_mask': False})


def prepare(config, state):
    state = Path(state).resolve()
    if state == (ROOT / config['target_state_dir']).resolve():
        raise StoragePause('Recovery state must differ from baseline')
    producer, rows, stats = baseline_rows(config)
    current = recovery_identity(config)
    if current['incremental_recovery']['baseline_producer_sha256'] != digest(producer):
        raise StoragePause('Current implementation differs from pinned baseline')
    state.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (state / 'retry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = jobs.open_index(state / 'index.sqlite')
        try:
            pin(db, 'producer', current); pin(db, 'dataset_id', config['dataset_id'])
            pin(db, 'source_state', str((ROOT / config['target_state_dir']).resolve()))
            pin(db, 'candidate_rows_sha256', digest(rows))
            db.execute('CREATE TABLE IF NOT EXISTS recovery_audit(sample_id TEXT PRIMARY KEY,record TEXT NOT NULL)')
            for row in rows:
                candidate = dict(row, state='pending' if row['error_code'] in EXECUTABLE else 'design_required',
                                 relative_path=None, sha256=None, bytes=None)
                cols = list(candidate)
                db.execute('INSERT OR IGNORE INTO samples(' + ','.join(cols) + ') VALUES(' + ','.join('?' for _ in cols) + ')', list(candidate.values()))
            db.commit()
            counts = dict(db.execute('SELECT state,COUNT(*) FROM samples GROUP BY state'))
            durable_json(state / 'plan.json', {'schema': 'b2d-target-incremental-retry-audit-v1',
                'baseline_stats': stats, 'baseline_producer_sha256': digest(producer),
                'recovery_producer_sha256': digest(current), 'candidates': len(rows),
                'queue_counts': counts, 'executable_rule': RULE,
                'excluded': ['insufficient_future', 'baseline complete'],
                'default_mode': 'validate; no tensor writes', 'partial_targets_trainable': False})
            return counts
        finally: db.close()


def validate(config, state, limit=32, apply=False, sample_ids=None):
    """Bounded retries, resumable pending-only cursor, append-only successes.

    `apply` writes complete four-task payloads only in this state's private
    cache. Validation never changes candidate state so the same audit can be
    promoted explicitly later. All attempts persist individual head results.
    """
    import fcntl
    import torch
    from tools.b2d_adapter import FrameReader
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration, bev_provider
    state = Path(state).resolve(); producer = recovery_identity(config)
    model, native = model_configuration(config), native_configuration(config)
    torch.set_num_threads(1)
    if state == (ROOT / config['target_state_dir']).resolve(): raise StoragePause('Baseline state forbidden')
    cache_root = state / 'cache'
    if cache_root == Path(config['target_root']['path']).resolve(): raise StoragePause('Baseline root forbidden')
    output_config = dict(config, target_root=dict(config['target_root'], path=str(cache_root),
        root_id='isolated_retry_audit', budget_gib=1, minimum_free_gib=0))
    # This bounded workspace validation does not inherit the data volume UUID.
    # Use the actual volume identity and validate it through the original IO gate.
    if apply:
        import subprocess
        mount = json.loads(subprocess.check_output(['findmnt','-J','-T',str(state),'-o','UUID,FSTYPE'], text=True))['filesystems'][-1]
        output_config['target_root'].update(uuid=mount['uuid'], fstype=mount['fstype'])
    provider = bev_provider(dict(config, bev_cache_dir=None)); results = []
    with (state / 'retry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = jobs.open_index(state / 'index.sqlite')
        try:
            pin(db, 'producer', producer)
            rows = list(db.execute("SELECT * FROM samples WHERE state='pending' ORDER BY route,frame"))
            if sample_ids is not None: rows = [r for r in rows if r['sample_id'] in sample_ids]
            if apply:
                core.storage_check(output_config['target_root'], True)
                pin(db, 'target_root', output_config['target_root'])
            current_route = None
            used = sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in cache_root.rglob('*') if p.is_file())
            for row in rows[:limit]:
                if (state / 'stop.request').exists(): break
                row = dict(row)
                if current_route != row['route']:
                    reader = FrameReader(jobs.AnnotationIndex.from_dataset_root(config['dataset_root'], row['route']))
                    builder = IrrelevantActorRecoveryBuilder(reader, native, model, provider)
                    current_route = row['route']
                ref = reader.index.ref(row['frame'])
                # Validate pinned source SHA, map and coordinate origin even in dry run.
                sources, mapped, future = core.source_identity(reader, ref, native, provider, Path(config['dataset_root']).resolve())
                reference = core.reference_pose(reader.read(ref), native)
                for key,value in [('sources_json',sources),('map_json',mapped),('reference_json',reference)]:
                    if row[key] and digest(json.loads(row[key])) != digest(value):
                        raise StoragePause('Pinned baseline source/reference changed: ' + row['sample_id'])
                result, tensors = component_checks(builder, ref)
                if digest(core.source_identity(reader, ref, native, provider, Path(config['dataset_root']).resolve())) != digest((sources,mapped,future)):
                    raise StoragePause('Sources changed during independent validation')
                result.update(sample_id=row['sample_id'],route=row['route'],frame=row['frame'],
                    original_error_code=row['error_code'],mode='apply' if apply else 'validate',
                    sources_sha256=digest(sources),reference_pose=reference,future_frame_ids=future,
                    checked=time.time())
                if apply and result['full_training_eligible']:
                    candidate = dict(row, state='pending',error_code=None,error=None)
                    with codec_scope(write=True):
                        cached, pixels, size = jobs.write_target(output_config,candidate,producer,reader,builder,native,provider,used)
                    payload = load_target_row(cached,output_config['target_root'],digest(producer))
                    if any(not torch.equal(tensors[k],payload['targets'][k]) for k in tensors):
                        raise StoragePause('Independent head validation differs from full TargetBuilder output')
                    cols = list(cached)
                    db.execute('INSERT OR REPLACE INTO samples(' + ','.join(cols) + ') VALUES(' + ','.join('?' for _ in cols) + ')',list(cached.values()))
                    used += size
                    result['saved_four_task_payload'] = True
                elif apply:
                    # Keep rejection reason and head evidence; explicit new rule required to retry.
                    db.execute("UPDATE samples SET state='still_rejected',updated=? WHERE sample_id=?",(time.time(),row['sample_id']))
                db.execute('INSERT OR REPLACE INTO recovery_audit VALUES(?,?)',(row['sample_id'],json.dumps(result)))
                db.commit(); results.append(result)
                with (state / 'attempts.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
                print(json.dumps({'frame':row['frame'],'eligible':result['full_training_eligible'],'mode':result['mode']}),flush=True)
            counts = dict(db.execute('SELECT state,COUNT(*) FROM samples GROUP BY state'))
            durable_json(state / 'retry_report.json', {'producer_sha256':digest(producer),'counts':counts,
                'attempted':len(results),'full_validated':sum(r['full_training_eligible'] for r in results),
                'baseline_modified':False,'cache_root':str(cache_root),'applied':apply,'results':results})
            return results
        finally: db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','diagnose','validate','apply','report'])
    p.add_argument('--config',type=Path,default=ROOT/'tools/training/diffusiondrive_v1/targets_full.json')
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--limit',type=int,default=32)
    p.add_argument('--full',action='store_true',help='New full recovery state; legacy completed state is never reused')
    p.add_argument('--all',action='store_true',help='Process the entire selected new queue')
    p.add_argument('--profile',choices=['compatible','projected'])
    p.add_argument('--town');p.add_argument('--route');p.add_argument('--error')
    p.add_argument('--pilot',action='store_true',help='Explicit bounded <=8 real pilot before profile certification')
    p.add_argument('--cache-root',type=Path,default=Path('/data/datasets/diffusiondrive_v1_targets_recovery_20261017'))
    p.add_argument('--output',type=Path,default=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/full_target_recovery')
    args = p.parse_args()
    if args.limit < 1: p.error('--limit must be positive')
    if args.pilot and (args.all or args.limit>8):p.error('Pilot must be explicitly limited to <=8 samples')
    if args.all:args.limit=None
    cfg = json.loads(args.config.read_text())
    if args.full:
        if args.command=='plan':result=full_plan(cfg,args.state,args.cache_root)
        elif args.command=='report':result=full_export(cfg,args.state,args.output)
        else:result=full_run(cfg,args.state,args,args.command)
        print(json.dumps(result,ensure_ascii=False));return
    if args.command not in ('plan','validate','apply'):p.error('diagnose/report require --full')
    if args.command == 'plan': print(json.dumps(prepare(cfg,args.state)))
    else: validate(cfg,args.state,args.limit,apply=args.command=='apply')


if __name__ == '__main__': main()
