"""Lossless disk codec and crash-recoverable conversion of full Targets caches.

Numeric producers/builders stay frozen. Only BEV storage changes; every reader
returns the original int64 tensor before existing contract/identity validation.
"""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time

import torch

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1 import target_cache as core
from tools.training.diffusiondrive_v1.features_cache import (
    StoragePause, GIB, digest, durable_json, file_digest, safe_path, directory_sync, pin)

CODEC = 'b2d-v1-targets-bev-uint8-v1'
RAW_LOAD, RAW_SAVE = torch.load, torch.save
CODEC_LOCK = threading.RLock()


def encode(payload):
    if not isinstance(payload, dict) or payload.get('schema') != core.SCHEMA:
        return payload
    core.validate_targets(payload['targets'])
    result = dict(payload, storage_encoding=CODEC)
    result['targets'] = dict(payload['targets'], bev_semantic_map=payload['targets']['bev_semantic_map'].to(torch.uint8))
    return result


def decode(payload):
    if not isinstance(payload, dict) or payload.get('schema') != core.SCHEMA:
        return payload
    encoding = payload.get('storage_encoding')
    if encoding is None: return payload
    if encoding != CODEC:
        raise AdapterError('invalid_targets_cache', 'Unsupported target storage encoding')
    bev = payload['targets']['bev_semantic_map']
    if (not isinstance(bev, torch.Tensor) or bev.dtype != torch.uint8 or bev.device.type != 'cpu'
        or tuple(bev.shape) != (128,256) or not (bev <= 6).all()):
        raise AdapterError('invalid_targets_cache', 'Invalid compact BEV values/shape/dtype')
    result = dict(payload)
    result['targets'] = dict(payload['targets'], bev_semantic_map=bev.to(torch.int64))
    core.validate_targets(result['targets'])
    return result


@contextmanager
def codec_scope(write=False):
    """Schema-specific serialization adapter in one CPU worker/reader process.

    Other torch payloads pass through. Scopes restore prior functions even on
    exceptions and serialize cache-reader scopes with a reentrant lock.
    """
    with CODEC_LOCK:
        old_load, old_save = torch.load, torch.save
        torch.load = lambda *a, **kw: decode(RAW_LOAD(*a, **kw))
        if write: torch.save = lambda value, *a, **kw: RAW_SAVE(encode(value), *a, **kw)
        try: yield
        finally: torch.load, torch.save = old_load, old_save


def load_target_row(row, spec, producer_sha):
    with codec_scope(): return core.load_target_row(row, spec, producer_sha)


def assert_equal(original, restored):
    if any(not torch.equal(original['targets'][k], restored['targets'][k]) for k in original['targets']):
        raise StoragePause('Lossless conversion changed target values')
    a = {k:v for k,v in original.items() if k not in ('targets','storage_encoding')}
    b = {k:v for k,v in restored.items() if k not in ('targets','storage_encoding')}
    if digest(a) != digest(b): raise StoragePause('Conversion changed identity/provenance')


def storage_preflight(config, state):
    state = Path(state); marker = Path(config['target_root']['path']) / 'target_storage_encoding.json'
    requested = config.get('target_storage_encoding')
    if requested not in (None, CODEC): raise StoragePause('Unknown target storage codec')
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['codec'] != requested or record['state_dir'] != str(state.resolve()):
            raise StoragePause('Target storage codec/config owner mismatch')
    if (state / 'index.sqlite').exists():
        db = sqlite3.connect('file:' + str((state/'index.sqlite').resolve()) + '?mode=ro',uri=True)
        try:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='codec_migrations'").fetchone():
                if db.execute("SELECT COUNT(*) FROM codec_migrations WHERE phase!='done'").fetchone()[0]:
                    raise StoragePause('Finish target_storage compact recovery before starting the generator')
        finally: db.close()


def _path_row(row, relative, sha, size):
    return dict(row,relative_path=relative,sha256=sha,bytes=size)


def _finish(db, row, entry, spec, producer_sha, fault_hook=None):
    """Old bytes remain linked until verified new bytes AND SQLite commit succeed."""
    dest = safe_path(spec,row['relative_path'])
    backup = safe_path(spec,entry['backup']); temp = safe_path(spec,entry['temp'])
    current_sha = file_digest(dest)
    if current_sha not in (entry['old_sha'],entry['new_sha']):
        raise StoragePause('Cache differs from both journaled versions: '+row['sample_id'])
    old_path = entry['backup'] if backup.exists() else row['relative_path']
    if current_sha == entry['new_sha'] and not backup.exists():
        raise StoragePause('Missing original backup before conversion commit')
    original = load_target_row(_path_row(row,old_path,entry['old_sha'],entry['old_bytes']),spec,producer_sha)
    if current_sha == entry['old_sha']:
        if not temp.exists():
            stream=io.BytesIO();RAW_SAVE(encode(original),stream);data=stream.getvalue()
            if hashlib.sha256(data).hexdigest()!=entry['new_sha']: raise StoragePause('Recreated conversion bytes differ')
            with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        restored=load_target_row(_path_row(row,entry['temp'],entry['new_sha'],entry['new_bytes']),spec,producer_sha)
        assert_equal(original,restored)
        if not backup.exists():os.link(dest,backup);directory_sync(dest.parent)
        core.storage_check(spec,True);os.replace(temp,dest);directory_sync(dest.parent)
        if fault_hook:fault_hook('renamed')
    restored=load_target_row(_path_row(row,row['relative_path'],entry['new_sha'],entry['new_bytes']),spec,producer_sha)
    assert_equal(original,restored)
    db.execute('UPDATE samples SET sha256=?,bytes=? WHERE sample_id=?',(entry['new_sha'],entry['new_bytes'],row['sample_id']))
    db.execute("UPDATE codec_migrations SET phase='done' WHERE sample_id=?",(row['sample_id'],));db.commit()
    if fault_hook:fault_hook('committed')
    core.storage_check(spec,True);backup.unlink(missing_ok=True);temp.unlink(missing_ok=True);directory_sync(dest.parent)


def _cleanup_done(row, entry, spec, producer_sha):
    backup=safe_path(spec,entry['backup']);temp=safe_path(spec,entry['temp'])
    if backup.exists():
        restored=load_target_row(row,spec,producer_sha)
        original=load_target_row(_path_row(row,entry['backup'],entry['old_sha'],entry['old_bytes']),spec,producer_sha)
        assert_equal(original,restored);core.storage_check(spec,True);backup.unlink();directory_sync(backup.parent)
    if temp.exists():core.storage_check(spec,True);temp.unlink();directory_sync(temp.parent)


def compact(config, state, limit=None, fault_hook=None):
    """Pause writer, then convert complete samples individually with a WAL journal."""
    import fcntl
    import signal
    from tools.training.diffusiondrive_v1.feature_service import alive,process_identity
    from tools.training.diffusiondrive_v1.runtime import native_configuration,model_configuration
    state=Path(state);state.mkdir(parents=True,exist_ok=True);spec=config['target_root'];db=None
    torch.set_num_threads(config.get('threads',1))
    if alive(state):
        current=json.loads((state/'pid.json').read_text())
        if current['pid']!=os.getpid():raise StoragePause('Stop Targets before compacting')
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.monotonic();status={'pid':os.getpid(),'state':'compacting','converted':0,'saved_bytes':0}
    def update():
        status.update(updated=time.time(),seconds=time.monotonic()-started)
        durable_json(state/'compaction_status.json',status)
    with (state/'launch.lock').open('a') as launch,(state/'writer.lock').open('a') as lock:
        fcntl.flock(launch,fcntl.LOCK_EX);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            core.storage_check(spec,True)
            db=core.open_index(state/'index.sqlite')
            producer=json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0]);sha=digest(producer)
            if sha!=digest(core.producer_identity(config,model_configuration(config),native_configuration(config))):
                raise StoragePause('Numeric target producer changed')
            db.execute('''CREATE TABLE IF NOT EXISTS codec_migrations(sample_id TEXT PRIMARY KEY,
              old_sha TEXT,old_bytes INTEGER,new_sha TEXT,new_bytes INTEGER,temp TEXT,backup TEXT,phase TEXT)''');db.commit()
            pin(db,'storage_codec',{'version':CODEC,'decoded_contract':'original int64 (128,256) 0..6'})
            marker=Path(spec['path'])/'target_storage_encoding.json'
            identity={'codec':CODEC,'state_dir':str(state.resolve()),'producer_sha256':sha}
            if marker.exists() and json.loads(marker.read_text())!=identity:raise StoragePause('Codec marker owner differs')
            if not marker.exists():durable_json(marker,identity)
            (state/'stop.request').unlink(missing_ok=True)
            durable_json(state/'pid.json',process_identity(os.getpid()))
            update()
            # Repair any rename/SQLite boundary interrupted by a previous run.
            for raw_entry in db.execute("SELECT * FROM codec_migrations WHERE phase!='done'").fetchall():
                entry=dict(raw_entry);row=dict(db.execute('SELECT * FROM samples WHERE sample_id=?',(entry['sample_id'],)).fetchone())
                _finish(db,row,entry,spec,sha)
            for raw_entry in db.execute("SELECT * FROM codec_migrations WHERE phase='done'"):
                entry=dict(raw_entry);row=dict(db.execute('SELECT * FROM samples WHERE sample_id=?',(entry['sample_id'],)).fetchone())
                _cleanup_done(row,entry,spec,sha)
            after=''
            while True:
                rows=db.execute("SELECT * FROM samples WHERE state='complete' AND sample_id>? AND sample_id NOT IN (SELECT sample_id FROM codec_migrations WHERE phase='done') ORDER BY sample_id LIMIT 256",(after,)).fetchall()
                if not rows:status['state']='complete';break
                for raw in rows:
                    if stopped or (state/'stop.request').exists():status['state']='stopped';break
                    if limit is not None and status['converted']>=limit:status['state']='batch_complete';break
                    row=dict(raw);after=row['sample_id'];status['last_sample_id']=after
                    original=load_target_row(row,spec,sha)
                    if original.get('storage_encoding')==CODEC:
                        # New generation already uses this codec; no re-encoding.
                        continue
                    stream=io.BytesIO();RAW_SAVE(encode(original),stream);data=stream.getvalue()
                    if len(data)>=row['bytes']:raise StoragePause('Codec failed to reduce sample size')
                    free=core.storage_check(spec,True)
                    if free-len(data)-65536<spec['minimum_free_gib']*GIB:raise StoragePause('Insufficient conversion reserve')
                    entry={'sample_id':after,'old_sha':row['sha256'],'old_bytes':row['bytes'],
                           'new_sha':hashlib.sha256(data).hexdigest(),'new_bytes':len(data),
                           'temp':row['relative_path']+'.uint8.partial','backup':row['relative_path']+'.legacy-backup','phase':'prepared'}
                    cols=list(entry);db.execute('INSERT INTO codec_migrations('+','.join(cols)+') VALUES('+','.join('?' for _ in cols)+')',list(entry.values()));db.commit()
                    temp=safe_path(spec,entry['temp'])
                    with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                    if fault_hook:fault_hook('prepared')
                    _finish(db,row,entry,spec,sha,fault_hook)
                    status['converted']+=1;status['saved_bytes']+=row['bytes']-len(data)
                    if status['converted']%100==0:update();print(json.dumps(status),flush=True)
                if status['state']!='compacting':break
            totals=db.execute("SELECT COUNT(*),COALESCE(SUM(old_bytes-new_bytes),0) FROM codec_migrations WHERE phase='done'").fetchone()
            core.storage_check(spec,True)
            status.update(total_converted=totals[0],total_saved_bytes=totals[1])
        except Exception as exc:
            status.update(state='paused',error=str(exc));import traceback;traceback.print_exc()
        finally:
            update()
            if db is not None:db.close()
    return status


def main():
    sys.path.insert(0,str(core.ROOT/'repositories/DiffusionDrive'))
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['compact','status'])
    p.add_argument('--config',type=Path,default=core.ROOT/'tools/training/diffusiondrive_v1/targets_full.json');p.add_argument('--limit',type=int)
    p.add_argument('--resume',action='store_true',help='Export fresh manifests and exec the original worker after successful full conversion')
    args=p.parse_args();config=json.loads(args.config.read_text());state=core.ROOT/config['target_state_dir']
    if args.action=='status':print((state/'compaction_status.json').read_text());return 0
    if config.get('target_storage_encoding')!=CODEC:raise StoragePause('Enable explicit target_storage_encoding in config first')
    result=compact(config,state,args.limit);print(json.dumps(result),flush=True)
    if args.resume and result['state']=='complete':
        import subprocess
        # Refuse continuation if current real capacity cannot cover even this
        # conservative remaining-size estimate plus the existing safety floor.
        db=sqlite3.connect('file:'+str((state/'index.sqlite').resolve())+'?mode=ro',uri=True)
        pending=db.execute("SELECT COUNT(*) FROM samples WHERE state='pending'").fetchone()[0]
        maximum=db.execute("SELECT COALESCE(MAX(bytes),65536) FROM samples WHERE state='complete'").fetchone()[0];db.close()
        forecast=pending*((maximum+4095)//4096*4096)+2*GIB
        free=core.storage_check(config['target_root'],True)
        if free-forecast<config['target_root']['minimum_free_gib']*GIB:
            raise StoragePause('Compact conversion finished, but continuation capacity forecast is insufficient')
        subprocess.run([config['python'],'-m','tools.training.diffusiondrive_v1.target_service','export',
                        '--config',str(args.config.resolve())],cwd=core.ROOT,check=True)
        command=[config['python'],'-u','-m','tools.training.diffusiondrive_v1.target_service','run',
                 '--config',str(args.config.resolve())]
        core.storage_check(config['target_root'],True)
        durable_json(state/'launch.json',{'pid':os.getpid(),'command':command,'started':time.time(),
                     'log':str(state/'worker.log'),'after_compaction':True})
        print(json.dumps({'event':'compact_complete_resume','forecast_gib':forecast/GIB,'free_gib':free/GIB}),flush=True)
        # Same detached PID/session; no duplicate writer or launcher race.
        os.execv(config['python'],command)
    return 2 if result['state']=='paused' else 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (StoragePause,OSError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
