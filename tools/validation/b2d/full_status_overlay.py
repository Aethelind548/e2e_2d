"""Version 2 full frozen-pair Status overlay; read-only mmap extraction of A0.

The archive's whole-file SHA is NOT revalidated here: pair/index identities,
embedded metadata, size, and stable stat are checked. Training reader must hash
actual payload bytes before using them. Offline reconstruction never certifies
navigation and this tool never admits a row for training.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.b2d_adapter.experimental_status_builder import build_status, COMMAND_ORDER

VERSION = 'full-status-overlay-v2.0.0'
SCHEMA = 'status-overlay-v2'
STATUS_ORDER = list(COMMAND_ORDER) + ['vx', 'vy', 'ax', 'ay']
A0_EVIDENCE = 'FROZEN_INDEX_MATCH_MMAP_EMBEDDED_IDENTITY_STABLE_STAT_FULL_HASH_DEFERRED'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    return db


def file_stat(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def extract_status(path, feature, pair, producer, dataset_id):
    """Extract actual old float32[8], without touching mapped camera/lidar pages."""
    import torch
    before = file_stat(path)
    if before[2] != feature['bytes']:
        raise ValueError('feature_size_mismatch')
    if (feature['sha256'] != pair['feature_sha'] or feature['state'] != 'complete'
            or feature['route'] != pair['route'] or feature['frame'] != pair['frame']
            or feature['sample_id'] != pair['sample_id']):
        raise ValueError('feature_index_pair_mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (payload.get('schema') != 'b2d-v1-features-only-v1'
            or payload.get('producer_sha256') != producer
            or payload.get('dataset_id') != dataset_id
            or any(payload.get(k) != pair[k] for k in ('sample_id', 'route', 'frame'))
            or digest(payload.get('sources')) != digest(json.loads(feature['source_json']))):
        raise ValueError('feature_embedded_identity_mismatch')
    status = payload['features']['status_feature']
    if (not isinstance(status, torch.Tensor) or status.dtype != torch.float32
            or tuple(status.shape) != (8,) or status.device.type != 'cpu'
            or not torch.isfinite(status).all()):
        raise ValueError('old_status_shape_dtype_nonfinite')
    value = status.numpy().copy()
    if not np.isin(value[:4], (0., 1.)).all() or value[:4].sum() != 1:
        raise ValueError('old_navigation_not_one_hot')
    if file_stat(path) != before:
        raise ValueError('feature_changed_during_mmap_read')
    return value


def compose(pair, preview, old_status, identities, motion_error=None):
    identity = {k: pair[k] for k in ('sample_id', 'route', 'frame', 'split')}
    navigation_valid = bool(preview and preview.get('preview_valid') is True)
    command = json.loads(preview['command4_json']) if navigation_valid else None
    result = build_status(command, None if old_status is None else old_status[4:],
                          'VALID' if navigation_valid else 'MISSING',
                          'VALID' if old_status is not None else 'MISSING',
                          'OFFLINE_EXPERT_RECONSTRUCTION', identity,
                          navigation_provenance='NOT_VERIFIED',
                          diagnostic_metadata={'motion_method': 'A0_FROZEN'})
    reasons = list(result.diagnostic_metadata['failure_reasons'])
    if preview is None:
        reasons.append('ROUTE_FRAME_NOT_IN_FULL_PREVIEW')
    elif not navigation_valid:
        reasons.append('PREVIEW_' + str(preview.get('preview_reason')))
    if motion_error:
        reasons.append('A0_READ_' + motion_error)
    row = result.as_dict()
    row.update(schema=SCHEMA, sample_id=pair['sample_id'],
               producer_version=VERSION, status_dtype='float32',
               navigation_order=list(COMMAND_ORDER), status_order=STATUS_ORDER,
               old_status8=None if old_status is None else old_status.tolist(),
               a0_motion4=None if old_status is None else old_status[4:].tolist(),
               motion_source='A0', motion_provenance=A0_EVIDENCE if old_status is not None else 'MISSING',
               preview_validity='MISSING' if preview is None else 'VALID' if navigation_valid else 'INVALID',
               navigation_source={'kind': 'OFFLINE_EXPERT_RECONSTRUCTION',
                                  'certification': 'NOT_VERIFIED',
                                  'reference_path_online_available': False,
                                  'evidence_sha256': None,
                                  'upstream_source': None if preview is None else preview.get('navigation_source')},
               source_feature_identity=identities,
               training_admitted=False, training_contract='NOT_ADMITTED_EXPERIMENT_ONLY',
               failure_reason=';'.join(reasons),
               physical_payload_hash_verification='NOT_RUN_DEFERRED_TO_STRICT_READER',
               preview_identity=None if preview is None else {'route': preview['route'], 'frame': preview['frame']})
    return row


def check_row(row, pair):
    checks = {}
    checks['sample_id_nonempty_unique_mapping'] = bool(row['sample_id']) and row['sample_id'] == pair['sample_id']
    checks['route_frame_split_identity'] = row['sample_identity'] == {k: pair[k] for k in ('sample_id','route','frame','split')}
    checks['navigation_order'] = row['navigation_order'] == list(COMMAND_ORDER)
    checks['required_metadata'] = all(k in row for k in ('preview_validity','navigation_source','status_validity','failure_reason','producer_version','training_admitted'))
    checks['no_near_fallback'] = row['fallback_source'] is None
    checks['uncertified_rejected_by_admission'] = row['training_admitted'] is False and row['navigation_source']['certification'] == 'NOT_VERIFIED'
    checks['source_sha_index_match'] = row['source_feature_identity']['feature_sha256'] == pair['feature_sha'] and row['source_feature_identity']['target_sha256'] == pair['target_sha']
    old, candidate = row['old_status8'], row['status8']
    checks['a0_old_payload_read'] = old is not None
    checks['candidate_or_explicit_invalid'] = (candidate is None) == (row['status_validity'] == 'INVALID')
    if candidate is not None:
        a = np.asarray(candidate)
        checks['float32_8_finite'] = a.shape == (8,) and np.isfinite(a).all() and np.array_equal(a, a.astype(np.float32).astype(np.float64))
        checks['command_one_hot'] = np.isin(a[:4], (0,1)).all() and a[:4].sum() == 1
        checks['motion4_exact_frozen_a0'] = old is not None and np.array_equal(a[4:].astype(np.float32), np.asarray(old[4:], dtype=np.float32))
    return {k:bool(v) for k,v in checks.items()}


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def build(args):
    import pyarrow.parquet as pq
    from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    pair_path = Path(manifest['pair_index'])
    if sha256(pair_path) != manifest['index_sha256']:
        raise ValueError('frozen_pair_index_sha_mismatch')
    navigation_manifest = args.preview_root / 'generation_manifest.json'
    if json.loads(navigation_manifest.read_text()).get('status') != 'DONE':
        raise ValueError('full_generation_not_done')
    output = args.output.resolve()
    protected = [args.preview_root.resolve(), pair_path.resolve().parent]
    protected += [Path(s['state']).resolve() for t in ('feature_sources','target_sources') for s in manifest[t].values()]
    if any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError('output_inside_protected_input')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'full_status_overlay_manifest.jsonl').exists():
        raise FileExistsError('overlay immutable; use separate versioned output')
    navigation = {}
    preview_file_shas = {}
    columns = ['route','frame','preview_valid','preview_reason','command4_json','navigation_source']
    for p in sorted((args.preview_root/'routes').glob('*/frame_route_preview.parquet')):
        before = file_stat(p)
        preview_file_shas[str(p)] = sha256(p)
        for row in pq.read_table(p, columns=columns).to_pylist():
            key = row['route'], row['frame']
            if key in navigation:
                raise ValueError('duplicate_full_route_frame:'+str(key))
            navigation[key] = row
        if file_stat(p) != before:
            raise ValueError('preview_changed_during_read')
    print(json.dumps({'phase':'navigation_loaded','frames':len(navigation)}), flush=True)
    source_indexes = {}; feature_dbs = {}; target_dbs = {}; roots = {}
    for kind in ('feature_sources','target_sources'):
        for name, s in manifest[kind].items():
            p = Path(s['state'])/'index.sqlite'
            db = readonly(p)
            meta = {r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
            if digest(meta['producer']) != s['producer_sha256'] or meta['dataset_id'] != manifest['dataset_id']:
                raise ValueError('source_index_producer_mismatch:'+name)
            source_indexes[str(p)] = {'sha256':sha256(p), 'stat':file_stat(p)}
            (feature_dbs if kind == 'feature_sources' else target_dbs)[name] = db
            if kind == 'feature_sources':
                roots[name] = effective_roots(manifest,s)
    counts = Counter(); split_counts = defaultdict(Counter); checks = defaultdict(Counter)
    seen = set(); paired_navigation = set(); failures = []
    overlay_path = output/'full_status_overlay_manifest.jsonl'
    start = time.monotonic()
    pair_db = readonly(pair_path)
    cursor = pair_db.execute('SELECT * FROM pairs ORDER BY feature_source,route,frame')
    from tools.training.diffusiondrive_v1.features_cache import sample_id, safe_path
    with ThreadPoolExecutor(max_workers=args.workers) as executor, overlay_path.open('x') as out, (output/'full_status_overlay_check_failures.jsonl').open('x') as failed_out:
        while True:
            batch = [dict(r) for r in cursor.fetchmany(512)]
            if not batch: break
            jobs = []
            for pair in batch:
                sid = pair['sample_id']
                if not sid or sid in seen or sid != sample_id(manifest['dataset_id'],pair['route'],pair['frame']):
                    raise ValueError('pair_sample_id_duplicate_or_identity_mismatch:'+str(sid))
                seen.add(sid)
                f = feature_dbs[pair['feature_source']].execute('SELECT * FROM samples WHERE sample_id=?',(sid,)).fetchone()
                t = target_dbs[pair['target_source']].execute('SELECT sample_id,route,frame,sha256,state,current_anno_sha256 FROM samples WHERE sample_id=?',(sid,)).fetchone()
                if t is None or t['state'] != 'complete' or t['sha256'] != pair['target_sha'] or t['route'] != pair['route'] or t['frame'] != pair['frame'] or t['current_anno_sha256'] != pair['current_anno_sha']:
                    raise ValueError('frozen_target_index_mismatch:'+sid)
                if f is None:
                    jobs.append((pair,None,'FEATURE_INDEX_MISSING'))
                else:
                    f = dict(f); spec = roots[pair['feature_source']][f['root_id']]
                    future = executor.submit(extract_status, safe_path(spec,f['relative_path']),f,pair,manifest['feature_sources'][pair['feature_source']]['producer_sha256'],manifest['dataset_id'])
                    jobs.append((pair,future,None))
            for pair,future,error in jobs:
                old = None
                if future is not None:
                    try: old = future.result()
                    except Exception as exc: error = type(exc).__name__+':'+str(exc)
                key = pair['route'],pair['frame']; preview = navigation.get(key)
                if preview is not None: paired_navigation.add(key)
                ids = {'manifest_sha256':manifest_sha,'pair_index_sha256':manifest['index_sha256'],
                       'feature_sha256':pair['feature_sha'],'target_sha256':pair['target_sha'],
                       'feature_producer_sha256':manifest['feature_sources'][pair['feature_source']]['producer_sha256'],
                       'target_producer_sha256':manifest['target_sources'][pair['target_source']]['producer_sha256']}
                row = compose(pair,preview,old,ids,error)
                out.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                for name,ok in check_row(row,pair).items():
                    checks[name]['passed' if ok else 'failed'] += 1
                    if not ok: failed_out.write(json.dumps({'sample_id':pair['sample_id'],'check':name,'reason':error or row['failure_reason']})+'\n')
                counts['rows']+=1; counts['preview_matched']+=preview is not None; counts['preview_valid']+=row['preview_validity']=='VALID'; counts['a0_read']+=old is not None
                counts['status8_candidates']+=row['status8'] is not None; counts['status_validity_'+row['status_validity']]+=1
                counts['legal_unknown']+=row['status8'] is not None and row['status8'][3]==1
                counts['navigation_changed']+=row['status8'] is not None and old is not None and row['status8'][:4]!=old[:4].tolist()
                counts['training_admitted']+=row['training_admitted']; split_counts[pair['split']]['rows']+=1
                split_counts[pair['split']]['candidates']+=row['status8'] is not None
            print(json.dumps({'phase':'overlay','rows':counts['rows'],'seconds':round(time.monotonic()-start,1)}),flush=True)
    pair_db.close()
    for db in list(feature_dbs.values())+list(target_dbs.values()): db.close()
    if sha256(pair_path)!=manifest['index_sha256'] or sha256(manifest_path)!=manifest_sha:
        raise ValueError('frozen_manifest_or_pair_index_changed')
    for p,identity in source_indexes.items():
        if file_stat(p)!=tuple(identity['stat']) or sha256(p)!=identity['sha256']:
            raise ValueError('source_index_changed:'+p)
    summary={'schema':VERSION,'manifest':str(manifest_path),'manifest_sha256':manifest_sha,
             'pair_index':str(pair_path),'pair_index_sha256':manifest['index_sha256'],
             'preview_root':str(args.preview_root.resolve()),'full_route_frame_count':len(navigation),
             'full_frames_outside_frozen_pairs':len(set(navigation)-paired_navigation),
             'counts':dict(counts),'splits':{k:dict(v) for k,v in split_counts.items()},
             'checks':{k:{'passed':v['passed'],'failed':v['failed']} for k,v in checks.items()},
             'mapping_rate':counts['preview_matched']/counts['rows'],
             'navigation_change_denominator':counts['status8_candidates'],
             'navigation_change_rate':counts['navigation_changed']/max(counts['status8_candidates'],1),
             'overlay_sha256':sha256(overlay_path),'source_indexes':source_indexes,
             'whole_physical_feature_target_hash_audit':'NOT_RUN; mandatory strict reader per loaded item',
             'motion_provenance':A0_EVIDENCE,'producer_sha256':sha256(__file__),
             'builder_sha256':sha256(ROOT/'tools/b2d_adapter/experimental_status_builder.py'),
             'elapsed_seconds':time.monotonic()-start,'training_admission':'NO-GO'}
    write_json(output/'full_status_overlay_summary.json',summary)
    write_json(output/'full_status_overlay_preview_shas.json',preview_file_shas)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--preview-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    print(json.dumps(build(p.parse_args()),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
