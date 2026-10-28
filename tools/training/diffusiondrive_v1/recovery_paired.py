"""Explicit multi-producer cache collection for the existing V1 fit entry.

The caller must whitelist exact Target producer hashes. Existing manifests and
the baseline-only factory path keep their original validation unchanged.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1.features_cache import digest,file_digest,durable_json,load_feature_row,directory_sync,feature_producer
from tools.training.diffusiondrive_v1.target_cache import current_annotation_source,ROOT
from tools.training.diffusiondrive_v1.target_storage import load_target_row
from tools.training.diffusiondrive_v1.target_retry_audit import readonly as _readonly

SCHEMA='b2d-v1-explicit-recovery-paired-v1'


@contextmanager
def readonly(path):
    db=_readonly(path)
    try:yield db
    finally:db.close()


def build(config,state,manifest_path,*,include_gravity6=False,target_source_names=None,storage_registration=None):
    from tools.training.diffusiondrive_v1.feature_jobs import config_and_state
    from tools.training.diffusiondrive_v1.runtime import native_configuration,route_group
    from tools.training.diffusiondrive_v1.features_cache import sample_id
    from tools.b2d_adapter.v1_label_rules import label_rules
    native=native_configuration(config)
    state=Path(state).resolve();manifest_path=Path(manifest_path).resolve()
    with readonly(state/'recovery.sqlite') as db:
        cache=Path(json.loads(db.execute("SELECT value FROM meta WHERE key='cache_root'").fetchone()[0]))
    # Metadata is versioned alongside the requested manifest, not in a tensor root.
    root=manifest_path.parent/(manifest_path.stem+'_index');root.mkdir(parents=True,exist_ok=True)
    dest=root/'index.sqlite';temp=root/'index.sqlite.partial'
    if dest.exists() or manifest_path.exists():raise FileExistsError('Use a new versioned paired index; existing manifest is immutable')
    if temp.exists():temp.unlink()  # Only this unpublished build's interrupted temporary index.
    sources={};features={};records={};counts={};duplicates=[]
    specs=[('baseline',ROOT/config['target_state_dir'],0),
        ('historical26',ROOT/'outputs/diffusiondrive_v1/targets_recovery_irrelevant_v1',1),
        ('historical28',ROOT/'outputs/diffusiondrive_v1/target_retry_audit_20261017',1),
        ('compatible',state/'compatible',2),('projected',state/'projected',3)]
    if target_source_names is not None:
        if not set(target_source_names)<={r[0] for r in specs}:raise ValueError('Unknown Target source selection')
        specs=[r for r in specs if r[0] in target_source_names]
    for source,folder,rank in specs:
        if not (folder/'index.sqlite').exists():continue
        with readonly(folder/'index.sqlite') as db:
            meta={r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
            producer=meta['producer']
            if (meta.get('dataset_id')!=config['dataset_id'] or producer['geometry']!=native.geometry.provenance()
                    or producer['rules']!=label_rules().identity()
                    or producer['maps']!={k:{'sha256':v['sha256'],'version':v['carla_version']} for k,v in config['map_paths'].items()}):
                raise AdapterError('recovery_pair_contract_mismatch',source+': dataset/reference/map/rules')
            sources[source]={'state':str(folder),'target_root':meta['target_root'],'producer':meta['producer'],
                'producer_sha256':digest(meta['producer']),'rank':rank}
            count=0
            for r in db.execute("SELECT sample_id,route,frame,current_anno_sha256,sha256,sources_json FROM samples WHERE state='complete'"):
                sid=r['sample_id']
                if (sid!=sample_id(config['dataset_id'],r['route'],r['frame'])
                        or current_annotation_source(json.loads(r['sources_json']),r['route'],r['frame'])['sha256']!=r['current_anno_sha256']):
                    raise AdapterError('pair_source_mismatch',sid)
                if sid in records:duplicates.append(sid);continue
                records[sid]=(source,r['route'],r['frame'],r['current_anno_sha256'],r['sha256'],rank);count+=1
            counts[source]={'complete':count,'paired':0}
    gravity_path=state/'gravity6_final/gravity6/manifest.json'
    gravity_validation=state/'gravity6_final/gravity6/pilot_validation.json'
    gravity=None
    if include_gravity6 and gravity_path.exists() and gravity_validation.exists():
        gravity=json.loads(gravity_path.read_text());checked=json.loads(gravity_validation.read_text())
        if checked['producer_sha256']!=gravity['producer_sha256']:raise AdapterError('recovery_pair_contract_mismatch','Gravity6 pilot gate differs')
        sources['gravity6']={'kind':'paired-gravity6','manifest':str(gravity_path),'manifest_sha256':file_digest(gravity_path),
            'producer':gravity['producer'],'producer_sha256':gravity['producer_sha256'],'rank':4,
            'root_spec':dict(config['target_root'],path=gravity['root'],root_id='gravity6_pairs')}
        counts['gravity6']={'complete':len(gravity['rows']),'paired':len(gravity['rows'])}
        for r in gravity['rows']:
            if r['sample_id'] in records:duplicates.append(r['sample_id'])
    if duplicates:raise AdapterError('recovery_pair_duplicate',duplicates[:5])
    index=sqlite3.connect(temp)
    index.execute('PRAGMA synchronous=FULL')
    index.execute('''CREATE TABLE pairs(sample_id TEXT PRIMARY KEY,route TEXT NOT NULL,frame INTEGER NOT NULL,
       split TEXT NOT NULL,target_source TEXT NOT NULL,feature_source TEXT NOT NULL,
       target_sha TEXT NOT NULL,feature_sha TEXT NOT NULL,current_anno_sha TEXT NOT NULL,source_rank INTEGER NOT NULL)''')
    feature_collection=json.loads((ROOT/config['features_collection']).read_text())
    fp=digest(feature_producer(config))
    if fp!=feature_collection['producer_sha256']:raise AdapterError('recovery_feature_contract_mismatch','producer')
    mismatches=[];feature_count=0;feature_ids=set();route_splits={};group_splits={}
    for entry in feature_collection['jobs']:
        cfg,folder=config_and_state(ROOT/entry['config'])
        features[entry['job_id']]={'state':str(folder),'config':entry['config'],'roots':cfg['cache_roots'],'producer_sha256':fp}
        with readonly(folder/'index.sqlite') as db:
            if digest(json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0]))!=fp:
                raise AdapterError('recovery_feature_contract_mismatch',entry['job_id'])
            for r in db.execute("SELECT sample_id,route,frame,source_json,sha256 FROM samples WHERE state='complete'"):
                feature_count+=1;sid=r['sample_id']
                if sid in feature_ids:raise AdapterError('recovery_pair_duplicate','Feature '+sid)
                feature_ids.add(sid)
                if sid not in records:continue
                source,route,frame,anno,tsha,rank=records[sid]
                f_anno=current_annotation_source(json.loads(r['source_json']),r['route'],r['frame'])['sha256']
                if route!=r['route'] or frame!=r['frame'] or anno!=f_anno:
                    mismatches.append(sid);continue
                town=re.search(r'_(Town\d+(?:HD)?)_',route).group(1)
                split='validation' if town=='Town05' else 'train'
                group=route_group(route)
                if route_splits.setdefault(route,split)!=split or group_splits.setdefault(group,split)!=split:
                    raise AdapterError('pair_split_leakage',route)
                index.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (sid,route,frame,split,source,entry['job_id'],tsha,r['sha256'],anno,rank))
                counts[source]['paired']+=1
    if mismatches:
        index.close()
        raise AdapterError('pair_source_mismatch',mismatches[:5])
    if gravity is not None:
        for row in gravity['rows']:
            anno=Path(config['dataset_root'])/row['route']/'anno'/f"{row['frame']:05d}.json.gz"
            index.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?,?,?)',(row['sample_id'],row['route'],row['frame'],
                'validation' if '_Town05_' in row['route'] else 'train','gravity6','gravity6',row['sha256'],row['sha256'],file_digest(anno),4))
    index.execute('CREATE INDEX paired_split ON pairs(split,source_rank DESC,route,frame)')
    index.commit();splits=dict(index.execute('SELECT split,COUNT(*) FROM pairs GROUP BY split'));index.close()
    with temp.open('rb') as f:os.fsync(f.fileno())
    os.replace(temp,dest);directory_sync(root)
    result={'schema':SCHEMA,'dataset_id':config['dataset_id'],'pair_index':str(dest),'index_sha256':file_digest(dest),
        'target_sources':sources,'feature_sources':features,'feature_producer_sha256':fp,
        'counts':counts,'split_counts':splits,'split_policy':{'validation_towns':['Town05'],'train':'all other Towns'},
        'features_complete':feature_count,'source_mismatches':mismatches,'duplicates':duplicates}
    result.update(gravity6_included=gravity is not None,
        gravity6_separate_manifest=str(gravity_path) if gravity_path.exists() and gravity is None else None,
        target_complete_unique=len(records),target_without_feature=len(records)-sum(v['paired'] for k,v in counts.items() if k!='gravity6'),
        feature_without_target=len(feature_ids-set(records)),
        route_split_counts={s:sum(v==s for v in route_splits.values()) for s in splits},
        route_group_split_counts={s:sum(v==s for v in group_splits.values()) for s in splits},
        raw_source_verification='Recorded current annotation SHA checked across all rows; actual raw files are verified on Dataset read.')
    if storage_registration:
        from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
        path=Path(storage_registration).resolve()
        result['feature_storage_registration']={'path':str(path),'sha256':file_digest(path)}
        for source in features.values():effective_roots(result,source)
    durable_json(manifest_path,result);return result


class RecoveryPairedDataset:
    def __init__(self,manifest,config,split):
        self.manifest,self.config,self.split=manifest,config,split
        with readonly(manifest['pair_index']) as db:
            self.rows=[dict(r) for r in db.execute('SELECT * FROM pairs WHERE split=? ORDER BY source_rank DESC,route,frame',(split,))]
        self.tokens=[str(Path(config['dataset_root'])/r['route'])+'::'+str(r['frame']) for r in self.rows]

    def __len__(self):return len(self.rows)

    def target_root(self, source):
        from tools.training.diffusiondrive_v1.target_relocation import TargetLocationResolver
        if not hasattr(self, '_target_locations'):
            self._target_locations = TargetLocationResolver(self.manifest, self.config)
        return self._target_locations.resolve(source['target_root'], source['producer_sha256'], source['state'])

    def __getitem__(self,i):
        from tools.training.diffusiondrive_v1.paired_dataset import pair_checks
        from tools.training.diffusiondrive_v1.runtime import native_configuration
        r=self.rows[i];source=self.manifest['target_sources'][r['target_source']]
        if source.get('kind')=='paired-gravity6':
            from tools.training.diffusiondrive_v1.gravity6_recovery import Gravity6Dataset
            from tools.training.diffusiondrive_v1.target_cache import storage_check
            storage_check(source['root_spec'])
            if file_digest(source['manifest'])!=source['manifest_sha256']:raise AdapterError('recovery_pair_index_mismatch','Gravity6 manifest changed')
            manifest=json.loads(Path(source['manifest']).read_text())
            entry=next((row for row in manifest['rows'] if row['sample_id']==r['sample_id']),None)
            if entry is None or entry['sha256']!=r['target_sha'] or entry['route']!=r['route'] or entry['frame']!=r['frame']:
                raise AdapterError('recovery_pair_index_mismatch',r['sample_id'])
            annotation=Path(self.config['dataset_root'])/r['route']/'anno'/f"{r['frame']:05d}.json.gz"
            if file_digest(annotation)!=r['current_anno_sha']:raise AdapterError('pair_source_mismatch',r['sample_id'])
            return Gravity6Dataset(dict(manifest,rows=[entry]))[0]
        fsource=self.manifest['feature_sources'][r['feature_source']]
        with readonly(Path(source['state'])/'index.sqlite') as db:target=db.execute('SELECT * FROM samples WHERE sample_id=?',(r['sample_id'],)).fetchone()
        with readonly(Path(fsource['state'])/'index.sqlite') as db:feature=db.execute('SELECT * FROM samples WHERE sample_id=?',(r['sample_id'],)).fetchone()
        if target is None or feature is None or target['state']!='complete' or feature['state']!='complete':raise AdapterError('recovery_pair_index_mismatch',r['sample_id'])
        target,feature=dict(target),dict(feature)
        if target['sha256']!=r['target_sha'] or feature['sha256']!=r['feature_sha'] or target['current_anno_sha256']!=r['current_anno_sha']:
            raise AdapterError('recovery_pair_index_mismatch',r['sample_id'])
        pair_checks({'row':feature,'dataset_id':self.config['dataset_id']},target,
            {'route':r['route'],'frame':r['frame']},native_configuration(self.config),source['producer'])
        from tools.b2d_adapter import FrameReader,RouteIndex
        from tools.training.diffusiondrive_v1.target_cache import reference_pose
        reader=FrameReader(RouteIndex.from_dataset_root(self.config['dataset_root'],r['route']))
        ref=reader.index.ref(r['frame'])
        if file_digest(reader.index.path(ref,'anno'))!=r['current_anno_sha']:
            raise AdapterError('pair_source_mismatch',r['sample_id'])
        if digest(reference_pose(reader.read(ref),native_configuration(self.config)))!=digest(json.loads(target['reference_json'])):
            raise AdapterError('pair_coordinate_mismatch',r['sample_id'])
        from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
        values=load_feature_row(feature,effective_roots(self.manifest,fsource),fsource['producer_sha256'])
        targets=load_target_row(target,self.target_root(source),source['producer_sha256'])['targets']
        return values,targets


def datasets(path,config,model):
    manifest=json.loads(Path(path).read_text())
    approved=set(config.get('allowed_recovery_producers',[]))
    actual={s['producer_sha256'] for s in manifest['target_sources'].values()}
    if manifest['schema']!=SCHEMA or manifest['dataset_id']!=config['dataset_id'] or approved!=actual:
        raise AdapterError('recovery_pair_contract_mismatch','Explicit complete producer whitelist required')
    if manifest['feature_producer_sha256']!=digest(feature_producer(config)) or file_digest(manifest['pair_index'])!=manifest['index_sha256']:
        raise AdapterError('recovery_pair_contract_mismatch','Feature/index digest changed')
    registration=manifest.get('feature_storage_registration')
    if registration and config.get('allowed_feature_storage_registration_sha256')!=registration['sha256']:
        raise AdapterError('recovery_pair_contract_mismatch','Storage relocation requires explicit registration SHA')
    for source in manifest['target_sources'].values():
        if source.get('kind')=='paired-gravity6':
            from tools.training.diffusiondrive_v1.gravity6_recovery import CONTRACT
            if CONTRACT['version'] not in config.get('approved_new_geometry_contracts',[]) or digest(source['producer'])!=source['producer_sha256']:
                raise AdapterError('recovery_pair_contract_mismatch','Gravity6 geometry requires explicit opt-in')
            if source['producer'].get('base_feature_producer_sha256')!=manifest['feature_producer_sha256'] or source['producer']['contract']!=CONTRACT or source['producer']['dataset_id']!=config['dataset_id']:
                raise AdapterError('recovery_pair_contract_mismatch','Gravity6 parent/reference differs')
            continue
        with readonly(Path(source['state'])/'index.sqlite') as db:
            meta={r['key']:json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
        if digest(source['producer'])!=source['producer_sha256'] or digest(meta['producer'])!=source['producer_sha256'] or meta['target_root']!=source['target_root']:
            raise AdapterError('recovery_pair_contract_mismatch','Target index producer/root differs')
        if 'geometry_contract' in source['producer'] and source['producer']['geometry_contract']['version'] not in config.get('approved_new_geometry_contracts',[]):
            raise AdapterError('recovery_pair_contract_mismatch','Projected geometry requires explicit opt-in')
    for source in manifest['feature_sources'].values():
        from tools.training.diffusiondrive_v1.feature_jobs import config_and_state
        current,_=config_and_state(ROOT/source['config'])
        if current['cache_roots']!=source['roots']:raise AdapterError('recovery_pair_contract_mismatch','Feature root registration changed')
        from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
        effective_roots(manifest,source)
    reader=RecoveryPairedDataset
    if config.get('cache_read_mode','audited')=='cache-only':
        from tools.training.diffusiondrive_v1.cache_only_reader import CacheOnlyPairedDataset
        reader=CacheOnlyPairedDataset
    elif config.get('cache_read_mode','audited')!='audited':
        raise ValueError('Unknown cache read mode')
    elif config.get('mount_check_mode','findmnt')=='procfs':
        from tools.training.diffusiondrive_v1.procfs_mount_reader import ProcfsMountPairedDataset
        reader=ProcfsMountPairedDataset
    elif config.get('mount_check_mode','findmnt')!='findmnt':
        raise ValueError('Unknown mount check mode')
    result={s:reader(manifest,config,s) for s in ['train','validation']}
    if any(not len(v) for v in result.values()):raise AdapterError('empty_paired_split','recovery')
    root=Path(config['dataset_root']).resolve()
    for split,dataset in result.items():
        expected=set()
        for entry in config['data'][split]:
            route=str((ROOT/entry['route']).resolve().relative_to(root))
            for frame in entry['frames']:
                if (route,frame) in expected:raise AdapterError('recovery_pair_split_mismatch','duplicate requested frame')
                expected.add((route,frame))
        if expected!={(r['route'],r['frame']) for r in dataset.rows}:raise AdapterError('recovery_pair_split_mismatch','Selection differs from authenticated full index')
    return result,{'path':str(Path(path).resolve()),'sha256':file_digest(path),'schema':SCHEMA,
        'counts':{s:len(v) for s,v in result.items()},'allowed_recovery_producers':sorted(actual),
        'feature_producer_sha256':manifest['feature_producer_sha256'],'explicit_new_geometry_opt_in':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-gravity6',action='store_true',help='Explicit opt-in only; default keeps the new reference in its independent manifest')
    p.add_argument('--baseline-only',action='store_true');p.add_argument('--storage-registration',type=Path)
    args=p.parse_args();config=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    if args.baseline_only and args.include_gravity6:p.error('Baseline excludes gravity6')
    print(json.dumps(build(config,args.state,args.output,include_gravity6=args.include_gravity6,
        target_source_names=['baseline'] if args.baseline_only else None,storage_registration=args.storage_registration)['counts']))


if __name__=='__main__':main()
