"""Publish new Baseline-only storage/configuration versions; never start training."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from tools.training.diffusiondrive_v1.features_cache import durable_json,file_digest

ROOT=Path(__file__).resolve().parents[3]


def configuration(manifest_path,output_path,audit):
    manifest=json.loads(manifest_path.read_text())
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    cfg['paired_manifest']=str(manifest_path);cfg.pop('prepared_manifest',None)
    cfg['allowed_recovery_producers']=[s['producer_sha256'] for s in manifest['target_sources'].values()]
    assert set(manifest['target_sources'])=={'baseline'}
    cfg['approved_new_geometry_contracts']=[]
    cfg['allowed_feature_storage_registration_sha256']=manifest['feature_storage_registration']['sha256']
    cfg['official_reference_commit']=audit['reference_commit'];cfg['official_source_sha256']={k:v['current_sha256'] for k,v in audit['sources'].items()}
    cfg['data']={'train':[],'validation':[]}
    db=sqlite3.connect(f"file:{manifest['pair_index']}?mode=ro",uri=True)
    for split,route in db.execute('select distinct split,route from pairs order by split,route'):
        frames=[r[0] for r in db.execute('select frame from pairs where split=? and route=? order by frame',(split,route))]
        cfg['data'][split].append({'route':str(Path(cfg['dataset_root'])/route),'frames':frames})
    counts=dict(db.execute('select split,count(*) from pairs group by split'));db.close()
    cfg['batch_size']=2;cfg['threads']=2;cfg['tensor_memory_items']=2
    cfg['reference_eval_batches']=4;cfg['evidence_every_steps']=1000;cfg['step_report_limit']=128;cfg['log_every_n_steps']=100
    cfg['strict_resume_audit']=True;cfg['csv_logger']=True;cfg['output_dir']=str(output_path)
    cfg['output_storage']={'path':str(output_path),'uuid':'F8E4AE34E4ADF552','fstype':'ntfs3','minimum_free_gib':100}
    cfg['training']=dict(cfg['training'],max_epochs=100,max_steps=math.ceil(counts['train']/cfg['batch_size'])*100,
        num_workers=2,persistent_workers=True,prefetch_factor=2,precision='32-true',validation_every_epochs=1,
        checkpoint_every_steps=5000)
    cfg['training'].pop('validation_every_steps',None)
    cfg['experiment_protocol']={'phase':'baseline-only','target_producer':cfg['allowed_recovery_producers'][0],
        'formal_training_started':False,'scheduler':'Official WarmupCosLR hardcoded epochs=100,warmup_epochs=3,min_lr=1e-6; epoch stepping',
        'validation_sample_ids_sha256':None,'batch_size_not_original_distributed_batch':True}
    with sqlite3.connect(f"file:{manifest['pair_index']}?mode=ro",uri=True) as db:
        ids=[r[0] for r in db.execute("select sample_id from pairs where split='validation' order by sample_id")]
    cfg['experiment_protocol']['validation_sample_ids_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--migration-verification',type=Path,required=True,
                   help='Successful post-remount all-file verification; checked before any publication')
    args=p.parse_args()
    out=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight'
    reg={'path':str(args.registration.resolve()),'sha256':file_digest(args.registration)}
    original=json.loads((out/'baseline_original_locations_manifest.json').read_text())
    assert original['split_counts'] and sum(original['split_counts'].values())==218622
    baseline=copy.deepcopy(original);baseline['feature_storage_registration']=reg
    registration=json.loads(args.registration.read_text())
    verification=json.loads(args.migration_verification.read_text())
    if (verification.get('status')!='passed' or verification.get('registration_sha256')!=reg['sha256']
            or verification.get('verified_files')!=registration['verified_files']
            or verification.get('verified_bytes')!=registration['verified_bytes']):
        raise RuntimeError('Migration must pass post-remount verification before training setup')
    from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
    from tools.training.diffusiondrive_v1.features_cache import active_mount
    for source in baseline['feature_sources'].values():
        for spec in effective_roots(baseline,source).values():active_mount(spec)
    baseline['migration_acceptance']={'path':str(args.migration_verification.resolve()),
                                      'sha256':file_digest(args.migration_verification)}
    baseline['storage_parent_manifest']={'path':str(out/'baseline_original_locations_manifest.json'),'sha256':file_digest(out/'baseline_original_locations_manifest.json')}
    baseline_path=out/'baseline_pssd_manifest.json'
    if baseline_path.exists():raise FileExistsError('Baseline publication exists')
    durable_json(baseline_path,baseline)
    old_multi=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/full_target_recovery/acceptance_20261019/multi_producer_manifest.json'
    multi=json.loads(old_multi.read_text());multi['feature_storage_registration']=reg
    multi['storage_parent_manifest']={'path':str(old_multi),'sha256':file_digest(old_multi)}
    durable_json(out/'multi_producer_pssd_locations_manifest.json',multi)
    audit=json.loads((out/'official_runtime_audit.json').read_text())
    experiment=Path('/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/baseline_original_7class_20261019')
    from tools.training.diffusiondrive_v1.target_cache import storage_check
    storage_check({'path':str(experiment),'uuid':'F8E4AE34E4ADF552','fstype':'ntfs3'},True)
    experiment.mkdir(parents=True,exist_ok=False)
    cfg=configuration(baseline_path,experiment,audit)
    formal=ROOT/'tools/training/diffusiondrive_v1/baseline_original_7class_20261019.json'
    if formal.exists():raise FileExistsError(formal)
    durable_json(formal,cfg);durable_json(experiment/'configuration.json',cfg)
    durable_json(experiment/'dataset_identity.json',{'manifest':str(baseline_path),'sha256':file_digest(baseline_path),'storage':reg,'long_training_started':False})
    # Bounded child index: 16 actual train examples spanning Towns and 8 Town05.
    source=sqlite3.connect(f"file:{baseline['pair_index']}?mode=ro",uri=True);source.row_factory=sqlite3.Row
    train=[dict(r) for r in source.execute("select * from pairs where split='train' order by route,frame")];chosen=[];towns=set()
    for row in train:
        town=re.search(r'_(Town\d+(?:HD)?)_',row['route']).group(1)
        if town not in towns:chosen.append(row);towns.add(town)
    ids={r['sample_id'] for r in chosen}
    remaining=baseline['feature_sources']['remaining_20261015']['state']
    with sqlite3.connect(f'file:{remaining}/index.sqlite?mode=ro',uri=True) as db:
        moved_ids={r[0] for r in db.execute("select sample_id from samples where root_id='cache_system' and state='complete'")}
    moved=next(r for r in train if r['sample_id'] in moved_ids)
    if moved['sample_id'] not in ids:chosen.append(moved);ids.add(moved['sample_id'])
    for row in train:
        if len(chosen)>=16:break
        if row['sample_id'] not in ids:chosen.append(row);ids.add(row['sample_id'])
    chosen += [dict(r) for r in source.execute("select * from pairs where split='validation' order by route,frame limit 8")]
    idx=out/'preflight_subset.sqlite'
    target=sqlite3.connect(idx);target.execute(source.execute("select sql from sqlite_master where name='pairs'").fetchone()[0])
    for r in chosen:target.execute('insert into pairs values('+','.join('?' for _ in r)+')',list(r.values()))
    target.commit();target.close();source.close()
    subset=copy.deepcopy(baseline);subset['pair_index']=str(idx);subset['index_sha256']=file_digest(idx)
    subset['split_counts']={'train':16,'validation':8};subset['counts']['baseline']['paired']=24
    subset['selection']={'purpose':'bounded preflight only','parent_manifest':str(baseline_path),'parent_sha256':file_digest(baseline_path),'train_towns':sorted(towns)}
    for key in ['target_without_feature','feature_without_target']:subset.pop(key,None)
    path=out/'preflight_subset_manifest.json';durable_json(path,subset)
    preflight=configuration(path,Path('/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/preflight_baseline_20261019'),audit)
    preflight.update(reference_eval_batches=2,evidence_every_steps=1,step_report_limit=128,log_every_n_steps=1)
    preflight['training'].update(max_steps=32,checkpoint_every_steps=4)
    durable_json(out/'preflight_configuration.json',preflight)
    print(json.dumps({'formal_config':str(formal),'split_counts':baseline['split_counts'],'preflight_counts':subset['split_counts'],'formal_max_steps':cfg['training']['max_steps'],'experiment':str(experiment)}))


if __name__=='__main__':main()
