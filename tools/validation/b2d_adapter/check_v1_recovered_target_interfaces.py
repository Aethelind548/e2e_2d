"""World geometry, de-duplicated union and real paired reads of isolated recovery."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def main():
    import torch
    from tools.b2d_adapter import FrameReader
    from tools.b2d_adapter.navsim_bridge import candidate_vehicle_boxes
    from tools.b2d_adapter.v1_builders import official_v1_module
    from tools.training.diffusiondrive_v1.target_retry_audit import readonly
    from tools.training.diffusiondrive_v1.target_jobs import AnnotationIndex
    from tools.training.diffusiondrive_v1.target_storage import load_target_row
    from tools.training.diffusiondrive_v1.features_cache import digest,durable_json
    from tools.training.diffusiondrive_v1.runtime import model_configuration,native_configuration,initialization_manifest
    from tools.training.diffusiondrive_v1.paired_dataset import PairedDataset,pair_checks
    from tools.validation.b2d_adapter.audit_v1_target_rejections import trajectory_checks,interface,OUT
    from tools.training.diffusiondrive_v1.feature_jobs import config_and_state
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    model,native=model_configuration(cfg),native_configuration(cfg)
    state=ROOT/'outputs/diffusiondrive_v1/target_retry_audit_20261017'
    with readonly(state/'index.sqlite') as db:
        producer=json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
        spec=json.loads(db.execute("SELECT value FROM meta WHERE key='target_root'").fetchone()[0])
        rows=[dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY route,frame")]
    isolated=dict(cfg,target_state_dir=str(state),target_root=spec)
    official=official_v1_module('transfuser_features').TransfuserTargetBuilder(model)
    features={};metadata_union={};job_counts=[]
    with readonly(ROOT/cfg['target_state_dir']/'index.sqlite') as db:
        baseline={r[0]:r[1] for r in db.execute("SELECT sample_id,current_anno_sha256 FROM samples WHERE state='complete'")}
    sources=[(rows,spec,digest(producer))]
    old_cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_recovery_irrelevant_v1.json').read_text())
    with readonly(ROOT/old_cfg['target_state_dir']/'index.sqlite') as db:
        old_sha=digest(json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0]))
        old_rows=[dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete'")]
    sources.append((old_rows,old_cfg['target_root'],old_sha))
    all_recovery={r['sample_id']:r for rows_,_,_ in sources for r in rows_}
    assert not baseline.keys() & all_recovery.keys()
    assert len(all_recovery)==len(rows)+len(old_rows)
    ids=set(all_recovery)
    placeholders=','.join('?' for _ in ids)
    from tools.training.diffusiondrive_v1.target_cache import current_annotation_source
    from tools.training.diffusiondrive_v1.features_cache import feature_producer
    collection=json.loads((ROOT/cfg['features_collection']).read_text())
    assert collection['producer_sha256']==digest(feature_producer(cfg))
    base_intersection=0;feature_count=0;pair_mismatches=[]
    for entry in collection['jobs']:
        fcfg,fstate=config_and_state(ROOT/entry['config'])
        with readonly(fstate/'index.sqlite') as db:
            for rec in db.execute("SELECT * FROM samples WHERE state='complete'"):
                row=dict(rec);feature_count+=1;sid=row['sample_id']
                if sid in baseline:
                    if current_annotation_source(json.loads(row['source_json']),row['route'],row['frame'])['sha256']==baseline[sid]:base_intersection+=1
                    else:pair_mismatches.append(sid)
                if sid in ids:
                    f={'row':row,'job_id':entry['job_id'],'dataset_id':cfg['dataset_id'],'config':entry['config'],
                        'producer_sha256':collection['producer_sha256']}
                    features[sid]=f
    records=[];paired_rows=[]
    for row in rows:
        reader=FrameReader(AnnotationIndex.from_dataset_root(cfg['dataset_root'],row['route']))
        payload=load_target_row(row,spec,digest(producer))
        geometry=trajectory_checks(payload,reader,native)
        raw=reader.read(reader.index.ref(row['frame']))
        states,labels=official._compute_agent_targets(candidate_vehicle_boxes(raw,native.geometry))
        assert torch.equal(states,payload['targets']['agent_states']) and torch.equal(labels,payload['targets']['agent_labels'])
        rec={'sample_id':row['sample_id'],'route':row['route'],'frame':row['frame'],'geometry':geometry,
            'detection_equal_unfiltered_official':True,'feature_available':row['sample_id'] in features}
        records.append(rec)
        if row['sample_id'] in features:
            f=features[row['sample_id']];pair_checks(f,row,{'route':row['route'],'frame':row['frame']},native,producer)
            paired_rows.append({'sample_id':row['sample_id'],'dataset_id':cfg['dataset_id'],'route':row['route'],'frame':row['frame'],
                'feature_config':f['config'],'feature_root_id':f['job_id']+':'+f['row']['root_id'],
                'target_root_id':spec['root_id'],'feature_producer_sha':f['producer_sha256'],'target_producer_sha':digest(producer),
                'feature_row':f['row'],'target_row':row})
    valid_old=0
    for r in old_rows:
        if r['sample_id'] in features:
            f=features[r['sample_id']]
            assert current_annotation_source(json.loads(f['row']['source_json']),r['route'],r['frame'])['sha256']==r['current_anno_sha256']
            valid_old+=1
    dataset=PairedDataset(paired_rows,isolated,producer)
    pairs=[dataset[i] for i in range(len(dataset))]
    image_init=initialization_manifest(cfg,model)
    assert len(pairs)>=2
    # Reuse the identical original forward/loss check; keep baseline evidence distinct.
    interface(pairs,isolated,model,output_path=OUT/'recovered_interface_validation.json')
    durable_json(OUT/'recovered_geometry_and_intersection.json',{
        'new_complete':len(rows),'previous_recovery_complete':len(old_rows),'combined_target_count':len(baseline)+len(all_recovery),
        'new_geometry_checked':records,'new_paired_dataset_reads':len(pairs),'previous_recovery_valid_pairs':valid_old,
        'features_complete_snapshot':feature_count,'baseline_valid_intersection':base_intersection,
        'new_recovery_valid_intersection':len(pairs),'union_valid_intersection':base_intersection+len(pairs)+valid_old,
        'baseline_pair_source_mismatches':pair_mismatches,'deduplicated':True,'baseline_still_unchanged':True,
        'pretrained_initialization':image_init,'recovery_producer_sha256':digest(producer),
        'note':'Explicit PairedDataset for isolated producer works; production paired_datasets factory remains frozen baseline-only'})


if __name__=='__main__':main()
