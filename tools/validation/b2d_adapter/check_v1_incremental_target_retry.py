"""Probe recorded tilted-actor rejects only, then apply a bounded isolated batch.

This is an explicit validation command, not a background worker. Existing
complete recovery IDs are excluded so reported additions are genuinely new.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def main():
    import numpy as np
    import torch
    from tools.b2d_adapter import FrameReader,AdapterError
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1TargetBuilder
    from tools.training.diffusiondrive_v1.target_jobs import AnnotationIndex,failure_state
    from tools.training.diffusiondrive_v1.target_recovery import irrelevant_tilt
    from tools.training.diffusiondrive_v1.target_retry_audit import readonly,prepare,validate,baseline_rows
    from tools.training.diffusiondrive_v1.features_cache import durable_json,digest
    from tools.training.diffusiondrive_v1.runtime import model_configuration,native_configuration,bev_provider
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    state=ROOT/'outputs/diffusiondrive_v1/target_retry_audit_20261017'
    out=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/target_recovery_audit'
    prepare(cfg,state)
    torch.set_num_threads(1);model,native=model_configuration(cfg),native_configuration(cfg)
    base,rows,stats=baseline_rows(cfg);rows=[r for r in rows if r['error_code']=='invalid_vehicle_box']
    with readonly(ROOT/'outputs/diffusiondrive_v1/targets_recovery_irrelevant_v1/index.sqlite') as db:
        prior_ids={r[0] for r in db.execute("SELECT sample_id FROM samples WHERE state='complete'")}
    records=[];candidates={};current_route=None
    for index,row in enumerate(rows):
        if current_route!=row['route']:
            reader=FrameReader(AnnotationIndex.from_dataset_root(cfg['dataset_root'],row['route']))
            builder=DiffusionDriveV1TargetBuilder(reader,native,model,None);current_route=row['route']
        record={'sample_id':row['sample_id'],'route':row['route'],'frame':row['frame'],'actors':[]}
        try:
            raw=reader.read(reader.index.ref(row['frame']))
            for box in raw.annotation['bounding_boxes']:
                if box.get('class')=='vehicle' and np.max(np.abs(box['rotation'][:2]))>5:
                    record['actors'].append(irrelevant_tilt(box,raw,native.geometry,builder.official_builder))
            record['all_tilted_actors_irrelevant']=bool(record['actors']) and all(p['safely_irrelevant'] for p in record['actors'])
            if record['all_tilted_actors_irrelevant'] and row['sample_id'] not in prior_ids:
                import re
                town=re.search(r'_(Town\d+(?:HD)?)_',row['route']).group(1)
                candidates.setdefault(town,[]).append(row['sample_id'])
        except AdapterError as exc:
            failure_state(exc)
            record.update(error_code=exc.code,error=str(exc),all_tilted_actors_irrelevant=False)
        records.append(record)
        if index%500==0:print(json.dumps({'actor_probe':index+1,'of':len(rows)}),flush=True)
    # Spread a small full-rebuild batch across available Towns, not just the first route.
    selected=[]
    for depth in range(32):
        for town in sorted(candidates):
            if depth<len(candidates[town]) and len(selected)<32:selected.append(candidates[town][depth])
    result={'actor_probe_rows':len(rows),'safe_actor_gate_frames':sum(r['all_tilted_actors_irrelevant'] for r in records),
        'already_recovered_ids_excluded':len(prior_ids),'new_actor_gate_by_town':{k:len(v) for k,v in candidates.items()},
        'full_rebuild_selected':selected,'note':'Actor-only proof is not full Target acceptance; independent full heads run below',
        'frames':records}
    durable_json(out/'tilted_actor_recovery_probe.json',result)
    validated=validate(cfg,state,limit=32,apply=False,sample_ids=set(selected))
    applied=validate(cfg,state,limit=32,apply=True,sample_ids=set(selected))
    assert [r['full_training_eligible'] for r in validated]==[r['full_training_eligible'] for r in applied]
    with readonly(state/'index.sqlite') as db:
        complete=[dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete'")]
        file_before={r['sample_id']:(r['sha256'],r['bytes'],(state/'cache'/r['relative_path']).stat().st_mtime_ns) for r in complete}
    # Same selection resume must not read or rewrite an already complete row.
    repeated=validate(cfg,state,limit=32,apply=True,sample_ids=set(selected))
    assert not repeated
    assert all((state/'cache'/r['relative_path']).stat().st_mtime_ns==file_before[r['sample_id']][2] for r in complete)
    with readonly(ROOT/cfg['target_state_dir']/'index.sqlite') as db:
        after_stats=dict(db.execute('SELECT state,COUNT(*) FROM samples GROUP BY state'))
        after_producer=json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
        assert stats==after_stats and digest(base)==digest(after_producer)
    durable_json(out/'incremental_retry_validation.json',{
        'status':'pass','baseline_before':stats,'baseline_after':after_stats,'baseline_producer_unchanged':True,
        'actor_probe_rows':len(rows),'safe_actor_gate_frames':result['safe_actor_gate_frames'],
        'new_actor_gate_by_town':result['new_actor_gate_by_town'],
        'selected':len(selected),'new_complete_cache_files':len(complete),
        'complete_sample_ids':[r['sample_id'] for r in complete],
        'still_rejected_in_batch':sum(not r['full_training_eligible'] for r in applied),
        'retry_complete_rewrites':0,'isolated_state':str(state),'mode':'bounded explicit apply only; no background process',
        'full_results':applied})


if __name__=='__main__':main()
