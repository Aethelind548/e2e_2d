"""Authoritative union report over target-only indexes and verified 6DoF pairs.

Queue state is retained alongside final union state: a frozen-Feature queue can
still say needs_feature_contract while a separate new-Feature pair is complete.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def main():
    import torch
    from tools.training.diffusiondrive_v1.target_retry_audit import full_export,readonly
    from tools.training.diffusiondrive_v1.features_cache import durable_json,file_digest
    from tools.training.diffusiondrive_v1.gravity6_recovery import Gravity6Dataset
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);args=p.parse_args()
    state=args.state.resolve();out=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/full_target_recovery'
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    full_export(cfg,state,out)
    body=json.loads((out/'recovery_full_diagnosis.json').read_text());samples=body['samples']
    byid={s['sample_id']:s for s in samples}
    manifest_path=state/'gravity6_final/gravity6/manifest.json'
    proof_path=state/'gravity6_final/gravity6/pilot_validation.json'
    if manifest_path.exists() and proof_path.exists():
        manifest=json.loads(manifest_path.read_text());proof=json.loads(proof_path.read_text())
        assert proof['producer_sha256']==manifest['producer_sha256']
        dataset=Gravity6Dataset(manifest)
        for i,row in enumerate(manifest['rows']):
            dataset[i]
            sample=byid[row['sample_id']]
            assert sample['final_state']=='needs_feature_contract'
            sample['queue_state']=sample['final_state'];sample['final_state']='complete_gravity6'
            sample['attempts']=sample.get('attempts',0)+1
            sample['gravity6_complete']={'manifest':str(manifest_path),'producer_sha256':manifest['producer_sha256'],
                'relative_path':row['relative_path'],'sha256':row['sha256'],'features_regenerated':True}
    states=Counter(s['final_state'] for s in samples);byoriginal={};combinations=Counter();rules=Counter()
    for s in samples:
        byoriginal.setdefault(s['original_error_code'],Counter())[s['final_state']]+=1
        codes=tuple(sorted({e['code'] for key in ['trajectory_errors','detection_errors','bev_errors','source_errors'] for e in s.get(key,[])}))
        combinations[codes]+=1
        if s['final_state'].startswith('complete_'):
            if s['final_state']=='complete_gravity6':applied=['gravity6_reference_and_regenerated_features']
            else:
                applied=[];profile=s['validation']['validated_profile']
                head=s['validation']['head_checks']['components']
                agents=head['detection']['provenance'];events=head['bev']['provenance']['coverage']['recovery_events']
                if agents.get('recovery',{}).get('excluded_actor_ids'):applied.append('irrelevant_actor')
                if agents.get('projected_actors'):applied.append('projected_actor')
                applied+=sorted({e['rule'] for e in events})
                if not applied:applied=['qualified_full_rebuild']
            s['applied_rules']=sorted(applied);rules[tuple(sorted(applied))]+=1
    new=sum(n for st,n in states.items() if st.startswith('complete_'))
    summary={'original_nonfuture':len(samples),'historical_complete':states['historical_complete'],
        'new_complete':new,'cumulative_recovered':new+states['historical_complete'],
        'final_complete_targets':221100+new+states['historical_complete'],'states':dict(states),
        'by_original':{k:dict(v) for k,v in byoriginal.items()},'diagnosed':len(samples),
        'mutually_exclusive_final_states':True,'all_source_indexes_explicit':True,
        'note':'Final state is the union of independent Target indexes and explicitly validated new-Feature pairs; queue_state is retained for the latter.'}
    body['states']=dict(states);body['count']=len(samples);body['union_summary']=summary
    durable_json(out/'recovery_full_diagnosis.json',body);durable_json(out/'summary.json',summary)
    with (out/'recovery_candidate_manifest.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['sample_id','route','frame','original_code','final_state','queue_state','attempts','trajectory_errors','detection_errors','bev_errors','source_errors'])
        for s in samples:writer.writerow([s.get(k) for k in ['sample_id','route','frame','original_error_code','final_state','queue_state','attempts']]+[','.join(sorted({e['code'] for e in s.get(key,[])})) for key in ['trajectory_errors','detection_errors','bev_errors','source_errors']])
    lines=['# 多重错误分布与最终互斥状态','','原始诊断保留所有错误；每帧只计一个最终状态。','',
        '| 完整错误集合 | 帧数 |','|---|---:|']
    lines += [f"| {' + '.join(combo) or 'none'} | {n} |" for combo,n in combinations.most_common()]
    lines += ['','## 最终状态','', '| 状态 | 帧数 |','|---|---:|']+[f'| {k} | {v} |' for k,v in states.items()]
    lines += ['','## 新恢复的规则组合贡献（互斥）','', '| 规则组合 | 完整新增帧数 |','|---|---:|']
    lines += [f"| {' + '.join(combo)} | {n} |" for combo,n in rules.most_common()]
    (out/'multi_error_distribution.md').write_text('\n'.join(lines)+'\n')
    # Account for the specifically requested 1,027 old actor-only candidates.
    probe=json.loads((ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/target_recovery_audit/tilted_actor_recovery_probe.json').read_text())
    initial=json.loads((out/'initial_identity.json').read_text());historical=set()
    for name in ['targets_recovery_irrelevant_v1','target_retry_audit_20261017']:
        with readonly(ROOT/'outputs/diffusiondrive_v1'/name/'index.sqlite') as db:
            historical.update(r[0] for r in db.execute("SELECT sample_id FROM samples WHERE state='complete'"))
    previous=json.loads((ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/target_recovery_audit/incremental_retry_validation.json').read_text())
    four_failed={r['sample_id'] for r in previous['full_results'] if not r['full_training_eligible']}
    specific={r['sample_id'] for r in probe['frames'] if r['all_tilted_actors_irrelevant']}-historical-four_failed
    assert len(specific)==1027
    durable_json(out/'original_1027_outcomes.json',{'count':len(specific),'states':dict(Counter(byid[sid]['final_state'] for sid in specific)),
        'samples':[{'sample_id':sid,'final_state':byid[sid]['final_state']} for sid in sorted(specific)]})
    protected={}
    for name,expected in initial['indexes'].items():
        with readonly(expected['index_path']) as db:
            rows=[tuple(r) for r in db.execute("SELECT sample_id,sha256,bytes,relative_path FROM samples WHERE state='complete' ORDER BY sample_id")]
            meta=dict(db.execute('SELECT key,value FROM meta'))
        same=hashlib.sha256(json.dumps(rows).encode()).hexdigest()==expected['complete_rows_sha256']
        assert same and meta==expected['meta']
        protected[name]={'complete_rows_unchanged':len(rows),'producer_and_root_meta_unchanged':True}
    durable_json(out/'protected_baseline_check.json',protected)
    flag=[s for s in samples if s['final_state']=='blocked_source_semantics']
    tokens={(s['route'],fid) for s in flag for fid in s['trajectory_metrics']['flag_frames']}
    durable_json(out/'teleport_evidence.json',{'windows':len(flag),'distinct_flagged_route_frames':len(tokens),
        'max_window_xyz_speed_mps':max((s['trajectory_metrics']['max_xyz_speed_mps'] for s in flag),default=0),
        'max_window_xy_speed_mps':max((s['trajectory_metrics']['max_xy_speed_mps'] for s in flag),default=0),
        'max_window_yaw_rate_radps':max((s['trajectory_metrics']['max_yaw_rate_radps'] for s in flag),default=0),
        'producer_semantics':'not authenticated for V0.0.4; no flag bypass enabled',
        'flagged_tokens':[{'route':r,'frame':f} for r,f in sorted(tokens)]})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
