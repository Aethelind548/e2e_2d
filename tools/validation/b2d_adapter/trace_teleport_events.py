"""Evidence for each recorded Route/flag frame; never relaxes a rejection rule."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def main():
    import numpy as np
    from tools.b2d_adapter import FrameReader,RouteIndex
    from tools.training.diffusiondrive_v1.features_cache import file_digest,durable_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    windows=defaultdict(list)
    with sqlite3.connect(f'file:{args.state}/recovery.sqlite?mode=ro',uri=True) as db:
        for sid,route,frame,text in db.execute("select sample_id,route,frame,diagnosis from tasks where status='blocked_source_semantics'"):
            d=json.loads(text)
            for flag in d['trajectory_metrics']['flag_frames']:
                windows[route,flag].append({'sample_id':sid,'frame':frame})
    events=[];readers={}
    for (route,fid),affected in sorted(windows.items()):
        if route not in readers:readers[route]=FrameReader(RouteIndex.from_dataset_root(cfg['dataset_root'],route))
        reader=readers[route];current=reader.read(reader.index.ref(fid));a=current.annotation
        ids=[str(v) for v in a.get('teleport_ids',[])];ego_id=str(current.ego_box.get('id'))
        context=[];raws=[]
        for frame in range(max(0,fid-2),fid+3):
            if frame not in reader.index.frame_ids:continue
            raw=reader.read(reader.index.ref(frame));raws.append(raw)
            boxes={str(b.get('id')):b for b in raw.annotation['bounding_boxes'] if isinstance(b,dict)}
            context.append({'frame':frame,'annotation_sha256':file_digest(reader.index.path(raw.ref,'anno')),
                'ego_id':str(raw.ego_box.get('id')),'ego_location':raw.ego_box['location'],'ego_rotation':raw.ego_box['rotation'],
                'teleport_flag':raw.state.get('teleport_flag'), 'teleport_ids':raw.annotation.get('teleport_ids'),
                'disappear_flag':raw.annotation.get('disappear_flag'),'disappear_ids':raw.annotation.get('disappear_ids'),
                'explicit_time_fields':{k:v for k,v in raw.annotation.items() if any(t in k.lower() for t in ['timestamp','game_time','elapsed_seconds','frame_id'])},
                'affected_actors':{aid:{k:boxes[aid].get(k) for k in ['class','type_id','location','center','rotation','extent']} if aid in boxes else None for aid in ids}})
        steps=[]
        for left,right in zip(raws[:-1],raws[1:]):
            delta=np.asarray(right.ego_box['location'])-left.ego_box['location']
            yaw=np.radians(right.ego_box['rotation'][2]-left.ego_box['rotation'][2]);yaw=(yaw+np.pi)%(2*np.pi)-np.pi
            steps.append({'from':left.ref.frame_id,'to':right.ref.frame_id,'ego_xyz_step_m':float(np.linalg.norm(delta)),
                'ego_xy_step_m':float(np.linalg.norm(delta[:2])),'ego_yaw_step_rad':float(abs(yaw)),
                'same_ego_id':str(left.ego_box.get('id'))==str(right.ego_box.get('id'))})
        actor_steps={}
        for aid in ids:
            entries=[(r['frame'],r['affected_actors'][aid]) for r in context if r['affected_actors'][aid] is not None]
            jumps=[]
            for (f,b),(nf,nb) in zip(entries[:-1],entries[1:]):
                jumps.append({'from':f,'to':nf,'xyz_step_m':float(np.linalg.norm(np.asarray(nb['location'])-b['location']))})
            actor_steps[aid]={'present_frames':[f for f,b in entries],'steps':jumps}
        events.append({'route':route,'flag_frame':fid,'affected_windows':affected,
            'source_teleport_flag':a.get('teleport_flag'),'source_teleport_ids':ids,
            'ego_id':ego_id,'ego_listed_in_teleport_ids':ego_id in ids,'teleport_ids_nonempty':bool(ids),
            'context':context,'ego_steps':steps,'affected_actor_steps':actor_steps,
            'decision':'blocked_source_semantics','reason':'Actor IDs and continuity are evidence; flag emitter/reset semantics and exact collection pipeline still required.'})
    repo=Path('/home/aethelind/work/environment-build_test/Bench2Drive')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    terms=['teleport_flag','teleport_ids','disappear_flag','disappear_ids']
    search=subprocess.run(['git','-C',str(repo),'grep','-n']+sum((['-e',t] for t in terms),[])+['HEAD','--'],text=True,capture_output=True)
    assert search.returncode in (0,1)
    files={}
    for rel in ['tools/data_collect.py','docs/v004_update.md','docs/anno.md']:
        path=repo/rel
        files[rel]={'path':str(path),'sha256':file_digest(path),
            'tracked_diff':subprocess.run(['git','-C',str(repo),'diff','--name-only','HEAD','--',rel],capture_output=True,text=True).stdout.strip(),
            'primary_url':f'https://github.com/Thinklab-SJTU/Bench2Drive/blob/{commit}/{rel}'}
    result={'window_count':len({w['sample_id'] for e in events for w in e['affected_windows']}),'route_flag_event_count':len(events),
        'route_count':len(readers),'ego_in_teleport_ids_events':sum(e['ego_listed_in_teleport_ids'] for e in events),
        'all_ids_nonempty':all(e['teleport_ids_nonempty'] for e in events),
        'explicit_time_fields_present':any(r['explicit_time_fields'] for e in events for r in e['context']),
        'collector_trace':{'local_repo':str(repo),'commit':commit,'files':files,'tracked_flag_search_matches':search.stdout,
            'conclusion':'The public 0.0.4 checkout collector does not emit these four fields; the actual flag-generating collection/postprocessing implementation is not established.'},
        'events':events,'rejection_policy_changed':False,
        'next_evidence':['Versioned flag-producing collector or postprocessor and its reset lifetime.',
            'Whether the mark refers only to listed actors or also to ego/sensor/time reset.',
            'Independent timestamps or authoritative cadence and synchronized sensor evidence around each event.',
            'Per-head effects: real ego future vs dynamic actor/map labels; any future rule requires an independent contract.']}
    durable_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['events','collector_trace','next_evidence']}))


if __name__=='__main__':main()
