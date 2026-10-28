"""CPU-only causal replay of frozen route references and isolated Status overlays.

Raw annotations are read in frame order. Frozen reconstructed references remain
offline, NOT_VERIFIED evidence; this runner never certifies their online lineage.
The optional bounded trial replays full sequences for selected panel routes and
writes only experimental JSONL, without importing a training cache writer.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.b2d_adapter.coordinate_transform import actor_pose, actor_to_reference
from tools.b2d_adapter.experimental_route_progress import ReferenceSegment, RouteProgressTracker
from tools.b2d_adapter.experimental_status_builder import (
    build_status, build_overlay_record, write_status_overlay,
)
from tools.validation.b2d.route_preview_v11 import (
    BASELINE, PANEL, EXAMPLES, dedupe_points, jsonl, recover_candidate,
)

V11 = ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261022_191514_v11_panel'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def failed(reason):
    return dict(preview_valid=False, preview_validity='INVALID', command4=None,
                failure_reason=reason, navigation_source='NOT_VERIFIED')


def verify_panel_pairs(rows, manifest):
    """Bind experiment identities to the original immutable pair index."""
    index = Path(manifest['pair_index'])
    if sha256(index) != manifest['index_sha256']:
        raise ValueError('pair_index_sha_mismatch')
    with sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        db.row_factory = sqlite3.Row
        for row in rows:
            saved = db.execute('SELECT * FROM pairs WHERE sample_id=?', (row['sample_id'],)).fetchone()
            if saved is None or any(saved[key] != value for key, value in row['row'].items()):
                raise ValueError('panel_pair_identity_mismatch:' + row['sample_id'])


def verify_protected_snapshot(snapshot):
    """Compare source/config/index/checkpoint bytes without writing those paths."""
    changed = []
    for filename, expected in snapshot['protected_files'].items():
        path = Path(filename)
        if (not path.is_file() or path.stat().st_size != expected['bytes']
                or sha256(path) != expected['sha256']):
            changed.append(filename)
    return changed


def read_pose(path):
    with gzip.open(path, 'rt') as stream:
        anno = json.load(stream)
    ego = [box for box in anno['bounding_boxes'] if box.get('class') == 'ego_vehicle']
    if len(ego) != 1:
        raise ValueError('ego_count')
    if anno.get('teleport_flag'):
        raise ValueError('teleport_flag')
    pose = actor_pose(ego[0])
    if max(abs(float(v)) for v in ego[0]['rotation'][:2]) > 5:
        raise ValueError('nonplanar_pose')
    rear = actor_to_reference(pose, (-1.388632877, 0.0))
    if not np.isfinite([rear.x, rear.y, rear.yaw]).all():
        raise ValueError('invalid_pose')
    return rear


def segment(points, segment_id, source):
    return ReferenceSegment.from_mapping(dict(
        points_xy=dedupe_points(points), segment_id=segment_id,
        end_kind='LOCAL_SEGMENT_END', navigation_source=source,
    ), navigation_source=source, source_status='NOT_VERIFIED')


def report(path, title, body):
    header = (f'# {title}\n\n'
              '> **建立前提：** Status 上游隔离实验，正式 A0 和八维合同保持冻结。\n'
              '> **解决问题：** 记录因果路线匹配、有效性和准入证据。\n'
              '> **适用范围：** 2026-10-22 固定 81 Route / 192 Frame 面板及指定容量试运行。\n'
              '> **不包含：** 全量覆盖、物理时钟认证、真实在线导航认证或训练。\n'
              '> **当前状态：** 实验审计，不授予训练准入。\n'
              '> **相关代码：** `tools/validation/b2d/status_upstream_audit.py`、'
              '`tools/b2d_adapter/experimental_route_progress.py`、'
              '`tools/b2d_adapter/experimental_status_builder.py`。\n'
              '> **最后更新：** 2026-10-22。\n\n')
    with path.open('x') as stream:
        stream.write(header + body + '\n')


def trailing_motion4(frame_ids, poses, annotations):
    """Use A0 unchanged, rejecting unavailable/discontinuous causal windows."""
    from tools.b2d_adapter.experimental_causal_state import A0, causal_state
    from tools.b2d_adapter.config import CandidateConfig
    limits = CandidateConfig()
    if len(frame_ids) != 3 or list(frame_ids) != list(range(frame_ids[-1]-2,frame_ids[-1]+1)):
        raise ValueError('missing_or_nonconsecutive_motion_history')
    if len(annotations) != 3:
        raise ValueError('missing_motion_annotations')
    for anno in annotations:
        ego = [b for b in anno.get('bounding_boxes',[]) if b.get('class')=='ego_vehicle']
        if len(ego)!=1 or ego[0].get('type_id')!=limits.axle_vehicle_type:
            raise ValueError('motion_reference_vehicle_mismatch')
        if anno.get('teleport_flag') or max(abs(float(v)) for v in ego[0]['rotation'][:2])>limits.max_planar_tilt_deg:
            raise ValueError('motion_history_pose_invalid')
    poses = np.asarray(poses,dtype=np.float64)
    if poses.shape != (3,3) or not np.isfinite(poses).all():
        raise ValueError('invalid_motion_poses')
    headings = np.abs((np.diff(poses[:,2])+np.pi)%(2*np.pi)-np.pi)/.1
    if (np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1)/.1>limits.max_speed_mps).any() or (headings>limits.max_yaw_rate_radps).any():
        raise ValueError('motion_history_discontinuity')
    state = causal_state(poses,method=A0,dt=.1)
    return np.concatenate((state.velocity,state.acceleration)).tolist()


def reconstruction_trial(args):
    """Bounded HD-map reconstruction -> causal tracker -> A0 -> overlay trial.

    Offline reconstruction can see the expert route; that fact is recorded and
    excludes every result from training. Once references are frozen, the status
    calculation receives only the current and trailing poses.
    """
    from tools.b2d_adapter.experimental_route_navigation import CompactMap, reconstruct
    from tools.validation.b2d.route_navigation_preview import read_route, town_of
    from tools.validation.b2d.route_preview_v11 import COMPACT
    panel = json.loads(args.panel.read_text())
    manifest = json.loads(Path(panel['manifest']).read_text())
    verify_panel_pairs([row for split in ('validation','train') for row in panel[split]], manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    names = sorted({row['route'] for split in ('validation','train')
                    for row in panel[split]})[:args.trial_routes]
    if len(names) != args.trial_routes:
        raise ValueError('insufficient_route_count')
    cache, summaries = {}, []
    start = time.monotonic()
    dataset_id = manifest['dataset_id']
    for name in names:
        route_start = time.monotonic()
        town = town_of(name)
        if town not in cache:
            cache[town] = CompactMap(COMPACT/f'{town}_centers.npz',
                                      COMPACT/f'{town}_center_metadata.json', town+'_compact')
        frames, poses, heights, records = read_route(args.source_root/name)
        reconstructed = reconstruct(cache[town], poses, heights, route_name=name)
        segments = reconstructed.get('segments') or []
        trackers = {s['segment_id']: RouteProgressTracker([ReferenceSegment.from_mapping(s)],
                    max_projection_m=5.0) for s in segments}
        output = args.output/name
        output.mkdir()
        with (output/'reference_segments.jsonl').open('x') as stream:
            for item in segments:
                stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
        # Freeze the complete route-reconstruction source identity. These
        # hashes describe offline references, not causal online feature inputs.
        source_paths = sorted((args.source_root/name/'anno').glob('*.json.gz'))
        dump(output/'reference_identity.json',dict(
            source='RECONSTRUCTED_ROUTE_OFFLINE', online_lineage='NOT_VERIFIED',
            expert_future_used_for_reference_reconstruction=True,
            compact_sha256=sha256(COMPACT/f'{town}_centers.npz'),
            compact_metadata_sha256=sha256(COMPACT/f'{town}_center_metadata.json'),
            annotation_sha256={p.name:sha256(p) for p in source_paths}))
        overlays, frame_rows = [], []
        for index, (frame, pose) in enumerate(zip(frames, poses)):
            selected = next((s for s in segments if s['start_frame'] <= index <= s['end_frame']), None)
            try:
                rear = read_pose(args.source_root/name/'anno'/f'{frame:05d}.json.gz')
                preview = (trackers[selected['segment_id']].update((rear.x,rear.y),rear.yaw)
                           if selected else failed('not_in_reference_segment'))
                motion = None
                motion_failure = 'insufficient_motion_history'
                if index >= 2:
                    try:
                        motion = trailing_motion4(frames[index-2:index+1],poses[index-2:index+1],
                            [records[f]['anno'] for f in frames[index-2:index+1]])
                        motion_failure = None
                    except ValueError as exc:
                        motion_failure = str(exc)
            except (OSError, ValueError, KeyError, EOFError) as exc:
                preview, motion = failed(type(exc).__name__+':'+str(exc)), None
                motion_failure = 'current_pose_invalid'
            identity_data = dict(dataset=dataset_id,route=name,frame=int(frame))
            sample_id = hashlib.sha256(json.dumps(identity_data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            valid = bool(preview.get('preview_valid'))
            result = build_status(preview.get('command4'),motion,'VALID' if valid else 'INVALID',
                'NOT_VERIFIED' if motion is not None else 'MISSING',
                preview.get('navigation_source') or 'NOT_VERIFIED',
                dict(sample_id=sample_id,route=name,frame=int(frame)),
                diagnostic_metadata=dict(preview=preview,motion_method='A0',
                    dt_assumption_s=.1,clock_certified=False,causal_motion_frame_ids=list(range(frame-2,frame+1)),
                    motion_failure_reason=motion_failure,
                    future_ego_used_by_status=False,reference_reconstructed_offline=True))
            overlays.append(build_overlay_record(result,None,feature_sha256=None,target_sha256=None,
                feature_producer_sha256=manifest['feature_producer_sha256'],
                target_producer_sha256=manifest['target_sources']['baseline']['producer_sha256'],
                manifest_sha256=panel['manifest_sha256']))
            frame_rows.append(dict(frame=int(frame),**preview))
        # Keep unreadable annotations as explicit failed rows rather than
        # silently dropping them from the trial denominator.
        unavailable = len(source_paths)-len(frames)
        for path in source_paths:
            frame = int(path.name.split('.')[0])
            if frame in records:
                continue
            identity_data = dict(dataset=dataset_id,route=name,frame=frame)
            sample_id = hashlib.sha256(json.dumps(identity_data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            preview = failed('annotation_unreadable_or_invalid')
            result = build_status(None,None,'INVALID','MISSING','NOT_VERIFIED',
                dict(sample_id=sample_id,route=name,frame=frame),diagnostic_metadata=dict(preview=preview))
            overlays.append(build_overlay_record(result,None,feature_sha256=None,target_sha256=None,
                manifest_sha256=panel['manifest_sha256']))
            frame_rows.append(dict(frame=frame,**preview))
        write_status_overlay(output/'status_overlay_manifest.jsonl',overlays)
        summary = dict(route=name,annotation_files=len(source_paths),readable_frames=len(frames),
            unreadable_frames=unavailable,segments=len(segments),
            preview_valid=sum(r.get('preview_valid',False) for r in frame_rows),
            shape_valid_status=sum(r['status8'] is not None for r in overlays),
            failure_counts=dict(Counter(r.get('failure_reason') or 'GEOMETRY_VALID' for r in frame_rows)),
            elapsed_s=time.monotonic()-route_start)
        dump(output/'summary.json',summary)
        summaries.append(summary)
        print(json.dumps(dict(event='reconstruction_trial_route',**summary)),flush=True)
    result = dict(schema='status-upstream-reconstruction-trial-v1',routes=len(summaries),
        annotation_files=sum(r['annotation_files'] for r in summaries),
        readable_frames=sum(r['readable_frames'] for r in summaries),
        unreadable_frames=sum(r['unreadable_frames'] for r in summaries),
        preview_valid=sum(r['preview_valid'] for r in summaries),
        shape_valid_status=sum(r['shape_valid_status'] for r in summaries),
        certified_status=0,training_admitted=0,elapsed_s=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        output_bytes=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
        route_summaries=summaries,source_root=str(args.source_root),
        reference_lineage='NOT_VERIFIED_OFFLINE_EXPERT_RECONSTRUCTION',
        cache_payloads_read=False,full_generation_started=False,
        generator_sources={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
            ROOT/'tools/b2d_adapter/experimental_route_progress.py',
            ROOT/'tools/b2d_adapter/experimental_status_builder.py',
            ROOT/'tools/b2d_adapter/experimental_route_navigation.py']})
    dump(args.output/'summary.json',result)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'status_overlay_manifest.jsonl').exists():
        raise FileExistsError('Use a fresh output directory; overlays are never overwritten')
    panel = json.loads(args.panel.read_text())
    manifest = json.loads(Path(panel['manifest']).read_text())
    if sha256(panel['manifest']) != panel['manifest_sha256']:
        raise ValueError('frozen_manifest_sha_mismatch')
    rows = [dict(row, split=split) for split in ('validation', 'train') for row in panel[split]]
    verify_panel_pairs(rows, manifest)
    by_route = defaultdict(list)
    for row in rows:
        by_route[row['route']].append(row)
    paths = {r['route']: r['points'] for r in jsonl(args.baseline / 'route_reference_paths.jsonl')}
    # V1.1 contains null sample_id on recovered rows. Join the comparison by
    # route/frame, and take the authoritative sample identity from the panel.
    old = {(r['route'], int(r['frame'])): r for r in jsonl(args.v11 / 'frame_route_preview.jsonl')}
    examples = {r['sample_id']: r for r in json.loads(args.examples.read_text())['records']}
    trial_routes = set(sorted(set(paths) & set(by_route))[:args.trial_routes])
    previews, transitions, trial_rows, pose_failures, trial_summaries = {}, [], [], [], []
    start = time.monotonic()
    source_hash_matches = 0
    read_count = 0
    for route, samples in sorted(by_route.items()):
        tracker = None
        if route in paths:
            tracker = RouteProgressTracker([segment(paths[route], 'full:' + route,
                'RECONSTRUCTED_ROUTE_OFFLINE')], max_projection_m=5.0)
        wanted = {int(row['frame']): row for row in samples}
        last = max(wanted)
        prior_command = None
        prior_lateral = None
        prior_scope = None
        trial_start = time.monotonic()
        trial_begin_index = len(trial_rows)
        files = sorted((args.source_root / route / 'anno').glob('*.json.gz'),
                       key=lambda p: int(p.name.split('.')[0]))
        for path in files:
            frame = int(path.name.split('.')[0])
            if frame > last and route not in trial_routes:
                break
            current = wanted.get(frame)
            read_count += 1
            try:
                rear = read_pose(path)
                if current is not None:
                    if sha256(path) != current['row']['current_anno_sha']:
                        raise ValueError('current_annotation_sha_mismatch')
                    source_hash_matches += 1
                preview = tracker.update((rear.x, rear.y), rear.yaw) if tracker else failed('path_unavailable')
                primary_validity = preview.get('preview_validity')
                primary_failure = preview.get('failure_reason')
                # A local candidate remains scoped to this current anchor. It
                # must not replace the frozen full route or invent global s.
                if current is not None and not preview.get('preview_valid'):
                    ex = examples.get(current['sample_id'])
                    candidate, info = recover_candidate(ex) if ex else (None, {})
                    if candidate and candidate.get('preview_valid'):
                        index = info['selected_candidate_index']
                        points = ex['candidates'][index]['points_xyz']
                        local = RouteProgressTracker([segment(
                            [p[:2] for p in points], 'local:' + current['sample_id'],
                            'LOCAL_REFERENCE_SEGMENT')], max_projection_m=5.0)
                        preview = local.update((rear.x, rear.y), rear.yaw)
                        preview['recovery_reason'] = info['recovery_reason']
                        preview['progress_scope'] = 'LOCAL_ANCHOR_ONLY'
                    elif not preview.get('preview_valid'):
                        preview['candidate_failure_reason'] = info.get('recovery_reason')
                        if info.get('recovery_reason') == 'candidate_disagreement_at_20m':
                            preview = dict(failed('candidate_disagreement_at_20m'),
                                           preview_validity='AMBIGUOUS')
                preview['primary_preview_validity'] = primary_validity
                preview['primary_tracker_failure_reason'] = primary_failure
                command = preview.get('command4') if preview.get('preview_valid') else None
                lateral = preview.get('preview_lateral_y')
                scope = preview.get('reference_segment_id')
                if (command is not None and prior_command is not None and scope == prior_scope
                        and command != prior_command):
                    transitions.append(dict(route=route, frame=frame, previous_command4=prior_command,
                        command4=command, previous_lateral_y=prior_lateral, lateral_y=lateral,
                        boundary_near=bool(lateral is not None and abs(abs(lateral)-2.0) <= .25),
                        reason='unmodified_20m_geometry_current_body_yaw',
                        source_status='NOT_VERIFIED'))
                prior_command = command
                prior_lateral = lateral
                prior_scope = scope
            except (OSError, ValueError, KeyError, EOFError) as exc:
                preview = failed(type(exc).__name__ + ':' + str(exc))
                pose_failures.append(dict(route=route, frame=frame, reason=preview['failure_reason']))
                prior_command = None
                prior_scope = None
            if current is not None:
                previews[current['sample_id']] = preview
            if route in trial_routes:
                trial_rows.append(dict(route=route, frame=frame, **preview))
        if route in trial_routes:
            records = trial_rows[trial_begin_index:]
            trial_summaries.append(dict(route=route, frames=len(records),
                valid=sum(r.get('preview_valid',False) for r in records),
                failure_counts=dict(Counter(r.get('failure_reason') or 'GEOMETRY_VALID' for r in records)),
                elapsed_s=time.monotonic()-trial_start))
        print(json.dumps(dict(event='route_replay', route=route, panel_frames=len(samples),
                              trial=route in trial_routes,
                              elapsed_s=round(time.monotonic()-trial_start, 3))), flush=True)
    overlays, frame_rows = [], []
    for row in rows:
        preview = previews.get(row['sample_id'], failed('annotation_frame_missing'))
        valid = bool(preview.get('preview_valid'))
        prior = old[(row['route'], int(row['frame']))]
        old_command = json.loads(prior['command4_json']) if prior.get('command4_json') else None
        result = build_status(preview.get('command4'), row['status'][4:8],
            'VALID' if valid else ('AMBIGUOUS' if 'AMBIGUOUS' in preview.get('preview_validity', '') else 'INVALID'), 'NOT_VERIFIED',
            preview.get('navigation_source') or 'NOT_VERIFIED',
            dict(sample_id=row['sample_id'], route=row['route'], frame=int(row['frame']), split=row['split']),
            navigation_provenance='NOT_VERIFIED', diagnostic_metadata=dict(
                preview=preview, motion_method='A0_FROZEN',
                motion_clock='ASSUMED_0.1S_NOT_CERTIFIED',
                old_feature_status_source='frozen_behavior_panel',
                future_ego_used_by_preview=False,
                frozen_reference_online_lineage='NOT_VERIFIED_OFFLINE_RECONSTRUCTION'))
        overlays.append(build_overlay_record(result, row['status'],
            feature_sha256=row['row']['feature_sha'], target_sha256=row['row']['target_sha'],
            feature_producer_sha256=manifest['feature_producer_sha256'],
            target_producer_sha256=manifest['target_sources']['baseline']['producer_sha256'],
            manifest_sha256=panel['manifest_sha256']))
        frame_rows.append(dict(sample_id=row['sample_id'], route=row['route'], frame=row['frame'],
            primary_class=row['primary_class'], split=row['split'], **preview,
            baseline_v10_valid=prior['baseline_preview_status']=='PASS',
            baseline_v11_valid=bool(prior['preview_valid']),
            command_changed_from_v11=bool(valid and prior['preview_valid'] and preview['command4'] != old_command),
            command_changed_from_near=bool(valid and preview['command4'] != row['status'][:4]),
            shape_valid_status=result.status8 is not None, status_validity=result.status_validity))
    write_status_overlay(args.output / 'status_overlay_manifest.jsonl', overlays)
    for name, values in [('frame_route_preview.jsonl', frame_rows),
                         ('command_transitions.jsonl', transitions), ('capacity_trial_frames.jsonl', trial_rows)]:
        with (args.output / name).open('x') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    counts = Counter(row.get('failure_reason') or 'GEOMETRY_VALID' for row in frame_rows)
    good = sum(row.get('preview_valid', False) for row in frame_rows)
    summary = dict(schema='status-upstream-audit-v1', routes=len(by_route), panel_frames=len(rows),
        preview_generated=sum(row.get('preview_xy_rear') is not None for row in frame_rows),
        preview_geometry_valid=good, source_certified_navigation=0,
        shape_valid_status=sum(r['shape_valid_status'] for r in frame_rows),
        status_valid=sum(r['status_validity']=='VALID' for r in frame_rows),
        status_not_verified=sum(r['status_validity']=='NOT_VERIFIED' for r in frame_rows),
        near_fallback=0, baseline_v10=164, baseline_v11=190,
        delta_from_v10=good-164, delta_from_v11=good-190,
        newly_recovered_from_v10=sum(r.get('preview_valid',False) and not r['baseline_v10_valid'] for r in frame_rows),
        lost_from_v11=sum(not r.get('preview_valid',False) and r['baseline_v11_valid'] for r in frame_rows),
        command_changed_from_v11=sum(r['command_changed_from_v11'] for r in frame_rows),
        command_changed_from_near=sum(r['command_changed_from_near'] for r in frame_rows),
        source_counts=dict(Counter(r.get('navigation_source') for r in frame_rows)),
        failure_counts=dict(counts), current_annotation_sha_matches=source_hash_matches,
        primary_failure_counts=dict(Counter(r.get('primary_tracker_failure_reason') or 'GEOMETRY_VALID'
                                           for r in frame_rows)),
        baseline_v11_null_sample_ids=sum(r.get('sample_id') is None for r in old.values()),
        panel_sha256=sha256(args.panel), manifest_sha256=sha256(panel['manifest']),
        input_reference_sha256=sha256(args.baseline/'route_reference_paths.jsonl'),
        read_annotations=read_count, pose_failures=pose_failures,
        command_transitions=len(transitions), threshold_near_transitions=sum(r['boundary_near'] for r in transitions),
        trial_routes=len(trial_routes), trial_frames=len(trial_rows),
        trial_valid=sum(r.get('preview_valid',False) for r in trial_rows),
        trial_failure_counts=dict(Counter(r.get('failure_reason') or 'GEOMETRY_VALID' for r in trial_rows)),
        trial_route_summaries=trial_summaries,
        trial_elapsed_s=sum(r['elapsed_s'] for r in trial_summaries),
        elapsed_s=time.monotonic()-start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        output_bytes=sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()),
        source_root=str(args.source_root), free_bytes=shutil.disk_usage(args.output).free,
        training_admission='NO_GO', full_generation_started=False,
        limitations=['References reconstructed offline; task-plan lineage NOT_VERIFIED',
                    'No certified physical timestamp; no online CARLA replay',
                    'Trial measures replay/overlay capacity with existing frozen references, not HD-map reconstruction',
                    'No formal cache payload scan; sample SHA lineage inherited from frozen panel'])
    dump(args.output/'summary.json', summary)
    report(args.output/'status_upstream_contract.md', 'Status 上游合同与准入',
        'Status 固定 `float32[8]=[Left,Straight,Right,Unknown,vx,vy,ax,ay]`。正式 A0 不变；A2V 仍为候选。\n\n'
        '几何通过只产生未认证候选；只有导航来源认证、导航及 Motion4 有效性均通过才为 VALID。'
        '本面板全部重建路径来源未认证、物理时钟未认证，因此训练准入为 **NO-GO**。'
        '合法 Unknown 与 INVALID/AMBIGUOUS 失败分离，失败 Status 为 null，不编码默认 Straight/Unknown。'
        '本轮不使用 Near Fallback；显式启用时必须记录 NEAR_FALLBACK。\n\n'
        'Overlay 按 sample_id 关联旧值、Feature/Target SHA、producer 与 Manifest SHA；'
        '每行保留生成器源码 SHA，training_admitted=false。未接入训练 reader，不写正式缓存。')
    report(args.output/'route_progress_audit.md', '路线进度审计',
        f'连续因果重放 {read_count} 份标注；当前面板标注 SHA 匹配 {source_hash_matches}/{len(rows)}。'
        '参考路径提前冻结，路线来源仍为离线重建；匹配只用当前后轴/真实车身 yaw 及历史状态。\n\n'
        '静止噪声、路口等待、蠕行、起步、弯中停车、路口驶出以 CPU 合成回归覆盖；'
        '真实面板不含这六类场景的独立逐帧导航真值，不能宣称真实错误均已修复。'
        '倒车、重定位、重规划显式处理；未认证断裂不直线填补。'
        '只有显式 TASK_END 才认定任务终点；冻结参考路径端点默认为 LOCAL_SEGMENT_END。\n\n'
        f'失败分类：`{json.dumps(dict(counts),ensure_ascii=False)}`；逐帧匹配见 frame_route_preview.jsonl。')
    report(args.output/'command_transition_report.md', 'Command 变化审计',
        f'与 V1.1 同时几何有效帧中命令变化 {summary["command_changed_from_v11"]}；'
        f'与旧 near 命令不同 {summary["command_changed_from_near"]}。'
        f'连续重放检测变化 {len(transitions)} 次，其中距 ±2m 边界 ≤0.25m 的变化 '
        f'{summary["threshold_near_transitions"]} 次。\n\n'
        '未修改 ±2m 分类阈值，未加 command 平滑。逐次记录当前车身姿态和路线预瞄几何的影响；'
        '缺失/失败帧打断相邻比较。局部候选仅在其面板锚点使用，不作为连续可信路线。'
        '所有变化仍需独立 expected-route 真值验证；统计不能解释为误跳率。')
    report(args.output/'preview_coverage_report.md', 'Preview 覆盖与容量试运行',
        f'81 Route / 192 Frame：几何有效 {good}/192，相对 V1.0 164 帧净增 {good-164}，'
        f'相对已有 V1.1 190 帧净增 {good-190}。新恢复 '
        f'{summary["newly_recovered_from_v10"]} 帧，V1.1 成功但本轮拒绝 '
        f'{summary["lost_from_v11"]} 帧。\n\n'
        f'来源分层：`{json.dumps(summary["source_counts"],ensure_ascii=False)}`。'
        f'八维数值可组合 {summary["shape_valid_status"]}；认证导航 0；有效认证 Status 0；'
        'Near Fallback 0；未认证样本不得进入微调。局部恢复沿用唯一拓扑候选或 20m 候选一致，'
        '没有把断裂直线填平，也没有证明这些候选与原始任务意图同源。\n\n'
        f'容量试运行 {len(trial_routes)} Route / {len(trial_rows)} Frame；'
        f'整个集成执行 {summary["elapsed_s"]:.2f}s，峰值 RSS {summary["max_rss_kib"]} KiB。'
        f'试运行单独用时 {summary["trial_elapsed_s"]:.2f}s，几何有效 '
        f'{summary["trial_valid"]}/{summary["trial_frames"]}；未通过帧明确保留失败，不称作全链路质量通过。'
        '使用已有冻结参考路径并重新读取原始标注，测量 tracker/overlay 重放容量；'
        '不替代重新构建 HD Map 路径的全链路试运行。canonical /data 挂载故障；'
        'alternate 源只读，实验输出只写工程 outputs。')
    print(json.dumps({k:summary[k] for k in ('preview_geometry_valid','shape_valid_status','status_valid',
        'delta_from_v10','delta_from_v11','trial_routes','trial_frames','elapsed_s')},ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel', type=Path, default=PANEL)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--examples', type=Path, default=EXAMPLES)
    parser.add_argument('--v11', type=Path, default=V11)
    parser.add_argument('--trial-routes', type=int, default=20)
    parser.add_argument('--reconstruction-trial-only',action='store_true',
                        help='bounded 20-50 Route reconstruction/tracker/A0/overlay trial')
    args = parser.parse_args()
    if not 0 <= args.trial_routes <= 50:
        parser.error('--trial-routes must be in 0..50')
    if args.reconstruction_trial_only:
        if args.trial_routes < 20:
            parser.error('reconstruction trial requires 20-50 routes')
        reconstruction_trial(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
