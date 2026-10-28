"""Read-only Route-end classification for an already frozen trial.

This tool does not inspect or write the live full-generation target.  It
classifies failed trial frames using three independent pieces of evidence:
  * the source driving-clip frame boundary;
  * reconstructed reference-segment frame ranges and break metadata;
  * the failure reason/preview validity from the tracker.

The output is deliberately conservative: a failure near a local segment end is
not called EXPECTED_SEQUENCE_END unless it is also near the source clip end.
"""
import argparse
import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRIAL = ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261022_status_upstream'
DEFAULT_SOURCE = Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/v1')
PREVIEW_M = 20.0
END_FRAME_TOLERANCE = 20


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_frames(source_root, route):
    try:
        return sorted(int(path.name.split('.')[0])
                      for path in (Path(source_root) / route / 'anno').glob('*.json.gz')
                      if path.is_file())
    except OSError:
        return []


def classify_failure(row, segments, source_last, tolerance=END_FRAME_TOLERANCE):
    frame = int(row['frame'])
    reason = str(row.get('failure_reason') or row.get('reason') or '')
    validity = str(row.get('preview_validity') or '')
    segment = next((item for item in segments
                    if int(item['start_frame']) <= frame <= int(item['end_frame'])), None)
    next_segment = next((item for item in segments
                         if int(item['start_frame']) > frame), None)
    tail = (source_last - frame) if source_last is not None else None
    if 'AMBIGUOUS' in validity.upper() or 'AMBIGUOUS' in reason.upper() or 'candidate_disagreement' in reason:
        return 'AMBIGUOUS_TOPOLOGY', 'ambiguous_preview_or_candidate', segment, next_segment, tail
    if source_last is not None and frame >= source_last - tolerance:
        if reason in ('local_segment_end', 'route_end_before_preview', 'LOCAL_SEGMENT_END'):
            return 'EXPECTED_SEQUENCE_END', 'within_preview_horizon_of_source_clip_end', segment, next_segment, tail
    if reason in ('local_segment_end', 'route_end_before_preview', 'LOCAL_SEGMENT_END'):
        if segment is not None and next_segment is not None:
            if int(segment['end_frame']) < int(next_segment['start_frame']) - 1:
                return 'INTERNAL_ROUTE_GAP', 'failure_at_segment_boundary_with_following_segment', segment, next_segment, tail
        return 'INTERNAL_SEGMENT_TRUNCATION', 'local_path_end_before_source_clip_end', segment, next_segment, tail
    if segment is None:
        if next_segment is not None:
            return 'INTERNAL_ROUTE_GAP', 'frame_before_next_verified_segment', segment, next_segment, tail
        return 'OTHER_INVALID_GEOMETRY', 'no_segment_and_not_an_end_failure', segment, next_segment, tail
    return 'OTHER_INVALID_GEOMETRY', reason or 'non_end_geometry_failure', segment, next_segment, tail


def classify_rows(rows, segments_by_route, source_last_by_route, tolerance=END_FRAME_TOLERANCE):
    classified = []
    for row in rows:
        if row.get('preview_valid'):
            continue
        route = row['route']
        result, evidence, segment, next_segment, tail = classify_failure(
            row, segments_by_route.get(route, []), source_last_by_route.get(route), tolerance)
        classified.append({
            'route': route, 'frame': int(row['frame']),
            'failure_reason': row.get('failure_reason') or row.get('reason'),
            'preview_validity': row.get('preview_validity'),
            'classification': result, 'evidence': evidence,
            'source_last_frame': source_last_by_route.get(route),
            'tail_frames': tail,
            'segment_id': None if segment is None else segment.get('segment_id'),
            'segment_end_frame': None if segment is None else int(segment['end_frame']),
            'next_segment_start_frame': None if next_segment is None else int(next_segment['start_frame']),
        })
    return classified


def audit_terminal_routes(source_root, output, dataset_id):
    """Audit every original clip, without opening frozen tensor payloads or queues."""
    from tools.b2d_adapter.frame_reader import FrameReader
    from tools.b2d_adapter.config import CandidateConfig
    from tools.b2d_adapter.terminal_policy import POLICY, inspect_terminal, file_sha
    from tools.training.diffusiondrive_v1.target_jobs import AnnotationIndex
    from tools.training.diffusiondrive_v1.features_cache import sample_id
    geometry = CandidateConfig()
    horizon = geometry.stride * geometry.trajectory_count
    routes = sorted(p for p in source_root.iterdir() if p.is_dir() and (p/'anno').is_dir())
    if not routes:
        raise ValueError('No original Route annotations found')
    output.mkdir(parents=True, exist_ok=False)
    details, records, frame_rows = [], [], []
    with (output/'route_terminal_frames.jsonl').open('x') as frames_file:
        for number, route in enumerate(routes, 1):
            index = None
            try:
                index = AnnotationIndex(route)
                detail = inspect_terminal(FrameReader(index), geometry)
                missing = [f for f in index.frame_ids if f+horizon > index.frame_ids[-1]]
                row = dict(route=route.name, category=detail['category'], reason=detail['reason'],
                           status=detail['record'].get('status'), entry_status=detail['entry_status'],
                           termination_reason=';'.join(
                               k+':'+json.dumps(detail['record'].get('infractions', {}).get(k))
                               for k in ('route_timeout', 'vehicle_blocked', 'route_dev', 'scenario_timeouts')
                               if detail['record'].get('infractions', {}).get(k)) or 'not_recorded',
                           score_route=detail['record'].get('scores', {}).get('score_route'),
                           duration_game_s=detail['record'].get('meta', {}).get('duration_game'),
                           first_frame=index.frame_ids[0], last_frame=index.frame_ids[-1],
                           annotation_count=len(index.frame_ids), missing_target_frames=len(missing),
                           potential_recoverable_samples=len(missing) if detail['category']=='A' else 0,
                           result_sha256=detail['result_sha256'], **detail['metrics'])
                row['duration_vs_nominal_residual_s'] = row['duration_game_s']-(len(index.frame_ids)+1)*geometry.source_dt_s
                for state in detail['last40']:
                    path = index.files['anno'][state['frame']]
                    sid = sample_id(dataset_id, route.name, state['frame'])
                    frow = dict(route=route.name, sample_id=sid, **state,
                                nominal_offset_from_end_s=(state['frame']-index.frame_ids[-1])*geometry.source_dt_s,
                                annotation_path=str(path), annotation_sha256=file_sha(path))
                    frames_file.write(json.dumps(frow)+'\n')
                    frame_rows.append(frow)
                details.append(detail)
            except Exception as exc:
                row = dict(route=route.name, category='E', reason=type(exc).__name__+':'+str(exc),
                           missing_target_frames=sum(f+horizon > index.frame_ids[-1] for f in index.frame_ids) if index else None,
                           potential_recoverable_samples=0)
            records.append(row)
            if number % 100 == 0:
                print(json.dumps(dict(phase='terminal_audit', routes=number)), flush=True)
    fields = sorted(set().union(*(r.keys() for r in records)))
    with (output/'route_terminal_audit.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    (output/'route_terminal_details.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
    summary = dict(schema='b2d-terminal-audit-v1', source_root=str(source_root.resolve()),
                   dataset_id=dataset_id, policy=POLICY, route_count=len(records),
                   categories={k: dict(routes=sum(r['category']==k for r in records),
                       missing_target_frames=sum((r.get('missing_target_frames') or 0) for r in records if r['category']==k),
                       potential_recoverable_samples=sum(r['potential_recoverable_samples'] for r in records if r['category']==k)) for k in 'ABCDE'},
                   original_status_counts=dict(Counter(r.get('status','READ_ERROR') for r in records)),
                   sampling=dict(source_dt_s_engineering=geometry.source_dt_s, stride=geometry.stride,
                                 num_poses=geometry.trajectory_count, interval_s=geometry.target_dt_s,
                                 required_future_source_frames=horizon, target_horizon_s=4.0),
                   tail_frame_rows=len(frame_rows), observed_timestamps=sum(r['timestamp_s'] is not None for r in frame_rows),
                   clock_statement='40 frames contain 39 intervals: nominal 3.9s; physical timestamps absent means true elapsed time UNVERIFIED.',
                   scope='Raw annotation future-horizon deficits, NOT verified full Feature/Agent/BEV training pairs.')
    (output/'route_terminal_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    # This task performs a small pilot only. All non-A attempts must reject.
    pilot_terminal_targets(output, details, geometry, dataset_id)
    return summary


def pilot_terminal_targets(output, details, geometry, dataset_id):
    """Use the existing V1 trajectory path; a zero-eligible pilot stays empty."""
    import torch
    import numpy as np
    import sys
    sys.path.insert(0, str(ROOT/'repositories/DiffusionDrive'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.b2d_adapter.frame_reader import FrameReader
    from tools.b2d_adapter.coordinate_transform import ActorPose, target_relative
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1TargetBuilder, v1_native_config, official_v1_module
    from tools.training.diffusiondrive_v1.target_jobs import AnnotationIndex
    from tools.training.diffusiondrive_v1.features_cache import sample_id
    from tools.b2d_adapter.errors import AdapterError
    derived = output/'derived_targets'
    derived.mkdir()
    chosen = []
    # Include the near-zero Completed endpoint, a failed stopped clip, and motion.
    for kind in ('A', 'B', 'E', 'D', 'C'):
        matches = [r for r in details if r['category']==kind]
        chosen.extend(matches[:2 if kind=='A' else 1])
    accepted, attempts = [], []
    fig, axes = plt.subplots(max(len(chosen), 1), 3, figsize=(16, 4*max(len(chosen), 1)), squeeze=False)
    for rownum, item in enumerate(chosen):
        index = AnnotationIndex(Path(item['result_path']).parent)
        builder = DiffusionDriveV1TargetBuilder(FrameReader(index), v1_native_config(),
                                              official_v1_module('transfuser_config').TransfuserConfig())
        for offset in (39, 17, 0):
            frame = index.frame_ids[-1]-offset
            ref = index.ref(frame)
            sid = sample_id(dataset_id, item['route'], frame)
            old_reason = None
            try:
                builder.trajectory_builder.build_targets(ref)
            except AdapterError as exc:
                old_reason = exc.code
            try:
                result = builder.trajectory_builder.build_targets(ref, terminal_padding=True)
                value = result.tensors['trajectory']
                if tuple(value.shape)!=(8,3) or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                    raise AssertionError('Invalid derived trajectory')
                relative = sid+'.pt'
                with (derived/relative).open('xb') as stream:
                    torch.save(dict(schema='b2d-terminal-trajectory-derived-v1', sample_id=sid,
                                    targets=result.tensors, provenance=result.provenance,
                                    complete_training_targets=False), stream)
                p = result.provenance['trajectory_source']['terminal_padding']
                accepted.append(dict(sample_id=sid, route=item['route'], frame=frame, relative_path=relative,
                                     real_point_count=p['real_point_count'], padded_point_count=p['padded_point_count']))
                decision = 'accepted_trajectory_only'
            except AdapterError as exc:
                decision = exc.code
                if item['category']=='A':
                    raise
            attempts.append(dict(route=item['route'], category=item['category'], frame=frame,
                                 sample_id=sid, original_builder=old_reason or 'accepted', decision=decision))
        rows = item['last40']
        origin = ActorPose(rows[0]['rear_x_m'], rows[0]['rear_y_m'], rows[0]['rear_yaw_rad'])
        local = np.array([target_relative(origin, ActorPose(r['rear_x_m'],r['rear_y_m'],r['rear_yaw_rad'])) for r in rows])
        t = np.arange(len(rows))*geometry.source_dt_s
        axes[rownum,0].plot(local[:,0],local[:,1],'.-',label='Observed rear-axle poses')
        axes[rownum,0].scatter(local[-1,0],local[-1,1],marker='x',c='red',label='True endpoint')
        axes[rownum,0].set_title(item['category']+' / '+item['route'].split('_Town')[0])
        axes[rownum,0].set_xlabel('forward (m)');axes[rownum,0].set_ylabel('left (m)');axes[rownum,0].axis('equal')
        axes[rownum,1].plot(t,[r['speed_mps'] for r in rows],label='Recorded speed (m/s)')
        axes[rownum,1].plot(t[1:],np.hypot(np.diff(local[:,0]),np.diff(local[:,1]))/geometry.source_dt_s,label='Pose speed / nominal dt')
        axes[rownum,1].set_xlabel('Nominal seconds, physical clock UNVERIFIED')
        axes[rownum,2].plot(t,np.unwrap(local[:,2]),label='Observed relative yaw (rad)')
        axes[rownum,2].set_xlabel('Nominal seconds')
        axes[rownum,2].set_title('Padding '+('eligible' if item['category']=='A' else 'REJECTED; no padded target generated'))
        for ax in axes[rownum]:
            ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'terminal_trajectory_examples.png',dpi=150);plt.close(fig)
    with (derived/'sample_map.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample_id','route','frame','relative_path','real_point_count','padded_point_count'])
        writer.writeheader();writer.writerows(accepted)
    sources = ['terminal_policy.py','trajectory_target.py','temporal_sampler.py','native_builders.py','v1_builders.py']
    manifest = dict(schema='b2d-terminal-trajectory-derived-v1', dataset_id=dataset_id,
                    accepted_targets=len(accepted), complete_training_targets=False,
                    source_sha256={p:hashlib.sha256((ROOT/'tools/b2d_adapter'/p).read_bytes()).hexdigest() for p in sources},
                    sample_map='sample_map.csv', attempts=attempts,
                    registration='Isolated pilot only; no frozen cache, overlay, pair index or split modified')
    (derived/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path,
                        default=DEFAULT_TRIAL / 'capacity_trial_frames.jsonl')
    parser.add_argument('--segments-root', type=Path,
                        default=DEFAULT_TRIAL / 'reconstruction_trial')
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--end-frame-tolerance', type=int, default=END_FRAME_TOLERANCE)
    parser.add_argument('--terminal-source-root', type=Path,
                        help='Audit original terminal states of ALL Routes; independent of preview failures')
    parser.add_argument('--dataset-id', default='Bench2Drive-V0.0.4-local-v1')
    args = parser.parse_args()
    if args.terminal_source_root is not None:
        summary = audit_terminal_routes(args.terminal_source_root, args.output, args.dataset_id)
        print(json.dumps(summary, ensure_ascii=False))
        return
    if args.output.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    args.output.mkdir(parents=True)
    rows = load_jsonl(args.frames)
    routes = sorted({row['route'] for row in rows})
    segments_by_route = {}
    source_last = {}
    for route in routes:
        segment_path = args.segments_root / route / 'reference_segments.jsonl'
        segments_by_route[route] = load_jsonl(segment_path) if segment_path.is_file() else []
        frames = source_frames(args.source_root, route)
        source_last[route] = max(frames) if frames else max(
            (int(row['frame']) for row in rows if row['route'] == route), default=None)
    classified = classify_rows(rows, segments_by_route, source_last,
                               args.end_frame_tolerance)
    (args.output / 'route_end_classification.jsonl').write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in classified) + '\n')
    summary = {
        'schema': 'route-end-classification-v1',
        'source': str(args.frames.resolve()),
        'segments_root': str(args.segments_root.resolve()),
        'routes': len(routes),
        'trial_frames': len(rows),
        'failed_frames': len(classified),
        'classification_counts': dict(Counter(row['classification'] for row in classified)),
        'by_route': {route: dict(Counter(row['classification'] for row in classified
                                         if row['route'] == route)) for route in routes},
        'end_frame_tolerance': args.end_frame_tolerance,
        'source_last_frame_by_route': source_last,
        'boundary_rule': ('EXPECTED_SEQUENCE_END requires both a local end failure and '
                          'frame within tolerance of the source clip end; a local segment '
                          'end elsewhere is INTERNAL_SEGMENT_TRUNCATION.'),
        'source_boundary': 'Frozen trial only; full-generation target not read or modified.',
    }
    (args.output / 'route_end_classification_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    fields = ['route', 'frame', 'failure_reason', 'preview_validity', 'classification',
              'evidence', 'source_last_frame', 'tail_frames', 'segment_id',
              'segment_end_frame', 'next_segment_start_frame']
    with (args.output / 'route_end_classification.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in classified)
    print(json.dumps({'failed_frames': len(classified),
                      'counts': summary['classification_counts'],
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
