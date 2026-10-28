"""Frozen full Route Preview audit v1; writes only an independent evidence root.

Reuses the existing failure and truncation classifiers. Frame-index segment
bounds are explicitly converted through readable source frame IDs. The old
20-frame tolerance is retained only as a diagnostic; sequence-end acceptance
uses measured ego arc length and the actual source boundary. No navigation
certification is inferred from complete offline expert trajectories.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.b2d_adapter.experimental_route_navigation import (  # noqa: E402
    CompactMap, _normal_runs, build_segments, match_poses,
)
from tools.b2d_adapter.route_preview_contract import preview_command4  # noqa: E402
from tools.b2d_adapter.experimental_route_preview import RoutePolyline  # noqa: E402
from tools.validation.b2d.route_end_classifier import classify_failure, source_frames  # noqa: E402
from tools.validation.b2d.route_truncation_root_cause import build_row, sha256  # noqa: E402
from tools.validation.b2d.route_navigation_preview import read_route, town_of  # noqa: E402

VERSION = 'full-route-preview-audit-v1'
TOWNS = {}
SHARD_FILES = ('frame_route_preview.parquet', 'route_segment_manifest.jsonl',
               'manifest.json', 'DONE.json', 'failures.jsonl')


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def snapshot(root):
    """Content hashes, sizes and modification times of every generation file."""
    return {str(path.relative_to(root)): {'sha256': sha256(path),
            'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            for path in sorted(Path(root).rglob('*')) if path.is_file()}


def segment_frame_bounds(segments, frames):
    converted = []
    for segment in segments:
        start, end = int(segment['start_frame']), int(segment['end_frame'])
        if not (0 <= start <= end < len(frames)):
            raise ValueError('segment_source_index_out_of_bounds')
        converted.append({**segment, 'source_start_index': start, 'source_end_index': end,
                          'start_frame': int(frames[start]), 'end_frame': int(frames[end])})
    return converted


def measured_failure(row, segments, frames, cumulative, continuous):
    """Reuse taxonomy, never promote a 20-frame heuristic to physical evidence."""
    frame = int(row['frame'])
    source_last = frames[-1] if frames else None
    legacy = classify_failure(row, segments, source_last, tolerance=20)[0]
    kind, evidence, segment, following, tail = classify_failure(
        row, segments, source_last, tolerance=0)
    index = frames.index(frame) if frame in frames else None
    remaining = None if index is None else float(cumulative[-1] - cumulative[index])
    final_segment = bool(segment and segment['end_frame'] == source_last and following is None)
    is_end = row.get('failure_reason') in ('local_segment_end', 'route_end_before_preview',
                                         'LOCAL_SEGMENT_END')
    if is_end and final_segment and continuous and remaining is not None and remaining < 20.0:
        kind = 'EXPECTED_SEQUENCE_END'
        evidence = 'final_source_segment_continuous_measured_future_ego_arc_lt_20m'
    elif kind == 'EXPECTED_SEQUENCE_END':
        kind, evidence = 'INTERNAL_SEGMENT_TRUNCATION', 'source_boundary_not_physically_verified'
    return {'route': row['route'], 'frame': frame, 'failure_reason': row.get('failure_reason'),
            'preview_validity': row.get('preview_validity'), 'classification': kind,
            'evidence': evidence, 'source_last_frame': source_last, 'tail_frames': tail,
            'future_ego_arc_m': remaining, 'source_continuous': continuous,
            'legacy_20_frame_classification': legacy, 'physical_time_standard_certified': False,
            'segment_id': None if segment is None else segment.get('segment_id'),
            'navigation_certification': 'NOT_VERIFIED'}


def center_repeat_evidence(centers, matches, poses, cumulative, town):
    """Repeated map IDs are candidates, not automatic rejection of real loops."""
    seen, repeats = {}, []
    observed = [(index, int(match.center_id)) for index, match in enumerate(matches)
                if match is not None]
    cursor = 0
    occurrence = []
    for center in centers:
        hit = next(((index, cid) for index, cid in observed
                    if index >= cursor and cid == center), None)
        occurrence.append(None if hit is None else hit[0])
        if hit is not None:
            cursor = hit[0] + 1
    for index, center in enumerate(centers):
        if center in seen:
            earlier = seen[center]
            a, b = occurrence[earlier], occurrence[index]
            travel = None if a is None or b is None else float(cumulative[b] - cumulative[a])
            loop_length = 0.0
            for cid in centers[earlier:index]:
                points = town.center_points(cid)[:, :2]
                loop_length += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            # Both projected ego endpoints can differ by up to the 5 m
            # generation projection threshold. This is an evidence bound,
            # not an invented physical time or loop-length standard.
            supported = travel is not None and travel + 10.0 >= loop_length
            repeats.append({'center_id': center, 'from_occurrence': earlier,
                            'to_occurrence': index, 'a_b_a': index == earlier + 2,
                            'map_loop_arc_m': loop_length, 'observed_ego_arc_m': travel,
                            'assessment': 'OBSERVED_TRAVEL_SUPPORTS_LOOP' if supported
                            else 'NOT_VERIFIED_REPEATED_CENTER',
                            'actual_frame_indices': [a, b]})
        seen[center] = index
    return repeats


def audit_route(job):
    route, root_s, source_s, compact_s = job
    root, source, compact = map(Path, (root_s, source_s, compact_s))
    shard = root / 'routes' / route
    issues = []
    for filename in SHARD_FILES:
        if not (shard / filename).is_file():
            raise FileNotFoundError(str(shard / filename))
    manifest = json.loads((shard / 'manifest.json').read_text())
    done = json.loads((shard / 'DONE.json').read_text())
    rows = pd.read_parquet(shard / 'frame_route_preview.parquet').to_dict('records')
    stored_segments = jsonl(shard / 'route_segment_manifest.jsonl')
    frames, poses, z_values, records = read_route(source / route)
    raw_frames = source_frames(source, route)
    if [int(row['frame']) for row in rows] != raw_frames:
        issues.append('frame_list_not_equal_source_annotations')
    if len({int(row['frame']) for row in rows}) != len(rows):
        issues.append('duplicate_frame')
    if any(row['route'] != route for row in rows):
        issues.append('row_route_identity_mismatch')
    signature = sha256(source / route / 'result.json')
    if signature != manifest.get('source_signature') or signature != done.get('source_signature'):
        issues.append('source_result_signature_mismatch')
    if done.get('version') != 'route-preview-v1.1' or done.get('route') != route:
        issues.append('done_version_or_route_mismatch')
    valid = sum(bool(row['preview_valid']) for row in rows)
    if valid != manifest['preview_valid'] or len(rows) - valid != manifest['preview_failed']:
        issues.append('route_manifest_preview_count_mismatch')
    if len(raw_frames) != manifest['annotation_files'] or len(frames) != manifest['frames']:
        issues.append('route_manifest_source_count_mismatch')
    if len(stored_segments) != manifest['segment_count']:
        issues.append('route_manifest_segment_count_mismatch')
    town_name = town_of(route)
    if town_name not in TOWNS:
        TOWNS[town_name] = CompactMap(compact / f'{town_name}_centers.npz',
                                    compact / f'{town_name}_center_metadata.json', town_name)
    town = TOWNS[town_name]
    matches = match_poses(town, poses, z_values)
    rebuilt = build_segments(town, _normal_runs(matches), route_name=route)
    if len(rebuilt) != len(stored_segments):
        issues.append('rebuilt_segment_count_mismatch')
    segment_evidence = []
    for old, fresh in zip(stored_segments, rebuilt):
        equal = (old['start_frame'] == fresh['start_frame'] and
                 old['end_frame'] == fresh['end_frame'] and
                 np.array_equal(old['points_xy'], fresh['points_xy']))
        if not equal:
            issues.append(f"rebuilt_segment_geometry_mismatch:{old['segment_id']}")
        points = np.asarray(old['points_xy'], dtype=np.float64)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        computed_length = float(lengths.sum())
        if not np.isfinite(points).all() or np.any(lengths <= 1e-9):
            issues.append(f"invalid_segment_points:{old['segment_id']}")
        if abs(computed_length - old['length_m']) > 1e-5:
            issues.append(f"segment_length_mismatch:{old['segment_id']}")
        if max(old['join_gaps_m'], default=0) > 1.0 + 1e-9:
            issues.append(f"center_join_gap_exceeded:{old['segment_id']}")
        lo, hi = int(old['start_frame']), int(old['end_frame'])
        ego_arc = float(np.linalg.norm(np.diff(np.asarray(poses)[lo:hi + 1, :2], axis=0), axis=1).sum())
        segment_evidence.append({'segment_id': old['segment_id'], 'points': len(points),
                                 'length_m': computed_length, 'ego_arc_m': ego_arc,
                                 'reference_to_ego_ratio': computed_length / ego_arc if ego_arc else None,
                                 'max_point_spacing_m': float(max(lengths, default=0)),
                                 'max_center_join_gap_m': max(old['join_gaps_m'], default=0),
                                 'rebuilt_geometry_equal': equal,
                                 'centers': fresh['centers'],
                                 'center_repeats': []})
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.asarray(poses)[:, :2], axis=0), axis=1))]
    continuous = bool(raw_frames == frames and all(b - a == 1 for a, b in zip(frames, frames[1:]))
                  and not any(item['anno'].get('teleport_flag') for item in records.values())
                  and max(np.diff(cumulative), default=0.0) <= 2.0)
    for evidence, fresh in zip(segment_evidence, rebuilt):
        evidence['center_repeats'] = center_repeat_evidence(fresh['centers'], matches, poses,
                                                          cumulative, town)
    converted = segment_frame_bounds(stored_segments, frames)
    paths = {int(segment['segment_id']): RoutePolyline(segment['points_xy'], VERSION)
             for segment in stored_segments}
    indices = {frame: index for index, frame in enumerate(frames)}
    geometry_ok, mismatches, classifications = 0, [], []
    status_numeric, categories = 0, Counter()
    state = {'town': town, 'frames': frames, 'poses': poses, 'z_values': z_values,
             'records': records, 'matches': matches, 'source_frames': raw_frames,
             'cumulative': cumulative.tolist(), 'frame_index': indices,
             'segments': segment_frame_bounds(rebuilt, frames),
             'reverse_links': {}, 'branch_extension_cache': {}, 'segment_path_cache': {}}
    for row in rows:
        frame = int(row['frame'])
        categories[str(row['preview_category'])] += 1
        sid = row.get('segment_id')
        sid = None if sid is None or (isinstance(sid, float) and math.isnan(sid)) else int(sid)
        if sid in paths and frame in indices:
            pose = poses[indices[frame]]
            recomputed = preview_command4(paths[sid], pose[:2], pose[2], max_projection_m=5.0)
            identical = (bool(row['preview_valid']) == recomputed['preview_valid'] and
                         row['preview_reason'] == recomputed['reason'] and
                         row['preview_category'] == recomputed['category'])
            if row['preview_valid']:
                target = json.loads(row['preview_point_xy_json'])
                identical = identical and np.allclose(target, recomputed['preview_point_xy'], atol=1e-6, rtol=0)
            if not identical:
                mismatches.append(frame)
            elif recomputed['preview_valid']:
                geometry_ok += 1
        if row.get('status8_json') is not None:
            vector = np.asarray(json.loads(row['status8_json']), dtype=np.float32)
            status_numeric += int(vector.shape == (8,) and np.isfinite(vector).all())
        if not row['preview_valid']:
            normalized = {**row, 'failure_reason': row['preview_reason'],
                          'preview_validity': row['preview_status']}
            item = measured_failure(normalized, converted, frames, cumulative, continuous)
            if item['classification'] == 'INTERNAL_SEGMENT_TRUNCATION':
                detail = build_row(normalized, state, item)
                item.update(truncation_subtype=detail['truncation_subtype'],
                            subtype_reason=detail['subtype_reason'],
                            branch_count=detail.get('branch_count'),
                            future_observed_branch_ids=detail.get('future_observed_branch_ids'),
                            source_discontinuity=detail.get('source_discontinuity'),
                            reconstruction_repair_eligible_offline=detail.get('repair_eligible', False),
                            online_route_intent_certified=False)
            classifications.append(item)
    if mismatches:
        issues.append('preview_geometry_recomputation_mismatch')
    return {'route': route, 'frames': len(rows), 'preview_valid': valid,
            'preview_failed': len(rows) - valid, 'geometric_valid': geometry_ok,
            'status_numeric_float32_convertible': status_numeric,
            'navigation_source_certified': 0, 'training_admitted': 0,
            'source_continuous': continuous, 'readable_source_frames': len(frames),
            'source_frame_gaps': [[a, b] for a, b in zip(raw_frames, raw_frames[1:]) if b - a != 1],
            'schema_columns': sorted(rows[0]) if rows else [],
            'issues': issues, 'geometry_mismatch_frames': mismatches,
            'segments': segment_evidence, 'preview_categories': dict(categories),
            'classifications': classifications}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--compact-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.root.resolve() == args.output.resolve() or args.root.resolve() in args.output.resolve().parents:
        raise ValueError('output_must_be_outside_generation_root')
    args.output.mkdir(parents=True, exist_ok=False)
    before = snapshot(args.root)
    dump(args.output / 'generation_sha_before.json', before)
    manifest = jsonl(args.root / 'route_navigation_manifest.jsonl')
    generation = json.loads((args.root / 'generation_manifest.json').read_text())
    routes = [item['route'] for item in manifest]
    if generation['status'] != 'DONE' or len(set(routes)) != len(routes):
        raise ValueError('generation_not_frozen_done_or_duplicate_route_manifest')
    jobs = [(route, str(args.root), str(args.source_root), str(args.compact_root)) for route in routes]
    counts, classes, subtypes, issues, schemas = Counter(), Counter(), Counter(), [], set()
    with (args.output / 'full_route_geometry_audit.jsonl').open('w') as geo, (
            args.output / 'full_preview_failure_classification.jsonl').open('w') as failures:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for number, result in enumerate(pool.map(audit_route, jobs), 1):
                for key in ('frames', 'preview_valid', 'preview_failed', 'geometric_valid',
                            'status_numeric_float32_convertible', 'navigation_source_certified', 'training_admitted'):
                    counts[key] += result[key]
                schemas.add(tuple(result['schema_columns']))
                for item in result.pop('classifications'):
                    failures.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
                    classes[item['classification']] += 1
                    if 'truncation_subtype' in item:
                        subtypes[item['truncation_subtype']] += 1
                if result['issues']:
                    issues.append({'route': result['route'], 'issues': result['issues']})
                for segment in result['segments']:
                    counts['segments'] += 1
                    counts['center_repeat_candidates'] += len(segment['center_repeats'])
                    counts['unsupported_center_repeats'] += sum(
                        row['assessment'].startswith('NOT_VERIFIED') for row in segment['center_repeats'])
                geo.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                geo.flush()
                failures.flush()
                if number % 25 == 0:
                    print(json.dumps({'event': 'audit_progress', 'routes': number, 'total': len(routes),
                                      'frames': counts['frames'], 'issue_routes': len(issues)}), flush=True)
    after = snapshot(args.root)
    dump(args.output / 'generation_sha_after.json', after)
    shard_routes = sorted(path.name for path in (args.root / 'routes').iterdir() if path.is_dir())
    raw_routes = sorted(path.name for path in args.source_root.iterdir() if path.is_dir())
    global_integrity = {
        'generation_done': generation['status'] == 'DONE', 'routes': len(routes),
        'unique_manifest_routes': len(set(routes)), 'route_inventory_equal_source': sorted(routes) == raw_routes,
        'route_inventory_equal_shards': sorted(routes) == shard_routes,
        'quarantine_files': len(list((args.root / 'quarantine').glob('*'))),
        'quarantine_manifest_rows': len(jsonl(args.root / 'quarantine_manifest.jsonl')),
        'generation_failure_rows': len(jsonl(args.root / 'failures.jsonl')),
        'temporary_shards': [name for name in shard_routes if name.startswith('.tmp-') or '.old-' in name],
        'all_file_sha_unchanged': before == after, 'hashed_files': len(before),
        'historical_generator_sha_available': False,
        'historical_generator_sha_status': 'NOT_VERIFIED: generation metadata stores version, not source SHA',
        'schema_variants': len(schemas), 'route_issues': issues}
    summary = {'schema': VERSION, 'generation_root': str(args.root.resolve()),
               'source_root': str(args.source_root.resolve()), 'integrity': global_integrity,
               'counts': dict(counts), 'failure_classification_counts': dict(classes),
               'internal_truncation_subtype_counts': dict(subtypes),
               'rates': {'geometric_coverage': counts['geometric_valid'] / counts['frames'],
                         'navigation_source_certification': 0.0, 'training_admission': 0.0},
               'physical_end_rule': 'final source-covered segment + continuous source + measured future ego arc <20m',
               '20_frame_tolerance': 'legacy diagnostic only; no physical-time certification',
               'source_navigation_certification': 'NOT_VERIFIED: offline complete expert route is not online navigation authority',
               'float32_note': 'JSON vectors are float32-convertible candidates, not typed payload or admission proof',
               'current_code_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in [
                   Path(__file__), ROOT / 'tools/b2d_adapter/experimental_route_navigation.py',
                   ROOT / 'tools/b2d_adapter/route_preview_contract.py',
                   ROOT / 'tools/validation/b2d/route_end_classifier.py',
                   ROOT / 'tools/validation/b2d/route_truncation_root_cause.py']}}
    dump(args.output / 'full_navigation_quality_summary.json', summary)
    header = ('> 建立前提：全量生成DONE后只读审计。解决问题：区分生成完整性、几何及训练认证。\n'
              '> 适用范围：2026-10-23本次1100 Route冻结快照。不包含：修改生成数据、导航来源认证或启动训练。\n'
              '> 当前状态：已执行CPU审计。相关代码：tools/validation/b2d/full_route_preview_audit.py。最后更新：2026-10-23。\n\n')
    (args.output / 'full_route_preview_integrity_report.md').write_text(
        '# Full Route Preview integrity\n\n' + header +
        f'Routes: {len(routes)}; frames: {counts["frames"]}; valid: {counts["preview_valid"]}; failed: {counts["preview_failed"]}.\n\n' +
        f'全部 {len(before)} 个生成文件执行前后 SHA/size/mtime 比对：{before == after}。独立输出，原生成目录未写入。\n\n' +
        '```json\n' + json.dumps(global_integrity, ensure_ascii=False, indent=2) + '\n```\n\n' +
        '生成元数据记录 route-preview-v1.1，但未记录历史源码 SHA；当前源码 SHA 仅证明本次重算实现，不能追认历史生成器身份。隔离环境挂载视图只读；宿主挂载状态由主任务独立核验。\n')
    (args.output / 'full_preview_coverage_report.md').write_text(
        '# Full Preview coverage\n\n' + header +
        f'几何重算有效 {counts["geometric_valid"]}/{counts["frames"]} ({100*counts["geometric_valid"]/counts["frames"]:.6f}%)；来源认证 0；训练准入 0。\n\n' +
        '```json\n' + json.dumps({'failure_classes': dict(classes), 'internal_subtypes': dict(subtypes),
                                  'counts': dict(counts)}, ensure_ascii=False, indent=2) + '\n```\n\n' +
        'EXPECTED_SEQUENCE_END 仅认可真实片段最终参考段覆盖到末帧、源序列连续且实际未来 Ego 累计行程不足20 m。旧20帧容差仅输出诊断，不作为物理时间标准。保守未通过的末端仍留缺陷类，不代表已证实需要补齐。\n\n' +
        'RECONSTRUCTION_STOP 表示离线未来轨迹/唯一拓扑支持继续重建，不等于在线导航路线意图认证。MAP_MATCH_AMBIGUOUS 保留真实路线意图歧义/地图匹配歧义证据，不选择分支。TOPOLOGY_LINK_MISSING 为地图连接缺失；SOURCE_DISCONTINUITY 为原始数据不连续；UNRESOLVED 为其他未认证。\n\n' +
        '全部参考段重算点数、长度、相邻Center几何间隙及相同范围Ego行程；逐帧重新投影并计算20 m Preview。Center重复访问按实际Ego行程提供支持或NOT_VERIFIED，不禁止合法回环。详细证据见 full_route_geometry_audit.jsonl。\n')
    print(json.dumps({'event': 'audit_done', 'counts': dict(counts), 'issues': len(issues),
                      'unchanged': before == after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
