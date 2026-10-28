"""Independent repair candidates for frozen Route Preview truncations.

Only a ``RECONSTRUCTION_STOP`` row with a single certified map branch is
allowed to produce a selected repair path.  Multiple map branches are reported
as candidates with their geometry and preview status, but no branch is
selected.  The tool never interpolates a straight line over a break, never
rewrites the frozen trial, and never touches the active full-generation output.
"""
import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.b2d_adapter.experimental_route_preview import RoutePolyline  # noqa: E402
from tools.b2d_adapter.route_preview_contract import preview_command4  # noqa: E402
from tools.validation.b2d.route_end_classifier import load_jsonl  # noqa: E402
from tools.validation.b2d.route_truncation_root_cause import (  # noqa: E402
    DEFAULT_COMPACT, DEFAULT_SOURCE, DEFAULT_TRIAL, branch_extension,
    load_route_state, sha256,
)


DEFAULT_ROOT_CAUSE = (
    ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/'
    '20261022_truncation_root_cause/route_truncation_root_cause.jsonl'
)
DEFAULT_TRIAL_FRAMES = (
    ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/'
    '20261022_status_upstream/capacity_trial_frames.jsonl'
)
MAX_PROJECTION_M = 5.0


def _branch_by_id(row, branch_id):
    if branch_id is None:
        return None
    return next((item for item in row.get('branches', [])
                 if int(item['normal_center_id']) == int(branch_id)), None)


def _candidate_preview(town, segment, branch, pose):
    extension = branch_extension(town, segment, branch)
    if extension is None or not extension['valid_join']:
        return {
            'geometry_valid': False,
            'reason': 'branch_join_gap_exceeds_limit',
            'join_gaps_m': [] if extension is None else extension['join_gaps_m'],
            'max_join_gap_m': None if extension is None else extension['max_join_gap_m'],
            'preview_valid': False,
        }
    path = RoutePolyline(extension['points_xy'], source='repair_candidate', dense=True)
    preview = preview_command4(path, pose[:2], pose[2], max_projection_m=MAX_PROJECTION_M)
    return {
        'geometry_valid': True,
        'join_gaps_m': [round(float(value), 6) for value in extension['join_gaps_m']],
        'max_join_gap_m': round(float(extension['max_join_gap_m']), 6),
        'candidate_path_length_m': round(float(path.length_m), 6),
        'preview_valid': bool(preview.get('preview_valid')),
        'category': preview.get('category'),
        'reason': preview.get('reason'),
        'command4': preview.get('command4'),
        'route_progress_m': None if preview.get('route_progress_m') is None
        else round(float(preview['route_progress_m']), 6),
        'remaining_path_m': None if preview.get('route_progress_m') is None
        else round(float(path.length_m - preview['route_progress_m']), 6),
        'projection_distance_m': preview.get('projection_distance_m'),
        'lateral_m': preview.get('lateral_m'),
        'longitudinal_m': preview.get('longitudinal_m'),
    }


def build_repair_row(row, state):
    route = row['route']
    frame = int(row['frame'])
    town = state['town']
    index = state['frame_index'].get(frame)
    segment = next((item for item in state['segments']
                    if int(item['start_frame']) <= frame <= int(item['end_frame'])), None)
    base = {
        'schema': 'route-truncation-repair-candidate-v1',
        'route': route,
        'frame': frame,
        'route_end_class': row.get('route_end_class'),
        'truncation_subtype': row.get('truncation_subtype'),
        'source_last_frame': row.get('source_last_frame'),
        'candidate_status': 'NO_CERTIFIED_REPAIR',
        'certified': False,
        'recovered': False,
        'selected_branch_id': None,
        'selected_preview': None,
        'branch_candidates': [],
        'crosses_unverified_break': False,
        'source_unchanged': True,
    }
    if index is None or segment is None:
        base.update(candidate_status='UNRESOLVED_MISSING_SEGMENT_OR_FRAME',
                    reason='missing_current_frame_or_segment')
        return base
    pose = state['poses'][index]
    branches = row.get('branches', [])
    subtype = row.get('truncation_subtype')
    if subtype == 'RECONSTRUCTION_STOP':
        selected_id = row.get('repair_selected_branch_id')
        selected = _branch_by_id(row, selected_id)
        if selected is None:
            base.update(candidate_status='NO_SELECTED_BRANCH',
                        reason='reconstruction_stop_without_selected_branch')
            return base
        preview = _candidate_preview(town, segment, selected, pose)
        base.update(
            candidate_status='RECONSTRUCTION_STOP_REPAIR',
            certified=bool(preview['geometry_valid'] and preview['preview_valid']),
            recovered=bool(preview['geometry_valid'] and preview['preview_valid']),
            selected_branch_id=int(selected['normal_center_id']),
            selected_preview=preview,
            reason=None if preview['preview_valid'] else preview.get('reason'),
            crosses_unverified_break=False,
        )
        return base
    if subtype == 'MAP_MATCH_AMBIGUOUS':
        candidates = []
        for branch in branches:
            preview = _candidate_preview(town, segment, branch, pose)
            candidates.append({
                'branch_id': int(branch['normal_center_id']),
                'road': int(branch['road']),
                'lane': int(branch['lane']),
                'path_center_ids': [int(value) for value in branch['path_center_ids']],
                'selected': False,
                **preview,
            })
        base.update(
            candidate_status='CANDIDATES_ONLY_AMBIGUOUS',
            reason=row.get('subtype_reason'),
            branch_candidates=candidates,
            crosses_unverified_break=False,
        )
        return base
    if subtype == 'TOPOLOGY_LINK_MISSING':
        base.update(candidate_status='NO_TOPOLOGY_BRANCH',
                    reason=row.get('subtype_reason'))
        return base
    if subtype == 'SOURCE_DISCONTINUITY':
        base.update(candidate_status='NO_REPAIR_SOURCE_DISCONTINUITY',
                    reason=row.get('subtype_reason'))
        return base
    base.update(candidate_status='UNRESOLVED_NO_EVIDENCE',
                reason=row.get('subtype_reason'))
    return base


def summarize(rows, root_cause_path, trial_frames_path):
    trial_rows = load_jsonl(trial_frames_path) if trial_frames_path.is_file() else []
    trial_total = len(trial_rows)
    trial_valid = sum(bool(row.get('preview_valid')) for row in trial_rows)
    recovered = sum(bool(row.get('recovered')) for row in rows)
    certified = sum(bool(row.get('certified')) for row in rows)
    candidate_only = sum(bool(row.get('branch_candidates')) for row in rows)
    selected = sum(row.get('selected_branch_id') is not None for row in rows)
    status_counts = Counter(row.get('candidate_status') for row in rows)
    subtype_counts = Counter(row.get('truncation_subtype') for row in rows)
    branch_counts = Counter(len(row.get('branch_candidates', [])) for row in rows)
    branch_preview_valid = sum(
        any(candidate.get('preview_valid') for candidate in row.get('branch_candidates', []))
        for row in rows)
    branch_candidates = [candidate for row in rows
                         for candidate in row.get('branch_candidates', [])]
    branch_candidate_valid = sum(bool(candidate.get('preview_valid'))
                                 for candidate in branch_candidates)
    branch_command_distribution = Counter(
        candidate.get('category') for candidate in branch_candidates)
    command_disagreement_rows = 0
    all_branch_candidates_valid_rows = 0
    for row in rows:
        candidates = row.get('branch_candidates', [])
        if not candidates:
            continue
        categories = {candidate.get('category') for candidate in candidates}
        if len(categories) > 1:
            command_disagreement_rows += 1
        if all(candidate.get('preview_valid') for candidate in candidates):
            all_branch_candidates_valid_rows += 1
    coverage_before = trial_valid / trial_total if trial_total else None
    coverage_after = ((trial_valid + recovered) / trial_total
                      if trial_total else None)
    return {
        'schema': 'route-truncation-repair-summary-v1',
        'root_cause': str(root_cause_path.resolve()),
        'root_cause_sha256': sha256(root_cause_path),
        'trial_frames': str(trial_frames_path.resolve()),
        'trial_total_frames': trial_total,
        'trial_valid_frames_before': trial_valid,
        'trial_valid_frames_after_certified_repair': trial_valid + recovered,
        'coverage_before': coverage_before,
        'coverage_after_certified_repair': coverage_after,
        'coverage_delta': None if coverage_before is None else coverage_after - coverage_before,
        'truncation_rows': len(rows),
        'subtype_counts': dict(subtype_counts),
        'candidate_status_counts': dict(status_counts),
        'selected_branch_rows': selected,
        'certified_repair_rows': certified,
        'certified_recoverable_rows': certified,
        'recovered_rows': recovered,
        'successfully_recovered_rows': recovered,
        'uncertified_candidate_recoverable_rows': branch_preview_valid,
        'ambiguous_candidate_rows': candidate_only,
        'ambiguous_rows_with_at_least_one_preview_valid': branch_preview_valid,
        'branch_candidate_total': len(branch_candidates),
        'branch_candidate_preview_valid': branch_candidate_valid,
        'branch_candidate_command_distribution': dict(branch_command_distribution),
        'rows_with_branch_command_disagreement': command_disagreement_rows,
        'rows_all_branch_candidates_valid': all_branch_candidates_valid_rows,
        'branch_candidate_count_distribution': dict(branch_counts),
        'unresolved_training_rows': len(rows) - recovered,
        'crosses_unverified_break_rows': sum(bool(row.get('crosses_unverified_break')) for row in rows),
        'source_unchanged_rows': sum(bool(row.get('source_unchanged')) for row in rows),
        'rule': {
            'selected_repair': 'only RECONSTRUCTION_STOP with one certified branch',
            'ambiguous': 'branches are reported with preview status, but never selected',
            'no_cross_break': 'no straight-line interpolation and no branch with join gap > 1 m',
        },
        'source_boundary': 'Frozen trial only; no active generation, cache, model, checkpoint or training writes.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-cause', type=Path, default=DEFAULT_ROOT_CAUSE)
    parser.add_argument('--trial-root', type=Path, default=DEFAULT_TRIAL)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--compact-dir', type=Path, default=DEFAULT_COMPACT)
    parser.add_argument('--trial-frames', type=Path, default=DEFAULT_TRIAL_FRAMES)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output_files = (args.output / 'route_truncation_repair_candidates.jsonl',
                    args.output / 'route_truncation_repair_summary.json')
    if any(path.exists() for path in output_files):
        raise FileExistsError('Repair output files already exist; use a fresh directory')
    root_rows = load_jsonl(args.root_cause)
    if not root_rows:
        raise ValueError('root-cause JSONL is empty')
    args.output.mkdir(parents=True, exist_ok=True)
    cache = {}
    rows = []
    for route, group in itertools.groupby(
            sorted(root_rows, key=lambda value: value['route']),
            key=lambda value: value['route']):
        group = list(group)
        state = load_route_state(route, args.source_root, args.compact_dir,
                                 args.trial_root, cache)
        for row in group:
            rows.append(build_repair_row(row, state))
    summary = summarize(rows, args.root_cause, args.trial_frames)
    with (args.output / 'route_truncation_repair_candidates.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    (args.output / 'route_truncation_repair_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'truncation_rows': summary['truncation_rows'],
        'subtype_counts': summary['subtype_counts'],
        'candidate_status_counts': summary['candidate_status_counts'],
        'certified_repair_rows': summary['certified_repair_rows'],
        'recovered_rows': summary['recovered_rows'],
        'ambiguous_rows_with_at_least_one_preview_valid':
            summary['ambiguous_rows_with_at_least_one_preview_valid'],
        'coverage_before': summary['coverage_before'],
        'coverage_after_certified_repair': summary['coverage_after_certified_repair'],
        'output': str(args.output),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
