"""Check whether the online and offline Navigation4 rules are the same rule.

The leaderboard hands the agent a dense 1 m plan and the agent keeps a
``downsample_route(..., 50)`` view of 3-4 points.  This tool answers three
questions with evidence instead of assumption:

1. Does the dense plan, evaluated with the frozen preview calculator, reproduce
   the offline overlay Command4 at the same rear pose?
2. Does the agent's downsampled view reproduce it?
3. Which calculator parameters (``max_projection_m`` 5.0 vs 10.0) are being
   compared, and how many rows flip only because of that parameter?

It also reports the geometry that drives each decision (rear arc, projection
distance, preview arc, lateral/longitudinal offset, heading delta) so a
disagreement can be attributed to coverage, projection, or a real turn.

Read-only: consumes session captures and comparison rows; writes one JSON
report per run.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from tools.validation.b2d.certification_level_audit import (
    command_of,
    load_sessions,
)
from tools.validation.b2d.route_source_certification import (
    dedupe_points,
    preview_command4,
    project_progress,
)

VERSION = 'navigation-semantics-check-v1'
PREVIEW_M = 20.0
LATERAL_THRESHOLD_M = 2.0


def option_change_view(points, sample_factor=50.0):
    """Approximate the leaderboard rule without CARLA objects.

    The real ``downsample_route`` keeps the first point, lane-change points,
    RoadOption change points, points every ``sample_factor`` metres and the
    last point.  Without RoadOption here only the distance rule and the
    endpoints apply, which is exactly what an all-LANEFOLLOW plan produces.
    """
    if len(points) < 2:
        return list(points)
    kept = [points[0]]
    dist = 0.0
    for prev, current in zip(points, points[1:]):
        dist += math.dist(prev, current)
        if dist > sample_factor:
            kept.append(current)
            dist = 0.0
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


def evaluate(points, rear, yaw, max_projection_m):
    preview = preview_command4(points, rear, yaw, preview_m=PREVIEW_M,
                               lateral_threshold_m=LATERAL_THRESHOLD_M,
                               max_projection_m=max_projection_m)
    details = {
        'status': preview.get('status'),
        'reason': preview.get('reason'),
        'command4': preview.get('command4'),
        'category': preview.get('category'),
        'preview_point_xy': preview.get('preview_point_xy'),
        'lateral_m': preview.get('lateral_m'),
        'longitudinal_m': preview.get('longitudinal_m'),
        'heading_delta_deg': preview.get('heading_delta_deg'),
    }
    if preview.get('status') == 'PASS':
        rear_arc, rear_projection = project_progress(points, rear)
        preview_arc, _ = project_progress(points, preview['preview_point_xy'])
        details['rear_arc_m'] = round(rear_arc, 3)
        details['rear_projection_m'] = round(rear_projection, 3)
        details['preview_arc_m'] = round(preview_arc, 3)
        details['horizon_m'] = round(preview_arc - rear_arc, 3)
    return details


def compare_views(dense, agent, rear, yaw):
    """Compare four candidate rules at one frame."""
    rules = {
        'dense_projection5': evaluate(dense, rear, yaw, 5.0),
        'dense_projection10': evaluate(dense, rear, yaw, 10.0),
        'agent_projection5': evaluate(agent, rear, yaw, 5.0) if len(agent) > 1 else
        {'status': 'FAIL', 'reason': 'agent_view_too_short'},
        'agent_projection10': evaluate(agent, rear, yaw, 10.0) if len(agent) > 1 else
        {'status': 'FAIL', 'reason': 'agent_view_too_short'},
    }
    return rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--session-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.comparison.open() if line.strip()]
    sessions = load_sessions(args.session_dir)
    report, summary = [], Counter()
    per_route = {}

    for row in rows:
        session = sessions.get(row['route'])
        if session is None:
            continue
        dense = session['points']
        agent = session['agent_points']
        rules = compare_views(dense, agent, row['rear_xy'], row['yaw_rad'])
        offline = row.get('offline_command4')
        record = {
            'route': row['route'],
            'frame': row['frame'],
            'sample_id': row['sample_id'],
            'offline_command': command_of(offline),
            'dense_point_count': len(dense),
            'agent_point_count': len(agent),
            'rules': {name: {**value, 'command': command_of(value.get('command4'))}
                      for name, value in rules.items()},
        }
        record['match'] = {name: bool(value.get('command4') == offline and offline is not None)
                           for name, value in rules.items()}
        for name, matched in record['match'].items():
            summary[f'match:{name}'] += int(matched)
            summary[f'total:{name}'] += 1
        summary['frames'] += 1
        per_route.setdefault(row['route'], Counter())[
            'dense5_match'] += int(record['match']['dense_projection5'])
        report.append(record)

    # Frames where the dense plan and the agent view disagree: attribute to geometry.
    disagreements = [r for r in report if r['rules']['dense_projection5'].get('command')
                     != r['rules']['agent_projection5'].get('command')]
    attribution = Counter()
    for record in disagreements:
        dense = record['rules']['dense_projection5']
        agent = record['rules']['agent_projection5']
        if dense.get('status') != 'PASS':
            attribution['dense_preview_invalid'] += 1
        elif agent.get('status') != 'PASS':
            attribution[f'agent_preview_invalid:{agent.get("reason")}'] += 1
        else:
            horizon = dense.get('horizon_m') or 0.0
            lateral = agent.get('lateral_m')
            chord = record['agent_point_count']
            if chord <= 4:
                attribution['agent_chord_shortcut'] += 1
            if abs((dense.get('heading_delta_deg') or 0.0)
                   - (agent.get('heading_delta_deg') or 0.0)) > 20.0:
                attribution['heading_delta_divergence'] += 1
            if horizon and dense.get('preview_arc_m') and agent.get('preview_arc_m'):
                if abs(agent['preview_arc_m'] - dense['preview_arc_m']) > 1.0:
                    attribution['arc_parameterization_differs'] += 1
            if lateral is not None and abs(abs(lateral) - LATERAL_THRESHOLD_M) < 0.5:
                attribution['near_lateral_threshold'] += 1

    parameter_flips = sum(
        1 for r in report
        if r['rules']['dense_projection5'].get('command') != r['rules']['dense_projection10'].get('command'))
    summary['dense_projection_rule_flips'] = parameter_flips
    summary['dense_vs_agent_disagreements'] = len(disagreements)
    summary['route_dense5_matches'] = {route: dict(counter) for route, counter in per_route.items()}
    summary['version'] = VERSION
    summary['attribution'] = dict(attribution)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'navigation_semantics_rows.jsonl').open('w') as stream:
        for record in report:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
    (args.output / 'navigation_semantics_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in (
        'frames', 'match:dense_projection5', 'match:dense_projection10',
        'match:agent_projection5', 'match:agent_projection10',
        'dense_projection_rule_flips', 'dense_vs_agent_disagreements', 'attribution')},
        sort_keys=True))


if __name__ == '__main__':
    main()
