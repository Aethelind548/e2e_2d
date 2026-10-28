"""Turn-focused comparison of several status_route_eval outputs.

Reads `samples.jsonl` from two or more checkpoint evaluation directories, keeps
only frames whose expert future trajectory actually turns (cumulative heading
change across the 8 target waypoints), and reports ADE/FDE/heading per
checkpoint and per route. This is offline open-loop analysis, not CARLA.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def cumulative_turn_deg(target):
    """Sum of absolute heading changes between consecutive target segments."""
    headings = []
    for first, second in zip(target, target[1:]):
        headings.append(math.atan2(second[1] - first[1], second[0] - first[0]))
    total = 0.0
    for previous, current in zip(headings, headings[1:]):
        delta = (current - previous + math.pi) % (2 * math.pi) - math.pi
        total += abs(math.degrees(delta))
    return total


def load_samples(directory):
    path = Path(directory) / 'samples.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row['turn_deg'] = cumulative_turn_deg(row['target'])
    return rows


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float('nan')


def summarize(rows, min_turn):
    kept = [row for row in rows if row['turn_deg'] >= min_turn]
    return {
        'frames': len(kept),
        'ade_m': mean(row['ade_m'] for row in kept),
        'fde_m': mean(row['fde_m'] for row in kept),
        'heading_error_deg': mean(row['heading_error_deg'] for row in kept),
    }


def ensure_same_frames(payload):
    """Only identical frame sets may be compared across checkpoints."""
    reference_name = next(iter(payload))
    reference = {row['sample_id'] for row in payload[reference_name]}
    for name, rows in payload.items():
        if {row['sample_id'] for row in rows} != reference:
            raise SystemExit('sample_id sets differ between %s and %s; rerun the evaluation '
                             'with identical routes/frames before comparing'
                             % (reference_name, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+',
                        help='label=route-eval-directory (label defaults to the directory name)')
    parser.add_argument('--min-turn', type=float, default=60.0,
                        help='minimum cumulative expert turn in degrees (default 60)')
    parser.add_argument('--output', type=Path, default=None,
                        help='optional CSV summary path')
    args = parser.parse_args()

    labelled = []
    for item in args.directories:
        name, _, path = item.partition('=')
        labelled.append((name, Path(path)) if path else (Path(item).name, Path(item)))
    payload = {name: load_samples(path) for name, path in labelled}
    ensure_same_frames(payload)

    first_name = labelled[0][0]
    turns = [row['turn_deg'] for row in payload[first_name]]
    print('frames=%d cumulative-turn %.0f..%.0f deg, threshold >=%.0f'
          % (len(turns), min(turns), max(turns), args.min_turn))

    summary = {name: summarize(rows, args.min_turn) for name, rows in payload.items()}
    print('\ncheckpoint                     n    ADE     FDE   heading')
    for name, stats in summary.items():
        print('%-28s %4d  %5.2f  %6.2f  %6.1f' % (
            name, stats['frames'], stats['ade_m'], stats['fde_m'], stats['heading_error_deg']))

    routes = []
    for row in payload[first_name]:
        if row['route'] not in routes:
            routes.append(row['route'])
    print('\nper route')
    per_route = []
    for route in routes:
        stats = {name: summarize([row for row in rows if row['route'] == route], args.min_turn)
                 for name, rows in payload.items()}
        counts = {values['frames'] for values in stats.values()}
        if len(counts) != 1:
            raise SystemExit('threshold keeps different frame counts per checkpoint on %s' % route)
        print('%-46s %s' % (route[:46], ' | '.join(
            '%5.2f/%6.2f' % (values['ade_m'], values['fde_m']) for values in stats.values())))
        per_route.append({'route': route, 'frames': counts.pop(), **stats})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['route', 'frames'] + [f'{name}_ade_m' for name, _ in labelled]
                            + [f'{name}_fde_m' for name, _ in labelled]
                            + [f'{name}_heading_deg' for name, _ in labelled])
            for entry in per_route:
                writer.writerow([entry['route'], entry['frames']]
                                + [entry[name]['ade_m'] for name, _ in labelled]
                                + [entry[name]['fde_m'] for name, _ in labelled]
                                + [entry[name]['heading_error_deg'] for name, _ in labelled])
        print('\nwrote %s' % args.output)

    return summary


if __name__ == '__main__':
    main()
