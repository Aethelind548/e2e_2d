"""Build the versioned Status training sample index from frozen assets.

The frozen pair index and the frozen Status Overlay v2 stay untouched.  This tool
only *selects* rows that satisfy the declared filters and writes a new versioned
index/manifest:

* Feature/Target identity exists in the frozen pair index (sample_id + SHAs);
* Status8 is present, float32-convertible, four-element one-hot;
* Motion4 equals the frozen A0 values bitwise;
* preview validity is VALID (no Unknown padding for invalid previews);
* the route is not one of the 12 geometry-defect routes;
* the frame is not inside a confirmed command-mismatch / invalid-preview interval
  (only available for routes with a real session capture);
* the original train/validation route split is preserved.

Rows are labelled ``navigation_source=RECONSTRUCTED_EXPERT_ROUTE`` and
``online_source_certified=false``; this records the experiment assumption, it is
not a source certification.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'status-training-manifest-v1'
INDEX_SCHEMA = 'status-training-sample-index-v1'
MANIFEST_SCHEMA = 'status-training-dataset-manifest-v1'
NAVIGATION_SOURCE = 'RECONSTRUCTED_EXPERT_ROUTE'
EXPERIMENT_SCOPE = 'OFFLINE_ORACLE_ROUTE_IL'
CLASSES = ('Left', 'Straight', 'Right', 'Unknown')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def class_of(status8):
    if not status8 or len(status8) < 4 or 1.0 not in status8[:4]:
        return None
    return CLASSES[status8[:4].index(1.0)]


def town_of(route):
    if '_Town' not in route:
        return None
    value = route.split('_Town', 1)[1].split('_', 1)[0]
    return 'Town' + value


def scenario_of(route):
    return route.split('_Town', 1)[0]


def load_pair_index(path):
    import sqlite3
    connection = sqlite3.connect(f'file:{Path(path)}?mode=ro', uri=True)
    try:
        return {row[0]: {'route': row[1], 'frame': row[2], 'split': row[3],
                         'feature_sha': row[5], 'target_sha': row[4]}
                for row in connection.execute(
                    'SELECT sample_id, route, frame, split, target_sha, feature_sha FROM pairs')}
    finally:
        connection.close()


def load_exclusion_frames(path):
    frames = set()
    routes = set()
    if not path or not Path(path).is_file():
        return frames, routes
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            routes.add(row['route'])
            for sample in row.get('sample_ids') or []:
                frames.add(sample)
    return frames, routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--pair-index', type=Path, required=True)
    parser.add_argument('--defect-exclusion', type=Path, required=True)
    parser.add_argument('--exclusion-intervals', type=Path, default=None)
    parser.add_argument('--dataset-id', default='Bench2Drive-V0.0.4-local-v1')
    parser.add_argument('--version', default='status-training-v1-20261023')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    defects = {row['route'] for row in
               json.loads(args.defect_exclusion.read_text())['defect_routes']}
    excluded_frames, interval_routes = load_exclusion_frames(args.exclusion_intervals)
    pairs = load_pair_index(args.pair_index)

    excluded = Counter()
    rows = []
    seen = set()
    per_class = Counter()
    per_split_class = defaultdict(Counter)
    per_town = Counter()
    per_route = Counter()
    per_split_route = defaultdict(set)
    turn_frames = Counter()
    duration = Counter()

    with args.overlay.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get('sample_identity') or {}
            sample = row.get('sample_id')
            route = identity.get('route')
            frame = identity.get('frame')
            split = identity.get('split')
            status8 = row.get('status8')
            if sample in seen:
                excluded['duplicate_sample'] += 1
                continue
            seen.add(sample)
            pair = pairs.get(sample)
            if pair is None:
                excluded['missing_pair_index'] += 1
                continue
            if pair['route'] != route or pair['frame'] != frame or pair['split'] != split:
                excluded['pair_identity_mismatch'] += 1
                continue
            if route in defects:
                excluded['geometry_defect_route'] += 1
                continue
            if sample in excluded_frames:
                excluded['confirmed_command_or_preview_interval'] += 1
                continue
            if not isinstance(status8, list) or len(status8) != 8:
                excluded['status8_absent'] += 1
                continue
            if sum(status8[:4]) != 1.0 or any(v not in (0.0, 1.0) for v in status8[:4]):
                excluded['navigation_not_one_hot'] += 1
                continue
            a0 = row.get('a0_motion4')
            if not isinstance(a0, list) or len(a0) != 4 or list(status8[4:]) != list(a0):
                excluded['motion4_differs_from_a0'] += 1
                continue
            if row.get('preview_validity') != 'VALID':
                excluded['preview_not_valid'] += 1
                continue
            category = class_of(status8)
            if category is None or category == 'Unknown':
                excluded['navigation_unknown'] += 1
                continue
            record = {
                'sample_id': sample,
                'route': route,
                'frame': frame,
                'split': split,
                'town': town_of(route),
                'scenario': scenario_of(route),
                'class4': category,
                'status8': status8,
                'old_status8': row.get('old_status8'),
                'a0_motion4': a0,
                'preview_category': (row.get('diagnostic_metadata') or {}).get('command_category'),
                'feature_sha': pair['feature_sha'],
                'target_sha': pair['target_sha'],
                'navigation_source': NAVIGATION_SOURCE,
                'online_source_certified': False,
                'experiment_scope': EXPERIMENT_SCOPE,
            }
            rows.append(record)
            per_class[category] += 1
            per_split_class[split][category] += 1
            per_town[town_of(route)] += 1
            per_route[route] += 1
            per_split_route[split].add(route)
            if category in ('Left', 'Right'):
                turn_frames[category] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / 'status_training_sample_index.jsonl'
    index_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    index_sha = sha256_file(index_path)

    split_counts = Counter(row['split'] for row in rows)
    route_overlap = sorted(per_split_route['train'] & per_split_route['validation'])
    manifest = {
        'schema': MANIFEST_SCHEMA,
        'version': args.version,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'dataset_id': args.dataset_id,
        'navigation_source': NAVIGATION_SOURCE,
        'online_source_certified': False,
        'experiment_scope': EXPERIMENT_SCOPE,
        'sources': {
            'overlay': {'path': str(args.overlay.resolve()),
                        'sha256': sha256_file(args.overlay)},
            'pair_index': {'path': str(args.pair_index.resolve()),
                           'sha256': sha256_file(args.pair_index)},
            'defect_exclusion': {'path': str(args.defect_exclusion.resolve()),
                                 'sha256': sha256_file(args.defect_exclusion),
                                 'defect_route_count': len(defects)},
            'exclusion_intervals': (None if not args.exclusion_intervals else {
                'path': str(args.exclusion_intervals.resolve()),
                'sha256': sha256_file(args.exclusion_intervals),
                'routes': len(interval_routes),
                'sample_ids': len(excluded_frames)}),
        },
        'filters': {
            'pair_identity': 'sample_id + route + frame + split must match the frozen pair index',
            'navigation': 'four-element one-hot, Unknown excluded',
            'motion': 'Motion4 bitwise equal to frozen A0',
            'preview': 'preview_validity == VALID',
            'geometry': 'exclude 12 frozen defect routes and confirmed mismatch intervals',
            'split': 'unchanged frozen split; no route in both train and validation',
        },
        'counts': {
            'rows': len(rows),
            'splits': dict(split_counts),
            'routes': {split: len(routes) for split, routes in per_split_route.items()},
            'classes': dict(per_class),
            'classes_by_split': {split: dict(counter) for split, counter in per_split_class.items()},
            'turn_frames': dict(turn_frames),
            'towns': dict(per_town),
        },
        'route_overlap_train_validation': route_overlap,
        'exclusions': dict(excluded),
        'index': {'path': str(index_path.resolve()), 'sha256': index_sha},
        'training_admitted': False,
        'note': 'Offline oracle-route imitation-learning experiment; not an online-source certification.',
    }
    manifest_path = args.output_dir / 'status_training_dataset_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'rows': len(rows), 'splits': dict(split_counts),
                      'routes': manifest['counts']['routes'],
                      'classes': dict(per_class), 'exclusions': dict(excluded),
                      'route_overlap': route_overlap,
                      'index_sha256': index_sha}, sort_keys=True))


if __name__ == '__main__':
    main()
