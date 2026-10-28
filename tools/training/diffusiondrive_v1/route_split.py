"""Build a deterministic Route-level 80/10/10 split and derived indexes.

The source Status index, Feature/Target pair index and manifests are frozen.
This tool writes new files in a new evidence directory; it never edits the
source index, source pair index or any tensor cache.

Tail policy: a Route's last ``--tail-frames`` frames may remain in train, but
are excluded from validation/test.  They are not moved into train when the
Route belongs to validation/test, so Route isolation is preserved.
"""
import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'diffusiondrive-v1-route-split-v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def split_routes(routes, seed, train_fraction=0.8, val_fraction=0.1):
    ordered = sorted(routes)
    random.Random(seed).shuffle(ordered)
    total = len(ordered)
    n_train = int(total * train_fraction)
    n_val = int(total * val_fraction)
    if total >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= total:
            n_train = total - 2
    return {
        'train': ordered[:n_train],
        'validation': ordered[n_train:n_train + n_val],
        'test': ordered[n_train + n_val:],
    }


def route_stats(rows):
    counts = Counter(row['split'] for row in rows)
    routes = {split: {row['route'] for row in rows if row['split'] == split}
              for split in sorted(counts)}
    return {'frames': dict(sorted(counts.items())),
            'routes_with_frames': {split: len(value) for split, value in sorted(routes.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-index', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    parser.add_argument('--route-index', type=Path, required=True)
    parser.add_argument('--pair-index', type=Path, required=True)
    parser.add_argument('--pair-manifest', type=Path, required=True)
    parser.add_argument('--feature-registration', type=Path, required=True)
    parser.add_argument('--target-registration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tail-frames', type=int, default=40)
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f'output directory already exists: {output}')
    output.mkdir(parents=True)

    candidate_manifest = json.loads(args.candidate_manifest.read_text())
    if sha256_file(args.candidate_index) != candidate_manifest['index']['sha256']:
        raise SystemExit('candidate index SHA mismatch')
    candidate_rows = read_jsonl(args.candidate_index)
    if len(candidate_rows) != candidate_manifest['counts']['rows']:
        raise SystemExit('candidate row count mismatch')
    sample_ids = [row['sample_id'] for row in candidate_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit('candidate sample_id is not unique')

    route_rows = read_jsonl(args.route_index)
    route_lengths = {}
    for row in route_rows:
        route = row['route']
        if route in route_lengths:
            raise SystemExit(f'duplicate route in route index: {route}')
        # ``validation_frames`` is an admitted/validation subset, not always
        # the physical route tail.  The candidate range is the authoritative
        # observed route range for this split; the route index only supplies
        # the complete Route universe.
        route_lengths[route] = 0
    candidate_routes = {row['route'] for row in candidate_rows}
    missing = sorted(candidate_routes - set(route_lengths))
    if missing:
        raise SystemExit(f'candidate routes missing from route index: {missing[:3]}')
    observed_max_frame = {}
    for row in candidate_rows:
        route = row['route']
        observed_max_frame[route] = max(observed_max_frame.get(route, -1), int(row['frame']))
    for route, frame in observed_max_frame.items():
        route_lengths[route] = frame + 1

    assignment = split_routes(route_lengths, args.seed, args.train_fraction, args.val_fraction)
    route_to_split = {route: split for split, routes in assignment.items() for route in routes}
    if set(route_to_split) != set(route_lengths):
        raise SystemExit('route assignment is incomplete')
    if any(set(assignment[a]) & set(assignment[b])
           for a in assignment for b in assignment if a < b):
        raise SystemExit('route overlap between splits')

    # Rewrite only the split label.  Train keeps the tail; validation/test
    # drop their own tail frames instead of leaking them into train.
    rewritten = []
    tail_excluded = 0
    for row in candidate_rows:
        row = dict(row)
        route = row['route']
        split = route_to_split[route]
        length = route_lengths.get(route, 0)
        is_tail = length > 0 and int(row['frame']) >= max(0, length - args.tail_frames)
        if split in ('validation', 'test') and is_tail:
            row['split'] = 'excluded_route_tail'
            tail_excluded += 1
        else:
            row['split'] = split
        rewritten.append(row)

    with args.pair_index.open('rb') as stream:
        source_index_sha = hashlib.sha256(stream.read()).hexdigest()
    source_manifest = json.loads(args.pair_manifest.read_text())
    if source_index_sha != source_manifest['index_sha256']:
        raise SystemExit('pair index SHA mismatch')
    with sqlite3.connect(args.pair_index.resolve().as_uri() + '?mode=ro', uri=True) as source:
        source.row_factory = sqlite3.Row
        pair_map = {row['sample_id']: dict(row)
                    for row in source.execute('SELECT * FROM pairs')}
    absent = [sample_id for sample_id in sample_ids if sample_id not in pair_map]
    if absent:
        raise SystemExit(f'candidate rows absent from pair index: {absent[:3]}')

    split_index = output / 'status_training_sample_index.jsonl'
    with split_index.open('w') as stream:
        for row in rewritten:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')

    pair_path = output / 'paired_index.sqlite'
    database = sqlite3.connect(pair_path)
    database.execute('PRAGMA synchronous=FULL')
    database.execute('''CREATE TABLE pairs(sample_id TEXT PRIMARY KEY,route TEXT NOT NULL,frame INTEGER NOT NULL,
       split TEXT NOT NULL,target_source TEXT NOT NULL,feature_source TEXT NOT NULL,
       target_sha TEXT NOT NULL,feature_sha TEXT NOT NULL,current_anno_sha TEXT NOT NULL,source_rank INTEGER NOT NULL)''')
    for row in rewritten:
        pair = pair_map[row['sample_id']]
        if (pair['route'] != row['route'] or int(pair['frame']) != int(row['frame'])
                or pair['target_sha'] != row.get('target_sha')
                or pair['feature_sha'] != row.get('feature_sha')):
            raise SystemExit(f'candidate/pair identity mismatch: {row["sample_id"]}')
        database.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?,?,?)',
                         (row['sample_id'], row['route'], int(row['frame']), row['split'],
                          pair['target_source'], pair['feature_source'], pair['target_sha'],
                          pair['feature_sha'], pair['current_anno_sha'], pair['source_rank']))
    database.execute('CREATE INDEX paired_split ON pairs(split,source_rank DESC,route,frame)')
    database.commit()
    pair_counts = dict(database.execute('SELECT split,COUNT(*) FROM pairs GROUP BY split'))
    database.close()

    split_counts = dict(Counter(row['split'] for row in rewritten))
    for split in ('train', 'validation', 'test'):
        if not split_counts.get(split):
            raise SystemExit(f'empty required split: {split}')
    split_routes_with_frames = {
        split: sorted({row['route'] for row in rewritten if row['split'] == split})
        for split in ('train', 'validation', 'test')
    }
    route_lists = {
        'train': sorted(assignment['train']),
        'validation': sorted(assignment['validation']),
        'test': sorted(assignment['test']),
    }
    route_file_names = {'train': 'train_routes.json', 'validation': 'val_routes.json',
                        'test': 'test_routes.json'}
    for name, routes in route_lists.items():
        write_json(output / route_file_names[name], {
            'schema': SCHEMA, 'seed': args.seed, 'split': name,
            'routes': routes, 'route_count': len(routes),
            'routes_with_frames': len(split_routes_with_frames[name]),
        })

    feature_registration = {
        'path': str(args.feature_registration.resolve()),
        'sha256': sha256_file(args.feature_registration),
    }
    target_registration = {
        'path': str(args.target_registration.resolve()),
        'sha256': sha256_file(args.target_registration),
    }
    source_index = {
        'path': str(args.candidate_index.resolve()),
        'sha256': sha256_file(args.candidate_index),
    }
    split_manifest_path = output / 'dataset_split_manifest.json'
    split_manifest = {
        'schema': SCHEMA,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'seed': args.seed,
        'unit': 'route',
        'requested_ratio': {'train': args.train_fraction, 'validation': args.val_fraction,
                            'test': round(1.0 - args.train_fraction - args.val_fraction, 10)},
        'route_universe': len(route_lengths),
        'route_counts': {name: len(routes) for name, routes in route_lists.items()},
        'frames': split_counts,
        'routes_with_frames': {name: len(routes) for name, routes in split_routes_with_frames.items()},
        'tail_policy': {
            'frames': args.tail_frames,
            'train': 'retain route tail',
            'validation_test': 'exclude route tail; do not move into train',
            'excluded_route_tail': tail_excluded,
        },
        'route_overlap': {
            'train_validation': sorted(set(route_lists['train']) & set(route_lists['validation'])),
            'train_test': sorted(set(route_lists['train']) & set(route_lists['test'])),
            'validation_test': sorted(set(route_lists['validation']) & set(route_lists['test'])),
        },
        'sample_ids_unique': True,
        'samples': len(rewritten),
        'sources': {
            'candidate_index': source_index,
            'candidate_manifest': {'path': str(args.candidate_manifest.resolve()),
                                   'sha256': sha256_file(args.candidate_manifest)},
            'route_index': {'path': str(args.route_index.resolve()),
                            'sha256': sha256_file(args.route_index)},
            'pair_index': {'path': str(args.pair_index.resolve()), 'sha256': source_index_sha},
            'pair_manifest': {'path': str(args.pair_manifest.resolve()),
                              'sha256': sha256_file(args.pair_manifest)},
            'feature_registration': feature_registration,
            'target_registration': target_registration,
        },
        'files': {
            'status_index': {'path': str(split_index.resolve()),
                             'sha256': sha256_file(split_index)},
            'paired_index': {'path': str(pair_path.resolve()),
                             'sha256': sha256_file(pair_path)},
            'train_routes': {'path': str((output/'train_routes.json').resolve()),
                             'sha256': sha256_file(output/'train_routes.json')},
            'val_routes': {'path': str((output/'val_routes.json').resolve()),
                           'sha256': sha256_file(output/'val_routes.json')},
            'test_routes': {'path': str((output/'test_routes.json').resolve()),
                            'sha256': sha256_file(output/'test_routes.json')},
        },
    }
    write_json(split_manifest_path, split_manifest)

    status_manifest = dict(candidate_manifest)
    status_manifest['created'] = split_manifest['created']
    status_manifest['counts'] = dict(candidate_manifest['counts'])
    status_manifest['counts']['rows'] = len(rewritten)
    status_manifest['counts']['splits'] = {key: value for key, value in split_counts.items()
                                           if key != 'excluded_route_tail'}
    status_manifest['counts']['routes'] = {
        name: len(routes) for name, routes in split_routes_with_frames.items()
    }
    status_manifest['counts']['excluded_route_tail'] = tail_excluded
    status_manifest['counts']['classes_by_split'] = {
        split: dict(sorted(Counter(row['class4'] for row in rewritten
                                   if row['split'] == split).items()))
        for split in ('train', 'validation', 'test')
    }
    status_manifest['index'] = {'path': str(split_index.resolve()),
                                'sha256': split_manifest['files']['status_index']['sha256']}
    status_manifest['split_policy'] = {
        'unit': 'route', 'seed': args.seed, 'ratios': split_manifest['requested_ratio'],
        'tail_frames': args.tail_frames,
        'tail_rule': 'train retains tail; validation/test tail excluded, never moved to train',
        'test_usage': 'not loaded during training, checkpoint selection or tuning',
    }
    status_manifest['route_split_manifest'] = {
        'path': str(split_manifest_path.resolve()), 'sha256': sha256_file(split_manifest_path)}
    status_manifest['route_overlap_train_validation'] = split_manifest['route_overlap']['train_validation']
    status_manifest['route_overlap_train_test'] = split_manifest['route_overlap']['train_test']
    status_manifest['route_overlap_validation_test'] = split_manifest['route_overlap']['validation_test']
    status_manifest_path = output / 'status_training_dataset_manifest.json'
    write_json(status_manifest_path, status_manifest)

    pair_manifest = dict(source_manifest)
    pair_manifest['derived_from'] = {
        'path': str(args.pair_manifest.resolve()),
        'sha256': sha256_file(args.pair_manifest),
        'purpose': 'Route-level 8:1:1 split; frozen source manifest and tensor caches unchanged',
    }
    pair_manifest['pair_index'] = str(pair_path.resolve())
    pair_manifest['index_sha256'] = split_manifest['files']['paired_index']['sha256']
    pair_manifest['split_counts'] = pair_counts
    pair_manifest['split_policy'] = status_manifest['split_policy']
    pair_manifest['route_split_counts'] = {name: len(routes) for name, routes in route_lists.items()}
    pair_manifest['route_split_manifest'] = {
        'path': str(split_manifest_path.resolve()), 'sha256': sha256_file(split_manifest_path)}
    pair_manifest['feature_storage_registration'] = feature_registration
    pair_manifest['target_storage_registration'] = target_registration
    pair_manifest_path = output / 'paired_manifest.json'
    write_json(pair_manifest_path, pair_manifest)

    report = {
        'schema': SCHEMA,
        'output_dir': str(output),
        'route_counts': split_manifest['route_counts'],
        'frames': split_counts,
        'routes_with_frames': split_manifest['routes_with_frames'],
        'excluded_route_tail': tail_excluded,
        'route_overlap': split_manifest['route_overlap'],
        'files': {
            'split_manifest': {'path': str(split_manifest_path), 'sha256': sha256_file(split_manifest_path)},
            'status_manifest': {'path': str(status_manifest_path), 'sha256': sha256_file(status_manifest_path)},
            'paired_manifest': {'path': str(pair_manifest_path), 'sha256': sha256_file(pair_manifest_path)},
            'status_index': split_manifest['files']['status_index'],
            'paired_index': split_manifest['files']['paired_index'],
        },
    }
    write_json(output / 'split_report.json', report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
