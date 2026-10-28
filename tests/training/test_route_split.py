import hashlib
import json
import sqlite3
import sys

from tools.training.diffusiondrive_v1 import route_split


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_split_routes_is_deterministic_and_disjoint():
    routes = {f'Scenario_Town01_Route{i}_Weather0_00-00-00-00-00' for i in range(10)}
    first = route_split.split_routes(routes, 0)
    second = route_split.split_routes(routes, 0)
    assert first == second
    assert [len(first[name]) for name in ('train', 'validation', 'test')] == [8, 1, 1]
    assert not (set(first['train']) & set(first['validation']))
    assert not (set(first['train']) & set(first['test']))
    assert not (set(first['validation']) & set(first['test']))


def test_route_split_keeps_train_tail_and_excludes_eval_tail(tmp_path, monkeypatch):
    routes = [f'Scenario_Town01_Route{i}_Weather0_00-00-00-00-00' for i in range(10)]
    frames = list(range(50))
    rows = []
    pairs = []
    for route in routes:
        for frame in frames:
            sid = hashlib.sha256(f'{route}:{frame}'.encode()).hexdigest()
            rows.append({'sample_id': sid, 'route': route, 'frame': frame, 'split': 'train',
                         'class4': 'Straight', 'target_sha': 't', 'feature_sha': 'f'})
            pairs.append((sid, route, frame, 'train', 'baseline', 'baseline',
                          't', 'f', 'a', 0))
    index_path = tmp_path / 'index.jsonl'
    index_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    pair_index = tmp_path / 'pairs.sqlite'
    with sqlite3.connect(pair_index) as db:
        db.execute('''CREATE TABLE pairs(sample_id TEXT PRIMARY KEY,route TEXT NOT NULL,frame INTEGER NOT NULL,
           split TEXT NOT NULL,target_source TEXT NOT NULL,feature_source TEXT NOT NULL,
           target_sha TEXT NOT NULL,feature_sha TEXT NOT NULL,current_anno_sha TEXT NOT NULL,source_rank INTEGER NOT NULL)''')
        db.executemany('INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?,?,?)', pairs)
        db.commit()
    candidate_manifest = tmp_path / 'candidate.json'
    candidate_manifest.write_text(json.dumps({
        'schema': 'status-training-dataset-manifest-v1', 'version': 'test',
        'navigation_source': 'RECONSTRUCTED_EXPERT_ROUTE', 'online_source_certified': False,
        'counts': {'rows': len(rows), 'splits': {'train': len(rows)}, 'routes': {},
                   'classes_by_split': {}},
        'index': {'sha256': sha(index_path)}, 'sources': {},
    }))
    pair_manifest = tmp_path / 'pair.json'
    pair_manifest.write_text(json.dumps({
        'schema': 'b2d-v1-explicit-recovery-paired-v1', 'index_sha256': sha(pair_index),
        'target_sources': {}, 'feature_sources': {},
    }))
    route_index = tmp_path / 'routes.jsonl'
    route_index.write_text(''.join(json.dumps({'route': route, 'validation_frames': 50}) + '\n'
                                   for route in routes))
    feature_registration = tmp_path / 'feature_registration.json'
    feature_registration.write_text('{}')
    target_registration = tmp_path / 'target_registration.json'
    target_registration.write_text('{}')
    output = tmp_path / 'split'
    monkeypatch.setattr(sys, 'argv', [
        'route_split',
        '--candidate-index', str(index_path),
        '--candidate-manifest', str(candidate_manifest),
        '--route-index', str(route_index),
        '--pair-index', str(pair_index),
        '--pair-manifest', str(pair_manifest),
        '--feature-registration', str(feature_registration),
        '--target-registration', str(target_registration),
        '--output-dir', str(output),
        '--seed', '0', '--tail-frames', '40',
    ])
    route_split.main()
    split = json.loads((output / 'dataset_split_manifest.json').read_text())
    assert split['route_counts'] == {'train': 8, 'validation': 1, 'test': 1}
    assert split['frames'] == {'train': 400, 'validation': 10, 'test': 10,
                               'excluded_route_tail': 80}
    assert split['route_overlap'] == {'train_validation': [], 'train_test': [],
                                      'validation_test': []}
    assert (output / 'val_routes.json').is_file()
    rewritten = [json.loads(line) for line in (output / 'status_training_sample_index.jsonl').read_text().splitlines()]
    assert {row['split'] for row in rewritten} == {'train', 'validation', 'test',
                                                   'excluded_route_tail'}
