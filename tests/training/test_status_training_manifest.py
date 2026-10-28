"""CPU tests for the versioned Status training sample index builder."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.training.diffusiondrive_v1 import status_training_manifest as builder

STRAIGHT = [0.0, 1.0, 0.0, 0.0]
LEFT = [1.0, 0.0, 0.0, 0.0]
MOTION = [-0.1, -0.2, -0.3, -0.4]


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def make_pair_index(path, rows):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE pairs (sample_id TEXT, route TEXT, frame INTEGER, split TEXT, '
               'target_sha TEXT, feature_sha TEXT)')
    for row in rows:
        db.execute('INSERT INTO pairs VALUES (?,?,?,?,?,?)',
                   (row['sample_id'], row['route'], row['frame'], row['split'],
                    'target' + row['sample_id'][:6], 'feature' + row['sample_id'][:6]))
    db.commit()
    db.close()


def overlay_row(sample, route, frame, split, status8=STRAIGHT + MOTION, a0=MOTION,
                validity='VALID'):
    return {'sample_id': sample,
            'sample_identity': {'sample_id': sample, 'route': route, 'frame': frame, 'split': split},
            'status8': status8, 'old_status8': status8,
            'a0_motion4': a0, 'preview_validity': validity,
            'diagnostic_metadata': {'command_category': (
                'Left' if status8 and status8[:4] == LEFT else 'Straight')}}


def setup(tmp_path):
    rows = [
        overlay_row('a' * 64, 'Good_Town01_Route1', 1, 'train'),
        overlay_row('b' * 64, 'Good_Town01_Route1', 2, 'train', status8=LEFT + MOTION),
        overlay_row('c' * 64, 'Defect_Town02_Route2', 3, 'train'),
        overlay_row('d' * 64, 'Good_Town01_Route1', 4, 'train', status8=None),
        overlay_row('e' * 64, 'Good_Town01_Route1', 5, 'train', a0=[0.0, 0.0, 0.0, 0.0]),
        overlay_row('f' * 64, 'Good_Town01_Route1', 6, 'train', validity='INVALID'),
        overlay_row('g' * 64, 'Session_Town03_Route3', 7, 'train'),
    ]
    write_jsonl(tmp_path / 'overlay.jsonl', rows)
    make_pair_index(tmp_path / 'pairs.sqlite', [
        {'sample_id': row['sample_id'], 'route': row['sample_identity']['route'],
         'frame': row['sample_identity']['frame'], 'split': row['sample_identity']['split']}
        for row in rows])
    (tmp_path / 'defects.json').write_text(json.dumps(
        {'defect_routes': [{'route': 'Defect_Town02_Route2'}]}))
    write_jsonl(tmp_path / 'intervals.jsonl',
                [{'route': 'Session_Town03_Route3', 'kind': 'COMMAND_MISMATCH',
                  'sample_ids': ['g' * 64]}])
    return rows


def run(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'builder',
        '--overlay', str(tmp_path / 'overlay.jsonl'),
        '--pair-index', str(tmp_path / 'pairs.sqlite'),
        '--defect-exclusion', str(tmp_path / 'defects.json'),
        '--exclusion-intervals', str(tmp_path / 'intervals.jsonl'),
        '--output-dir', str(tmp_path / 'out'),
    ])
    builder.main()


def test_filters_and_counts(tmp_path, monkeypatch, capsys):
    setup(tmp_path)
    run(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (tmp_path / 'out' / 'status_training_sample_index.jsonl').open()]
    assert [row['sample_id'][0] for row in rows] == ['a', 'b']
    assert all(row['navigation_source'] == 'RECONSTRUCTED_EXPERT_ROUTE' for row in rows)
    assert all(row['online_source_certified'] is False for row in rows)
    manifest = json.loads((tmp_path / 'out' / 'status_training_dataset_manifest.json').read_text())
    assert manifest['counts']['rows'] == 2
    assert manifest['exclusions']['geometry_defect_route'] == 1
    assert manifest['exclusions']['confirmed_command_or_preview_interval'] == 1
    assert manifest['exclusions']['status8_absent'] == 1
    assert manifest['exclusions']['motion4_differs_from_a0'] == 1
    assert manifest['exclusions']['preview_not_valid'] == 1
    assert manifest['route_overlap_train_validation'] == []


def test_index_sha_is_pinned(tmp_path, monkeypatch):
    setup(tmp_path)
    run(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / 'out' / 'status_training_dataset_manifest.json').read_text())
    assert manifest['index']['sha256'] == builder.sha256_file(
        tmp_path / 'out' / 'status_training_sample_index.jsonl')
    assert manifest['sources']['overlay']['sha256'] == builder.sha256_file(tmp_path / 'overlay.jsonl')
