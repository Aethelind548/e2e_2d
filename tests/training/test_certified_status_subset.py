"""CPU tests for the certification manifest and certified Status subset builder."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.training.diffusiondrive_v1 import certified_status_subset as builder


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def pair_db(path, rows):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE pairs (sample_id TEXT PRIMARY KEY, route TEXT, frame INTEGER, '
               'split TEXT, feature_sha TEXT, target_sha TEXT)')
    for row in rows:
        db.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?)',
                   (row['sample_id'], row['route'], row['frame'], row['split'], 'f' * 64, 't' * 64))
    db.commit()
    db.close()
    return path


def status_row(sample_id, old, new):
    return {'sample_id': sample_id, 'old_status8': old, 'status8': new}


def test_manifest_marks_missing_capture_as_insufficient(tmp_path):
    comparison = write_jsonl(tmp_path / 'c.jsonl', [{
        'sample_id': 's1', 'route': 'R1', 'frame': 2, 'xml_id': '1', 'xml_source': 'x.xml',
        'xml_preview_status': 'PASS', 'xml_command4': [0, 1, 0, 0], 'offline_command4': [0, 1, 0, 0],
        'old_status_command4': [0, 1, 0, 0], 'new_status_command4': [0, 1, 0, 0],
        'xml_vs_offline_point_distance_m': 1.0, 'xml_horizon_m': 20.0}])
    pairs = {'s1': {'sample_id': 's1', 'route': 'R1', 'frame': 2, 'split': 'train'}}
    out = tmp_path / 'manifest.jsonl'
    result = builder.build_manifest(comparison, pairs, set(), {}, out)
    record = json.loads(out.read_text().splitlines()[0])
    assert result['counts'] == {'INSUFFICIENT_EVIDENCE': 1, 'total': 1}
    assert record['certification_status'] == 'INSUFFICIENT_EVIDENCE'
    assert record['certification_reason'] == 'NO_ONLINE_PLAN_CAPTURED'
    assert record['training_admitted'] is False


def test_manifest_excludes_geometry_defect_routes(tmp_path):
    comparison = write_jsonl(tmp_path / 'c.jsonl', [{
        'sample_id': 's1', 'route': 'BAD', 'frame': 2, 'xml_preview_status': 'PASS'}])
    pairs = {'s1': {'sample_id': 's1', 'route': 'BAD', 'frame': 2, 'split': 'train'}}
    out = tmp_path / 'manifest.jsonl'
    builder.build_manifest(comparison, pairs, {'BAD'}, {'BAD': {'sha256': 'x'}}, out)
    record = json.loads(out.read_text().splitlines()[0])
    assert record['certification_reason'] == 'EXCLUDED_GEOMETRY_DEFECT'


def test_manifest_drops_samples_outside_frozen_pairs(tmp_path):
    comparison = write_jsonl(tmp_path / 'c.jsonl', [{'sample_id': 'ghost', 'route': 'R', 'frame': 1}])
    result = builder.build_manifest(comparison, {}, set(), {}, tmp_path / 'manifest.jsonl')
    assert result['rows'] == 0
    assert result['counts']['not_in_pair_index'] == 1


def test_subset_is_empty_without_verified_rows(tmp_path):
    manifest = write_jsonl(tmp_path / 'manifest.jsonl', [{
        'sample_id': 's1', 'route': 'R1', 'certification_status': 'INSUFFICIENT_EVIDENCE'}])
    overlay = write_jsonl(tmp_path / 'overlay.jsonl', [status_row('s1', [0, 1, 0, 0, 1, 0, 0, 0],
                                                                  [0, 1, 0, 0, 1, 0, 0, 0])])
    pairs = {'s1': {'sample_id': 's1', 'route': 'R1', 'frame': 2, 'split': 'train'}}
    result = builder.build_subset(manifest, pairs, set(), overlay, tmp_path / 'subset.jsonl')
    assert result['verified_rows'] == 0
    assert result['staged_rows'] == 0
    assert result['reader_ready'] is False
    assert (tmp_path / 'subset.jsonl').read_text() == ''


def test_subset_includes_verified_row_with_identical_motion(tmp_path):
    manifest = write_jsonl(tmp_path / 'manifest.jsonl', [{
        'sample_id': 's1', 'route': 'R1', 'certification_status': builder.VERIFIED,
        'online_capture': {'sha256': 'a' * 64}}])
    overlay = write_jsonl(tmp_path / 'overlay.jsonl', [status_row('s1', [0, 1, 0, 0, 1, 0, 0, 0],
                                                                  [1, 0, 0, 0, 1, 0, 0, 0])])
    pairs = {'s1': {'sample_id': 's1', 'route': 'R1', 'frame': 2, 'split': 'train'}}
    result = builder.build_subset(manifest, pairs, set(), overlay, tmp_path / 'subset.jsonl')
    row = json.loads((tmp_path / 'subset.jsonl').read_text().splitlines()[0])
    assert result['verified_rows'] == 1
    assert row['status8'][:4] == [1, 0, 0, 0]
    assert row['a0_motion4'] == [1, 0, 0, 0]
    assert row['staged_only'] is True
    assert row['training_admitted'] is False


def test_subset_rejects_motion_change_and_defect_routes(tmp_path):
    manifest = write_jsonl(tmp_path / 'manifest.jsonl', [
        {'sample_id': 's1', 'route': 'R1', 'certification_status': builder.VERIFIED},
        {'sample_id': 's2', 'route': 'BAD', 'certification_status': builder.VERIFIED}])
    overlay = write_jsonl(tmp_path / 'overlay.jsonl', [
        status_row('s1', [0, 1, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 2, 0, 0, 0]),
        status_row('s2', [0, 1, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0])])
    pairs = {'s1': {'sample_id': 's1', 'route': 'R1', 'frame': 2, 'split': 'train'},
             's2': {'sample_id': 's2', 'route': 'BAD', 'frame': 3, 'split': 'train'}}
    result = builder.build_subset(manifest, pairs, {'BAD'}, overlay, tmp_path / 'subset.jsonl')
    assert result['verified_rows'] == 0
    assert result['rejected_count'] == 1
    assert result['excluded'] == {'geometry_defect': 1}


def test_capture_index_reads_only_online_records(tmp_path):
    good = {'schema': 'online-global-plan-v1', 'sha256': 'a' * 64, 'world_point_count': 7,
            'metadata': {'route': 'R1'}}
    bad = {'schema': 'something-else', 'metadata': {'route': 'R2'}}
    (tmp_path / 'a.json').write_text(json.dumps(good))
    (tmp_path / 'b.json').write_text(json.dumps(bad))
    index = builder.capture_index(tmp_path)
    assert set(index) == {'R1'}
    assert index['R1']['points'] == 7
