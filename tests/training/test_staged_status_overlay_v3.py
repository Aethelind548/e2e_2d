"""CPU tests for the staged Status overlay v3 builder."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.training.diffusiondrive_v1 import staged_status_overlay_v3 as builder

STRAIGHT = [0.0, 1.0, 0.0, 0.0]
MOTION = [-0.1, -0.2, -0.3, -0.4]


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def make_inputs(tmp_path):
    subset = [
        {'sample_id': 'a' * 64,
         'sample_identity': {'sample_id': 'a' * 64, 'route': 'Route_A', 'frame': 1, 'split': 'train'},
         'status8': STRAIGHT + MOTION, 'old_status8': STRAIGHT + MOTION, 'a0_motion4': MOTION},
        {'sample_id': 'b' * 64,
         'sample_identity': {'sample_id': 'b' * 64, 'route': 'Route_B', 'frame': 2, 'split': 'validation'},
         'status8': STRAIGHT + MOTION, 'old_status8': STRAIGHT + MOTION, 'a0_motion4': MOTION},
        {'sample_id': 'c' * 64,
         'sample_identity': {'sample_id': 'c' * 64, 'route': 'Route_C', 'frame': 3, 'split': 'train'},
         'status8': STRAIGHT + MOTION, 'old_status8': STRAIGHT + MOTION, 'a0_motion4': MOTION},
    ]
    audit = [
        {'sample_id': 'a' * 64, 'certification_level': 'CERTIFIED', 'route_source_identity': {},
         'session_capture_sha256': 's1', 'planner_capture_sha256': 'p1', 'command_match': True,
         'preview_point_distance_m': 0.01, 'horizon_m': 20.0, 'rear_projection_distance_m': 0.5,
         'certification_reason': 'SESSION_PLAN_COMMAND_VERIFIED'},
        {'sample_id': 'b' * 64, 'certification_level': 'CARLA_PLANNER_REPRODUCED', 'route_source_identity': {},
         'session_capture_sha256': None, 'planner_capture_sha256': 'p2', 'command_match': True,
         'preview_point_distance_m': 0.02, 'horizon_m': 20.0, 'rear_projection_distance_m': 0.4,
         'certification_reason': 'PLANNER_ONLY_NO_SESSION_CAPTURE'},
    ]
    overlay = [
        {'sample_id': 'a' * 64, 'status8': STRAIGHT + MOTION, 'old_status8': STRAIGHT + MOTION,
         'a0_motion4': MOTION, 'status_dtype': 'float32',
         'status_order': ['Left', 'Straight', 'Right', 'Unknown', 'vx', 'vy', 'ax', 'ay'],
         'navigation_order': ['Left', 'Straight', 'Right', 'Unknown']},
        {'sample_id': 'b' * 64, 'status8': STRAIGHT + MOTION, 'old_status8': STRAIGHT + MOTION,
         'a0_motion4': MOTION, 'status_dtype': 'float32',
         'status_order': ['Left', 'Straight', 'Right', 'Unknown', 'vx', 'vy', 'ax', 'ay'],
         'navigation_order': ['Left', 'Straight', 'Right', 'Unknown']},
        {'sample_id': 'c' * 64, 'status8': STRAIGHT + MOTION, 'old_status8': STRAIGHT + MOTION,
         'a0_motion4': MOTION, 'status_dtype': 'float32',
         'status_order': ['Left', 'Straight', 'Right', 'Unknown', 'vx', 'vy', 'ax', 'ay'],
         'navigation_order': ['Left', 'Straight', 'Right', 'Unknown']},
    ]
    write_jsonl(tmp_path / 'subset.jsonl', subset)
    write_jsonl(tmp_path / 'audit.jsonl', audit)
    write_jsonl(tmp_path / 'overlay.jsonl', overlay)
    return subset, audit, overlay


def run_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'builder',
        '--subset', str(tmp_path / 'subset.jsonl'),
        '--audit-rows', str(tmp_path / 'audit.jsonl'),
        '--overlay', str(tmp_path / 'overlay.jsonl'),
        '--output-dir', str(tmp_path / 'out'),
    ])
    builder.main()


def test_builder_marks_every_row_not_admitted(tmp_path, monkeypatch):
    make_inputs(tmp_path)
    run_builder(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (tmp_path / 'out' / 'staged_status_overlay_v3.jsonl').open()]
    assert len(rows) == 2
    assert all(row['training_admitted'] is False for row in rows)
    assert all(row['staged_only'] is True for row in rows)
    assert all(row['frozen_overlay_sha256'] for row in rows)


def test_builder_copies_levels_and_reports_missing_audit(tmp_path, monkeypatch):
    make_inputs(tmp_path)
    run_builder(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (tmp_path / 'out' / 'staged_status_overlay_v3.jsonl').open()]
    levels = {row['sample_id']: row['certification_level'] for row in rows}
    assert levels['a' * 64] == 'CERTIFIED'
    assert levels['b' * 64] == 'CARLA_PLANNER_REPRODUCED'
    summary = json.loads((tmp_path / 'out' / 'staged_status_overlay_v3_summary.json').read_text())
    assert summary['rows_in'] == 3
    assert summary['rows_out'] == 2
    assert summary['missing_audit_rows'] == 1
    assert summary['training_admitted_rows'] == 0
    assert summary['reader_ready'] is False


def test_builder_detects_route_overlap_and_certified_split(tmp_path, monkeypatch):
    make_inputs(tmp_path)
    run_builder(tmp_path, monkeypatch)
    summary = json.loads((tmp_path / 'out' / 'staged_status_overlay_v3_summary.json').read_text())
    assert summary['route_overlap_train_validation'] == []
    assert summary['certified_rows_by_split'] == {'train': 1}
    assert summary['blocking_reason'] == 'CERTIFICATION_CONTRACT_V2_NOT_APPROVED'
