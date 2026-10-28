"""CPU tests for the continuous session-frame audit."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.b2d import session_continuous_audit as audit

STRAIGHT = [0.0, 1.0, 0.0, 0.0]
LEFT = [1.0, 0.0, 0.0, 0.0]
MOTION = [-0.1, -0.2, -0.3, -0.4]


def session():
    return {
        'sha256': 'session-sha',
        'points': [(float(i), 0.0) for i in range(0, 80)],
        'metadata': {'dataset_route': 'Route_A', 'town': 'Town01'},
    }


def overlay_row(motion=MOTION):
    return {
        'sample_identity': {'split': 'train', 'route': 'Route_A', 'frame': 1},
        'status8': STRAIGHT + list(motion),
        'old_status8': STRAIGHT + MOTION,
        'a0_motion4': MOTION,
    }


def test_matching_frame_with_identity_is_certified():
    record = audit.classify(session(), 1, (0.0, 0.0), 0.0,
                            STRAIGHT, MOTION, overlay_row())
    assert record['level'] == 'CERTIFIED'
    assert record['reason'] == 'SESSION_PLAN_COMMAND_VERIFIED'
    assert record['a0_motion_identity'] is True


def test_offline_invalid_frame_is_insufficient():
    record = audit.classify(session(), 1, (0.0, 0.0), 0.0, None, None, overlay_row())
    assert record['level'] == 'INSUFFICIENT_EVIDENCE'
    assert record['reason'] == 'OFFLINE_PREVIEW_INVALID'


def test_command_disagreement_is_mismatch():
    record = audit.classify(session(), 1, (0.0, 0.0), 0.0, LEFT, MOTION, overlay_row())
    assert record['level'] == 'COMMAND_MISMATCH'
    assert record['reason'] == 'SESSION_COMMAND_DIFFERS_FROM_OFFLINE'


def test_missing_status_row_caps_at_command_match_verified():
    record = audit.classify(session(), 1, (0.0, 0.0), 0.0, STRAIGHT, MOTION, None)
    assert record['level'] == 'COMMAND_MATCH_VERIFIED'
    assert record['reason'] == 'NO_FROZEN_STATUS_ROW'


def test_motion_identity_violation_blocks_certification():
    record = audit.classify(session(), 1, (0.0, 0.0), 0.0, STRAIGHT, MOTION,
                            overlay_row(motion=[0.0, 0.0, 0.0, 0.0]))
    assert record['level'] == 'COMMAND_MATCH_VERIFIED'
    assert record['reason'] == 'MOTION4_DIFFERS_FROM_FROZEN_A0'


def test_window_stats_counts_runs_and_switches():
    records = [
        {'frame': 0, 'preview_command': 'Straight'},
        {'frame': 1, 'preview_command': 'Straight'},
        {'frame': 2, 'preview_command': 'Left'},
        {'frame': 3, 'preview_command': 'Left'},
        {'frame': 4, 'preview_command': None},
    ]
    stats = audit.window_stats(records)
    assert stats['runs'] == 3
    assert stats['switch_count'] == 2
    assert stats['switch_frames'] == [2, 4]
    assert stats['invalid_runs'] == 1
    assert stats['invalid_frames'] == 1
