import math

from tools.validation.b2d.route_preview_v11 import recover_candidate


def record(candidates):
    return {
        'route': 'synthetic',
        'frame': 1,
        'actor_yaw_deg': 0.0,
        'preview_origin': {'rear_xyz': [0.0, 0.0, 0.0]},
        'candidates': [{'points_xyz': [[x, y, 0.0] for x, y in points]}
                       for points in candidates],
        'candidate_status': 'ambiguous' if len(candidates) > 1 else 'unique',
    }


def test_candidate_consensus_only_when_20m_target_and_command_agree():
    # Both candidates have the same 20 m point, then diverge after 20 m.
    preview, info = recover_candidate(record([
        [(0, 0), (50, 0)],
        [(0, 0), (20, 0), (20, 20)],
    ]))
    assert preview is not None
    assert preview['preview_valid'] is True
    assert info['candidate_consensus'] is True
    assert info['recovery_reason'] == 'candidate_consensus_at_20m'


def test_candidate_conflict_at_20m_is_not_recovered():
    preview, info = recover_candidate(record([
        [(0, 0), (50, 0)],
        [(0, 0), (10, 0), (10, 20)],
    ]))
    assert preview is None
    assert info['candidate_consensus'] is False
    assert info['recovery_reason'] == 'candidate_disagreement_at_20m'
