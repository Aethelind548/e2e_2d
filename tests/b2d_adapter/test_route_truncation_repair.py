from types import SimpleNamespace

import numpy as np

from tools.validation.b2d.route_truncation_repair import build_repair_row


class FakeTown:
    def __init__(self):
        self.centers = {
            1: SimpleNamespace(id=1, road=1, lane=1, topology_type='Normal', is_junction=False,
                               points=[[0.0, 0.0], [10.0, 0.0]]),
            3: SimpleNamespace(id=3, road=3, lane=1, topology_type='Normal', is_junction=False,
                               points=[[10.0, 0.0], [30.0, 0.0]]),
        }

    def center_points(self, center_id):
        return np.asarray(self.centers[center_id].points, dtype=np.float64)


def _state():
    town = FakeTown()
    segment = {
        'segment_id': 0, 'start_frame': 0, 'end_frame': 0,
        'points_xy': [[0.0, 0.0], [10.0, 0.0]],
    }
    return {
        'town': town, 'frames': [0], 'poses': [(2.0, 0.0, 0.0)],
        'z_values': [0.0], 'matches': [None], 'records': {},
        'segments': [segment], 'source_frames': [0], 'cumulative': [0.0],
        'frame_index': {0: 0}, 'reverse_links': {},
        'branch_extension_cache': {}, 'segment_path_cache': {},
    }


def _branch():
    return {
        'normal_center_id': 3, 'road': 3, 'lane': 1,
        'topology_type': 'Normal', 'path_center_ids': [1, 3],
        'valid_join': True, 'join_gaps_m': [0.0], 'max_join_gap_m': 0.0,
    }


def _row(subtype):
    return {
        'route': 'route-A', 'frame': 0,
        'route_end_class': 'INTERNAL_SEGMENT_TRUNCATION',
        'truncation_subtype': subtype, 'source_last_frame': 0,
        'branches': [_branch()], 'repair_selected_branch_id': 3,
        'subtype_reason': 'test',
    }


def test_unique_branch_produces_a_certified_candidate():
    result = build_repair_row(_row('RECONSTRUCTION_STOP'), _state())
    assert result['candidate_status'] == 'RECONSTRUCTION_STOP_REPAIR'
    assert result['certified'] is True
    assert result['recovered'] is True
    assert result['selected_preview']['preview_valid'] is True
    assert result['crosses_unverified_break'] is False


def test_ambiguous_branch_is_never_selected():
    result = build_repair_row(_row('MAP_MATCH_AMBIGUOUS'), _state())
    assert result['candidate_status'] == 'CANDIDATES_ONLY_AMBIGUOUS'
    assert result['certified'] is False
    assert result['recovered'] is False
    assert result['selected_branch_id'] is None
    assert result['branch_candidates'][0]['selected'] is False
    assert result['branch_candidates'][0]['preview_valid'] is True
