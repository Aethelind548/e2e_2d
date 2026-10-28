from types import SimpleNamespace

import numpy as np

from tools.validation.b2d.route_truncation_root_cause import (
    _clean_points, branch_extension, classify_subtype, successor_branches,
)


class FakeTown:
    def __init__(self, centers, links):
        self.centers = centers
        self.links = links

    def center_points(self, center_id):
        return np.asarray(self.centers[center_id].points, dtype=np.float64)


def _center(center_id, topology='Normal', points=None):
    return SimpleNamespace(id=center_id, road=center_id, lane=1,
                           topology_type=topology, is_junction=topology == 'Junction',
                           points=points or [[0.0, 0.0], [1.0, 0.0]])


def _town(branches=2):
    centers = {
        1: _center(1, points=[[-1.0, 0.0], [0.0, 0.0]]),
        2: _center(2, topology='Junction', points=[[0.0, 0.0], [0.1, 0.0]]),
        3: _center(3, points=[[0.1, 0.0], [3.0, 0.0]]),
        4: _center(4, points=[[0.1, 1.0], [3.0, 1.0]]),
    }
    links = {1: [(2, 0.05)], 2: [(3, 0.01)]}
    if branches > 1:
        links[2].append((4, 0.01))
    return FakeTown(centers, links)


def test_geometry_cleanup_keeps_strict_join_threshold_and_preserves_input():
    points = [[0, 0, 9], [0, 0, 8], [1e-6, 0, 7], [2e-6, 0, 6]]
    assert _clean_points(points) == [[0., 0.], [2e-6, 0.]]
    assert points == [[0, 0, 9], [0, 0, 8], [1e-6, 0, 7], [2e-6, 0, 6]]
    assert _clean_points([]) == []


def test_successor_branches_returns_distinct_normal_successors():
    town = _town(branches=2)
    branches = successor_branches(town, 1)
    assert [item['normal_center_id'] for item in branches] == [3, 4]
    assert all(item['path_center_ids'][:2] == [1, 2] for item in branches)


def test_branch_extension_preserves_map_geometry_and_rejects_large_gap():
    town = _town(branches=1)
    segment = {'points_xy': [[-1.0, 0.0], [0.0, 0.0]]}
    branch = successor_branches(town, 1)[0]
    extension = branch_extension(town, segment, branch)
    assert extension is not None
    assert extension['valid_join'] is True
    assert extension['max_join_gap_m'] < 1e-9
    town.centers[3].points = [[2.5, 0.0], [3.0, 0.0]]
    extension = branch_extension(town, segment, branch)
    assert extension['valid_join'] is False
    assert extension['max_join_gap_m'] > 1.0


def test_classification_requires_future_resolution_for_multiple_branches():
    evidence = {
        'segment_present': True,
        'current_match_present': True,
        'source_discontinuity': {'present': False},
        'map_candidate_conflict': False,
        'successor_links': [{'target_center_id': 2}],
        'branches': [
            {'normal_center_id': 3, 'valid_join': True},
            {'normal_center_id': 4, 'valid_join': True},
        ],
        'future_observed_branch_ids': [],
    }
    assert classify_subtype(evidence)[0] == 'MAP_MATCH_AMBIGUOUS'
    evidence['future_observed_branch_ids'] = [3]
    assert classify_subtype(evidence)[0] == 'RECONSTRUCTION_STOP'


def test_unique_branch_is_reconstruction_stop_and_missing_branch_is_topology_missing():
    base = {
        'segment_present': True,
        'current_match_present': True,
        'source_discontinuity': {'present': False},
        'map_candidate_conflict': False,
        'future_observed_branch_ids': [],
        'branches': [{'normal_center_id': 3, 'valid_join': True}],
        'successor_links': [{'target_center_id': 2}],
    }
    assert classify_subtype(base)[0] == 'RECONSTRUCTION_STOP'
    base['branches'] = []
    assert classify_subtype(base)[0] == 'TOPOLOGY_LINK_MISSING'
    base['successor_links'] = []
    assert classify_subtype(base)[0] == 'TOPOLOGY_LINK_MISSING'


def test_source_discontinuity_has_priority():
    evidence = {
        'segment_present': True,
        'current_match_present': True,
        'source_discontinuity': {'present': True, 'reasons': ['teleport_flag']},
        'map_candidate_conflict': False,
        'successor_links': [],
        'branches': [],
        'future_observed_branch_ids': [],
    }
    assert classify_subtype(evidence)[0] == 'SOURCE_DISCONTINUITY'


def test_missing_evidence_is_unresolved():
    assert classify_subtype({'segment_present': False})[0] == 'UNRESOLVED'
