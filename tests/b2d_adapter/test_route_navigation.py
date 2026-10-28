import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.b2d_adapter.experimental_route_navigation import (
    CompactMap, build_segments, local_preview, maneuver_from_delta,
)


def test_maneuver_thresholds_and_signs():
    assert maneuver_from_delta(0) == 'Straight'
    assert maneuver_from_delta(19.9) == 'Straight'
    assert maneuver_from_delta(20.1) == 'Right'
    assert maneuver_from_delta(-20.1) == 'Left'
    assert maneuver_from_delta(170) == 'UTurn'
    assert maneuver_from_delta(-170) == 'UTurn'


def test_local_preview_straight_left_right_and_end():
    straight = [[0, 0], [50, 0]]
    p = local_preview(straight, (0, 0), 0, preview_m=20)
    assert p['category'] == 'Straight'
    assert p['v1_compatible'] is True

    left = [[0, 0], [20, 0], [20, -30]]
    p = local_preview(left, (0, 0), 0, preview_m=25)
    assert p['category'] == 'Left'

    right = [[0, 0], [20, 0], [20, 30]]
    p = local_preview(right, (0, 0), 0, preview_m=25)
    assert p['category'] == 'Right'

    p = local_preview([[0, 0], [10, 0]], (0, 0), 0, preview_m=20)
    assert p['status'] == 'FAIL'
    assert p['category'] == 'Unknown'
    assert p['reason'] == 'route_end_before_preview'


def test_local_preview_uturn_is_not_v1_compatible():
    path = [[0, 0], [20, 0], [0, 0]]
    p = local_preview(path, (0, 0), 0, preview_m=20)
    assert p['category'] == 'UTurn'
    assert p['v1_compatible'] is False
    assert p['command4_uturn_order'][-1] == 1
    assert p['command4_v1_order'][-1] == 1


def test_compact_map_best_match_reads_compact_export(tmp_path):
    points = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    offsets = np.array([0, 3], dtype=np.int64)
    np.savez_compressed(tmp_path / 'centers.npz', points=points, offsets=offsets)
    metadata = {
        'centers': [{
            'id': 0, 'road': 1, 'lane': 1, 'topology_type': 'Normal',
            'topology': [], 'left': [None, None], 'right': [None, None],
        }],
        'links': [],
    }
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    town = CompactMap(tmp_path / 'centers.npz', tmp_path / 'metadata.json', 'synthetic')
    match = town.best_match(1.3, 0.1, 0.0)
    assert match is not None
    assert match.road == 1
    assert match.lane == 1
    assert match.lateral_m < .4


def test_local_preview_rejects_duplicate_path_points():
    with pytest.raises(ValueError):
        local_preview([[0, 0], [0, 0], [10, 0]], (0, 0), 0)


def test_segments_split_only_at_unbridged_transition():
    class FakeTown:
        points = {
            1: np.array([[0., 0.], [10., 0.]]),
            2: np.array([[20., 0.], [30., 0.]]),
            9: np.array([[10., 0.], [20., 0.]]),
        }

        def center_points(self, center_id):
            return self.points[center_id]

        def center_path(self, a, b, max_hops=10):
            return [a, 9, b] if a == 1 and b == 2 else None

    runs = [
        {'indices': [0, 1], 'matches': [type('M', (), {'center_id': 1})()]},
        {'indices': [5, 6], 'matches': [type('M', (), {'center_id': 2})()]},
    ]
    segments = build_segments(FakeTown(), runs, route_name='synthetic')
    assert len(segments) == 1
    assert segments[0]['start_frame'] == 0
    assert segments[0]['end_frame'] == 6
    assert segments[0]['centers'] == [1, 9, 2]
    assert segments[0]['max_join_gap_m'] == 0.0

    FakeTown.center_path = lambda self, a, b, max_hops=10: None
    split = build_segments(FakeTown(), runs, route_name='synthetic')
    assert len(split) == 2
    assert split[0]['end_frame'] == 1
    assert split[1]['start_frame'] == 5
    assert split[1]['break_before'] == 'topology_path_missing'
