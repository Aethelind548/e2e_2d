from types import SimpleNamespace

import numpy as np

from tools.b2d_adapter.experimental_route_navigation import _stable_centers
from tools.b2d_adapter.experimental_route_preview import RoutePolyline


class FakeTown:
    def __init__(self, centers):
        self.centers = {
            center_id: SimpleNamespace(id=center_id, road=614, lane=-1,
                                       topology_type='Normal', is_junction=False)
            for center_id in centers
        }
        self._points = {center_id: np.asarray(points, dtype=np.float64)
                        for center_id, points in centers.items()}

    def center_points(self, center_id):
        return self._points[center_id]


def _matches(*center_ids):
    return [SimpleNamespace(center_id=center_id) for center_id in center_ids]


def test_stable_centers_collapses_duplicate_shared_endpoint_records():
    town = FakeTown({
        1368: [[786.66, 2284.11], [785.34, -69.04]],
        1369: [[783.23, 2206.11], [785.34, -69.04]],
    })
    assert _stable_centers(town, _matches(1369, 1368, 1369, 1368)) == [1369]


def test_stable_centers_keeps_a_real_sequential_transition():
    town = FakeTown({
        1: [[0.0, 0.0], [0.0, 10.0]],
        2: [[0.0, 10.0], [0.0, 20.0]],
    })
    assert _stable_centers(town, _matches(1, 1, 2, 2)) == [1, 2]


def test_stable_centers_does_not_repeat_same_center():
    town = FakeTown({1: [[0.0, 0.0], [0.0, 10.0]]})
    assert _stable_centers(town, _matches(1, 1, 1, 1)) == [1]


def test_vectorized_projection_uses_clamped_polyline_parameters():
    route = RoutePolyline([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], 'test', dense=True)
    arc, distance, projected, index = route.project([5.0, 1.0])
    assert index == 0
    assert abs(arc - 5.0) < 1e-9
    assert abs(distance - 1.0) < 1e-9
    assert np.allclose(projected, [5.0, 0.0])
    arc, distance, projected, index = route.project([10.0, 20.0])
    assert index == 1
    assert abs(arc - 20.0) < 1e-9
    assert abs(distance - 10.0) < 1e-9
    assert np.allclose(projected, [10.0, 10.0])
