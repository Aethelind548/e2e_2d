import pytest

from tools.validation.b2d.route_preview_generate import (
    DEFAULT_MAX_PREVIEW_WORK, DEFAULT_MAX_ROUTE_POINTS,
    DEFAULT_MAX_SEGMENT_LENGTH_M, DEFAULT_MAX_SEGMENT_POINTS,
    RouteResourceExceeded, enforce_route_resources, quarantine_path,
    read_quarantine, remove_quarantine, route_resource_profile,
    write_quarantine,
)


class SizedPoints:
    def __init__(self, count):
        self.count = int(count)

    def __len__(self):
        return self.count


def _segment(points, length_m, start_frame=0, end_frame=99):
    return {
        'points_xy': SizedPoints(points),
        'length_m': float(length_m),
        'start_frame': start_frame,
        'end_frame': end_frame,
    }


def test_normal_segment_passes_limits():
    limits = {
        'max_segment_points': DEFAULT_MAX_SEGMENT_POINTS,
        'max_route_points': DEFAULT_MAX_ROUTE_POINTS,
        'max_segment_length_m': DEFAULT_MAX_SEGMENT_LENGTH_M,
        'max_preview_work': DEFAULT_MAX_PREVIEW_WORK,
    }
    profile = enforce_route_resources([_segment(100, 1000.0)], limits)
    assert profile['max_segment_points'] == 100
    assert profile['preview_work_units'] == 10_000


def test_pathological_segment_is_rejected_before_preview():
    limits = {
        'max_segment_points': 200_000,
        'max_route_points': 400_000,
        'max_segment_length_m': 100_000.0,
        'max_preview_work': 100_000_000,
    }
    with pytest.raises(RouteResourceExceeded) as raised:
        enforce_route_resources([_segment(1_199_092, 1_196_400.49, 0, 624)], limits)
    details = raised.value.details
    assert 'max_segment_points' in details['violations']
    assert 'max_segment_length_m' in details['violations']
    assert 'max_preview_work' in details['violations']


def test_quarantine_marker_is_source_signature_specific(tmp_path):
    route = 'ControlLoss_Town12_Route23404_Weather5_02-12-22-54-10'
    marker = write_quarantine(tmp_path, route, 'source-a',
                              'resource_limit_exceeded',
                              {'resource_profile': {'max_segment_points': 10}})
    assert marker.is_file()
    assert read_quarantine(tmp_path, route, 'source-a')['reason'] == 'resource_limit_exceeded'
    assert read_quarantine(tmp_path, route, 'source-b') is None
    assert quarantine_path(tmp_path, route) == marker
    remove_quarantine(tmp_path, route)
    assert not marker.exists()
