"""Physical source boundaries and stable frame identity for full audit v1."""
import numpy as np
import pytest
from types import SimpleNamespace

from tools.validation.b2d.full_route_preview_audit import (
    center_repeat_evidence, measured_failure, segment_frame_bounds,
)


def _failure(frame):
    return {'route': 'route', 'frame': frame, 'failure_reason': 'route_end_before_preview',
            'preview_validity': 'FAIL'}


def test_snapshot_hashes_content_and_preserves_source_metadata(tmp_path):
    from hashlib import sha256
    from tools.validation.b2d.full_route_preview_audit import snapshot
    path = tmp_path / 'manifest.json'
    payload = b'{"contract":"frozen"}\n'
    path.write_bytes(payload)
    before = path.stat()
    result = snapshot(tmp_path)['manifest.json']
    assert result == {'sha256': sha256(payload).hexdigest(),
                      'size': len(payload), 'mtime_ns': before.st_mtime_ns}
    assert path.read_bytes() == payload
    assert path.stat().st_mtime_ns == before.st_mtime_ns


def test_nonzero_source_frames_convert_segment_indices_without_guessing():
    segments = [{'segment_id': 0, 'start_frame': 0, 'end_frame': 2}]
    converted = segment_frame_bounds(segments, [40, 42, 44])
    assert converted[0]['start_frame'] == 40
    assert converted[0]['end_frame'] == 44
    with pytest.raises(ValueError, match='out_of_bounds'):
        segment_frame_bounds(segments, [40, 42])


def test_stationary_tail_over_twenty_frames_can_be_expected():
    frames = list(range(100))
    segments = [{'segment_id': 0, 'start_frame': 0, 'end_frame': 99}]
    result = measured_failure(_failure(10), segments, frames, np.linspace(0, 5, 100), True)
    assert result['classification'] == 'EXPECTED_SEQUENCE_END'
    assert result['legacy_20_frame_classification'] == 'INTERNAL_SEGMENT_TRUNCATION'
    assert result['physical_time_standard_certified'] is False


def test_internal_segment_near_source_tail_is_not_expected():
    frames = list(range(100))
    segments = [{'segment_id': 0, 'start_frame': 0, 'end_frame': 90}]
    result = measured_failure(_failure(85), segments, frames, np.linspace(0, 5, 100), True)
    assert result['classification'] == 'INTERNAL_SEGMENT_TRUNCATION'
    assert result['legacy_20_frame_classification'] == 'EXPECTED_SEQUENCE_END'


def test_discontinuous_source_or_long_future_arc_blocks_expected():
    segments = [{'segment_id': 0, 'start_frame': 0, 'end_frame': 99}]
    frames = list(range(100))
    assert measured_failure(_failure(50), segments, frames, np.linspace(0, 5, 100), False)[
        'classification'] == 'INTERNAL_SEGMENT_TRUNCATION'
    assert measured_failure(_failure(50), segments, frames, np.linspace(0, 100, 100), True)[
        'classification'] == 'INTERNAL_SEGMENT_TRUNCATION'


def test_observed_loop_is_supported_but_stationary_center_chatter_is_not():
    town = SimpleNamespace(center_points=lambda cid: np.asarray([[0, 0], [20, 0]]))
    matches = [SimpleNamespace(center_id=cid) for cid in [1, 2, 1]]
    supported = center_repeat_evidence([1, 2, 1], matches, [], [0, 25, 50], town)
    assert supported[0]['assessment'] == 'OBSERVED_TRAVEL_SUPPORTS_LOOP'
    assert supported[0]['a_b_a'] is True
    unsupported = center_repeat_evidence([1, 2, 1], matches, [], [0, 0.5, 1], town)
    assert unsupported[0]['assessment'] == 'NOT_VERIFIED_REPEATED_CENTER'
