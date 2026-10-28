"""Numerical evaluation contracts; no model, GPU, real cache or training required."""
import math
import numpy as np
import pytest

from tools.training.diffusiondrive_v1.evaluate_checkpoint import (
    Metrics, case_groups, content_identity, detection_counts, kinematic_baselines, semantic_counts,
    trajectory_errors, trajectory_flags, wrapped_angle,
)


def test_wrapped_heading_boundary():
    gt = np.zeros((8, 3)); pred = gt.copy()
    gt[:, 2] = math.pi - .01; pred[:, 2] = -math.pi + .01
    assert trajectory_errors(pred, gt)['heading_mae_rad'] == pytest.approx(.02)


def test_causal_cv_and_stationary():
    status = np.array([0, 1, 0, 0, 3., 4., 999, 999])
    b = kinematic_baselines(status, np.arange(1, 9) * .5)
    np.testing.assert_allclose(b['constant_velocity'][-1], [12, 16, math.atan2(4, 3)])
    assert not b['stationary'].any()


def test_trajectory_axes_and_point_errors():
    gt = np.zeros((8, 3)); p = gt.copy(); p[:, :2] = [3, 4]
    e = trajectory_errors(p, gt)
    assert e['ade_m'] == 5 and e['fde_m'] == 5
    assert e['longitudinal_mae_m'] == 3 and e['lateral_mae_m'] == 4
    assert e['point_error_m'] == [5] * 8


def test_nonfinite_and_shape_fail_closed():
    t = np.zeros((8, 3)); p = t.copy(); p[0, 0] = np.nan
    assert trajectory_errors(p, t) is None
    assert trajectory_flags(p, t)['nonfinite']
    with pytest.raises(ValueError): trajectory_errors(t[:7], t)
    with pytest.raises(ValueError): trajectory_errors(t, p)


def test_real_stop_not_degenerate():
    t = np.zeros((8, 3))
    assert not trajectory_flags(t, t)['degenerate_moving']
    gt = kinematic_baselines([0, 1, 0, 0, 3, 0, 0, 0], np.arange(1, 9) * .5)['constant_velocity']
    assert trajectory_flags(t, gt)['degenerate_moving']


def boxes(xy):
    return np.array([[x, y, 0, 4, 2] for x, y in xy], dtype=float).reshape(-1, 5)


def test_duplicate_detections_count_fp():
    r = detection_counts(boxes([(0, 0), (0, 0)]), [10, 10], boxes([(0, 0)]), [True])
    assert (r['tp'], r['fp'], r['fn']) == (1, 1, 0)


def test_distance_gate_maximizes_cardinality():
    # Valid assignment (0->1,1->0) has two matches. Invalid low-cost permutation
    # must not steal a match before thresholding.
    r = detection_counts(boxes([(0, 0), (1.8, 0)]), [10, 10], boxes([(.1, 0), (-1.8, 0)]), [True, True])
    assert r['tp'] == 2


def test_empty_and_invalid_detection_boxes():
    assert detection_counts(boxes([]), [], boxes([]), [])['tp'] == 0
    p = boxes([(0, 0)]); p[0, 3] = -1
    r = detection_counts(p, [1], boxes([(0, 0)]), [True])
    assert (r['tp'], r['fp'], r['fn']) == (0, 1, 1)
    p[0, 0] = np.nan
    assert detection_counts(p, [1], boxes([]), [])['invalid_frames'] == 1


def test_detection_threshold_and_gt_padding():
    r = detection_counts(boxes([(0, 0), (9, 0)]), [10, -10], boxes([(0, 0), (9, 0)]), [True, False])
    assert (r['tp'], r['fp'], r['fn']) == (1, 0, 0)


def test_bev_pixel_pooling_and_invalid_pixels():
    logits = np.zeros((7, 128, 256)); gt = np.zeros((128, 256), dtype=int)
    logits[1] = 1; gt[:64] = 1; logits[1, 0, 0] = np.nan
    cm, invalid = semantic_counts(logits, gt)
    assert invalid == 1 and cm.sum() == 32767
    assert cm[1, 1] == 16383 and cm[0, 1] == 16384
    gt[0, 0] = 7
    with pytest.raises(ValueError): semantic_counts(logits, gt)


def test_sample_weighted_sums_not_batch_mean():
    m = Metrics(); target = np.zeros((8, 3))
    for x in [0, 0, 9]:
        pred = target.copy(); pred[:, 0] = x
        error = trajectory_errors(pred, target)
        record = {'errors': dict(model=error, stationary=error, constant_velocity=error),
                  'flags': trajectory_flags(pred, target), 'invalid_bev_pixels': 0}
        cm = np.zeros((7, 7), dtype=int); cm[0, 0] = 1
        m.add(record, cm, dict(tp=1, fp=0, fn=1, frames=1, invalid_frames=0))
    result = m.result()
    assert result['trajectory']['model']['ade_m'] == 3  # not 4.5 for batch means 0 and 9
    assert result['bev']['iou']['road'] is None
    assert result['detection']['recall'] == .5
    assert result['detection']['f1'] == pytest.approx(2/3)


def test_validation_group_scope_and_speed_bins():
    groups = case_groups('v1/Scenario_Name_Town05_Route123_Weather1', 5.)
    assert groups == ['town:Town05', 'scenario:Scenario_Name', 'speed_mps:5-10']
    assert wrapped_angle(3 * math.pi) == pytest.approx(math.pi)


def test_read_atime_is_not_a_content_change():
    from types import SimpleNamespace
    a = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=5, st_atime_ns=6)
    b = SimpleNamespace(**dict(vars(a), st_atime_ns=7))
    assert content_identity(a) == content_identity(b)
    b.st_mtime_ns += 1
    assert content_identity(a) != content_identity(b)
