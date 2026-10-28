import numpy as np

from tools.training.diffusiondrive_v1.evaluate_checkpoint import trajectory_errors
from tools.training.diffusiondrive_v1.status_route_eval_core import _match, bev_metrics


def _gt(x=1.0, y=0.0, heading=0.0):
    states = np.zeros((30, 5), dtype=np.float32)
    valid = np.zeros(30, dtype=bool)
    states[0] = [x, y, heading, 4.0, 2.0]
    valid[0] = True
    return states, valid


def test_match_perfect_prediction():
    gt, valid = _gt()
    pred = gt[:1].copy()
    result = _match(pred, np.array([2.0]), gt, valid, .5, 2.0)
    assert (result['tp'], result['fp'], result['fn']) == (1, 0, 0)
    assert result['matched_center_error_m'][0] < 1e-6


def test_match_no_gt():
    gt, valid = _gt()
    valid[:] = False
    result = _match(np.array([[1., 0., 0., 4., 2.]]), np.array([2.]), gt, valid, .5, 2.0)
    assert (result['tp'], result['fp'], result['fn']) == (0, 1, 0)


def test_match_duplicate_predictions_are_one_tp_one_fp():
    gt, valid = _gt()
    pred = np.repeat(gt[:1], 2, axis=0)
    result = _match(pred, np.array([2., 2.]), gt, valid, .5, 2.0)
    assert (result['tp'], result['fp'], result['fn']) == (1, 1, 0)


def test_threshold_rejects_low_confidence_prediction():
    gt, valid = _gt()
    result = _match(gt[:1], np.array([-1.0]), gt, valid, .5, 2.0)
    assert (result['selected'], result['tp'], result['fp'], result['fn']) == (0, 0, 0, 1)


def test_heading_wrap_is_reported_separately_from_raw_difference():
    gt, valid = _gt(heading=-np.pi + 0.01)
    pred = gt[:1].copy()
    pred[0, 2] = np.pi - 0.01
    result = _match(pred, np.array([2.]), gt, valid, .5, 2.0)
    assert result['matched_heading_raw_abs_rad'][0] > 6.0
    assert result['matched_heading_wrapped_abs_rad'][0] < 0.03


def test_bev_missing_class_is_na_not_perfect():
    cm = np.zeros((7, 7), dtype=np.int64)
    cm[0, 0] = 10
    cm[1, 1] = 5
    metrics = bev_metrics(cm, 0)
    assert metrics['iou']['background'] == 1.0
    assert metrics['iou']['road'] == 1.0
    assert metrics['iou']['pedestrian'] is None
    assert metrics['miou_present_union'] == 1.0


def test_perfect_trajectory_has_zero_error():
    target = np.zeros((8, 3), dtype=np.float32)
    target[:, 0] = np.arange(1, 9) * .5
    result = trajectory_errors(target, target)
    assert result['ade_m'] == 0.0 and result['fde_m'] == 0.0
    assert result['heading_mae_rad'] == 0.0
