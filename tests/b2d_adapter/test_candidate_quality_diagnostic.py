import math
import os

import numpy as np
import pytest

import tools.validation.b2d.candidate_quality_diagnostic as diag


def official_anchor_index(target, anchor):
    """Independent re-implementation of multimodal_loss.LossComputer lines 133-135."""
    target = np.asarray(target, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    dist = np.linalg.norm(target[None, :, :2] - anchor, axis=-1)
    dist = dist.mean(-1)
    return int(np.argmin(dist))


def test_anchor_mode_matches_official_assignment():
    rng = np.random.default_rng(7)
    anchor = rng.normal(scale=8.0, size=(20, 8, 2))
    for _ in range(200):
        # Heading must not influence the official positive assignment.
        target = np.column_stack((rng.normal(scale=8.0, size=(8, 2)),
                                  rng.normal(scale=2.0, size=8)))
        assert diag.anchor_mode(target, anchor) == official_anchor_index(target, anchor)


def test_anchor_mode_ignores_heading_channel():
    anchor = np.zeros((20, 8, 2), dtype=np.float64)
    anchor[5] = 1.0
    target = np.column_stack((np.ones((8, 2)), np.linspace(-3, 3, 8)))
    assert diag.anchor_mode(target, anchor) == 5
    shifted = target.copy()
    shifted[:, 2] += math.pi
    assert diag.anchor_mode(shifted, anchor) == 5


def test_trajectory_geometry_straight_and_stationary():
    straight = np.column_stack((np.arange(1, 9) * 2.0, np.zeros(8)))
    g = diag.trajectory_geometry(straight)
    assert g['endpoint_m'] == pytest.approx(16.0)
    assert g['path_m'] == pytest.approx(16.0)
    assert g['longitudinal_end_m'] == pytest.approx(16.0)
    assert g['lateral_end_m'] == pytest.approx(0.0)
    assert g['second_diff_m'] == pytest.approx(0.0)
    assert g['detour_ratio'] == pytest.approx(1.0)
    assert g['total_yaw_deg'] == pytest.approx(0.0)
    assert g['backward_steps'] == 0

    stationary = np.zeros((8, 2))
    s = diag.trajectory_geometry(stationary)
    assert s['endpoint_m'] == pytest.approx(0.0)
    assert s['path_m'] == pytest.approx(0.0)
    assert s['detour_ratio'] is None
    assert 'near_zero' in diag.geometry_flags(s)


def test_trajectory_geometry_detects_reversal_and_extreme_step():
    # Forward then back to the origin: path is long, endpoint short.
    x = np.array([4.0, 8.0, 12.0, 8.0, 4.0, 0.0, -4.0, -8.0])
    geometry = diag.trajectory_geometry(np.column_stack((x, np.zeros(8))))
    assert geometry['backward_steps'] > 0
    assert geometry['detour_ratio'] is not None and geometry['detour_ratio'] > 1.0
    flags = diag.geometry_flags(geometry)
    assert 'reversal' in flags

    jumping = np.zeros((8, 2))
    jumping[4:, 0] = 40.0
    assert 'extreme_step' in diag.geometry_flags(diag.trajectory_geometry(jumping))


def test_anchor_class_is_report_only_taxonomy():
    assert diag.anchor_class({'endpoint_m': .2, 'total_yaw_deg': 0.0}) == 'stationary_like'
    assert diag.anchor_class({'endpoint_m': 5.0, 'total_yaw_deg': 0.0}) == 'low_speed_like'
    assert diag.anchor_class({'endpoint_m': 30.0, 'total_yaw_deg': 0.0}) == 'cruise_like'
    assert diag.anchor_class({'endpoint_m': 30.0, 'total_yaw_deg': 40.0}) == 'left_turn_like'
    assert diag.anchor_class({'endpoint_m': 30.0, 'total_yaw_deg': -40.0}) == 'right_turn_like'


def test_ade_fde_uses_xy_only_and_rejects_malformed():
    target = np.column_stack((np.arange(1, 9) * 1.0, np.zeros(8), np.zeros(8)))
    candidate = target.copy()
    candidate[:, 1] = 2.0
    candidate[:, 2] = 1.5              # heading error must not change ADE/FDE
    error = diag.ade_fde(candidate, target)
    assert error['ade_m'] == pytest.approx(2.0)
    assert error['fde_m'] == pytest.approx(2.0)
    assert len(error['point_error_m']) == 8
    with pytest.raises(ValueError):
        diag.ade_fde(target[:, :2], target)


def test_ade_fde_returns_none_for_nonfinite_candidate():
    target = np.zeros((8, 3))
    candidate = np.zeros((8, 3))
    candidate[3, 0] = np.nan
    assert diag.ade_fde(candidate, target) is None


def test_official_anchor_file_reports_no_zero_displacement_prior():
    path = os.environ.get('B2D_OFFICIAL_ANCHOR')
    if not path:
        pytest.skip('Set B2D_OFFICIAL_ANCHOR for optional official asset inspection')
    anchor = diag.load_anchor(path)
    assert anchor.shape == (20, 8, 2)
    endpoints = np.linalg.norm(anchor[:, -1, :], axis=-1)
    # Documented structural fact used by the report, not a training rule.
    assert endpoints.min() == pytest.approx(3.22, abs=0.01)
    assert endpoints.argmin() == 8


def test_module_exposes_no_training_or_selection_hook():
    source = (diag.__file__ and open(diag.__file__).read()) or ''
    for forbidden in ('backward()', 'optimizer.step', 'loss_computer(',
                      'cross_entropy', 'focal', 'select_mode', 'rerank'):
        assert forbidden not in source, forbidden
    assert diag.METRIC_PREFIX == 'diagnostic_only_'
