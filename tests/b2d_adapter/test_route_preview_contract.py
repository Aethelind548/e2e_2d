import math

import pytest

from tools.b2d_adapter.route_preview_contract import (
    COMMAND_ORDER, UNSUPPORTED_UTURN, preview_command4, route_progress,
    stable_preview_command4, status8, status8_from_preview,
)


def test_shared_contract_keeps_v1_order_and_status8_motion():
    assert COMMAND_ORDER == ('Left', 'Straight', 'Right', 'Unknown')
    assert status8([0, 1, 0, 0], [1.5, -0.2, 0.1, 0.0]) == (
        0.0, 1.0, 0.0, 0.0, 1.5, -0.2, 0.1, 0.0)


def test_shared_contract_straight_left_right_and_end():
    straight = [[0, 0], [50, 0]]
    r = preview_command4(straight, (0, 0), 0, preview_m=20)
    assert r['status'] == 'PASS'
    assert r['category'] == 'Straight'
    assert r['command4'] == [0, 1, 0, 0]
    assert r['route_progress_m'] == pytest.approx(0)
    assert r['preview_point_xy'][0] == pytest.approx(20)

    left = [[0, 0], [20, 0], [20, -30]]
    r = preview_command4(left, (0, 0), 0, preview_m=25)
    assert r['category'] == 'Left'
    assert r['command4'] == [1, 0, 0, 0]
    assert r['lateral_m'] > 2

    right = [[0, 0], [20, 0], [20, 30]]
    r = preview_command4(right, (0, 0), 0, preview_m=25)
    assert r['category'] == 'Right'
    assert r['command4'] == [0, 0, 1, 0]
    assert r['lateral_m'] < -2

    r = preview_command4([[0, 0], [10, 0]], (0, 0), 0, preview_m=20)
    assert r['status'] == 'FAIL'
    assert r['preview_valid'] is False
    assert r['reason'] == 'route_end_before_preview'
    assert r['command4'] == [0, 0, 0, 1]


def test_shared_contract_uturn_is_explicit_and_not_status8():
    path = [[0, 0], [20, 0], [0, 0]]
    r = preview_command4(path, (0, 0), 0, preview_m=20)
    assert r['status'] == UNSUPPORTED_UTURN
    assert r['uturn_status'] == UNSUPPORTED_UTURN
    assert r['preview_valid'] is False
    assert r['command4'] == [0, 0, 0, 1]
    with pytest.raises(ValueError):
        status8_from_preview(r, [0, 0, 0, 0])


def test_shared_contract_progress_rejects_large_backtrack_and_projection():
    path = [[0, 0], [50, 0]]
    p = route_progress(path, (20, 0), previous_progress_m=22)
    assert p.valid is False
    assert p.reason == 'progress_regression'
    p = route_progress(path, (10, 4.1), max_projection_m=4)
    assert p.valid is False
    assert p.reason == 'projection_distance_exceeded'


def test_shared_contract_status8_rejects_non_one_hot():
    with pytest.raises(ValueError):
        status8([0.5, 0.5, 0, 0], [0, 0, 0, 0])
    with pytest.raises(ValueError):
        status8([0, 1, 0, 0], [math.nan, 0, 0, 0])


def test_v3_stable_preview_accepts_corroborated_shorter_end_horizon():
    # The route ends before 20 m; 5/4/3 m all give the same command.
    path = [[0, 0], [5.2, 0]]
    r = stable_preview_command4(path, (0, 0), 0, min_preview_m=5)
    assert r['status'] == 'PASS'
    assert r['stable'] is True
    assert r['selected']['category'] == 'Straight'
    assert r['selected']['preview_m'] == pytest.approx(5)
    assert r['same_category_horizons'] == [5.0, 4.0, 3.0]


def test_v3_stable_preview_rejects_too_short_end_and_unstable_turn():
    short = stable_preview_command4([[0, 0], [4.2, 0]], (0, 0), 0,
                                    min_preview_m=5)
    assert short['status'] == 'FAIL'
    assert short['stable'] is False
    assert short['reason'] == 'route_remaining_too_short'

    # 20 m is Left, but 15 m and below are already Straight: not stable.
    turn = [[0, 0], [15, 0], [15, -30]]
    unstable = stable_preview_command4(turn, (0, 0), 0, min_preview_m=5)
    assert unstable['status'] == 'FAIL'
    assert unstable['stable'] is False
    assert unstable['reason'] == 'command_not_stable_across_horizons'


def test_v3_stable_preview_does_not_hide_unknown_or_uturn():
    # 15/12/10 m are U-turn; 8 m and below are Straight.  Shortening must not
    # hide the longer-horizon U-turn.
    uturn = [[0, 0], [10, 0], [0, 0]]
    r = stable_preview_command4(uturn, (0, 0), 0, min_preview_m=5)
    assert r['status'] == 'FAIL'
    assert r['stable'] is False
    assert r['reason'] == 'longer_horizon_not_route_end'
    assert r['hidden_longer_horizons'] == [20.0, 15.0, 12.0, 10.0]
