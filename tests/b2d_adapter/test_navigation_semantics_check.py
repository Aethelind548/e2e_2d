"""CPU tests for the navigation-semantics rule comparison."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.b2d import navigation_semantics_check as semantics

STRAIGHT = [0.0, 1.0, 0.0, 0.0]
LEFT = [1.0, 0.0, 0.0, 0.0]


def test_option_change_view_keeps_endpoints_and_spacing():
    points = [(float(i), 0.0) for i in range(0, 131)]
    kept = semantics.option_change_view(points, sample_factor=50.0)
    assert kept[0] == (0.0, 0.0)
    assert kept[-1] == (130.0, 0.0)
    assert len(kept) == 4
    gaps = [b[0] - a[0] for a, b in zip(kept, kept[1:])]
    assert all(gap >= 50.0 for gap in gaps[:-1])


def test_dense_plan_and_agent_chord_diverge_geometrically():
    dense = [(float(i), 0.0) for i in range(0, 10)] + \
            [(10.0 + i, -float(i)) for i in range(0, 40)]
    agent = semantics.option_change_view(dense, sample_factor=50.0)
    rules = semantics.compare_views(dense, agent, (0.0, 0.0), 0.0)
    dense_preview = rules['dense_projection5']
    agent_preview = rules['agent_projection5']
    assert dense_preview['status'] == 'PASS'
    assert agent_preview['status'] == 'PASS'
    # The 50 m chord cuts the corner: the 20 m preview point must move materially,
    # which is exactly why the agent view cannot stand in for the dense plan.
    dense_point = dense_preview['preview_point_xy']
    agent_point = agent_preview['preview_point_xy']
    assert ((dense_point[0] - agent_point[0]) ** 2
            + (dense_point[1] - agent_point[1]) ** 2) ** 0.5 > 2.0
    assert abs((dense_preview['heading_delta_deg'] or 0.0)
               - (agent_preview['heading_delta_deg'] or 0.0)) > 5.0


def test_straight_plan_matches_under_both_rules():
    dense = [(float(i), 0.0) for i in range(0, 80)]
    agent = semantics.option_change_view(dense, sample_factor=50.0)
    rules = semantics.compare_views(dense, agent, (0.0, 0.0), 0.0)
    assert rules['dense_projection5']['command4'] == STRAIGHT
    assert rules['agent_projection5']['command4'] == STRAIGHT
