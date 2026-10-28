"""CPU tests for the certification evidence-level audit."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.b2d import certification_level_audit as audit

STRAIGHT = [0.0, 1.0, 0.0, 0.0]
LEFT = [1.0, 0.0, 0.0, 0.0]


def base_row(**overrides):
    row = {
        'route': 'Route_Town01_Route1',
        'town': 'Town01',
        'frame': 0,
        'xml_id': None,
        'xml_source': 'ONLINE_CAPTURED_PLAN',
        'plan_metadata_source': 'CARLA_GLOBAL_ROUTE_PLANNER_FROM_LOCAL_XML',
        'plan_sha256': 'planner-sha',
        'xml_preview_status': 'PASS',
        'rear_projection_distance_m': 1.0,
        'xml_command4': STRAIGHT,
        'offline_command4': STRAIGHT,
        'xml_vs_offline_point_distance_m': 0.01,
        'xml_horizon_m': 20.0,
        'rear_xy': [0.0, 0.0],
        'yaw_rad': 0.0,
        'xml_preview_point_xy': [20.0, 0.0],
        'map_name': 'Carla/Maps/Town01',
    }
    row.update(overrides)
    return row


def association(accepted=True):
    return {'Route_Town01_Route1': {'accepted': accepted, 'best': {'xml_id': '1'}}}


def planner():
    return {
        'sha256': 'planner-sha',
        'points': [(0.0, 0.0), (20.0, 0.0), (40.0, 0.0)],
        'metadata': {'xml_sha256': 'xml-sha', 'xml_id': '1', 'route_id': '1'},
    }


def session(command_left=False):
    points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    if command_left:
        points = [(0.0, 0.0), (10.0, 0.0), (15.0, -15.0), (15.0, -30.0)]
    return {
        'sha256': 'session-sha',
        'points': points,
        'agent_points': [(0.0, 0.0), (15.0, 15.0)],
        'metadata': {
            'dataset_route': 'Route_Town01_Route1',
            'town': 'Town01',
            'xml_sha256': 'xml-sha',
            'map_name': 'Carla/Maps/Town01',
            'leaderboard_evaluator_sha256': 'eval-sha',
            'route_scenario_sha256': 'scenario-sha',
        },
    }


def test_planner_only_never_reaches_certified():
    level, reason, details = audit.classify_row(base_row(), None, planner(), association(), set())
    assert level == 'CARLA_PLANNER_REPRODUCED'
    assert reason == 'PLANNER_ONLY_NO_SESSION_CAPTURE'
    assert details['command_match'] is True


def test_planner_only_mismatch_is_command_mismatch():
    row = base_row(offline_command4=LEFT)
    level, reason, _ = audit.classify_row(row, None, planner(), association(), set())
    assert level == 'COMMAND_MISMATCH'
    assert reason == 'PLANNER_COMMAND_DIFFERS_FROM_OFFLINE'


def test_session_capture_with_identity_and_match_is_certified():
    level, reason, details = audit.classify_row(base_row(), session(), planner(), association(), set())
    assert level == 'CERTIFIED'
    assert reason == 'SESSION_PLAN_COMMAND_VERIFIED'
    assert details['route_identity']['same_route_xml_sha'] is True


def test_session_capture_without_identity_stays_command_match_verified():
    level, reason, _ = audit.classify_row(base_row(), session(), planner(), association(accepted=False), set())
    assert level == 'COMMAND_MATCH_VERIFIED'
    assert reason == 'ROUTE_IDENTITY_NOT_PINNED'


def test_session_capture_command_mismatch_is_not_certified():
    level, reason, details = audit.classify_row(base_row(), session(command_left=True), planner(), association(), set())
    assert level == 'COMMAND_MISMATCH'
    assert reason == 'SESSION_COMMAND_DIFFERS_FROM_OFFLINE'
    assert details['session_command'] == 'Left'


def test_missing_planner_sha_and_defect_routes_are_insufficient():
    level, reason, _ = audit.classify_row(base_row(plan_sha256=None), None, planner(), association(), set())
    assert (level, reason) == ('INSUFFICIENT_EVIDENCE', 'NO_PLANNER_EVIDENCE')
    level, reason, _ = audit.classify_row(base_row(), session(), planner(), association(),
                                          {'Route_Town01_Route1'})
    assert (level, reason) == ('INSUFFICIENT_EVIDENCE', 'EXCLUDED_GEOMETRY_DEFECT')


def test_coverage_and_projection_failures_are_insufficient():
    level, reason, _ = audit.classify_row(
        base_row(xml_preview_status='FAIL'), session(), planner(), association(), set())
    assert (level, reason) == ('INSUFFICIENT_EVIDENCE', 'PLAN_COVERAGE_INSUFFICIENT')
    level, reason, _ = audit.classify_row(
        base_row(rear_projection_distance_m=9.0), session(), planner(), association(), set())
    assert (level, reason) == ('INSUFFICIENT_EVIDENCE', 'COORDINATE_PROJECTION_TOO_FAR')
