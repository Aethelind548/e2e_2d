"""CPU tests for the original-route XML certification helper logic."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.b2d import route_source_certification as cert

XML = """<routes>
  <route id="10" town="Town01">
    <waypoints>
      <position x="0.0" y="0.0" z="0.0" />
      <position x="10.0" y="0.0" z="0.0" />
      <position x="20.0" y="0.0" z="0.0" />
    </waypoints>
  </route>
</routes>"""


def test_load_xml_requires_two_points(tmp_path):
    for name in cert.XML_FILES:
        (tmp_path / name).write_text(XML)
    routes = cert.load_xml(tmp_path)
    assert set(routes) == {('Town01', '10')}
    assert routes[('Town01', '10')]['points'] == [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    assert routes[('Town01', '10')]['source'] == cert.XML_FILES[0]


def test_load_xml_skips_degenerate_routes(tmp_path):
    for name in cert.XML_FILES:
        (tmp_path / name).write_text(
            '<routes><route id="1" town="Town02"><waypoints>'
            '<position x="0" y="0" z="0" /></waypoints></route></routes>')
    assert cert.load_xml(tmp_path) == {}


def test_polyline_distance_projects_onto_segments():
    points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    distances = cert.polyline_distance(points, [(5.0, 2.0), (25.0, 0.0), (-3.0, 0.0)])
    assert distances == pytest.approx([2.0, 5.0, 3.0])


def test_accept_match_thresholds():
    assert cert.accept_match(1.0, 0.5)
    assert cert.accept_match(0.8, 8.0)
    assert not cert.accept_match(0.79, 0.0)
    assert not cert.accept_match(1.0, 8.01)


def test_tool_never_emits_certified_rows():
    text = Path(cert.__file__).read_text()
    assert "'CERTIFIED'" not in text
    assert 'certified_rows\': 0' in text


def test_dedupe_points_drops_consecutive_duplicates():
    points = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (10.0, 1e-9), (20.0, 0.0)]
    assert cert.dedupe_points(points) == [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    assert cert.dedupe_points([(1.0, 2.0)]) == [(1.0, 2.0)]
    assert cert.dedupe_points([(1.0, 2.0), (1.0, 2.0)]) == [(1.0, 2.0)]
