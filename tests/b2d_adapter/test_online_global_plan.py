"""CPU tests for the online global-plan capture serializer (no CARLA needed)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validation.b2d import online_global_plan as ogp


class Loc:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class Rot:
    def __init__(self, yaw, pitch=0.0, roll=0.0):
        self.yaw, self.pitch, self.roll = yaw, pitch, roll


class Tr:
    def __init__(self, loc, rot):
        self.location, self.rotation = loc, rot


class Opt:
    def __init__(self, name):
        self.name = name


def plan(count=5):
    world = [(Tr(Loc(float(i), 0.0, 0.5), Rot(0.0)), Opt('LANEFOLLOW')) for i in range(count)]
    world[count - 2] = (Tr(Loc(float(count - 2), 0.0, 0.5), Rot(0.0)), Opt('LEFT'))
    gps = [((10.0 + i * 1e-5, 20.0, 30.0), option) for i, (_, option) in enumerate(world)]
    return gps, world


def test_serialize_counts_options_and_length():
    gps, world = plan()
    record = ogp.serialize_plan(gps, world, {'route_id': 'R1'})
    assert record['world_point_count'] == record['gps_point_count'] == 5
    assert record['road_option_counts'] == {'LANEFOLLOW': 4, 'LEFT': 1}
    assert record['world_arc_length_m'] == pytest.approx(4.0)
    assert record['world_points'][3][:3] == [3.0, 0.0, 0.5]
    assert record['gps_points'][0][:3] == [10.0, 20.0, 30.0]
    assert record['certification_effect'].startswith('ONLINE_PLAN_CAPTURED')


def test_record_sha_is_content_addressed():
    gps, world = plan()
    first = ogp.serialize_plan(gps, world, {'route_id': 'R1'})
    second = ogp.serialize_plan(gps, world, {'route_id': 'R2'})
    assert ogp.record_sha(first) != ogp.record_sha(second)
    first['sha256'] = ogp.record_sha(first)
    assert ogp.record_sha(first) == first['sha256']


def test_write_capture_round_trip(tmp_path):
    gps, world = plan()
    record = ogp.serialize_plan(gps, world, {'route_id': 'R1', 'capture_id': 'R1'})
    path = ogp.write_capture(tmp_path, record)
    loaded = json.loads(path.read_text())
    assert loaded['sha256'] == ogp.record_sha(loaded)
    assert path.name == 'R1.json'


def test_hook_calls_original_and_writes_capture(tmp_path):
    calls = []

    class Agent:
        def set_global_plan(self, gps, world):
            calls.append((len(gps), len(world)))
            return 'ok'

    ogp.install_hook(Agent, tmp_path, metadata_fn=lambda agent: {'route_id': 'RouteX', 'capture_id': 'RouteX'})
    gps, world = plan()
    assert Agent().set_global_plan(gps, world) == 'ok'
    assert calls == [(5, 5)]
    assert (tmp_path / 'RouteX.json').is_file()


def test_hook_is_idempotent(tmp_path):
    class Agent:
        def set_global_plan(self, gps, world):
            return None

    first = ogp.install_hook(Agent, tmp_path)
    second = ogp.install_hook(Agent, tmp_path)
    assert first is second
