"""Capture and serialize the real online navigation plan handed to an agent.

In the Bench2Drive leaderboard the authoritative plan is what
``leaderboard_evaluator`` passes to ``set_global_plan``:

* ``route_scenario.route``   -- dense ``(carla.Transform, RoadOption)`` list from
  ``GlobalRoutePlanner.trace_route`` at 1 m hop, built from the route XML keypoints;
* ``route_scenario.gps_route`` -- the same points converted to ``(lat, lon, alt)``.

``AutonomousAgent.set_global_plan`` then downsamples the world list to <=50 m
spacing (always keeping RoadOption changes) before the agent sees it.  The XML
keypoints alone are therefore *not* the plan: they lack interpolation and
RoadOption.

This module only serializes/writes evidence.  It never modifies a plan, never
drives, and works without a CARLA import (duck-typed transforms), so its
serialization can be unit-tested on CPU.
"""
import hashlib
import json
import math
import os
from pathlib import Path

VERSION = 'online-global-plan-capture-v1'
SCHEMA = 'online-global-plan-v1'
ENV_VAR = 'B2D_CAPTURE_PLAN_DIR'


def _vec(value, names):
    for name in names:
        if hasattr(value, name):
            value = getattr(value, name)
        else:
            return None
    return value


def transform_fields(transform):
    """Return (x, y, z, yaw, pitch, roll) from a CARLA-like transform."""
    location = getattr(transform, 'location', transform)
    rotation = getattr(transform, 'rotation', None)
    x, y, z = float(location.x), float(location.y), float(location.z)
    yaw = float(getattr(rotation, 'yaw', 0.0)) if rotation is not None else 0.0
    pitch = float(getattr(rotation, 'pitch', 0.0)) if rotation is not None else 0.0
    roll = float(getattr(rotation, 'roll', 0.0)) if rotation is not None else 0.0
    return x, y, z, yaw, pitch, roll


def road_option_name(option):
    name = getattr(option, 'name', None)
    return str(name if name is not None else option)


def gps_fields(entry):
    """Accept a {lat, lon, z} dict, (lat, lon, alt) or a CARLA-like Location."""
    if isinstance(entry, dict):
        return float(entry['lat']), float(entry['lon']), float(entry.get('z', 0.0))
    if isinstance(entry, (list, tuple)):
        lat, lon = float(entry[0]), float(entry[1])
        alt = float(entry[2]) if len(entry) > 2 else 0.0
        return lat, lon, alt
    return float(entry.latitude), float(entry.longitude), float(entry.altitude)


def arc_length(points):
    total = 0.0
    for previous, current in zip(points, points[1:]):
        total += math.dist(previous[:3], current[:3])
    return total


def serialize_plan(gps_route, world_route, metadata=None):
    """Build the evidence record for one captured ``set_global_plan`` call."""
    world, world_options = [], []
    for transform, option in world_route:
        x, y, z, yaw, pitch, roll = transform_fields(transform)
        world.append([x, y, z, yaw, pitch, roll])
        world_options.append(road_option_name(option))
    gps = []
    for entry, option in gps_route:
        lat, lon, alt = gps_fields(entry[0] if isinstance(entry, tuple) and len(entry) == 2 else entry)
        gps.append([lat, lon, alt, road_option_name(option)])
    option_counts = {}
    for name in world_options:
        option_counts[name] = option_counts.get(name, 0) + 1
    record = {
        'schema': SCHEMA,
        'capture_version': VERSION,
        'source': 'leaderboard_evaluator.set_global_plan',
        'plan_kind': 'GlobalRoutePlanner.trace_route@1m + RoadOption',
        'gps_point_count': len(gps),
        'world_point_count': len(world),
        'world_arc_length_m': round(arc_length(world), 3),
        'road_option_counts': option_counts,
        'world_points': world,
        'world_road_options': world_options,
        'gps_points': gps,
        'metadata': dict(metadata or {}),
        'certification_effect': 'ONLINE_PLAN_CAPTURED (match verification still required)',
    }
    return record


def record_sha(record):
    payload = dict(record)
    payload.pop('sha256', None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_capture(out_dir, record):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record['sha256'] = record_sha(record)
    name = record['metadata'].get('capture_id') or record['metadata'].get('route_id') or 'plan'
    path = out_dir / f'{name}.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return path


def install_hook(agent_cls, out_dir, metadata_fn=None):
    """Wrap ``agent_cls.set_global_plan`` so every call is captured to disk.

    ``metadata_fn(agent)`` may return extra metadata (route id, town, weather,
    dataset version, carla version).  The original method is always called first,
    so agent behaviour is unchanged.
    """
    original = agent_cls.set_global_plan
    if getattr(original, '_b2d_capture', False):
        return original

    def wrapper(self, global_plan_gps, global_plan_world_coord):
        result = original(self, global_plan_gps, global_plan_world_coord)
        try:
            metadata = {'capture_id': None}
            if metadata_fn is not None:
                metadata.update(dict(metadata_fn(self)))
            metadata.setdefault('carla_version', getattr(self, 'carla_version', None))
            record = serialize_plan(global_plan_gps, global_plan_world_coord, metadata)
            write_capture(out_dir, record)
        except Exception as exc:  # capture must never break the evaluation
            print('PLAN_CAPTURE_FAILED: ' + type(exc).__name__ + ': ' + str(exc))
        return result

    wrapper._b2d_capture = True
    agent_cls.set_global_plan = wrapper
    return wrapper


class _Location:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Rotation:
    def __init__(self, yaw, pitch=0.0, roll=0.0):
        self.yaw, self.pitch, self.roll = yaw, pitch, roll


class _Transform:
    def __init__(self, location, rotation):
        self.location, self.rotation = location, rotation


def _self_test(out_dir):
    """CPU-only proof that serialization/hashing works without CARLA."""
    class _Option:
        def __init__(self, name):
            self.name = name

    world = [(_Transform(_Location(float(i), 0.0, 0.5), _Rotation(0.0)), _Option('LANEFOLLOW'))
             for i in range(5)]
    world[3] = (_Transform(_Location(3.0, 0.0, 0.5), _Rotation(0.0)), _Option('LEFT'))
    gps = [((10.0 + i * 1e-5, 20.0, 30.0), option) for i, (_, option) in enumerate(world)]
    record = serialize_plan(gps, world, {'route_id': 'SELFTEST', 'town': 'Town01'})
    path = write_capture(out_dir, record)
    loaded = json.loads(path.read_text())
    assert loaded['world_point_count'] == 5
    assert loaded['gps_point_count'] == 5
    assert abs(loaded['world_arc_length_m'] - 4.0) < 1e-6
    assert loaded['road_option_counts'] == {'LANEFOLLOW': 4, 'LEFT': 1}
    assert loaded['sha256'] == record_sha(loaded)
    print(json.dumps({'self_test': 'PASS', 'path': str(path), 'sha256': loaded['sha256']}))
    return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='serialize a synthetic plan and exit')
    parser.add_argument('--output', type=Path, default=Path('/tmp/b2d_plan_capture_selftest'))
    args = parser.parse_args()
    if args.self_test:
        return _self_test(args.output)
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
