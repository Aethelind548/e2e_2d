"""Minimal agent that records the real ``set_global_plan`` call of a session.

This module is imported *inside* the Bench2Drive leaderboard process
(``leaderboard_evaluator.py -a <this file> --track MAP``).  It exists for one
purpose: capture the exact arguments the evaluator hands to
``autonomous_agent.set_global_plan`` in a real route session, including the
downsampled view the agent itself can read.

It is not a driving agent for benchmarking.  With ``B2D_CAPTURE_DRIVE=1`` it
follows the captured plan with a small pure-pursuit controller so a route can
progress; otherwise it holds the brake and the evaluator times the route out.
Either way the capture happens at the first ``set_global_plan`` call.

Evidence only: nothing here modifies the plan, the evaluator, the dataset or the
frozen overlay.
"""
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = 'leaderboard-session-plan-v1'
VERSION = 'session-plan-agent-v1'
DRIVE_ENV = 'B2D_CAPTURE_DRIVE'
CAPTURE_DIR_ENV = 'B2D_CAPTURE_PLAN_DIR'

ROOT = Path('/home/aethelind/work/e2e_ad')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, '') else default


def _option_name(option):
    return str(getattr(option, 'name', option))


def _option_int(option):
    return int(getattr(option, 'value', option))


def _transform_point(transform):
    location = getattr(transform, 'location', transform)
    rotation = getattr(transform, 'rotation', None)
    return [
        float(location.x), float(location.y), float(location.z),
        float(getattr(rotation, 'yaw', 0.0)),
        float(getattr(rotation, 'pitch', 0.0)),
        float(getattr(rotation, 'roll', 0.0)),
    ]


def _gps_point(entry):
    if isinstance(entry, dict):
        return [float(entry['lat']), float(entry['lon']), float(entry.get('z', 0.0))]
    if isinstance(entry, (list, tuple)):
        alt = float(entry[2]) if len(entry) > 2 else 0.0
        return [float(entry[0]), float(entry[1]), alt]
    return [float(entry.latitude), float(entry.longitude), float(entry.altitude)]


def _unpack_gps(entry):
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry[0]
    return entry


def _arc_length(points):
    return sum(math.dist(a[:3], b[:3]) for a, b in zip(points, points[1:]))


def build_record(gps_route, world_route, downsampled_indices, metadata=None):
    """Serialize a captured ``set_global_plan`` call and the agent-visible view."""
    world_points = []
    world_options = []
    for transform, option in world_route:
        world_points.append(_transform_point(transform))
        world_options.append(_option_name(option))
    gps_points = []
    for entry, option in gps_route:
        gps_points.append(_gps_point(_unpack_gps(entry)) + [_option_name(option)])

    indices = list(downsampled_indices)
    ds_points = [_transform_point(world_route[i][0]) for i in indices]
    ds_options = [_option_name(world_route[i][1]) for i in indices]

    option_counts = {}
    for name in world_options:
        option_counts[name] = option_counts.get(name, 0) + 1
    ds_counts = {}
    for name in ds_options:
        ds_counts[name] = ds_counts.get(name, 0) + 1

    record = {
        'schema': SCHEMA,
        'capture_version': VERSION,
        'capture_point': 'leaderboard_evaluator.py -> agent_instance.set_global_plan(gps_route, route)',
        'raw': {
            'world_point_count': len(world_points),
            'gps_point_count': len(gps_points),
            'world_arc_length_m': round(_arc_length(world_points), 3),
            'road_option_counts': option_counts,
            'world_points': world_points,
            'world_road_options': world_options,
            'world_road_option_ints': [_option_int(o) for _, o in world_route],
            'gps_points': gps_points,
        },
        'agent_view': {
            'downsample_factor_m': 50,
            'downsampled_indices': indices,
            'point_count': len(ds_points),
            'arc_length_m': round(_arc_length(ds_points), 3),
            'road_option_counts': ds_counts,
            'world_points': ds_points,
            'world_road_options': ds_options,
            'gps_points': [_gps_point(_unpack_gps(gps_route[i])) + [_option_name(gps_route[i][1])] for i in indices],
        },
        'metadata': dict(metadata or {}),
        'certification_effect': 'SESSION_PLAN_CAPTURED (command agreement still required)',
    }
    return record


def record_sha(record):
    payload = dict(record)
    payload.pop('sha256', None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_record(out_dir, record):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record['sha256'] = record_sha(record)
    capture_id = record['metadata'].get('capture_id') or 'session'
    path = out_dir / f'{capture_id}.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
    with (out_dir / 'session_capture_index.jsonl').open('a') as stream:
        stream.write(json.dumps({
            'capture_id': capture_id,
            'sha256': record['sha256'],
            'route': record['metadata'].get('dataset_route'),
            'town': record['metadata'].get('town'),
            'raw_points': record['raw']['world_point_count'],
            'agent_points': record['agent_view']['point_count'],
        }, sort_keys=True) + '\n')
    return path


def collect_metadata(agent, downsampled_indices):
    """Describe where the plan came from and what the agent can actually read."""
    import carla

    metadata = {
        'capture_id': _env('B2D_CAPTURE_ID'),
        'dataset_route': _env('B2D_DATASET_ROUTE'),
        'route_id': _env('B2D_ROUTE_ID'),
        'town': _env('B2D_TOWN'),
        'weather': _env('B2D_WEATHER'),
        'xml_file': _env('B2D_XML_FILE'),
        'xml_id': _env('B2D_XML_ID'),
        'xml_sha256': _env('B2D_XML_SHA256'),
        'dataset_split': _env('B2D_DATASET_SPLIT'),
        'leaderboard_repo_head': _env('B2D_REPO_HEAD'),
        'leaderboard_evaluator_sha256': _env('B2D_EVALUATOR_SHA'),
        'route_scenario_sha256': _env('B2D_ROUTE_SCENARIO_SHA'),
        'route_manipulation_sha256': _env('B2D_ROUTE_MANIPULATION_SHA'),
        'carla_client_version': getattr(carla, '__version__', None),
        'carla_server_version': None,
        'map_name': None,
        'agent_module_sha256': _sha256(__file__),
        'capture_tool_sha256': _sha256(ROOT / 'tools/validation/b2d/leaderboard_session_capture.py')
        if (ROOT / 'tools/validation/b2d/leaderboard_session_capture.py').exists() else None,
        'drive_enabled': _env(DRIVE_ENV, '0') == '1',
        'downsample_factor_m': 50,
        'downsampled_index_count': len(list(downsampled_indices)),
    }
    try:
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        metadata['map_name'] = CarlaDataProvider.get_map().name
        metadata['carla_server_version'] = CarlaDataProvider.get_client().get_server_version()
    except Exception as error:  # pragma: no cover - only inside a live session
        metadata['probe_error'] = f'{type(error).__name__}: {error}'
    return metadata


def build_agent_view_probe(agent):
    """Record what navigation representations the agent can actually read."""
    def describe(value):
        if value is None:
            return {'type': 'NoneType', 'length': 0}
        try:
            length = len(value)
        except TypeError:
            length = None
        first = None
        if length:
            first = repr(value[0])[:120]
        return {'type': type(value).__name__, 'length': length, 'first': first}

    return {
        'attributes': {
            '_global_plan': describe(getattr(agent, '_global_plan', None)),
            '_global_plan_world_coord': describe(getattr(agent, '_global_plan_world_coord', None)),
            '_plan_gps_HACK': describe(getattr(agent, '_plan_gps_HACK', None)),
        },
        'semantics': {
            '_global_plan': 'downsampled (gps_point, RoadOption) pairs the agent consumes',
            '_global_plan_world_coord': 'downsampled (carla.Transform, RoadOption) pairs',
            '_plan_gps_HACK': 'undownsampled gps plan kept by the base class',
        },
    }


class SessionPlanCaptureAgent:
    """Leaderboard agent that captures plans; subclass created lazily.

    The CARLA base class is imported at module import time by the leaderboard, so
    this module is only importable inside that environment.  Tests exercise
    ``build_record`` directly.
    """


def make_agent_class():
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
    from leaderboard.utils.route_manipulation import downsample_route

    class CaptureAgent(AutonomousAgent):
        def setup(self, path_to_conf_file):
            self.track = Track.MAP
            self._capture_written = False

        def sensors(self):
            return []

        def set_global_plan(self, global_plan_gps, global_plan_world_coord):
            super().set_global_plan(global_plan_gps, global_plan_world_coord)
            if getattr(self, '_capture_written', False):
                return
            out_dir = _env(CAPTURE_DIR_ENV)
            if not out_dir:
                print('PLAN_CAPTURE_SKIPPED: B2D_CAPTURE_PLAN_DIR is unset', flush=True)
                return
            try:
                indices = downsample_route(global_plan_world_coord, 50)
                metadata = collect_metadata(self, indices)
                metadata['agent_view_probe'] = build_agent_view_probe(self)
                record = build_record(global_plan_gps, global_plan_world_coord, indices, metadata)
                path = write_record(out_dir, record)
                self._capture_written = True
                print(f'PLAN_CAPTURE_OK {path} sha256={record["sha256"]}', flush=True)
            except Exception as error:  # pragma: no cover - live session only
                print(f'PLAN_CAPTURE_FAILED {type(error).__name__}: {error}', flush=True)

        def run_step(self, input_data, timestamp):
            import carla
            control = carla.VehicleControl()
            if _env(DRIVE_ENV, '0') != '1' or self.hero_actor is None:
                control.brake = 1.0
                return control
            try:
                return self._pursuit_control(control, carla)
            except Exception as error:  # pragma: no cover - live session only
                print(f'PURSUIT_FAILED {type(error).__name__}: {error}', flush=True)
                control.brake = 1.0
                return control

        def _pursuit_control(self, control, carla):
            plan = getattr(self, '_global_plan_world_coord', None) or []
            if len(plan) < 2:
                control.brake = 1.0
                return control
            points = [(p[0].location.x, p[0].location.y) for p in plan]
            transform = self.hero_actor.get_transform()
            x, y = transform.location.x, transform.location.y
            yaw = math.radians(transform.rotation.yaw)

            index = getattr(self, '_pursuit_index', 0)
            while index < len(points) - 1 and math.dist((x, y), points[index]) < 6.0:
                index += 1
            self._pursuit_index = index

            target = points[index]
            for candidate in points[index:]:
                if math.dist((x, y), candidate) >= 8.0:
                    target = candidate
                    break
            dx, dy = target[0] - x, target[1] - y
            angle = math.atan2(dy, dx) - yaw
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            control.steer = max(-0.6, min(0.6, angle * 1.2))

            velocity = self.hero_actor.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            target_speed = 7.0 if abs(angle) < 0.5 else 4.0
            error = target_speed - speed
            if error > 0:
                control.throttle = min(0.65, 0.28 * error + 0.15)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(0.6, -0.2 * error)
            return control

    return CaptureAgent


_CACHED = {}


def get_entry_point():
    return 'CaptureAgent'


def __getattr__(name):
    if name == 'CaptureAgent':
        if 'cls' not in _CACHED:
            _CACHED['cls'] = make_agent_class()
        return _CACHED['cls']
    raise AttributeError(name)
