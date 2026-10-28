"""Capture authoritative global plans from a running CARLA server.

The leaderboard builds the plan handed to ``set_global_plan`` with
``interpolate_trajectory``: ``GlobalRoutePlanner.trace_route`` on the real map at
1 m hop, producing ``(Transform, RoadOption)`` plus the GPS view.  This tool runs
the *same* CARLA APIs in the same order, using the local route XML keypoints, and
writes one evidence JSON per route through ``online_global_plan``.

Scope and honesty:

* The plan is built by the real CARLA map and the same planner the leaderboard
  uses, so it carries the true road topology and RoadOption decisions.
* The keypoints come from the local route XML, not from the original data-capture
  run; a capture is therefore labelled
  ``CARLA_GLOBAL_ROUTE_PLANNER_FROM_LOCAL_XML``.  It is stronger than a raw XML
  proxy but is not proof that the original run used the same file.
* Read-only with respect to the dataset: only CARLA is queried and evidence is
  written to an explicit output directory.

Run inside the CARLA client environment (``carla`` + ``agents`` on PYTHONPATH).
"""
import argparse
import hashlib
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validation.b2d.online_global_plan import record_sha, serialize_plan, write_capture  # noqa: E402

VERSION = 'carla-global-plan-capture-v1'
EARTH_RADIUS_EQUA = 6378137.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def latlon_ref(world):
    """Same reference extraction as leaderboard ``_get_latlon_ref``."""
    odr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(odr))
    lat_ref, lon_ref = 42.0, 2.0
    for opendrive in tree.iter('OpenDRIVE'):
        for header in opendrive.iter('header'):
            for georef in header.iter('geoReference'):
                if georef.text:
                    for item in georef.text.split(' '):
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def location_to_gps(lat_ref, lon_ref, location):
    """Same conversion as leaderboard ``_location_to_gps``."""
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y
    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    return {'lat': lat, 'lon': lon, 'z': location.z}


def downsample_ids(route, sample_factor=50):
    """Same selection rule as leaderboard ``downsample_route`` (keeps RoadOption changes)."""
    ids, prev_option, dist = [], None, 0.0
    for index, point in enumerate(route):
        curr_option = point[1]
        if index == 0:
            ids.append(index)
        elif curr_option != prev_option:
            ids.append(index)
            dist = 0.0
        elif index == len(route) - 1:
            ids.append(index)
            dist = 0.0
        else:
            dist += point[0].location.distance(route[index - 1][0].location)
        prev_option = curr_option
    return ids


def keypoints_from_xml(xml_dir, xml_file, town, route_id, carla):
    root = ET.parse(Path(xml_dir) / xml_file).getroot()
    for route in root.findall('route'):
        if route.get('town') == town and route.get('id') == route_id:
            points = []
            for position in route.find('waypoints').iter('position'):
                points.append(carla.Location(x=float(position.attrib['x']),
                                             y=float(position.attrib['y']),
                                             z=float(position.attrib.get('z', 0.0))))
            return points
    raise LookupError(f'route {town}/{route_id} not found in {xml_file}')


def build_plan(grp, keypoints, lat_ref, lon_ref):
    """Same loop as leaderboard ``interpolate_trajectory``."""
    route, gps_route = [], []
    for index in range(len(keypoints) - 1):
        trace = grp.trace_route(keypoints[index], keypoints[index + 1])
        for waypoint, connection in trace:
            route.append((waypoint.transform, connection))
            gps_route.append((location_to_gps(lat_ref, lon_ref, waypoint.transform.location), connection))
    return gps_route, route


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--association', type=Path,
                        default=ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261023_status_integration/online_certification_20261023/xml_preview_comparison/original_route_source_association.json')
    parser.add_argument('--xml-dir', type=Path,
                        default=Path('/home/aethelind/work/environment-build_test/Bench2Drive/leaderboard/data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--towns', default=None, help='comma-separated subset of towns')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true',
                        help='reuse an existing output directory and skip captured routes')
    parser.add_argument('--skip-load', action='store_true',
                        help='assume the server already runs the requested town map')
    parser.add_argument('--timeout', type=float, default=120.0)
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use a fresh explicit output directory (or pass --resume)')
    args.output.mkdir(parents=True, exist_ok=True)
    previous = {}
    if args.resume:
        for path in sorted(args.output.glob('*.json')):
            if path.name == 'capture_index.json':
                continue
            try:
                record = json.loads(path.read_text())
                meta = record.get('metadata') or {}
                route = meta.get('route')
                if route and record.get('schema') == 'online-global-plan-v1':
                    previous[route] = {
                        'route': route, 'town': meta.get('town'),
                        'xml_id': meta.get('xml_id'), 'capture': path.name,
                        'sha256': record.get('sha256'),
                        'world_points': record.get('world_point_count'),
                        'agent_view_points': meta.get('agent_view_point_count'),
                        'world_arc_length_m': record.get('world_arc_length_m'),
                        'road_option_counts': record.get('road_option_counts'),
                    }
            except (OSError, ValueError):
                continue

    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    association = json.loads(args.association.read_text())
    accepted = [record for record in association['records'] if record['accepted']]
    if args.towns:
        keep = set(args.towns.split(','))
        accepted = [record for record in accepted if record['town'] in keep]
    if args.limit:
        accepted = accepted[:args.limit]
    by_town = {}
    skipped = 0
    for record in accepted:
        if record['route'] in previous:
            skipped += 1
            continue
        by_town.setdefault(record['town'], []).append(record)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    server_version = client.get_server_version()
    index = {'schema': VERSION, 'carla_server_version': server_version,
             'client_library': 'carla-0.9.15', 'association': str(args.association),
             'association_sha256': sha256(args.association),
             'records': sorted(previous.values(), key=lambda r: r['route']),
             'skipped_existing': skipped}
    started = time.monotonic()
    failures = []
    for town, records in sorted(by_town.items()):
        if not records:
            continue
        print(json.dumps({'phase': 'load_world', 'town': town, 'routes': len(records)}), flush=True)
        if args.skip_load:
            world = client.get_world()
        else:
            world = client.load_world(town)
        carla_map = world.get_map()
        lat_ref, lon_ref = latlon_ref(world)
        planner = GlobalRoutePlanner(carla_map, 1.0)
        for record in records:
            route = record['route']
            try:
                keypoints = keypoints_from_xml(args.xml_dir, record['best']['source'],
                                               town, record['best']['xml_id'], carla)
                gps_route, route_plan = build_plan(planner, keypoints, lat_ref, lon_ref)
                keep = downsample_ids(route_plan, 50)
                metadata = {
                    'capture_id': route,
                    'route': route,
                    'route_id': record.get('route_id'),
                    'town': town,
                    'xml_file': record['best']['source'],
                    'xml_sha256': sha256(args.xml_dir / record['best']['source']),
                    'xml_id': record['best']['xml_id'],
                    'map_name': carla_map.name,
                    'carla_server_version': server_version,
                    'plan_kind': 'GlobalRoutePlanner.trace_route@1m+RoadOption',
                    'source': 'CARLA_GLOBAL_ROUTE_PLANNER_FROM_LOCAL_XML',
                    'capture_method': 'carla_client_interpolate_trajectory',
                    'keypoint_count': len(keypoints),
                    'agent_view_point_count': len(keep),
                    'lat_lon_ref': [lat_ref, lon_ref],
                }
                plan_record = serialize_plan(gps_route, route_plan, metadata)
                plan_record['sha256'] = record_sha(plan_record)
                path = write_capture(args.output, plan_record)
                index['records'].append({
                    'route': route, 'town': town, 'xml_id': record['best']['xml_id'],
                    'capture': path.name, 'sha256': plan_record['sha256'],
                    'world_points': plan_record['world_point_count'],
                    'agent_view_points': len(keep),
                    'world_arc_length_m': plan_record['world_arc_length_m'],
                    'road_option_counts': plan_record['road_option_counts'],
                })
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures.append({'route': route, 'town': town,
                                 'error_type': type(exc).__name__, 'error': str(exc)[:300]})
        print(json.dumps({'phase': 'town_done', 'town': town,
                          'captured': len(index['records']), 'failed': len(failures),
                          'seconds': round(time.monotonic() - started, 1)}), flush=True)

    index['captured'] = len(index['records'])
    index['failed'] = failures
    index['failure_count'] = len(failures)
    index['elapsed_seconds'] = round(time.monotonic() - started, 1)
    (args.output / 'capture_index.json').write_text(
        json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'captured': index['captured'], 'failed': index['failure_count'],
                      'output': str(args.output)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
