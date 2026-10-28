"""Offline/online-proxy Route Preview alignment for V1.1.

The only independent ordered plans available in the frozen audit are the two
exact XML matches.  They are original Bench2Drive route XML points, but this
runner cannot prove that the live CARLA Agent loaded the same plan.  Every
result is therefore tagged ``ORIGINAL_ROUTE_XML_PROXY`` and the live/online
reproduction flag remains NOT_VERIFIED.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.b2d_adapter.coordinate_transform import ActorPose, actor_to_reference  # noqa: E402
from tools.b2d_adapter.route_preview_contract import preview_command4  # noqa: E402

V11 = ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261022_191514_v11_panel'
ASSOCIATION = ROOT / 'outputs/diffusiondrive_v1/route_preview_data_audit/20261022_readonly/route_source_association.json'
EXAMPLES = ROOT / 'outputs/diffusiondrive_v1/route_preview_data_audit/20261022_readonly/route_preview_examples.json'
BASELINE = ROOT / 'outputs/diffusiondrive_v1/route_preview_reconstruction/20261022_183306_route_preview'


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dedupe(points):
    out = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        if not out or math.hypot(x - out[-1][0], y - out[-1][1]) > 1e-6:
            out.append((x, y))
    return out


def rear_from_example(record):
    if 'preview_origin' in record:
        rear = record['preview_origin']['rear_xyz']
        return (float(rear[0]), float(rear[1])), math.radians(float(record['actor_yaw_deg']))
    actor = record['actor_xyz']
    yaw = math.radians(float(record['actor_yaw_deg']))
    rear = actor_to_reference(ActorPose(actor[0], actor[1], yaw), (-1.388632877, 0.0))
    return (float(rear.x), float(rear.y)), yaw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--v11', type=Path, default=V11)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    args.output.mkdir(parents=True)
    v11 = {(row['route'], int(row['frame'])): row
           for row in jsonl(args.v11 / 'frame_route_preview.jsonl')}
    examples = {(row['route'], int(row['frame'])): row
                for row in json.loads(EXAMPLES.read_text())['records']}
    paths = {row['route']: dedupe(row['points'])
             for row in jsonl(BASELINE / 'route_reference_paths.jsonl')}
    association = json.loads(ASSOCIATION.read_text())
    records = []
    for item in association['routes']:
        matches = item.get('xml_matches') or []
        route = item['route']
        if not matches:
            continue
        frame_rows = [(key, row) for key, row in v11.items() if key[0] == route]
        if not frame_rows:
            continue
        for key, row in frame_rows:
            example = examples.get(key)
            if example is None:
                continue
            rear, yaw = rear_from_example(example)
            xml_points = dedupe(matches[0]['points'])
            xml_preview = preview_command4(xml_points, rear, yaw, max_projection_m=10.0)
            offline_points = paths.get(route)
            offline_source = 'FULL_RECONSTRUCTED_ROUTE'
            if offline_points is None or row['navigation_source'] != 'RECONSTRUCTED_ROUTE':
                candidates = example.get('candidates') or []
                index = row.get('selected_candidate_index')
                if index is None and len(candidates) == 1:
                    index = 0
                if index is not None and index < len(candidates):
                    offline_points = dedupe([(p[0], p[1]) for p in
                                             candidates[index].get('points_xyz', [])])
                    offline_source = 'LOCAL_REFERENCE_SEGMENT'
            offline_preview = (preview_command4(offline_points, rear, yaw, max_projection_m=10.0)
                               if offline_points and len(offline_points) >= 2 else None)
            status_command = json.loads(row['status_command4_json']) if row.get('status_command4_json') else None
            xml_command = xml_preview.get('command4')
            offline_command = offline_preview.get('command4') if offline_preview else None
            records.append({
                'route': route,
                'town': item.get('town'),
                'route_id': item.get('route_id'),
                'frame': key[1],
                'sample_id': row.get('sample_id'),
                'xml_source': matches[0].get('source'),
                'xml_point_count': len(xml_points),
                'xml_route_status': 'ORIGINAL_ROUTE_XML_PROXY',
                'offline_source': offline_source,
                'live_agent_source_certified': False,
                'online_input_reproducibility': 'NOT_VERIFIED',
                'same_rear_pose': True,
                'rear_xy': list(rear),
                'yaw_rad': yaw,
                'xml_preview_status': xml_preview.get('status'),
                'xml_preview_command4': xml_command,
                'xml_preview_point_xy': xml_preview.get('preview_point_xy'),
                'offline_preview_status': offline_preview.get('status') if offline_preview else None,
                'offline_preview_command4': offline_command,
                'offline_preview_point_xy': offline_preview.get('preview_point_xy') if offline_preview else None,
                'status_command4': status_command,
                'near_category': row.get('near_category'),
                'far_category': row.get('far_category'),
                'xml_vs_offline_command_match': xml_command == offline_command,
                'xml_vs_status_command_match': xml_command == status_command,
                'xml_vs_offline_point_distance_m': (
                    float(np.linalg.norm(np.asarray(xml_preview['preview_point_xy'])
                                         - np.asarray(offline_preview['preview_point_xy'])))
                    if xml_preview.get('preview_point_xy') and offline_preview
                    and offline_preview.get('preview_point_xy') else None),
                'xml_route_progress_m': xml_preview.get('route_progress_m'),
                'offline_route_progress_m': offline_preview.get('route_progress_m') if offline_preview else None,
                'xml_projection_distance_m': xml_preview.get('projection_distance_m'),
                'acceptance': ('PASS' if xml_preview.get('preview_valid')
                               and offline_preview and offline_preview.get('preview_valid')
                               and xml_command == offline_command else 'SUSPECT'),
            })
    (args.output / 'offline_online_alignment.json').write_text(
        json.dumps({'schema': 'route-preview-v11-alignment-v1',
                    'source_boundary': ('Original XML route is an offline proxy; '
                                        'live CARLA set_global_plan identity is NOT_VERIFIED.'),
                    'records': records}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'event': 'alignment_done', 'records': len(records),
                      'matched': sum(row['xml_vs_offline_command_match'] for row in records),
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
