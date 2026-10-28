"""Read-only, bounded M2 annotation/map audit; never regenerates training labels.

Output lives outside the adapter and cache directories so producer/cache
identities and training contracts remain unchanged. Counts are object-frame
records, not unique actors. Individual official box masks diagnose clipping and
overwrites against existing NPZ labels; they are not new supervision targets.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


class AnnotationPaths:
    """Only explicitly configured files; no sensor-directory indexing or access."""
    def path(self, ref, modality):
        if modality != 'anno':
            raise RuntimeError('audit may only read annotations')
        return ref.route / 'anno' / ('%05d.json.gz' % ref.frame_id)


def attributes(box):
    # Scalars/vectors useful for independent metadata, without camera/LiDAR/bones payloads.
    fields = ('class', 'type_id', 'id', 'state', 'speed', 'brake', 'light_state', 'affects_ego',
              'location', 'center', 'extent', 'rotation', 'semantic_tags', 'base_type', 'age', 'gender',
              'road_id', 'lane_id', 'section_id', 'num_points', 'distance', 'trigger_volume_location',
              'trigger_volume_rotation', 'trigger_volume_extent')
    return {key: box[key] for key in fields if key in box}


def map_audit(config, hd_root):
    import numpy as np
    result = {}
    for town, spec in config['map_paths'].items():
        path = Path(spec['path'])
        identity = sha(path)
        if identity != spec['sha256']:
            raise ValueError('map changed since M2: ' + town)
        tree = ET.parse(path)
        roads = tree.findall('road')
        lanes = [lane for road in roads for section in road.findall('lanes/laneSection')
                 for side in ('left', 'right') for lane in section.findall(side + '/lane')]
        marks = [mark for lane in lanes for mark in lane.findall('roadMark')]
        objects = [o for road in roads for o in road.findall('objects/object')]
        vector_lanes = [v for lane in lanes for v in lane.findall('userData/vectorLane')]
        data = {'source': str(path), 'sha256': identity, 'carla_version': spec['carla_version'],
                'scope': 'entire configured map metadata, not dataset routes or local pixel counts',
                'roads': len(roads), 'noncenter_lane_section_records': len(lanes),
                'lane_types': dict(Counter(lane.get('type', 'missing') for lane in lanes)),
                'lane_id_signs': dict(Counter('positive' if int(l.get('id')) > 0 else 'negative' for l in lanes)),
                'explicit_lane_directions': dict(Counter(l.get('direction', 'absent') for l in lanes)),
                'vector_lane_travel_directions': dict(Counter(v.get('travelDir', 'absent') for v in vector_lanes)),
                'road_rules': dict(Counter(r.get('rule', 'absent') for r in roads)),
                'road_mark_types': dict(Counter(m.get('type', 'missing') for m in marks)),
                'road_mark_colors': dict(Counter(m.get('color', 'absent') for m in marks)),
                'road_mark_lane_change': dict(Counter(m.get('laneChange', 'absent') for m in marks)),
                'road_predecessors': sum(r.find('link/predecessor') is not None for r in roads),
                'road_successors': sum(r.find('link/successor') is not None for r in roads),
                'lane_predecessors': sum(l.find('link/predecessor') is not None for l in lanes),
                'lane_successors': sum(l.find('link/successor') is not None for l in lanes),
                'junctions': len(tree.findall('junction')),
                'junction_lane_links': len(tree.findall('junction/connection/laneLink')),
                'objects': len(objects), 'object_types': dict(Counter(o.get('type', 'absent') for o in objects)),
                'object_names_by_type': {kind: dict(Counter(o.get('name', '') for o in objects if o.get('type') == kind))
                                         for kind in ('parking', 'parkingSpace', 'crosswalk', 'barrier')},
                'object_geometry_fields': {key: sum(key in o.attrib for o in objects)
                                           for key in ('s', 't', 'zOffset', 'hdg', 'length', 'width', 'height')},
                'object_outline_count': sum(o.find('outline') is not None for o in objects),
                'parking_spaces': sum(len(o.findall('parkingSpace')) for o in objects),
                'signals': len(tree.findall('road/signals/signal'))}
        hd_path = hd_root / (town + '_HD_map.npz')
        with zipfile.ZipFile(hd_path) as archive:
            info = archive.getinfo('arr.npy')
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
            data['hd_map'] = {'path': str(hd_path), 'archive_bytes': hd_path.stat().st_size,
                              'uncompressed_bytes': info.file_size, 'shape': list(shape), 'dtype': str(dtype),
                              'payload_inspected': False}
        result[town] = data
    # One small existing trusted dataset archive suffices for actual schema evidence.
    # Do not unpickle multi-GB Town11/12/13 arrays.
    hd = result['Town02']['hd_map']
    if hd['uncompressed_bytes'] > 32 * 1024 * 1024:
        raise ValueError('bounded HD-map metadata probe exceeded 32MiB limit')
    with np.load(hd['path'], allow_pickle=True) as archive:
        pairs = archive['arr']
    lane_records = []
    triggers = []
    for road_id, road in pairs:
        for lane_id, records in road.items():
            (triggers if lane_id == 'Trigger_Volumes' else lane_records).extend(records)
    hd.update(payload_inspected=True, scope='Town02 only; other towns header-only',
              lane_record_count=len(lane_records), trigger_record_count=len(triggers),
              lane_record_fields=dict(Counter(key for row in lane_records for key in row)),
              marking_types=dict(Counter(str(row.get('Type')) for row in lane_records)),
              marking_colors=dict(Counter(str(row.get('Color')) for row in lane_records)),
              topology_types=dict(Counter(str(row.get('TopologyType')) for row in lane_records if 'TopologyType' in row)),
              trigger_types=dict(Counter(str(row.get('Type')) for row in triggers)),
              center_sample_point=(next((row['Points'][0] for row in lane_records if row.get('Type') == 'Center'), None)))
    return result


def main():
    import numpy as np
    import torch
    from shapely.geometry import box as rectangle
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from tools.b2d_adapter.frame_reader import FrameReader
    from tools.b2d_adapter.route_index import FrameRef
    from tools.b2d_adapter.coordinate_transform import world_to_ego, wrap_angle
    from tools.b2d_adapter.v1_builders import official_v1_module, v1_native_config
    from tools.b2d_adapter.v1_bev import bev_actor_name, bev_annotations
    from tools.b2d_adapter.errors import AdapterError
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'tools/training/diffusiondrive_v1/m2_subset.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/semantic_audit')
    parser.add_argument('--hd-map-root', type=Path, default=Path('/data/datasets/Bench2Drive-V0.0.4/maps'))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for forbidden in (ROOT / 'tools/b2d_adapter', ROOT / 'repositories', ROOT / 'outputs'):
        if output == forbidden or forbidden in output.parents:
            raise ValueError('audit outputs must be separate from model/adapter/training/cache files')
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    manifest_path = ROOT / config['prepared_manifest']
    manifest = json.loads(manifest_path.read_text())
    accepted = {row['token']: row for rows in manifest['accepted'].values() for row in rows}
    rejected = {row['token']: row for row in json.loads((manifest_path.parent / 'preparation_result.json').read_text())['failures']}
    model = official_v1_module('transfuser_config').TransfuserConfig()
    official = official_v1_module('transfuser_features').TransfuserTargetBuilder(model)
    geometry = v1_native_config().geometry
    torch.set_num_threads(2)
    protected = [ROOT / 'tools/b2d_adapter/v1_bev.py', ROOT / 'tools/b2d_adapter/v1_builders.py',
                 ROOT / 'tools/training/diffusiondrive_v1/runtime.py', ROOT / 'tools/training/diffusiondrive_v1/fit.py',
                 ROOT / 'tools/training/diffusiondrive_v1/prepared.py', manifest_path,
                 Path(official_v1_module('transfuser_config').__file__), Path(official_v1_module('transfuser_features').__file__),
                 Path(official_v1_module('transfuser_loss').__file__), Path(official_v1_module('transfuser_model_v2').__file__)]
    cache_paths = {token: ROOT / config['bev_cache_dir'] / (row['provenance']['targets']['bev']['cache_key'] + '.npz')
                   for token, row in accepted.items()}
    protected += list(cache_paths.values())
    before = {str(path): sha(path) for path in protected}
    fields = defaultdict(Counter)
    types = defaultdict(Counter)
    values = defaultdict(lambda: defaultdict(Counter))
    finite = defaultdict(Counter)
    counters = {name: Counter() for name in ('raw_classes', 'mapped_names', 'mapped_labels', 'ignored_classes',
                                             'unknown_classes', 'dispositions_all', 'dispositions_accepted', 'raster_visible_labels')}
    records = []
    frames = []
    raw_by_route = defaultdict(list)
    instance_keys = defaultdict(set)
    labels_by_name = {'vehicle': 5, 'pedestrian': 6, 'traffic_cone': 4, 'barrier': 4, 'czone_sign': 4, 'generic_object': 4}
    view = rectangle(0., -32., 32., 32.)
    reader = FrameReader(AnnotationPaths())
    for split, entries in config['data'].items():
        for entry in entries:
            for frame_id in entry['frames']:
                ref = FrameRef(ROOT / entry['route'], frame_id)
                raw = reader.read(ref)  # Explicitly no camera/LiDAR decoding.
                annotation_path = reader.index.path(ref, 'anno')
                annotation_sha = sha(annotation_path)
                raw_by_route[str(ref.route)].append(raw)
                existing_labels = None
                cache_validation = None
                provenance = None
                if ref.token in accepted:
                    row = accepted[ref.token]
                    expected = row['provenance']['targets']['bev']['cache_identity']['annotation_sha256']
                    if expected != annotation_sha:
                        raise ValueError('annotation changed since cache creation: ' + ref.token)
                    with np.load(cache_paths[ref.token], allow_pickle=False) as archive:
                        existing_labels = archive['labels'].copy()
                        metadata = json.loads(str(archive['metadata'].item()))
                    if hashlib.sha256(existing_labels.tobytes()).hexdigest() != metadata['labels_sha256']:
                        raise ValueError('BEV cache checksum failure: ' + ref.token)
                    cache_validation = metadata['validation']
                    provenance = metadata['provenance']
                frame_counts = Counter()
                for ordinal, actor in enumerate(raw.annotation['bounding_boxes']):
                    cls = actor.get('class', 'missing')
                    counters['raw_classes'][cls] += 1
                    fields[cls].update(actor.keys())
                    types[cls][str(actor.get('type_id', 'missing'))] += 1
                    instance_keys[cls].add((str(ref.route), str(actor.get('id', 'missing'))))
                    for key in ('state', 'speed', 'brake', 'light_state', 'affects_ego', 'base_type'):
                        if key in actor:
                            values[cls][key][json.dumps(actor[key], sort_keys=True)] += 1
                        if key in actor and numeric(actor[key]):
                            finite[cls][key] += 1
                    record = {'token': ref.token, 'split': split, 'ordinal': ordinal, 'attributes': attributes(actor),
                              'in_accepted_training_data': existing_labels is not None, 'annotation_sha256': annotation_sha}
                    try:
                        name = bev_actor_name(actor)
                    except AdapterError as exc:
                        counters['unknown_classes'][cls] += 1
                        record.update(mapped_name=None, disposition='unknown_mapping', error_code=exc.code, error=str(exc))
                        name = 'unknown'
                    if name is None:
                        counters['ignored_classes'][cls] += 1
                        record.update(mapped_name=None, disposition='mapping_ignored')
                    elif name != 'unknown':
                        label = labels_by_name[name]
                        counters['mapped_names'][name] += 1
                        counters['mapped_labels'][str(label)] += 1
                        record.update(mapped_name=name, mapped_label=label)
                        try:
                            one = SimpleNamespace(annotation={'bounding_boxes': [actor]}, actor_pose=raw.actor_pose, ego_box=raw.ego_box)
                            converted, _ = bev_annotations(one, geometry)
                            bbox = converted.boxes[0]
                            polygon = OrientedBox(StateSE2(float(bbox[0]), float(bbox[1]), float(bbox[6])),
                                                  float(bbox[3]), float(bbox[4]), float(bbox[5])).geometry
                            mask = official._compute_box_mask(converted, model.bev_semantic_classes[label][1])
                            pixels = int(mask.sum())
                            survives = int((mask & (existing_labels == label)).sum()) if existing_labels is not None else None
                            geometrically_intersects = bool(polygon.intersects(view))
                            if pixels == 0:
                                reason = 'raster_empty' if geometrically_intersects else 'outside_spatial_range'
                            elif existing_labels is None:
                                reason = 'rejected_full_sample'
                            elif survives == 0:
                                reason = 'overwritten_by_higher_class'
                            else:
                                reason = 'visible_class_pixels'
                                counters['raster_visible_labels'][str(label)] += 1
                            tilt = max(abs(float(v)) for v in actor['rotation'][:2])
                            record.update(disposition=reason, box_target_frame=bbox.tolist(),
                                          geometric_bev_intersection=geometrically_intersects, individual_mask_pixels=pixels,
                                          surviving_class_pixels=survives, above_planar_tilt_policy=tilt > geometry.max_planar_tilt_deg,
                                          center_height_from_ego_world_m=float(actor['center'][2] - raw.ego_box['location'][2]))
                            if existing_labels is not None and pixels:
                                record['covered_final_labels'] = dict(Counter(map(str, existing_labels[mask].tolist())))
                        except (AdapterError, ValueError, TypeError, KeyError) as exc:
                            record.update(disposition='invalid_geometry', error=str(exc))
                    if name is None:
                        # Optional spatial status for ignored objects does not change primary ignore reason.
                        try:
                            center = np.asarray(actor['center'], dtype=np.float64)
                            extent = np.asarray(actor['extent'], dtype=np.float64)
                            rotation = np.asarray(actor['rotation'], dtype=np.float64)
                            if any(v.shape != (3,) for v in (center, extent, rotation)) or not np.isfinite(np.r_[center, extent, rotation]).all() or (extent <= 0).any():
                                raise ValueError('invalid ignored-object geometry')
                            xy = world_to_ego(center[:2], raw.actor_pose)
                            p = OrientedBox(StateSE2(float(xy[0]-geometry.axle_dx_m), float(-(xy[1]-geometry.axle_dy_m)),
                                                    wrap_angle(raw.actor_pose.yaw-math.radians(rotation[2]))),
                                            float(2*extent[0]), float(2*extent[1]), float(2*extent[2])).geometry
                            record['ignored_geometry_intersects_bev'] = bool(p.intersects(view))
                        except (ValueError, TypeError, KeyError) as exc:
                            record['ignored_geometry_error'] = str(exc)
                    reason = record['disposition']
                    counters['dispositions_all'][reason] += 1
                    if existing_labels is not None:
                        counters['dispositions_accepted'][reason] += 1
                    frame_counts[reason] += 1
                    records.append(record)
                frame = {'token': ref.token, 'annotation_source': str(annotation_path), 'annotation_sha256': annotation_sha,
                         'accepted': existing_labels is not None, 'object_dispositions': dict(frame_counts),
                         'top_level_fields': sorted(raw.annotation),
                         'source_motion_flags': {key: raw.annotation[key] for key in
                             ('teleport_flag', 'disappear_flag', 'teleport_ids', 'disappear_ids') if key in raw.annotation},
                         'cached_validation': cache_validation,
                         'coverage_audit': {'schema': 'b2d-v1-independent-semantic-audit-v1',
                             'scope': 'M2 configured current annotation; object mapping, raster visibility and field availability',
                             'original_real_labels_validated_scope': 'map ego alignment and required target/schema checks, not full semantic coverage',
                             'complete_dataset_semantics_certified': False, 'dynamic_motion_ground_truth_certified': False}}
                if existing_labels is None:
                    frame['prior_rejection'] = rejected[ref.token]
                else:
                    frame['cached_class_pixels'] = np.bincount(existing_labels.flatten(), minlength=7).tolist()
                    frame['cached_map_coverage'] = provenance['coverage']
                frames.append(frame)
    # Consecutive existing pairs only; dt is the accepted adapter assumption, not new timestamp certification.
    pairs = defaultdict(list)
    for route, raw_frames in raw_by_route.items():
        raw_frames.sort(key=lambda raw: raw.ref.frame_id)
        for left, right in zip(raw_frames, raw_frames[1:]):
            if right.ref.frame_id != left.ref.frame_id + 1:
                continue
            previous = {(b['class'], str(b.get('id'))): b for b in left.annotation['bounding_boxes']}
            for actor in right.annotation['bounding_boxes']:
                cls = actor['class']
                key = cls, str(actor.get('id'))
                if cls not in ('vehicle', 'walker') or key not in previous:
                    continue
                old = previous[key]
                displacement = np.asarray(actor['location'], dtype=float) - np.asarray(old['location'], dtype=float)
                candidate_speed = float(np.linalg.norm(displacement[:2]) / geometry.source_dt_s)
                yaw = math.radians(float(actor['rotation'][2]))
                forward = float(np.dot(displacement[:2] / geometry.source_dt_s, [math.cos(yaw), math.sin(yaw)]))
                reported = float(actor['speed']) if numeric(actor.get('speed')) else None
                pairs[cls].append({'route': route, 'actor_id': key[1], 'frames': [left.ref.frame_id, right.ref.frame_id],
                                   'state': actor.get('state'), 'reported_speed_mps': reported,
                                   'candidate_displacement_speed_mps': candidate_speed, 'candidate_forward_speed_mps': forward,
                                   'absolute_forward_difference_mps': abs(reported-forward) if reported is not None else None,
                                   'absolute_magnitude_difference_mps': abs(abs(reported)-candidate_speed) if reported is not None else None,
                                   'source_teleport_flag': bool(left.annotation.get('teleport_flag') or right.annotation.get('teleport_flag')),
                                   'source_dt_s_assumed': geometry.source_dt_s})
    map_info = map_audit(config, args.hd_map_root)
    with (output / 'objects.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    (output / 'frames.json').write_text(json.dumps(frames, ensure_ascii=False, indent=2) + '\n')
    (output / 'map_inventory.json').write_text(json.dumps(map_info, ensure_ascii=False, indent=2) + '\n')
    (output / 'paired_motion.json').write_text(json.dumps(dict(pairs), ensure_ascii=False, indent=2) + '\n')
    dynamic = {}
    for cls in ('vehicle', 'walker', 'traffic_light', 'traffic_sign', 'ego_vehicle'):
        dynamic[cls] = {'records': counters['raw_classes'][cls], 'unique_route_scoped_ids': len(instance_keys[cls]),
                        'field_presence': dict(fields[cls]), 'finite_numeric_counts': dict(finite[cls]),
                        'selected_value_counts': {key: dict(value) for key, value in values[cls].items()}}
    after = {str(path): sha(path) for path in protected}
    if before != after:
        raise RuntimeError('protected training/model/cache files changed during audit')
    summary = {'schema': 'b2d-v1-independent-semantic-audit-v1', 'status': 'pass',
               'scope': {'routes': sum(len(es) for es in config['data'].values()), 'frames': len(frames),
                         'accepted_frames': len(accepted), 'rejected_frames': len(rejected),
                         'count_unit': 'object-frame records; unique IDs additionally scoped by full route path',
                         'annotations_only': True, 'camera_lidar_decoded': False, 'labels_regenerated': False,
                         'whole_dataset_scanned': False}, 'counts': {key: dict(value) for key, value in counters.items()},
               'raw_type_counts': {key: dict(value) for key, value in types.items()}, 'dynamic_fields': dynamic,
               'paired_motion_counts': {key: len(value) for key, value in pairs.items()},
               'raw_annotation_fields_observed': sorted(set(key for frame in frames for key in frame['top_level_fields'])),
               'protected_sha256': before, 'protected_files_unchanged': True,
               'manifest_source': str(manifest_path), 'config_source': str(args.config.resolve())}
    (output / 'audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (output / 'actor_mapping.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['raw_class', 'raw_type_id', 'records', 'bev_actor_name', 'v1_label', 'action'])
        for cls, counts in types.items():
            for type_id, count in sorted(counts.items()):
                try:
                    name = bev_actor_name({'class': cls, 'type_id': type_id})
                    action = 'ignored' if name is None else 'mapped'
                except AdapterError:
                    name = None
                    action = 'unknown_rejected'
                writer.writerow([cls, type_id, count, name, labels_by_name.get(name, ''), action])
    print(json.dumps({'scope': summary['scope'], 'counts': summary['counts'],
                      'paired_motion_counts': summary['paired_motion_counts'], 'protected_files_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
