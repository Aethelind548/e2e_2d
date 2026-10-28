"""Content-addressed, complete V1 tensors; no raw-data access in training workers."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from tools.b2d_adapter.errors import AdapterError

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'b2d-v1-prepared-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def producer_identity(model_config, label_config=None, camera_policy='b2d-proportional-crop-v1'):
    import numpy
    import PIL
    import torch
    from tools.b2d_adapter.v1_builders import validate_v1_config, v1_native_config
    files = list((ROOT / 'tools/b2d_adapter').glob('*.py'))
    files += [Path(__file__), ROOT / 'tools/training/diffusiondrive_v1/runtime.py']
    files += [ROOT / 'repositories/DiffusionDrive/navsim/agents/diffusiondrive' / name
              for name in ('transfuser_features.py', 'transfuser_config.py')]
    result = {'schema': SCHEMA, 'adapter': validate_v1_config(v1_native_config(camera_policy), model_config),
            'implementation': {str(path.relative_to(ROOT)): file_digest(path) for path in sorted(files)},
            'libraries': {'torch': torch.__version__, 'numpy': numpy.__version__, 'pillow': PIL.__version__}}
    from tools.b2d_adapter.v1_label_rules import label_rules
    rules = label_rules(label_config)
    if rules.enriched:
        result['label_rules'] = rules.identity()
    if camera_policy == 'b2d-proportional-crop-v1':
        import cv2
        import torchvision
        result['camera_preprocessing'] = {'policy': camera_policy, 'source_wh': [1600,900],
            'crop_top_bottom': 23, 'crop_side_left_right': 346, 'stitched_wh': [3416,854],
            'resize_wh': [1024,256], 'interpolation': 'cv2.INTER_LINEAR', 'tensor': 'ToTensor RGB /255 once'}
        result['libraries'].update(opencv=cv2.__version__, torchvision=torchvision.__version__)
    return result


def sample_sources(index, current, native_config):
    """Every annotation consumed by continuity/trajectory checks and current sensors."""
    from tools.b2d_adapter.route_index import CAMERAS
    geometry = native_config.geometry
    end = current.frame_id + geometry.trajectory_count * geometry.stride
    paths = [index.path(index.ref(i), 'anno') for i in range(current.frame_id - 2, end + 1)]
    paths += [index.path(current, modality) for modality in (*CAMERAS, 'lidar')]
    return [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_digest(path)} for path in paths]


def validate_tensors(features, targets):
    import torch
    expected = {'features': {'camera_feature': ((3, 256, 1024), torch.float32),
                            'lidar_feature': ((1, 256, 256), torch.float32),
                            'status_feature': ((8,), torch.float32)},
                'targets': {'trajectory': ((8, 3), torch.float32), 'agent_states': ((30, 5), torch.float32),
                            'agent_labels': ((30,), torch.bool), 'bev_semantic_map': ((128, 256), torch.int64)}}
    for name, values in (('features', features), ('targets', targets)):
        if set(values) != set(expected[name]):
            raise AdapterError('invalid_prepared_tensors', name + ': wrong keys')
        for key, (shape, dtype) in expected[name].items():
            tensor = values[key]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise AdapterError('invalid_prepared_tensors', key + ': shape/dtype mismatch')
            if tensor.device.type != 'cpu' or not torch.isfinite(tensor).all():
                raise AdapterError('invalid_prepared_tensors', key + ': finite CPU tensor required')
    for key in ('camera_feature', 'lidar_feature'):
        if not ((features[key] >= 0).all() and (features[key] <= 1).all()):
            raise AdapterError('invalid_prepared_tensors', key + ': outside [0,1]')
    bev = targets['bev_semantic_map']
    if not ((bev >= 0).all() and (bev < 7).all()):
        raise AdapterError('invalid_prepared_tensors', 'BEV outside 0..6')


def load_artifact(row):
    import torch
    path = Path(row['tensor_path'])
    if file_digest(path) != row['tensor_sha256']:
        raise AdapterError('invalid_prepared_cache', str(path) + ': checksum mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if set(payload) != {'features', 'targets'}:
        raise AdapterError('invalid_prepared_cache', str(path) + ': payload keys')
    validate_tensors(payload['features'], payload['targets'])
    return payload['features'], payload['targets']


def compatible_producer(prior, current, rules):
    if digest(prior) == digest(current):
        return True
    # A reviewed code bridge never changes camera/native/model semantics.
    if digest(prior.get('adapter')) != digest(current.get('adapter')):
        return False
    camera_bridge_path = ROOT / 'tools/training/diffusiondrive_v1/camera_original_compatibility.json'
    policy = current['adapter']['native_policy']['parameters']['camera_policy']
    if policy == 'navsim-pixel-crop-v1' and camera_bridge_path.exists():
        camera_bridge = json.loads(camera_bridge_path.read_text())
        if any(row['historical_producer_sha256'] == digest(prior)
               and row['reviewed_current_producer_sha256'] == digest(current)
               and row['label_rules'] == rules.identity()
               and row['camera_policy'] == policy
               for row in camera_bridge['producer_bridges']):
            return True
    # Reviewed, exact M2 -> M4 original-label bridge. Both revisions are pinned;
    # future code/library changes fail closed. This never applies to enriched labels.
    bridge_path = ROOT / 'tools/training/diffusiondrive_v1/m4_original_compatibility.json'
    if rules.enriched or not bridge_path.exists():
        return False
    bridge = json.loads(bridge_path.read_text())
    return (digest(prior) == bridge['historical_producer_sha256']
            and digest(current) == bridge['reviewed_current_producer_sha256'])


class PreparedDataset:
    def __init__(self, rows):
        self.rows = rows
        self.tokens = [row['token'] for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return load_artifact(self.rows[index])


def prepared_datasets(path, config, model_config):
    """Use an explicitly reviewed accepted manifest; never silently skip bad samples."""
    from tools.training.diffusiondrive_v1.runtime import route_group
    from tools.b2d_adapter.v1_builders import BEV_CLASSES
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    from tools.b2d_adapter.v1_label_rules import label_rules
    from tools.training.diffusiondrive_v1.runtime import native_configuration
    rules = label_rules(config.get('label_rules'))
    camera_policy = native_configuration(config).camera_policy
    if (manifest.get('schema') != SCHEMA or not compatible_producer(
            manifest['producer'], producer_identity(model_config, rules.identity(), camera_policy), rules)):
        raise AdapterError('prepared_contract_mismatch', 'schema or current adapter implementation differs')
    if manifest['requested_data_sha256'] != digest(config['data']) or manifest['map_paths_sha256'] != digest(config['map_paths']):
        raise AdapterError('prepared_contract_mismatch', 'requested route split or map identity differs')
    tokens = set()
    groups = {}
    result = {}
    for split in ('train', 'validation'):
        requested = {str((ROOT / entry['route']).resolve()) + '::' + str(frame)
                     for entry in config['data'][split] for frame in entry['frames']}
        rows = manifest['accepted'][split]
        if not rows:
            raise AdapterError('empty_prepared_split', split)
        for row in rows:
            if row['token'] not in requested or row['token'] in tokens or row['cache_key'] != digest(row['identity']):
                raise AdapterError('invalid_prepared_manifest', row['token'])
            if row['identity']['producer_sha256'] != digest(manifest['producer']):
                raise AdapterError('invalid_prepared_manifest', 'sample producer mismatch')
            bev = row['provenance']['targets']['bev']
            if row['provenance']['features']['camera']['parameters']['camera_policy'] != camera_policy:
                raise AdapterError('prepared_contract_mismatch', 'sample camera policy differs: ' + row['token'])
            if label_rules(bev.get('label_rules')).identity() != rules.identity():
                raise AdapterError('prepared_contract_mismatch', 'sample label rules differ: ' + row['token'])
            town = row['route_group'].split(':')[0]
            spec = config['map_paths'][town]
            if not isinstance(spec, dict) or bev['map_sha256'] != spec['sha256'] or bev['map_version'] != spec['carla_version']:
                raise AdapterError('prepared_contract_mismatch', 'sample map differs: ' + row['token'])
            if tuple(bev['class_names']) != BEV_CLASSES or not row['provenance']['targets']['complete_training_targets']:
                raise AdapterError('invalid_prepared_manifest', 'complete seven-class targets required')
            if row['route_group'] != route_group(row['token'].rsplit('::', 1)[0]):
                raise AdapterError('invalid_prepared_manifest', 'route group mismatch')
            tokens.add(row['token'])
        groups[split] = {row['route_group'] for row in rows}
        result[split] = PreparedDataset(rows)
    if groups['train'] & groups['validation']:
        raise AdapterError('route_split_leakage', groups)
    return result, {'path': str(path), 'sha256': file_digest(path),
                    'accepted_counts': {key: len(value) for key, value in result.items()},
                    'failed_count': manifest['failed_count'], 'actual_route_groups': {k: sorted(v) for k, v in groups.items()}}
