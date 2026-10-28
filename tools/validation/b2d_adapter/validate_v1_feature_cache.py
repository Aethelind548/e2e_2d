"""Independent real two-root reading and selected fresh Adapter comparisons."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def main():
    import torch
    from tools.b2d_adapter import FrameReader, RouteIndex, AdapterError
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1FeatureBuilder
    from tools.training.diffusiondrive_v1.features_cache import FeatureCacheDataset, feature_producer, index_summary, durable_json
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum', type=int, default=100)
    args = parser.parse_args()
    config = json.loads((ROOT / 'tools/training/diffusiondrive_v1/features_cache.json').read_text())
    state = ROOT / config['state_dir']
    torch.set_num_threads(2)
    producer = feature_producer(config)
    dataset = FeatureCacheDataset(state / 'index.sqlite', config['cache_roots'], producer)
    assert len(dataset) >= args.minimum
    before = time.monotonic()
    root_reads = {}
    selected = {}
    for i, row in enumerate(dataset.rows):
        start = time.monotonic()
        values = dataset[i]
        root = row['root_id']
        record = root_reads.setdefault(root, {'samples': 0, 'bytes': 0, 'read_seconds': 0.})
        record['samples'] += 1
        record['bytes'] += row['bytes']
        record['read_seconds'] += time.monotonic() - start
        if root not in selected:
            selected[root] = (i, values)
    assert set(root_reads) == {'cache_a', 'cache_b'}
    fresh_stats = []
    for root, (i, cached) in selected.items():
        row = dataset.rows[i]
        reader = FrameReader(RouteIndex.from_dataset_root(config['dataset_root'], row['route']))
        native = native_configuration(config)
        builder = DiffusionDriveV1FeatureBuilder(reader, native, model_configuration(config), config['lidar_decoder_python'])
        with patch('tools.b2d_adapter.v1_bev.B2DOpenDriveBEVBuilder.build_targets', side_effect=AssertionError('Features must not call BEV')):
            fresh = builder.build_features(builder.read_current(reader.index.ref(row['frame']))).tensors
        assert all(torch.equal(fresh[k], cached[k]) for k in fresh)
        fresh_stats.append({'root': root, 'sample_id': row['sample_id'], 'bit_equal_fresh': True,
                            'stats': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'min': float(v.min()),
                                          'max': float(v.max()), 'mean': float(v.mean()), 'std': float(v.std())}
                                      for k, v in cached.items()}})
    wrong = dict(producer, camera_rule=dict(producer['camera_rule'], side_left_right=416))
    try:
        FeatureCacheDataset(state / 'index.sqlite', config['cache_roots'], wrong)
    except AdapterError as exc:
        assert exc.code == 'feature_contract_mismatch'
    else:
        raise AssertionError('Wrong feature producer accepted')
    db = sqlite3.connect(state / 'index.sqlite')
    snapshot = index_summary(db)
    events = [json.loads(s) for s in (state / 'attempts.jsonl').read_text().splitlines()]
    writes = {root: [e for e in events if e.get('root_id') == root and e['status'] == 'complete'] for root in root_reads}
    for root, record in root_reads.items():
        record['end_to_end_write_samples_per_s'] = len(writes[root]) / sum(e['seconds'] for e in writes[root])
        record['read_mib_per_s'] = record['bytes'] / max(record['read_seconds'], .001) / 1024 ** 2
    db.close()
    result = {'status': 'pass', 'samples_verified': len(dataset), 'index': snapshot, 'root_reads': root_reads,
              'mean_sample_bytes': sum(r['bytes'] for r in dataset.rows) / len(dataset),
              'independent_two_disk_reader': True, 'map_generation_not_called': True,
              'wrong_camera_identity_rejected': True, 'fresh_samples': fresh_stats,
              'seconds': time.monotonic() - before, 'workers': 1}
    durable_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
