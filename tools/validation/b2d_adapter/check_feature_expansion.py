"""Real expansion acceptance without modifying old tensors, navigation or producer."""
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def main():
    import torch
    from tools.b2d_adapter import FrameReader, RouteIndex
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1FeatureBuilder
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration
    from tools.training.diffusiondrive_v1.features_cache import FeatureCacheDataset, digest, feature_producer, file_digest, safe_path, durable_json
    from tools.training.diffusiondrive_v1.feature_jobs import FeatureJobsDataset, config_and_state, readonly_db
    old, old_state = config_and_state(ROOT / 'tools/training/diffusiondrive_v1/features_cache.json')
    new, new_state = config_and_state(ROOT / 'tools/training/diffusiondrive_v1/features_remaining_20261015.json')
    out = ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/features_expansion'
    protected = json.loads((out / 'protected_baseline_before.json').read_text())
    torch.set_num_threads(2)
    producer = feature_producer(new)
    assert digest(producer) == protected['baseline_producer_sha256']
    assert digest(feature_producer(old)) == digest(producer)
    assert not set(old['routes']) & set(new['routes'])
    db = readonly_db(old_state / 'index.sqlite')
    old_rows = [dict(r) for r in db.execute('SELECT * FROM samples ORDER BY sample_id')]
    assert digest(old_rows) == protected['baseline_rows_sha256']
    db.close()
    assert file_digest(ROOT / 'tools/training/diffusiondrive_v1/features_cache.json') == protected['baseline_config_sha256']
    assert all(file_digest(ROOT / p) == h for p, h in protected['core_files'].items())
    new_dataset = FeatureCacheDataset(new_state / 'index.sqlite', new['cache_roots'], producer)
    assert len(new_dataset) == 20
    for i, row in enumerate(new_dataset.rows):
        features = new_dataset[i]
        assert row['route'] in new['routes']
        path = safe_path(new['cache_roots'][row['root_id']], row['relative_path'])
        assert not path.is_relative_to(Path(old['cache_roots'][row['root_id']]['path']))
    row = new_dataset.rows[0]
    reader = FrameReader(RouteIndex.from_dataset_root(new['dataset_root'], row['route']))
    builder = DiffusionDriveV1FeatureBuilder(reader, native_configuration(new), model_configuration(new), new['lidar_decoder_python'])
    fresh = builder.build_features(builder.read_current(reader.index.ref(row['frame']))).tensors
    cached = new_dataset[0]
    assert all(torch.equal(fresh[k], cached[k]) for k in fresh)
    combined = FeatureJobsDataset(ROOT / 'tools/training/diffusiondrive_v1/features_jobs.json', producer)
    assert len(combined) == 4915 + 20
    assert len(set(combined.sample_ids)) == len(combined)
    selections = [combined.sample_ids[0], row['sample_id']]
    selected = FeatureJobsDataset(ROOT / 'tools/training/diffusiondrive_v1/features_jobs.json', producer, selections)
    assert selected.sample_ids == selections
    assert selected[0]['camera_feature'].shape == selected[1]['camera_feature'].shape == (3, 256, 1024)
    # Check all candidates, including failed/pending, for overlap without loading raw tensors.
    check = sqlite3.connect('file:' + str((new_state / 'index.sqlite').resolve()) + '?mode=ro', uri=True)
    check.execute('ATTACH DATABASE ? AS old', (str(old_state / 'index.sqlite'),))
    overlap = check.execute('SELECT COUNT(*) FROM samples n JOIN old.samples o USING(sample_id)').fetchone()[0]
    check.close()
    assert overlap == 0
    result = {'status': 'pass', 'new_samples_verified': 20, 'old_samples_all_readable': 4915,
              'old_full_read_check': 'old_cache_read_check.json', 'old_index_rows_unchanged': True,
              'old_config_and_numeric_modules_unchanged': True, 'producer_unchanged': digest(producer),
              'route_sets_disjoint': True, 'all_candidate_sample_id_overlap': overlap,
              'new_locations_separate': True, 'fresh_features_bit_equal': True,
              'combined_dataset_length': len(combined), 'combined_old_new_samples_readable': True,
              'new_features': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'min': float(v.min()),
                                  'max': float(v.max())} for k, v in cached.items()}}
    durable_json(out / 'small_validation.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
