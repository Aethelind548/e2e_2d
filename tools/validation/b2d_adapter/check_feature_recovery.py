"""Inject recoverable failures in designated test cache artifacts, never raw data."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))


def main():
    import torch
    from tools.training.diffusiondrive_v1.features_cache import FeatureCacheDataset, feature_producer, safe_path, active_mount, digest, durable_json
    from tools.training.diffusiondrive_v1.feature_service import alive
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['inject', 'verify'])
    args = p.parse_args()
    config = json.loads((ROOT / 'tools/training/diffusiondrive_v1/features_cache.json').read_text())
    state = ROOT / config['state_dir']
    evidence = ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/tensor_service'
    evidence.mkdir(exist_ok=True)
    baseline = evidence / 'recovery_before.pt'
    dataset = FeatureCacheDataset(state / 'index.sqlite', config['cache_roots'], feature_producer(config))
    if args.action == 'inject':
        if alive(state):
            raise RuntimeError('Stop worker before recovery test')
        root_a = [r for r in dataset.rows if r['root_id'] == 'cache_a']
        root_b = [r for r in dataset.rows if r['root_id'] == 'cache_b']
        chosen = [root_a[0], root_b[0], root_a[1]]
        selected = FeatureCacheDataset(state / 'index.sqlite', config['cache_roots'], feature_producer(config),
                                      sample_ids=[r['sample_id'] for r in chosen])
        torch.save({'rows': chosen, 'features': [selected[i] for i in range(3)]}, baseline)
        for r in chosen:
            active_mount(config['cache_roots'][r['root_id']], writing=True)
        # These are newly generated, purpose-specific validation cache files.
        safe_path(config['cache_roots'][chosen[0]['root_id']], chosen[0]['relative_path']).unlink()
        safe_path(config['cache_roots'][chosen[1]['root_id']], chosen[1]['relative_path']).write_bytes(b'interrupted-test')
        db = sqlite3.connect(state / 'index.sqlite')
        db.execute("UPDATE samples SET state='pending' WHERE sample_id=?", (chosen[2]['sample_id'],))
        db.commit()
        db.close()
        partial = safe_path(config['cache_roots']['cache_b'], str(Path(chosen[1]['relative_path']).parent / 'injected.partial'))
        partial.write_bytes(b'incomplete tensor; must not be accepted')
        durable_json(evidence / 'recovery_injection.json', {'ids': [r['sample_id'] for r in chosen],
                     'cases': ['missing verified file', 'checksum-corrupted file', 'rename-before-index-commit'],
                     'partial': str(partial), 'raw_data_modified': False})
        print('Injected designated cache recovery tests')
    else:
        saved = torch.load(baseline, map_location='cpu', weights_only=True)
        selected = FeatureCacheDataset(state / 'index.sqlite', config['cache_roots'], feature_producer(config),
                                      sample_ids=[r['sample_id'] for r in saved['rows']])
        for i, row in enumerate(selected.rows):
            assert row['root_id'] == saved['rows'][i]['root_id']
            assert row['relative_path'] == saved['rows'][i]['relative_path']
            assert all(torch.equal(v, selected[i][k]) for k, v in saved['features'][i].items())
        injection = json.loads((evidence / 'recovery_injection.json').read_text())
        assert not Path(injection['partial']).exists()
        status = json.loads((state / 'status.json').read_text())
        assert status['cleaned_partial_files'] == 1
        report = {'status': 'pass', 'tested': injection['cases'], 'partial_cleaned': True,
                  'same_fixed_locations': True, 'features_bit_equal': True, 'raw_data_modified': False,
                  'run_counts': status['counts']}
        durable_json(evidence / 'recovery_check.json', report)
        print(json.dumps(report))


if __name__ == '__main__':
    main()
