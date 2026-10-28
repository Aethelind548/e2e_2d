"""Bounded real target roundtrips and strict pairing; sensor generation forbidden."""
import copy
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')


def main():
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from tools.b2d_adapter import AdapterError, FrameReader, RouteIndex
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1TargetBuilder
    from tools.training.diffusiondrive_v1.target_cache import generate, load_target_row, producer_identity, validate_targets
    from tools.training.diffusiondrive_v1.paired_dataset import paired_datasets
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration, bev_provider
    from tools.training.diffusiondrive_v1.features_cache import digest, durable_json, file_digest, feature_producer
    from tools.training.diffusiondrive_v1.prepared import load_artifact
    cfg = json.loads((ROOT / 'tools/training/diffusiondrive_v1/paired_check.json').read_text())
    state = ROOT / cfg['target_state_dir']; out = ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/targets_interface'
    torch.set_num_threads(2)
    model, native = model_configuration(cfg), native_configuration(cfg)
    producer = producer_identity(cfg, model, native)
    manifest_path = ROOT / cfg['paired_manifest']
    datasets, identity = paired_datasets(manifest_path, cfg, model)
    records = []; samples = []; protected_files = {}
    for dataset in datasets.values():
        for i, row in enumerate(dataset.rows):
            features, targets = dataset[i]; validate_targets(targets)
            payload = load_target_row(row['target_row'], cfg['target_root'], digest(producer))
            assert all(torch.equal(v, payload['targets'][k]) for k, v in targets.items())
            future = payload['future_frame_ids']
            assert future == [row['frame'] + n * native.geometry.stride for n in range(1, 9)]
            record = {'sample_id': row['sample_id'], 'route': row['route'], 'frame': row['frame'], 'future_frame_ids': future,
                      'future_seconds': [.5 * n for n in range(1, 9)], 'agent_positive': int(targets['agent_labels'].sum()),
                      'bev_pixels': torch.bincount(targets['bev_semantic_map'].flatten(), minlength=7).tolist(),
                      'current_anno_sha256': payload['current_annotation_sha256'],
                      'targets': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'finite': bool(torch.isfinite(v).all())} for k, v in targets.items()}}
            records.append(record); samples.append((row, features, targets))
            from tools.training.diffusiondrive_v1.feature_jobs import config_and_state
            fcfg, _ = config_and_state(ROOT / row['feature_config'])
            fpath = Path(fcfg['cache_roots'][row['feature_row']['root_id']]['path']) / row['feature_relative_path']
            protected_files[str(fpath)] = row['feature_row']['sha256']
    assert len(records) == 9 and any(r['bev_pixels'][4] for r in records) and any(r['bev_pixels'][6] for r in records)
    assert len(identity['unavailable']) == 1 and identity['unavailable'][0]['reason'] == 'insufficient_future'
    # Targets-only repeat must neither open image/LAZ data nor call a feature builder.
    original_read = FrameReader.read
    def annotation_only(self, ref, cameras=False, lidar=False):
        assert not cameras and not lidar
        return original_read(self, ref, cameras=False, lidar=False)
    with patch.object(FrameReader, 'read', annotation_only), patch(
            'tools.b2d_adapter.v1_builders.DiffusionDriveV1FeatureBuilder.build_features', side_effect=AssertionError('No Features rebuild')):
        repeated = generate(cfg, state)
        assert repeated['counts'] == {'success': 0, 'failed': 0, 'skipped': 9, 'processed': 0}
        row, _, cached = samples[0]
        reader = FrameReader(RouteIndex.from_dataset_root(cfg['dataset_root'], row['route']))
        fresh = DiffusionDriveV1TargetBuilder(reader, native, model, bev_provider(dict(cfg, bev_cache_dir=None))).compute_targets(reader.index.ref(row['frame']))
        assert all(torch.equal(v, cached[k]) for k, v in fresh.items())
    # Compare against already existing full prepared labels, no raw feature processing.
    full = json.loads((ROOT / 'outputs/diffusiondrive_v1/camera_proportional_check_prepared/b2d-proportional-crop-v1/accepted_manifest.json').read_text())
    full_rows = {r['token']: r for rows in full['accepted'].values() for r in rows}
    compared = 0
    for row, _, targets in samples:
        token = str(Path(cfg['dataset_root']) / row['route']) + '::' + str(row['frame'])
        if token in full_rows:
            _, full_targets = load_artifact(full_rows[token]); assert all(torch.equal(v, full_targets[k]) for k, v in targets.items()); compared += 1
    assert compared >= 3
    mismatches = {}
    manifest = json.loads(manifest_path.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        for case in ['frame', 'annotation', 'label_version']:
            bad = copy.deepcopy(manifest)
            if case == 'frame': bad['rows'][0]['target_row']['frame'] += 1
            if case == 'annotation': bad['rows'][0]['target_row']['current_anno_sha256'] = '0' * 64
            if case == 'label_version': bad['target_producer']['rules']['name'] = 'v1_semantic_enriched'
            path = Path(tmp) / (case + '.json'); path.write_text(json.dumps(bad))
            try: paired_datasets(path, cfg, model)
            except AdapterError as exc: mismatches[case] = exc.code
            else: raise AssertionError('Mismatched pair accepted: ' + case)
    assert all(file_digest(p) == h for p, h in protected_files.items())
    frozen = json.loads((out / 'features_protected_before.json').read_text())
    assert all(file_digest(ROOT / p) == h for p, h in frozen['core_files'].items())
    assert digest(feature_producer(cfg)) == frozen['baseline_producer_sha256']
    # At most three representative true BEV labels in one figure.
    choices = [next(s for s in samples if 'Town04' in s[0]['route']),
               next(s for s in samples if 'PedestrianCrossing_Town05' in s[0]['route']),
               next(s for s in samples if 'Town13' in s[0]['route'])]
    cmap = ListedColormap(['#eeeeee','#666666','#66c2a5','#ffcc44','#aa77bb','#4477dd','#ee5566'])
    fig, axes = plt.subplots(1, 3, figsize=(10, 8), constrained_layout=True)
    for ax, (row, _, values) in zip(axes, choices):
        im = ax.imshow(values['bev_semantic_map'].numpy().T, origin='lower', extent=(0,32,-32,32),
                       cmap=cmap, norm=BoundaryNorm(np.arange(-.5,7.5),7), interpolation='nearest', aspect='equal')
        ax.set_title(row['route'].split('_Weather')[0] + '\nframe ' + str(row['frame']), fontsize=8)
        ax.set_xlabel('forward x (m)'); ax.set_ylabel('left y (m)')
    fig.colorbar(im, ax=axes, ticks=list(range(7)), label='Original V1 class ID')
    fig.savefig(out / 'bev_three_samples.png', dpi=150); plt.close(fig)
    report = {'status': 'pass', 'valid_samples': records, 'paired': identity,
              'targets_repeat_skipped': 9, 'sensor_decodes_during_repeat': 0, 'fresh_target_bit_equal': True,
              'full_prepared_target_comparisons': compared, 'mismatches_rejected': mismatches,
              'features_files_unchanged': len(protected_files), 'feature_producer_unchanged': digest(feature_producer(cfg)),
              'visualized_samples': 3}
    durable_json(out / 'validation.json', report)
    print(json.dumps({'status':'pass','valid_targets':9,'unavailable':1,'comparisons':compared,'mismatch_rejections':mismatches}))


if __name__ == '__main__': main()
