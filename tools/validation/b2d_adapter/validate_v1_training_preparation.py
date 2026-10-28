"""Bounded real V1 regression and pilot cache/loader measurements after cleanup."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'repositories/DiffusionDrive'))
sys.dont_write_bytecode = True
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')
REPORT = ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/training_preparation'


def regression():
    import torch
    from torch.utils.data import default_collate
    from tools.b2d_adapter import FrameReader, RouteIndex
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1Dataset, v1_native_config, official_v1_module
    from tools.training.diffusiondrive_v1.runtime import fit_configuration, model_configuration, bev_provider, initialization_manifest, native_configuration
    from tools.training.diffusiondrive_v1.prepared import prepared_datasets, validate_tensors, atomic_json
    torch.set_num_threads(4)
    config, _ = fit_configuration(ROOT/'tools/training/diffusiondrive_v1/m2_subset.json')
    model = model_configuration(config)
    initialization = initialization_manifest(config, model)
    datasets, identity = prepared_datasets(ROOT/config['prepared_manifest'], config, model)
    provider = bev_provider(config)
    timings = []
    source_bytes = []
    samples = []
    for i in range(4):
        # Four known accepted real frames; no data-root scan or new acceptance claim.
        prepared = datasets['train'][i]
        token = datasets['train'].tokens[i]
        route, frame = token.rsplit('::', 1)
        reader = FrameReader(RouteIndex(route))
        dataset = DiffusionDriveV1Dataset(reader, [reader.index.ref(int(frame))],
                                        native_configuration(config), model, config['lidar_decoder_python'], provider)
        start = time.monotonic()
        actual = dataset[0]
        timings.append(time.monotonic()-start)
        validate_tensors(*actual)
        assert all(torch.equal(actual[side][key], prepared[side][key]) for side in (0,1) for key in actual[side]), token
        samples.append(actual)
        row = datasets['train'].rows[i]
        source_bytes.append(sum(s['bytes'] for s in row['identity']['sources']))
    checkpoint_path = ROOT/'outputs/diffusiondrive_v1/m2_fit_checked/checkpoints/last.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    agent = official_v1_module('transfuser_agent').TransfuserAgent(model, config['learning_rate'], checkpoint_path=None)
    agent.load_state_dict({k.removeprefix('agent.'): v for k,v in checkpoint['state_dict'].items()}, strict=True)
    del checkpoint
    # Official optimizer/scheduler must remain usable; no optimizer step here.
    optimizers = agent.get_optimizers()
    features, targets = default_collate(samples[:2])
    agent.train()
    with torch.no_grad():
        prediction = agent.forward(features, targets=targets)
        losses = agent.compute_loss(features, targets, prediction)
    assert prediction['bev_semantic_map'].shape == (2,7,128,256)
    assert prediction['trajectory'].shape == (2,8,3)
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values())
    deleted_imports = [name for name in sys.modules if name.startswith(('tools.bndp','tools.training.bndp','adapters.'))]
    assert not deleted_imports
    optimizer = optimizers['optimizer']
    scheduler = optimizers['lr_scheduler']
    if isinstance(scheduler, dict):
        scheduler = scheduler['scheduler']
    result = {'status': 'pass', 'old_manifest': identity, 'real_samples': datasets['train'].tokens[:4],
              'raw_vs_historical_prepared_all_tensors_equal': True, 'batch_size': 2,
              'seven_class_prediction_shape': list(prediction['bev_semantic_map'].shape),
              'checkpoint_strict_load': str(checkpoint_path), 'optimizer_steps': 0,
              'optimizer': type(optimizer).__name__, 'scheduler': type(scheduler).__name__,
              'weighted_official_losses': {k:float(v) for k,v in losses.items()},
              'deleted_research_imports': deleted_imports, 'initialization': initialization,
              'raw_loading': {'seconds_per_sample': timings, 'samples_per_second': len(timings)/sum(timings),
                              'source_bytes_per_sample': source_bytes,
                              'scope': 'known accepted raw frames, cached BEV, sensor decode+features+targets; first map parse included; filesystem page cache not flushed'}}
    atomic_json(REPORT/'regression.json', result)
    print(json.dumps({k:result[k] for k in ('status','raw_vs_historical_prepared_all_tensors_equal','optimizer','scheduler','raw_loading')}))


def benchmark(config_path):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tools.training.diffusiondrive_v1.runtime import fit_configuration, model_configuration
    from tools.training.diffusiondrive_v1.prepared import prepared_datasets, atomic_json
    torch.set_num_threads(4)
    config, groups = fit_configuration(config_path)
    model = model_configuration(config)
    datasets, identity = prepared_datasets(ROOT/config['prepared_manifest'], config, model)
    source = Subset(datasets['train'], list(range(min(128, len(datasets['train'])))))
    runs = []
    for workers in (0,2):
        options = {'num_workers': workers}
        if workers:
            options.update(multiprocessing_context='spawn', prefetch_factor=2, timeout=30)
        loader = DataLoader(source, batch_size=config['batch_size'], shuffle=False, **options)
        start = previous = time.monotonic()
        batches, count, intervals = 0, 0, []
        for features, targets in loader:
            now = time.monotonic()
            intervals.append(now-previous)
            previous = now
            count += len(features['status_feature'])
            batches += 1
        seconds = time.monotonic()-start
        remaining_count = count-min(config['batch_size'],count)
        runs.append({'workers': workers, 'samples': count, 'batches': batches, 'seconds': seconds,
                     'samples_per_second_including_startup_shutdown': count/seconds,
                     'first_batch_seconds': intervals[0],
                     'steady_samples_per_second_excluding_first_batch': remaining_count/sum(intervals[1:]),
                     'scope': 'CPU tensors + per-item SHA/dtype/shape/range validation; no model forward or GPU transfer; filesystem cache not flushed'})
    manifest = json.loads((ROOT/config['prepared_manifest']).read_text())
    rows = [r for values in manifest['accepted'].values() for r in values]
    tensor_bytes = sum(Path(r['tensor_path']).stat().st_size for r in rows)
    feature_bytes = sum(t.numel()*t.element_size() for t in datasets['train'][0][0].values())
    target_bytes = sum(t.numel()*t.element_size() for t in datasets['train'][0][1].values())
    compact_label_bytes = sum(p.stat().st_size for p in (ROOT/config['bev_cache_dir']).glob('*.npz'))
    result = {'status': 'pass', 'manifest': identity, 'counts': {k:len(v) for k,v in datasets.items()},
              'route_groups': {k:sorted(v) for k,v in groups.items()}, 'runs': runs,
              'prepared_tensor_bytes': tensor_bytes, 'per_sample_tensor_bytes': tensor_bytes/len(rows),
              'features_bytes_per_sample': feature_bytes, 'targets_bytes_per_sample': target_bytes,
              'bev_npz_cache_bytes': compact_label_bytes,
              'estimated_prepared_100k_frames_gib': tensor_bytes/len(rows)*100000/1024**3,
              'policy': 'Pilot full tensors only. Formal full-dataset tensor preparation remains gated on measured disk space and explicit manifest.'}
    atomic_json(REPORT/'loader_benchmark.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-config', type=Path)
    args = parser.parse_args()
    benchmark(args.benchmark_config) if args.benchmark_config else regression()
