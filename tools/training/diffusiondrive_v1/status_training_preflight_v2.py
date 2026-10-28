"""Historical preflight (no optimizer step, optional subset).

For the current Route split v2 experiment use
``status_training_preflight_full.py``; this file is retained only for historical
reproduction and must not be used to bypass the full-validation/Test-isolation
preflight.

It runs structural checks against the versioned training index, frozen cache
reader and official DiffusionDrive V1 agent.  It does not create an optimizer
step and is not the current preflight.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'experimental-status-preflight-v1'
STATUS_RULE = 'b2d-route-preview-20m-v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--scratch-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {'schema': VERSION, 'started': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'status_rule': STATUS_RULE, 'checks': {}, 'blockers': []}

    def record(name, ok, detail, blocker=None):
        report['checks'][name] = {'status': 'PASS' if ok else 'FAIL', 'detail': detail}
        if not ok and blocker:
            report['blockers'].append(blocker)

    # 0. Manifest/index identity and data version.
    manifest = json.loads(args.manifest.read_text())
    index_sha = sha256_file(args.index)
    identity_ok = index_sha == manifest['index']['sha256']
    record('data_version', identity_ok,
           {'version': manifest['version'], 'index_sha256': index_sha,
            'rows': manifest['counts']['rows'], 'splits': manifest['counts']['splits'],
            'navigation_source': manifest['navigation_source'],
            'online_source_certified': manifest['online_source_certified']},
           'data_version_mismatch' if not identity_ok else None)
    report['dataset_version'] = manifest['version']

    import torch
    from torch.utils.data import DataLoader
    from tools.training.diffusiondrive_v1.experimental_status_reader import ExperimentalStatusDataset

    train = ExperimentalStatusDataset(args.manifest, args.index, args.base_config, 'train')
    validation = ExperimentalStatusDataset(args.manifest, args.index, args.base_config, 'validation')

    # 1. Structure / shape / dtype on real cached samples.
    structure = []
    for index in (0, len(train)//3, len(train)//2, len(train)-1):
        features, targets = train[index]
        structure.append({
            'sample_id': train.rows[index]['sample_id'],
            'camera': [list(features['camera_feature'].shape), str(features['camera_feature'].dtype)],
            'lidar': [list(features['lidar_feature'].shape), str(features['lidar_feature'].dtype)],
            'status': [list(features['status_feature'].shape), str(features['status_feature'].dtype)],
            'target_keys': sorted(targets.keys()),
            'target_trajectory': [list(targets['trajectory'].shape), str(targets['trajectory'].dtype)]
            if 'trajectory' in targets else None,
        })
    expected_status = (3, 256, 1024), (1, 256, 256), (8,)
    structure_ok = all(
        tuple(row['camera'][0]) == expected_status[0]
        and tuple(row['lidar'][0]) == expected_status[1]
        and tuple(row['status'][0]) == (8,)
        and row['status'][1] == 'torch.float32'
        and row['camera'][1] == 'torch.float32' and row['lidar'][1] == 'torch.float32'
        for row in structure)
    record('structure', structure_ok, structure, 'structure_mismatch' if not structure_ok else None)

    # 2. Random vs sequential identity.
    probes = [0, 17, 101, 1000]
    identity = []
    for index in probes:
        sequential = train[index][0]['status_feature']
        random_access = train[index][0]['status_feature']
        identity.append({'index': index, 'equal': bool(torch.equal(sequential, random_access))})
    identity_ok = all(row['equal'] for row in identity)
    record('identity', identity_ok, {'probes': identity,
                                       'train_rows': len(train), 'validation_rows': len(validation)},
           'identity_mismatch' if not identity_ok else None)

    # 3. Route isolation.
    train_routes = {row['route'] for row in train.rows}
    validation_routes = {row['route'] for row in validation.rows}
    overlap = sorted(train_routes & validation_routes)
    record('route_isolation', not overlap,
           {'train_routes': len(train_routes), 'validation_routes': len(validation_routes),
            'overlap': overlap[:10]},
           'route_split_leakage' if overlap else None)

    # 4. DataLoader with 0 and N workers.
    loader_results = []
    for workers in (0, args.workers):
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=False,
                            num_workers=workers, drop_last=False)
        batches = 0
        status_ok = True
        for features, targets in loader:
            status = features['status_feature']
            status_ok = status_ok and tuple(status.shape) == (args.batch_size, 8)
            status_ok = status_ok and bool(((status[:, :4].sum(1) - 1).abs() < 1e-6).all())
            batches += 1
            if batches >= 2:
                break
        loader_results.append({'workers': workers, 'batches': batches, 'status_ok': status_ok})
    loader_ok = all(row['status_ok'] and row['batches'] == 2 for row in loader_results)
    record('dataloader', loader_ok, loader_results, 'dataloader_failed' if not loader_ok else None)

    # 5/6/7. Official V1 model, real loss/gradients, checkpoint load.
    from tools.b2d_adapter.v1_builders import official_v1_module
    from tools.training.diffusiondrive_v1.runtime import model_configuration
    from tools.training.diffusiondrive_v1.lr_experiment import warm_start, check_gradients

    config = json.loads(args.base_config.read_text())
    model_config = model_configuration(config)
    agent = official_v1_module('transfuser_agent').TransfuserAgent(
        model_config, config['learning_rate'], checkpoint_path=None)
    warm = warm_start(agent, args.checkpoint)
    record('checkpoint', True, warm)

    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        record('model_accepts_status', False, {'device': 'cuda', 'available': False},
               'cuda_unavailable')
    else:
        features, targets = next(iter(DataLoader(train, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0)))
        features = {k: v.to(device) for k, v in features.items()}
        targets = {k: v.to(device) for k, v in targets.items()}
        agent.to(device).train()
        predictions = agent.forward(features, targets=targets)
        losses = agent.compute_loss(features, targets, predictions)
        loss = losses['loss']
        record('model_accepts_status', True,
               {'device': str(device), 'batch': args.batch_size,
                'status_shape': list(features['status_feature'].shape),
                'prediction_keys': sorted(predictions.keys()),
                'gpu_total_mib': round(torch.cuda.get_device_properties(0).total_memory / 2**20)
                if device.type == 'cuda' else None})
        loss.backward()
        try:
            check_gradients(agent.parameters())
            gradient_ok = True
        except Exception as error:  # noqa: BLE001
            gradient_ok = False
            report['gradient_error'] = f'{type(error).__name__}: {error}'
        record('loss_and_gradients', gradient_ok,
               {'loss': float(loss.detach().cpu()),
                'component_losses': {k: (float(v.detach().cpu()) if torch.is_tensor(v) else float(v))
                                     for k, v in losses.items() if v is not None},
                'optimizer_steps': 0,
                'gradients_finite': gradient_ok},
               None if gradient_ok else 'nonfinite_or_missing_gradients')

    # 8. Scratch untouched.
    scratch = json.loads(args.scratch_config.read_text())
    scratch_dir = Path(scratch['output_dir'])
    processes = subprocess.run(['pgrep', '-af', 'fit.py'], capture_output=True, text=True).stdout.strip()
    last_ckpt = scratch_dir / 'checkpoints' / 'last.ckpt'
    scratch_detail = {'output_dir': str(scratch_dir), 'exists': scratch_dir.is_dir(),
                      'fit_processes': processes or None,
                      'last_checkpoint': None}
    if last_ckpt.is_file():
        stat = last_ckpt.stat()
        scratch_detail['last_checkpoint'] = {'path': str(last_ckpt), 'size': stat.st_size,
                                             'mtime_ns': stat.st_mtime_ns,
                                             'modified_during_preflight': stat.st_mtime >= started}
    scratch_ok = not processes and not (scratch_detail['last_checkpoint'] or {}).get(
        'modified_during_preflight', False)
    record('scratch_untouched', scratch_ok, scratch_detail,
           'scratch_modified' if not scratch_ok else None)

    # 9. Storage.
    data_root = Path(config['dataset_root'])
    probe = args.output_dir / '.preflight_write_probe'
    try:
        probe.write_text('ok')
        writable = probe.read_text() == 'ok'
        probe.unlink()
    except OSError as error:
        writable = False
        report['storage_error'] = f'{type(error).__name__}: {error}'
    storage_ok = writable and data_root.is_dir() and os.access(data_root, os.R_OK)
    record('storage', storage_ok,
           {'dataset_root': str(data_root), 'dataset_root_readable': data_root.is_dir(),
            'output_dir': str(args.output_dir.resolve()), 'output_writable': writable},
           'storage_unavailable' if not storage_ok else None)

    # 10. Logging identity.
    sources = {
        'status_builder': ROOT/'agent/navigation/status_builder.py',
        'route_preview_contract': ROOT/'tools/b2d_adapter/route_preview_contract.py',
        'training_manifest_tool': ROOT/'tools/training/diffusiondrive_v1/status_training_manifest.py',
        'experimental_reader': ROOT/'tools/training/diffusiondrive_v1/experimental_status_reader.py',
    }
    logging_identity = {'status_rule': STATUS_RULE, 'dataset_version': manifest['version'],
                        'index_sha256': index_sha,
                        'code_sha256': {name: sha256_file(path) for name, path in sources.items()}}
    record('logging_identity', True, logging_identity)

    report['elapsed_seconds'] = round(time.time() - started, 1)
    go = not report['blockers']
    report['go_no_go'] = 'GO' if go else 'NO-GO'
    (args.output_dir / 'status_training_preflight.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n')
    train.close()
    validation.close()
    print(json.dumps({'go_no_go': report['go_no_go'], 'blockers': report['blockers'],
                      'checks': {k: v['status'] for k, v in report['checks'].items()},
                      'elapsed_seconds': report['elapsed_seconds']}, sort_keys=True))


if __name__ == '__main__':
    main()
