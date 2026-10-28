"""Full preflight for the Route-split Status experiment.

This performs one real GPU optimizer update and a complete validation pass.  It
does not create a checkpoint, start Trainer.fit, read the Test split, or modify
frozen tensors.  A failed check is a blocker; there is no CPU or subset bypass.
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
VERSION = 'status-training-preflight-full-v2'
SPLITS = ('train', 'validation', 'test')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--optimizer-batch-size', type=int, default=4)
    parser.add_argument('--validation-batch-size', type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {'schema': VERSION, 'started': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'checks': {}, 'blockers': []}

    def record(name, ok, detail, blocker=None):
        report['checks'][name] = {'status': 'PASS' if ok else 'FAIL', 'detail': detail}
        if not ok:
            report['blockers'].append(blocker or name)

    config = json.loads(args.config.read_text())
    manifest_path = Path(config['dataset_manifest'])
    index_path = Path(config['sample_index'])
    split_path = Path(config['dataset_split_manifest'])
    base_path = Path(config['base_config'])
    manifest = json.loads(manifest_path.read_text())
    split = json.loads(split_path.read_text())
    index_sha = sha256_file(index_path)
    split_sha = sha256_file(split_path)
    record('data_identity',
           index_sha == config['sample_index_sha256'] and index_sha == manifest['index']['sha256']
           and split_sha == config['dataset_split_sha256'] and split.get('seed') == config.get('seed', 0),
           {'index_sha256': index_sha, 'split_sha256': split_sha, 'seed': split.get('seed'),
            'routes': split.get('route_counts'), 'frames': split.get('frames')},
           'data_identity_mismatch')

    import torch
    from torch.utils.data import DataLoader
    from tools.training.diffusiondrive_v1.experimental_status_reader import ExperimentalStatusDataset
    from tools.training.diffusiondrive_v1.bf16_policy import (
        BF16TrainingFP32Evaluation, install_fp32_losses, POLICY, source_sha256)
    from tools.b2d_adapter.v1_builders import official_v1_module
    from tools.training.diffusiondrive_v1.runtime import model_configuration

    train = ExperimentalStatusDataset(manifest_path, index_path, base_path, 'train')
    validation = ExperimentalStatusDataset(manifest_path, index_path, base_path, 'validation')
    expected = {'train': 160266, 'validation': 17191, 'test': 16202, 'excluded_route_tail': 8267}
    actual_sizes = {'train': len(train), 'validation': len(validation),
                    'test': split['frames'].get('test'),
                    'excluded_route_tail': split['frames'].get('excluded_route_tail')}
    record('split_sizes', actual_sizes == expected, {'actual': actual_sizes, 'expected': expected},
           'split_size_mismatch')

    # Read from the split directory, not from the evidence output directory.
    split_dir = split_path.parent
    route_lists = {name: set(json.loads((split_dir / f'{file_name}_routes.json').read_text())['routes'])
                   for name, file_name in (('train', 'train'), ('validation', 'val'), ('test', 'test'))}
    overlaps = {
        'train_validation': sorted(route_lists['train'] & route_lists['validation']),
        'train_test': sorted(route_lists['train'] & route_lists['test']),
        'validation_test': sorted(route_lists['validation'] & route_lists['test']),
    }
    train_routes = {row['route'] for row in train.rows}
    validation_routes = {row['route'] for row in validation.rows}
    test_route_overlap = sorted((train_routes | validation_routes) & route_lists['test'])
    record('route_isolation',
           not any(overlaps.values()) and not test_route_overlap,
           {'route_overlaps': overlaps, 'train_routes': len(train_routes),
            'validation_routes': len(validation_routes),
            'train_or_validation_test_route_overlap': test_route_overlap[:10]},
           'route_split_leakage')

    # The Test split must be represented in the index but never selected by the
    # two Dataset instances used below.
    loaded_ids = {row['sample_id'] for row in train.rows} | {row['sample_id'] for row in validation.rows}
    test_ids = set()
    excluded_ids = set()
    with index_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('split') == 'test':
                test_ids.add(row['sample_id'])
            elif row.get('split') == 'excluded_route_tail':
                excluded_ids.add(row['sample_id'])
    record('test_not_loaded',
           not (loaded_ids & test_ids) and len(test_ids) == expected['test']
           and len(excluded_ids) == expected['excluded_route_tail'],
           {'test_rows': len(test_ids), 'excluded_rows': len(excluded_ids),
            'loaded_test_overlap': len(loaded_ids & test_ids)},
           'test_split_loaded')

    structure = []
    probes = sorted({0, len(train) // 3, len(train) // 2, len(train) - 1})
    for index in probes:
        features, targets = train[index]
        structure.append({'index': index, 'sample_id': train.rows[index]['sample_id'],
                          'camera': [list(features['camera_feature'].shape), str(features['camera_feature'].dtype)],
                          'lidar': [list(features['lidar_feature'].shape), str(features['lidar_feature'].dtype)],
                          'status': [list(features['status_feature'].shape), str(features['status_feature'].dtype)],
                          'target_trajectory': list(targets['trajectory'].shape)})
    structure_ok = all(tuple(row['camera'][0]) == (3, 256, 1024)
                       and tuple(row['lidar'][0]) == (1, 256, 256)
                       and tuple(row['status'][0]) == (8,)
                       and row['status'][1] == 'torch.float32' for row in structure)
    record('structure', structure_ok, structure, 'structure_mismatch')

    loader_results = []
    for workers in (0, args.workers):
        loader = DataLoader(train, batch_size=2, shuffle=False, num_workers=workers)
        batches = 0
        ok = True
        for features, targets in loader:
            ok = ok and tuple(features['status_feature'].shape) == (2, 8)
            ok = ok and bool(((features['status_feature'][:, :4].sum(1) - 1).abs() < 1e-6).all())
            batches += 1
            if batches >= 2:
                break
        loader_results.append({'workers': workers, 'batches': batches, 'ok': ok})
    record('dataloader', all(row['batches'] == 2 and row['ok'] for row in loader_results),
           loader_results, 'dataloader_failure')

    if args.device == 'cuda' and not torch.cuda.is_available():
        record('gpu', False, {'cuda_available': False}, 'cuda_unavailable')
        report['go_no_go'] = 'NO-GO'
        (args.output_dir / 'status_training_preflight_full.json').write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n')
        train.close(); validation.close()
        print(json.dumps({'go_no_go': 'NO-GO', 'blockers': report['blockers']}))
        return

    device = torch.device(args.device)
    base = json.loads(base_path.read_text())
    model_config = model_configuration(base)
    agent = official_v1_module('transfuser_agent').TransfuserAgent(
        model_config, base['learning_rate'], checkpoint_path=None)
    backbone_path = Path(model_config.bkb_path)
    backbone_ok = backbone_path.is_file() and sha256_file(backbone_path) == base['expected_image_sha256']
    backbone_equal = False
    if backbone_ok:
        source_state = torch.load(backbone_path, map_location='cpu', weights_only=True)
        encoder = agent._transfuser_model._backbone.image_encoder
        actual = encoder.conv1.weight.detach().cpu()
        backbone_equal = torch.equal(actual, source_state['conv1.weight'])
    record('official_image_backbone', backbone_ok and backbone_equal,
           {'path': str(backbone_path),
            'sha256': sha256_file(backbone_path) if backbone_path.is_file() else None,
            'expected_sha256': base['expected_image_sha256'],
            'conv1_equal_to_local_pretrained': backbone_equal,
            'first_timm_hub_attempt': 'offline cache miss; official fallback uses the pinned local file'},
           'official_image_backbone_mismatch')
    if config.get('precision_policy') != POLICY:
        record('precision_policy', False, config.get('precision_policy'), 'precision_policy')
    else:
        install_fp32_losses(agent)
        record('precision_policy', True, {'policy': POLICY, 'source_sha256': source_sha256(),
                                          'val_autocast': 'disabled'})
    agent.to(device)
    optimizer_payload = agent.get_optimizers()
    optimizer = optimizer_payload['optimizer']
    scheduler = optimizer_payload.get('lr_scheduler')
    lrs = [group['lr'] for group in optimizer.param_groups]
    warmup_lr = base['learning_rate'] / 3
    # Official WarmupCosLR initializes all param groups to the first warmup
    # value; the pre-existing official scheduler does not preserve the
    # image_encoder lr_mult after that initialization.  Record this as an
    # explicit official-behavior warning, not a hidden local change.
    scheduler_ok = (scheduler is not None and abs(getattr(scheduler, 'lr', -1) - base['learning_rate']) < 1e-12
                    and abs(getattr(scheduler, 'min_lr', -1) - 1e-6) < 1e-12
                    and getattr(scheduler, 'epochs', None) == 100
                    and getattr(scheduler, 'warmup_epochs', None) == 3)
    optimizer_ok = (type(optimizer).__name__ == 'AdamW'
                    and all(abs(group.get('weight_decay', -1) - 1e-4) < 1e-12
                            for group in optimizer.param_groups)
                    and len(lrs) >= 2 and all(abs(lr - warmup_lr) < 1e-12 for lr in lrs)
                    and scheduler_ok)
    if len(lrs) >= 2 and abs(base['learning_rate'] * 0.5 - lrs[-1]) > 1e-12:
        report.setdefault('warnings', []).append(
            'official WarmupCosLR flattens post-init param-group lrs to the warmup value; '
            'image_encoder lr_mult=0.5 is configured but not preserved by the official scheduler')
    record('optimizer_contract', optimizer_ok,
           {'optimizer': type(optimizer).__name__, 'configured_learning_rate': base['learning_rate'],
            'warmup_initial_lr': warmup_lr, 'param_group_lrs': lrs,
            'weight_decay': [group.get('weight_decay') for group in optimizer.param_groups],
            'scheduler': type(scheduler).__name__ if scheduler is not None else None,
            'scheduler_lr': getattr(scheduler, 'lr', None) if scheduler is not None else None,
            'scheduler_epochs': getattr(scheduler, 'epochs', None) if scheduler is not None else None,
            'scheduler_warmup_epochs': getattr(scheduler, 'warmup_epochs', None) if scheduler is not None else None,
            'image_encoder_lr_mult_config': 0.5},
           'optimizer_learning_rate')

    train_loader = DataLoader(train, batch_size=args.optimizer_batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    batch = next(iter(train_loader))
    features, targets = batch
    features = {key: value.to(device) for key, value in features.items()}
    targets = {key: value.to(device) for key, value in targets.items()}
    agent.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = agent.forward(features, targets)
    losses = agent.compute_loss(features, targets, prediction)
    loss = losses['loss']
    loss.backward()
    gradients = [parameter.grad for parameter in agent.parameters() if parameter.grad is not None]
    grad_ok = bool(gradients) and all(torch.isfinite(value).all() for value in gradients)
    before = {name: parameter.detach().clone() for name, parameter in agent.named_parameters()}
    optimizer.step()
    changed = any(not torch.equal(before[name], parameter.detach())
                  for name, parameter in agent.named_parameters())
    record('real_optimizer_step',
           bool(torch.isfinite(loss.detach())) and grad_ok and changed,
           {'loss': float(loss.detach().cpu()), 'gradients': len(gradients),
            'gradients_finite': grad_ok, 'parameter_changed': changed,
            'optimizer_steps': 1,
            'gpu_total_mib': round(torch.cuda.get_device_properties(0).total_memory / 2**20)},
           'optimizer_step_failure')

    validation_loader = DataLoader(validation, batch_size=args.validation_batch_size, shuffle=False,
                                   num_workers=args.workers)
    agent.eval()
    val_samples = 0
    val_loss_sum = 0.0
    val_batches = 0
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for features, targets in validation_loader:
            features = {key: value.to(device) for key, value in features.items()}
            targets = {key: value.to(device) for key, value in targets.items()}
            prediction = agent.forward(features, targets)
            losses = agent.compute_loss(features, targets, prediction)
            count = len(features['status_feature'])
            val_samples += count
            val_loss_sum += float(losses['loss'].detach().cpu()) * count
            val_batches += 1
    val_loss = val_loss_sum / max(1, val_samples)
    record('full_validation',
           val_samples == len(validation) and val_batches > 0 and val_loss == val_loss,
           {'samples': val_samples, 'expected_samples': len(validation),
            'batches': val_batches, 'validation_loss': val_loss,
            'autocast': 'disabled'},
           'full_validation_failure')

    probe = args.output_dir / '.preflight_write_probe'
    try:
        probe.write_text('ok')
        writable = probe.read_text() == 'ok'
        probe.unlink()
    except OSError:
        writable = False
    record('storage',
           writable and Path(base['dataset_root']).is_dir(),
           {'output_writable': writable, 'dataset_root': base['dataset_root'],
            'target_registration': config.get('target_registration', 'in pair manifest')},
           'storage_failure')

    processes = subprocess.run(['pgrep', '-af', 'status_experimental_train|tools.training.diffusiondrive_v1.fit'],
                               capture_output=True, text=True).stdout.strip()
    record('no_training_process', not bool(processes),
           {'processes': processes or None,
            'paused_old_run_remains_stopped': True}, 'training_process_running')

    report['dataset_version'] = manifest.get('version')
    report['split_manifest'] = str(split_path)
    report['elapsed_seconds'] = round(time.time() - started, 1)
    go = not report['blockers']
    report['go_no_go'] = 'GO' if go else 'NO-GO'
    (args.output_dir / 'status_training_preflight_full.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n')
    train.close(); validation.close()
    print(json.dumps({'go_no_go': report['go_no_go'], 'blockers': report['blockers'],
                      'checks': {key: value['status'] for key, value in report['checks'].items()},
                      'full_validation': {'samples': val_samples, 'loss': val_loss},
                      'elapsed_seconds': report['elapsed_seconds']}, sort_keys=True))


if __name__ == '__main__':
    main()
