"""Read-only, target-free official V1 checkpoint evaluation on frozen validation caches.

No Trainer, optimizer, backward, raw-data rebuilding or CARLA execution. Detection
uses an explicitly engineering-defined centre-distance criterion, not official AP.
"""
import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / 'outputs/diffusiondrive_v1/checkpoint_evaluation'
CLASS_NAMES = ['background', 'road', 'walkway', 'centerline', 'static', 'vehicle', 'pedestrian']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def content_identity(info):
    # Reading may update atime on relatime mounts; that does not change contents.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def wrapped_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def kinematic_baselines(status, times):
    """Causal rear-frame velocity only; no future target enters either baseline."""
    velocity = np.asarray(status, dtype=np.float64)[4:6]
    cv = np.zeros((len(times), 3), dtype=np.float64)
    cv[:, :2] = np.asarray(times)[:, None] * velocity
    if np.linalg.norm(velocity) > 1e-8:
        cv[:, 2] = math.atan2(velocity[1], velocity[0])
    return {'stationary': np.zeros_like(cv), 'constant_velocity': cv}


def trajectory_errors(prediction, target):
    p, t = np.asarray(prediction), np.asarray(target)
    if p.shape != (8, 3) or t.shape != (8, 3) or not np.isfinite(t).all():
        raise ValueError('Expected complete finite 8x3 trajectory target and matching prediction shape')
    if not np.isfinite(p).all():
        return None
    delta = p[:, :2].astype(np.float64) - t[:, :2]
    distance = np.linalg.norm(delta, axis=-1)
    return {'ade_m': float(distance.mean()), 'fde_m': float(distance[-1]),
            'point_error_m': distance.tolist(),
            'heading_mae_rad': float(np.abs(wrapped_angle(p[:, 2] - t[:, 2])).mean()),
            'longitudinal_mae_m': float(np.abs(delta[:, 0]).mean()),
            'lateral_mae_m': float(np.abs(delta[:, 1]).mean())}


def trajectory_flags(prediction, target):
    if not np.isfinite(prediction).all():
        return {'nonfinite': True, 'degenerate_moving': False, 'abnormal': True}
    p = np.asarray(prediction, dtype=np.float64)
    velocity = np.diff(np.vstack([np.zeros((1, 2)), p[:, :2]]), axis=0) / .5
    acceleration = np.diff(velocity, axis=0) / .5
    yaw_rate = np.abs(wrapped_angle(np.diff(np.r_[0., p[:, 2]]))) / .5
    travel = lambda xy: np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0), axis=1).sum()
    degenerate = bool(travel(p[:, :2]) < .5 and travel(np.asarray(target)[:, :2]) > 5.)
    abnormal = bool(degenerate or np.linalg.norm(velocity, axis=-1).max() > 40.
                    or np.linalg.norm(acceleration, axis=-1).max() > 12. or yaw_rate.max() > 2.)
    return {'nonfinite': False, 'degenerate_moving': degenerate, 'abnormal': abnormal}


def detection_counts(states, logits, gt_states, gt_valid, score_threshold=.5, distance_m=2.):
    """One-to-one max-cardinality gated matching, then minimum centre distance.

    Positive-confidence invalid boxes count as FP. Nonfinite head output invalidates
    the frame, explicitly counted separately. Only the 30-slot vehicle target is GT.
    """
    from scipy.optimize import linear_sum_assignment
    states, logits, gt_states, gt_valid = map(np.asarray, (states, logits, gt_states, gt_valid))
    if not np.isfinite(states).all() or not np.isfinite(logits).all():
        return {'invalid_frames': 1, 'frames': 0, 'tp': 0, 'fp': 0, 'fn': 0}
    selected = logits >= math.log(score_threshold / (1 - score_threshold))
    predictions = states[selected]
    truth = gt_states[gt_valid.astype(bool)]
    usable = (predictions[:, 3:5] > 0).all(-1)
    valid_boxes = predictions[usable]
    tp = 0
    if len(valid_boxes) and len(truth):
        distance = np.linalg.norm(valid_boxes[:, None, :2] - truth[None, :, :2], axis=-1)
        k = min(distance.shape)
        # One invalid edge must cost more than all possible valid edges together.
        cost = np.where(distance <= distance_m, distance, (k + 1) * (distance_m + 1))
        a, b = linear_sum_assignment(cost)
        tp = int((distance[a, b] <= distance_m).sum())
    return {'invalid_frames': 0, 'frames': 1, 'tp': tp, 'fp': len(predictions) - tp, 'fn': len(truth) - tp}


def semantic_counts(logits, truth):
    logits, truth = np.asarray(logits), np.asarray(truth)
    if logits.shape != (7, 128, 256) or truth.shape != (128, 256):
        raise ValueError('Seven-class BEV contract differs')
    if not np.isin(truth, np.arange(7)).all():
        raise ValueError('Unknown target class: no silent ignore/background replacement')
    valid = np.isfinite(logits).all(0)
    pred = np.argmax(np.where(np.isfinite(logits), logits, -np.inf), axis=0)
    cm = np.bincount((truth[valid].astype(np.int64) * 7 + pred[valid]).ravel(), minlength=49).reshape(7, 7)
    return cm, int((~valid).sum())


def fraction(a, b):
    return float(a / b) if b else None


class Metrics:
    """Accumulate sums/counts, never an unweighted average of batch means."""
    def __init__(self):
        self.n = 0
        self.errors = {name: [] for name in ('model', 'stationary', 'constant_velocity')}
        self.flags = defaultdict(int)
        self.det = defaultdict(int)
        self.cm = np.zeros((7, 7), dtype=np.int64)
        self.invalid_pixels = 0

    def add(self, record, cm, det):
        self.n += 1
        for name, error in record['errors'].items():
            if error is not None:
                self.errors[name].append(error)
        for key, value in record['flags'].items():
            self.flags[key] += int(value)
        for key, value in det.items():
            self.det[key] += value
        self.cm += cm
        self.invalid_pixels += record['invalid_bev_pixels']

    def result(self):
        trajectories = {}
        for name, rows in self.errors.items():
            if not rows:
                trajectories[name] = {'valid_samples': 0}
                continue
            values = {key: np.asarray([r[key] for r in rows], dtype=np.float64) for key in rows[0]}
            result = {key: value.mean(axis=0).tolist() for key, value in values.items()}
            result.update(valid_samples=len(rows), heading_mae_deg=result['heading_mae_rad'] * 180 / np.pi,
                          ade_quantiles_m=np.quantile(values['ade_m'], [.5, .9, .95, .99]).tolist(),
                          fde_quantiles_m=np.quantile(values['fde_m'], [.5, .9, .95, .99]).tolist())
            trajectories[name] = result
        tp, fp, fn = (self.det[k] for k in ('tp', 'fp', 'fn'))
        union = self.cm.sum(0) + self.cm.sum(1) - self.cm.diagonal()
        ious = [fraction(int(a), int(b)) for a, b in zip(self.cm.diagonal(), union)]
        defined = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
        return {'samples': self.n, 'trajectory': trajectories,
                'flags': {key: {'count': self.flags[key], 'rate': fraction(self.flags[key], self.n)}
                          for key in ('nonfinite', 'degenerate_moving', 'abnormal')},
                'detection': dict(self.det, precision=fraction(tp, tp + fp), recall=fraction(tp, tp + fn),
                                  f1=fraction(2 * tp, 2 * tp + fp + fn)),
                'bev': {'confusion_matrix_gt_rows': self.cm.tolist(), 'iou': dict(zip(CLASS_NAMES, ious)),
                        'miou_present_union': defined(ious), 'foreground_miou_present_union': defined(ious[1:]),
                        'valid_pixels': int(self.cm.sum()), 'nonfinite_pixels': self.invalid_pixels}}


def case_groups(route, speed):
    town = re.search(r'(Town\d+(?:HD)?)', route)
    scenario = Path(route).name.split('_Town')[0]
    speed_bin = '0-.5' if speed < .5 else '.5-5' if speed < 5 else '5-10' if speed < 10 else '10-20' if speed < 20 else '20+'
    return ['town:' + (town.group(1) if town else 'unknown'), 'scenario:' + scenario, 'speed_mps:' + speed_bin]


def stable_checkpoint(path):
    import torch
    path = Path(path).resolve()
    before = path.stat()
    checksum = sha256(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if content_identity(before) != content_identity(path.stat()):
        raise RuntimeError('Checkpoint changed during read: ' + str(path))
    meta = checkpoint['b2d_v1_fit']
    if digest(meta['semantic']) != meta['contract_sha256']:
        raise ValueError('Checkpoint semantic digest mismatch')
    return checkpoint, {'path': str(path), 'sha256': checksum, 'global_step': checkpoint['global_step'],
                        'epoch_zero_based': checkpoint['epoch'], 'contract_sha256': meta['contract_sha256']}


def inventory(root):
    import torch
    rows, validations, curves = [], [], []
    for folder in sorted(Path(root).glob('baseline_bs8*')):
        report_path = folder / 'fit_result.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            validations.extend(dict(v, run=folder.name) for v in report.get('validation', []))
        for path in sorted(folder.rglob('*.ckpt')):
            try:
                ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                rows.append({'path': str(path), 'global_step': ck['global_step'], 'epoch_zero_based': ck['epoch'],
                             'contract_sha256': ck['b2d_v1_fit']['contract_sha256'], 'bytes': path.stat().st_size})
                del ck
            except (FileNotFoundError, RuntimeError, EOFError) as exc:
                rows.append({'path': str(path), 'read_error': str(exc)})
        metrics = folder / 'metrics/version_0/metrics.csv'
        if metrics.exists():
            with metrics.open() as stream:
                for row in csv.DictReader(stream):
                    if row.get('train/loss_epoch') or row.get('val/loss_epoch'):
                        curves.append(dict(run=folder.name, **{k: float(v) for k, v in row.items() if v}))
    return {'checkpoints': rows, 'validation_records': validations, 'epoch_curves': curves,
            'epoch_caveat': 'Pause-end train epoch means may cover only the active segment; CSV step is zero-based.'}


def plot_results(output, curves, samples, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for metric, ax, title in [('train/loss_epoch', axes[0, 0], 'Training objective (epoch/active segment)'),
                              ('val/loss_epoch', axes[0, 1], 'Validation inference L1 + auxiliary loss'),
                              ('val/ade_m', axes[0, 2], 'Full validation ADE (m)')]:
        rows = sorted((r for r in curves if metric in r), key=lambda r: r['step'])
        ax.plot([r['step'] for r in rows], [r[metric] for r in rows], '.-')
        ax.set_title(title); ax.set_xlabel('CSV step'); ax.grid(alpha=.2)
    for label in labels:
        rows = [r for r in samples if r['checkpoint'] == label and r['errors']['model'] is not None]
        for ax, key in [(axes[1, 0], 'ade_m'), (axes[1, 1], 'fde_m')]:
            ax.hist([r['errors']['model'][key] for r in rows], bins=60, histtype='step', label=label)
            ax.set_xlabel(key); ax.set_ylabel('samples'); ax.legend()
        axes[1, 2].plot(np.arange(1, 9) * .5,
                        np.mean([r['errors']['model']['point_error_m'] for r in rows], axis=0), label=label)
    axes[1, 2].set(xlabel='Future time (s)', ylabel='Position MAE (m)'); axes[1, 2].legend()
    fig.tight_layout(); fig.savefig(output / 'curves.png', dpi=160); plt.close(fig)
    # Same low/median/high-error sample IDs from Best, across all checkpoints.
    reference = [r for r in samples if r['checkpoint'] == ('best' if 'best' in labels else labels[0]) and r['errors']['model']]
    reference.sort(key=lambda r: r['errors']['model']['ade_m'])
    chosen = [reference[int((len(reference) - 1) * q)] for q in (.1, .5, .99)]
    lookups = {(r['checkpoint'], r['sample_id']): r for r in samples}
    fig, axes = plt.subplots(len(labels), 3, figsize=(16, 4.5 * len(labels)), squeeze=False)
    for i, label in enumerate(labels):
        for j, ref in enumerate(chosen):
            row = lookups[label, ref['sample_id']]; ax = axes[i, j]
            for name, points in [('GT', row['target']), ('model', row['prediction']),
                                 ('CV', row['cv']), ('stationary', np.zeros((8, 3)))]:
                points = np.asarray(points, dtype=float)
                ax.plot(points[:, 1], points[:, 0], '.-', label=name)
            err = row['errors']['model']
            ax.set_title(f"{label}: Best-error quantile {(.1, .5, .99)[j]:.0%}\n{Path(row['route']).name.split('_Town')[0]} f{row['frame']}\nADE {err['ade_m']:.2f} m" if err else label + ': nonfinite')
            ax.set(xlabel='Left (m)', ylabel='Forward (m)'); ax.invert_xaxis(); ax.axis('equal'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(output / 'trajectory_cases.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--limit', type=int, help='Explicit deterministic validation subset; absent means full validation')
    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--sleep-seconds', type=float, default=.02, help='Yield between batches when sharing a training GPU')
    parser.add_argument('--score-threshold', type=float, default=.5)
    parser.add_argument('--matching-distance-m', type=float, default=2.)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1) or args.sleep_seconds < 0:
        parser.error('Positive batch/limit and nonnegative sleep required')
    if not 0 < args.score_threshold < 1 or args.matching_distance_m <= 0:
        parser.error('Invalid detection criterion')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ['metrics.json', 'evaluation_protocol.json', 'samples.jsonl']):
        raise FileExistsError('Use a fresh explicitly approved output directory; evaluation never overwrites results')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
    import torch
    from torch.utils.data import DataLoader, Subset
    from tools.b2d_adapter.v1_builders import official_v1_module
    from tools.training.diffusiondrive_v1.runtime import model_configuration, initialization_manifest
    from tools.training.diffusiondrive_v1.official_v1_audit import verify_pins
    from tools.training.diffusiondrive_v1.paired_dataset import paired_datasets
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    paths = {}
    for value in args.checkpoint:
        label, path = value.split('=', 1)
        if not re.fullmatch('[a-zA-Z0-9_-]+', label) or label in paths:
            raise ValueError('Checkpoint labels must be unique simple names')
        paths[label] = Path(path).resolve()
    # Mapped private readers remain valid even if the trainer later prunes a periodic
    # source. No source is renamed, modified, cloned to disk or locked against training.
    checkpoints = {}; checkpoint_records = {}
    for label, path in paths.items():
        checkpoints[label], checkpoint_records[label] = stable_checkpoint(path)
    first = next(iter(checkpoints.values()))['b2d_v1_fit']
    compatible = lambda semantic: {k: v for k, v in semantic.items()
                                  if k not in ('precision', 'precision_policy', 'precision_policy_source_sha256')}
    for ck in checkpoints.values():
        if compatible(ck['b2d_v1_fit']['semantic']) != compatible(first['semantic']):
            raise ValueError('Cross-checkpoint data/model/seed/initialization contracts differ')
        verify_pins(ck['b2d_v1_fit']['config'])
    config = dict(first['config'])
    # Execution-only reader selection, not a data/provenance/geometry override.
    config.update(cache_read_mode='cache-only', mount_check_mode='procfs', cache_file_read_mode='verified-single-pass')
    manifest_path = ROOT / config['paired_manifest']
    if sha256(manifest_path) != first['semantic']['prepared_data']['sha256']:
        raise ValueError('Frozen paired manifest SHA differs')
    manifest = json.loads(manifest_path.read_text())
    if set(manifest['target_sources']) != {'baseline'}:
        raise ValueError('This acceptance evaluator requires Baseline-only supervision')
    # Do not build or consume training batches. Factory verifies the existing full
    # index/config selection and producer whitelist, then discard its train reader.
    model_config = model_configuration(config)
    initialization_manifest(config, model_config)
    datasets, identity = paired_datasets(manifest_path, config, model_config)
    validation = datasets.pop('validation'); del datasets
    selected = np.arange(len(validation))
    if args.limit and args.limit < len(validation):
        selected = np.sort(np.random.default_rng(args.seed).choice(selected, args.limit, replace=False))
    rows = [validation.rows[int(i)] for i in selected]
    train_groups = {re.search(r'(Town\d+(?:HD)?_Route\d+)', e['route']).group(1) for e in config['data']['train']}
    validation_groups = {re.search(r'(Town\d+(?:HD)?_Route\d+)', r['route']).group(1) for r in validation.rows}
    if train_groups & validation_groups or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('RouteGroup leakage or duplicate evaluation sample')
    # Paths can have different depths (pause vs checkpoints); use their common experiment ancestor.
    common = Path(os.path.commonpath([str(p.parent) for p in paths.values()]))
    inventory_result = inventory(common)
    write_json(output / 'checkpoint_inventory.json', inventory_result)
    protocol = {'schema': 'b2d-v1-readonly-openloop-v1', 'checkpoints': checkpoint_records,
                'created_at_unix': time.time(), 'seed': args.seed, 'batch_size': args.batch_size,
                'noise_policy': 'manual_seed(seed + batch_index); equal batch size/order across checkpoints; changing batching changes noise',
                'device': args.device, 'precision': 'FP32 evaluation for every checkpoint, including BF16-trained checkpoints',
                'targets_supplied_to_forward': False, 'optimizer_steps': 0, 'split': 'validation',
                'validation_population': len(validation), 'evaluated_samples': len(rows),
                'selection': 'complete validation' if len(rows) == len(validation) else 'seeded uniform subset without replacement',
                'sample_id_sha256': digest([r['sample_id'] for r in rows]), 'paired_identity': identity,
                'route_group_overlap': [], 'towns': sorted({case_groups(r['route'], 0)[0] for r in rows}),
                'times_s': (np.arange(1, 9) * .5).tolist(),
                'coordinates': 'engineering rear axle: x forward, y left, heading left positive, metres/radians',
                'detection': {'target_scope': 'up to 30 nearest vehicles in official target ROI; not all Actor classes or official AP',
                              'score_threshold': args.score_threshold, 'max_center_distance_m': args.matching_distance_m,
                              'matching': 'max cardinality one-to-one gated centre distance; invalid dimensions positive predictions are FP'},
                'aggregation': 'trajectory sums per finite sample/time; detection pooled TP/FP/FN; BEV pooled pixel confusion matrix',
                'warning_thresholds_engineering_not_paper': {'speed_mps': 40, 'acceleration_mps2': 12, 'yaw_rate_radps': 2,
                    'degenerate_pred_travel_m_below': .5, 'degenerate_gt_travel_m_above': 5,
                    'review_if_worse_than_cv': True, 'any_nonfinite_requires_review': True},
                'baseline_heading': 'stationary heading=0; CV heading=atan2(vy,vx) except zero speed; neither estimates yaw dynamics',
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__).resolve(),
                    ROOT/'repositories/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py',
                    ROOT/'repositories/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_loss.py']},
                'torch': torch.__version__, 'sleep_seconds': args.sleep_seconds}
    write_json(output / 'evaluation_protocol.json', protocol)
    agent = official_v1_module('transfuser_agent').TransfuserAgent(model_config, config['learning_rate'], checkpoint_path=None)
    agent.to(args.device).eval().requires_grad_(False)
    loader = DataLoader(Subset(validation, selected.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=args.device == 'cuda')
    results = {}; samples = []
    with (output / 'samples.jsonl').open('x') as sample_stream:
        for label, ck in checkpoints.items():
            state = ck['state_dict']
            if any(not k.startswith('agent.') for k in state):
                raise ValueError('Unexpected Lightning state_dict prefix')
            agent.load_state_dict({k[len('agent.'):]: v for k, v in state.items()}, strict=True)
            agent.eval()
            versions = {k: v._version for k, v in agent.state_dict().items()}
            total = Metrics(); groups = defaultdict(Metrics); offset = 0; start = time.monotonic()
            with torch.inference_mode():
                for batch_index, (features, targets) in enumerate(loader):
                    torch.manual_seed(args.seed + batch_index)
                    f = {k: v.to(args.device, non_blocking=True) for k, v in features.items()}
                    if agent.training or any(m.training for m in agent.modules()):
                        raise RuntimeError('Evaluation model unexpectedly entered training mode')
                    # The targets never enter forward. No compute_loss/train/backward/optimizer call.
                    prediction = agent.forward(f)
                    prediction = {k: v.detach().cpu().numpy() for k, v in prediction.items() if isinstance(v, torch.Tensor)}
                    target = {k: v.numpy() for k, v in targets.items()}
                    for j in range(len(features['status_feature'])):
                        pair = rows[offset]; offset += 1
                        pred, truth = prediction['trajectory'][j], target['trajectory'][j]
                        baselines = kinematic_baselines(features['status_feature'][j].numpy(), protocol['times_s'])
                        errors = {name: trajectory_errors(p, truth) for name, p in dict(model=pred, **baselines).items()}
                        flags = trajectory_flags(pred, truth)
                        cm, invalid_pixels = semantic_counts(prediction['bev_semantic_map'][j], target['bev_semantic_map'][j])
                        det = detection_counts(prediction['agent_states'][j], prediction['agent_labels'][j],
                            target['agent_states'][j], target['agent_labels'][j], args.score_threshold, args.matching_distance_m)
                        record = {'checkpoint': label, 'sample_id': pair['sample_id'], 'route': pair['route'], 'frame': pair['frame'],
                                  'feature_sha256': pair['feature_sha'], 'target_sha256': pair['target_sha'],
                                  'speed_mps': float(np.linalg.norm(features['status_feature'][j, 4:6].numpy())),
                                  'errors': errors, 'flags': flags, 'detection': det, 'invalid_bev_pixels': invalid_pixels,
                                  'prediction': np.where(np.isfinite(pred), pred, None).tolist(), 'target': truth.tolist(),
                                  'cv': baselines['constant_velocity'].tolist()}
                        total.add(record, cm, det)
                        for group in case_groups(pair['route'], record['speed_mps']):
                            groups[group].add(record, cm, det)
                        sample_stream.write(json.dumps(record, allow_nan=False) + '\n'); samples.append(record)
                    if batch_index % 250 == 0:
                        print(json.dumps({'checkpoint': label, 'processed': offset, 'total': len(rows),
                                          'elapsed_seconds': time.monotonic()-start}), flush=True)
                    del f, prediction
                    if args.sleep_seconds:
                        time.sleep(args.sleep_seconds)
            if versions != {k: v._version for k, v in agent.state_dict().items()}:
                raise RuntimeError('Model parameters/buffers changed during evaluation')
            result = total.result()
            result.update(checkpoint=checkpoint_records[label], seconds=time.monotonic()-start,
                          groups={k: v.result() for k, v in sorted(groups.items())}, parameters_unchanged=True)
            results[label] = result
            sample_stream.flush()
            print(json.dumps({'completed': label, 'samples': result['samples'],
                              'trajectory': result['trajectory'], 'detection': result['detection']}), flush=True)
            gc.collect()
    write_json(output / 'metrics.json', results)
    with (output / 'comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['checkpoint', 'global_step', 'samples', 'ade_m', 'fde_m', 'heading_mae_deg',
            'lateral_mae_m', 'longitudinal_mae_m', 'cv_ade_m', 'stationary_ade_m', 'detection_precision', 'detection_recall',
            'detection_f1', 'bev_miou', 'nonfinite_rate', 'abnormal_rate'])
        writer.writeheader()
        for label, result in results.items():
            traj = result['trajectory']
            writer.writerow(dict(checkpoint=label, global_step=result['checkpoint']['global_step'], samples=result['samples'],
                **{k: traj['model'].get(k) for k in ['ade_m', 'fde_m', 'heading_mae_deg', 'lateral_mae_m', 'longitudinal_mae_m']},
                cv_ade_m=traj['constant_velocity'].get('ade_m'), stationary_ade_m=traj['stationary'].get('ade_m'),
                **{'detection_'+k: result['detection'][k] for k in ['precision', 'recall', 'f1']},
                bev_miou=result['bev']['miou_present_union'], nonfinite_rate=result['flags']['nonfinite']['rate'],
                abnormal_rate=result['flags']['abnormal']['rate']))
    plot_results(output, inventory_result['epoch_curves'], samples, list(paths))
    print('Evaluation complete. Original training, caches and checkpoints were not modified.', flush=True)


if __name__ == '__main__':
    main()
