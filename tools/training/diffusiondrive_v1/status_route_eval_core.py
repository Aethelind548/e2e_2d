"""Implementation of the current Route-split Status evaluation.

The public CLI remains ``status_route_eval.py``.  This module reuses the
trajectory, detection and BEV metric primitives in ``evaluate_checkpoint.py``
instead of introducing a second evaluator.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CLASS_NAMES = ['background', 'road', 'walkway', 'centerline', 'static', 'vehicle', 'pedestrian']
TIME_S = (np.arange(1, 9) * 0.5).tolist()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def travel(xy):
    xy = np.asarray(xy, dtype=np.float64)
    return float(np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0), axis=1).sum())


def load_weights_only(agent, path):
    """Strictly load a Status checkpoint's 763 agent tensors; no optimizer state."""
    import torch
    path = Path(path).resolve()
    before = sha256(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    state = checkpoint.get('state_dict') or {}
    if not state or not all(key.startswith('agent.') for key in state):
        raise RuntimeError('unexpected Status state_dict layout')
    weights = {key[len('agent.'):]: value for key, value in state.items()}
    if len(weights) != len(state):
        raise RuntimeError('mixed state_dict prefixes')
    agent.load_state_dict(weights, strict=True)
    actual = agent.state_dict()
    if set(actual) != set(weights):
        raise RuntimeError('state_dict keys differ after strict load')
    return {
        'path': str(path), 'sha256': before,
        'global_step': int(checkpoint.get('global_step', -1)),
        'epoch': int(checkpoint.get('epoch', -1)),
        'state_tensors': len(weights), 'optimizer_restored': False,
        'scheduler_restored': False, 'loader': 'strict-weight-only',
    }


def speed_group(speed, target_travel):
    if speed < 0.5:
        return 'stop' if target_travel <= 0.5 else 'near_stop_moving'
    if speed < 2.0:
        return 'creep'
    return 'normal'


def command_group(status):
    return ('Left', 'Straight', 'Right', 'Unknown')[int(np.argmax(np.asarray(status)[:4]))]


def group_names(record):
    speed = 'speed:' + record['speed_group']
    command = 'command:' + record['command']
    return [speed, command, 'cross:' + record['speed_group'] + ':' + record['command']]


def _match(states, logits, gt_states, gt_valid, threshold, distance_m):
    import torch
    from scipy.optimize import linear_sum_assignment
    states = np.asarray(states)
    logits = np.asarray(logits)
    gt_states = np.asarray(gt_states)
    gt_valid = np.asarray(gt_valid).astype(bool)
    result = {
        'invalid_frame': False, 'invalid_frames': 0, 'selected': 0, 'valid_boxes': 0,
        'invalid_dimension_predictions': 0, 'tp': 0, 'fp': 0, 'fn': 0,
        'matched_center_error_m': [], 'matched_x_error_m': [], 'matched_y_error_m': [],
        'matched_length_error_m': [], 'matched_width_error_m': [],
        'matched_heading_raw_abs_rad': [], 'matched_heading_wrapped_abs_rad': [],
        'matched_confidence': [], 'unmatched_confidence': [], 'all_confidence': [],
    }
    if not np.isfinite(states).all() or not np.isfinite(logits).all():
        result['invalid_frame'] = True
        result['invalid_frames'] = 1
        return result
    threshold_logit = math.log(threshold / (1.0 - threshold))
    selected = logits >= threshold_logit
    predictions = states[selected]
    scores = 1.0 / (1.0 + np.exp(-logits[selected]))
    usable = (predictions[:, 3:5] > 0).all(-1)
    boxes = predictions[usable]
    box_scores = scores[usable]
    truth = gt_states[gt_valid]
    result['selected'] = int(selected.sum())
    result['valid_boxes'] = int(len(boxes))
    result['invalid_dimension_predictions'] = int((~usable).sum())
    result['all_confidence'] = [float(x) for x in (1.0 / (1.0 + np.exp(-logits)))]
    tp = 0
    matched_pred = set()
    if len(boxes) and len(truth):
        distance = np.linalg.norm(boxes[:, None, :2] - truth[None, :, :2], axis=-1)
        k = min(distance.shape)
        cost = np.where(distance <= distance_m, distance, (k + 1) * (distance_m + 1))
        a, b = linear_sum_assignment(cost)
        good = distance[a, b] <= distance_m
        for pi, gi in zip(a[good], b[good]):
            matched_pred.add(int(pi))
            pb, gtb = boxes[pi], truth[gi]
            result['matched_center_error_m'].append(float(distance[pi, gi]))
            result['matched_x_error_m'].append(float(abs(pb[0] - gtb[0])))
            result['matched_y_error_m'].append(float(abs(pb[1] - gtb[1])))
            result['matched_length_error_m'].append(float(abs(pb[3] - gtb[3])))
            result['matched_width_error_m'].append(float(abs(pb[4] - gtb[4])))
            raw = float(pb[2] - gtb[2])
            result['matched_heading_raw_abs_rad'].append(abs(raw))
            result['matched_heading_wrapped_abs_rad'].append(
                abs(float(np.arctan2(np.sin(raw), np.cos(raw)))))
            result['matched_confidence'].append(float(box_scores[pi]))
        tp = int(good.sum())
    result['tp'] = tp
    result['fp'] = int(len(predictions) - tp)
    result['fn'] = int(len(truth) - tp)
    result['unmatched_confidence'] = [float(box_scores[i]) for i in range(len(boxes))
                                      if i not in matched_pred]
    return result


def official_agent_components(pred_states, pred_logits, gt_states, gt_valid, config):
    """Reproduce the official Hungarian CE/L1 terms and split L1 by coordinate."""
    import torch
    from scipy.optimize import linear_sum_assignment
    from navsim.agents.diffusiondrive.transfuser_loss import _get_ce_cost, _get_l1_cost
    from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex

    pred_states = torch.as_tensor(pred_states, dtype=torch.float32)
    pred_logits = torch.as_tensor(pred_logits, dtype=torch.float32)
    gt_states = torch.as_tensor(gt_states, dtype=torch.float32)
    gt_valid = torch.as_tensor(gt_valid, dtype=torch.bool)
    if config.latent:
        rad_to_ego = torch.arctan2(gt_states[..., BoundingBox2DIndex.Y],
                                   gt_states[..., BoundingBox2DIndex.X])
        gt_valid = torch.logical_and(
            torch.logical_and(-config.latent_rad_thresh <= rad_to_ego,
                              rad_to_ego <= config.latent_rad_thresh), gt_valid)
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = int(gt_valid.sum())
    denom = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1
    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)
    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    indices = [linear_sum_assignment(c) for c in cost.cpu()]
    src = torch.cat([torch.as_tensor(i, dtype=torch.int64) for i, _ in indices])
    dst = torch.cat([torch.as_tensor(j, dtype=torch.int64) for _, j in indices])
    batch_idx = torch.cat([torch.full_like(torch.as_tensor(i, dtype=torch.int64), b)
                            for b, (i, _) in enumerate(indices)])
    pred = pred_states[batch_idx, src]
    truth = torch.cat([gt_states[b, j] for b, (_, j) in enumerate(indices)], dim=0)
    valid = torch.cat([gt_valid[b, j] for b, (_, j) in enumerate(indices)], dim=0).float()
    delta = (pred - truth).abs()
    weighted = delta * valid[:, None]
    components = {
        'box_x': float(weighted[:, BoundingBox2DIndex.X].sum() / denom),
        'box_y': float(weighted[:, BoundingBox2DIndex.Y].sum() / denom),
        'box_length': float(weighted[:, BoundingBox2DIndex.LENGTH].sum() / denom),
        'box_width': float(weighted[:, BoundingBox2DIndex.WIDTH].sum() / denom),
        'box_heading_raw': float(weighted[:, BoundingBox2DIndex.HEADING].sum() / denom),
        'box_heading_wrapped': float(
            (torch.atan2(torch.sin((pred[:, 2] - truth[:, 2])),
                         torch.cos((pred[:, 2] - truth[:, 2]))).abs() * valid).sum() / denom),
        'ce': float(torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits[batch_idx, src], valid, reduction='none').view(batch_dim, -1).mean())
        if len(src) else 0.0,
    }
    components['box_total'] = sum(components[k] for k in
        ('box_x', 'box_y', 'box_length', 'box_width', 'box_heading_raw'))
    components['valid_gt'] = num_gt_instances
    components['empty_frames'] = int((gt_valid.sum(dim=1) == 0).sum())
    components['matched'] = int(len(src))
    return components


def trajectory_stats(records, name):
    values = [r['errors'][name] for r in records if r['errors'].get(name) is not None]
    if not values:
        return {'valid_samples': 0}
    ade = np.asarray([v['ade_m'] for v in values], dtype=np.float64)
    fde = np.asarray([v['fde_m'] for v in values], dtype=np.float64)
    heading = np.asarray([v['heading_mae_rad'] for v in values], dtype=np.float64)
    return {
        'valid_samples': len(values), 'ade_m': float(ade.mean()), 'fde_m': float(fde.mean()),
        'heading_mae_deg': float(np.degrees(heading.mean())),
        'ade_p50_m': float(np.quantile(ade, .5)), 'ade_p90_m': float(np.quantile(ade, .9)),
        'ade_p95_m': float(np.quantile(ade, .95)),
        'fde_p50_m': float(np.quantile(fde, .5)), 'fde_p90_m': float(np.quantile(fde, .9)),
        'fde_p95_m': float(np.quantile(fde, .95)),
    }


def agent_metrics(records, batches):
    total = defaultdict(float)
    counts = defaultdict(int)
    for batch in batches:
        n = batch['samples']
        counts['samples'] += n
        for key in ('loss', 'trajectory_loss', 'diffusion_loss', 'agent_class_loss',
                    'agent_box_loss', 'bev_semantic_loss'):
            total[key] += batch[key] * n
        counts['valid_gt'] += batch['valid_gt']
        counts['empty_frames'] += batch['empty_frames']
    n = max(1, counts['samples'])
    result = {key: total[key] / n for key in ('loss', 'trajectory_loss', 'diffusion_loss',
                                               'agent_class_loss', 'agent_box_loss',
                                               'bev_semantic_loss')}
    # Raw coordinate components use the same official batch matching.
    for key in ('box_x', 'box_y', 'box_length', 'box_width', 'box_heading_raw',
                'box_heading_wrapped', 'box_total', 'ce'):
        result['raw_' + key] = sum(b[key] * b['samples'] for b in batches) / n
    result.update(samples=counts['samples'], valid_gt=counts['valid_gt'],
                  empty_frames=counts['empty_frames'],
                  empty_frame_rate=counts['empty_frames'] / n)
    # Quality matching uses centre-distance gating, explicitly not official AP.
    for key in ('tp', 'fp', 'fn', 'selected', 'valid_boxes', 'invalid_dimension_predictions',
                'invalid_frames'):
        result[key] = int(sum(r['agent'][key] for r in records))
    tp, fp, fn = result['tp'], result['fp'], result['fn']
    result.update(precision=float(tp / (tp + fp)) if tp + fp else None,
                  recall=float(tp / (tp + fn)) if tp + fn else None,
                  f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None)
    for key in ('matched_center_error_m', 'matched_x_error_m', 'matched_y_error_m',
                'matched_length_error_m', 'matched_width_error_m',
                'matched_heading_raw_abs_rad', 'matched_heading_wrapped_abs_rad',
                'matched_confidence', 'unmatched_confidence', 'all_confidence'):
        values = [v for r in records for v in r['agent'][key]]
        result[key] = {
            'count': len(values),
            'mean': float(np.mean(values)) if values else None,
            'p50': float(np.quantile(values, .5)) if values else None,
            'p90': float(np.quantile(values, .9)) if values else None,
            'p95': float(np.quantile(values, .95)) if values else None,
        }
    return result


def bev_metrics(cm, invalid_pixels):
    cm = np.asarray(cm, dtype=np.int64)
    diag = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - diag
    ious = [None if int(u) == 0 else float(int(d) / int(u)) for d, u in zip(diag, union)]
    present = [x for x in ious if x is not None]
    return {
        'confusion_matrix_gt_rows': cm.tolist(),
        'class_names': CLASS_NAMES,
        'gt_pixels': cm.sum(1).tolist(), 'pred_pixels': cm.sum(0).tolist(),
        'iou': {name: value for name, value in zip(CLASS_NAMES, ious)},
        'miou_present_union': float(np.mean(present)) if present else None,
        'foreground_miou_present_union': float(np.mean([x for x in ious[1:] if x is not None]))
        if any(x is not None for x in ious[1:]) else None,
        'valid_pixels': int(cm.sum()), 'nonfinite_pixels': int(invalid_pixels),
    }


def add_record(record, records, cm):
    records.append(record)
    cm += np.asarray(record['bev_cm'], dtype=np.int64)


def summarize_groups(records):
    groups = defaultdict(list)
    for record in records:
        for group in group_names(record):
            groups[group].append(record)
        groups['town:' + record['town']].append(record)
    rows = []
    for group, subset in sorted(groups.items()):
        model = trajectory_stats(subset, 'model')
        stationary = trajectory_stats(subset, 'stationary')
        cv = trajectory_stats(subset, 'constant_velocity')
        rows.append({'group': group, 'n': len(subset),
                     'n_routes': len({r['route'] for r in subset}),
                     **model,
                     'stationary_ade_m': stationary.get('ade_m'),
                     'stationary_fde_m': stationary.get('fde_m'),
                     'cv_ade_m': cv.get('ade_m'), 'cv_fde_m': cv.get('fde_m'),
                     'stop_pred_travel_gt_1m_rate': (
                         float(np.mean([r['pred_travel_m'] > 1.0 for r in subset]))
                         if group == 'speed:stop' else None)})
    return rows


def route_stats(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record['route']].append(record)
    rows = []
    for route, subset in sorted(grouped.items()):
        model = trajectory_stats(subset, 'model')
        rows.append({'route': route, 'n': len(subset), **model})
    return rows


def route_equal(records, key):
    values = defaultdict(list)
    for record in records:
        value = record.get(key)
        if value is not None:
            values[record['route']].append(float(value))
    means = [float(np.mean(v)) for v in values.values() if v]
    return {'n_routes': len(means), 'mean': float(np.mean(means)) if means else None}


def evaluate_model(agent, dataset, indices, args, label, config, output):
    import torch
    from torch.utils.data import DataLoader, Subset
    from tools.training.diffusiondrive_v1.evaluate_checkpoint import (
        kinematic_baselines, semantic_counts, trajectory_errors, trajectory_flags)

    device = torch.device(args.device)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == 'cuda')
    records = []
    batches = []
    cm = np.zeros((7, 7), dtype=np.int64)
    invalid_pixels = 0
    failures_path = output / f'failures_{label}.jsonl'
    with failures_path.open('w') as failures, (output / f'samples_{label}.jsonl').open('w') as samples:
        offset = 0
        started = time.monotonic()
        with torch.inference_mode():
            for batch_index, (features, targets) in enumerate(loader):
                torch.manual_seed(args.seed + batch_index)
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(args.seed + batch_index)
                f = {key: value.to(device, non_blocking=True) for key, value in features.items()}
                t = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
                prediction = agent.forward(f)
                losses = agent.compute_loss(f, t, prediction)
                pred = {key: value.detach().float().cpu() for key, value in prediction.items()
                        if isinstance(value, torch.Tensor)}
                target = {key: value.detach().float().cpu() for key, value in t.items()}
                n = len(features['status_feature'])
                comp = official_agent_components(
                    pred['agent_states'].numpy(), pred['agent_labels'].numpy(),
                    target['agent_states'].numpy(), target['agent_labels'].numpy(),
                    config)
                batch = {'batch': batch_index, 'samples': n,
                         **{key: (float(losses[key].detach().cpu())
                                  if torch.is_tensor(losses[key]) else float(losses[key]))
                            for key in
                            ('loss', 'trajectory_loss', 'diffusion_loss', 'agent_class_loss',
                             'agent_box_loss', 'bev_semantic_loss')},
                         **comp}
                batches.append(batch)
                status_batch = features['status_feature'].cpu().numpy()
                for j in range(n):
                    try:
                        pair = dataset.rows[indices[offset]]
                        pred_traj = pred['trajectory'][j].numpy()
                        truth = target['trajectory'][j].numpy()
                        status = status_batch[j]
                        command = command_group(status)
                        speed = float(np.linalg.norm(status[4:6]))
                        target_travel = travel(truth[:, :2])
                        pred_travel = travel(pred_traj[:, :2])
                        baselines = kinematic_baselines(status, TIME_S)
                        errors = {name: trajectory_errors(value, truth)
                                  for name, value in dict(model=pred_traj, **baselines).items()}
                        details = _match(pred['agent_states'][j].numpy(), pred['agent_labels'][j].numpy(),
                                         target['agent_states'][j].numpy(),
                                         target['agent_labels'][j].numpy(),
                                         args.score_threshold, args.matching_distance_m)
                        cm_row, invalid = semantic_counts(pred['bev_semantic_map'][j].numpy(),
                                                          target['bev_semantic_map'][j].numpy())
                        record = {
                            'checkpoint': label, 'sample_id': pair['sample_id'], 'route': pair['route'],
                            'frame': int(pair['frame']), 'command': command,
                            'speed_mps': speed, 'target_travel_m': target_travel,
                            'pred_travel_m': pred_travel, 'speed_group': speed_group(speed, target_travel),
                            'town': pair.get('town'), 'errors': errors,
                            'flags': trajectory_flags(pred_traj, truth),
                            'agent': details, 'bev_cm': cm_row.tolist(),
                            'invalid_bev_pixels': invalid,
                            'prediction': pred_traj.tolist(), 'target': truth.tolist(),
                        }
                        samples.write(json.dumps(record, allow_nan=False) + '\n')
                        add_record(record, records, cm)
                        invalid_pixels += invalid
                        offset += 1
                    except Exception as error:  # noqa: BLE001
                        failures.write(json.dumps({
                            'batch': batch_index, 'index': offset,
                            'sample_id': dataset.rows[indices[offset]]['sample_id'],
                            'error': f'{type(error).__name__}: {error}'}) + '\n')
                        raise
                print(json.dumps({'checkpoint': label, 'processed': offset, 'total': len(indices),
                                  'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    metrics = {
        'checkpoint': label, 'samples': len(records),
        'trajectory': {name: trajectory_stats(records, name)
                       for name in ('model', 'stationary', 'constant_velocity')},
        'groups': summarize_groups(records), 'routes': route_stats(records),
        'agent': agent_metrics(records, batches),
        'bev': bev_metrics(cm, invalid_pixels),
        'seconds': time.monotonic() - started,
    }
    return metrics, records, batches


def write_summary_csv(path, rows):
    fields = ['group', 'n', 'n_routes', 'valid_samples', 'ade_m', 'fde_m', 'heading_mae_deg',
              'ade_p50_m', 'ade_p90_m', 'ade_p95_m', 'fde_p50_m', 'fde_p90_m', 'fde_p95_m',
              'stationary_ade_m', 'stationary_fde_m', 'cv_ade_m', 'cv_fde_m',
              'stop_pred_travel_gt_1m_rate']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def pair_models(best, latest, output):
    best_by = {r['sample_id']: r for r in best}
    latest_by = {r['sample_id']: r for r in latest}
    if set(best_by) != set(latest_by):
        raise RuntimeError('Best/Latest sample sets differ')
    rows = []
    for b in best:
        sid = b['sample_id']
        l = latest_by[sid]
        if b['route'] != l['route'] or b['frame'] != l['frame']:
            raise RuntimeError('Sample identity differs: ' + sid)
        row = {
            'sample_id': sid, 'route': b['route'], 'frame': b['frame'],
            'command': b['command'], 'speed_group': b['speed_group'],
            'speed_mps': b['speed_mps'], 'target_travel_m': b['target_travel_m'],
            'best_ade_m': b['errors']['model']['ade_m'], 'latest_ade_m': l['errors']['model']['ade_m'],
            'delta_ade_m': l['errors']['model']['ade_m'] - b['errors']['model']['ade_m'],
            'best_fde_m': b['errors']['model']['fde_m'], 'latest_fde_m': l['errors']['model']['fde_m'],
            'delta_fde_m': l['errors']['model']['fde_m'] - b['errors']['model']['fde_m'],
            'best_heading_deg': np.degrees(b['errors']['model']['heading_mae_rad']),
            'latest_heading_deg': np.degrees(l['errors']['model']['heading_mae_rad']),
        }
        rows.append(row)
    with (output / 'paired_samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    global_result = {
        'samples': len(rows),
        'frame_weighted': {'delta_ade_m': float(np.mean([r['delta_ade_m'] for r in rows])),
                           'delta_fde_m': float(np.mean([r['delta_fde_m'] for r in rows]))},
    }
    by_route = defaultdict(list)
    for row in rows:
        by_route[row['route']].append(row)
    route_rows = []
    for route, subset in sorted(by_route.items()):
        route_rows.append({'route': route, 'n': len(subset),
                           'delta_ade_m': float(np.mean([r['delta_ade_m'] for r in subset])),
                           'delta_fde_m': float(np.mean([r['delta_fde_m'] for r in subset]))})
    with (output / 'paired_by_route.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['route', 'n', 'delta_ade_m', 'delta_fde_m'])
        writer.writeheader(); writer.writerows(route_rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped['speed:' + row['speed_group']].append(row)
        grouped['command:' + row['command']].append(row)
        grouped['cross:' + row['speed_group'] + ':' + row['command']].append(row)
    group_rows = []
    for group, subset in sorted(grouped.items()):
        group_rows.append({'group': group, 'n': len(subset),
                           'delta_ade_m': float(np.mean([r['delta_ade_m'] for r in subset])),
                           'delta_fde_m': float(np.mean([r['delta_fde_m'] for r in subset]))})
    with (output / 'paired_by_group.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group', 'n', 'delta_ade_m', 'delta_fde_m'])
        writer.writeheader(); writer.writerows(group_rows)
    global_result.update({
        'route_equal': {'delta_ade_m': float(np.mean([r['delta_ade_m'] for r in route_rows])),
                        'delta_fde_m': float(np.mean([r['delta_fde_m'] for r in route_rows]))},
        'routes': len(route_rows), 'groups': group_rows,
        'worst_routes': sorted(route_rows, key=lambda r: r['delta_ade_m'], reverse=True)[:10],
    })
    write_json(output / 'paired_global.json', global_result)
    return rows, global_result


def plot_results(output, best, latest, best_metrics, latest_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not best:
        return
    best_by = {r['sample_id']: r for r in best}
    picks = sorted(best_by, key=lambda sid: abs(latest[[r['sample_id'] for r in latest].index(sid)]['errors']['model']['ade_m']
                                                   - best_by[sid]['errors']['model']['ade_m']), reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, sid in zip(axes.flat, picks):
        b = best_by[sid]; l = next(r for r in latest if r['sample_id'] == sid)
        for values, label, color in ((b['target'], 'GT', 'tab:green'),
                                     (b['prediction'], 'Best', 'tab:blue'),
                                     (l['prediction'], 'Latest', 'tab:red')):
            xy = np.asarray(values)[:, :2]
            ax.plot(np.r_[0, xy[:, 1]], np.r_[0, xy[:, 0]], '.-', color=color, label=label, markersize=3)
        ax.scatter([0], [0], marker='s', s=20, color='black')
        ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.25)
        ax.set_title(f"{b['route'].split('_Town')[0][:28]} f{b['frame']}\n"
                     f"Best {b['errors']['model']['ade_m']:.2f} / Latest {l['errors']['model']['ade_m']:.2f} m",
                     fontsize=8)
        ax.set_xlabel('Left y (m)'); ax.set_ylabel('Forward x (m)')
    for ax in axes.flat[len(picks):]:
        ax.axis('off')
    fig.savefig(output / 'paired_trajectory_examples.png', dpi=160); plt.close(fig)

    labels = ['Best', 'Latest']
    agent = [best_metrics['agent'], latest_metrics['agent']]
    fields = [('matched_center_error_m', 'centre m'), ('matched_length_error_m', 'length m'),
              ('matched_width_error_m', 'width m'), ('matched_heading_wrapped_abs_rad', 'yaw rad')]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    for ax, (key, title) in zip(axes, fields):
        ax.bar(labels, [float(m[key]['mean'] or 0.0) for m in agent])
        ax.set_title(title); ax.grid(axis='y', alpha=.25)
    fig.savefig(output / 'agent_matched_errors.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metrics, title in zip(axes, (best_metrics, latest_metrics), labels):
        cm = np.asarray(metrics['bev']['confusion_matrix_gt_rows'])
        ax.imshow(cm, cmap='Blues')
        ax.set_title(f'{title} BEV confusion')
        ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    fig.savefig(output / 'bev_confusion.png', dpi=160); plt.close(fig)


def training_reference(config, output):
    run = Path(config['output_dir'])
    candidates = sorted((run / 'logs').glob('lightning_logs/version_*/metrics.csv'))
    if not candidates:
        return {}
    rows = []
    with candidates[-1].open() as stream:
        for row in csv.DictReader(stream):
            if row.get('val/loss'):
                rows.append({key: (float(value) if value not in ('', None) else None)
                             for key, value in row.items()})
    return {'metrics_csv': str(candidates[-1]), 'rows': rows}


def compare_reference(results, checkpoint_records, reference):
    keys = {'loss': 'val/loss', 'trajectory_loss': 'val/trajectory_loss',
            'agent_class_loss': 'val/agent_class_loss', 'agent_box_loss': 'val/agent_box_loss',
            'bev_semantic_loss': 'val/bev_semantic_loss',
            'diffusion_loss': 'val/diffusion_loss'}
    rows = reference.get('rows', [])
    output = {'metrics_csv': reference.get('metrics_csv'), 'comparisons': {}}
    for label, result in results.items():
        step = int(checkpoint_records[label]['global_step'])
        if not rows:
            output['comparisons'][label] = {'step': step, 'reference': None}
            continue
        nearest = min(rows, key=lambda row: abs(float(row.get('step') or -1) - step))
        values = {}
        for key, source in keys.items():
            reference_value = nearest.get(source)
            evaluated = result['agent'].get(key)
            values[key] = {
                'evaluated': evaluated, 'training_csv': reference_value,
                'delta': None if evaluated is None or reference_value is None
                else float(evaluated - reference_value),
                'relative_delta': None if evaluated is None or not reference_value
                else float((evaluated - reference_value) / reference_value),
            }
        output['comparisons'][label] = {
            'checkpoint_step': step, 'reference_step': nearest.get('step'),
            'reference_epoch': nearest.get('epoch'), 'values': values,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-config', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--sample-ids-file', type=Path)
    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--score-threshold', type=float, default=.5)
    parser.add_argument('--matching-distance-m', type=float, default=2.)
    args = parser.parse_args()
    if args.batch_size != 16:
        raise SystemExit('batch-size must remain 16 to reproduce the original validation loss')
    if not 0 < args.score_threshold < 1 or args.matching_distance_m <= 0:
        raise SystemExit('invalid matching criterion')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))

    import torch
    from tools.training.diffusiondrive_v1.experimental_status_reader import ExperimentalStatusDataset
    from tools.training.diffusiondrive_v1.runtime import model_configuration
    from tools.b2d_adapter.v1_builders import official_v1_module

    experiment = json.loads(args.experiment_config.read_text())
    base_path = Path(experiment['base_config'])
    manifest_path = Path(experiment['dataset_manifest'])
    index_path = Path(experiment['sample_index'])
    split_path = Path(experiment['dataset_split_manifest'])
    for key, path in (('sample_index_sha256', index_path), ('dataset_split_sha256', split_path)):
        if sha256(path) != experiment[key]:
            raise SystemExit(f'{key} mismatch')
    base = json.loads(base_path.read_text())
    dataset = ExperimentalStatusDataset(manifest_path, index_path, base_path, 'validation')
    all_ids = [row['sample_id'] for row in dataset.rows]
    if args.sample_ids_file:
        selected_ids = [line.strip() for line in args.sample_ids_file.read_text().splitlines() if line.strip()]
        position = {sid: i for i, sid in enumerate(all_ids)}
        if len(selected_ids) != len(set(selected_ids)) or any(sid not in position for sid in selected_ids):
            raise SystemExit('sample IDs file is not a unique validation subset')
        indices = [position[sid] for sid in selected_ids]
        selection = 'explicit sample-id file, original dataset order preserved'
    elif args.limit:
        indices = list(range(min(args.limit, len(dataset))))
        selection = f'first {len(indices)} validation samples in original order'
    else:
        indices = list(range(len(dataset)))
        selection = 'complete validation set in original order'
    checkpoints = {}
    for value in args.checkpoint:
        label, path = value.split('=', 1)
        if not label or label in checkpoints:
            raise SystemExit('checkpoint labels must be unique')
        checkpoints[label] = Path(path).resolve()
    if not checkpoints:
        raise SystemExit('at least one checkpoint required')
    agent = official_v1_module('transfuser_agent').TransfuserAgent(
        model_configuration(base), base['learning_rate'], checkpoint_path=None)
    agent.to(args.device).eval().requires_grad_(False)
    checkpoint_records = {}
    results = {}
    all_records = {}
    for label, path in checkpoints.items():
        from tools.training.diffusiondrive_v1.status_route_eval_core import load_weights_only
        checkpoint_records[label] = load_weights_only(agent, path)
        metrics, records, batches = evaluate_model(agent, dataset, indices, args, label,
                                                   agent._config, output)
        results[label] = metrics
        all_records[label] = records
        write_json(output / f'metrics_{label}.json', metrics)
        write_summary_csv(output / f'summary_{label}.csv', metrics['groups'])
    protocol = {
        'schema': 'status-route-eval-v2',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'experiment_config': {'path': str(args.experiment_config.resolve()),
                              'sha256': sha256(args.experiment_config)},
        'base_config': {'path': str(base_path.resolve()), 'sha256': sha256(base_path)},
        'dataset_manifest': {'path': str(manifest_path.resolve()), 'sha256': sha256(manifest_path)},
        'sample_index': {'path': str(index_path.resolve()), 'sha256': sha256(index_path)},
        'split_manifest': {'path': str(split_path.resolve()), 'sha256': sha256(split_path)},
        'validation_population': len(dataset), 'evaluated_samples': len(indices),
        'selection': selection, 'sample_id_sha256': hashlib.sha256(
            ''.join(all_ids[i] for i in indices).encode()).hexdigest(),
        'checkpoints': checkpoint_records, 'device': args.device,
        'precision': 'FP32 evaluation; no BF16 autocast; targets never passed to forward',
        'batch_size': args.batch_size, 'seed': args.seed,
        'noise_policy': 'manual_seed(seed + batch_index) for both checkpoints; fixed order',
        'optimizer_steps': 0, 'targets_supplied_to_forward': False,
        'score_threshold': args.score_threshold, 'matching_distance_m': args.matching_distance_m,
        'group_definitions': {'stop': 'speed < 0.5 m/s and target travel <= 0.5 m',
                              'near_stop_moving': 'speed < 0.5 m/s and target travel > 0.5 m',
                              'creep': '0.5 <= speed < 2.0 m/s', 'normal': 'speed >= 2.0 m/s'},
    }
    write_json(output / 'evaluation_protocol.json', protocol)
    write_json(output / 'metrics.json', results)
    if set(checkpoints) == {'best', 'latest'}:
        paired, global_result = pair_models(all_records['best'], all_records['latest'], output)
        plot_results(output, all_records['best'], all_records['latest'],
                     results['best'], results['latest'])
        reference = training_reference(experiment, output)
        write_json(output / 'training_validation_reference.json', reference)
        write_json(output / 'loss_reproduction.json',
                   compare_reference(results, checkpoint_records, reference))
        print(json.dumps({'paired': global_result, 'protocol': str(output/'evaluation_protocol.json'),
                          'metrics': str(output/'metrics.json')}, indent=2))
    else:
        print(json.dumps({'checkpoints': list(results), 'output': str(output)}, indent=2))
    print('Evaluation complete; no optimizer, backward, raw data or frozen cache was modified.', flush=True)


if __name__ == '__main__':
    main()
