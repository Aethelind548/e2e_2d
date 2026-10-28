"""Read-only candidate-trajectory quality and anchor-coverage diagnostic.

The tool never trains, never updates parameters, never rebuilds caches and never
writes into a training output directory. It reads the frozen behaviour panel and
the authenticated pair manifest, runs CPU/FP32 inference with targets withheld
from `forward`, and captures the official 20 candidates plus the official mode
scores through a read-only forward hook.

Evaluation-only quantities (ADE/FDE/oracle/selection gap/behaviour labels) are
computed after inference and are written to the diagnostic output only. They must
never enter the training loss, the anchor-based positive assignment, the official
argmax, sample weights, the sampler, or any checkpoint-selection branch.

Subcommands
    anchors   anchor geometry + official anchor-mode coverage (no model)
    evaluate  candidate capture for the frozen panel (model, no gradients)
    report    rebuild CSVs/plots/summary from the saved sample JSONL
"""
import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.training.diffusiondrive_v1.evaluate_checkpoint import digest, sha256 as file_digest
CONFIG = ROOT / 'tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_scratch.json'
ANCHOR = ROOT / 'outputs/diffusiondrive_v1/assets/official_kmeans_navsim_traj_20.npy'
PANEL = ROOT / ('outputs/diffusiondrive_v1/scratch_attribution/'
                '20261022_142557_panel/behavior_panel.json')
DEFAULT_CHECKPOINTS = {
    'best': Path('/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/'
                 'batch64_lr1e4_scratch_20261021_210751/'
                 'checkpoints/best-epoch002-step009882.ckpt'),
    'late': Path('/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/'
                 'batch64_lr1e4_scratch_20261021_210751/'
                 'health/diagnostic-step000058500.ckpt'),
}
EXPECTED_ANCHOR_SHA256 = '2d341e525d7882ee5e916c67d37bea571e4c793cf801f0fff3eff59d38b20d98'
TIMES = np.arange(1, 9) * .5
# Engineering report thresholds. They only group/annotate diagnostic output.
THRESHOLDS = {
    'near_zero_end_m': 1.0,        # candidate/anchor counted as near-zero displacement
    'turn_yaw_deg': 20.0,          # candidate/anchor counted as a turn
    'turn_adequate_ratio': .5,     # candidate must reach this share of |GT yaw|
    'insufficient_ratio': .5,      # candidate below this share of |GT path| is "short"
    'smooth_second_diff_m': .25,   # mean |2nd difference| above this is annotated rough
    'detour_ratio': 1.5,           # path / endpoint above this is annotated as detour
    'max_step_m': 15.0,            # single 0.5 s step above this is annotated extreme
}

CLASSES = ('stationary', 'approaching_stop', 'low_creep', 'low_depart',
           'left_turn', 'right_turn', 'moving_straight', 'interaction_decision')
STOP_CLASSES = ('stationary', 'low_creep', 'low_depart', 'approaching_stop')
TURN_CLASSES = ('left_turn', 'right_turn')
METRIC_PREFIX = 'diagnostic_only_'


# ---------------------------------------------------------------- utilities

def readonly(path):
    db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def wrap(value):
    return float(np.arctan2(np.sin(value), np.cos(value)))


def mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def fraction(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


# ------------------------------------------------------- trajectory geometry

def trajectory_geometry(xy, yaw=None):
    """Pure geometry of one (T,2) xy sequence. No ground truth involved."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('expected (T,2) xy, got %r' % (xy.shape,))
    finite = bool(np.isfinite(xy).all())
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0), axis=1)
    path = float(steps.sum())
    endpoint = float(np.linalg.norm(xy[-1]))
    offset = np.linalg.norm(xy, axis=1)
    second = (xy[2:] - 2 * xy[1:-1] + xy[:-2]) if len(xy) >= 3 else np.zeros((0, 2))
    smooth = float(np.linalg.norm(second, axis=1).mean()) if len(second) else 0.0
    total_yaw = None
    if yaw is not None:
        total_yaw = wrap(float(np.asarray(yaw, dtype=np.float64)[-1]))
    else:
        total_yaw = wrap(float(math.atan2(xy[-1, 1], xy[-1, 0])))
    return {
        'finite': finite,
        'endpoint_m': endpoint,
        'path_m': path,
        'max_offset_m': float(offset.max()),
        'longitudinal_end_m': float(xy[-1, 0]),
        'lateral_end_m': float(xy[-1, 1]),
        'lateral_absmax_m': float(np.abs(xy[:, 1]).max()),
        'max_step_m': float(steps.max()),
        'second_diff_m': smooth,
        'total_yaw_deg': total_yaw * 180 / math.pi,
        'detour_ratio': float(path / endpoint) if endpoint > 1e-6 else None,
        'backward_steps': int(sum(1 for d in np.diff(xy[:, 0]) if d < -0.05)),
    }


def geometry_flags(geometry):
    """Report-only annotations; never used to gate or alter inference."""
    if not geometry['finite']:
        return 'nonfinite'
    flags = []
    if geometry['endpoint_m'] <= THRESHOLDS['near_zero_end_m']:
        flags.append('near_zero')
    if geometry['total_yaw_deg'] >= THRESHOLDS['turn_yaw_deg']:
        flags.append('left_turn')
    elif geometry['total_yaw_deg'] <= -THRESHOLDS['turn_yaw_deg']:
        flags.append('right_turn')
    if geometry['second_diff_m'] > THRESHOLDS['smooth_second_diff_m']:
        flags.append('rough')
    if geometry['detour_ratio'] is not None and geometry['detour_ratio'] > THRESHOLDS['detour_ratio']:
        flags.append('detour')
    if geometry['max_step_m'] > THRESHOLDS['max_step_m']:
        flags.append('extreme_step')
    if geometry['backward_steps'] > 0:
        flags.append('reversal')
    return '|'.join(flags) if flags else 'normal'


def anchor_class(geometry):
    """Report-only anchor taxonomy. Not a training label and not an inference rule."""
    if geometry['endpoint_m'] <= THRESHOLDS['near_zero_end_m']:
        return 'stationary_like'
    if geometry['endpoint_m'] <= 8.0:
        base = 'low_speed_like'
    else:
        base = 'cruise_like'
    yaw = geometry['total_yaw_deg']
    if yaw >= THRESHOLDS['turn_yaw_deg']:
        return 'left_turn_like'
    if yaw <= -THRESHOLDS['turn_yaw_deg']:
        return 'right_turn_like'
    return base


def ade_fde(candidate, target):
    """Evaluation-only XY error. Never enters any loss or selection branch."""
    p = np.asarray(candidate, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape or p.shape[1] != 3:
        raise ValueError('expected matching (T,3) trajectories')
    if not np.isfinite(p).all():
        return None
    distance = np.linalg.norm(p[:, :2] - t[:, :2], axis=-1)
    return {'ade_m': float(distance.mean()), 'fde_m': float(distance[-1]),
            'point_error_m': distance.tolist()}


def anchor_mode(target, anchor):
    """Exact reuse of multimodal_loss.LossComputer: XY, 8-point mean L2, argmin."""
    gt = np.asarray(target, dtype=np.float64)[None, :, :2]
    dist = np.linalg.norm(gt - np.asarray(anchor, dtype=np.float64)[None], axis=-1)
    return int(np.argmin(dist.mean(-1)))


def load_anchor(path=ANCHOR, verify=True):
    path = Path(path)
    if verify and hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_ANCHOR_SHA256:
        raise ValueError('official anchor SHA changed: ' + str(path))
    array = np.load(path, allow_pickle=False)
    if array.shape != (20, 8, 2) or not np.isfinite(array).all():
        raise ValueError('unexpected anchor array %r' % (array.shape,))
    return array


# ------------------------------------------------------- frozen data helpers

def load_manifest(config_path):
    config = json.loads(Path(config_path).read_text())
    manifest_path = ROOT / config['paired_manifest']
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'b2d-v1-explicit-recovery-paired-v1':
        raise ValueError('authenticated recovery pair manifest required')
    return config, manifest, manifest_path


def pair_rows(manifest, split):
    with readonly(manifest['pair_index']) as db:
        return [dict(r) for r in db.execute(
            'SELECT * FROM pairs WHERE split=? ORDER BY source_rank DESC,route,frame',
            (split,))]


def load_target(manifest, row):
    from tools.training.diffusiondrive_v1.target_storage import load_target_row
    source = manifest['target_sources'][row['target_source']]
    with readonly(Path(source['state']) / 'index.sqlite') as db:
        target = db.execute('SELECT * FROM samples WHERE sample_id=?',
                            (row['sample_id'],)).fetchone()
    if target is None or target['state'] != 'complete' or target['sha256'] != row['target_sha']:
        raise ValueError('target index mismatch: ' + row['sample_id'])
    return load_target_row(dict(target), source['target_root'],
                           source['producer_sha256'])['targets']


def target_stats(target):
    xy = np.asarray(target['trajectory'][:, :2], dtype=np.float64)
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0), axis=1)
    heading = np.asarray(target['trajectory'][:, 2], dtype=np.float64)
    return {'path_m': float(steps.sum()), 'endpoint_m': float(np.linalg.norm(xy[-1])),
            'first_speed_mps': float(steps[0] / .5),
            'last_speed_mps': float(steps[-1] / .5),
            'heading_change_rad': wrap(float(heading[-1])),
            'max_step_m': float(steps.max())}


def load_status(manifest, row):
    from tools.training.diffusiondrive_v1.features_cache import load_feature_row
    from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
    source = manifest['feature_sources'][row['feature_source']]
    with readonly(Path(source['state']) / 'index.sqlite') as db:
        feature = db.execute('SELECT * FROM samples WHERE sample_id=?',
                             (row['sample_id'],)).fetchone()
    if feature is None or feature['state'] != 'complete' or feature['sha256'] != row['feature_sha']:
        raise ValueError('feature index mismatch: ' + row['sample_id'])
    values = load_feature_row(dict(feature), effective_roots(manifest, source),
                              source['producer_sha256'])
    return np.asarray(values['status_feature'], dtype=np.float64).reshape(-1)


# ------------------------------------------------------------ anchors phase

def anchors_command(config_path, output, coverage_split='validation'):
    anchor = load_anchor()
    config, manifest, manifest_path = load_manifest(config_path)

    rows = []
    for mode in range(len(anchor)):
        geometry = trajectory_geometry(anchor[mode, :, :2])
        rows.append({'mode': mode, 'cluster_class': anchor_class(geometry),
                     'flags': geometry_flags(geometry), **geometry,
                     'xy': anchor[mode].tolist()})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'anchor_geometry.csv').open('x', newline='') as stream:
        fields = [k for k in rows[0] if k != 'xy']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

    # Official anchor-mode distribution. Only targets are read; no features,
    # no model, no gradients.
    coverage = defaultdict(Counter)
    per_sample = []
    if coverage_split in ('validation', 'train', 'both'):
        splits = ['validation', 'train'] if coverage_split == 'both' else [coverage_split]
        for split in splits:
            for index, row in enumerate(pair_rows(manifest, split)):
                target = load_target(manifest, row)
                stats = target_stats(target)
                mode = anchor_mode(target['trajectory'], anchor)
                key = behaviour_key(stats)
                coverage[(split, key)][mode] += 1
                coverage[(split, 'ALL')][mode] += 1
                per_sample.append({'split': split, 'sample_id': row['sample_id'],
                                   'route': row['route'], 'frame': row['frame'],
                                   'anchor_mode': mode, 'behaviour': key, **stats})
                if (index + 1) % 1000 == 0:
                    print(json.dumps({'event': 'coverage', 'split': split,
                                      'processed': index + 1}), flush=True)
    with (output / 'anchor_coverage.csv').open('x', newline='') as stream:
        fields = ['split', 'behaviour', 'anchor_mode', 'cluster_class', 'samples', 'share']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (split, key), counter in sorted(coverage.items()):
            total = sum(counter.values())
            for mode, count in sorted(counter.items()):
                writer.writerow({'split': split, 'behaviour': key, 'anchor_mode': mode,
                                 'cluster_class': rows[mode]['cluster_class'],
                                 'samples': count, 'share': count / total})
    (output / 'anchor_coverage_samples.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in per_sample))
    protocol = {
        'schema': 'candidate-quality-anchor-v1',
        'anchor': {'path': str(ANCHOR), 'sha256': file_digest(ANCHOR),
                   'shape': list(anchor.shape), 'dtype': str(anchor.dtype)},
        'config': str(Path(config_path).resolve()), 'config_sha256': file_digest(config_path),
        'manifest': str(manifest_path.resolve()), 'manifest_sha256': file_digest(manifest_path),
        'coverage_split': coverage_split, 'thresholds': THRESHOLDS,
        'note': 'anchor classes are report-only annotations, never training labels',
    }
    (output / 'anchor_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'event': 'anchors_written', 'output': str(output.resolve()),
                      'samples': len(per_sample)}, ensure_ascii=False), flush=True)


def behaviour_key(stats):
    """Behaviour bucketing reused from the frozen panel definition."""
    path = stats['path_m']
    dh = abs(stats['heading_change_rad'])
    if path <= .5 and dh < math.radians(10):
        return 'stationary'
    if stats['first_speed_mps'] > .5 and stats['last_speed_mps'] < .3 and dh < math.radians(20):
        return 'approaching_stop'
    if path > 2 and dh >= math.radians(20):
        return 'left_turn' if stats['heading_change_rad'] > 0 else 'right_turn'
    if .5 < path <= 5 and dh < math.radians(10):
        return 'low_creep'
    if path > 5 and stats['first_speed_mps'] < 1.0 and dh < math.radians(10):
        return 'low_depart'
    return 'moving_straight'


# ----------------------------------------------------------- evaluate phase

def evaluate_command(panel_path, output, limit=0, device='cpu', include_case=False):
    panel = json.loads(Path(panel_path).read_text())
    config = json.loads(Path(panel['config']).read_text())
    manifest = json.loads(Path(panel['manifest']).read_text())
    from tools.training.diffusiondrive_v1.cache_only_reader import CacheOnlyPairedDataset
    from tools.training.diffusiondrive_v1.runtime import (model_configuration,
                                                          initialization_manifest)
    from tools.training.diffusiondrive_v1.official_v1_audit import verify_pins
    from tools.training.diffusiondrive_v1.evaluate_checkpoint import stable_checkpoint
    from tools.b2d_adapter.v1_builders import official_v1_module
    import torch
    torch.set_num_threads(1)
    verify_pins(config)
    model_config = model_configuration(config)
    initialization_manifest(config, model_config)
    anchor = load_anchor()

    selected_paths = panel.get('checkpoints') or DEFAULT_CHECKPOINTS
    checkpoints, records = {}, {}
    for label in ('best', 'late'):
        checkpoints[label], records[label] = stable_checkpoint(Path(selected_paths[label]))
    datasets = {split: CacheOnlyPairedDataset(manifest, config, split)
                for split in ('validation', 'train')}
    sample_maps = {split: {r['sample_id']: i for i, r in enumerate(ds.rows)}
                   for split, ds in datasets.items()}
    todo = []
    for split in ('validation', 'train'):
        for rec in panel[split]:
            todo.append(dict(rec, split=split))
    if include_case:
        # The documented SequentialLaneChange frame is reported as its own case
        # and never fills a behaviour quota.
        for row in datasets['validation'].rows:
            if 'SequentialLaneChange' in row['route'] and int(row['frame']) == 317:
                status = load_status(manifest, row)
                todo.append({'row': row, 'sample_id': row['sample_id'], 'route': row['route'],
                             'frame': row['frame'], 'stats': target_stats(load_target(manifest, row)),
                             'primary_class': 'case_study', 'split': 'validation',
                             'speed_mps': float(np.linalg.norm(status[4:6]))})
    if limit:
        todo = todo[:limit]

    agent = official_v1_module('transfuser_agent').TransfuserAgent(
        model_config, config['learning_rate'], checkpoint_path=None)
    agent.to(device).eval().requires_grad_(False)
    captured = {}

    def capture(module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError('unexpected diff_decoder output')
        regs, classes = output
        captured['reg'] = regs[-1].detach().to('cpu')
        captured['cls'] = classes[-1].detach().to('cpu')

    handle = agent._transfuser_model._trajectory_head.diff_decoder.register_forward_hook(capture)
    all_rows = []
    try:
        for label, checkpoint in checkpoints.items():
            state = checkpoint['state_dict']
            agent.load_state_dict({k[len('agent.'):]: v for k, v in state.items()}, strict=True)
            agent.eval()
            for index, rec in enumerate(todo):
                split, dataset = rec['split'], datasets[rec['split']]
                features, targets = dataset[sample_maps[split][rec['sample_id']]]
                f = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
                torch.manual_seed(panel['seed'] + index)
                captured.clear()
                with torch.inference_mode():
                    prediction = agent.forward(f)
                if 'reg' not in captured or 'cls' not in captured:
                    raise RuntimeError('candidate hook captured nothing')
                candidates = captured['reg'][0].numpy()
                scores = captured['cls'][0].numpy()
                all_rows.append(sample_record(rec, label, records[label], prediction,
                                              candidates, scores, targets, anchor))
                print(json.dumps({'event': 'sample', 'checkpoint': label,
                                  'split': split, 'class': rec['primary_class'],
                                  'index': index, 'sample_id': rec['sample_id']}), flush=True)
    finally:
        handle.remove()

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'candidate_samples.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_rows))
    protocol = {
        'schema': 'candidate-quality-evaluate-v1',
        'panel': str(Path(panel_path).resolve()), 'panel_sha256': file_digest(panel_path),
        'config': str(Path(panel['config']).resolve()), 'config_sha256': file_digest(panel['config']),
        'manifest': panel['manifest'], 'manifest_sha256': panel['manifest_sha256'],
        'checkpoints': records, 'seed': panel['seed'], 'device': device, 'precision': 'FP32',
        'targets_supplied_to_forward': False, 'optimizer_steps': 0,
        'candidate_capture': 'read-only forward hook on official diff_decoder',
        'candidate_count': 20, 'selection_rule': 'official argmax of the 20 mode scores',
        'oracle_definition': 'post-forward minimum-ADE candidate; upper bound only',
        'samples': len(all_rows), 'thresholds': THRESHOLDS,
        'noise_policy': 'torch.manual_seed(panel.seed + sample_index)',
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'event': 'evaluate_written', 'output': str(output.resolve()),
                      'rows': len(all_rows)}, ensure_ascii=False), flush=True)


def sample_record(rec, label, checkpoint_record, prediction, candidates, scores,
                  targets, anchor):
    truth = np.asarray(targets['trajectory'], dtype=np.float64)
    selected_xy = np.asarray(prediction['trajectory'][0], dtype=np.float64)
    selected_mode = int(np.argmax(scores))
    anchor_mode_index = anchor_mode(truth, anchor)

    candidate_geometry, candidate_error, candidate_delta = [], [], []
    for candidate in candidates:
        geometry = trajectory_geometry(np.asarray(candidate)[:, :2])
        candidate_geometry.append(geometry)
        candidate_error.append(ade_fde(candidate, truth))
    for shift, mode_anchor in zip(candidates, anchor):
        candidate_delta.append(float(np.linalg.norm(
            np.asarray(shift, np.float64)[:, :2] - mode_anchor).mean()))

    finite = [(i, e) for i, e in enumerate(candidate_error) if e is not None]
    if not finite:
        raise RuntimeError('no finite candidate: ' + rec['sample_id'])
    oracle_mode, oracle_error = min(finite, key=lambda kv: kv[1]['ade_m'])
    selected_error = ade_fde(selected_xy, truth)
    if selected_error is None:
        raise RuntimeError('non-finite official output: ' + rec['sample_id'])
    matches = bool(np.allclose(selected_xy[:, :2], np.asarray(candidates[selected_mode],
                                                             np.float64)[:, :2]))
    geometry = candidate_geometry
    gt_geometry = trajectory_geometry(truth[:, :2], truth[:, 2])
    near_zero = [i for i, g in enumerate(geometry)
                 if g['finite'] and g['endpoint_m'] <= THRESHOLDS['near_zero_end_m']]
    turns_left = [i for i, g in enumerate(geometry)
                  if g['finite'] and g['total_yaw_deg'] >= THRESHOLDS['turn_yaw_deg']]
    turns_right = [i for i, g in enumerate(geometry)
                   if g['finite'] and g['total_yaw_deg'] <= -THRESHOLDS['turn_yaw_deg']]
    return {
        'checkpoint': label, 'global_step': checkpoint_record['global_step'],
        'checkpoint_sha256': checkpoint_record['sha256'], 'split': rec['split'],
        'primary_class': rec['primary_class'], 'sample_id': rec['sample_id'],
        'route': rec['route'], 'frame': int(rec['frame']),
        'feature_sha256': rec['row']['feature_sha'], 'target_sha256': rec['row']['target_sha'],
        'speed_mps': float(rec['speed_mps']),
        'anchor_mode': anchor_mode_index,
        'selected_mode': selected_mode,
        'oracle_mode': int(oracle_mode),
        'selected_argmax_matches': matches,
        'score_margin': float(np.sort(scores)[-1] - np.sort(scores)[-2]),
        'selected_error': selected_error, 'oracle_error': oracle_error,
        'anchor_error': candidate_error[anchor_mode_index],
        'target_geometry': gt_geometry,
        'candidate_geometry': geometry,
        'candidate_flags': [geometry_flags(g) for g in geometry],
        'candidate_delta_from_anchor_m': candidate_delta,
        'candidate_ade_m': [None if e is None else e['ade_m'] for e in candidate_error],
        'candidate_fde_m': [None if e is None else e['fde_m'] for e in candidate_error],
        'candidate_scores': np.asarray(scores).tolist(),
        'near_zero_candidates': near_zero,
        'left_turn_candidates': turns_left,
        'right_turn_candidates': turns_right,
        'candidates': [np.asarray(c).tolist() for c in candidates],
        'anchor_xy': anchor.tolist(),
        'target': truth.tolist(),
    }


# ------------------------------------------------------------- report phase

def enrich(rows):
    """Add heading-channel angles and re-derive flags consistently.

    The 20 anchors carry no heading channel, so their angle is the displacement
    chord. Candidates and targets do carry `[...,2]`, so the turn analysis uses
    that channel and reports the channel-vs-chord disagreement explicitly.
    """
    for r in rows:
        candidates = np.asarray(r['candidates'], dtype=np.float64)
        channel = [wrap(float(c[-1, 2])) * 180 / math.pi for c in candidates]
        r['candidate_yaw_deg'] = channel
        r['candidate_yaw_chord_gap_deg'] = [
            abs(wrap((c - g['total_yaw_deg'] / 180 * math.pi))) * 180 / math.pi
            for c, g in zip(channel, r['candidate_geometry'])]
        r['yaw_inconsistent'] = [gap > 30.0 for gap in r['candidate_yaw_chord_gap_deg']]
        r['candidate_flags'] = [
            geometry_flags(dict(g, total_yaw_deg=y, finite=g['finite']))
            + ('|yaw_mismatch' if bad else '')
            for g, y, bad in zip(r['candidate_geometry'], channel, r['yaw_inconsistent'])]
        r['target_yaw_deg'] = wrap(float(np.asarray(r['target'])[-1, 2])) * 180 / math.pi
        anchor = np.asarray(r['anchor_xy'], dtype=np.float64)
        truth = np.asarray(r['target'], dtype=np.float64)
        # What the raw prior alone would produce, with no decoder contribution.
        own = np.linalg.norm(anchor[r['anchor_mode']][:, :2] - truth[:, :2], axis=-1).mean()
        r['anchor_own_ade_m'] = float(own)
        r['anchor_own_end_m'] = float(np.linalg.norm(anchor[r['anchor_mode']][-1]))
        r['anchor_ade_m'] = r['anchor_error']['ade_m']
        r['decoder_gain_m'] = float(own - r['anchor_error']['ade_m'])
    return rows


def summarise(rows):
    def share(predicate, denominator=None):
        pool = denominator if denominator is not None else rows
        return fraction(sum(1 for r in pool if predicate(r)), len(pool))

    selected = [r['selected_error'] for r in rows]
    oracle = [r['oracle_error'] for r in rows]
    anchors = [r['anchor_error'] for r in rows]
    return {
        'samples': len(rows),
        'selected_ade': mean([e['ade_m'] for e in selected]),
        'oracle_ade': mean([e['ade_m'] for e in oracle]),
        'anchor_ade': mean([e['ade_m'] for e in anchors]),
        'selection_gap': mean([s['ade_m'] - o['ade_m'] for s, o in zip(selected, oracle)]),
        'anchor_gap': mean([a['ade_m'] - o['ade_m'] for a, o in zip(anchors, oracle)]),
        'selected_oracle_agree': share(lambda r: r['selected_mode'] == r['oracle_mode']),
        'anchor_oracle_agree': share(lambda r: r['anchor_mode'] == r['oracle_mode']),
        'anchor_selected_agree': share(lambda r: r['anchor_mode'] == r['selected_mode']),
        'min_candidate_end_m': mean([min(g['endpoint_m'] for g in r['candidate_geometry'])
                                     for r in rows]),
        'min_candidate_path_m': mean([min(g['path_m'] for g in r['candidate_geometry'])
                                      for r in rows]),
        'near_zero_available': share(lambda r: len(r['near_zero_candidates']) > 0),
        'near_zero_selected': share(lambda r: r['selected_mode'] in r['near_zero_candidates']),
        'nonfinite_candidates': share(lambda r: any(not g['finite']
                                                    for g in r['candidate_geometry'])),
        'selected_end_m': mean([r['candidate_geometry'][r['selected_mode']]['endpoint_m']
                                for r in rows]),
        'oracle_end_m': mean([r['candidate_geometry'][r['oracle_mode']]['endpoint_m']
                              for r in rows]),
        'target_end_m': mean([r['target_geometry']['endpoint_m'] for r in rows]),
        'target_path_m': mean([r['target_geometry']['path_m'] for r in rows]),
        'yaw_channel_chord_gap_deg': mean([g for r in rows for g in r['candidate_yaw_chord_gap_deg']]),
        'yaw_mismatch_rate': share(lambda r: any(r['yaw_inconsistent'])),
        'selected_delta_from_anchor_m': mean([r['candidate_delta_from_anchor_m'][r['selected_mode']]
                                              for r in rows]),
        'anchor_own_ade': mean([r['anchor_own_ade_m'] for r in rows]),
        'anchor_own_end_m': mean([r['anchor_own_end_m'] for r in rows]),
        'decoder_gain_m': mean([r['decoder_gain_m'] for r in rows]),
        'selected_end_over_target': mean([r['candidate_geometry'][r['selected_mode']]['endpoint_m']
                                          / max(r['target_geometry']['endpoint_m'], .1)
                                          for r in rows]),
    }


def turn_summary(rows):
    out = []
    for r in rows:
        gt_yaw = r['target_yaw_deg']
        if abs(gt_yaw) < THRESHOLDS['turn_yaw_deg']:
            continue
        sign = 1 if gt_yaw > 0 else -1
        same = [v * sign for v in r['candidate_yaw_deg']]
        adequate = [v for v in same
                    if v >= THRESHOLDS['turn_yaw_deg'] * THRESHOLDS['turn_adequate_ratio']]
        selected_yaw = r['candidate_yaw_deg'][r['selected_mode']] * sign
        oracle_yaw = r['candidate_yaw_deg'][r['oracle_mode']] * sign
        out.append({'samples': 1, 'gt_yaw': abs(gt_yaw),
                    'adequate': 1 if adequate else 0,
                    'adequate_count': len(adequate),
                    'selected_yaw': selected_yaw, 'oracle_yaw': oracle_yaw,
                    'insufficient': 1 if selected_yaw < THRESHOLDS['turn_yaw_deg'] else 0,
                    'max_same_yaw': max(same) if same else None})
    return out


def write_report(output, rows, panel_path, checkpoint_records, anchor_rows):
    rows = enrich(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['checkpoint'], r['split'], r['primary_class'])].append(r)
        groups[(r['checkpoint'], r['split'], 'ALL')].append(r)

    fields = ['checkpoint', 'split', 'primary_class'] + list(summarise(rows[:1]))
    with (output / 'behaviour_candidate_summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (label, split, cls), group in sorted(groups.items()):
            writer.writerow({'checkpoint': label, 'split': split, 'primary_class': cls,
                             **summarise(group)})

    fields = ['checkpoint', 'split', 'primary_class', 'samples', 'gt_turn_yaw_deg',
              'adequate_turn_coverage', 'mean_adequate_candidates', 'selected_turn_yaw_deg',
              'oracle_turn_yaw_deg', 'selected_insufficient_rate', 'max_same_sign_yaw_deg']
    with (output / 'turn_candidate_summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (label, split, cls), group in sorted(groups.items()):
            if cls not in TURN_CLASSES:
                continue
            stats = turn_summary(group)
            writer.writerow({'checkpoint': label, 'split': split, 'primary_class': cls,
                             'samples': len(stats),
                             'gt_turn_yaw_deg': mean([s['gt_yaw'] for s in stats]),
                             'adequate_turn_coverage': fraction(sum(s['adequate'] for s in stats),
                                                                len(stats)),
                             'mean_adequate_candidates': mean([s['adequate_count'] for s in stats]),
                             'selected_turn_yaw_deg': mean([s['selected_yaw'] for s in stats]),
                             'oracle_turn_yaw_deg': mean([s['oracle_yaw'] for s in stats]),
                             'selected_insufficient_rate': fraction(sum(s['insufficient'] for s in stats),
                                                                    len(stats)),
                             'max_same_sign_yaw_deg': mean([s['max_same_yaw'] for s in stats])})

    # Per-sample × per-mode geometry.
    fields = ['checkpoint', 'split', 'primary_class', 'sample_id', 'route', 'frame', 'mode',
              'is_selected', 'is_oracle', 'is_anchor_mode', 'flags', 'ade_m', 'fde_m',
              'anchor_delta_m', 'yaw_channel_deg', 'yaw_chord_gap_deg']
    with (output / 'candidate_geometry.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + list(
            k for k in rows[0]['candidate_geometry'][0] if k != 'finite'))
        writer.writeheader()
        for r in rows:
            for mode in range(20):
                writer.writerow({
                    'checkpoint': r['checkpoint'], 'split': r['split'],
                    'primary_class': r['primary_class'], 'sample_id': r['sample_id'],
                    'route': r['route'], 'frame': r['frame'], 'mode': mode,
                    'is_selected': r['selected_mode'] == mode,
                    'is_oracle': r['oracle_mode'] == mode,
                    'is_anchor_mode': r['anchor_mode'] == mode,
                    'flags': r['candidate_flags'][mode],
                    'ade_m': r['candidate_ade_m'][mode], 'fde_m': r['candidate_fde_m'][mode],
                    'anchor_delta_m': r['candidate_delta_from_anchor_m'][mode],
                    'yaw_channel_deg': r['candidate_yaw_deg'][mode],
                    'yaw_chord_gap_deg': r['candidate_yaw_chord_gap_deg'][mode],
                    **{k: v for k, v in r['candidate_geometry'][mode].items() if k != 'finite'}})

    # Best vs late, same sample and same noise seed.
    best = {(r['split'], r['sample_id']): r for r in rows if r['checkpoint'] == 'best'}
    late = {(r['split'], r['sample_id']): r for r in rows if r['checkpoint'] == 'late'}
    keys = sorted(set(best) & set(late))
    fields = ['split', 'primary_class', 'sample_id', 'route', 'frame',
              'best_selected_ade', 'late_selected_ade', 'best_oracle_ade', 'late_oracle_ade',
              'best_sel_mode', 'late_sel_mode', 'best_min_end_m', 'late_min_end_m',
              'best_near_zero_count', 'late_near_zero_count',
              'best_selected_end_m', 'late_selected_end_m',
              'best_oracle_end_m', 'late_oracle_end_m',
              'best_geometry_signature', 'late_geometry_signature']
    with (output / 'best_late_comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in keys:
            b, l = best[key], late[key]
            writer.writerow({
                'split': b['split'], 'primary_class': b['primary_class'],
                'sample_id': b['sample_id'], 'route': b['route'], 'frame': b['frame'],
                'best_selected_ade': b['selected_error']['ade_m'],
                'late_selected_ade': l['selected_error']['ade_m'],
                'best_oracle_ade': b['oracle_error']['ade_m'],
                'late_oracle_ade': l['oracle_error']['ade_m'],
                'best_sel_mode': b['selected_mode'], 'late_sel_mode': l['selected_mode'],
                'best_min_end_m': min(g['endpoint_m'] for g in b['candidate_geometry']),
                'late_min_end_m': min(g['endpoint_m'] for g in l['candidate_geometry']),
                'best_near_zero_count': len(b['near_zero_candidates']),
                'late_near_zero_count': len(l['near_zero_candidates']),
                'best_selected_end_m': b['candidate_geometry'][b['selected_mode']]['endpoint_m'],
                'late_selected_end_m': l['candidate_geometry'][l['selected_mode']]['endpoint_m'],
                'best_oracle_end_m': b['candidate_geometry'][b['oracle_mode']]['endpoint_m'],
                'late_oracle_end_m': l['candidate_geometry'][l['oracle_mode']]['endpoint_m'],
                'best_geometry_signature': digest([round(g['endpoint_m'], 3)
                                                   for g in b['candidate_geometry']]),
                'late_geometry_signature': digest([round(g['endpoint_m'], 3)
                                                   for g in l['candidate_geometry']])})

    plots(output, rows, best, late)
    protocol = {
        'schema': 'candidate-quality-diagnostic-v1',
        'panel': str(Path(panel_path).resolve()), 'panel_sha256': file_digest(panel_path),
        'checkpoints': checkpoint_records, 'rows': len(rows), 'thresholds': THRESHOLDS,
        'anchor_geometry_rows': len(anchor_rows),
        'note': ('All ADE/FDE/oracle/gap/behaviour quantities are evaluation-only and '
                 'never enter training, scoring, argmax, sampling or checkpoint selection.'),
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'event': 'report_written', 'output': str(output.resolve())},
                     ensure_ascii=False), flush=True)


def plots(output, rows, best, late):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cases = output / 'representative_cases'
    cases.mkdir(exist_ok=True)

    def draw_trajectories(ax, row, title):
        ax.axhline(0, color='black', lw=.6)
        ax.axvline(0, color='black', lw=.6)
        ax.plot(0, 0, 's', color='black', ms=5)
        target = np.asarray(row['target'])
        ax.plot(target[:, 1], target[:, 0], 'o-', color='tab:green', ms=3, label='GT')
        for mode in range(20):
            candidate = np.asarray(row['candidates'][mode])
            ax.plot(candidate[:, 1], candidate[:, 0], '-', color='lightgray', lw=.7)
        oracle = np.asarray(row['candidates'][row['oracle_mode']])
        ax.plot(oracle[:, 1], oracle[:, 0], '--', color='tab:orange', lw=1.6, label='oracle')
        selected = np.asarray(row['candidates'][row['selected_mode']])
        ax.plot(selected[:, 1], selected[:, 0], '-', color='tab:red', lw=2.0, label='selected')
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('lateral y (m), +left')
        ax.set_ylabel('forward x (m)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=6)

    # One extreme failure per behaviour, same sample for both checkpoints.
    picked = []
    for cls in ('stationary', 'low_depart', 'left_turn', 'right_turn'):
        pool = [r for r in rows if r['checkpoint'] == 'best'
                and r['split'] == 'validation' and r['primary_class'] == cls]
        if pool:
            picked.append((cls, max(pool, key=lambda r: r['selected_error']['ade_m'])))
    if picked:
        fig, axes = plt.subplots(len(picked), 2, figsize=(11, 4.4 * len(picked)),
                                 squeeze=False, constrained_layout=True)
        for row_index, (cls, anchor_row) in enumerate(picked):
            key = (anchor_row['split'], anchor_row['sample_id'])
            for column, (label, table) in enumerate((('best', best), ('late', late))):
                row = table.get(key)
                if row is None:
                    continue
                draw_trajectories(
                    axes[row_index][column], row,
                    f'{cls} {label} step {row["global_step"]}\n'
                    f'selected ADE {row["selected_error"]["ade_m"]:.2f} m / '
                    f'oracle {row["oracle_error"]["ade_m"]:.2f} m / '
                    f'anchor {row["anchor_ade_m"]:.2f} m')
        fig.savefig(cases / 'behaviour_extremes.png', dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(picked), figsize=(4.4 * len(picked), 4.6),
                                 squeeze=False, constrained_layout=True)
        for ax, (cls, row) in zip(axes.flat, picked):
            ends = [g['endpoint_m'] for g in row['candidate_geometry']]
            colours = ['tab:red' if i == row['selected_mode'] else
                       ('tab:orange' if i == row['oracle_mode'] else 'tab:gray') for i in range(20)]
            ax.bar(range(20), ends, color=colours)
            ax.axhline(row['target_geometry']['endpoint_m'], color='tab:green', ls='--',
                       label='GT endpoint')
            ax.axhline(THRESHOLDS['near_zero_end_m'], color='black', ls=':', lw=.8)
            ax.set_title(f'{cls} best: per-mode endpoint', fontsize=8)
            ax.set_xlabel('mode')
            ax.set_ylabel('m')
            ax.grid(alpha=.25, axis='y')
            ax.legend(fontsize=6)
        fig.savefig(cases / 'per_mode_endpoint.png', dpi=140)
        plt.close(fig)

    # Frame317 style check: the documented severe sequential-lane-change frame.
    case = [r for r in rows if 'SequentialLaneChange' in r['route'] and int(r['frame']) == 317]
    if case:
        fig, axes = plt.subplots(1, len(case), figsize=(5 * len(case), 4.8),
                                 squeeze=False, constrained_layout=True)
        for ax, row in zip(axes.flat, case):
            draw_trajectories(ax, row, f'{row["checkpoint"]} step {row["global_step"]}\n'
                                       f'selected ADE {row["selected_error"]["ade_m"]:.2f} m')
        fig.savefig(cases / 'frame317.png', dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    classes = list(CLASSES)
    for label, colour in (('best', 'tab:blue'), ('late', 'tab:red')):
        for ax, key in zip(axes, ('min_candidate_end_m', 'selected_end_m', 'target_end_m')):
            values = []
            for cls in classes:
                pool = [r for r in rows if r['checkpoint'] == label
                        and r['split'] == 'validation' and r['primary_class'] == cls]
                values.append(summarise(pool)[key] if pool else None)
            ax.plot(range(len(classes)), values, 'o-', color=colour, label=label)
    for ax, title in zip(axes, ('min candidate endpoint', 'selected endpoint', 'GT endpoint')):
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylabel('m')
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(output / 'endpoint_by_behaviour.png', dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('anchors')
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--coverage-split', default='validation',
                   choices=['validation', 'train', 'both'])
    p = sub.add_parser('evaluate')
    p.add_argument('--panel', type=Path, default=PANEL)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--include-case', action='store_true')
    p = sub.add_parser('report')
    p.add_argument('--panel', type=Path, default=PANEL)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--anchors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action != 'report' and args.output.exists():
        raise FileExistsError('use a fresh explicit output directory: ' + str(args.output))
    if args.action == 'anchors':
        anchors_command(args.config, args.output, args.coverage_split)
    elif args.action == 'evaluate':
        evaluate_command(args.panel, args.output, args.limit, args.device, args.include_case)
    else:
        from tools.training.diffusiondrive_v1.evaluate_checkpoint import stable_checkpoint
        panel = json.loads(Path(args.panel).read_text())
        rows = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
        records = {}
        for label, path in (panel.get('checkpoints') or DEFAULT_CHECKPOINTS).items():
            _, records[label] = stable_checkpoint(Path(path))
        with (args.anchors / 'anchor_geometry.csv').open() as stream:
            anchor_rows = list(csv.DictReader(stream))
        args.output.mkdir(parents=True, exist_ok=True)
        write_report(args.output, rows, args.panel, records, anchor_rows)


if __name__ == '__main__':
    main()
