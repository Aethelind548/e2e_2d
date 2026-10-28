"""Read-only behavior-balanced checkpoint attribution for the Scratch V1 run.

The tool never trains, never updates parameters, never rebuilds caches and never
writes to the training output. It builds a fixed behavior panel from the
authenticated pair index, evaluates selected checkpoints without supplying
targets to forward, and can capture the 20 frozen trajectory candidates through
a read-only forward hook on the official diffusion decoder. The `report`
subcommand rebuilds CSV/plot/protocol artifacts from saved sample JSONL without
repeating inference.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
CONFIG = ROOT / 'tools/training/diffusiondrive_v1/baseline_bs8_acc8_lr1e4_scratch.json'
DEFAULT_CHECKPOINTS = {
    'best': Path('/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_scratch_20261021_210751/checkpoints/best-epoch002-step009882.ckpt'),
    'late': Path('/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/batch64_lr1e4_scratch_20261021_210751/health/diagnostic-step000058500.ckpt'),
}
CLASSES = ('stationary', 'approaching_stop', 'low_creep', 'low_depart',
           'left_turn', 'right_turn', 'moving_straight', 'interaction_decision')
TRAIN_PER_CLASS = 8
VAL_PER_CLASS = 16
SEED = 20261022
TIMES = np.arange(1, 9) * .5
INTERACTION_HINTS = (
    'priority', 'yield', 'pedestrian', 'blocked', 'merging', 'obstacle',
    'accident', 'hazard', 'construction', 'parked', 'crossing', 'dynamicobject',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def readonly(path):
    db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def wrap(value):
    return float(np.arctan2(np.sin(value), np.cos(value)))


def scenario(route):
    return Path(route).name.split('_Town')[0]


def load_manifest(config_path):
    config = json.loads(Path(config_path).read_text())
    manifest_path = ROOT / config['paired_manifest']
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'b2d-v1-explicit-recovery-paired-v1':
        raise ValueError('This attribution tool requires the authenticated recovery pair manifest')
    return config, manifest, manifest_path


def pair_rows(manifest, split):
    with readonly(manifest['pair_index']) as db:
        rows = [dict(r) for r in db.execute(
            'SELECT * FROM pairs WHERE split=? ORDER BY source_rank DESC,route,frame', (split,))]
    return rows


def load_target(manifest, row):
    from tools.training.diffusiondrive_v1.target_storage import load_target_row
    source = manifest['target_sources'][row['target_source']]
    with readonly(Path(source['state']) / 'index.sqlite') as db:
        target = db.execute('SELECT * FROM samples WHERE sample_id=?', (row['sample_id'],)).fetchone()
    if target is None or target['state'] != 'complete' or target['sha256'] != row['target_sha']:
        raise ValueError('Target index mismatch: ' + row['sample_id'])
    return load_target_row(dict(target), source['target_root'], source['producer_sha256'])['targets']


def load_status(manifest, row):
    from tools.training.diffusiondrive_v1.features_cache import load_feature_row
    from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
    source = manifest['feature_sources'][row['feature_source']]
    with readonly(Path(source['state']) / 'index.sqlite') as db:
        feature = db.execute('SELECT * FROM samples WHERE sample_id=?', (row['sample_id'],)).fetchone()
    if feature is None or feature['state'] != 'complete' or feature['sha256'] != row['feature_sha']:
        raise ValueError('Feature index mismatch: ' + row['sample_id'])
    values = load_feature_row(dict(feature), effective_roots(manifest, source), source['producer_sha256'])
    status = np.asarray(values['status_feature'], dtype=np.float64).reshape(-1)
    if status.shape != (8,):
        raise ValueError('Unexpected status_feature shape: ' + str(status.shape))
    return status


def target_stats(target):
    xy = np.asarray(target['trajectory'][:, :2], dtype=np.float64)
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0), axis=1)
    heading = np.asarray(target['trajectory'][:, 2], dtype=np.float64)
    return {
        'path_m': float(steps.sum()),
        'first_speed_mps': float(steps[0] / .5),
        'last_speed_mps': float(steps[-1] / .5),
        'heading_change_rad': wrap(float(heading[-1])),
        'max_step_m': float(steps.max()),
    }


def tags_for(stats, route):
    name = scenario(route).lower()
    tags = set()
    dh = abs(stats['heading_change_rad'])
    path = stats['path_m']
    if path <= .5 and dh < math.radians(10): tags.add('stationary')
    if stats['first_speed_mps'] > .5 and stats['last_speed_mps'] < .3 and dh < math.radians(20):
        tags.add('approaching_stop')
    if .5 < path <= 5 and dh < math.radians(10): tags.add('low_creep')
    if path > 5 and stats['first_speed_mps'] < 1.0 and dh < math.radians(10): tags.add('low_depart')
    if path > 2 and dh >= math.radians(20):
        tags.add('left_turn' if stats['heading_change_rad'] > 0 else 'right_turn')
    if path > 5 and dh < math.radians(10) and stats['first_speed_mps'] >= 1.0:
        tags.add('moving_straight')
    if any(hint in name for hint in INTERACTION_HINTS) and dh < math.radians(25) and path > 1:
        tags.add('interaction_decision')
    return tags


def stable_order(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256((r['sample_id'] + salt).encode()).hexdigest())


def behavior_ok(cls, speed, stats):
    if cls == 'stationary':
        return speed < .5
    if cls == 'low_creep':
        return speed < 1.0
    if cls == 'low_depart':
        return speed < .5
    if cls == 'approaching_stop':
        return speed > .5 and stats['last_speed_mps'] < .5
    if cls == 'moving_straight':
        return speed >= 1.0
    if cls in ('left_turn', 'right_turn'):
        return abs(stats['heading_change_rad']) >= math.radians(15)
    return True


def choose(records, cls, count, used, manifest, max_per_route=2):
    pool = stable_order([r for r in records if cls in r['tags'] and r['sample_id'] not in used], cls + str(SEED))
    chosen = []
    route_count = Counter()
    for rec in pool:
        if route_count[rec['route']] >= max_per_route:
            continue
        try:
            status = load_status(manifest, rec['row'])
        except Exception as exc:
            rec.setdefault('status_errors', []).append(str(exc))
            continue
        speed = float(np.linalg.norm(status[4:6]))
        if not behavior_ok(cls, speed, rec['stats']):
            continue
        rec = dict(rec, speed_mps=speed, speed_velocity=status[4:6].tolist(),
                   acceleration=status[6:8].tolist(), status=status.tolist(), primary_class=cls)
        chosen.append(rec)
        used.add(rec['sample_id'])
        route_count[rec['route']] += 1
        if len(chosen) == count:
            return chosen
    if len(chosen) < count:
        # Keep the class honest: report shortage; do not duplicate samples or
        # relax the behavior threshold merely to fill a quota.
        raise RuntimeError(f'{cls}: only {len(chosen)} eligible samples, requested {count}')
    return chosen


def select(config_path, output):
    config, manifest, manifest_path = load_manifest(config_path)
    validation_rows = pair_rows(manifest, 'validation')
    print(json.dumps({'event': 'validation_rows', 'count': len(validation_rows)}), flush=True)
    validation_records = []
    for index, row in enumerate(validation_rows):
        target = load_target(manifest, row)
        stats = target_stats(target)
        validation_records.append({'row': row, 'sample_id': row['sample_id'], 'route': row['route'],
                                   'frame': row['frame'], 'stats': stats, 'tags': sorted(tags_for(stats, row['route']))})
        if (index + 1) % 500 == 0:
            print(json.dumps({'event': 'target_scan', 'processed': index + 1, 'total': len(validation_rows)}), flush=True)
    used = set()
    validation = []
    for cls in CLASSES:
        validation.extend(choose(validation_records, cls, VAL_PER_CLASS, used, manifest))

    # The existing 64-frame attribution panel supplies the four low-speed
    # training strata. Additional training classes are selected from the
    # authenticated pair index without duplicating those sample_ids.
    train_rows = pair_rows(manifest, 'train')
    by_id = {r['sample_id']: r for r in train_rows}
    panel_path = ROOT / 'outputs/diffusiondrive_v1/checkpoint_evaluation/attribution_samples.jsonl'
    panel_ids = []
    if panel_path.exists():
        for line in panel_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get('split') == 'train' and rec['sample_id'] in by_id:
                panel_ids.append(rec['sample_id'])
    train_records = []
    selected_train_ids = set()
    for sid in panel_ids:
        row = by_id[sid]
        target = load_target(manifest, row)
        stats = target_stats(target)
        train_records.append({'row': row, 'sample_id': row['sample_id'], 'route': row['route'],
                              'frame': row['frame'], 'stats': stats, 'tags': sorted(tags_for(stats, row['route']))})
    # Scan a deterministic subset of train turn/interaction routes. The scan is
    # for behavior classification only; it does not consume training tensors.
    candidates = [r for r in train_rows if ('turn' in scenario(r['route']).lower()
                  or any(h in scenario(r['route']).lower() for h in INTERACTION_HINTS))]
    candidates = stable_order(candidates, 'train-scan')[:4000]
    for index, row in enumerate(candidates):
        if row['sample_id'] in {r['row']['sample_id'] for r in train_records}:
            continue
        try:
            target = load_target(manifest, row)
        except Exception:
            continue
        stats = target_stats(target)
        train_records.append({'row': row, 'sample_id': row['sample_id'], 'route': row['route'],
                              'frame': row['frame'], 'stats': stats, 'tags': sorted(tags_for(stats, row['route']))})
        if (index + 1) % 500 == 0:
            print(json.dumps({'event': 'train_scan', 'processed': index + 1, 'total': len(candidates)}), flush=True)
    # Choose exactly 8 per class through the same deterministic class order.
    # The existing low-speed panel is a seed pool, not a separate quota.
    used = set()
    train = []
    for cls in CLASSES:
        train.extend(choose(train_records, cls, TRAIN_PER_CLASS, used, manifest))

    result = {
        'schema': 'scratch-behavior-panel-v1',
        'config': str(Path(config_path).resolve()),
        'config_sha256': file_digest(config_path),
        'manifest': str(manifest_path.resolve()),
        'manifest_sha256': file_digest(manifest_path),
        'seed': SEED,
        'times_s': TIMES.tolist(),
        'thresholds': {
            'stationary_path_m_max': .5, 'heading_straight_deg': 10,
            'low_creep_path_m': [.5, 5], 'low_depart_path_m_min': 5,
            'low_depart_first_speed_mps_max': 1.0,
            'approach_first_speed_mps_min': .5, 'approach_last_speed_mps_max': .3,
            'turn_heading_deg_min': 20, 'turn_selection_heading_deg_min': 15,
            'stationary_low_creep_low_depart_current_speed_mps_max': 1.0,
            'interaction_route_hints': list(INTERACTION_HINTS),
        },
        'validation': validation,
        'train': train,
        'counts': {'validation': dict(Counter(r['primary_class'] for r in validation)),
                   'train': dict(Counter(r['primary_class'] for r in train))},
        'route_overlap': sorted({r['route'] for r in validation} & {r['route'] for r in train}),
        'panel_scan_note': 'Validation scans all 7846 targets; training uses the existing 64-frame panel plus a deterministic turn/interaction route subset.',
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'behavior_panel.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (output / 'behavior_panel.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split', 'primary_class', 'tags', 'sample_id', 'route', 'frame',
                                                    'speed_mps', 'path_m', 'first_speed_mps', 'last_speed_mps',
                                                    'heading_change_rad', 'feature_sha', 'target_sha'])
        writer.writeheader()
        for split in ('validation', 'train'):
            for rec in result[split]:
                row = rec['row']
                writer.writerow({'split': split, 'primary_class': rec['primary_class'],
                                 'tags': '|'.join(rec['tags']), 'sample_id': row['sample_id'],
                                 'route': row['route'], 'frame': row['frame'],
                                 'speed_mps': rec['speed_mps'], 'path_m': rec['stats']['path_m'],
                                 'first_speed_mps': rec['stats']['first_speed_mps'],
                                 'last_speed_mps': rec['stats']['last_speed_mps'],
                                 'heading_change_rad': rec['stats']['heading_change_rad'],
                                 'feature_sha': row['feature_sha'], 'target_sha': row['target_sha']})
    print(json.dumps({'event': 'panel_written', 'output': str(output.resolve()),
                      'counts': result['counts']}, ensure_ascii=False), flush=True)


def load_panel(path):
    return json.loads(Path(path).read_text())


def mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def pairwise_delta(best, late, key):
    if best.get(key) is None or late.get(key) is None:
        return None
    return float(late[key] - best[key])


def candidate_metrics(prediction, target, candidates, scores):
    from tools.training.diffusiondrive_v1.evaluate_checkpoint import trajectory_errors, detection_counts, semantic_counts
    selected = np.asarray(prediction['trajectory'][0], dtype=np.float64)
    truth = np.asarray(target['trajectory'], dtype=np.float64)
    selected_error = trajectory_errors(selected, truth)
    candidate_errors = [trajectory_errors(np.asarray(c, dtype=np.float64), truth) for c in candidates]
    finite = [(i, e) for i, e in enumerate(candidate_errors) if e is not None]
    if not finite:
        raise RuntimeError('No finite candidate trajectory')
    oracle_mode, oracle_error = min(finite, key=lambda item: item[1]['ade_m'])
    selected_mode = int(np.argmax(scores))
    top = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    cm, invalid_pixels = semantic_counts(prediction['bev_semantic_map'][0], target['bev_semantic_map'])
    det = detection_counts(prediction['agent_states'][0], prediction['agent_labels'][0],
                            target['agent_states'], target['agent_labels'])
    diag = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - diag
    ious = [float(a / b) if b else None for a, b in zip(diag, union)]
    return {'selected': selected.tolist(), 'target': truth.tolist(),
            'selected_error': selected_error, 'oracle_error': oracle_error,
            'selected_mode': selected_mode, 'oracle_mode': int(oracle_mode),
            'selected_argmax_matches': bool(np.allclose(selected, np.asarray(candidates[selected_mode], dtype=np.float64))),
            'score_margin': float(top[0] - top[1]) if len(top) > 1 else None,
            'candidate_ades': [None if e is None else e['ade_m'] for e in candidate_errors],
            'candidate_fdes': [None if e is None else e['fde_m'] for e in candidate_errors],
            'candidate_scores': np.asarray(scores).tolist(),
            'candidates': [np.asarray(c).tolist() for c in candidates],
            'detection': det, 'bev_confusion': cm.tolist(), 'invalid_bev_pixels': invalid_pixels,
            'bev_iou': dict(zip(CLASS_NAMES, ious)),
            'bev_miou': mean([x for x in ious if x is not None]),
            'bev_foreground_miou': mean([x for x in ious[1:] if x is not None]),
            'selection_gap': float(selected_error['ade_m'] - oracle_error['ade_m']),
            'oracle_gain_fde': float(selected_error['fde_m'] - oracle_error['fde_m'])}


CLASS_NAMES = ['background', 'road', 'walkway', 'centerline', 'static', 'vehicle', 'pedestrian']


def write_reports(output, all_rows, panel, panel_path, checkpoint_records):
    """Write aggregates, failure cases, plots, and the frozen protocol."""
    # The case study is reported separately and never fills a behavior quota.
    normal_rows = [r for r in all_rows if not r.get('case_study', False)]
    groups = defaultdict(list)
    for r in normal_rows:
        groups[(r['checkpoint'], r['split'], r['primary_class'])].append(r)
    comparison_groups = defaultdict(list)
    for r in normal_rows:
        comparison_groups[(r['checkpoint'], r['split'])].append(r)

    def aggregate(rows):
        selected = [r['selected_error'] for r in rows]
        oracle = [r['oracle_error'] for r in rows]
        det_tp = sum(r['detection'].get('tp', 0) for r in rows)
        det_fp = sum(r['detection'].get('fp', 0) for r in rows)
        det_fn = sum(r['detection'].get('fn', 0) for r in rows)
        return {'samples': len(rows),
                'selected_ade': mean([e['ade_m'] for e in selected]),
                'selected_fde': mean([e['fde_m'] for e in selected]),
                'oracle_ade': mean([e['ade_m'] for e in oracle]),
                'oracle_fde': mean([e['fde_m'] for e in oracle]),
                'selection_gap': mean([r['selection_gap'] for r in rows]),
                'longitudinal_mae_m': mean([e['longitudinal_mae_m'] for e in selected]),
                'lateral_mae_m': mean([e['lateral_mae_m'] for e in selected]),
                'heading_mae_deg': mean([e['heading_mae_rad'] * 180 / np.pi for e in selected]),
                'detection_f1': (2 * det_tp / (2 * det_tp + det_fp + det_fn)) if (2 * det_tp + det_fp + det_fn) else None,
                'bev_miou': mean([r['bev_miou'] for r in rows]),
                'nonfinite': sum(1 for r in rows if not np.isfinite(r['selected_error']['ade_m']))}

    with (output / 'checkpoint_comparison.csv').open('x', newline='') as stream:
        fields = ['checkpoint', 'split', 'samples', 'selected_ade', 'selected_fde', 'oracle_ade', 'oracle_fde',
                  'selection_gap', 'longitudinal_mae_m', 'lateral_mae_m', 'heading_mae_deg', 'detection_f1',
                  'bev_miou', 'nonfinite']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for (label, split), rows in sorted(comparison_groups.items()):
            if split == 'validation':
                writer.writerow({'checkpoint': label, 'split': split, **aggregate(rows)})
    with (output / 'behavior_metrics.csv').open('x', newline='') as stream:
        fields = ['checkpoint', 'split', 'primary_class', 'samples', 'selected_ade', 'selected_fde',
                  'oracle_ade', 'oracle_fde', 'selection_gap', 'longitudinal_mae_m', 'lateral_mae_m',
                  'heading_mae_deg', 'detection_f1', 'bev_miou', 'false_motion_rate', 'missed_depart_rate']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for (label, split, cls), rows in sorted(groups.items()):
            agg = aggregate(rows)
            false_motion = [r for r in rows if r['primary_class'] == 'stationary' and r['target_path_m'] <= .5 and np.linalg.norm(np.asarray(r['selected'])[-1, :2]) > 1.]
            missed_depart = [r for r in rows if r['primary_class'] == 'low_depart' and r['target_path_m'] > 5 and np.linalg.norm(np.asarray(r['selected'])[-1, :2]) < 1.]
            writer.writerow({**{k: agg.get(k) for k in fields}, 'checkpoint': label, 'split': split,
                             'primary_class': cls, 'false_motion_rate': len(false_motion)/len(rows) if rows else None,
                             'missed_depart_rate': len(missed_depart)/len(rows) if rows else None})
    with (output / 'failure_cases.csv').open('x', newline='') as stream:
        fields = ['checkpoint', 'split', 'primary_class', 'sample_id', 'route', 'frame', 'speed_mps',
                  'selected_ade', 'selected_fde', 'oracle_ade', 'oracle_fde', 'selection_gap',
                  'longitudinal_mae_m', 'lateral_mae_m', 'heading_mae_deg', 'selected_mode',
                  'oracle_mode', 'score_margin']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r in sorted(all_rows, key=lambda x: x['selected_error']['ade_m'], reverse=True):
            e, o = r['selected_error'], r['oracle_error']
            writer.writerow({**{k: r.get(k) for k in fields}, 'selected_ade': e['ade_m'],
                             'selected_fde': e['fde_m'], 'oracle_ade': o['ade_m'],
                             'oracle_fde': o['fde_m'], 'longitudinal_mae_m': e['longitudinal_mae_m'],
                             'lateral_mae_m': e['lateral_mae_m'],
                             'heading_mae_deg': e['heading_mae_rad'] * 180 / np.pi})
    # Per-class selected/oracle curves.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    classes = list(CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, key in zip(axes, ('selected_ade', 'selection_gap')):
        for label in ('best', 'late'):
            vals = []
            for cls in classes:
                rows = groups.get((label, 'validation', cls), [])
                agg = aggregate(rows) if rows else {}
                vals.append(agg.get(key))
            ax.plot(range(len(classes)), vals, 'o-', label=label)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
        ax.set_title(key); ax.grid(alpha=.25); ax.legend()
    fig.savefig(output / 'behavior_selected_oracle.png', dpi=160)
    plt.close(fig)
    # Frame317 and representative turn panels.
    case_rows = [r for r in all_rows if 'SequentialLaneChange' in r['route'] and int(r['frame']) == 317]
    if case_rows:
        fig, axes = plt.subplots(1, len(case_rows), figsize=(6 * len(case_rows), 5), squeeze=False)
        for ax, r in zip(axes.flat, case_rows):
            for name, color in (('target', 'tab:green'), ('selected', 'tab:red')):
                xy = np.asarray(r[name]); ax.plot(xy[:, 1], xy[:, 0], 'o-', color=color, label=name)
            cand = np.asarray(r['candidates'][r['oracle_mode']]); ax.plot(cand[:, 1], cand[:, 0], '--', color='tab:orange', label='oracle')
            ax.scatter([0], [0], marker='s', color='black', label='t0')
            ax.set_title(f"{r['checkpoint']} step {r['global_step']} selected ADE {r['selected_error']['ade_m']:.2f} m")
            ax.set_xlabel('lateral y (m)'); ax.set_ylabel('forward x (m)')
            ax.set_aspect('auto'); ax.invert_xaxis(); ax.grid(alpha=.25); ax.legend(fontsize=7)
        fig.savefig(output / 'case_frame317.png', dpi=160); plt.close(fig)
    protocol = {'schema': 'scratch-behavior-attribution-v1', 'panel': str(Path(panel_path).resolve()),
                'panel_sha256': file_digest(panel_path), 'config': str(Path(panel['config']).resolve()),
                'config_sha256': file_digest(panel['config']), 'manifest': panel['manifest'],
                'manifest_sha256': panel['manifest_sha256'],
                'checkpoints': checkpoint_records,
                'seed': SEED, 'device': 'cpu', 'precision': 'FP32', 'targets_supplied_to_forward': False,
                'optimizer_steps': 0, 'candidate_capture': 'read-only forward hook on official diff_decoder',
                'candidate_count': 20, 'selection_rule': 'argmax of the 20 mode scores',
                'oracle_definition': 'minimum-ADE candidate computed after forward with target; upper bound only, not deployable',
                'oracle_uses_target_after_forward': True, 'samples': len(all_rows),
                'noise_policy': 'manual_seed(SEED + sample_index); same order across checkpoints',
                'panel_balance': {'validation': VAL_PER_CLASS * len(CLASSES),
                                  'train': TRAIN_PER_CLASS * len(CLASSES),
                                  'validation_classes': len(CLASSES), 'train_classes': len(CLASSES)}}
    (output / 'evaluation_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'event': 'evaluation_written', 'output': str(output.resolve()),
                      'rows': len(all_rows)}, ensure_ascii=False), flush=True)


def evaluate(panel_path, output):
    panel = load_panel(panel_path)
    config = json.loads(Path(panel['config']).read_text())
    manifest = json.loads(Path(panel['manifest']).read_text())
    from tools.training.diffusiondrive_v1.cache_only_reader import CacheOnlyPairedDataset
    from tools.training.diffusiondrive_v1.runtime import model_configuration, initialization_manifest
    from tools.training.diffusiondrive_v1.official_v1_audit import verify_pins
    from tools.training.diffusiondrive_v1.evaluate_checkpoint import stable_checkpoint
    from tools.b2d_adapter.v1_builders import official_v1_module
    import torch
    torch.set_num_threads(1)
    verify_pins(config)
    model_config = model_configuration(config)
    initialization_manifest(config, model_config)
    selected_paths = panel.get('checkpoints') or DEFAULT_CHECKPOINTS
    best_path = Path(selected_paths['best'])
    late_path = Path(selected_paths['late'])
    checkpoints = {}; checkpoint_records = {}
    for label, path in (('best', best_path), ('late', late_path)):
        checkpoints[label], checkpoint_records[label] = stable_checkpoint(path)
    datasets = {split: CacheOnlyPairedDataset(manifest, config, split) for split in ('validation', 'train')}
    sample_maps = {split: {r['sample_id']: i for i, r in enumerate(ds.rows)} for split, ds in datasets.items()}
    records = []
    for split in ('validation', 'train'):
        for rec in panel[split]:
            records.append(dict(rec, split=split, case_study=False))
    # Explicit severe case from the user's E3 analysis; it is not used to fill
    # any behavior quota and is reported separately.
    for row in datasets['validation'].rows:
        if 'SequentialLaneChange' in row['route'] and int(row['frame']) == 317:
            stats = target_stats(load_target(manifest, row))
            status = load_status(manifest, row)
            records.append({'row': row, 'sample_id': row['sample_id'], 'route': row['route'],
                            'frame': row['frame'], 'stats': stats, 'tags': ['case_study'],
                            'primary_class': 'case_study', 'speed_mps': float(np.linalg.norm(status[4:6])),
                            'status': status.tolist(), 'split': 'validation', 'case_study': True})
    agent = official_v1_module('transfuser_agent').TransfuserAgent(model_config, config['learning_rate'], checkpoint_path=None)
    agent.to('cpu').eval().requires_grad_(False)
    captured = {}
    def capture(module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError('Unexpected diff_decoder output')
        regs, classes = output
        captured['reg'] = regs[-1].detach().cpu()
        captured['cls'] = classes[-1].detach().cpu()
    handle = agent._transfuser_model._trajectory_head.diff_decoder.register_forward_hook(capture)
    all_rows = []
    try:
        for label, checkpoint in checkpoints.items():
            state = checkpoint['state_dict']
            agent.load_state_dict({k[len('agent.'):]: v for k, v in state.items()}, strict=True)
            agent.eval()
            for index, rec in enumerate(records):
                split = rec['split']; dataset = datasets[split]; row_index = sample_maps[split][rec['sample_id']]
                features, targets = dataset[row_index]
                f = {k: v.unsqueeze(0).to('cpu') for k, v in features.items()}
                torch.manual_seed(SEED + index)
                captured.clear()
                with torch.inference_mode():
                    prediction = agent.forward(f)
                if 'reg' not in captured or 'cls' not in captured:
                    raise RuntimeError('Candidate hook did not capture diff_decoder output')
                candidates = captured['reg'][0].detach().cpu().numpy()
                scores = captured['cls'][0].detach().cpu().numpy()
                metrics = candidate_metrics(prediction, targets, candidates, scores)
                speed = float(np.linalg.norm(np.asarray(features['status_feature'], dtype=np.float64).reshape(-1)[4:6]))
                row = {'checkpoint': label, 'global_step': checkpoint_records[label]['global_step'],
                       'checkpoint_sha256': checkpoint_records[label]['sha256'], 'split': split,
                       'primary_class': rec['primary_class'], 'tags': rec['tags'],
                       'sample_id': rec['sample_id'], 'route': rec['route'], 'frame': int(rec['frame']),
                       'feature_sha256': rec['row']['feature_sha'], 'target_sha256': rec['row']['target_sha'],
                       'case_study': bool(rec.get('case_study', False)),
                       'speed_mps': speed, 'target_path_m': rec['stats']['path_m'],
                       'target_first_speed_mps': rec['stats']['first_speed_mps'],
                       'target_last_speed_mps': rec['stats']['last_speed_mps'],
                       'target_heading_change_rad': rec['stats']['heading_change_rad'],
                       **metrics}
                all_rows.append(row)
                print(json.dumps({'event': 'sample', 'checkpoint': label, 'split': split,
                                  'class': rec['primary_class'], 'index': index, 'sample_id': rec['sample_id'],
                                  'selected_ade': metrics['selected_error']['ade_m'],
                                  'oracle_ade': metrics['oracle_error']['ade_m']}), flush=True)
    finally:
        handle.remove()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output / 'selected_oracle_samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_rows))
    # Per-sample selected/oracle CSV.
    with (output / 'selected_oracle_metrics.csv').open('x', newline='') as stream:
        fields = ['checkpoint', 'global_step', 'split', 'primary_class', 'sample_id', 'route', 'frame',
                  'speed_mps', 'selected_ade', 'selected_fde', 'oracle_ade', 'oracle_fde', 'selection_gap',
                  'selected_mode', 'oracle_mode', 'score_margin', 'longitudinal_mae_m', 'lateral_mae_m',
                  'heading_mae_deg', 'detection_f1', 'bev_miou']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r in all_rows:
            e, o = r['selected_error'], r['oracle_error']
            writer.writerow({**{k: r.get(k) for k in fields},
                             'selected_ade': e['ade_m'], 'selected_fde': e['fde_m'],
                             'oracle_ade': o['ade_m'], 'oracle_fde': o['fde_m'],
                             'longitudinal_mae_m': e['longitudinal_mae_m'],
                             'lateral_mae_m': e['lateral_mae_m'],
                             'heading_mae_deg': e['heading_mae_rad'] * 180 / np.pi,
                             'detection_f1': r['detection'].get('f1'),
                             'bev_miou': r['bev_miou']})
    write_reports(output, all_rows, panel, panel_path, checkpoint_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('select'); p.add_argument('--config', type=Path, default=CONFIG); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('evaluate'); p.add_argument('--panel', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('report'); p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    if args.action == 'select':
        select(args.config, args.output)
    elif args.action == 'evaluate':
        evaluate(args.panel, args.output)
    else:
        from tools.training.diffusiondrive_v1.evaluate_checkpoint import stable_checkpoint
        panel_data = load_panel(args.panel)
        rows = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
        records = {}
        for label, path in (panel_data.get('checkpoints') or DEFAULT_CHECKPOINTS).items():
            _, records[label] = stable_checkpoint(Path(path))
        args.output.mkdir(parents=True, exist_ok=False)
        write_reports(args.output, rows, panel_data, args.panel, records)


if __name__ == '__main__':
    main()
