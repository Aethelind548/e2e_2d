"""Plot train/validation curves for ``status_experimental_train`` runs.

Reads every ``logs/lightning_logs/version_*/metrics.csv`` under one run directory,
merges versions (pause/resume creates new version folders) and writes PNGs to
``<run>/curves`` by default. Read-only: never touches checkpoints or metrics.
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np


TRAIN_PANELS = [
    ('train/loss_step', 'train/trajectory_loss_step', 'Total / trajectory loss'),
    ('train/agent_class_loss_step', 'train/agent_box_loss_step', 'Agent class / box loss'),
    ('train/bev_semantic_loss_step', 'train/diffusion_loss_step', 'BEV semantic / diffusion loss'),
    ('train/trajectory_loss_0_step', 'train/trajectory_loss_1_step', 'Trajectory loss 0 / 1'),
]


def load_rows(run):
    rows = []
    for path in sorted(glob.glob(f'{run}/logs/lightning_logs/version_*/metrics.csv')):
        with open(path) as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def series(rows, key, x_key='step'):
    xs, ys = [], []
    for row in rows:
        value = row.get(key)
        if value in (None, '', 'nan'):
            continue
        xs.append(int(row[x_key]))
        ys.append(float(value))
    if not xs:
        return np.array([]), np.array([])
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


def smooth(y, window):
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def plot_training(rows, save_dir, window):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (first, second, title) in zip(axes.flat, TRAIN_PANELS):
        for key in (first, second):
            xs, ys = series(rows, key)
            if not len(xs):
                continue
            label = key.replace('train/', '').replace('_step', '')
            ax.plot(xs, ys, color='0.75', linewidth=0.6)
            if len(ys) >= window:
                ax.plot(xs[window - 1:], smooth(ys, window), linewidth=1.6,
                        label=f'{label} (mean {window})')
            else:
                ax.plot(xs, ys, linewidth=1.4, label=label)
        ax.set_title(title)
        ax.set_xlabel('Optimizer step')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.savefig(save_dir / 'train_losses.png', dpi=160)
    plt.close(fig)


def plot_validation(rows, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row.get('val/ade_m')
             and (not row.get('val/samples') or float(row['val/samples']) >= 512)
             and row.get('val/ade_left_m') and row.get('val/ade_straight_m')
             and row.get('val/ade_right_m')]
    valid.sort(key=lambda row: int(row['step']))
    if not valid:
        return None
    steps = np.array([int(row['step']) for row in valid])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, keys, title in (
            (axes[0, 0], ('val/loss', 'val/trajectory_loss'), 'Validation loss'),
            (axes[0, 1], ('val/ade_m', 'val/fde_m'), 'Validation ADE / FDE'),
            (axes[1, 0], ('val/ade_left_m', 'val/ade_straight_m', 'val/ade_right_m'),
             'Validation ADE by command (Left / Straight / Right)'),
            (axes[1, 1], ('val/fde_left_m', 'val/fde_straight_m', 'val/fde_right_m'),
             'Validation FDE by command (Left / Straight / Right)')):
        for key in keys:
            xs, ys = series(valid, key)
            if len(xs):
                ax.plot(xs, ys, 'o-', markersize=4, label=key.replace('val/', ''))
        ax.set_title(title)
        ax.set_xlabel('Optimizer step')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    # annotate the latest point so early sparse curves are readable
    for ax in axes.flat:
        for line in ax.get_lines():
            xs, ys = line.get_xdata(), line.get_ydata()
            if len(xs):
                ax.annotate(f'{ys[-1]:.2f}', (xs[-1], ys[-1]),
                            textcoords='offset points', xytext=(4, 4), fontsize=7)
    fig.savefig(save_dir / 'validation_curves.png', dpi=160)
    plt.close(fig)
    return steps[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--save-dir', type=Path, default=None)
    parser.add_argument('--smooth', type=int, default=50)
    args = parser.parse_args()

    run = args.output_dir.resolve()
    rows = load_rows(run)
    if not rows:
        raise SystemExit(f'no metrics.csv under {run}/logs/lightning_logs')
    save_dir = (args.save_dir or run / 'curves').resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_training(rows, save_dir, max(1, args.smooth))
    latest = plot_validation(rows, save_dir)
    valid = [row for row in rows if row.get('val/ade_m')]
    print({'run': str(run), 'rows': len(rows), 'validation_points': len(valid),
           'latest_validation_step': latest, 'save_dir': str(save_dir)})
    for path in sorted(save_dir.glob('*.png')):
        print(path)


if __name__ == '__main__':
    main()
