import csv
from pathlib import Path

from tools.training.diffusiondrive_v1 import status_training_curves as curves


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['epoch', 'step', 'train/loss_step', 'train/trajectory_loss_step',
              'train/agent_class_loss_step', 'train/agent_box_loss_step',
              'train/bev_semantic_loss_step', 'train/diffusion_loss_step',
              'train/trajectory_loss_0_step', 'train/trajectory_loss_1_step',
              'val/loss', 'val/trajectory_loss', 'val/ade_m', 'val/fde_m',
              'val/ade_left_m', 'val/ade_straight_m', 'val/ade_right_m',
              'val/fde_left_m', 'val/fde_straight_m', 'val/fde_right_m']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fields})


def test_curves_merge_versions_and_write_png(tmp_path):
    run = tmp_path / 'run'
    train = {'train/loss_step': 80, 'train/trajectory_loss_step': 70,
             'train/agent_class_loss_step': .5, 'train/agent_box_loss_step': 4,
             'train/bev_semantic_loss_step': 3, 'train/diffusion_loss_step': 0,
             'train/trajectory_loss_0_step': 2.5, 'train/trajectory_loss_1_step': 2.2}
    first = [dict(train, epoch=0, step=10 * i) for i in range(1, 6)]
    first += [dict(train, epoch=0, step=50, **{
        'val/loss': 27, 'val/trajectory_loss': 11, 'val/ade_m': 2.4, 'val/fde_m': 5.2,
        'val/ade_left_m': 3.4, 'val/ade_straight_m': 2.1, 'val/ade_right_m': 2.3,
        'val/fde_left_m': 7.3, 'val/fde_straight_m': 4.8, 'val/fde_right_m': 4.9})]
    second = [dict(train, epoch=0, step=50 + 10 * i) for i in range(1, 6)]
    second += [dict(train, epoch=1, step=100, **{
        'val/loss': 25, 'val/trajectory_loss': 9, 'val/ade_m': 2.0, 'val/fde_m': 4.4,
        'val/ade_left_m': 3.2, 'val/ade_straight_m': 1.6, 'val/ade_right_m': 2.2,
        'val/fde_left_m': 6.9, 'val/fde_straight_m': 3.6, 'val/fde_right_m': 4.8})]
    write_metrics(run / 'logs/lightning_logs/version_0/metrics.csv', first)
    write_metrics(run / 'logs/lightning_logs/version_1/metrics.csv', second)

    rows = curves.load_rows(run)
    assert len([row for row in rows if row.get('val/ade_m')]) == 2
    curves.plot_training(rows, tmp_path, window=3)
    latest = curves.plot_validation(rows, tmp_path)
    assert latest == 100
    assert (tmp_path / 'train_losses.png').stat().st_size > 0
    assert (tmp_path / 'validation_curves.png').stat().st_size > 0
