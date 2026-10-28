import csv
import json
from pathlib import Path

from tools.training.diffusiondrive_v1 import status_training_monitor as monitor


def build_run(tmp_path, state='running', step=1600):
    out = tmp_path / 'run'
    (out / 'logs/lightning_logs/version_0').mkdir(parents=True)
    (out / 'checkpoints').mkdir()
    (out / 'training_progress.json').write_text(json.dumps({
        'state': state, 'epoch': 3, 'global_step': step,
        'max_steps': 10000, 'schema': 'status-experimental-progress-v1'}))
    (out / 'train.log').write_text('Epoch 3:  10%|#  | 100/1000 [00:05<00:45, 18.00it/s, '
                                   'train/loss_step=41.25]\n')
    with (out / 'logs/lightning_logs/version_0/metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'val/samples', 'val/ade_m',
                                                    'val/fde_m', 'val/heading_error_deg'])
        writer.writeheader()
        writer.writerow({'step': 1599, 'val/samples': 1024, 'val/ade_m': 2.05,
                         'val/fde_m': 4.2, 'val/heading_error_deg': 8.1})
    (out / 'checkpoints/checkpoint_index.json').write_text(json.dumps({
        'best': [{'step': 1500, 'score': 1.86, 'epoch': 3, 'path': 'x'}]}))
    return out


def test_parse_train_line_reads_last_loss():
    text = 'a\rEpoch 1\rEpoch 2: 5%|#| 5/100 [00:01<00:19, 12.00it/s, train/loss_step=33.5]\n'
    assert monitor.parse_train_line(text) == 33.5
    assert monitor.parse_train_line('') is None


def test_format_line_reports_state_loss_validation_and_best(tmp_path, monkeypatch):
    out = build_run(tmp_path)
    monkeypatch.setattr(monitor, 'process_count', lambda tag: 1)
    line = monitor.format_line(out, [], 'tag')
    assert 'state=running' in line and 'step=1,600/10,000 (16.00%)' in line
    assert 'loss=41.2' in line and 'pid=alive' in line
    assert 'val@1,599 ADE=2.05 FDE=4.20 head=8.1' in line
    assert 'best=1,500 (1.860)' in line


def test_format_line_warns_when_state_running_without_process(tmp_path, monkeypatch):
    out = build_run(tmp_path)
    monkeypatch.setattr(monitor, 'process_count', lambda tag: 0)
    line = monitor.format_line(out, [], 'tag')
    assert 'pid=none' in line and 'WARNING=state-running-without-process' in line


def test_rate_uses_history_span(tmp_path, monkeypatch):
    out = build_run(tmp_path, step=1600)
    monkeypatch.setattr(monitor, 'process_count', lambda tag: 1)
    now = monitor.time.time()
    history = [(now - 200.0, 1000)]
    line = monitor.format_line(out, history, 'tag', window=300.0, min_span=120.0)
    assert 'rate=3.00 step/s' in line and 'eta=' in line


def test_log_tail_missing_file(tmp_path):
    assert monitor.log_tail(tmp_path / 'missing.log') == ''
