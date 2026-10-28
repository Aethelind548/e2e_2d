"""One-line training monitor for status_experimental_train runs.

Reads `training_progress.json`, the tail of `train.log`, the newest
`metrics.csv` validation row and `checkpoint_index.json`, then prints a single
status line. `--watch N` repeats every N seconds and adds throughput/ETA.
No writes, no GPU use; safe to run while training is active.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_OUTPUT = ('/run/media/aethelind/PSSD/diffusiondrive_v1_experiments/'
                  'status_preview_full_bf16_bs16_20261023')
DEFAULT_TAG = 'status_experimental_train'
LOG_TAIL_BYTES = 20000


def log_tail(path, limit=LOG_TAIL_BYTES):
    if not path.exists():
        return ''
    with path.open('rb') as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode('utf-8', 'ignore').replace('\r', '\n')


def parse_train_line(text):
    line = next((item for item in reversed(text.splitlines()) if 'it/s' in item), '')
    loss = re.search(r'train/loss_step=([0-9.]+)', line)
    return float(loss.group(1)) if loss else None


def _validation_rows(output):
    for path in reversed(sorted(glob.glob(str(output / 'logs/lightning_logs/version_*/metrics.csv')))):
        with open(path) as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row.get('val/ade_m') and row.get('val/samples')
                    and float(row['val/samples']) >= 1000]
        if rows:
            return rows
    return []


def last_validation(output):
    rows = _validation_rows(output)
    if not rows:
        return None
    row = rows[-1]
    result = {'step': int(float(row['step'])),
              'ade': float(row['val/ade_m']),
              'fde': float(row['val/fde_m']),
              'heading': float(row['val/heading_error_deg'])}
    if row.get('val/loss'):
        result['loss'] = float(row['val/loss'])
    return result


def best_checkpoint(output):
    path = output / 'checkpoints/checkpoint_index.json'
    if path.exists():
        best = json.loads(path.read_text()).get('best') or []
        if best:
            return best[0]
    rows = _validation_rows(output)
    scored = [row for row in rows if row.get('val/loss')]
    if not scored:
        return None
    row = min(scored, key=lambda item: float(item['val/loss']))
    step = int(float(row['step']))
    epoch = int(float(row.get('epoch') or 0))
    candidate = output / 'checkpoints' / f'best-epoch{epoch:03d}-step{step:06d}.ckpt'
    return {'step': step, 'epoch': epoch, 'score': float(row['val/loss']),
            'path': str(candidate) if candidate.is_file() else None}


def process_count(tag):
    result = subprocess.run(['pgrep', '-f', tag], stdout=subprocess.PIPE, text=True)
    return len([line for line in result.stdout.splitlines() if line.strip()])


def format_line(output, history, tag, window=300.0, samples_per_step=16, min_span=120.0):
    progress = json.loads((output / 'training_progress.json').read_text())
    loss = parse_train_line(log_tail(output / 'train.log'))
    validation = last_validation(output)
    best = best_checkpoint(output)
    running = process_count(tag)
    now = time.time()
    history.append((now, progress['global_step']))
    while len(history) > 2 and now - history[0][0] > window:
        history.pop(0)
    rate = None
    if len(history) > 1 and now - history[0][0] >= min_span:
        delta = history[-1][1] - history[0][1]
        if delta > 0:
            rate = delta / (now - history[0][0])
    step, total = progress['global_step'], progress['max_steps']
    parts = [time.strftime('%H:%M:%S'),
             'state=%s' % progress['state'],
             'epoch=%s' % progress['epoch'],
             'step=%s/%s (%.2f%%)' % (f'{step:,}', f'{total:,}', 100.0 * step / total),
             'loss=%s' % ('-' if loss is None else f'{loss:.1f}')]
    if rate:
        parts.append('rate=%.2f step/s (%.1f samples/s)' % (rate, rate * samples_per_step))
        parts.append('eta=%.1f h' % ((total - step) / rate / 3600.0))
    parts.append('pid=%s' % ('alive' if running else 'none'))
    if progress['state'] == 'running' and not running:
        parts.append('WARNING=state-running-without-process')
    if validation:
        val_loss = ' loss=%.2f' % validation['loss'] if 'loss' in validation else ''
        parts.append('val@%s%s ADE=%.2f FDE=%.2f head=%.1f' % (
            f"{validation['step']:,}", val_loss, validation['ade'],
            validation['fde'], validation['heading']))
    if best:
        parts.append('best=%s (%.3f)' % (f"{best['step']:,}", best['score']))
    return ' '.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(os.environ.get('DD_BS16_OUT', DEFAULT_OUTPUT)))
    parser.add_argument('--watch', type=float, default=0.0,
                        help='seconds between reports; 0 prints once')
    parser.add_argument('--window', type=float, default=300.0,
                        help='seconds of history used for the measured rate')
    parser.add_argument('--min-span', type=float, default=120.0,
                        help='minimum history span before a rate is printed')
    parser.add_argument('--tag', default=os.environ.get('DD_RUN_TAG', DEFAULT_TAG),
                        help='pgrep pattern used to detect the trainer process')
    args = parser.parse_args()

    if not (args.output_dir / 'training_progress.json').exists():
        raise SystemExit('missing training_progress.json under %s' % args.output_dir)
    history = []
    while True:
        print(format_line(args.output_dir, history, args.tag, args.window,
                          min_span=args.min_span), flush=True)
        if args.watch <= 0:
            return
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
