"""Active-run throughput and ETA, with validation time reported separately."""
from collections import deque
import math
import time

from tools.training.diffusiondrive_v1.features_cache import durable_json


class TrainingProgress:
    def __init__(self, output, horizon, train_samples, batch_size, config):
        self.output = output
        self.horizon = horizon
        self.accumulation = config['training'].get('accumulate_grad_batches', 1)
        self.micro_batches_per_epoch = math.ceil(train_samples / batch_size)
        self.epoch_steps = math.ceil(self.micro_batches_per_epoch / self.accumulation)
        self.batch_size = batch_size
        self.consumed_samples = 0
        self.consumed_micro_batches = 0
        self.validation_interval = config['training'].get('validation_every_steps')
        if not self.validation_interval:
            self.validation_interval = self.epoch_steps * config['training']['validation_every_epochs']
        self.val_durations = []
        self.val_count = 0
        self.points = deque()
        self.started = None
        self.last_write = 0
        self.val_started = None
        self.val_seconds = 0
        self.val_batches = 0

    def start(self, step, restored=None):
        self.started = time.monotonic()
        self.val_count = step // self.validation_interval
        if restored:
            self.val_count = restored['validation_count']
            self.val_durations = restored['validation_seconds']
            self.consumed_samples = restored.get('consumed_samples', step * self.batch_size * self.accumulation)
            self.consumed_micro_batches = restored.get('consumed_micro_batches', step * self.accumulation)
        self.points.append((step, 0.0, self.consumed_samples))

    def commit(self, samples, micro_batches):
        self.consumed_samples += samples
        self.consumed_micro_batches += micro_batches

    def state_dict(self):
        return {'validation_count': self.val_count, 'validation_seconds': self.val_durations[-10:],
                'consumed_samples': self.consumed_samples, 'consumed_micro_batches': self.consumed_micro_batches}

    def validation_start(self, batches):
        self.val_started = time.monotonic()
        self.val_batches = batches

    def validation_end(self, replay=False):
        duration = time.monotonic() - self.val_started
        self.val_seconds += duration
        if not replay:
            self.val_durations.append(duration)
            self.val_count += 1
        self.val_started = None

    def update(self, step, epoch, phase='training', force=False, validation_batch=0):
        now = time.monotonic()
        if not force and now - self.last_write < 5:
            return
        active_val = now - self.val_started if self.val_started is not None else 0
        active_train = now - self.started - self.val_seconds - active_val
        if self.points[-1][0] != step:
            self.points.append((step, active_train, self.consumed_samples))
        while len(self.points) > 2 and self.points[1][1] < active_train - 300:
            self.points.popleft()
        n = step - self.points[0][0]
        elapsed = active_train - self.points[0][1]
        rate = n / elapsed if n > 0 and elapsed > 0 else None
        sample_rate = (self.consumed_samples-self.points[0][2])/elapsed if n > 0 and elapsed > 0 else None
        remaining = max(0, self.horizon - step)
        train_eta = remaining / rate if rate else None
        future_vals = max(0, self.horizon // self.validation_interval - self.val_count)
        val_mean = sum(self.val_durations[-3:]) / len(self.val_durations[-3:]) if self.val_durations else None
        val_remaining = future_vals * val_mean if val_mean is not None else None
        if active_val and validation_batch >= 10:
            projected_full = active_val * self.val_batches / validation_batch
            val_remaining = max(0, future_vals - 1) * (val_mean or projected_full) + max(0, projected_full - active_val)
        total_eta = train_eta + val_remaining if train_eta is not None and val_remaining is not None else None
        if future_vals == 0:
            total_eta = train_eta
        result = dict(state=phase, global_step=step, epoch=epoch, max_steps=self.horizon,
                      epoch_steps=self.epoch_steps, micro_batches_per_epoch=self.micro_batches_per_epoch,
                      accumulate_grad_batches=self.accumulation, effective_batch_size=self.batch_size*self.accumulation,
                      consumed_samples=self.consumed_samples, consumed_micro_batches=self.consumed_micro_batches,
                      samples_per_second=sample_rate,
                      steps_per_second=rate, train_only_remaining_seconds=train_eta,
                      estimated_remaining_seconds=total_eta, remaining_validation_runs=future_vals,
                      observed_validation_seconds=val_mean, cooperative_pause_supported=True,
                      updated_at=time.time(), run_active_seconds=now-self.started,
                      eta_scope='Active runtime; excludes pauses. Until validation is measured only training ETA is available.',
                      checkpoint=None)
        durable_json(self.output / 'training_progress.json', result)
        self.last_write = now
        return result


def normalize_committed_progress(checkpoint):
    """Align Lightning batch completion with an already committed optimizer step.

    Periodic ModelCheckpoint runs in on_train_batch_end, before Lightning's
    completed counter increments. Our sampler has already acknowledged that batch.
    Modify only the loaded loop bookkeeping, never the stored checkpoint/tensors.
    """
    loop = checkpoint['loops']['fit_loop']
    progress = loop['epoch_loop.batch_progress']
    optimizer = loop['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']
    accumulation = checkpoint.get('b2d_v1_fit', {}).get('semantic', {}).get('accumulation')
    committed = checkpoint.get('b2d_v1_fit', {}).get('accumulation_state', {})
    if accumulation and (committed.get('pending_micro_batches') != 0
                         or committed.get('last_committed_optimizer_step') != checkpoint['global_step']):
        raise RuntimeError('Cannot restore an incomplete accumulation window')
    if optimizer['total']['completed'] != checkpoint['global_step']:
        raise RuntimeError('Optimizer step counter differs from checkpoint global_step')
    normalized = False
    for scope in ('total', 'current'):
        p = progress[scope]
        if p['completed'] == p['processed'] - 1:
            valid_update = optimizer[scope]['completed'] == p['processed'] if not accumulation else (
                committed.get('pending_micro_batches') == 0
                and committed.get('last_committed_optimizer_step') == checkpoint['global_step'])
            if p['ready'] != p['processed'] or p['started'] != p['processed'] or not valid_update:
                raise RuntimeError('Uncertain checkpoint batch commit phase')
            p['completed'] = p['processed']
            normalized = True
        elif p['completed'] != p['processed']:
            raise RuntimeError('Unsupported checkpoint batch progress')
    return normalized
