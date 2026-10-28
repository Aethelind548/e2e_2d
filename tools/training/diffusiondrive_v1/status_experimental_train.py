"""Experimental trainer: official V1 agent on the versioned Status index.

Scope is intentionally narrow and explicit:

* dataset = ``ExperimentalStatusDataset`` (research index, no certification);
* model = official ``TransfuserAgent`` built from the frozen anchor/backbone;
  initialization is the official constructor with the official timm ImageNet
  backbone, never an old Status checkpoint;
* training = the official ``AgentLightningModule`` step/loss, unmodified;
* validation = full validation set every epoch; ADE/FDE are diagnostics only;
* no changes to network structure, loss weights or trajectory targets.

It refuses to run unless the config declares ``experiment_scope`` and pins the
dataset manifest/index SHA.  This entry is prepared for a later explicit start;
it is not launched by the preparation round.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'status-experimental-train-v6'
REQUIRED_SCOPE = 'OFFLINE_ORACLE_ROUTE_IL'
PAPER_VALIDATION = 'paper-v1'
PAPER_SEED_POLICY = 'epoch*100003+batch_idx'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


PARAM_GROUP_PREFIXES = (
    ('backbone', '_backbone.'),
    ('bev_head', '_bev_semantic_head.'),
    ('agent_head', '_agent_head.'),
    ('trajectory_head', '_trajectory_head.'),
)

# Official ``TransfuserConfig`` loss weights, kept here so the run identity can
# record the values a resume actually used instead of only the override.
LOSS_WEIGHT_NAMES = ('trajectory_weight', 'trajectory_cls_weight',
                     'trajectory_reg_weight', 'diff_loss_weight',
                     'agent_class_weight', 'agent_box_weight',
                     'bev_semantic_weight')


def paramwise_lr_scales(model, paramwise_cfg):
    """Map official ``opt_paramwise_cfg`` name patterns to per-parameter lr multipliers.

    The official agent writes ``lr_mult`` into the extra parameter groups but never
    records ``lr_scale``, so ``WarmupCosLR`` (which only honours ``lr_scale``) flattens
    every group to the base lr.  This helper recomputes the intended multiplier per
    parameter name without touching the official sources.
    """
    patterns = []
    for pattern, group_cfg in (paramwise_cfg or {}).get('name', {}).items():
        patterns.append((pattern, float(group_cfg.get('lr_mult', 1.0))))
    scales = {}
    for name, _ in model.named_parameters():
        scale = 1.0
        for pattern, multiplier in patterns:
            if pattern in name:
                scale = multiplier
                break
        scales[name] = scale
    return scales


def apply_paramwise_lr_scale(optimizer, model, paramwise_cfg):
    """Write ``lr_scale`` into every optimizer group and return the applied scales.

    ``WarmupCosLR.get_lr`` only checks ``optimizer.param_groups[0]`` for ``lr_scale``,
    so the key must exist on the first group; each group must also be homogeneous,
    otherwise one group would mix backbone and head learning rates.
    """
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    scales = paramwise_lr_scales(model, paramwise_cfg)
    applied = []
    for group in optimizer.param_groups:
        values = {scales.get(names.get(id(parameter)), 1.0)
                  for parameter in group['params']}
        if len(values) != 1:
            raise RuntimeError(f'parameter group mixes lr_mult values: {sorted(values)}')
        group['lr_scale'] = values.pop()
        applied.append(group['lr_scale'])
    return applied


def install_paramwise_lr_scale(agent, epochs=100, warmup_epochs=3, min_lr=1e-6):
    """Wrap the official optimizer builder so the scheduler keeps ``lr_mult``.

    The official ``get_coslr_optimizers`` creates ``WarmupCosLR`` itself; that
    scheduler only scales per group when ``lr_scale`` exists, which the official
    builder never sets.  The wrapper sets ``lr_scale`` on the freshly built
    optimizer and rebuilds the same official scheduler, so the official files stay
    byte-identical.
    """
    from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR

    original = agent.get_coslr_optimizers

    def get_coslr_optimizers_with_lr_scale():
        result = original()
        optimizer = result['optimizer']
        applied = apply_paramwise_lr_scale(
            optimizer, agent._transfuser_model, agent._config.opt_paramwise_cfg)
        scheduler = WarmupCosLR(optimizer=optimizer, lr=agent._lr, min_lr=min_lr,
                                epochs=epochs, warmup_epochs=warmup_epochs)
        target = result.get('lr_scheduler')
        if isinstance(target, dict):
            target['scheduler'] = scheduler
        else:
            result['lr_scheduler'] = scheduler
        result['paramwise_lr_scale'] = applied
        return result

    agent.get_coslr_optimizers = get_coslr_optimizers_with_lr_scale
    return agent


def grouped_gradient_norms(model):
    """Unclipped gradient L2 norm per branch, used to catch a silent loss jump."""
    squared = {name: 0.0 for name, _ in PARAM_GROUP_PREFIXES}
    squared['other'] = 0.0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        bucket = 'other'
        for key, prefix in PARAM_GROUP_PREFIXES:
            if prefix in name:
                bucket = key
                break
        squared[bucket] += float(gradient.detach().float().pow(2).sum())
    return {name: math.sqrt(value) for name, value in squared.items()}


class ModelEMA:
    """Exponential moving average of every trainable parameter.

    Validation runs against these smoothed weights, so the monitored objective is
    less sensitive to the single-epoch swings that the raw weights showed in the
    previous runs.  Training itself always keeps the raw weights, therefore the
    optimizer state stays consistent across resumes.
    """

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {name: parameter.detach().clone()
                       for name, parameter in model.named_parameters()
                       if parameter.requires_grad}

    def _effective_decay(self):
        # Warm up so the first steps do not lock onto the random init.
        if self.num_updates < 100:
            return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        return self.decay

    def update(self, model):
        import torch
        if not self.shadow:
            return
        decay = self._effective_decay()
        self.num_updates += 1
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)

    def copy_to(self, model):
        import torch
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    parameter.data.copy_(self.shadow[name])

    def state_dict(self):
        return {'decay': self.decay, 'num_updates': self.num_updates,
                'shadow': {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state):
        if not state:
            return
        self.decay = float(state.get('decay', self.decay))
        self.num_updates = int(state.get('num_updates', 0))
        shadow = state.get('shadow') or {}
        for name, value in shadow.items():
            if name in self.shadow:
                # Checkpoint payloads arrive on whatever device the loader used
                # (typically CPU), while the live shadow tracks the model on its
                # compute device.  Casting here keeps ``update`` from raising a
                # cross-device error on the first step after a resume.
                target = self.shadow[name]
                self.shadow[name] = value.detach().to(device=target.device,
                                                      dtype=target.dtype)


class EMACallback:
    """Lightning callback wrapper; kept thin so ``ModelEMA`` stays testable.

    Ordering contract: EMA weights are swapped in at validation start, the smoothed
    copy is written as ``ema-<epoch>-<step>.ckpt``, and the raw weights are restored
    inside ``on_validation_end``.  Because this callback is registered *before*
    ``ModelCheckpoint``, the checkpoint saves the raw weights that match the saved
    optimizer state, while the monitored score comes from the EMA weights.
    """

    def __init__(self, decay=0.999, directory=None):
        self.decay = float(decay)
        self.directory = Path(directory) if directory is not None else None
        self.ema = None
        self.backup = None
        self.last_step = -1
        self.saved = []
        self.pending_ema_state = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == self.last_step:
            return                      # gradient accumulation window, no step yet
        self.last_step = step
        model = pl_module.agent._transfuser_model if hasattr(pl_module, 'agent') else pl_module
        if self.ema is None:
            self.ema = ModelEMA(model, self.decay)
            # ``self.ema`` only exists once the first optimizer step reveals the
            # model, so a payload restored before that point is applied here.
            # Dropping it would silently restart the average from the raw weights
            # after every resume.
            if self.pending_ema_state is not None:
                self.ema.load_state_dict(self.pending_ema_state)
                self.ema.decay = self.decay
                self.pending_ema_state = None
            return
        self.ema.update(model)

    def on_validation_start(self, trainer, pl_module):
        if self.ema is None or not self.ema.shadow:
            return
        model = pl_module.agent._transfuser_model if hasattr(pl_module, 'agent') else pl_module
        self.backup = {name: parameter.detach().clone()
                       for name, parameter in model.named_parameters()}
        self.ema.copy_to(model)

    def on_validation_end(self, trainer, pl_module):
        """Persist the smoothed weights, then restore the raw ones.

        The restore must happen here (not at the next epoch start) so that
        ``ModelCheckpoint`` — registered after this callback — writes the raw
        weights that match the saved optimizer state.
        """
        import torch
        if self.ema is None or self.backup is None:
            return
        if self.directory is not None:
            path = self.directory / f'ema-{int(trainer.current_epoch):03d}-{int(trainer.global_step):06d}.ckpt'
            if not path.exists():
                torch.save({'schema': 'status-ema-weights-v1',
                            'epoch': int(trainer.current_epoch),
                            'global_step': int(trainer.global_step),
                            'decay': float(self.ema.decay),
                            'num_updates': int(self.ema.num_updates),
                            'state_dict': {k: v.clone() for k, v in self.ema.shadow.items()}},
                           path)
                self.saved.append(str(path))
        model = self._model(pl_module)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.backup:
                    parameter.data.copy_(self.backup[name])
        self.backup = None

    @staticmethod
    def _model(pl_module):
        return pl_module.agent._transfuser_model if hasattr(pl_module, 'agent') else pl_module

    def state_dict(self):
        return {'ema': self.ema.state_dict() if self.ema is not None else None,
                'last_step': self.last_step, 'decay': self.decay}

    def load_state_dict(self, state):
        if not state:
            return
        self.last_step = int(state.get('last_step', -1))
        payload = state.get('ema')
        if not payload:
            return
        if self.ema is not None:
            self.ema.load_state_dict(payload)
            # The decay horizon belongs to the run's configuration, not to the
            # checkpoint: a resumed run may deliberately widen or narrow the
            # averaging window, and the old value would silently win.
            self.ema.decay = self.decay
        else:
            self.pending_ema_state = payload


def make_ema_lightning_callback(decay=0.999, directory=None):
    """Wrap the testable :class:`EMACallback` logic in a real ``pl.Callback``.

    Lightning requires an actual ``pl.Callback`` subclass (it inspects the parent
    to decide whether a callback is stateful), while this module keeps
    ``pytorch_lightning`` out of the import path so non-training commands stay fast.
    """
    import pytorch_lightning as pl

    class _LightningEMA(pl.Callback):
        def __init__(self):
            super().__init__()
            self.impl = EMACallback(decay=decay, directory=directory)

        @property
        def saved(self):
            return self.impl.saved

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            return self.impl.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        def on_validation_start(self, trainer, pl_module):
            return self.impl.on_validation_start(trainer, pl_module)

        def on_validation_end(self, trainer, pl_module):
            return self.impl.on_validation_end(trainer, pl_module)

        def state_dict(self):
            return self.impl.state_dict()

        def load_state_dict(self, state):
            return self.impl.load_state_dict(state)

    return _LightningEMA()


def request_graceful_stop(trainer):
    """End the run after the current batch.

    Lightning 2.x removed the writable ``Trainer.should_stop`` that older
    recipes used: on 2.2.1 assigning it only creates an inert instance
    attribute, so ``pause.request`` saved its checkpoint and then kept
    training.  Capping the epoch loop's ``max_steps`` at the current step is
    what actually makes ``_FitLoop.done`` true.  Returns True when the cap was
    applied.
    """
    try:
        trainer.should_stop = True          # harmless on versions that honour it
    except Exception:                       # noqa: BLE001 - never mask the pause
        pass
    epoch_loop = getattr(getattr(trainer, 'fit_loop', None), 'epoch_loop', None)
    if epoch_loop is None:
        return False
    epoch_loop.max_steps = int(trainer.global_step)
    return True


def live_learning_rates(trainer):
    """Current per-group lr from the live optimizer.

    ``training_progress.json`` used to echo a snapshot taken in ``on_train_start``,
    so every later epoch reported epoch-0 values.  Reading the optimizer directly
    keeps the progress file truthful.
    """
    optimizer = trainer.optimizers[0] if getattr(trainer, 'optimizers', None) else None
    if optimizer is None:
        return None
    return [float(group['lr']) for group in optimizer.param_groups]


def audit_lr_scale(optimizer, agent_model, paramwise_cfg, scheduler=None,
                   scheduler_epochs=None):
    """Re-apply ``lr_scale`` after a resume rewrote ``param_groups`` from disk.

    ``torch.optim.Optimizer.load_state_dict`` replaces the groups wholesale, so a
    checkpoint written before the fix restores groups without ``lr_scale`` and the
    official scheduler would flatten the backbone again.  Returns the audited
    payload so the caller can persist it.

    ``scheduler_epochs`` exists because ``WarmupCosLR.state_dict`` stores the
    horizon and ``load_state_dict`` restores it verbatim: a longer run resumed
    from a checkpoint always comes back with the official ``epochs=100``, which
    makes the cosine decay almost flat.  The planned horizon is re-injected here
    before the schedule is recomputed.
    """
    applied = apply_paramwise_lr_scale(optimizer, agent_model, paramwise_cfg)
    injected = None
    if scheduler is not None:
        current = int(getattr(scheduler, 'epochs', -1))
        if scheduler_epochs is not None and current != int(scheduler_epochs):
            injected = {'from': current, 'to': int(scheduler_epochs)}
            scheduler.epochs = int(scheduler_epochs)
        base = float(getattr(scheduler, 'lr', 0.0) or 0.0)
        if base > 0:
            scheduler.base_lrs = [base * value for value in applied]
        values = [float(value) for value in scheduler.get_lr()]
        for group, value in zip(optimizer.param_groups, values):
            group['lr'] = value
    payload = {'param_group_lr_scale': [float(value) for value in applied],
               'learning_rates': [float(group['lr']) for group in optimizer.param_groups],
               'scheduler_base_lrs': [float(value) for value in
                                      getattr(scheduler, 'base_lrs', [])],
               'scheduler_base_lr': float(getattr(scheduler, 'lr', 0.0) or 0.0),
               'scheduler_last_epoch': int(getattr(scheduler, 'last_epoch', -1)),
               'scheduler_epochs': int(getattr(scheduler, 'epochs', -1))}
    if injected is not None:
        payload['scheduler_epochs_injected'] = injected
    return payload


def expected_lr_plan(base_lr, paramwise_cfg, warmup_epochs=3):
    """Analytic warmup/peak learning rate per branch, recorded before training."""
    plan = {}
    for pattern, group_cfg in (paramwise_cfg or {}).get('name', {}).items():
        scale = float(group_cfg.get('lr_mult', 1.0))
        plan[pattern] = {'lr_scale': scale,
                         'first_step_lr': base_lr * scale / warmup_epochs,
                         'peak_lr': base_lr * scale}
    plan['default'] = {'lr_scale': 1.0,
                       'first_step_lr': base_lr / warmup_epochs,
                       'peak_lr': base_lr}
    return plan


def optimizer_contract(agent, paramwise_cfg, warmup_epochs=3, lr_scale_enabled=True):
    """Build the real optimizer once and prove every group keeps its ``lr_mult``.

    Raises when a declared multiplier is missing, which is exactly the silent
    regression that let image_encoder train at the full ``base_lr``.  The
    diagnostic arm with ``lr_scale_enabled=False`` asserts the opposite (flat lr).
    """
    result = agent.get_coslr_optimizers()
    optimizer = result['optimizer']
    scheduler = result.get('lr_scheduler')
    if isinstance(scheduler, dict):
        scheduler = scheduler.get('scheduler')
    base_lr = float(getattr(scheduler, 'lr', 0.0))
    scales = [float(group.get('lr_scale', 1.0)) for group in optimizer.param_groups]
    lrs = [float(group['lr']) for group in optimizer.param_groups]
    declared = [float(group_cfg.get('lr_mult', 1.0))
                for group_cfg in (paramwise_cfg or {}).get('name', {}).values()]
    if lr_scale_enabled:
        missing = [value for value in declared if value not in scales]
        if missing:
            raise SystemExit(f'paramwise lr_mult {missing} missing from optimizer groups {scales}')
    elif any(abs(value - 1.0) > 1e-12 for value in scales):
        raise SystemExit(f'diagnostic flat-lr arm unexpectedly has scales {scales}')
    if base_lr <= 0 or abs(base_lr - float(agent._lr)) > 1e-12:
        raise SystemExit(f'scheduler base lr {base_lr} differs from configured {agent._lr}')
    for group, scale, lr in zip(optimizer.param_groups, scales, lrs):
        expected = base_lr * scale / warmup_epochs
        if abs(lr - expected) > 1e-12:
            raise SystemExit(f'warmup lr {lr} != {expected} for lr_scale {scale}')
    payload = {'paramwise_lr_scale_applied': bool(lr_scale_enabled),
               'param_group_lr_scale': scales,
               'param_group_warmup_lr': lrs,
               'param_group_peak_lr': [base_lr * scale for scale in scales],
               'param_group_params': [len(group['params']) for group in optimizer.param_groups],
               'scheduler_base_lr': base_lr,
               'scheduler_base_lrs': [float(value) for value in
                                      getattr(scheduler, 'base_lrs', [])],
               'scheduler': type(scheduler).__name__ if scheduler is not None else None}
    del result, optimizer, scheduler
    return payload


def resume_contract(resume_path, experiment_root, resume_sha256=None):
    """Validate a full-resume checkpoint and return its warm-start record.

    A checkpoint inside the experiment output directory is accepted as-is.  A
    continuation run starts in a *new* directory and therefore resumes from a
    sibling experiment; that path is only accepted with an explicit SHA256 pin,
    so a swapped or rewritten checkpoint cannot be resumed silently.
    """
    if resume_path is None:
        return None
    resume_path = Path(resume_path).resolve()
    if not resume_path.is_file():
        raise SystemExit(f'resume checkpoint missing: {resume_path}')
    digest = sha256_file(resume_path)
    root = Path(experiment_root).resolve()
    inside = root in resume_path.parents
    if resume_sha256 is not None and digest != resume_sha256.lower():
        raise SystemExit(f'--resume-sha256 mismatch: {digest}')
    if not inside and resume_sha256 is None:
        raise SystemExit('external resume requires --resume-sha256 '
                         '(continuation from another experiment directory)')
    return {'policy': 'full-resume', 'path': str(resume_path), 'sha256': digest,
            'origin': 'inside-experiment' if inside else 'external-pinned',
            'optimizer_restored': True, 'scheduler_restored': True,
            'sampler_restored': False, 'rng_restored': False}


def prepare_output_dir(output, resuming=False):
    """Create the experiment output directory honoring the resume policy.

    A same-directory full-resume has to reuse its own output tree, so the
    directory is expected to already exist.  A fresh run must still refuse an
    existing directory, otherwise a stale experiment would be overwritten
    without any warning.
    """
    output = Path(output).resolve()
    # Never let a missing external disk turn into a silent system-disk directory:
    # ``mkdir(parents=True)`` would hide the real mount point once it returns.
    if not output.parent.parent.is_dir():
        raise SystemExit(f'output root missing, is the disk mounted? {output.parent.parent}')
    output.mkdir(parents=True, exist_ok=resuming)
    return output


def training_form_loss(agent, features, targets, seed):
    """Compute the training-form objective loss on eval-mode inputs.

    Validation uses ``forward_test`` which returns no loss terms, so
    ``transfuser_loss`` falls back to a single-step plain L1 for the trajectory
    branch (about 1/4.5 of the real objective).  This helper temporarily switches
    *only* the trajectory head to its training branch, freezes dropout, fixes the
    diffusion sampling seed and restores every state afterwards, so the returned
    loss matches the objective that actually produced the gradients.
    """
    import torch

    head = agent._transfuser_model._trajectory_head
    previous_branch = head.training
    dropout_states = {module: module.training for module in head.modules()
                      if isinstance(module, torch.nn.Dropout)}
    rng_state = torch.get_rng_state()
    head.training = True
    try:
        for module in dropout_states:
            module.training = False
        torch.manual_seed(int(seed))
        prediction = agent.forward(features, targets)
        return agent.compute_loss(features, targets, prediction)
    finally:
        head.training = previous_branch
        for module, state in dropout_states.items():
            module.training = state
        torch.set_rng_state(rng_state)


def paper_seed(epoch, batch_index):
    """Deterministic diffusion-sampling seed shared by every validation epoch."""
    return (int(epoch) * 100003 + int(batch_index)) % (2 ** 31)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path, default=None,
                        help='full checkpoint from this experiment; restores optimizer/scheduler/global_step')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='override the config output directory (must be a new directory)')
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='gradient norm clip before each optimizer step; default from config or 1.0')
    parser.add_argument('--init-weights', type=Path, default=None,
                        help='weights-only initialization (no optimizer/scheduler/step restore)')
    parser.add_argument('--init-weights-sha256', default=None,
                        help='required SHA256 pin for --init-weights')
    parser.add_argument('--resume-sha256', default=None,
                        help='required SHA256 pin when --resume points outside the experiment output directory')
    parser.add_argument('--no-paramwise-lr-scale', action='store_true',
                        help='diagnostic arm: keep the official flattened param-group lr')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate config/data/model wiring and exit before Trainer.fit')
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if config.get('experiment_scope') != REQUIRED_SCOPE:
        raise SystemExit(f'experiment_scope must be {REQUIRED_SCOPE}')
    if config.get('init_checkpoint') or config.get('init_weights'):
        raise SystemExit('old Status checkpoint initialization is forbidden in this experiment')
    if (args.init_weights is None) != (args.init_weights_sha256 is None):
        raise SystemExit('--init-weights and --init-weights-sha256 must be given together')
    if args.init_weights is not None and args.resume is not None:
        raise SystemExit('choose either --resume (full state) or --init-weights (weights only)')
    if args.init_weights is not None and not args.init_weights.is_file():
        raise SystemExit(f'initial weights checkpoint missing: {args.init_weights}')
    manifest_path = Path(config['dataset_manifest'])
    index_path = Path(config['sample_index'])
    if sha256_file(index_path) != config['sample_index_sha256']:
        raise SystemExit('sample index SHA mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest['index']['sha256'] != config['sample_index_sha256']:
        raise SystemExit('manifest/index SHA mismatch')
    if manifest['navigation_source'] != 'RECONSTRUCTED_EXPERT_ROUTE':
        raise SystemExit('unexpected navigation source')
    split_manifest_path = Path(config['dataset_split_manifest'])
    split_manifest = json.loads(split_manifest_path.read_text())
    if split_manifest.get('schema') != 'diffusiondrive-v1-route-split-v1':
        raise SystemExit('unexpected dataset split schema')
    if config.get('dataset_split_sha256') and sha256_file(split_manifest_path) != config['dataset_split_sha256']:
        raise SystemExit('dataset split SHA mismatch')
    if split_manifest.get('seed') != config.get('seed', 0):
        raise SystemExit('dataset split seed mismatch')
    if any(split_manifest['route_overlap'].values()):
        raise SystemExit('dataset split route overlap')
    for split in ('train', 'validation', 'test'):
        if not split_manifest['frames'].get(split):
            raise SystemExit(f'empty required split: {split}')

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint
    from navsim.planning.training.agent_lightning_module import AgentLightningModule
    from tools.b2d_adapter.v1_builders import official_v1_module
    from tools.training.diffusiondrive_v1.experimental_status_reader import ExperimentalStatusDataset
    from tools.training.diffusiondrive_v1.runtime import model_configuration

    pl.seed_everything(config.get('seed', 0), workers=True)
    base = json.loads(Path(config['base_config']).read_text())
    train = ExperimentalStatusDataset(manifest_path, index_path, config['base_config'], 'train')
    validation = ExperimentalStatusDataset(manifest_path, index_path, config['base_config'],
                                           'validation')
    train_loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], drop_last=True,
                              persistent_workers=config['num_workers'] > 0,
                              prefetch_factor=config.get('prefetch_factor', 2)
                              if config['num_workers'] > 0 else None)
    # The validation loader keeps its workers alive so that Lightning 2.2.1 never
    # tears them down mid-run (a non-persistent loader is not reliably joined and
    # leaked a fresh generation every epoch).  It deliberately uses far fewer
    # workers than training: validation is forward-only, so two readers already
    # saturate the GPU, and each extra worker costs ~3.4 GB RSS on a 30 GB box
    # where the persistent training workers already hold ~23 GB.
    val_workers = min(2, int(config['num_workers']))
    val_loader = DataLoader(validation, batch_size=config['batch_size'], shuffle=False,
                            num_workers=val_workers,
                            persistent_workers=val_workers > 0)

    model_config = model_configuration(base)
    # Experiment-scoped override of the official paramwise policy.  The base config
    # cannot carry it (``model_configuration`` builds a default ``TransfuserConfig``),
    # so a run that needs an extra multiplier declares the full name map here.
    opt_paramwise_override = config.get('opt_paramwise_cfg')
    if opt_paramwise_override is not None:
        if not isinstance(opt_paramwise_override, dict) or 'name' not in opt_paramwise_override:
            raise SystemExit('opt_paramwise_cfg override must be {"name": {...}}')
        model_config.opt_paramwise_cfg = opt_paramwise_override
    # Experiment-scoped override of the official loss weights.  Raising a branch
    # weight is the only lever that changes how much gradient that branch receives
    # (the paramwise lr only scales the resulting step), so a run that wants the
    # BEV head to stop being starved declares the override here.
    loss_weight_override = config.get('loss_weights')
    if loss_weight_override is not None:
        if not isinstance(loss_weight_override, dict) or not loss_weight_override:
            raise SystemExit('loss_weights override must be a non-empty mapping')
        for weight_name, weight_value in loss_weight_override.items():
            if not weight_name.endswith('_weight'):
                raise SystemExit(f'loss weight must end with _weight: {weight_name}')
            if not hasattr(model_config, weight_name):
                raise SystemExit(f'unknown loss weight: {weight_name}')
            setattr(model_config, weight_name, float(weight_value))
    agent = official_v1_module('transfuser_agent').TransfuserAgent(
        model_config, config['learning_rate'], checkpoint_path=None)
    # ``WarmupCosLR`` in the official agent is hard-coded to ``epochs=100``.  When a
    # run is shorter than that the cosine term only travels a few degrees and the
    # learning rate stays near its peak for the whole run.  ``scheduler_epochs``
    # binds the decay horizon to the actual budget; it defaults to ``max_epochs``.
    scheduler_epochs = int(config.get('scheduler_epochs', config['max_epochs']))
    if scheduler_epochs <= 3:
        raise SystemExit('scheduler_epochs must exceed the 3-epoch warmup')
    if scheduler_epochs < int(config['max_epochs']):
        raise SystemExit(f'scheduler_epochs {scheduler_epochs} < max_epochs '
                         f'{config["max_epochs"]}: the cosine horizon must cover '
                         'the whole planned run')
    lr_scale_enabled = bool(config.get('paramwise_lr_scale', True)) \
        and not args.no_paramwise_lr_scale
    if lr_scale_enabled:
        install_paramwise_lr_scale(agent, epochs=scheduler_epochs)
    init_record = None
    if args.init_weights is not None:
        from tools.training.diffusiondrive_v1.status_route_eval_core import load_weights_only
        init_record = load_weights_only(agent, args.init_weights)
        if init_record['sha256'] != args.init_weights_sha256.lower():
            raise SystemExit('--init-weights-sha256 mismatch: ' + init_record['sha256'])
        init_record['policy'] = 'weights-only-from-checkpoint'
    precision_policy = config.get('precision_policy')
    if precision_policy is not None:
        from tools.training.diffusiondrive_v1.bf16_policy import (
            BF16TrainingFP32Evaluation, install_fp32_losses, POLICY, source_sha256)
        if precision_policy != POLICY:
            raise SystemExit(f'unsupported precision_policy: {precision_policy}')
        if args.device != 'cuda':
            raise SystemExit('BF16 convolution policy requires --device cuda')
        install_fp32_losses(agent)
        precision_plugins = [BF16TrainingFP32Evaluation()]
        precision_policy_sha = source_sha256()
    else:
        precision_plugins = None
        precision_policy_sha = None
    resume_record = resume_contract(args.resume, config['output_dir'], args.resume_sha256)
    resume_path = Path(resume_record['path']) if resume_record is not None else None
    if resume_record is not None:
        warm = resume_record
    elif init_record is not None:
        warm = dict(init_record)
        warm['optimizer_restored'] = False
        warm['scheduler_restored'] = False
        warm['sampler_restored'] = False
        warm['rng_restored'] = False
        warm['new_global_step'] = 0
    else:
        warm = {'policy': 'official-scratch',
                'checkpoint_path': None,
                'image_backbone': 'official timm ResNet34 ImageNet pretrained',
                'optimizer_restored': False, 'scheduler_restored': False,
                'sampler_restored': False, 'rng_restored': False, 'new_global_step': 0}

    class StatusLightningModule(AgentLightningModule):
        """Official step/loss plus validation trajectory metrics."""

        def on_validation_epoch_start(self):
            self._val_batch_index = 0

        def _log_paper_loss(self, features, targets, batch_index):
            """Log the training-form objective on this validation batch."""
            losses = training_form_loss(
                self.agent, features, targets,
                paper_seed(self.trainer.current_epoch, batch_index))
            size = len(features['status_feature'])
            for key in ('loss', 'trajectory_loss', 'agent_class_loss',
                        'agent_box_loss', 'bev_semantic_loss'):
                value = losses.get(key)
                if value is None:
                    continue
                name = 'paper_loss' if key == 'loss' else f'paper_{key}'
                self.log(f'val/{name}', value, on_step=False, on_epoch=True,
                         prog_bar=(key == 'loss'), sync_dist=True, batch_size=size)

        def _step(self, batch, logging_prefix):
            if logging_prefix != 'val':
                return super()._step(batch, logging_prefix)
            features, targets = batch
            prediction = self.agent.forward(features, targets)
            loss_dict = self.agent.compute_loss(features, targets, prediction)
            size = len(features['status_feature'])
            self.log('val/samples', float(size), on_step=False, on_epoch=True,
                     sync_dist=True, reduce_fx='sum', batch_size=1)
            for key, value in loss_dict.items():
                if value is not None:
                    self.log(f'val/{key}', value, on_step=False, on_epoch=True,
                             prog_bar=True, sync_dist=True, batch_size=size)
            predicted = prediction['trajectory']
            truth = targets['trajectory']
            error = torch.linalg.norm(predicted[..., :2] - truth[..., :2], dim=-1)
            self.log('val/ade_m', error.mean(), on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=size)
            self.log('val/fde_m', error[:, -1].mean(), on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=size)
            heading = torch.atan2(torch.sin(predicted[:, -1, 2] - truth[:, -1, 2]),
                                  torch.cos(predicted[:, -1, 2] - truth[:, -1, 2]))
            self.log('val/heading_error_deg', heading.abs().mean() * 180.0 / torch.pi,
                     on_step=False, on_epoch=True, sync_dist=True, batch_size=size)
            command = features['status_feature'][:, :4].argmax(dim=1)
            for index, name in enumerate(('left', 'straight', 'right', 'unknown')):
                mask = command == index
                if bool(mask.any()):
                    count = int(mask.sum())
                    self.log(f'val/ade_{name}_m', error[mask].mean(), on_step=False,
                             on_epoch=True, sync_dist=True, batch_size=count)
                    self.log(f'val/fde_{name}_m', error[mask][:, -1].mean(),
                             on_step=False, on_epoch=True, sync_dist=True, batch_size=count)
            index = int(getattr(self, '_val_batch_index', 0))
            self._val_batch_index = index + 1
            self._log_paper_loss(features, targets, index)
            return loss_dict['loss']

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError('non-finite training loss at step '
                                   f'{int(self.trainer.global_step)}; stopped before optimizer.step')
            return loss

    module = StatusLightningModule(agent)

    output = prepare_output_dir(args.output_dir or config['output_dir'],
                                resuming=args.resume is not None)
    max_epochs = int(config['max_epochs'])
    if max_epochs <= 0:
        raise SystemExit('positive max_epochs required')
    if scheduler_epochs < max_epochs:
        raise SystemExit('scheduler_epochs must be >= max_epochs')
    accumulate = int(config.get('accumulate_grad_batches', 1))
    if accumulate <= 0:
        raise SystemExit('positive accumulate_grad_batches required')
    batches_per_epoch = len(train_loader)
    estimated_steps = math.ceil(batches_per_epoch / accumulate) * max_epochs
    split_sha = sha256_file(split_manifest_path)
    grad_clip_val = args.grad_clip if args.grad_clip is not None else float(
        config.get('gradient_clip_val', 1.0))
    if not (grad_clip_val > 0):
        raise SystemExit('gradient clip value must be positive')
    monitor = str(config.get('checkpoint_monitor', 'val/loss'))
    if monitor not in ('val/loss', 'val/paper_loss'):
        raise SystemExit(f'unsupported checkpoint monitor: {monitor}')
    save_top_k = int(config.get('checkpoint_keep', 1))
    if save_top_k < 1:
        raise SystemExit('checkpoint_keep must be >= 1')
    validation_objective = PAPER_VALIDATION if monitor == 'val/paper_loss' else 'legacy-l1'
    paramwise_cfg = agent._config.opt_paramwise_cfg
    lr_contract = optimizer_contract(agent, paramwise_cfg, lr_scale_enabled=lr_scale_enabled)
    identity = {
        'schema': VERSION,
        'dataset_version': manifest['version'],
        'sample_index_sha256': config['sample_index_sha256'],
        'manifest_sha256': sha256_file(manifest_path),
        'dataset_split_manifest': {'path': str(split_manifest_path),
                                   'sha256': split_sha,
                                   'seed': split_manifest['seed'],
                                   'route_counts': split_manifest['route_counts'],
                                   'frames': split_manifest['frames'],
                                   'tail_policy': split_manifest['tail_policy']},
        'status_rule': config['status_rule'],
        'navigation_source': manifest['navigation_source'],
        'online_source_certified': manifest['online_source_certified'],
        'warm_start': warm,
        'resume': str(resume_path) if resume_path is not None else None,
        'batch_size': config['batch_size'],
        'accumulate_grad_batches': accumulate,
        'effective_batch_size': config['batch_size'] * accumulate,
        'num_workers': config['num_workers'],
        'learning_rate': config['learning_rate'],
        'learning_rate_policy': {'schema': 'status-experimental-lr-policy-v1',
                                 'scheduler': (f'official WarmupCosLR {scheduler_epochs} epochs/'
                                               'warmup 3/min_lr 1e-6'),
                                 'scheduler_epochs': scheduler_epochs,
                                 'paramwise_lr_scale_applied': lr_scale_enabled,
                                 'paramwise_cfg': paramwise_cfg,
                                 'expected_lr': expected_lr_plan(config['learning_rate'],
                                                               paramwise_cfg),
                                 'optimizer_contract': lr_contract},
        'gradient_clip': {'val': grad_clip_val, 'algorithm': 'norm',
                          'position': 'before every optimizer step'},
        'loss_weights': {'values': {name: float(getattr(model_config, name))
                                    for name in LOSS_WEIGHT_NAMES},
                         'override': loss_weight_override},
        'ema': ({'enabled': True, 'decay': float(config.get('ema_decay', 0.999)),
                 'window_steps': round(1.0 / (1.0 - float(config.get('ema_decay', 0.999)))),
                 'window_epochs': round((1.0 / (1.0 - float(config.get('ema_decay', 0.999))))
                                        / max(1.0, len(train_loader) / accumulate), 2),
                 'validates_with': 'ema-weights',
                 'checkpoint_weights': 'raw-weights-matching-optimizer'}
                if bool(config.get('ema_enabled', False)) else {'enabled': False}),
        'precision_policy': precision_policy,
        'precision_policy_source_sha256': precision_policy_sha,
        'max_epochs': max_epochs,
        'scheduler_epochs': scheduler_epochs,
        'batches_per_epoch': batches_per_epoch,
        'estimated_optimizer_steps': estimated_steps,
        'planned_epochs': max_epochs,
        'validation_full': True,
        'validation_rows': len(validation),
        'test_rows': split_manifest['frames']['test'],
        'test_loaded': False,
        'checkpoint_policy': {'monitor': monitor,
                              'validation_objective': validation_objective,
                              'paper_seed_policy': PAPER_SEED_POLICY,
                              'save_top_k': save_top_k,
                              'save_last': True,
                              'full_validation': True,
                              'validation_rows': len(validation),
                              'periodic_step_checkpoints': False,
                              'pause_request': 'pause.request'},
        'code_sha256': {
            'status_builder': sha256_file(ROOT/'agent/navigation/status_builder.py'),
            'reader': sha256_file(ROOT/'tools/training/diffusiondrive_v1/experimental_status_reader.py'),
            'trainer': sha256_file(Path(__file__)),
            'route_split': sha256_file(ROOT/'tools/training/diffusiondrive_v1/route_split.py'),
        },
    }
    (output / 'run_identity.json').write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'train_rows': len(train), 'val_rows': len(validation),
                          'batches_per_epoch': batches_per_epoch, 'max_epochs': max_epochs,
                          'scheduler_epochs': scheduler_epochs,
                          'estimated_optimizer_steps': estimated_steps,
                          'warm_start': warm['policy'],
                          'gradient_clip': {'val': grad_clip_val, 'algorithm': 'norm'},
                          'optimizer_contract': lr_contract,
                          'expected_lr': identity['learning_rate_policy']['expected_lr']}))
        return

    class ExperimentControl(pl.Callback):
        """Write progress JSON and honor a cooperative ``pause.request``."""

        def __init__(self, directory, total_steps, progress_every, grad_log_every,
                     lr_scale_enabled, scheduler_epochs=None):
            super().__init__()
            self.directory = Path(directory)
            self.total_steps = int(total_steps)
            self.progress_every = max(1, int(progress_every))
            self.grad_log_every = max(1, int(grad_log_every))
            self.lr_scale_enabled = bool(lr_scale_enabled)
            self.scheduler_epochs = None if scheduler_epochs is None else int(scheduler_epochs)
            self.paused = False
            self.learning_rates = None

        @staticmethod
        def _live_lrs(trainer):
            """Read the optimizer groups, never a snapshot taken before training."""
            return live_learning_rates(trainer)

        def _write(self, state, trainer, **extra):
            live = self._live_lrs(trainer)
            snapshot = self.learning_rates or {}
            payload = {'schema': 'status-experimental-progress-v1',
                       'state': state,
                       'global_step': int(trainer.global_step),
                       'epoch': int(trainer.current_epoch),
                       'max_steps': self.total_steps,
                       'epochs': self.scheduler_epochs,
                       'param_group_lr_scale': snapshot.get('param_group_lr_scale'),
                       'learning_rates': live if live is not None
                       else snapshot.get('learning_rates'),
                       'updated': time.strftime('%Y-%m-%dT%H:%M:%S')}
            payload.update(extra)
            temporary = self.directory / 'training_progress.json.tmp'
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
            temporary.replace(self.directory / 'training_progress.json')

        def _enforce_lr_scale(self, trainer, pl_module):
            """Restore ``lr_scale`` after a resume rewrites ``param_groups`` from disk."""
            optimizer = trainer.optimizers[0] if trainer.optimizers else None
            agent_module = getattr(pl_module, 'agent', None)
            if optimizer is None or agent_module is None or not hasattr(agent_module, '_config'):
                return
            configs = list(getattr(trainer, 'lr_scheduler_configs', []) or [])
            scheduler = configs[0].scheduler if configs else None
            if self.lr_scale_enabled:
                payload = audit_lr_scale(optimizer, agent_module._transfuser_model,
                                         agent_module._config.opt_paramwise_cfg, scheduler,
                                         scheduler_epochs=self.scheduler_epochs)
            else:
                payload = {'param_group_lr_scale': [float(group.get('lr_scale', 1.0))
                                                   for group in optimizer.param_groups],
                           'learning_rates': [float(group['lr'])
                                              for group in optimizer.param_groups],
                           'scheduler_base_lrs': [float(value) for value in
                                                  getattr(scheduler, 'base_lrs', [])],
                           'scheduler_base_lr': float(getattr(scheduler, 'lr', 0.0) or 0.0),
                           'scheduler_last_epoch': int(getattr(scheduler, 'last_epoch', -1))}
            payload['schema'] = 'status-experimental-lr-audit-v1'
            payload['global_step'] = int(trainer.global_step)
            payload['epoch'] = int(trainer.current_epoch)
            payload['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            with (self.directory / 'lr_schedule.jsonl').open('a') as stream:
                stream.write(json.dumps(payload, sort_keys=True) + '\n')
            self.learning_rates = payload

        def on_train_start(self, trainer, pl_module):
            # A checkpoint saved during validation can leave val_loop.restarting=True,
            # which replays only the remaining batches and yields misleading metrics.
            val_loop = getattr(trainer.fit_loop.epoch_loop, 'val_loop', None)
            if val_loop is not None:
                val_loop.restarting = False
            self._enforce_lr_scale(trainer, pl_module)
            self._write('running', trainer)

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            step = int(trainer.global_step)
            if step % self.grad_log_every != 0:
                return
            agent_module = getattr(pl_module, 'agent', None)
            if agent_module is None:
                return
            norms = grouped_gradient_norms(agent_module._transfuser_model)
            total = math.sqrt(sum(value * value for value in norms.values()))
            clip_scale = min(1.0, float(grad_clip_val) / total) if total > 0 else 1.0
            record = {'schema': 'status-experimental-grad-norms-v1',
                      'global_step': step, 'epoch': int(trainer.current_epoch),
                      'total': total, 'branches': norms,
                      'gradient_clip_val': float(grad_clip_val),
                      'clip_scale': clip_scale}
            with (self.directory / 'grad_norms.jsonl').open('a') as stream:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
            if not math.isfinite(total) or not all(math.isfinite(value)
                                                   for value in norms.values()):
                raise RuntimeError('non-finite gradient norm at step '
                                   f'{step}; stopped before optimizer.step')

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = int(trainer.global_step)
            request = self.directory / 'pause.request'
            if request.exists() and step > 0 and not self.paused:
                checkpoint = self.directory / f'pause-step{step:06d}.ckpt'
                trainer.save_checkpoint(checkpoint)
                request.replace(self.directory / 'pause.applied')
                self._write('paused', trainer, pause_checkpoint=str(checkpoint))
                self.paused = True
                request_graceful_stop(trainer)
                return
            if step % self.progress_every == 0:
                self._write('running', trainer)

    control = ExperimentControl(output, estimated_steps, config.get('progress_every_steps', 50),
                                config.get('gradient_log_every_steps', 10), lr_scale_enabled,
                                scheduler_epochs=scheduler_epochs)
    checkpoint = ModelCheckpoint(
        dirpath=output / 'checkpoints',
        monitor=monitor,
        mode='min',
        save_top_k=save_top_k,
        save_last=True,
        every_n_epochs=1,
        filename='best-{epoch:03d}-{step:06d}',
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint, control]
    ema_callback = None
    if bool(config.get('ema_enabled', False)):
        ema_decay = float(config.get('ema_decay', 0.999))
        if not 0.0 < ema_decay < 1.0:
            raise SystemExit('ema_decay must be in (0, 1)')
        ema_callback = make_ema_lightning_callback(decay=ema_decay, directory=output)
        # Registered BEFORE ModelCheckpoint so the raw weights are restored in
        # on_validation_end before the checkpoint writes them.
        callbacks = [ema_callback, checkpoint, control]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if args.device == 'cuda' else 'cpu',
        devices=1,
        precision=None if precision_plugins else '32',
        default_root_dir=str(output),
        callbacks=callbacks,
        log_every_n_steps=config['log_every_n_steps'],
        limit_val_batches=1.0,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=accumulate,
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm='norm',
        enable_checkpointing=True,
        logger=pl.loggers.CSVLogger(str(output/'logs')),
        num_sanity_val_steps=0,
        plugins=precision_plugins,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=str(resume_path) if resume_path is not None else None)
    paused = bool(control.paused)
    result = {'status': 'paused' if paused else 'completed',
              'global_step': trainer.global_step,
              'max_epochs': max_epochs,
              'validation_rows': len(validation),
              'gradient_clip_val': grad_clip_val,
              'learning_rate_audit': control.learning_rates,
              'lr_schedule_log': str(output / 'lr_schedule.jsonl'),
              'grad_norm_log': str(output / 'grad_norms.jsonl'),
              'output_dir': str(output),
              'run_identity': str(output/'run_identity.json'),
              'pause_checkpoint': str(output / f'pause-step{trainer.global_step:06d}.ckpt')
              if paused else None}
    (output / 'train_result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    control._write('paused' if paused else 'completed', trainer)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
