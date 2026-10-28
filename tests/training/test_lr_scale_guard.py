"""Regression guards for the Agent-box jump fix in the Status training entry.

These tests pin the local adapter that restores the official ``image_encoder``
``lr_mult`` (the official builder never writes ``lr_scale``, so ``WarmupCosLR``
flattens both groups to the base lr), plus the gradient-norm / non-finite
guards that make a silent branch collapse impossible.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'repositories/DiffusionDrive'))

from navsim.agents.diffusiondrive.modules.scheduler import WarmupCosLR
from tools.training.diffusiondrive_v1.status_experimental_train import (
    apply_paramwise_lr_scale,
    audit_lr_scale,
    expected_lr_plan,
    grouped_gradient_norms,
    install_paramwise_lr_scale,
    optimizer_contract,
    prepare_output_dir,
    resume_contract,
)

PARAMWISE = {'name': {'image_encoder': {'lr_mult': 0.5}}}
BASE_LR = 6e-4


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Linear(2, 2)
        self.fuser = torch.nn.Linear(2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = Backbone()
        self._bev_semantic_head = torch.nn.Linear(2, 2)
        self._agent_head = torch.nn.Linear(2, 2)
        self._trajectory_head = torch.nn.Linear(2, 2)


def official_style_optimizer(model, base_lr=BASE_LR):
    """Mirror ``TransfuserAgent.get_coslr_optimizers``: extra group lr, no ``lr_scale``."""
    matched = [p for n, p in model.named_parameters() if 'image_encoder' in n]
    rest = [p for n, p in model.named_parameters() if 'image_encoder' not in n]
    optimizer = torch.optim.AdamW([{'params': rest, 'lr': base_lr}], weight_decay=1e-4)
    optimizer.add_param_group({'params': matched, 'lr': base_lr * 0.5})
    return optimizer


class FakeAgent:
    def __init__(self, model, base_lr=BASE_LR):
        self._transfuser_model = model
        self._lr = base_lr
        self._config = type('Config', (), {'opt_paramwise_cfg': PARAMWISE})()

    def get_coslr_optimizers(self):
        optimizer = official_style_optimizer(self._transfuser_model, self._lr)
        scheduler = WarmupCosLR(optimizer, lr=self._lr, min_lr=1e-6,
                                epochs=100, warmup_epochs=3)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def test_lr_scale_is_written_and_scheduler_keeps_half_rate_backbone():
    model = Model()
    optimizer = official_style_optimizer(model)
    assert all('lr_scale' not in group for group in optimizer.param_groups)
    assert apply_paramwise_lr_scale(optimizer, model, PARAMWISE) == [1.0, 0.5]
    assert [group['lr_scale'] for group in optimizer.param_groups] == [1.0, 0.5]

    scheduler = WarmupCosLR(optimizer, min_lr=1e-6, lr=BASE_LR, warmup_epochs=3, epochs=100)
    assert scheduler.base_lrs == pytest.approx([BASE_LR, BASE_LR * 0.5])
    assert [group['lr'] for group in optimizer.param_groups] == pytest.approx([2e-4, 1e-4])
    optimizer.step(); scheduler.step()
    assert [group['lr'] for group in optimizer.param_groups] == pytest.approx([4e-4, 2e-4])
    optimizer.step(); scheduler.step()
    assert [group['lr'] for group in optimizer.param_groups] == pytest.approx([6e-4, 3e-4])
    optimizer.step(); scheduler.step()
    assert [group['lr'] for group in optimizer.param_groups] == pytest.approx([6e-4, 3e-4])


def test_install_wraps_official_builder_without_touching_sources():
    from tools.b2d_adapter.v1_builders import V1_ROOT
    scheduler_source = V1_ROOT / 'navsim/agents/diffusiondrive/modules/scheduler.py'
    before = scheduler_source.read_bytes()
    agent = install_paramwise_lr_scale(FakeAgent(Model()))
    result = agent.get_coslr_optimizers()
    assert result['paramwise_lr_scale'] == [1.0, 0.5]
    optimizer = result['optimizer']
    assert [group['lr_scale'] for group in optimizer.param_groups] == [1.0, 0.5]
    assert [group['lr'] for group in optimizer.param_groups] == pytest.approx([2e-4, 1e-4])
    assert result['lr_scheduler'].base_lrs == pytest.approx([BASE_LR, BASE_LR * 0.5])
    assert scheduler_source.read_bytes() == before


def test_restored_param_groups_regain_lr_scale_after_resume():
    """A checkpoint written before the fix stores groups without ``lr_scale``."""
    model = Model()
    agent = install_paramwise_lr_scale(FakeAgent(model))
    optimizer = agent.get_coslr_optimizers()['optimizer']
    saved = optimizer.state_dict()
    for group in saved['param_groups']:
        group.pop('lr_scale', None)
        group['lr'] = 5.7e-4
    restored = official_style_optimizer(model)
    restored.load_state_dict(saved)
    assert all('lr_scale' not in group for group in restored.param_groups)
    assert apply_paramwise_lr_scale(restored, model, PARAMWISE) == [1.0, 0.5]
    assert restored.param_groups[0]['lr_scale'] == 1.0


def test_mixed_multipliers_inside_one_group_are_rejected():
    model = Model()
    optimizer = torch.optim.AdamW([{'params': list(model.parameters()), 'lr': BASE_LR}])
    with pytest.raises(RuntimeError):
        apply_paramwise_lr_scale(optimizer, model, PARAMWISE)


def test_grouped_gradient_norms_split_branches_and_expose_non_finite():
    model = Model()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    norms = grouped_gradient_norms(model)
    assert set(norms) == {'backbone', 'bev_head', 'agent_head', 'trajectory_head', 'other'}
    backbone_elements = sum(p.numel() for n, p in model.named_parameters() if '_backbone.' in n)
    agent_elements = sum(p.numel() for n, p in model.named_parameters() if '_agent_head.' in n)
    assert norms['backbone'] == pytest.approx(math.sqrt(4.0 * backbone_elements))
    assert norms['agent_head'] == pytest.approx(math.sqrt(4.0 * agent_elements))
    assert norms['other'] == 0.0
    list(model._agent_head.parameters())[0].grad[0, 0] = float('nan')
    assert not math.isfinite(grouped_gradient_norms(model)['agent_head'])


def test_audit_lr_scale_repairs_resumed_groups_and_reports_peak():
    from tools.training.diffusiondrive_v1.status_experimental_train import audit_lr_scale
    model = Model()
    agent = install_paramwise_lr_scale(FakeAgent(model))
    result = agent.get_coslr_optimizers()
    saved_optimizer = result['optimizer'].state_dict()
    saved_scheduler = result['lr_scheduler'].state_dict()
    for group in saved_optimizer['param_groups']:      # checkpoint written before the fix
        group.pop('lr_scale', None)
        group['lr'] = 5.697e-4
    saved_scheduler['base_lrs'] = [BASE_LR, BASE_LR]
    saved_scheduler['last_epoch'] = 3

    restored = official_style_optimizer(model)
    live = WarmupCosLR(restored, min_lr=1e-6, lr=BASE_LR, epochs=100, warmup_epochs=3)
    restored.load_state_dict(saved_optimizer)
    live.load_state_dict(saved_scheduler)
    assert [group['lr'] for group in restored.param_groups] == pytest.approx([5.697e-4, 5.697e-4])

    audit = audit_lr_scale(restored, model, PARAMWISE, live)
    assert audit['param_group_lr_scale'] == [1.0, 0.5]
    assert audit['learning_rates'] == pytest.approx([BASE_LR, 3e-4])
    assert audit['scheduler_base_lrs'] == pytest.approx([BASE_LR, 3e-4])
    assert audit['scheduler_last_epoch'] == 3


def test_scheduler_epochs_override_replaces_the_restored_100_epoch_horizon():
    """A resumed checkpoint always restores ``epochs=100``; the planned horizon wins.

    Without the override the cosine term only advances a few degrees and the
    learning rate stays at ~2.6e-4 for the whole continuation.
    """
    model = Model()
    optimizer = official_style_optimizer(model)
    apply_paramwise_lr_scale(optimizer, model, PARAMWISE)
    live = WarmupCosLR(optimizer, min_lr=1e-6, lr=BASE_LR, epochs=100, warmup_epochs=3)
    live.last_epoch = 26          # epoch 26 of the parent run, resumed from disk
    live._step_count = 27

    flat = audit_lr_scale(optimizer, model, PARAMWISE, live)
    assert flat['scheduler_epochs'] == 100
    assert 'scheduler_epochs_injected' not in flat
    flat_expected = 1e-6 + 0.5 * (BASE_LR - 1e-6) * (
        1 + math.cos(math.pi * (26 - 3) / (100 - 3)))
    assert flat['learning_rates'][0] == pytest.approx(flat_expected, rel=1e-9)
    assert flat_expected > 2e-4

    injected = audit_lr_scale(optimizer, model, PARAMWISE, live, scheduler_epochs=50)
    assert injected['scheduler_epochs'] == 50
    assert injected['scheduler_epochs_injected'] == {'from': 100, 'to': 50}
    assert live.epochs == 50
    expected = 1e-6 + 0.5 * (BASE_LR - 1e-6) * (
        1 + math.cos(math.pi * (26 - 3) / (50 - 3)))
    assert injected['learning_rates'][0] == pytest.approx(expected, rel=1e-9)
    assert injected['learning_rates'][1] == pytest.approx(expected * 0.5, rel=1e-9)
    assert injected['learning_rates'][0] < flat['learning_rates'][0]


def test_scheduler_epochs_override_is_idempotent():
    model = Model()
    optimizer = official_style_optimizer(model)
    apply_paramwise_lr_scale(optimizer, model, PARAMWISE)
    live = WarmupCosLR(optimizer, min_lr=1e-6, lr=BASE_LR, epochs=50, warmup_epochs=3)
    live.last_epoch = 26
    payload = audit_lr_scale(optimizer, model, PARAMWISE, live, scheduler_epochs=50)
    assert 'scheduler_epochs_injected' not in payload
    assert payload['scheduler_epochs'] == 50


def test_progress_learning_rates_are_read_from_the_live_optimizer(tmp_path):
    """``training_progress.json`` must never report a stale pre-training snapshot."""
    from tools.training.diffusiondrive_v1.status_experimental_train import live_learning_rates

    class FakeOptimizer:
        param_groups = [{'lr': 2.7531e-4}, {'lr': 1.3765e-4}]

    class FakeTrainer:
        optimizers = [FakeOptimizer()]

    assert live_learning_rates(FakeTrainer()) == pytest.approx([2.7531e-4, 1.3765e-4])

    class NoOptimizer:
        optimizers = []

    assert live_learning_rates(NoOptimizer()) is None


def test_graceful_stop_caps_epoch_loop_max_steps():
    """Lightning 2.x has no writable ``Trainer.should_stop``.

    Setting it silently does nothing, which let ``pause.request`` save a
    checkpoint and then keep training.  The stop must cap ``max_steps``.
    """
    from tools.training.diffusiondrive_v1.status_experimental_train import request_graceful_stop

    class EpochLoop:
        max_steps = 125200

    class FitLoop:
        epoch_loop = EpochLoop()

    class Trainer:
        fit_loop = FitLoop()
        global_step = 67668

    trainer = Trainer()
    assert request_graceful_stop(trainer) is True
    assert trainer.fit_loop.epoch_loop.max_steps == 67668
    # Repeating must stay idempotent at the same step.
    assert request_graceful_stop(trainer) is True
    assert trainer.fit_loop.epoch_loop.max_steps == 67668

    class NoLoop:
        global_step = 10

    assert request_graceful_stop(NoLoop()) is False


def test_graceful_stop_really_ends_real_lightning():
    """End-to-end proof on the installed Lightning version."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from tools.training.diffusiondrive_v1.status_experimental_train import request_graceful_stop

    class Module(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)

        def training_step(self, batch, batch_idx):
            return self.layer(batch[0]).pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-2)

    class StopAt(pl.Callback):
        def __init__(self, step):
            super().__init__()
            self.step = step

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if int(trainer.global_step) >= self.step:
                request_graceful_stop(trainer)

    loader = DataLoader(TensorDataset(torch.randn(64, 2)), batch_size=8)
    trainer = pl.Trainer(max_epochs=50, accelerator='cpu', logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         limit_val_batches=0, callbacks=[StopAt(3)])
    trainer.fit(Module(), loader)
    assert trainer.global_step == 3


def test_ema_tracks_parameters_with_warmup_decay():
    """EMA must follow the moving average and warm up before using the full decay."""
    from tools.training.diffusiondrive_v1.status_experimental_train import ModelEMA

    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, decay=0.999)
    start = {n: p.detach().clone() for n, p in model.named_parameters()}

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)                    # num_updates 0 -> effective decay 0.1
    moved = (ema.shadow['weight'] - start['weight']).mean().item()
    assert 0.0 < moved < 1.0, f'first update moved {moved}'

    for _ in range(200):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01)
        ema.update(model)
    assert ema.num_updates == 201
    # After warmup the shadow lags the live weights.
    live = model.weight.detach()
    assert not torch.allclose(ema.shadow['weight'], live, atol=1e-3)


def test_ema_copy_to_restores_exactly():
    from tools.training.diffusiondrive_v1.status_experimental_train import ModelEMA

    model = torch.nn.Linear(3, 2)
    ema = ModelEMA(model, decay=0.9)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(5.0)
    ema.copy_to(model)
    for name, p in model.named_parameters():
        assert torch.allclose(p, original[name]), name


def test_ema_state_roundtrip():
    from tools.training.diffusiondrive_v1.status_experimental_train import ModelEMA

    model = torch.nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.995)
    ema.update(model)
    state = ema.state_dict()
    restored = ModelEMA(model, decay=0.5)
    restored.load_state_dict(state)
    assert restored.decay == pytest.approx(0.995)
    assert restored.num_updates == ema.num_updates
    assert torch.allclose(restored.shadow['weight'], ema.shadow['weight'])


def test_ema_callback_swaps_in_for_validation_and_restores_raw(tmp_path):
    """Validation scores must come from EMA weights while the checkpoint keeps raw ones."""
    from tools.training.diffusiondrive_v1.status_experimental_train import EMACallback

    model = torch.nn.Linear(2, 1)
    raw = {n: p.detach().clone() for n, p in model.named_parameters()}

    class Module:
        agent = type('A', (), {'_transfuser_model': model})

    class Trainer:
        global_step = 0
        current_epoch = 0

    module, trainer = Module(), Trainer()
    callback = EMACallback(decay=0.9, directory=tmp_path)

    # first batch builds the shadow, later steps update it
    for step in range(1, 6):
        trainer.global_step = step
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        callback.on_train_batch_end(trainer, module, None, None, 0)
    assert callback.ema is not None

    with torch.no_grad():
        for p in model.parameters():
            p.add_(100.0)                     # simulate latest step
    callback.on_validation_start(trainer, module)
    assert torch.allclose(model.weight.detach(), callback.ema.shadow['weight']), \
        'validation must run on EMA weights'

    callback.on_validation_end(trainer, module)
    for name, p in model.named_parameters():
        assert torch.allclose(p, raw[name] + 105.0), f'{name} not restored'

    saved = list(tmp_path.glob('ema-*.ckpt'))
    assert len(saved) == 1
    payload = torch.load(saved[0], map_location='cpu', weights_only=False)
    assert payload['schema'] == 'status-ema-weights-v1'
    assert payload['num_updates'] == 4    # first batch only builds the shadow


def test_ema_is_skipped_without_optimizer_step():
    """Gradient accumulation windows must not advance the average."""
    from tools.training.diffusiondrive_v1.status_experimental_train import EMACallback

    model = torch.nn.Linear(2, 1)

    class Module:
        agent = type('A', (), {'_transfuser_model': model})

    class Trainer:
        global_step = 7
        current_epoch = 0

    callback = EMACallback(decay=0.9)
    trainer, module = Trainer(), Module()
    callback.on_train_batch_end(trainer, module, None, None, 0)   # builds shadow
    callback.on_train_batch_end(trainer, module, None, None, 1)   # same step -> skipped
    callback.on_train_batch_end(trainer, module, None, None, 2)   # same step -> skipped
    assert callback.ema.num_updates == 0


def test_ema_state_survives_resume_before_the_shadow_exists():
    """A restored payload must not be dropped while ``self.ema`` is still None.

    Load order in Lightning is ``load_state_dict`` -> first ``on_train_batch_end``,
    and the shadow is only built on that first batch (the model is unknown before
    it).  Reading the payload only when ``self.ema`` already exists silently
    restarts the average from the raw weights on every resume.
    """
    from tools.training.diffusiondrive_v1.status_experimental_train import EMACallback

    model = torch.nn.Linear(2, 1)

    class Module:
        agent = type('A', (), {'_transfuser_model': model})

    class Trainer:
        global_step = 100
        current_epoch = 3

    source = EMACallback(decay=0.9)
    source_trainer = Trainer()
    for step in range(5):
        source_trainer.global_step = step
        source.on_train_batch_end(source_trainer, Module(), None, None, 0)
    source_payload = source.state_dict()
    assert source_payload['ema']['num_updates'] == 4

    restored = EMACallback(decay=0.9)
    restored.load_state_dict(source_payload)
    assert restored.ema is None and restored.pending_ema_state is not None

    restored_trainer = Trainer()
    restored_trainer.global_step = 5
    restored.on_train_batch_end(restored_trainer, Module(), None, None, 0)
    assert restored.ema.num_updates == 4, 'shadow was rebuilt instead of restored'
    assert restored.pending_ema_state is None
    for name, value in source_payload['ema']['shadow'].items():
        assert torch.allclose(restored.ema.shadow[name], value)

    # A callback that already owns a shadow applies the payload immediately.
    live = EMACallback(decay=0.9)
    live.on_train_batch_end(Trainer(), Module(), None, None, 0)
    live.load_state_dict(source_payload)
    assert live.ema.num_updates == 4 and live.pending_ema_state is None


def test_ema_restore_casts_shadow_to_the_live_device_and_dtype():
    """A restored payload must land on the shadow's device, not the loader's.

    Checkpoints are usually read onto CPU while the live shadow follows the model
    onto the accelerator, so a plain ``clone()`` leaves ``update`` raising
    "Expected all tensors to be on the same device".
    """
    from tools.training.diffusiondrive_v1.status_experimental_train import ModelEMA

    model = torch.nn.Linear(3, 2)
    ema = ModelEMA(model, decay=0.9)
    payload = {'decay': 0.9, 'num_updates': 42,
               'shadow': {k: v.double() for k, v in ema.state_dict()['shadow'].items()}}

    ema.load_state_dict(payload)
    for name, value in ema.shadow.items():
        assert value.dtype == model.state_dict()[name].dtype, f'{name} dtype drifted'
        assert value.device == model.state_dict()[name].device
    assert ema.num_updates == 42

    ema.update(model)   # must not raise a cross-device / cross-dtype error
    assert ema.num_updates == 43


def test_expected_lr_plan_documents_half_rate_backbone():
    plan = expected_lr_plan(BASE_LR, PARAMWISE, warmup_epochs=3)
    assert plan['image_encoder'] == {'lr_scale': 0.5, 'first_step_lr': pytest.approx(1e-4),
                                     'peak_lr': pytest.approx(3e-4)}
    assert plan['default']['first_step_lr'] == pytest.approx(2e-4)
    assert plan['default']['peak_lr'] == pytest.approx(BASE_LR)


def test_optimizer_contract_accepts_fixed_builder_and_rejects_flat_lr():
    contract = optimizer_contract(install_paramwise_lr_scale(FakeAgent(Model())), PARAMWISE)
    assert contract['param_group_lr_scale'] == [1.0, 0.5]
    assert contract['param_group_warmup_lr'] == pytest.approx([2e-4, 1e-4])
    assert contract['param_group_peak_lr'] == pytest.approx([BASE_LR, 3e-4])
    assert contract['scheduler'] == 'WarmupCosLR'

    flat = FakeAgent(Model())
    with pytest.raises(SystemExit):
        optimizer_contract(flat, PARAMWISE)


def test_diagnostic_flat_lr_arm_contract_requires_flattened_groups():
    contract = optimizer_contract(FakeAgent(Model()), PARAMWISE, lr_scale_enabled=False)
    assert contract['paramwise_lr_scale_applied'] is False
    assert contract['param_group_lr_scale'] == [1.0, 1.0]
    assert contract['param_group_warmup_lr'] == pytest.approx([2e-4, 2e-4])
    assert contract['param_group_peak_lr'] == pytest.approx([BASE_LR, BASE_LR])
    with pytest.raises(SystemExit):
        optimizer_contract(install_paramwise_lr_scale(FakeAgent(Model())), PARAMWISE,
                           lr_scale_enabled=False)


def test_resume_contract_accepts_inside_and_pins_external(tmp_path):
    import hashlib
    root = tmp_path / 'experiment'
    (root / 'checkpoints').mkdir(parents=True)
    checkpoint = root / 'checkpoints' / 'last.ckpt'
    checkpoint.write_bytes(b'checkpoint-bytes')
    digest = hashlib.sha256(b'checkpoint-bytes').hexdigest()

    inside = resume_contract(checkpoint, root)
    assert inside['origin'] == 'inside-experiment'
    assert inside['sha256'] == digest and inside['optimizer_restored'] is True

    continuation = tmp_path / 'continuation'
    with pytest.raises(SystemExit):
        resume_contract(checkpoint, continuation)              # unpinned external
    pinned = resume_contract(checkpoint, continuation, digest.upper())
    assert pinned['origin'] == 'external-pinned'

    with pytest.raises(SystemExit):
        resume_contract(checkpoint, continuation, '0' * 64)    # wrong pin
    with pytest.raises(SystemExit):
        resume_contract(checkpoint, root, '0' * 64)            # wrong pin inside
    with pytest.raises(SystemExit):
        resume_contract(tmp_path / 'missing.ckpt', root)
    assert resume_contract(None, root) is None


def test_prepare_output_dir_allows_same_directory_resume_only(tmp_path):
    from tools.training.diffusiondrive_v1.status_experimental_train import prepare_output_dir
    root = tmp_path / 'experiments'
    root.mkdir()
    target = root / 'r1'

    assert prepare_output_dir(target) == target.resolve()
    assert target.is_dir()

    # A fresh run must not silently reuse an existing experiment tree.
    with pytest.raises(FileExistsError):
        prepare_output_dir(target)

    # A same-directory full-resume must accept the tree it already owns.
    assert prepare_output_dir(target, resuming=True) == target.resolve()

    with pytest.raises(SystemExit):
        prepare_output_dir(tmp_path / 'missing-mount' / 'child' / 'run')


class _FakeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(0.5)
        self.branch = None

    def forward(self, x):
        self.branch = 'train' if self.training else 'eval'
        return self.drop(x) * 2 + torch.randn(x.shape) * 0.01


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._trajectory_head = _FakeHead()


class _FakeAgent:
    def __init__(self):
        self._transfuser_model = _FakeModel()

    def forward(self, features, targets):
        return {'trajectory': self._transfuser_model._trajectory_head(features)}

    def compute_loss(self, features, targets, prediction):
        return {'loss': prediction['trajectory'].sum(),
                'trajectory_loss': prediction['trajectory'].sum()}


def test_paper_seed_is_deterministic_and_unique():
    from tools.training.diffusiondrive_v1.status_experimental_train import paper_seed
    assert paper_seed(3, 5) == 3 * 100003 + 5
    values = {paper_seed(e, b) for e in range(4) for b in range(5)}
    assert len(values) == 20
    assert all(0 <= value < 2 ** 31 for value in values)


def test_training_form_loss_uses_train_branch_and_restores_all_state():
    from tools.training.diffusiondrive_v1.status_experimental_train import (
        paper_seed, training_form_loss)
    agent = _FakeAgent()
    agent._transfuser_model.eval()
    head = agent._transfuser_model._trajectory_head
    assert head.training is False and head.drop.training is False

    rng_before = torch.get_rng_state().clone()
    features = torch.ones(4, 3)
    first = training_form_loss(agent, features, None, paper_seed(3, 5))

    assert head.branch == 'train'                       # training branch was used
    assert head.training is False and head.drop.training is False   # restored
    assert torch.equal(torch.get_rng_state(), rng_before)          # RNG restored
    assert 'loss' in first and 'trajectory_loss' in first

    again = training_form_loss(agent, features, None, paper_seed(3, 5))
    assert abs(first['loss'].item() - again['loss'].item()) < 1e-7  # same seed -> same noise
    other = training_form_loss(agent, features, None, paper_seed(3, 6))
    assert abs(first['loss'].item() - other['loss'].item()) > 1e-9  # different seed -> different noise

    agent.forward(features, None)
    assert head.branch == 'eval'                        # eval path untouched outside
