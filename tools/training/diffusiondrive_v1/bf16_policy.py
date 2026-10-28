"""Execution-only AMP boundaries and an explicit FP32 -> BF16 contract transition.

No official source or state_dict keys are changed. Validation remains FP32.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
from pathlib import Path

import torch
from pytorch_lightning.plugins.precision import MixedPrecision


POLICY = 'bf16-conv-fp32-fusion-heads-loss-eval-v1'


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def fp32_tree(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: fp32_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(fp32_tree(v) for v in value)
    if isinstance(value, list):
        return [fp32_tree(v) for v in value]
    return value


def fp32_call(function):
    @wraps(function)
    def call(*args, **kwargs):
        # Cast BEFORE the official reduction, without detaching the graph.
        with torch.autocast('cuda', enabled=False):
            return function(*fp32_tree(args), **fp32_tree(kwargs))
    return call


def install_fp32_losses(agent):
    if getattr(agent, '_b2d_precision_policy', None):
        raise RuntimeError('Precision boundaries already installed')
    computer = agent._transfuser_model._trajectory_head.loss_computer
    computer.forward = fp32_call(computer.forward)
    agent.compute_loss = fp32_call(agent.compute_loss)
    # Whole-network and whole-backbone AMP failed real geometry comparisons.
    # Keep fusion/normalization/heads in FP32; opt in only backbone Conv2d.
    # Function boundaries preserve module types, parameters and state_dict keys.
    network = agent._transfuser_model
    network.forward = fp32_call(network.forward)
    def wrap_convolution(convolution):
        original = convolution.forward
        @wraps(original)
        def forward(*args, **kwargs):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=convolution.training):
                result = original(*args, **kwargs)
            return fp32_tree(result)
        convolution.forward = forward
    for convolution in network._backbone.modules():
        if isinstance(convolution, torch.nn.Conv2d):wrap_convolution(convolution)
    agent._b2d_precision_policy = POLICY


class BF16TrainingFP32Evaluation(MixedPrecision):
    def __init__(self):
        super().__init__('bf16-mixed', 'cuda')

    def pre_backward(self, tensor, module):
        if not torch.isfinite(tensor.detach()).all():
            raise FloatingPointError('Nonfinite BF16 training loss; stopping before optimizer update')
        return super().pre_backward(tensor, module)

    @contextmanager
    def val_step_context(self):
        with torch.autocast('cuda', enabled=False):
            yield

    test_step_context = val_step_context
    predict_step_context = val_step_context


def semantic_digest(semantic):
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()


def validate_resume(checkpoint, semantic, allow_transition=False, allow_checkpoint_boundary_fix=False):
    saved = checkpoint['b2d_v1_fit']
    old = saved['semantic']
    if semantic_digest(old) != saved['contract_sha256']:
        raise ValueError('Parent checkpoint semantic SHA mismatch')
    if old == semantic:
        if allow_transition:
            raise ValueError('Precision transition requested for an unchanged contract')
        return None
    differences = {k for k in old.keys() | semantic.keys() if old.get(k) != semantic.get(k)}
    if (allow_transition and differences == {'precision', 'precision_policy', 'precision_policy_source_sha256'}
            and old['precision'] == '32-true' and 'precision_policy' not in old
            and semantic['precision'] == 'bf16-mixed' and semantic['precision_policy'] == POLICY
            and semantic['precision_policy_source_sha256'] == source_sha256()):
        return {'schema': 'explicit-precision-transition-v1', 'from': '32-true',
                'to': POLICY, 'parent_contract_sha256': saved['contract_sha256'],
                'child_contract_sha256': semantic_digest(semantic),
                'parent_global_step': checkpoint['global_step'],
                'preserve_optimizer_scheduler_rng_sampler': True}
    boundary_keys = {'execution_source_sha256', 'precision_policy_source_sha256'}
    if allow_checkpoint_boundary_fix and 'execution_source_sha256' in differences and differences <= boundary_keys:
        if not (old.get('precision') == semantic.get('precision') == 'bf16-mixed'
                and old.get('precision_policy') == semantic.get('precision_policy') == POLICY
                and semantic.get('precision_policy_source_sha256') == source_sha256()):
            raise ValueError('Checkpoint-boundary transition requires unchanged BF16 policy identity')
        old_sources = old.get('execution_source_sha256')
        new_sources = semantic.get('execution_source_sha256')
        if not isinstance(old_sources, dict) or not isinstance(new_sources, dict):
            raise ValueError('Checkpoint-boundary transition requires execution_source_sha256 maps')
        if set(old_sources) != set(new_sources):
            raise ValueError('Checkpoint-boundary transition changed the execution source set')
        changed = {key for key in old_sources if old_sources[key] != new_sources[key]}
        if changed != {'tools/training/diffusiondrive_v1/fit.py'}:
            raise ValueError('Checkpoint-boundary transition may change only fit.py source hash')
        return {'schema': 'explicit-checkpoint-boundary-fix-v1',
                'reason': 'defer ModelCheckpoint batch save until the accumulation window commits',
                'changed_sources': sorted(changed),
                'precision_policy_source_changed': old.get('precision_policy_source_sha256') != semantic.get('precision_policy_source_sha256'),
                'parent_contract_sha256': saved['contract_sha256'],
                'child_contract_sha256': semantic_digest(semantic),
                'parent_global_step': checkpoint['global_step'],
                'preserve_optimizer_scheduler_rng_sampler': True}
    raise ValueError('resume contract mismatch; supported transitions are explicit FP32 -> BF16 or checkpoint-boundary fit.py source fix')
