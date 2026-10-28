"""Weight-only initialization and explicit accumulation contract for V1 experiments.

The official model, optimizer, scheduler and loss implementations are untouched.
"""
import argparse
import json
from pathlib import Path
import re

from tools.training.diffusiondrive_v1.evaluate_checkpoint import stable_checkpoint


ACCUMULATION_SCHEMA = 'optimizer-boundary-commit-v1'


def accumulation_contract(config):
    count = config['training'].get('accumulate_grad_batches', 1)
    if type(count) is not int or count < 1:
        raise ValueError('accumulate_grad_batches must be a positive integer')
    if count == 1:
        return None  # Preserve the existing, frozen batch8 semantic contract.
    return {'schema': ACCUMULATION_SCHEMA, 'micro_batches': count,
            'effective_batch_size': count * config['batch_size'],
            'tail_rule': 'Lightning epoch-tail flush; each micro-loss divided by configured accumulation',
            'checkpoint_boundary': 'completed_optimizer_update_only'}


def training_budget_contract(config, horizon):
    training = config['training']
    if not training.get('freeze_budget'):
        return None
    if horizon != training['max_steps']:
        raise ValueError('Formal training horizon differs from frozen experiment budget')
    return {k: training[k] for k in ('max_steps', 'max_epochs', 'validation_every_epochs',
            'checkpoint_every_steps', 'checkpoint_monitor', 'health_checkpoint_keep')}


def prune_health_checkpoints(directory, keep):
    """Retain only this run's latest numbered diagnostics; never touch initial/best/foreign files."""
    if keep is None:
        return []
    if type(keep) is not int or keep < 1:
        raise ValueError('health_checkpoint_keep must be a positive integer')
    directory = Path(directory)
    paths = sorted(p for p in directory.glob('diagnostic-step*.ckpt')
                   if re.fullmatch(r'diagnostic-step\d{9}\.ckpt', p.name) and p.is_file() and not p.is_symlink())
    removed = []
    for path in paths[:-keep]:
        path.unlink()
        removed.append(str(path))
    return removed


def warm_start(agent, path):
    import torch
    checkpoint, identity = stable_checkpoint(path)
    state = checkpoint['state_dict']
    if not state or not all(k.startswith('agent.') for k in state):
        raise ValueError('Warm-start checkpoint has unexpected parameter prefixes')
    weights = {k[len('agent.'):]: v for k, v in state.items()}
    agent.load_state_dict(weights, strict=True)
    actual = agent.state_dict()
    if set(actual) != set(weights) or not all(torch.equal(actual[k].cpu(), v.cpu()) for k, v in weights.items()):
        raise RuntimeError('Warm-start parameters/buffers differ after strict loading')
    return dict(identity, schema='weight-only-warm-start-v1', strict=True,
                state_tensors=len(weights), optimizer_restored=False, scheduler_restored=False,
                sampler_restored=False, rng_restored=False, new_global_step=0)


class HealthStop(RuntimeError):
    """Controlled exit before an invalid optimizer update; resume last healthy checkpoint."""


def check_gradients(parameters):
    import torch
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
        raise HealthStop('Nonfinite/missing gradients: optimizer update was not executed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for path in sorted(args.output_dir.glob('smoke*/fit_result.json')):
        record = json.loads(path.read_text())
        print(json.dumps({'run': str(path.parent), **{k: record.get(k) for k in
              ('status', 'global_step', 'accumulation', 'warm_start', 'resume', 'last_checkpoint')}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
