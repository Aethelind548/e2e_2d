"""Fixed-cache, target-free V1 health diagnostics with observational hooks only.

Reuses the checkpoint evaluator's metrics and the approved attribution panel.
Engineering alarms are not paper acceptance standards. No loss enters training.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from tools.training.diffusiondrive_v1.evaluate_checkpoint import detection_counts, sha256, write_json


def alarms(current, baseline, previous):
    v, b = current['validation'], baseline['validation']
    warnings = []
    if v['query_std'] < min(.02, b['query_std'] * .1):
        warnings.append('agent_query_collapse')
    if v['qk_std'] > max(100., b['qk_std'] * 4) and v['attention_entropy'] < .05:
        warnings.append('saturated_cross_attention')
    degraded = v['ade_m'] > b['ade_m'] * 1.3 + .3 or v['fde_m'] > b['fde_m'] * 1.3 + .5
    if degraded:
        warnings.append('validation_degradation')
    sustained = degraded and bool(previous) and 'validation_degradation' in previous.get('warnings', [])
    growing = bool(previous) and v['qk_std'] > max(100., previous['validation']['qk_std'] * 2)
    if growing:
        warnings.append('rapid_qk_growth')
    stop = any(x in warnings for x in ['agent_query_collapse', 'saturated_cross_attention']) or sustained
    if growing and 'rapid_qk_growth' in previous.get('warnings', []):
        stop = True
    return warnings, stop


class HealthMonitor:
    def __init__(self, agent, datasets, config, output):
        self.agent, self.output = agent, Path(output)
        self.spec = config['health_monitor']; self.interval = self.spec['every_optimizer_steps']
        source = Path(self.spec['panel_source'])
        if sha256(source) != self.spec['panel_source_sha256']:
            raise ValueError('Health panel provenance SHA mismatch')
        lookup = {split: {row['sample_id']: i for i, row in enumerate(ds.rows)} for split, ds in datasets.items()}
        self.panel = []
        for line in source.open():
            row = json.loads(line)
            if row.get('attempt') == 2 and row['condition'] == 'best':
                split = row['split']; index = lookup[split][row['sample_id']]
                pair = datasets[split].rows[index]
                if pair.get('feature_sha', pair.get('feature_row', {}).get('sha256')) != row['feature_sha']:
                    raise ValueError('Health feature provenance mismatch')
                if pair.get('target_sha', pair.get('target_row', {}).get('sha256')) != row['target_sha']:
                    raise ValueError('Health target provenance mismatch')
                self.panel.append((row, datasets[split], index))
        if len(self.panel) != 64 or len({r['sample_id'] for r, _, _ in self.panel}) != 64:
            raise ValueError('Expected the approved 32 train + 32 validation attribution panel')
        self.history = []
        self.output.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint = None

    def state_dict(self):
        return {'history': self.history, 'last_checkpoint': self.last_checkpoint}

    def load_state_dict(self, state):
        self.history = state['history']; self.last_checkpoint = state.get('last_checkpoint')

    def run(self, step):
        network = self.agent._transfuser_model
        observations = {}
        def query_hook(module, inputs, output):
            observations['query_std'] = float(output[:, 1:].float().std(dim=1).mean())
        def attention_hook(module, inputs):
            query, key = inputs[:2]; d = module.embed_dim; h = module.num_heads
            w, b = module.in_proj_weight, module.in_proj_bias
            q = torch.nn.functional.linear(query, w[:d], None if b is None else b[:d])
            k = torch.nn.functional.linear(key, w[d:2*d], None if b is None else b[d:2*d])
            q = q.reshape(q.shape[0], q.shape[1], h, d//h).transpose(1, 2)
            k = k.reshape(k.shape[0], k.shape[1], h, d//h).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) / (d//h)**.5
            probabilities = scores.softmax(-1)
            observations.update(qk_std=float(scores.std()), attention_entropy=float(
                (-(probabilities[:, :, 1:] * probabilities[:, :, 1:].clamp_min(1e-30).log()).sum(-1)).mean()),
                bev_memory_std=float(key[:, :-1].std()),
                ego_status_attention=float(probabilities[:, :, 0, -1].mean()),
                q_projection_rms=float(w[:d].square().mean().sqrt()),
                k_projection_rms=float(w[d:2*d].square().mean().sqrt()))
        decoder = network._tf_decoder.layers[2]
        handles = [decoder.register_forward_hook(query_hook), decoder.multihead_attn.register_forward_pre_hook(attention_hook)]
        flags = [(m, m.training) for m in self.agent.modules()]
        versions = {n: t._version for n, t in list(self.agent.named_parameters()) + list(self.agent.named_buffers())}
        grad_versions = [(p.grad, p.grad._version) for p in self.agent.parameters() if p.grad is not None]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        rows = []
        device = next(self.agent.parameters()).device
        try:
            self.agent.eval()
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []), torch.inference_mode(), torch.autocast(device.type, enabled=False):
                for source, dataset, index in self.panel:
                    features, targets = dataset[index]
                    f = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
                    t = {k: v.unsqueeze(0).to(device) for k, v in targets.items()}
                    seed = source['seed']; torch.manual_seed(seed)
                    pred = self.agent.forward(f)
                    if not all(torch.isfinite(v).all().item() for v in pred.values() if isinstance(v, torch.Tensor)):
                        raise FloatingPointError('Nonfinite target-free health prediction')
                    measured = dict(observations)
                    losses = self.agent.compute_loss(f, t, pred)
                    if not all(torch.isfinite(torch.as_tensor(v)).all().item() for v in losses.values() if v is not None):
                        raise FloatingPointError('Nonfinite official health evaluation loss')
                    error = torch.linalg.norm(pred['trajectory'][0, :, :2] - t['trajectory'][0, :, :2], dim=-1)
                    xy = pred['trajectory'][0, :, :2]
                    travel = float(torch.linalg.norm(torch.diff(torch.cat([xy.new_zeros(1, 2), xy]), dim=0), dim=-1).sum())
                    det = detection_counts(pred['agent_states'][0].cpu().numpy(), pred['agent_labels'][0].cpu().numpy(),
                        t['agent_states'][0].cpu().numpy(), t['agent_labels'][0].cpu().numpy())
                    f2 = dict(f); f2['status_feature'] = f['status_feature'].clone(); f2['status_feature'][:, 4] += 5
                    torch.manual_seed(seed); alternative = self.agent.forward(f2)['trajectory']
                    if not torch.isfinite(alternative).all():raise FloatingPointError('Nonfinite status-sensitivity prediction')
                    sensitivity = float(torch.linalg.norm(alternative[..., :2] - pred['trajectory'][..., :2], dim=-1).mean())
                    rows.append(dict(measured, split=source['split'], group=source['group'], sample_id=source['sample_id'],
                        feature_sha=source['feature_sha'], target_sha=source['target_sha'],
                        ade_m=float(error.mean()), fde_m=float(error[-1]), predicted_travel_m=travel,
                        stationary_moved=source['group'] == 'stationary' and travel > 1.,
                        velocity_sensitivity_m=sensitivity, detection=det,
                        official_target_free_eval_loss=float(losses['loss'])))
        finally:
            for handle in handles: handle.remove()
            for module, flag in flags: module.training = flag
            random.setstate(python_rng); np.random.set_state(numpy_rng)
        assert versions == {n: t._version for n, t in list(self.agent.named_parameters()) + list(self.agent.named_buffers())}
        assert all(t._version == version for t, version in grad_versions)
        result = {'global_step': step, 'samples': len(rows), 'sample_ids': [r['sample_id'] for r in rows],
                  'warnings': [], 'stop': False, 'scope': 'Stratified diagnostic panel; contains four historical validation hard cases; not population metrics'}
        for split in ['train', 'validation']:
            rs = [r for r in rows if r['split'] == split]
            numeric = ['ade_m', 'fde_m', 'qk_std', 'attention_entropy', 'query_std', 'bev_memory_std',
                       'ego_status_attention', 'q_projection_rms', 'k_projection_rms', 'velocity_sensitivity_m', 'official_target_free_eval_loss']
            summary = {k: float(np.mean([r[k] for r in rs])) for k in numeric}
            det = Counter(); [det.update(r['detection']) for r in rs]
            denominator = 2*det['tp'] + det['fp'] + det['fn']
            summary['detection_f1'] = 2*det['tp']/denominator if denominator else None
            summary['groups'] = {}
            for group in ['stationary', 'low_creep', 'low_depart', 'moving']:
                gs = [r for r in rs if r['group'] == group]
                summary['groups'][group] = {'samples': len(gs), **{k:float(np.mean([r[k] for r in gs])) for k in ['ade_m', 'fde_m']}}
            stops = [r for r in rs if r['group'] == 'stationary']
            summary['stationary_moved_rate'] = sum(r['stationary_moved'] for r in stops)/len(stops)
            result[split] = summary
        if self.history:
            result['warnings'], result['stop'] = alarms(result, self.history[0], self.history[-1])
        self.history.append(result)
        write_json(self.output/f'step{step:09d}.json', {'summary': result, 'rows': rows})
        with (self.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
        write_json(self.output/'summary.json', {'history': self.history, 'motion_alarm': 'GT stationary path<=.5m, predicted path>1m',
                   'noise_and_model_rng_preserved': True, 'optimizer_or_gradient_updates': 0})
        self.plot()
        random.setstate(python_rng);np.random.set_state(numpy_rng)
        return result

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
        for ax, key in zip(axes.flat, ['ade_m', 'fde_m', 'qk_std', 'attention_entropy', 'query_std',
                                      'velocity_sensitivity_m', 'stationary_moved_rate', 'detection_f1', 'official_target_free_eval_loss']):
            for split in ['train', 'validation']:
                ax.plot([r['global_step'] for r in self.history], [r[split][key] for r in self.history], 'o-', label=split)
            ax.set_title(key); ax.set_xlabel('Optimizer step'); ax.grid(alpha=.3)
        axes[0, 0].legend(); fig.savefig(self.output/'health_curves.png', dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for path in sorted(args.output_dir.glob('**/health/summary.json')):
        data = json.loads(path.read_text())
        print(json.dumps({'path': str(path), 'history': data['history']}, ensure_ascii=False))


if __name__ == '__main__': main()
