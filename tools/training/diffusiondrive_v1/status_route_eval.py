"""Current read-only Status evaluation entry.

The public CLI delegates to ``status_route_eval_core`` and evaluates one or more
Status checkpoints on the frozen Validation split with FP32, target-free forward
calls and the existing trajectory/agent/BEV metric primitives.  The legacy
route-only implementation is retained below as ``legacy_main`` for historical
reproduction only; it is not the current entry point.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def legacy_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--base-config', type=Path,
                        default=ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261023_status_training/full_bf16_bs16_val250_config.json')
    parser.add_argument('--per-route', type=int, default=80)
    parser.add_argument('--routes', default='')
    parser.add_argument('--all-routes', action='store_true',
                        help='score every validation route (full validation split)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.training.diffusiondrive_v1.experimental_status_reader import ExperimentalStatusDataset
    from tools.training.diffusiondrive_v1.bf16_policy import install_fp32_losses
    from tools.training.diffusiondrive_v1.runtime import model_configuration
    from tools.training.diffusiondrive_v1.status_route_eval_core import load_weights_only
    from tools.b2d_adapter.v1_builders import official_v1_module

    config = json.loads(args.base_config.read_text())
    manifest_path = Path(config['dataset_manifest'])
    index_path = Path(config['sample_index'])
    dataset = ExperimentalStatusDataset(manifest_path, index_path, config['base_config'], 'validation',
                                        limit=0)
    rows = dataset.rows
    all_routes = []
    for row in rows:
        if row['route'] not in all_routes:
            all_routes.append(row['route'])
    # Routes covered by the fixed 1,024-row monitoring subset are excluded here.
    seen, subset_routes = set(), set()
    for row in rows[:1024]:
        if row['route'] not in seen:
            seen.add(row['route'])
            subset_routes.add(row['route'])
    if args.routes:
        chosen = [route for route in args.routes.split(',') if route]
    elif args.all_routes:
        chosen = list(all_routes)
    else:
        chosen = []
        for route in all_routes:
            if route in subset_routes:
                continue
            count = sum(1 for row in rows if row['route'] == route)
            if count >= args.per_route:
                chosen.append(route)
            if len(chosen) == 3:
                break

    base = json.loads(Path(config['base_config']).read_text())
    agent = official_v1_module('transfuser_agent').TransfuserAgent(
        model_configuration(base), config['learning_rate'], checkpoint_path=None)
    warm = load_weights_only(agent, args.checkpoint)
    install_fp32_losses(agent)
    agent.eval().cuda()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for route in chosen:
        positions = [i for i, row in enumerate(rows) if row['route'] == route]
        take = np.linspace(0, len(positions) - 1, min(args.per_route, len(positions))).round().astype(int)
        for pick in take:
            index = positions[int(pick)]
            features, targets = dataset[index]
            batch = {key: value.unsqueeze(0).cuda() for key, value in features.items()}
            with torch.no_grad():
                prediction = agent.forward(batch)
            predicted = prediction['trajectory'][0].float().cpu()
            truth = targets['trajectory'].float().cpu()
            error = torch.linalg.norm(predicted[:, :2] - truth[:, :2], dim=-1)
            heading = torch.atan2(torch.sin(predicted[-1, 2] - truth[-1, 2]),
                                  torch.cos(predicted[-1, 2] - truth[-1, 2]))
            status = rows[index]['status8']
            command = ('Left', 'Straight', 'Right', 'Unknown')[int(np.argmax(status[:4]))]
            records.append({'route': route, 'sample_id': rows[index]['sample_id'], 'command': command,
                            'ade_m': float(error.mean()), 'fde_m': float(error[-1]),
                            'heading_error_deg': float(abs(heading) * 180.0 / np.pi),
                            'prediction': [list(map(float, p)) for p in predicted.tolist()],
                            'target': [list(map(float, p)) for p in truth.tolist()]})

    with (output / 'samples.jsonl').open('w') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')

    def mean(key, subset):
        return float(np.mean([row[key] for row in subset])) if subset else float('nan')

    summary = []
    groups = [('all', records)]
    groups += [(route, [row for row in records if row['route'] == route]) for route in chosen]
    groups += [(command, [row for row in records if row['command'] == command])
               for command in ('Left', 'Straight', 'Right')]
    for name, subset in groups:
        if subset:
            summary.append({'group': name, 'n': len(subset), 'ade_m': mean('ade_m', subset),
                            'fde_m': mean('fde_m', subset),
                            'heading_error_deg': mean('heading_error_deg', subset)})
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group', 'n', 'ade_m', 'fde_m', 'heading_error_deg'])
        writer.writeheader()
        writer.writerows(summary)

    # Plot the last six samples (evenly across routes) with GT/prediction.
    picks = np.linspace(0, len(records) - 1, min(6, len(records))).round().astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, pick in zip(axes.flat, picks):
        record = records[int(pick)]
        target = np.array(record['target'])[:, :2]
        predicted = np.array(record['prediction'])[:, :2]
        for xy, label, color, style in ((target, 'GT', 'tab:green', 'o-'),
                                        (predicted, 'Prediction', 'tab:red', '.-')):
            ax.plot(np.r_[0, xy[:, 1]], np.r_[0, xy[:, 0]], style, color=color, label=label, markersize=3)
        ax.scatter([0], [0], marker='s', s=20, color='black')
        lateral = max(2.0, 1.12 * float(np.nanmax(np.abs(np.r_[target[:, 1], predicted[:, 1]]))))
        ax.set_xlim(lateral, -lateral)
        ax.set_ylim(min(-0.5, float(np.nanmin(np.r_[target[:, 0], predicted[:, 0]])) - 0.5),
                    max(3.0, float(np.nanmax(np.r_[target[:, 0], predicted[:, 0]])) * 1.1))
        ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.25)
        ax.set_title(f"{record['route'].split('_Town')[0][:30]} f{record['sample_id'].split(':')[-1]}\n"
                     f"{record['command']} | ADE {record['ade_m']:.2f} FDE {record['fde_m']:.2f} m",
                     fontsize=8)
        ax.set_xlabel('Left y (m)'); ax.set_ylabel('Forward x (m)')
    for ax in axes.flat[len(picks):]:
        ax.axis('off')
    fig.savefig(output / 'route_examples.png', dpi=160)
    plt.close(fig)

    print(json.dumps({'checkpoint': str(args.checkpoint), 'warm_start': warm['sha256'],
                      'routes': chosen, 'samples': len(records), 'output': str(output),
                      'summary': summary}, indent=2))


def main():
    """Current Route-split evaluation CLI."""
    from tools.training.diffusiondrive_v1.status_route_eval_core import main as core_main
    return core_main()


if __name__ == '__main__':
    main()
