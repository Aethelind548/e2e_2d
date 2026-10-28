"""Render existing verified seven-class targets; never rebuild labels or train."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')

COLORS = ['#eeeeee', '#666666', '#89cce0', '#ffdf00', '#eb5757', '#5268f5', '#40c97b']
NAMES = ['背景', '道路', '步行道', '车道中心线', '静态障碍物', '车辆', '行人']
SELECTION = [
    ('Town01', 'DynamicObjectCrossing', 81),
    ('Town02', 'ControlLoss', 80),
    ('Town04', 'BlockedIntersection', 80),
    ('Town05', 'PedestrianCrossing', 40),
    ('Town06', 'HighwayExit', 80),
    ('Town07', 'VanillaSignalizedTurnEncounterRedLight', 80),
    ('Town10HD', 'OppositeVehicleRunningRedLight', 80),
    ('Town12', 'Accident', 80),
    ('Town13', 'ConstructionObstacleTwoWays', 80),
    ('Town15', 'HighwayCutIn', 80),
]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from matplotlib import font_manager

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/m4/bev_examples')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    # Restrict visualization writes to analysis; avoid user-supplied training/cache destinations.
    if ROOT / 'analysis' not in out.parents:
        raise ValueError('visualization output must be inside analysis')
    out.mkdir(parents=True, exist_ok=True)
    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'font.family': font_name, 'axes.unicode_minus': False,
                         'font.size': 10, 'savefig.facecolor': 'white'})
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-.5, 7.5), 7)
    handles = [Patch(facecolor=c, edgecolor='#777777', label=f'{i} {name}') for i, (c, name) in enumerate(zip(COLORS, NAMES))]
    sources = {}
    generated = []

    def protect(path):
        path = Path(path)
        sources[str(path)] = file_sha(path)
        return path

    def load_labels(path):
        path = protect(path)
        with np.load(path, allow_pickle=False) as archive:
            labels = archive['labels'].copy()
            if 'metadata' in archive:
                metadata = json.loads(str(archive['metadata'].item()))
                if hashlib.sha256(labels.tobytes()).hexdigest() != metadata['labels_sha256']:
                    raise ValueError(f'Label checksum mismatch: {path}')
        check(labels)
        return labels

    def check(labels):
        if labels.shape != (128, 256) or labels.dtype not in (np.dtype('uint8'), np.dtype('int64')) or not ((labels >= 0) & (labels < 7)).all():
            raise ValueError('Expected numeric seven-class 128x256 labels')

    def frame(ax, labels, title, roi=None):
        # Stored rows increase with forward x; columns increase with left y.
        # Display forward upward and positive left on the left side of the page.
        ax.imshow(labels[:, ::-1], origin='lower', extent=(32, -32, 0, 32),
                  cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel('横向距离 / m（左为正）', fontsize=9)
        ax.set_ylabel('前向距离 / m', fontsize=9)
        ax.tick_params(labelsize=8)
        if roi:
            xmin, xmax, ymin, ymax = roi
            ax.set_xlim(ymax, ymin)
            ax.set_ylim(xmin, xmax)
        else:
            ax.set_xlim(32, -32)
            ax.set_ylim(0, 32)
            ax.set_xticks([32, 16, 0, -16, -32])
            ax.set_yticks([0, 8, 16, 24, 32])
            ax.annotate('自车参考点', xy=(0, 0), xytext=(0, 4.2), ha='center', fontsize=8,
                        arrowprops={'arrowstyle': '->', 'color': '#111111'}, color='#111111')

    def finish(fig, name, footer='真实监督标签；非模型预测。显示采用最近邻，数值标签未修改。'):
        fig.legend(handles=handles, loc='lower center', ncol=7, bbox_to_anchor=(.5, .026), fontsize=9)
        fig.text(.5, .006, footer, ha='center', fontsize=9, color='#555555')
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        generated.append(name)

    config = json.loads(protect(ROOT / 'tools/training/diffusiondrive_v1/m2_subset.json').read_text())
    manifest = json.loads(protect(ROOT / config['prepared_manifest']).read_text())
    rows = [row for split in manifest['accepted'].values() for row in split]
    samples = []
    for town, scenario, frame_id in SELECTION:
        row = next(row for row in rows if f'/{scenario}_{town}_' in row['token'] and row['token'].endswith(f'::{frame_id}'))
        cache = ROOT / config['bev_cache_dir'] / (row['provenance']['targets']['bev']['cache_key'] + '.npz')
        labels = load_labels(cache)
        counts = np.bincount(labels.ravel(), minlength=7).tolist()
        name = f'{town}_{scenario}_{frame_id}.png'
        fig, ax = plt.subplots(figsize=(10.2, 6.3))
        title = f'{town} · {scenario}\n帧 {frame_id} | 原版七类标签'
        frame(ax, labels, title)
        fig.subplots_adjust(left=.09, right=.97, bottom=.18, top=.83)
        fig.text(.5, .115, '像素数：' + '  '.join(f'{i}={n}' for i, n in enumerate(counts)), ha='center', fontsize=9)
        finish(fig, name)
        samples.append({'token': row['token'], 'town': town, 'scenario': scenario, 'frame_id': frame_id,
                        'image': name, 'labels_source': str(cache), 'class_pixel_counts': counts,
                        'labels': labels})

    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    for ax, sample in zip(axes.flat, samples):
        frame(ax, sample['labels'], f"{sample['town']} · {sample['scenario']}\n帧 {sample['frame_id']}  静态={sample['class_pixel_counts'][4]}px  车辆={sample['class_pixel_counts'][5]}px  行人={sample['class_pixel_counts'][6]}px")
    fig.suptitle('DiffusionDrive V1 · 10 个真实样本的七类 BEV 监督标签', fontsize=17, y=.986)
    fig.subplots_adjust(left=.06, right=.98, top=.953, bottom=.07, hspace=.55, wspace=.12)
    finish(fig, 'overview_10_real_frames.png')

    for town, target_class in [('Town01', 6), ('Town05', 6), ('Town13', 4)]:
        sample = next(sample for sample in samples if sample['town'] == town)
        labels = sample['labels']
        rr, cc = np.nonzero(labels == target_class)
        if not len(rr):
            raise ValueError(f'Expected class {target_class} for selected real sample')
        # Actual pixel ROI; nearest-neighbor zoom keeps the small 0.25m cells visible.
        pad = 16
        r0, r1 = max(0, rr.min()-pad), min(128, rr.max()+pad+1)
        c0, c1 = max(0, cc.min()-pad), min(256, cc.max()+pad+1)
        roi = (r0*.25, r1*.25, c0*.25-32, c1*.25-32)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
        frame(axes[0], labels, f'{town} · 完整前向范围')
        frame(axes[1], labels, f'类别 {target_class} {NAMES[target_class]} · 真实像素局部放大', roi)
        fig.subplots_adjust(left=.06, right=.97, bottom=.18, top=.85, wspace=.2)
        finish(fig, f'{town}_class{target_class}_zoom.png')

    # Two recorded-frame original/enriched comparisons, read only from prepared artifacts.
    import torch
    enriched_config = json.loads(protect(ROOT / 'tools/training/diffusiondrive_v1/m4_subset.json').read_text())
    enriched_manifest = json.loads(protect(ROOT / enriched_config['prepared_manifest']).read_text())
    comparisons = []
    for entries in enriched_manifest['accepted'].values():
        for row in entries:
            original_row = next(prior for prior in rows if prior['token'] == row['token'])
            original = load_labels(ROOT / config['bev_cache_dir'] / (original_row['provenance']['targets']['bev']['cache_key'] + '.npz'))
            artifact = protect(row['tensor_path'])
            if file_sha(artifact) != row['tensor_sha256']:
                raise ValueError('Prepared artifact checksum mismatch')
            payload = torch.load(artifact, map_location='cpu', weights_only=True)
            enhanced = payload['targets']['bev_semantic_map'].numpy()
            check(enhanced)
            town = re.search(r'_(Town\d+(?:HD)?)_', row['token']).group(1)
            changed = int(np.count_nonzero(original != enhanced))
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            frame(axes[0], original, f'{town} · 真实帧原版标签')
            frame(axes[1], enhanced, f'{town} · M4 增强标签\n变化像素={changed}')
            fig.subplots_adjust(left=.06, right=.98, bottom=.18, top=.83, wspace=.2)
            name = f'{town}_recorded_original_vs_enriched.png'
            finish(fig, name, '真实录制帧；变化像素为 0 表示该位置未触发增强，不代表额外生成了护栏。')
            comparisons.append({'token': row['token'], 'image': name, 'changed_pixels': changed})

    # Verified actual-map guardrail queries, separate from recorded actor samples.
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for i, town in enumerate(('Town12', 'Town13')):
        path = protect(ROOT / f'analysis/diffusiondrive_b2d_architecture/v1_validation/m4/{town}_GuardRail_labels.npz')
        with np.load(path, allow_pickle=False) as archive:
            original, enhanced = archive['original'].copy(), archive['enriched'].copy()
        check(original); check(enhanced)
        changed = original != enhanced
        rr, cc = np.nonzero(changed)
        roi = (max(0, rr.min()-12)*.25, min(128, rr.max()+13)*.25,
               max(0, cc.min()-12)*.25-32, min(256, cc.max()+13)*.25-32)
        frame(axes[i, 0], original, f'{town} · 护栏真实地图位置 · 原版')
        frame(axes[i, 1], enhanced, f'M4 增强 · 新增 Static {int(changed.sum())} 像素')
        frame(axes[i, 2], enhanced, '新增护栏表面 · 局部放大', roi)
    fig.subplots_adjust(left=.04, right=.99, bottom=.115, top=.95, hspace=.38, wspace=.17)
    finish(fig, 'guardrail_original_enriched_zoom.png', '真实地图几何 + 构造的查询位姿；不是 B2D 录制帧。仅显示已验证的护栏表面栅格。')

    if any(file_sha(path) != prior for path, prior in sources.items()):
        raise RuntimeError('A source changed during visualization')
    report = {'status': 'pass', 'type': 'supervision_label_visualizations_not_model_predictions',
              'image_count': len(generated), 'images': generated,
              'orientation': 'forward up; positive-left on left; origin at ego rear reference; 0.25m pixels',
              'classes': [{'id': i, 'name': name, 'color': color} for i, (name, color) in enumerate(zip(NAMES, COLORS))],
              'samples': [{k: v for k, v in sample.items() if k != 'labels'} for sample in samples],
              'recorded_comparisons': comparisons, 'protected_sources_sha256': sources,
              'training_executed': False, 'labels_modified': False}
    (out / 'index.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (out / 'README.md').write_text('# BEV 标签可视化\n\n这些是监督标签，不是模型预测。显示前方朝上、左侧为正，使用原始 0.25m 栅格的最近邻显示。\n\n'
        + '\n'.join(f'- [{name}]({name})' for name in generated) + '\n\n帧 token、类别像素数、来源校验和见 index.json。原版缓存与增强缓存均只读。\n')
    print(json.dumps({'status': 'pass', 'images': len(generated), 'output_dir': str(out),
                      'recorded_comparisons': comparisons}, ensure_ascii=False))


if __name__ == '__main__':
    main()
