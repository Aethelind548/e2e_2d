"""Verify the real Camera Builder and calibrated overlap without lost Pilot caches."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from tools.b2d_adapter import FrameReader, RouteIndex
from tools.b2d_adapter.native_builders import B2DObservation
from tools.b2d_adapter.v1_builders import DiffusionDriveV1FeatureBuilder, v1_native_config
from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration
from tools.validation.b2d_adapter.audit_v1_camera_alignment import geometry, signed_overlap, NAMES, stats


def main():
    torch.set_num_threads(4)
    config = json.loads((ROOT / 'tools/training/diffusiondrive_v1/camera_proportional_check.json').read_text())
    model = model_configuration(config)
    native = native_configuration(config)
    legacy = v1_native_config('navsim-pixel-crop-v1')
    assert native.camera_policy == 'b2d-proportional-crop-v1'
    for name, expected in [('pilot_proportional.json', native.camera_policy),
                           ('pilot_original.json', legacy.camera_policy)]:
        cfg = json.loads((ROOT / 'tools/training/diffusiondrive_v1' / name).read_text())
        assert native_configuration(cfg).camera_policy == expected
    out = ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/camera_crop_fix'
    out.mkdir(exist_ok=True)
    report = {'scope': 'real Camera Builder only; not full Dataset/model/loss acceptance',
              'camera_policy': native.camera_policy, 'frames': []}
    overview, axes = plt.subplots(5, 2, figsize=(16, 12), constrained_layout=True)
    for i, entry in enumerate(e for entries in config['data'].values() for e in entries):
        reader = FrameReader(RouteIndex(entry['route']))
        raw = reader.read(reader.index.ref(entry['frames'][0]), cameras=True)
        builders = [DiffusionDriveV1FeatureBuilder(reader, c, model) for c in (legacy, native)]
        results = [b.build_camera(B2DObservation(raw, None, c.provenance()['policy_sha256']))
                   for b, c in zip(builders, (legacy, native))]
        old, new = [r.tensors['camera_feature'] for r in results]
        assert tuple(new.shape) == (3, 256, 1024) and new.dtype == torch.float32
        assert torch.isfinite(new).all() and 0 <= new.min() <= new.max() <= 1
        second = reader.read(raw.ref, cameras=True)
        repeated = builders[1].build_camera(B2DObservation(second, None, native.provenance()['policy_sha256']))
        assert torch.equal(new, repeated.tensors['camera_feature'])
        images = [raw.cameras[name] for name in NAMES]
        for name, image in zip(NAMES, images):
            assert image.shape == (900, 1600, 3) and image.dtype == np.uint8
            assert np.array_equal(image, cv2.imread(str(reader.index.path(raw.ref, name)))[:, :, ::-1])
        parts = [a[23:-23, 346:-346] if j != 1 else a[23:-23, :] for j, a in enumerate(images)]
        stitched = np.concatenate(parts, axis=1)
        assert stitched.shape == (854, 3416, 3)
        assert np.array_equal(new.permute(1, 2, 0).numpy(),
                              cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255)
        folder = out / (Path(entry['route']).name.split('_Weather')[0] + '_frame00080')
        folder.mkdir(exist_ok=True)
        old_evidence = folder.parent.parent / 'camera_alignment' / folder.name / 'camera_tensor_evidence.npz'
        with np.load(old_evidence, allow_pickle=False) as saved:
            assert np.array_equal(old.numpy(), saved['current_camera_feature'])
        calibrated = {}
        for name in NAMES:
            previous = geometry(raw.cameras, raw.sensors[name], raw.ego_box, name)
            current = geometry(raw.cameras, raw.sensors[name], raw.ego_box, name, 346, 23)
            calibrated[name] = {'intrinsic': current['intrinsic'],
                                'camera_to_ego_carla': current['camera_to_ego_carla'],
                                'bearing_deg': current['relative_target_bearing_deg'],
                                'old_interval_deg': previous['current_angle_interval_target_deg'],
                                'new_interval_deg': current['current_angle_interval_target_deg']}
        overlaps = {policy: [signed_overlap(calibrated[a][policy + '_interval_deg'],
                                            calibrated[b][policy + '_interval_deg'])
                              for a, b in zip(NAMES[:-1], NAMES[1:])] for policy in ('old', 'new')}
        assert all(x < 0 for x in overlaps['old']) and all(x > 1.6 for x in overlaps['new'])
        for j, tensor in enumerate((old, new)):
            rgb = np.rint(tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(rgb).save(folder / ('old_input.png' if j == 0 else 'new_input.png'))
            axes[i, j].imshow(rgb)
            axes[i, j].axis('off')
            axes[i, j].set_title(folder.name + (' | old 28/416' if j == 0 else ' | new 23/346'), fontsize=9)
        Image.fromarray(stitched).save(folder / 'new_stitched.png')
        report['frames'].append({'token': raw.ref.token, 'sources': raw.provenance['source_files'],
                                 'crop_wh': [[a.shape[1], a.shape[0]] for a in parts],
                                 'stitched_wh': [3416, 854], 'aspect': 4,
                                 'scale_x': 1024 / 3416, 'scale_y': 256 / 854,
                                 'tensor_stats': stats(new.permute(1, 2, 0).numpy()),
                                 'tensor_shape': list(new.shape), 'calibration': calibrated,
                                 'adjacent_overlap_deg': overlaps,
                                 'legacy_tensor_matches_saved_audit': True,
                                 'rgb_verified': True, 'fresh_decode_reproducible': True})
    overview.savefig(out / 'overview_old_vs_new.png', dpi=150)
    plt.close(overview)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for policy, offset, color in [('old', 0, '#d45b42'), ('new', .3, '#238d75')]:
        for j, name in enumerate(NAMES):
            lo, hi = report['frames'][0]['calibration'][name][policy + '_interval_deg']
            ax.plot([lo, hi], [j + offset, j + offset], lw=7, color=color,
                    label=policy if j == 0 else None)
            ax.text((lo + hi) / 2, j + offset + .06, f'{lo:.3f} to {hi:.3f}', ha='center', fontsize=8)
    ax.set_yticks([.15, 1.15, 2.15], NAMES)
    ax.set_xlabel('Ego bearing (degrees, positive = left); pinhole far-field estimate')
    ax.set_xlim(-80, 80)
    ax.legend()
    fig.savefig(out / 'horizontal_coverage_old_vs_new.png', dpi=150)
    plt.close(fig)
    report.update(status='pass', limitations=['Pinhole far-field estimate; annotation lacks lens distortion.',
                                            'Near-field parallax remains.',
                                            'Historical Pilot caches/checkpoints missing; full compatibility validation pending.'])
    (out / 'camera_views_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'pass', 'frames': len(report['frames']),
                      'new_overlap_deg': [r['adjacent_overlap_deg']['new'] for r in report['frames']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
