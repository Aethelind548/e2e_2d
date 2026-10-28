"""Read-only dataset inventory for Route Preview V1.1 full generation.

Never writes to the source or derives a training Manifest.  If the source
mount is unavailable, it records the exact failure and still reports the
frozen Manifest reference counts so the blocker is auditable.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / 'outputs/diffusiondrive_v1/scratch_attribution/20261022_142557_panel/behavior_panel.json'
SOURCE = Path('/data/datasets/Bench2Drive-V0.0.4/v1')
TARGET = Path('/data/datasets/derived/Bench2Drive-V0.0.4/route_preview_v1')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return default if default is not None else {'error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--target', type=Path, default=TARGET)
    parser.add_argument('--panel', type=Path, default=PANEL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_error = None
    route_names = []
    try:
        entries = list(os.scandir(args.source))
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    route_names.append(entry.name)
            except OSError as exc:
                source_error = f'{entry.path}: {exc}'
        source_error = source_error or None
    except OSError as exc:
        source_error = f'{args.source}: {exc}'
    panel = read_json(args.panel, {})
    manifest_path = panel.get('manifest')
    manifest = read_json(manifest_path, {}) if manifest_path else {}
    usage = shutil.disk_usage(Path('/'))
    try:
        target_exists = args.target.exists()
        target_error = None
    except OSError as exc:
        target_exists = None
        target_error = str(exc)
    try:
        parent_exists = args.target.parent.exists()
    except OSError:
        parent_exists = None
    target = {
        'path': str(args.target),
        'exists': target_exists,
        'exists_error': target_error,
        'parent': str(args.target.parent),
        'parent_exists': parent_exists,
        'write_attempted': False,
    }
    payload = {
        'schema': 'route-preview-v11-dataset-inventory-v1',
        'source_root': str(args.source),
        'read_only': True,
        'source_scan': {
            'status': 'FAIL' if source_error else 'PASS',
            'error': source_error,
            'observed_route_dirs': len(route_names),
            'route_names_sample': route_names[:10],
        },
        'panel': {
            'path': str(args.panel),
            'routes': len({row['route'] for split in ('validation', 'train')
                           for row in panel.get(split, [])}),
            'frames': sum(len(panel.get(split, [])) for split in ('validation', 'train')),
            'split_counts': {split: len(panel.get(split, [])) for split in ('validation', 'train')},
        },
        'frozen_manifest': {
            'path': manifest_path,
            'schema': manifest.get('schema'),
            'counts': manifest.get('counts'),
            'split_counts': manifest.get('split_counts'),
            'route_split_counts': manifest.get('route_split_counts'),
            'unique_target_frames': manifest.get('target_complete_unique'),
            'features_complete': manifest.get('features_complete'),
            'route_total': (manifest.get('route_split_counts', {}).get('train', 0)
                            + manifest.get('route_split_counts', {}).get('validation', 0)),
        },
        'storage': {
            'device': '/',
            'total_bytes': usage.total,
            'used_bytes': usage.used,
            'free_bytes': usage.free,
            'budget_bytes': 2 * 1024 ** 3,
            'budget_fits': usage.free > 2 * 1024 ** 3,
        },
        'target': target,
        'blocking_reasons': ([source_error] if source_error else []) + [
            'No raw source read is possible while the source mount returns I/O errors.',
            'Full generation was intentionally not started; no target data was written.',
        ] if source_error else [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'event': 'inventory_done', 'source_status': payload['source_scan']['status'],
                      'routes_observed': payload['source_scan']['observed_route_dirs'],
                      'manifest_frames': payload['frozen_manifest']['unique_target_frames'],
                      'free_bytes': payload['storage']['free_bytes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
