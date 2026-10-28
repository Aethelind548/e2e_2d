"""Independent, versioned Status-only fine-tune entry (no fit.py changes).

The frozen production entry ``fit.py`` stays byte-identical.  This module is the
only place that may wrap the frozen datasets with ``StatusOverlayDataset``, and it
refuses to build anything unless the reader contract pins every identity and the
selected rows are already certified.

Design rules:

* Selection is always an explicit ``sample_id`` list per split; there is no
  implicit "admit the whole split" mode.
* Camera/LiDAR/Target payloads come from the unchanged frozen reader.
* A0 Motion4 and the ``[Left, Straight, Right, Unknown, vx, vy, ax, ay]`` order are
  enforced by the reader contract, not re-implemented here.
* No Lightning, no trainer, no optimizer: training launch stays a separate,
  explicitly authorized step.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = 'status-finetune-entry-v1'
REQUIRED_CONTRACT_KEYS = (
    'schema', 'reader_version', 'manifest_sha256', 'pair_index_sha256', 'overlay_sha256',
    'reader_source_sha256', 'contract_source_sha256',
    'allowed_overlay_producer_versions', 'allowed_navigation_evidence_sha256',
    'selected_sample_ids_by_split',
)


class EntryRejected(RuntimeError):
    """Raised when the entry cannot prove identities or certified selection."""


def _load_json(path):
    return json.loads(Path(path).read_text())


def validate_contract(contract, manifest_path, overlay_path):
    """Fail before touching any cache payload when the contract is incomplete."""
    missing = [key for key in REQUIRED_CONTRACT_KEYS if key not in contract]
    if missing:
        raise EntryRejected('contract_missing_fields:' + ','.join(missing))
    selection = contract['selected_sample_ids_by_split']
    if not isinstance(selection, dict) or not selection:
        raise EntryRejected('selection_missing')
    unknown = sorted(set(selection) - {'train', 'validation'})
    if unknown:
        raise EntryRejected('selection_unknown_split:' + ','.join(unknown))
    for split, ids in selection.items():
        if not isinstance(ids, list) or not ids:
            raise EntryRejected('selection_empty_or_not_list:' + split)
        if len(set(ids)) != len(ids) or any(not isinstance(sid, str) or not sid.strip() for sid in ids):
            raise EntryRejected('selection_invalid_ids:' + split)
    if not contract['allowed_navigation_evidence_sha256']:
        raise EntryRejected('navigation_evidence_allowlist_empty')
    if not contract['allowed_overlay_producer_versions']:
        raise EntryRejected('overlay_producer_allowlist_empty')
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or 'pair_index' not in manifest:
        raise EntryRejected('manifest_invalid')
    if not Path(overlay_path).is_file():
        raise EntryRejected('overlay_missing')
    if len({contract['manifest_sha256'], contract['pair_index_sha256'], contract['overlay_sha256']}) != 3:
        raise EntryRejected('contract_sha_not_distinct')
    return {'splits': sorted(selection), 'selected': {k: len(v) for k, v in selection.items()},
            'pair_index': manifest['pair_index']}


def build_dataset(manifest_path, config, contract_path, overlay_path, split):
    """Return the certified Status-replacing dataset for one split."""
    from tools.training.diffusiondrive_v1.status_overlay_reader import StatusOverlayDataset

    contract = _load_json(contract_path)
    validate_contract(contract, manifest_path, overlay_path)
    if split not in contract['selected_sample_ids_by_split']:
        raise EntryRejected('split_not_selected:' + split)
    return StatusOverlayDataset.from_files(manifest_path, config, overlay_path, contract_path, split)


def build_dataloader(manifest_path, config, contract_path, overlay_path, split,
                     batch_size=None, num_workers=0):
    """CPU/GPU-agnostic DataLoader factory; does not start training."""
    from torch.utils.data import DataLoader

    dataset = build_dataset(manifest_path, config, contract_path, overlay_path, split)
    options = {}
    if num_workers:
        options.update(prefetch_factor=config.get('training', {}).get('prefetch_factor', 2),
                       persistent_workers=bool(config.get('training', {}).get('persistent_workers', False)))
    return DataLoader(dataset, batch_size=batch_size or config.get('batch_size', 8),
                      shuffle=False, num_workers=num_workers, **options), dataset


def preflight(manifest_path, config_path, contract_path, overlay_path):
    """Report whether the entry could build without opening any tensor payload."""
    config = _load_json(config_path)
    contract = _load_json(contract_path)
    status = {'schema': 'status-finetune-entry-preflight-v1', 'entry_version': VERSION,
              'manifest': str(manifest_path), 'config': str(config_path),
              'contract': str(contract_path), 'overlay': str(overlay_path)}
    try:
        info = validate_contract(contract, manifest_path, overlay_path)
        status.update(validation='PASS', selection=info)
    except EntryRejected as exc:
        status.update(validation='REJECTED', reason=str(exc))
        status['admission'] = 'NO_GO'
        return status
    try:
        from tools.training.diffusiondrive_v1.status_overlay_reader import StatusOverlayDataset
        StatusOverlayDataset.from_files(manifest_path, config, overlay_path, contract_path, 'train')
    except Exception as exc:  # noqa: BLE001 - typed adapter rejection is the expected outcome today
        status.update(reader_gate='REJECTED', reader_error=type(exc).__name__ + ': ' + str(exc)[:300])
        status['admission'] = 'NO_GO'
        return status
    status.update(reader_gate='PASS', admission='READY_FOR_AUTHORIZED_LAUNCH')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    result = preflight(args.manifest, args.config, args.contract, args.overlay)
    if args.output is not None:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['admission'] == 'READY_FOR_AUTHORIZED_LAUNCH' else 3


if __name__ == '__main__':
    sys.exit(main())
