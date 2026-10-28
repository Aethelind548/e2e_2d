"""CPU tests for the independent Status fine-tune entry (no trainer, no GPU)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.training.diffusiondrive_v1 import status_finetune_entry as entry
from tools.training.diffusiondrive_v1 import status_overlay_contract, status_overlay_reader


def contract(tmp_path, **overrides):
    base = {
        'schema': status_overlay_contract.READER_SCHEMA,
        'reader_version': status_overlay_reader.VERSION,
        'manifest_sha256': 'a' * 64,
        'pair_index_sha256': 'b' * 64,
        'overlay_sha256': 'c' * 64,
        'reader_source_sha256': 'd' * 64,
        'contract_source_sha256': 'e' * 64,
        'allowed_overlay_producer_versions': ['full-status-overlay-v2.0.0'],
        'allowed_navigation_evidence_sha256': ['f' * 64],
        'selected_sample_ids_by_split': {'train': ['s1'], 'validation': ['s2']},
    }
    base.update(overrides)
    return base


def manifest(path):
    path.write_text(json.dumps({'pair_index': '/tmp/_pairs.sqlite', 'dataset_id': 'x'}))
    return path


def overlay(path):
    path.write_text('{}\n')
    return path


def test_contract_accepts_complete_pins(tmp_path):
    info = entry.validate_contract(contract(tmp_path), manifest(tmp_path / 'm.json'), overlay(tmp_path / 'o.jsonl'))
    assert info['selected'] == {'train': 1, 'validation': 1}
    assert info['splits'] == ['train', 'validation']


@pytest.mark.parametrize('overrides, reason', [
    ({'selected_sample_ids_by_split': {}}, 'selection_missing'),
    ({'selected_sample_ids_by_split': {'train': []}}, 'selection_empty_or_not_list'),
    ({'selected_sample_ids_by_split': {'test': ['x']}}, 'selection_unknown_split'),
    ({'selected_sample_ids_by_split': {'train': ['a', 'a']}}, 'selection_invalid_ids'),
    ({'allowed_navigation_evidence_sha256': []}, 'navigation_evidence_allowlist_empty'),
    ({'allowed_overlay_producer_versions': []}, 'overlay_producer_allowlist_empty'),
    ({'manifest_sha256': 'b' * 64}, 'contract_sha_not_distinct'),
])
def test_contract_rejections(tmp_path, overrides, reason):
    with pytest.raises(entry.EntryRejected, match=reason):
        entry.validate_contract(contract(tmp_path, **overrides), manifest(tmp_path / 'm.json'),
                                overlay(tmp_path / 'o.jsonl'))


def test_contract_requires_every_identity_key(tmp_path):
    broken = contract(tmp_path)
    broken.pop('reader_source_sha256')
    with pytest.raises(entry.EntryRejected, match='contract_missing_fields'):
        entry.validate_contract(broken, manifest(tmp_path / 'm.json'), overlay(tmp_path / 'o.jsonl'))


def test_preflight_reports_no_go_without_evidence(tmp_path):
    contract_path = tmp_path / 'c.json'
    contract_path.write_text(json.dumps(contract(tmp_path, allowed_navigation_evidence_sha256=[])))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'cache_read_mode': 'cache-only'}))
    result = entry.preflight(manifest(tmp_path / 'm.json'), config, contract_path, overlay(tmp_path / 'o.jsonl'))
    assert result['validation'] == 'REJECTED'
    assert result['admission'] == 'NO_GO'


def test_build_dataset_propagates_reader_gate(monkeypatch, tmp_path):
    contract_path = tmp_path / 'c.json'
    contract_path.write_text(json.dumps(contract(tmp_path)))
    calls = {}

    class Boom(Exception):
        pass

    def fake_from_files(*args, **kwargs):
        calls['args'] = args
        raise Boom('certified gate')

    monkeypatch.setattr(status_overlay_reader.StatusOverlayDataset, 'from_files', staticmethod(fake_from_files))
    with pytest.raises(Boom):
        entry.build_dataset(manifest(tmp_path / 'm.json'), {'cache_read_mode': 'cache-only'},
                            contract_path, overlay(tmp_path / 'o.jsonl'), 'train')
    assert calls['args'][-1] == 'train'


def test_build_dataloader_uses_explicit_batch(monkeypatch, tmp_path):
    contract_path = tmp_path / 'c.json'
    contract_path.write_text(json.dumps(contract(tmp_path)))

    class FakeDataset:
        rows = [{'sample_id': 's1'}]
        tokens = ['s1']

        def __len__(self):
            return 1

    monkeypatch.setattr(status_overlay_reader.StatusOverlayDataset, 'from_files',
                        staticmethod(lambda *a, **k: FakeDataset()))
    loader, dataset = entry.build_dataloader(manifest(tmp_path / 'm.json'),
                                             {'cache_read_mode': 'cache-only', 'batch_size': 8},
                                             contract_path, overlay(tmp_path / 'o.jsonl'), 'train',
                                             batch_size=1, num_workers=0)
    assert loader.batch_size == 1
    assert isinstance(dataset, FakeDataset)


def test_entry_does_not_import_training_loop():
    text = Path(entry.__file__).read_text()
    for forbidden in ('import lightning', 'from lightning', 'diffusiondrive_v1.fit', 'Trainer('):
        assert forbidden not in text
