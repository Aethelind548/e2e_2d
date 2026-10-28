"""Target relocation gates: preserve frozen logical identity and fail closed."""
import copy
import json
import pickle
from pathlib import Path
import sqlite3

import pytest

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1.features_cache import digest, file_digest
from tools.training.diffusiondrive_v1 import target_relocation as relocation


def fixture(tmp_path, monkeypatch):
    state = tmp_path / 'state'; state.mkdir()
    destination = tmp_path / 'pssd'; destination.mkdir()
    source = dict(path=str(tmp_path / 'old'), root_id='targets_data', uuid='old', fstype='ext4')
    target = dict(source, path=str(destination), uuid='pssd', fstype='ntfs3')
    producer = {'fixture': 'numeric-producer-unchanged'}
    with sqlite3.connect(state / 'index.sqlite') as db:
        db.execute('create table meta(key text primary key,value text)')
        db.executemany('insert into meta values(?,?)', [('target_root', json.dumps(source)), ('producer', json.dumps(producer))])
    owner = destination / 'target_root.json'
    owner.write_text(json.dumps(dict(state_id=str(state), root_id='targets_data')))
    entry = {'from': source, 'to': target, 'producer_sha256': digest(producer),
             'destination_owner_sha256': {'target_root.json': file_digest(owner)}}
    record = dict(schema=relocation.SCHEMA, verified=True, roots={str(state): entry})
    registry = tmp_path / 'registration.json'; registry.write_text(json.dumps(record))
    config = dict(target_storage_registration=dict(path=str(registry), sha256=file_digest(registry)),
                  allowed_target_storage_registration_sha256=file_digest(registry),
                  target_location_reader_sha256=file_digest(relocation.__file__))
    monkeypatch.setattr(relocation, 'storage_check', lambda spec: 1 << 40)
    return state, source, target, producer, registry, config, record


def test_no_registration_preserves_same_root_object():
    original = {'path': '/old', 'root_id': 'targets'}
    assert relocation.TargetLocationResolver({}, {}).resolve(original, 'sha', '/state') is original


@pytest.mark.parametrize('cache_only', [False, True])
def test_paired_readers_resolve_frozen_source_without_rewriting_manifest(tmp_path, monkeypatch, cache_only):
    from tools.training.diffusiondrive_v1.recovery_paired import RecoveryPairedDataset
    from tools.training.diffusiondrive_v1.cache_only_reader import CacheOnlyPairedDataset
    state, source, target, producer, _, config, _ = fixture(tmp_path, monkeypatch)
    cls = CacheOnlyPairedDataset if cache_only else RecoveryPairedDataset
    dataset = cls.__new__(cls)
    dataset.config = config; dataset.manifest = {}
    frozen = dict(target_root=source, producer_sha256=digest(producer), state=str(state))
    assert dataset.target_root(frozen) == target
    assert frozen['target_root'] == source


def test_legacy_paired_dataset_passes_new_root_to_target_loader(tmp_path, monkeypatch):
    from tools.training.diffusiondrive_v1 import paired_dataset as paired
    state, source, target, producer, _, config, _ = fixture(tmp_path, monkeypatch)
    dataset = paired.PairedDataset.__new__(paired.PairedDataset)
    dataset.config = dict(config, target_root=source, target_state_dir=str(state))
    dataset.target_sha = digest(producer); dataset.feature_sha = 'fixture'
    dataset.feature_roots = {'fixture': {}}
    dataset.rows = [dict(feature_row={}, feature_config='fixture', target_row={})]
    monkeypatch.setattr(paired, 'load_feature_row', lambda *args: {'unchanged': True})
    received = []
    def load(row, spec, producer_sha):
        received.append(spec)
        return {'targets': {'unchanged': True}}
    monkeypatch.setattr(paired, 'load_target_row', load)
    assert dataset[0] == ({'unchanged': True}, {'unchanged': True})
    assert received == [target]
    assert dataset.config['target_root'] == source


def test_registered_root_works_with_absent_source_and_pickles(tmp_path, monkeypatch):
    state, source, target, producer, _, config, _ = fixture(tmp_path, monkeypatch)
    resolver = relocation.TargetLocationResolver({}, config)
    assert resolver.resolve(source, digest(producer), state) == target
    restored = pickle.loads(pickle.dumps(resolver))
    assert restored._pid is None
    assert restored.resolve(source, digest(producer), state) == target
    assert source['uuid'] == 'old'


@pytest.mark.parametrize('key', ['allowed_target_storage_registration_sha256', 'target_location_reader_sha256'])
def test_missing_pin_rejected(tmp_path, monkeypatch, key):
    _, _, _, _, _, config, _ = fixture(tmp_path, monkeypatch)
    del config[key]
    with pytest.raises(AdapterError, match='target_relocation_registration'):
        relocation.TargetLocationResolver({}, config)


def test_empty_registration_is_not_implicit_legacy_fallback():
    with pytest.raises(AdapterError, match='target_relocation_registration'):
        relocation.TargetLocationResolver({}, {'target_storage_registration': {}})


def test_manifest_config_registration_disagreement(tmp_path, monkeypatch):
    _, _, _, _, _, config, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(AdapterError, match='target_relocation_registration'):
        relocation.TargetLocationResolver({'target_storage_registration': {'path': '/other', 'sha256': 'a'*64}}, config)


@pytest.mark.parametrize('mutation', ['unverified', 'producer', 'source_root', 'destination_root_id', 'owner_missing'])
def test_forged_registration_rejected(tmp_path, monkeypatch, mutation):
    state, source, _, producer, registry, config, record = fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(record); entry = changed['roots'][str(state)]
    if mutation == 'unverified': changed['verified'] = False
    elif mutation == 'producer': entry['producer_sha256'] = 'a'*64
    elif mutation == 'source_root': entry['from']['path'] = '/guessed'
    elif mutation == 'destination_root_id': entry['to']['root_id'] = 'other'
    else: entry['destination_owner_sha256'] = {}
    registry.write_text(json.dumps(changed))
    config['target_storage_registration']['sha256'] = file_digest(registry)
    config['allowed_target_storage_registration_sha256'] = file_digest(registry)
    with pytest.raises(AdapterError, match='target_relocation_'):
        relocation.TargetLocationResolver({}, config).resolve(source, digest(producer), state)


@pytest.mark.parametrize('change', ['owner', 'registration', 'index_producer'])
def test_changes_after_activation_are_rejected(tmp_path, monkeypatch, change):
    state, source, target, producer, registry, config, _ = fixture(tmp_path, monkeypatch)
    resolver = relocation.TargetLocationResolver({}, config)
    if change == 'index_producer':
        resolver.resolve(source, digest(producer), state)
        with sqlite3.connect(state/'index.sqlite') as db:
            db.execute("update meta set value=? where key='producer'", (json.dumps({'changed': True}),))
    else:
        resolver.resolve(source, digest(producer), state)
        path = registry if change == 'registration' else Path(target['path'])/'target_root.json'
        path.write_text(path.read_text() + ' ')
    with pytest.raises(AdapterError, match='target_relocation_'):
        resolver.resolve(source, digest(producer), state)
