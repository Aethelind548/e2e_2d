"""CPU fixtures for independent Overlay v2; no real sample is certified here."""
import copy
import json
import pickle
import sqlite3

import pytest
import torch
from torch.utils.data import DataLoader

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1.status_overlay_contract import NAVIGATION_ORDER, READER_SCHEMA, SCHEMA, STATUS_ORDER, validate_row
from tools.training.diffusiondrive_v1.status_overlay_reader import VERSION, SelectedSampleView, StatusOverlayDataset, file_identity, file_sha256


class FixtureDataset:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        self.features = {'camera_feature': torch.zeros(3, 4, 4), 'lidar_feature': torch.ones(1, 4, 4),
                         'status_feature': torch.tensor([0., 1., 0., 0., 1., 2., 3., 4.])}
        self.targets = {'trajectory': torch.zeros(8, 3)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.features, self.targets


def fixture():
    pair = {'sample_id': 'fixture-only-1', 'route': 'fixture-route', 'frame': 7, 'split': 'train',
            'feature_sha': 'a' * 64, 'target_sha': 'b' * 64, 'target_source': 'baseline'}
    context = {'manifest_sha256': 'c' * 64, 'pair_index_sha256': 'd' * 64,
               'feature_producer_sha256': 'e' * 64, 'target_producer_sha256': 'f' * 64,
               'allowed_overlay_producer_versions': ['fixture-only-v1'],
               'allowed_navigation_evidence_sha256': ['1' * 64]}
    row = {'schema': SCHEMA, 'sample_id': pair['sample_id'],
           'sample_identity': {k: pair[k] for k in ('sample_id', 'route', 'frame', 'split')},
           'source_feature_identity': {'feature_sha256': pair['feature_sha'], 'target_sha256': pair['target_sha'],
               **{k: v for k, v in context.items() if k.endswith('sha256')}},
           'status8': [1., 0., 0., 0., 1., 2., 3., 4.],
           'old_status8': [0., 1., 0., 0., 1., 2., 3., 4.],
           'a0_motion4': [1., 2., 3., 4.], 'motion_source': 'A0',
           'status_dtype': 'float32', 'status_order': STATUS_ORDER, 'navigation_order': NAVIGATION_ORDER,
           'navigation_source': {'kind': 'FIXTURE_ONLY_ONLINE_ROUTE', 'certification': 'CERTIFIED',
                                 'reference_path_online_available': True, 'evidence_sha256': '1' * 64},
           'preview_validity': 'VALID', 'status_validity': 'VALID', 'training_admitted': True,
           'failure_reason': None, 'fallback_source': None, 'producer_version': 'fixture-only-v1'}
    return pair, row, context


def dataset(pair=None, row=None, context=None):
    p, r, c = fixture()
    pair, row, context = pair or p, row or r, context or c
    return StatusOverlayDataset(FixtureDataset([pair]), [row], pair_rows=[pair], context=context)


def test_only_status_replaced_preserving_objects_and_original():
    wrapped = dataset()
    features, targets = wrapped[0]
    assert features is not wrapped.base_dataset.features
    assert features['camera_feature'] is wrapped.base_dataset.features['camera_feature']
    assert features['lidar_feature'] is wrapped.base_dataset.features['lidar_feature']
    assert targets is wrapped.base_dataset.targets
    assert features['status_feature'].dtype == torch.float32
    assert features['status_feature'].shape == (8,)
    assert torch.equal(features['status_feature'][4:], wrapped.base_dataset.features['status_feature'][4:])
    assert wrapped.base_dataset.features['status_feature'][1] == 1
    assert features['status_feature'][0] == 1


@pytest.mark.parametrize('workers', [0, 2])
def test_cpu_dataloader_and_pickle(workers):
    wrapped = pickle.loads(pickle.dumps(dataset()))
    options = {'multiprocessing_context': 'spawn', 'timeout': 30} if workers else {}
    features, targets = next(iter(DataLoader(wrapped, batch_size=1, num_workers=workers, **options)))
    assert features['status_feature'].shape == (1, 8)
    assert features['status_feature'][0, 0] == 1
    assert targets['trajectory'].shape == (1, 8, 3)


@pytest.mark.parametrize('mutation,reason', [
    ({'training_admitted': False}, 'training_not_admitted'),
    ({'status_validity': 'NOT_VERIFIED'}, 'valid_status_required'),
    ({'status_dtype': 'float64'}, 'status_layout'),
    ({'status8': [0.] * 7}, 'status_numeric'),
    ({'status8': [0., 1., 0., 0., float('nan'), 2., 3., 4.]}, 'status_numeric'),
    ({'status8': [0., 1., 0., 0., float('inf'), 2., 3., 4.]}, 'status_numeric'),
    ({'status8': ['0', 1., 0., 0., 1., 2., 3., 4.]}, 'status_numeric'),
    ({'status8': [0., 1., 0., 0., 9., 2., 3., 4.]}, 'a0_motion_changed'),
    ({'motion_source': 'A2V'}, 'a0_motion_identity'),
    ({'fallback_source': 'NEAR_FALLBACK'}, 'near_fallback_forbidden'),
    ({'producer_version': 'unapproved'}, 'overlay_producer_not_approved'),
    ({'navigation_order': ['Right', 'Left', 'Straight', 'Unknown']}, 'status_layout'),
    ({'failure_reason': 'some_failure'}, 'admitted_failure_reason'),
])
def test_invalid_rows_rejected(mutation, reason):
    pair, row, context = fixture()
    row.update(mutation)
    with pytest.raises(AdapterError, match=reason):
        dataset(pair, row, context)


@pytest.mark.parametrize('field', ['sample_id', 'route', 'frame', 'split'])
def test_identity_mismatch(field):
    pair, row, context = fixture()
    row['sample_identity'][field] = 'wrong'
    with pytest.raises(AdapterError, match='route_frame_split_mismatch'):
        dataset(pair, row, context)


@pytest.mark.parametrize('field', ['feature_sha256', 'target_sha256', 'feature_producer_sha256',
                                  'target_producer_sha256', 'manifest_sha256', 'pair_index_sha256'])
def test_sha_producer_identity(field):
    pair, row, context = fixture()
    row['source_feature_identity'][field] = '9' * 64
    with pytest.raises(AdapterError, match='cache_identity_mismatch'):
        dataset(pair, row, context)


def test_offline_status_refused_before_payload_read():
    pair, row, context = fixture()
    row['navigation_source'].update(kind='OFFLINE_EXPERT_RECONSTRUCTION', certification='NOT_VERIFIED',
                                     reference_path_online_available=False)
    row.update(status_validity='NOT_VERIFIED', training_admitted=False)
    assert not validate_row(row, pair, context)
    with pytest.raises(AdapterError, match='navigation_not_certified'):
        dataset(pair, row, context)


def test_forged_certification_without_pinned_evidence_refused():
    pair, row, context = fixture()
    context['allowed_navigation_evidence_sha256'] = []
    with pytest.raises(AdapterError, match='navigation_evidence_not_approved'):
        dataset(pair, row, context)


def test_legal_unknown_is_not_missing():
    pair, row, context = fixture()
    row['status8'][:4] = [0., 0., 0., 1.]
    assert dataset(pair, row, context)[0][0]['status_feature'][3] == 1


def test_missing_duplicate_and_foreign_overlay():
    pair, row, context = fixture()
    base = FixtureDataset([pair])
    for rows, code in [([], 'overlay_missing_sample'), ([row, row], 'overlay_duplicate_or_empty'),
                       ([dict(row, sample_id='foreign')], 'overlay_unknown_sample')]:
        with pytest.raises(AdapterError, match=code):
            StatusOverlayDataset(base, rows, pair_rows=[pair], context=context)


def test_reversed_overlay_uses_sample_id():
    pair, row, context = fixture()
    other = dict(pair, sample_id='fixture-only-2', frame=8)
    row2 = copy.deepcopy(row)
    row2['sample_id'] = other['sample_id']
    row2['sample_identity'].update(sample_id=other['sample_id'], frame=8)
    row2['status8'][:4] = [0., 0., 1., 0.]
    wrapped = StatusOverlayDataset(FixtureDataset([pair, other]), [row2, row], pair_rows=[pair, other], context=context)
    assert wrapped[0][0]['status_feature'][0] == 1
    assert wrapped[1][0]['status_feature'][2] == 1


def test_original_status_and_base_row_tampering_refused():
    wrapped = dataset()
    wrapped.base_dataset.features['status_feature'][4] = 8
    with pytest.raises(AdapterError, match='overlay_original_status_mismatch'):
        wrapped[0]
    wrapped = dataset()
    wrapped.base_dataset.rows[0]['split'] = 'validation'
    with pytest.raises(AdapterError, match='overlay_base_identity_changed'):
        wrapped[0]


def test_watched_file_tampering_refused(tmp_path):
    path = tmp_path / 'overlay.jsonl'
    path.write_text('{}\n')
    pair, row, context = fixture()
    wrapped = StatusOverlayDataset(FixtureDataset([pair]), [row], pair_rows=[pair], context=context,
                                   watched_files={str(path): file_identity(path)})
    path.write_text('{}\n{}\n')
    with pytest.raises(AdapterError, match='overlay_identity_changed'):
        wrapped[0]


def frozen_fixture(tmp_path):
    from tools.training.diffusiondrive_v1 import status_overlay_contract, status_overlay_reader
    pair, row, context = fixture()
    index = tmp_path / 'pairs.sqlite'
    db = sqlite3.connect(index)
    db.execute('CREATE TABLE pairs (sample_id TEXT,route TEXT,frame INTEGER,split TEXT,feature_sha TEXT,target_sha TEXT,target_source TEXT)')
    db.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?,?)', tuple(pair.values()))
    db.commit()
    db.close()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'pair_index': str(index), 'index_sha256': file_sha256(index),
                                   'feature_producer_sha256': context['feature_producer_sha256'],
                                   'target_sources': {'baseline': {'producer_sha256': context['target_producer_sha256']}}}))
    row['source_feature_identity'].update(manifest_sha256=file_sha256(manifest), pair_index_sha256=file_sha256(index))
    row.update(training_admitted=False, status_validity='NOT_VERIFIED')
    row['navigation_source'].update(certification='NOT_VERIFIED', reference_path_online_available=False)
    overlay = tmp_path / 'overlay.jsonl'
    overlay.write_text(json.dumps(row) + '\n')
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'schema': READER_SCHEMA, 'reader_version': VERSION,
        'manifest_sha256': file_sha256(manifest), 'pair_index_sha256': file_sha256(index),
        'overlay_sha256': file_sha256(overlay), 'reader_source_sha256': file_sha256(status_overlay_reader.__file__),
        'contract_source_sha256': file_sha256(status_overlay_contract.__file__),
        'allowed_overlay_producer_versions': context['allowed_overlay_producer_versions'],
        'allowed_navigation_evidence_sha256': []}))
    return manifest, overlay, contract


def test_from_files_refuses_before_production_factory(tmp_path, monkeypatch):
    import tools.training.diffusiondrive_v1.recovery_paired as formal
    monkeypatch.setattr(formal, 'datasets', lambda *args: pytest.fail('Must not open formal reader without certification'))
    manifest, overlay, contract = frozen_fixture(tmp_path)
    with pytest.raises(AdapterError, match='navigation_not_certified'):
        StatusOverlayDataset.from_files(manifest, {'cache_read_mode': 'cache-only'}, overlay, contract, 'train')


@pytest.mark.parametrize('key', ['manifest_sha256', 'pair_index_sha256', 'overlay_sha256', 'reader_source_sha256', 'contract_source_sha256'])
def test_factory_requires_all_frozen_sha(tmp_path, key):
    manifest, overlay, contract = frozen_fixture(tmp_path)
    value = json.loads(contract.read_text())
    value[key] = '0' * 64
    contract.write_text(json.dumps(value))
    with pytest.raises(AdapterError, match='overlay_frozen_file_sha'):
        StatusOverlayDataset.from_files(manifest, {'cache_read_mode': 'cache-only'}, overlay, contract, 'train')


def test_explicit_sample_selection_retains_identity_and_rejects_foreign_split():
    pair, row, context = fixture()
    other = dict(pair, sample_id='fixture-2', frame=8)
    base = FixtureDataset([pair, other])
    selected = SelectedSampleView(base, [other['sample_id'], pair['sample_id']])
    assert [p['sample_id'] for p in selected.rows] == ['fixture-2', 'fixture-only-1']
    assert selected[0][1] is base.targets
    for ids in [[], [pair['sample_id'], pair['sample_id']], ['other-split-id']]:
        with pytest.raises(AdapterError, match='overlay_selection_invalid'):
            SelectedSampleView(base, ids)
