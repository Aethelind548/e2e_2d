"""Release checks reject missing identities and never read data payloads."""
import hashlib
import json

import pytest

from tools.release.preflight import inspect_profile, small_text_sha256


def manifest(**profile):
    return {'schema': 'diffusiondrive-v1-release-resources-v1',
            'profiles': {'r2': profile}, 'resources': []}


def test_missing_entry_is_blocked(tmp_path):
    result = inspect_profile(tmp_path, manifest(entry_files=['missing.py']), 'r2')
    assert result['status'] == 'BLOCKED'


def test_inherited_fit_mismatch_does_not_masquerade_as_r2_acceptance(tmp_path):
    (tmp_path / 'fit.py').write_text('original = False\n')
    (tmp_path / 'config.json').write_text(json.dumps({'execution_source_sha256': {'fit.py': '0' * 64}}))
    config = manifest(config_files=['config.json'], fit_execution_contract=False)
    report = inspect_profile(tmp_path, config, 'r2')
    assert report['status'] == 'UNVERIFIED'
    assert report['training_or_agent_acceptance'] == 'NOT_VERIFIED'
    config['profiles']['r2']['fit_execution_contract'] = True
    assert inspect_profile(tmp_path, config, 'r2')['status'] == 'BLOCKED'


def test_resource_is_metadata_only_even_if_expected_sha_exists(tmp_path, monkeypatch):
    asset = tmp_path / 'weights.ckpt'
    asset.write_bytes(b'not a checkpoint to deserialize')
    config = manifest()
    config['resources'] = [{'id': 'weights', 'kind': 'file', 'profiles': ['r2'],
                            'path': '{workspace}/weights.ckpt', 'expected_sha256': '0' * 64}]
    monkeypatch.setattr(type(asset), 'read_bytes', lambda self: pytest.fail('resource contents read'))
    report = inspect_profile(tmp_path, config, 'r2')
    assert report['checks'][0]['detail']['payload_identity'].startswith('NOT_VERIFIED')
    assert asset.stat().st_size > 0


def test_missing_location_is_explicit_and_no_directory_is_created(tmp_path):
    config = manifest()
    config['resources'] = [{'id': 'data', 'kind': 'directory', 'profiles': ['r2'], 'path': '{data_root}/v1'}]
    assert inspect_profile(tmp_path, config, 'r2')['status'] == 'BLOCKED'
    assert list(tmp_path.iterdir()) == []


def test_source_identity_mismatch_is_blocked(tmp_path):
    source = tmp_path / 'entry.py'
    source.write_text('pass\n')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    config = manifest(source_sha256={'entry.py': sha})
    assert inspect_profile(tmp_path, config, 'r2')['status'] == 'STATIC_CHECKS_PASS'
    source.write_text('raise RuntimeError()\n')
    assert inspect_profile(tmp_path, config, 'r2')['status'] == 'BLOCKED'


def test_source_path_cannot_escape_release(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        inspect_profile(tmp_path, manifest(entry_files=['../outside.py']), 'r2')


def test_large_resource_cannot_be_hashed_as_small_source(tmp_path):
    path = tmp_path / 'large.ckpt'
    with path.open('wb') as stream:
        stream.truncate(9 * 1024 * 1024)
    with pytest.raises(ValueError, match='limit'):
        small_text_sha256(path)


def test_unknown_manifest_schema_rejected(tmp_path):
    with pytest.raises(ValueError, match='schema'):
        inspect_profile(tmp_path, {'schema': 'unknown'}, 'r2')
