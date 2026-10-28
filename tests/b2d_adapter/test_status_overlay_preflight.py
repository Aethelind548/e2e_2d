import copy

from tools.validation.b2d.status_overlay_preflight import check_rows


def _row():
    sid = 'sample-1'
    return {
        'sample_id': sid,
        'sample_identity': {'sample_id': sid, 'route': 'route-A', 'frame': 7,
                            'split': 'validation'},
        'status8': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        'old_status8': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        'status_validity': 'NOT_VERIFIED',
        'status_dtype': 'float32',
        'status_order': ['Left', 'Straight', 'Right', 'Unknown', 'vx', 'vy', 'ax', 'ay'],
        'navigation_source': 'LOCAL_REFERENCE_SEGMENT',
        'fallback_source': None,
        'diagnostic_metadata': {'upstream': {
            'motion_method': 'A0_FROZEN',
            'preview': {'preview_validity': 'VALID'},
        }},
        'source_identity': {
            'feature_sha256': 'a' * 64, 'target_sha256': 'b' * 64,
            'feature_producer_sha256': 'c' * 64,
            'target_producer_sha256': 'd' * 64,
            'manifest_sha256': 'e' * 64,
        },
        'training_admitted': False,
        'training_contract': 'NOT_ADMITTED_EXPERIMENT_ONLY',
    }


def _panel():
    return [{
        'sample_id': 'sample-1',
        'sample_identity': {'sample_id': 'sample-1', 'route': 'route-A', 'frame': 7,
                            'split': 'validation'},
        'row': {'feature_sha': 'a' * 64, 'target_sha': 'b' * 64},
        'split': 'validation',
    }]


def _index():
    return [{'sample_id': 'sample-1', 'route': 'route-A', 'frame': 7,
             'split': 'validation'}]


def test_preflight_accepts_complete_contract_row():
    counts = check_rows([_row()], _panel(), _index(), [])
    assert counts['sample_id_unique']['failed'] == 0
    assert counts['route_frame_split_match']['failed'] == 0
    assert counts['motion4_a0_match']['failed'] == 0
    assert counts['a0_motion_method']['failed'] == 0
    assert counts['status_command_one_hot']['failed'] == 0
    assert counts['reader_overlay_recognition']['reader_supports_overlay'] is False


def test_preflight_rejects_duplicate_and_identity_mismatch():
    rows = [_row(), copy.deepcopy(_row())]
    counts = check_rows(rows, _panel(), _index(), [])
    assert counts['sample_id_unique']['failed'] == 1
    bad = _row()
    bad['sample_identity'] = dict(bad['sample_identity'], route='wrong-route')
    counts = check_rows([bad], _panel(), _index(), [])
    assert counts['route_frame_split_match']['failed'] == 1


def test_preflight_rejects_bad_status_shape_and_motion():
    bad = _row()
    bad['status8'] = [0.0, 1.0, 0.0]
    counts = check_rows([bad], _panel(), _index(), [])
    assert counts['status8_shape_float32_8']['failed'] == 1
    bad = _row()
    bad['status8'][4] = 9.0
    counts = check_rows([bad], _panel(), _index(), [])
    assert counts['motion4_a0_match']['failed'] == 1


def test_preflight_rejects_invalid_row_with_status8():
    bad = _row()
    bad['status_validity'] = 'INVALID'
    counts = check_rows([bad], _panel(), _index(), [])
    assert counts['invalid_has_no_status8']['failed'] == 1


def test_preflight_requires_status_order_and_preview_validity():
    bad = _row()
    bad['status_order'] = ['Straight', 'Left', 'Right', 'Unknown', 'vx', 'vy', 'ax', 'ay']
    bad['diagnostic_metadata']['upstream'].pop('preview')
    counts = check_rows([bad], _panel(), _index(), [])
    assert counts['status_order_v1']['failed'] == 1
    assert counts['preview_validity_present']['failed'] == 1
