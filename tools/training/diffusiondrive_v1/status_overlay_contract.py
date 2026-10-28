"""Status Overlay v2: CPU-only, fail-closed identity and admission contract.

Version 2 adds explicit cache identities and navigation certification; an offline
expert reconstruction remains NOT_VERIFIED even when its geometry is valid.
"""
import re

import numpy as np

SCHEMA = 'status-overlay-v2'
READER_SCHEMA = 'status-overlay-reader-contract-v1'
NAVIGATION_ORDER = ['Left', 'Straight', 'Right', 'Unknown']
STATUS_ORDER = NAVIGATION_ORDER + ['vx', 'vy', 'ax', 'ay']


def sha256_value(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def vector(value, size):
    """Decode a JSON vector only when explicitly numeric and finite in float32."""
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    if any(isinstance(v, (bool, str)) or not isinstance(v, (int, float)) for v in value):
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.asarray(value, dtype=np.float32)
    return result if result.shape == (size,) and np.isfinite(result).all() else None


def validate_row(row, pair, context, *, require_admitted=False):
    """Return stable failure codes. Never infer identity from ordering or paths.

    ``pair`` is a frozen paired-index record. ``context`` contains the expected
    manifest/index and Feature/Target producer hashes. Admission additionally
    requires externally pinned producer and navigation evidence allowlists.
    """
    errors = []
    if not isinstance(row, dict):
        return ['overlay_not_object']
    if row.get('schema') != SCHEMA:
        errors.append('overlay_schema')
    sid = row.get('sample_id')
    if not isinstance(sid, str) or not sid.strip() or sid != pair.get('sample_id'):
        errors.append('sample_id_mismatch')
    identity = row.get('sample_identity')
    if not isinstance(identity, dict) or any(identity.get(k) != pair.get(k) for k in ('sample_id', 'route', 'frame', 'split')):
        errors.append('route_frame_split_mismatch')
    if isinstance(pair.get('frame'), bool) or not isinstance(pair.get('frame'), int):
        errors.append('frame_identity_type')
    if isinstance(identity, dict) and type(identity.get('frame')) is not int:
        errors.append('frame_identity_type')
    source = row.get('source_feature_identity')
    expected = {'feature_sha256': pair.get('feature_sha'), 'target_sha256': pair.get('target_sha'),
                **{k: context.get(k) for k in ('feature_producer_sha256', 'target_producer_sha256', 'manifest_sha256', 'pair_index_sha256')}}
    if not isinstance(source, dict) or any(not sha256_value(v) or source.get(k) != v for k, v in expected.items()):
        errors.append('cache_identity_mismatch')
    if row.get('status_dtype') != 'float32' or row.get('status_order') != STATUS_ORDER or row.get('navigation_order') != NAVIGATION_ORDER:
        errors.append('status_layout')
    candidate = vector(row.get('status8'), 8)
    old = vector(row.get('old_status8'), 8)
    a0 = vector(row.get('a0_motion4'), 4)
    if candidate is None:
        if row.get('status8') is not None or row.get('status_validity') != 'INVALID':
            errors.append('status_numeric')
    elif not np.isin(candidate[:4], [0., 1.]).all() or candidate[:4].sum() != 1:
        errors.append('navigation_one_hot')
    if row.get('motion_source') != 'A0' or old is None or a0 is None or not np.array_equal(old[4:], a0):
        errors.append('a0_motion_identity')
    if candidate is not None and (a0 is None or not np.array_equal(candidate[4:], a0)):
        errors.append('a0_motion_changed')
    nav = row.get('navigation_source')
    if not isinstance(nav, dict) or not isinstance(nav.get('kind'), str) or not nav.get('kind') or nav.get('certification') not in ('CERTIFIED', 'NOT_VERIFIED', 'INVALID'):
        errors.append('navigation_source_missing')
    if row.get('preview_validity') not in ('VALID', 'NOT_VERIFIED', 'INVALID', 'MISSING'):
        errors.append('preview_validity_missing')
    if row.get('status_validity') not in ('VALID', 'NOT_VERIFIED', 'INVALID'):
        errors.append('status_validity_missing')
    if not isinstance(row.get('training_admitted'), bool) or 'failure_reason' not in row:
        errors.append('admission_metadata_missing')
    if not isinstance(row.get('producer_version'), str) or not row.get('producer_version'):
        errors.append('overlay_producer_missing')
    if require_admitted:
        if row.get('training_admitted') is not True:
            errors.append('training_not_admitted')
        if candidate is None or row.get('status_validity') != 'VALID' or row.get('preview_validity') != 'VALID':
            errors.append('valid_status_required')
        if row.get('fallback_source') is not None or (isinstance(nav, dict) and 'NEAR' in nav.get('kind', '').upper()):
            errors.append('near_fallback_forbidden')
        if not isinstance(nav, dict) or nav.get('certification') != 'CERTIFIED' or nav.get('reference_path_online_available') is not True:
            errors.append('navigation_not_certified')
        if row.get('producer_version') not in context.get('allowed_overlay_producer_versions', []):
            errors.append('overlay_producer_not_approved')
        evidence = nav.get('evidence_sha256') if isinstance(nav, dict) else None
        if not sha256_value(evidence) or evidence not in context.get('allowed_navigation_evidence_sha256', []):
            errors.append('navigation_evidence_not_approved')
        if row.get('failure_reason') not in (None, ''):
            errors.append('admitted_failure_reason')
    return errors
