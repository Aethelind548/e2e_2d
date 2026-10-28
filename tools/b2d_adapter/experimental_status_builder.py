"""Experimental eight-channel Status assembly and append-free JSONL overlays.

This module has no production imports or cache writers. Geometry validity and
navigation provenance are separate: reconstructed routes can produce useful
eight-channel candidates without certifying them for training. Every overlay
row is excluded from training until a separate training contract admits it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SCHEMA = 'experimental-status-overlay-v1'
GENERATOR_VERSION = 'status-builder-v1'
COMMAND_ORDER = ('Left', 'Straight', 'Right', 'Unknown')
VALIDITIES = frozenset(('VALID', 'NOT_VERIFIED', 'INVALID', 'FAILED', 'MISSING', 'AMBIGUOUS'))
FAILURES = VALIDITIES - {'VALID', 'NOT_VERIFIED'}
PROVENANCE_LEVELS = frozenset(('CERTIFIED', 'NOT_VERIFIED', 'REJECTED'))
NEAR_FALLBACK = 'NEAR_FALLBACK'


def _json_copy(value, name):
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(name + ' must contain finite JSON values') from exc


def _identity(value):
    if not isinstance(value, dict):
        raise ValueError('sample_identity must be a dictionary')
    result = _json_copy(value, 'sample_identity')
    sample_id = result.get('sample_id')
    if not isinstance(sample_id, str) or not sample_id.strip() or '\x00' in sample_id:
        raise ValueError('sample_identity requires a nonempty string sample_id')
    for alias, primary in (('route_id', 'route'), ('frame_id', 'frame')):
        if alias in result and primary in result and result[alias] != result[primary]:
            raise ValueError('sample_identity contains contradictory ' + primary + ' aliases')
    route = result.get('route', result.get('route_id'))
    frame = result.get('frame', result.get('frame_id'))
    if route is not None and (not isinstance(route, str) or not route.strip()):
        raise ValueError('sample route must be a nonempty string')
    if frame is not None and (isinstance(frame, bool) or not isinstance(frame, (str, int))
                              or re.fullmatch('[0-9]+', str(frame)) is None):
        raise ValueError('sample frame must be a nonnegative integer or digit string')
    return result


def _vector(value, length, name):
    if value is None:
        return None
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu':
            raise ValueError('nonnumeric or complex vector')
        with np.errstate(over='ignore', invalid='ignore'):
            vector = raw.astype(np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a finite numeric vector') from exc
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(name + f' must be finite float32 with shape ({length},)')
    return vector.copy()


def _command(vector):
    if vector is not None and (not np.isin(vector, (0., 1.)).all() or vector.sum() != 1.):
        raise ValueError('command4 must be exactly one-hot in Left/Straight/Right/Unknown order')
    return vector


@dataclass(frozen=True)
class StatusResult:
    status8: np.ndarray | None
    status_validity: str
    fallback_source: str | None
    diagnostic_metadata: dict
    sample_identity: dict

    def as_dict(self):
        return {
            'status8': None if self.status8 is None else self.status8.tolist(),
            'status_validity': self.status_validity,
            'fallback_source': self.fallback_source,
            'diagnostic_metadata': _json_copy(self.diagnostic_metadata, 'diagnostic_metadata'),
            'sample_identity': _json_copy(self.sample_identity, 'sample_identity'),
        }


def build_status(command4, motion4, navigation_validity, motion_validity,
                 navigation_source, sample_identity, *,
                 navigation_provenance='NOT_VERIFIED', fallback_source=None,
                 diagnostic_metadata=None):
    """Combine checked inputs; never repair, smooth, or estimate an upstream value.

    ``VALID`` navigation means the geometry/command passed upstream checks;
    ``navigation_provenance='CERTIFIED'`` separately asserts task-plan lineage.
    The default provenance is deliberately unverified. A legal Unknown is a
    one-hot command, while failed/missing navigation produces ``status8=None``.
    NOT_VERIFIED inputs retain finite candidate vectors for offline comparison.
    The caller must explicitly identify a near fallback with NEAR_FALLBACK.
    """
    identity = _identity(sample_identity)
    if navigation_validity not in VALIDITIES or motion_validity not in VALIDITIES:
        raise ValueError('unknown navigation_validity or motion_validity')
    if navigation_provenance not in PROVENANCE_LEVELS:
        raise ValueError('unknown navigation_provenance')
    if not isinstance(navigation_source, str) or not navigation_source.strip():
        raise ValueError('navigation_source must be a nonempty string')
    if fallback_source not in (None, NEAR_FALLBACK):
        raise ValueError('fallback_source must be None or NEAR_FALLBACK')
    if navigation_source == NEAR_FALLBACK:
        fallback_source = NEAR_FALLBACK
    if fallback_source == NEAR_FALLBACK and navigation_provenance == 'CERTIFIED':
        # A near fallback has no certified route-preview lineage by itself.
        raise ValueError('NEAR_FALLBACK cannot claim certified route-preview provenance')
    command = _command(_vector(command4, 4, 'command4'))
    motion = _vector(motion4, 4, 'motion4')
    reasons = []
    if command is None:
        reasons.append('COMMAND_MISSING')
    if motion is None:
        reasons.append('MOTION_MISSING')
    if navigation_validity in FAILURES:
        reasons.append('NAVIGATION_' + navigation_validity)
    if motion_validity in FAILURES:
        reasons.append('MOTION_' + motion_validity)
    if navigation_provenance == 'REJECTED':
        reasons.append('NAVIGATION_PROVENANCE_REJECTED')
    failed = bool(reasons)
    status = None if failed else np.concatenate((command, motion)).astype(np.float32)
    if failed:
        validity = 'INVALID'
    elif (navigation_validity == motion_validity == 'VALID'
          and navigation_provenance == 'CERTIFIED' and fallback_source is None):
        validity = 'VALID'
    else:
        validity = 'NOT_VERIFIED'
        if navigation_provenance != 'CERTIFIED':
            reasons.append('NAVIGATION_PROVENANCE_NOT_VERIFIED')
        if navigation_validity == 'NOT_VERIFIED':
            reasons.append('NAVIGATION_NOT_VERIFIED')
        if motion_validity == 'NOT_VERIFIED':
            reasons.append('MOTION_NOT_VERIFIED')
        if fallback_source:
            reasons.append(NEAR_FALLBACK)
    caller_metadata = {} if diagnostic_metadata is None else diagnostic_metadata
    if not isinstance(caller_metadata, dict):
        raise ValueError('diagnostic_metadata must be a dictionary')
    metadata = {
        'navigation_source': navigation_source,
        'navigation_validity': navigation_validity,
        'navigation_provenance': navigation_provenance,
        'motion_validity': motion_validity,
        'command_category': None if command is None else COMMAND_ORDER[int(command.argmax())],
        'legal_unknown': bool(command is not None and command[3] == 1 and navigation_validity not in FAILURES),
        'shape_valid_candidate': status is not None,
        'failure_reasons': reasons,
        'upstream': _json_copy(caller_metadata, 'diagnostic_metadata'),
    }
    if status is not None:
        status.flags.writeable = False
    return StatusResult(status, validity, fallback_source, metadata, identity)


def generator_identity():
    return {
        'version': GENERATOR_VERSION,
        'source': 'tools/b2d_adapter/experimental_status_builder.py',
        'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _sha(value, name):
    if value is not None and (not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None):
        raise ValueError(name + ' must be a lowercase SHA256 or None for unavailable evidence')
    return value


def build_overlay_record(result, old_status8, *, feature_sha256, target_sha256,
                         feature_producer_sha256=None, target_producer_sha256=None,
                         manifest_sha256=None):
    """Create one auditable row without changing any source Feature or Target.

    Unavailable source Status/identities remain null and are recorded explicitly;
    an offline panel must not invent hashes or imply production cache membership.
    """
    if not isinstance(result, StatusResult):
        raise ValueError('result must be a StatusResult')
    old_status = _vector(old_status8, 8, 'old_status8')
    if old_status is not None:
        _command(old_status[:4])
    record = result.as_dict()
    record.update({
        'schema': SCHEMA,
        'sample_id': record['sample_identity']['sample_id'],
        'old_status8': None if old_status is None else old_status.tolist(),
        'status_dtype': 'float32',
        'status_order': list(COMMAND_ORDER) + ['vx', 'vy', 'ax', 'ay'],
        'navigation_source': result.diagnostic_metadata['navigation_source'],
        'generator': generator_identity(),
        'source_identity': {
            'feature_sha256': _sha(feature_sha256, 'feature_sha256'),
            'target_sha256': _sha(target_sha256, 'target_sha256'),
            'feature_producer_sha256': _sha(feature_producer_sha256, 'feature_producer_sha256'),
            'target_producer_sha256': _sha(target_producer_sha256, 'target_producer_sha256'),
            'manifest_sha256': _sha(manifest_sha256, 'manifest_sha256'),
        },
        'training_admitted': False,
        'training_contract': 'NOT_ADMITTED_EXPERIMENT_ONLY',
    })
    missing = sorted(key for key, value in record['source_identity'].items() if value is None)
    record['source_identity_validity'] = 'VALID' if not missing else 'NOT_VERIFIED'
    record['missing_source_evidence'] = missing + ([] if old_status is not None else ['old_status8'])
    return record


def _check_overlay_record(record):
    record = _json_copy(record, 'overlay record')
    if record.get('schema') != SCHEMA or record.get('training_admitted') is not False:
        raise ValueError('overlay schema invalid or training admission attempted')
    if record.get('training_contract') != 'NOT_ADMITTED_EXPERIMENT_ONLY':
        raise ValueError('overlay cannot declare a training contract')
    identity = _identity(record.get('sample_identity'))
    if record.get('sample_id') != identity['sample_id']:
        raise ValueError('sample_id does not match sample_identity')
    if record.get('status_dtype') != 'float32' or record.get('status_order') != list(COMMAND_ORDER) + ['vx', 'vy', 'ax', 'ay']:
        raise ValueError('overlay Status contract mismatch')
    candidate = _vector(record.get('status8'), 8, 'status8')
    validity = record.get('status_validity')
    if validity not in ('VALID', 'NOT_VERIFIED', 'INVALID') or ((candidate is None) != (validity == 'INVALID')):
        raise ValueError('status_validity and candidate presence disagree')
    if candidate is not None:
        _command(candidate[:4])
    old = _vector(record.get('old_status8'), 8, 'old_status8')
    if old is not None:
        _command(old[:4])
    metadata = record.get('diagnostic_metadata', {})
    rebuilt = build_status(
        None if candidate is None else candidate[:4],
        None if candidate is None else candidate[4:],
        metadata.get('navigation_validity'), metadata.get('motion_validity'),
        record.get('navigation_source'), identity,
        navigation_provenance=metadata.get('navigation_provenance'),
        fallback_source=record.get('fallback_source'))
    if rebuilt.status_validity != validity:
        raise ValueError('status_validity contradicts upstream validity or provenance')
    source_identity = record.get('source_identity')
    if not isinstance(source_identity, dict) or set(source_identity) != {
            'feature_sha256', 'target_sha256', 'feature_producer_sha256',
            'target_producer_sha256', 'manifest_sha256'}:
        raise ValueError('source_identity fields missing or unexpected')
    for key, value in source_identity.items():
        _sha(value, key)
    missing = sorted(key for key, value in source_identity.items() if value is None)
    if record.get('missing_source_evidence') != missing + ([] if old is not None else ['old_status8']):
        raise ValueError('source evidence completeness inconsistent')
    if record.get('source_identity_validity') != ('NOT_VERIFIED' if missing else 'VALID'):
        raise ValueError('source identity validity inconsistent')
    generator = record.get('generator', {})
    if generator != generator_identity():
        raise ValueError('generator identity differs from executing builder')
    return record


def write_status_overlay(path, records):
    """Validate all rows, then create a new JSONL file exclusively (no overwrite).

    Duplicate sample IDs and duplicate route/frame identities are rejected even
    if the caller assigns different IDs. This writer never opens source assets.
    """
    rows = []
    ids = set()
    physical_identities = set()
    for raw in records:
        row = _check_overlay_record(raw)
        if row['sample_id'] in ids:
            raise ValueError('duplicate sample_id: ' + row['sample_id'])
        ids.add(row['sample_id'])
        identity = row['sample_identity']
        route = identity.get('route', identity.get('route_id'))
        frame = identity.get('frame', identity.get('frame_id'))
        if route is not None and frame is not None:
            physical = json.dumps([identity.get('dataset_id'), route, int(frame)], sort_keys=True)
            if physical in physical_identities:
                raise ValueError('duplicate route/frame identity: ' + physical)
            physical_identities.add(physical)
        rows.append(json.dumps(row, allow_nan=False, sort_keys=True, separators=(',', ':')) + '\n')
    destination = Path(path)
    if not destination.parent.is_dir():
        raise ValueError('overlay parent must already exist as a dedicated experimental output directory')
    with destination.open('x', encoding='utf-8') as stream:
        stream.writelines(rows)
    return {'path': str(destination), 'rows': len(rows), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
