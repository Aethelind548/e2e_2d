"""Explicit, SHA-pinned Target locations; frozen logical roots remain unchanged.

Only storage addresses change. Numeric producers, payload SHA, pair identities
and original SQLite metadata are still checked by the existing cache readers.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1.features_cache import digest
from tools.training.diffusiondrive_v1.target_cache import storage_check

SCHEMA = 'b2d-v1-verified-target-relocation-v1'
VERSION = 'target-location-reader-v1'


def file_digest(path):
    # Registration/owner JSON is metadata, never a tensor buffer pending decode
    # in single_pass_scope, even if this module is first imported in that scope.
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def identity(path):
    st = Path(path).stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


class TargetLocationResolver:
    """Worker-local registration cache; small identities are checked every read."""

    def __init__(self, manifest, config):
        declared = manifest.get('target_storage_registration')
        configured = config.get('target_storage_registration')
        if declared is not None and configured is not None and declared != configured:
            raise AdapterError('target_relocation_registration', 'Manifest/config registrations differ')
        self.reference = configured if configured is not None else declared
        self.allowed_sha = config.get('allowed_target_storage_registration_sha256')
        self.reader_sha = config.get('target_location_reader_sha256')
        self._pid = None
        self._watched = {}
        self._roots = {}
        self._registration = None
        if self.reference is not None:
            if (not isinstance(self.reference, dict) or set(self.reference) != {'path', 'sha256'}
                    or self.allowed_sha != self.reference['sha256']
                    or not isinstance(self.allowed_sha, str) or len(self.allowed_sha) != 64
                    or self.reader_sha != file_digest(Path(__file__))):
                raise AdapterError('target_relocation_registration', 'Explicit registration and reader SHA required')

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_pid=None, _watched={}, _roots={}, _registration=None)
        return state

    def _watch_digest(self, path, expected):
        path = Path(path)
        before = identity(path)
        if path.is_symlink() or file_digest(path) != expected or identity(path) != before:
            raise AdapterError('target_relocation_identity', str(path))
        self._watched[str(path)] = before

    def _prepare(self):
        if self._pid == os.getpid():
            return
        self._watched, self._roots = {}, {}
        path = Path(self.reference['path'])
        self._watch_digest(path, self.allowed_sha)
        registration = json.loads(path.read_text())
        if identity(path) != self._watched[str(path)]:
            raise AdapterError('target_relocation_identity', str(path))
        if (registration.get('schema') != SCHEMA or registration.get('verified') is not True
                or not isinstance(registration.get('roots'), dict) or not registration['roots']):
            raise AdapterError('target_relocation_unverified', str(path))
        self._registration = registration
        self._pid = os.getpid()

    def resolve(self, spec, producer_sha256, state):
        if self.reference is None:
            return spec
        self._prepare()
        for path, expected in self._watched.items():
            if (identity(path) if Path(path).exists() else None) != expected:
                raise AdapterError('target_relocation_identity', path)
        state = str(Path(state).resolve())
        key = (state, producer_sha256, digest(spec))
        if key in self._roots:
            return dict(self._roots[key])
        entry = self._registration['roots'].get(state)
        if (not entry or entry.get('from') != spec
                or entry.get('producer_sha256') != producer_sha256):
            raise AdapterError('target_relocation_source', 'Unregistered source root/producer: ' + state)
        index = Path(state) / 'index.sqlite'
        index_before = identity(index)
        with sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True) as db:
            db.execute('PRAGMA query_only=ON')
            meta = {key: json.loads(value) for key, value in db.execute(
                "SELECT key,value FROM meta WHERE key IN ('target_root','producer')")}
        if identity(index) != index_before:
            raise AdapterError('target_relocation_identity', str(index))
        self._watched[str(index)] = index_before
        wal = Path(str(index) + '-wal')
        self._watched[str(wal)] = identity(wal) if wal.exists() else None
        if meta.get('target_root') != spec or digest(meta.get('producer')) != producer_sha256:
            raise AdapterError('target_relocation_source', 'Frozen source index root/producer differs')
        destination = entry.get('to')
        if (not isinstance(destination, dict) or destination.get('root_id') != spec.get('root_id')
                or Path(destination.get('path', '')).resolve() == Path(spec['path']).resolve()
                or not Path(destination.get('path', '')).is_absolute()):
            raise AdapterError('target_relocation_destination', state)
        storage_check(destination)
        owners = entry.get('destination_owner_sha256', {})
        if 'target_root.json' not in owners or set(owners) - {'target_root.json', 'target_storage_encoding.json'}:
            raise AdapterError('target_relocation_destination', 'Missing/unknown destination owner marker')
        for name, sha256 in owners.items():
            self._watch_digest(Path(destination['path']) / name, sha256)
        owner = json.loads((Path(destination['path']) / 'target_root.json').read_text())
        if owner.get('state_id') != state or owner.get('root_id') != spec['root_id']:
            raise AdapterError('target_relocation_destination', 'Destination owner state/root mismatch')
        for path, expected in self._watched.items():
            if (identity(path) if Path(path).exists() else None) != expected:
                raise AdapterError('target_relocation_identity', path)
        self._roots[key] = dict(destination)
        return dict(destination)
