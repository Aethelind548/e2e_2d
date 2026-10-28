"""Independent Status Overlay v2 reader; never selected by the production fit.

Only explicitly admitted rows replace status_feature. Frozen cache reader owns
artifact SHA, index, relocation and producer validation. No model/GPU is used.
"""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1.status_overlay_contract import READER_SCHEMA, validate_row

VERSION = 'status-overlay-reader-v1'


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_identity(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def load_overlay(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class SelectedSampleView:
    """Explicit sample_id selection over an unchanged authenticated Dataset."""
    def __init__(self, base, sample_ids):
        self.base_dataset = base
        if (not isinstance(sample_ids, list) or not sample_ids
                or any(not isinstance(s, str) or not s.strip() for s in sample_ids)
                or len(set(sample_ids)) != len(sample_ids)):
            raise AdapterError('overlay_selection_invalid', 'explicit unique nonempty sample_ids required')
        positions = {row['sample_id']: i for i, row in enumerate(base.rows)}
        if len(positions) != len(base.rows) or any(s not in positions for s in sample_ids):
            raise AdapterError('overlay_selection_invalid', 'sample outside requested frozen split')
        self._positions = [positions[s] for s in sample_ids]
        self.rows = [copy.deepcopy(base.rows[i]) for i in self._positions]
        self.tokens = [base.tokens[i] for i in self._positions] if getattr(base, 'tokens', None) else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        position = self._positions[index]
        if self.base_dataset.rows[position] != self.rows[index]:
            raise AdapterError('overlay_base_identity_changed', self.rows[index]['sample_id'])
        return self.base_dataset[position]

    def close(self):
        if hasattr(self.base_dataset, 'close'):
            self.base_dataset.close()


class StatusOverlayDataset:
    """Wrap an authenticated base dataset, matching rows by sample_id only.

    The constructor checks *every* selected row before any cache payload read.
    All overlay rows must match a frozen pair; unrelated/unselected split rows
    are allowed but duplicates and altered identities are always rejected.
    A base injection is intended for CPU fixtures; production uses from_files.
    """
    def __init__(self, base_dataset, overlay_rows, *, pair_rows, context, watched_files=None):
        self.base_dataset = base_dataset
        self.context = copy.deepcopy(context)
        self.rows = copy.deepcopy(base_dataset.rows)
        pairs = {}
        physical = set()
        for pair in pair_rows:
            sid = pair.get('sample_id')
            key = (pair.get('route'), pair.get('frame'))
            if not isinstance(sid, str) or not sid.strip() or sid in pairs or key in physical:
                raise AdapterError('overlay_pair_duplicate', sid)
            pairs[sid] = pair
            physical.add(key)
        overlay = {}
        for original in overlay_rows:
            row = copy.deepcopy(original)
            sid = row.get('sample_id') if isinstance(row, dict) else None
            if not isinstance(sid, str) or not sid.strip() or sid in overlay:
                raise AdapterError('overlay_duplicate_or_empty', sid)
            if sid not in pairs:
                raise AdapterError('overlay_unknown_sample', sid)
            errors = validate_row(row, pairs[sid], self._context(pairs[sid]), require_admitted=False)
            if errors:
                raise AdapterError('overlay_contract_rejected', sid + ':' + ','.join(errors))
            overlay[sid] = row
        self._overlay = {}
        selected = set()
        for pair in self.rows:
            sid = pair.get('sample_id')
            if sid in selected or sid not in pairs or pair != pairs[sid]:
                raise AdapterError('overlay_base_identity_mismatch', sid)
            selected.add(sid)
            if sid not in overlay:
                raise AdapterError('overlay_missing_sample', sid)
            row = overlay[sid]
            errors = validate_row(row, pair, self._context(pair), require_admitted=True)
            if errors:
                raise AdapterError('overlay_admission_rejected', sid + ':' + ','.join(errors))
            self._overlay[sid] = row
        if not selected:
            raise AdapterError('overlay_empty_selection', 'No admitted selected samples')
        self._watched_files = dict(watched_files or {})
        self.tokens = list(getattr(base_dataset, 'tokens', []))
        self.reader_mode = VERSION

    def _context(self, pair):
        c = dict(self.context)
        if 'target_producers' in c:
            c['target_producer_sha256'] = c['target_producers'].get(pair.get('target_source'))
        return c

    @classmethod
    def from_files(cls, manifest_path, config, overlay_path, contract_path, split):
        """Validate the unchanged formal reader contract, then wrap one split.

        An empty evidence allowlist intentionally makes this factory NO-GO for
        today's offline navigation. The contract does not confer certification.
        """
        from tools.training.diffusiondrive_v1.recovery_paired import datasets
        initial = {str(Path(p).resolve()): file_identity(p) for p in (contract_path, manifest_path)}
        contract = json.loads(Path(contract_path).read_text())
        manifest = json.loads(Path(manifest_path).read_text())
        if any(file_identity(p) != identity for p, identity in initial.items()):
            raise AdapterError('overlay_identity_changed', 'contract/manifest changed during metadata read')
        from tools.training.diffusiondrive_v1 import status_overlay_contract
        expected = {'manifest_sha256': manifest_path, 'pair_index_sha256': manifest['pair_index'], 'overlay_sha256': overlay_path,
                    'reader_source_sha256': __file__, 'contract_source_sha256': status_overlay_contract.__file__}
        if contract.get('schema') != READER_SCHEMA or contract.get('reader_version') != VERSION:
            raise AdapterError('overlay_reader_contract', 'schema/version')
        watched = {}
        for key, path in expected.items():
            before = file_identity(path)
            if contract.get(key) != file_sha256(path) or before != file_identity(path):
                raise AdapterError('overlay_frozen_file_sha', key)
            watched[str(Path(path).resolve())] = before
        if any(file_identity(p) != identity for p, identity in initial.items()):
            raise AdapterError('overlay_identity_changed', 'contract/manifest changed before hash validation')
        watched[str(Path(contract_path).resolve())] = file_identity(contract_path)
        if contract['pair_index_sha256'] != manifest.get('index_sha256'):
            raise AdapterError('overlay_frozen_index_sha', manifest['pair_index'])
        if config.get('cache_read_mode') != 'cache-only' or split not in ('train', 'validation'):
            raise AdapterError('overlay_reader_contract', 'cache-only and explicit train/validation split required')
        db = sqlite3.connect(Path(manifest['pair_index']).resolve().as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            pairs = [dict(row) for row in db.execute('SELECT * FROM pairs')]
        finally:
            db.close()
        context = dict(contract, feature_producer_sha256=manifest['feature_producer_sha256'],
                       target_producers={k: v['producer_sha256'] for k, v in manifest['target_sources'].items()})
        rows = load_overlay(overlay_path)
        # Fail certification before importing model/native code or opening any
        # tensor payload. This namespace is validation-only, never a read path.
        selection = contract.get('selected_sample_ids_by_split')
        planned = SimpleNamespace(rows=[p for p in pairs if p['split'] == split])
        if selection is not None:
            if not isinstance(selection, dict) or split not in selection:
                raise AdapterError('overlay_selection_invalid', 'requested split missing')
            planned = SelectedSampleView(planned, selection[split])
        cls(planned, rows,
            pair_rows=pairs, context=context, watched_files=watched)
        base, _ = datasets(manifest_path, config, None)
        selected_base = SelectedSampleView(base[split], selection[split]) if selection is not None else base[split]
        result = cls(selected_base, rows, pair_rows=pairs, context=context, watched_files=watched)
        for path, identity in watched.items():
            if file_identity(path) != identity:
                raise AdapterError('overlay_identity_changed', path)
        return result

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        for path, expected in self._watched_files.items():
            if file_identity(path) != expected:
                raise AdapterError('overlay_identity_changed', path)
        # Index selects an existing base row; sample_id provides overlay identity.
        pair = self.rows[index]
        if self.base_dataset.rows[index] != pair:
            raise AdapterError('overlay_base_identity_changed', pair['sample_id'])
        row = self._overlay[pair['sample_id']]
        errors = validate_row(row, pair, self._context(pair), require_admitted=True)
        if errors:
            raise AdapterError('overlay_admission_rejected', ','.join(errors))
        features, targets = self.base_dataset[index]
        original = features.get('status_feature')
        if not isinstance(original, torch.Tensor) or original.dtype != torch.float32 or tuple(original.shape) != (8,) or original.device.type != 'cpu' or not torch.isfinite(original).all():
            raise AdapterError('overlay_original_status', pair['sample_id'])
        old = torch.tensor(row['old_status8'], dtype=torch.float32)
        if not torch.equal(original, old):
            raise AdapterError('overlay_original_status_mismatch', pair['sample_id'])
        # Shallow copy retains camera/lidar tensors and target object unchanged.
        replaced = dict(features)
        replaced['status_feature'] = torch.tensor(row['status8'], dtype=torch.float32)
        return replaced, targets

    def close(self):
        close = getattr(self.base_dataset, 'close', None)
        if close is not None:
            close()
