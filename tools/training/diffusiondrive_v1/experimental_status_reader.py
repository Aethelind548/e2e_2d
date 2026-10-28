"""Research-only Status dataset for the offline expert-route experiment.

This reader is deliberately separate from the strict ``StatusOverlayDataset``:
it does **not** grant certification or admission.  It reads a versioned training
index (``status_training_sample_index.jsonl``) and replaces only
``status_feature`` with the indexed overlay values, while the frozen cache reader
still owns Feature/Target identity, producers, relocation and SHA validation.

Guarantees per row:

* the sample exists in the frozen pair index with the same route/frame/split;
* the original cached Status equals the indexed ``old_status8`` (no silent cache
  drift);
* the replacement is finite float32 ``(8,)`` with a valid one-hot Navigation4;
* Motion4 equals the indexed frozen A0 values.
"""
import hashlib
import json
from pathlib import Path

from tools.b2d_adapter.errors import AdapterError

VERSION = 'experimental-status-dataset-v1'
INDEX_SCHEMA = 'status-training-sample-index-v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_index(path, split):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') == split:
                rows.append(row)
    if not rows:
        raise AdapterError('experimental_status_empty_split', split)
    return rows


class ExperimentalStatusDataset:
    def __init__(self, manifest_path, index_path, base_config_path, split,
                 manifest_sha256=None, limit=0):
        import torch
        from tools.training.diffusiondrive_v1.cache_only_reader import CacheOnlyPairedDataset
        self.torch = torch
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('schema') != 'status-training-dataset-manifest-v1':
            raise AdapterError('experimental_status_manifest', 'unexpected schema')
        if manifest_sha256 and sha256_file(self.manifest_path) != manifest_sha256:
            raise AdapterError('experimental_status_manifest_sha', str(self.manifest_path))
        if sha256_file(index_path) != self.manifest['index']['sha256']:
            raise AdapterError('experimental_status_index_sha', str(index_path))
        self.index_path = Path(index_path)
        self.split = split
        config = json.loads(Path(base_config_path).read_text())
        config['cache_read_mode'] = 'cache-only'
        paired = json.loads(Path(config['paired_manifest']).read_text())
        self.base = CacheOnlyPairedDataset(paired, config, split)
        positions = {row['sample_id']: index for index, row in enumerate(self.base.rows)}
        rows = load_index(index_path, split)
        if limit:
            rows = rows[:limit]
        selected, mapping = [], []
        for row in rows:
            position = positions.get(row['sample_id'])
            if position is None:
                raise AdapterError('experimental_status_unknown_sample', row['sample_id'])
            if self.base.rows[position]['route'] != row['route']:
                raise AdapterError('experimental_status_route_mismatch', row['sample_id'])
            selected.append(row)
            mapping.append(position)
        self.rows = selected
        self.tokens = [self.base.tokens[i] for i in mapping]
        self._positions = mapping
        self.reader_mode = VERSION

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features, targets = self.base[self._positions[index]]
        status = self.torch.tensor(row['status8'], dtype=self.torch.float32)
        if tuple(status.shape) != (8,) or not self.torch.isfinite(status).all():
            raise AdapterError('experimental_status_shape', row['sample_id'])
        if float(status[:4].sum()) != 1.0 or not bool(((status[:4] == 0) | (status[:4] == 1)).all()):
            raise AdapterError('experimental_status_one_hot', row['sample_id'])
        if not self.torch.equal(status[4:], self.torch.tensor(row['a0_motion4'], dtype=self.torch.float32)):
            raise AdapterError('experimental_status_motion', row['sample_id'])
        original = features.get('status_feature')
        old = self.torch.tensor(row['old_status8'], dtype=self.torch.float32)
        if (not isinstance(original, self.torch.Tensor) or original.dtype != self.torch.float32
                or tuple(original.shape) != (8,) or not self.torch.equal(original, old)):
            raise AdapterError('experimental_status_original_mismatch', row['sample_id'])
        replaced = dict(features)
        replaced['status_feature'] = status
        return replaced, targets

    def close(self):
        close = getattr(self.base, 'close', None)
        if close is not None:
            close()
