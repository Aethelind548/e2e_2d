"""Read authenticated Feature/Target artifacts without reopening raw B2D data.

The parent paired manifest pins each sample's artifact SHA and annotation SHA.
Recorded cache lineage/reference checks remain strict; live raw-file auditing is
the separate RecoveryPairedDataset path. Numeric producers are never modified.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sqlite3

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1 import features_cache as fc
from tools.training.diffusiondrive_v1.recovery_paired import RecoveryPairedDataset
from tools.training.diffusiondrive_v1.system_feature_relocation import effective_roots
from tools.training.diffusiondrive_v1.target_storage import load_target_row


def file_identity(path):
    info = Path(path).stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class CacheOnlyPairedDataset(RecoveryPairedDataset):
    def __init__(self, manifest, config, split):
        super().__init__(manifest, config, split)
        if any(s.get('kind') == 'paired-gravity6' for s in manifest['target_sources'].values()):
            raise AdapterError('cache_only_contract_unsupported', 'gravity6 requires its own accepted reader')
        from tools.training.diffusiondrive_v1.runtime import native_configuration
        from tools.b2d_adapter.v1_label_rules import label_rules
        self.native = native_configuration(config)
        for source in manifest['target_sources'].values():
            if (source['producer']['geometry'] != self.native.geometry.provenance()
                    or source['producer']['rules'] != label_rules().identity()):
                raise AdapterError('pair_coordinate_mismatch', 'Frozen Target geometry/label contract differs')
        self._pid = None
        self._connections = {}
        self._roots = {}
        self._watched = {}
        self._index_inodes = {}
        self.reader_mode = 'cache-only-v1'
        self.mount_mode = config.get('mount_check_mode', 'procfs')
        self.file_read_mode = config.get('cache_file_read_mode', 'legacy')
        if self.file_read_mode not in ('legacy', 'verified-single-pass'):
            raise ValueError('Unknown cache file read mode')
        if self.mount_mode not in ('findmnt', 'procfs'):
            raise ValueError('Unknown mount check mode')

    def __getstate__(self):
        state = dict(self.__dict__)
        # Never pickle a SQLite connection or inherit a parent's open handle.
        state.update(_pid=None, _connections={}, _roots={}, _watched={}, _index_inodes={})
        return state

    def close(self):
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _prepare_worker(self):
        if self._pid == os.getpid():
            return
        self.close()
        self._roots = {}
        self._watched = {}
        self._index_inodes = {}
        index = Path(self.manifest['pair_index'])
        before = file_identity(index)
        if fc.file_digest(index) != self.manifest['index_sha256'] or file_identity(index) != before:
            raise AdapterError('cache_only_index_changed', str(index))
        self._watched[str(index)] = before
        registration = self.manifest.get('feature_storage_registration')
        if registration:
            path = Path(registration['path']); before = file_identity(path)
            if fc.file_digest(path) != registration['sha256'] or file_identity(path) != before:
                raise AdapterError('cache_only_registration_changed', str(path))
            self._watched[str(path)] = before
        # Validate relocation and owner identities once per worker. Changes to
        # these small files are still rejected on every subsequent item read.
        for name, source in self.manifest['feature_sources'].items():
            self._roots[name] = effective_roots(self.manifest, source)
            for root in self._roots[name].values():
                marker = Path(root['path']) / 'cache_root.json'
                self._watched[str(marker)] = file_identity(marker)
        for source in self.manifest['target_sources'].values():
            self.target_root(source)
        self._pid = os.getpid()

    def _row(self, source, sample_id, target=False):
        path = Path(source['state']) / 'index.sqlite'
        key = str(path)
        if key not in self._connections:
            # No long-lived read transaction: keep the connection but let each
            # SELECT finish, avoiding locks on unrelated service state updates.
            db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA query_only=ON')
            try:
                meta = {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM meta')}
                if (fc.digest(meta['producer']) != source['producer_sha256']
                        or meta['dataset_id'] != self.config['dataset_id']):
                    raise AdapterError('cache_only_producer_mismatch', key)
                if target and meta['target_root'] != source['target_root']:
                    raise AdapterError('cache_only_root_mismatch', key)
            except BaseException:
                db.close()
                raise
            self._connections[key] = db
            self._index_inodes[key] = file_identity(path)[:2]
        if file_identity(path)[:2] != self._index_inodes[key]:
            raise AdapterError('cache_only_index_replaced', key)
        return self._connections[key].execute('SELECT * FROM samples WHERE sample_id=?', (sample_id,)).fetchone()

    def __getitem__(self, index):
        self._prepare_worker()
        for path, expected in self._watched.items():
            if file_identity(path) != expected:
                raise AdapterError('cache_only_identity_changed', path)
        pair = self.rows[index]
        target_source = self.manifest['target_sources'][pair['target_source']]
        feature_source = self.manifest['feature_sources'][pair['feature_source']]
        target = self._row(target_source, pair['sample_id'], target=True)
        feature = self._row(feature_source, pair['sample_id'])
        if target is None or feature is None or target['state'] != 'complete' or feature['state'] != 'complete':
            raise AdapterError('recovery_pair_index_mismatch', pair['sample_id'])
        target, feature = dict(target), dict(feature)
        if (target['sha256'] != pair['target_sha'] or feature['sha256'] != pair['feature_sha']
                or target['current_anno_sha256'] != pair['current_anno_sha']):
            raise AdapterError('recovery_pair_index_mismatch', pair['sample_id'])
        from tools.training.diffusiondrive_v1.paired_dataset import pair_checks
        pair_checks({'row': feature, 'dataset_id': self.config['dataset_id']}, target,
                    {'route': pair['route'], 'frame': pair['frame']}, self.native, target_source['producer'])
        if self.mount_mode == 'procfs':
            from tools.training.diffusiondrive_v1.procfs_mount_reader import mount_scope
            scope = mount_scope()
        else:
            scope = nullcontext()
        if getattr(self, 'file_read_mode', 'legacy') == 'verified-single-pass':
            from tools.training.diffusiondrive_v1.single_pass_cache_reader import single_pass_scope
            file_scope = single_pass_scope()
        else:
            file_scope = nullcontext()
        with scope, file_scope:
            features = fc.load_feature_row(feature, self._roots[pair['feature_source']], feature_source['producer_sha256'])
            targets = load_target_row(target, self.target_root(target_source), target_source['producer_sha256'])['targets']
        # No RouteIndex, FrameReader, raw annotation, JPG, LAZ or map reads.
        return features, targets
