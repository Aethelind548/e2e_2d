"""One physical Features root per disk, explicitly shared by registered job indexes.

Delegate storage ownership only. Reuse the frozen cache implementation, payload
contract, checksum verification, atomic writes, quotas and whole-root locks.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from tools.training.diffusiondrive_v1 import features_cache as core

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / 'tools/training/diffusiondrive_v1/features_shared_storage.json'
REGISTRY_SCHEMA = 'b2d-v1-shared-features-storage-v1'
ORIGINAL_STORAGE = core.CacheStorage


def read_registry(path=None):
    registry = json.loads(Path(path or REGISTRY).read_text())
    if registry['schema'] != REGISTRY_SCHEMA:
        raise core.StoragePause('Unknown shared Features storage registry')
    return registry


def marker(registry, root_id):
    return {'schema':core.SCHEMA, 'root_id':root_id, 'dataset_id':registry['dataset_id'],
            'index_id':registry['roots'][root_id]['storage_owner_id']}


def check_owner(registry, root_id, spec, index_id, dataset_id, producer_sha):
    entry = registry['roots'][root_id]
    if (spec != entry['spec'] or dataset_id != registry['dataset_id']
            or producer_sha != registry['producer_sha256']
            or index_id not in {v['index_id'] for v in entry['members']}):
        raise core.StoragePause('Unregistered shared cache owner/spec/producer: ' + root_id)
    core.active_mount(spec)
    if json.loads((Path(spec['path'])/'cache_root.json').read_text()) != marker(registry,root_id):
        raise core.StoragePause('Shared cache storage owner changed: ' + root_id)


def validate_registration(config, index_id=None, producer_sha=None):
    path = config.get('shared_feature_roots')
    if not path: return {}
    registry = read_registry(ROOT/path)
    state = (ROOT/config['state_dir']).resolve()
    with sqlite3.connect('file:'+str(state/'index.sqlite')+'?mode=ro',uri=True) as db:
        meta = {k:json.loads(v) for k,v in db.execute("SELECT key,value FROM meta WHERE key IN ('index_id','producer')")}
    if index_id is not None and index_id != meta['index_id']:
        raise core.StoragePause('Shared cache job index changed')
    actual_producer = core.digest(meta['producer'])
    if producer_sha is not None and producer_sha != actual_producer:
        raise core.StoragePause('Shared cache numeric producer changed')
    for root_id, entry in registry['roots'].items():
        if not any(v['index_id']==meta['index_id'] and Path(v['state']).resolve()==state for v in entry['members']):
            raise core.StoragePause('Unregistered job state for shared root')
        check_owner(registry,root_id,config['cache_roots'][root_id],meta['index_id'],config['dataset_id'],actual_producer)
    return registry['roots']


class SharedCacheStorage(ORIGINAL_STORAGE):
    def __init__(self, config, producer_sha, index_id):
        shared = validate_registration(config,index_id,producer_sha)
        private_config = dict(config,cache_roots={k:v for k,v in config['cache_roots'].items() if k not in shared})
        super().__init__(private_config,producer_sha,index_id)
        try:
            for root_id, entry in shared.items():
                # Both jobs use the same original whole-root lock and quota.
                # The registry authorizes delegation to this storage owner ID.
                item = ORIGINAL_STORAGE(dict(config,cache_roots={root_id:entry['spec']}),producer_sha,entry['storage_owner_id'])
                self.used.update(item.used)
                self.locks.extend(item.locks)
                item.locks = []
            self.config = config
            self.roots = config['cache_roots']
        except BaseException:
            self.close()
            raise


@contextmanager
def storage_scope(config):
    original = core.CacheStorage
    if config.get('shared_feature_roots'): core.CacheStorage = SharedCacheStorage
    try:
        yield
    finally:
        core.CacheStorage = original
