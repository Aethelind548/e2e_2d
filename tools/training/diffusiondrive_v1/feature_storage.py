"""Register/expand the system-NVMe Features tier without changing its producer."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import time

from tools.training.diffusiondrive_v1.features_cache import (
    SCHEMA, GIB, CacheStorage, StoragePause, active_mount, durable_json, digest, feature_producer)

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / 'tools/training/diffusiondrive_v1/features_system_storage.json'
CONFIG = ROOT / 'tools/training/diffusiondrive_v1/features_remaining_20261015.json'
ROOT_ID = 'cache_system'
SYSTEM_PATH = Path('/data/diffusiondrive_v1_tensors/features')
SYSTEM_UUID = '932410d9-a546-40dd-aeda-09e390df924b'


def capacity_check(budget_gib, minimum_free_gib, used_bytes, free_bytes):
    """Grant only capacity that physically exists, including the free-space floor."""
    if budget_gib <= 0 or minimum_free_gib < 40:
        raise StoragePause('System budget must be positive; reserve at least 40 GiB')
    budget = int(budget_gib * GIB)
    if budget < used_bytes:
        raise StoragePause('Requested quota is smaller than existing system cache')
    required = budget - used_bytes + int(minimum_free_gib * GIB)
    if free_bytes < required:
        raise StoragePause('System capacity not yet available: need %.2f GiB free, have %.2f GiB' %
                           (required / GIB, free_bytes / GIB))


def index_identity(config):
    path = ROOT / config['state_dir'] / 'index.sqlite'
    db = sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True)
    try:
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta WHERE key IN ('index_id','producer','dataset_id')")}
    finally:
        db.close()
    if meta['dataset_id'] != config['dataset_id']:
        raise StoragePause('System tier dataset differs from existing index')
    return meta


def validate_extension(config):
    """Called by both launch paths; registered roots keep their original owner."""
    from tools.training.diffusiondrive_v1.feature_shared_storage import validate_registration
    validate_registration(config)
    extensions = config.get('storage_extensions', {})
    if not extensions:
        if ROOT_ID in config['cache_roots']:
            raise StoragePause('System Features tier is not registered')
        return
    if set(extensions) != {ROOT_ID}:
        raise StoragePause('Unknown storage extension; register it explicitly')
    plan = json.loads((ROOT / extensions[ROOT_ID]['plan']).read_text())
    spec = config['cache_roots'].get(ROOT_ID)
    if (plan['schema'] != 'b2d-v1-system-features-storage-v1'
        or plan['dataset_id'] != config['dataset_id']
        or plan['state_dir'] != config['state_dir'] or spec != plan['spec']
        or not spec or spec['path'] != str(SYSTEM_PATH) or spec['mountpoint'] != '/'
        or spec['uuid'] != SYSTEM_UUID or spec['fstype'] != 'ext4'
        or list(config['cache_roots'])[0] != ROOT_ID):
        raise StoragePause('System tier config/registered identity or priority differs')
    meta = index_identity(config)
    if meta['index_id'] != plan['index_id'] or digest(meta['producer']) != plan['producer_sha256']:
        raise StoragePause('System tier index/producer owner differs')
    marker = Path(spec['path']) / 'cache_root.json'
    expected = {'schema': SCHEMA, 'root_id': ROOT_ID, 'dataset_id': config['dataset_id'], 'index_id': meta['index_id']}
    if not marker.exists() or json.loads(marker.read_text()) != expected:
        raise StoragePause('System Features cache owner marker differs')


def configure(config_path, budget_gib=250, minimum_free_gib=40):
    """Stop old writer first; update only routing/quotas, never route/sample IDs."""
    import fcntl
    from tools.training.diffusiondrive_v1.feature_service import alive
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text()); state = ROOT / config['state_dir']
    with (state / 'launch.lock').open('a') as launch:
        fcntl.flock(launch, fcntl.LOCK_EX)
        if alive(state): raise StoragePause('Stop Features before changing storage quotas')
        with (state / 'writer.lock').open('a') as writer:
            try: fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise StoragePause('Features writer still active') from exc
            meta = index_identity(config)
            if digest(feature_producer(config)) != digest(meta['producer']):
                raise StoragePause('Existing numeric Features producer changed')
            spec = {'path': str(SYSTEM_PATH), 'mountpoint': '/', 'uuid': SYSTEM_UUID,
                    'fstype': 'ext4', 'budget_gib': budget_gib, 'minimum_free_gib': minimum_free_gib}
            if not SYSTEM_PATH.is_dir() or not os.access(SYSTEM_PATH, os.W_OK):
                raise StoragePause('Prepare writable directory first: ' + str(SYSTEM_PATH))
            mounted = active_mount(spec, writing=True)
            marker = SYSTEM_PATH / 'cache_root.json'
            expected = {'schema': SCHEMA, 'root_id': ROOT_ID, 'dataset_id': config['dataset_id'], 'index_id': meta['index_id']}
            if marker.exists():
                if json.loads(marker.read_text()) != expected: raise StoragePause('System cache owned by another index')
            elif any(SYSTEM_PATH.iterdir()):
                raise StoragePause('Do not adopt an unmarked nonempty system cache directory')
            used = CacheStorage.measure(SYSTEM_PATH)
            capacity_check(budget_gib, minimum_free_gib, used, mounted['free_bytes'])
            if PLAN.exists():
                prior = json.loads(PLAN.read_text())
                if (prior['index_id'] != meta['index_id'] or prior['dataset_id'] != config['dataset_id']
                    or prior['state_dir'] != config['state_dir']
                    or prior['producer_sha256'] != digest(meta['producer'])
                    or any(prior['spec'][k] != spec[k] for k in ('path','mountpoint','uuid','fstype'))):
                    raise StoragePause('Existing system tier registration belongs to another identity')
            if ROOT_ID in config['cache_roots'] and any(config['cache_roots'][ROOT_ID][k] != spec[k]
                                                       for k in ('path','mountpoint','uuid','fstype')):
                raise StoragePause('Do not redirect an existing system tier to a different volume')
            plan = {'schema': 'b2d-v1-system-features-storage-v1', 'dataset_id': config['dataset_id'],
                    'state_dir': config['state_dir'], 'config': str(config_path.relative_to(ROOT)),
                    'index_id': meta['index_id'], 'producer_sha256': digest(meta['producer']),
                    'spec': spec, 'planned_additional_gib': 300, 'planned_budget_after_expansion_gib': 550,
                    'updated': time.time(), 'verified_free_bytes': mounted['free_bytes']}
            old = json.loads(json.dumps(config))
            config['cache_roots'] = {ROOT_ID: spec, **{k: v for k, v in config['cache_roots'].items() if k != ROOT_ID}}
            config['storage_extensions'] = {ROOT_ID: {'plan': str(PLAN.relative_to(ROOT))}}
            backup = state / 'config_before_system_storage.json'
            if not backup.exists(): durable_json(backup, old)
            if not marker.exists(): durable_json(marker, expected)
            # Both files are atomic. A crash between them leaves a safe launch
            # mismatch; configure can be rerun after checking this recorded plan.
            durable_json(PLAN, plan); durable_json(config_path, config)
            durable_json(SYSTEM_PATH.parent / 'storage_plan.json', plan)
            validate_extension(config)
            return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['configure', 'expand', 'status'])
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--budget-gib', type=int)
    args = p.parse_args()
    if args.action in ('configure','expand'):
        if args.action == 'expand' and args.budget_gib is None: p.error('expand requires --budget-gib')
        plan = configure(args.config, args.budget_gib if args.budget_gib is not None else 250)
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    cfg = json.loads(args.config.read_text())
    if ROOT_ID not in cfg['cache_roots']: raise StoragePause('System Features tier not configured yet')
    validate_extension(cfg)
    spec = cfg['cache_roots'][ROOT_ID]; info = active_mount(spec)
    print(json.dumps({'root': spec, 'mounted': info,
                      'used_gib': round(CacheStorage.measure(SYSTEM_PATH) / GIB, 3),
                      'free_gib': round(info['free_bytes'] / GIB, 3)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try: main()
    except (StoragePause, OSError) as exc:
        print('Features system storage remains unchanged/paused: ' + str(exc)); raise SystemExit(2)
