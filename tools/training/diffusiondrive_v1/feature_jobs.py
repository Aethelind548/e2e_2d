"""Independent route scheduling and unified reading across immutable feature jobs.

Numeric builders and features_cache.py remain unchanged. New jobs own separate
state/root directories; quotas reserve existing usage within global disk limits.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))

from tools.training.diffusiondrive_v1.features_cache import (
    GIB, CacheStorage, FeatureCacheDataset, StoragePause, active_mount, connect,
    digest, discover, durable_json, feature_producer, index_summary, pin)
from tools.training.diffusiondrive_v1.feature_service import alive


def readonly_db(path):
    db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def config_and_state(path):
    cfg = json.loads(Path(path).read_text())
    return cfg, (ROOT / cfg['state_dir']).resolve()


def require_writable_root(spec):
    """Keep the producer's mount gate; add actionable CLI diagnostics only."""
    try:
        return active_mount(spec, writing=True)
    except StoragePause as exc:
        details = ''
        try:
            mount = Path(spec['mountpoint'])
            result = subprocess.run(['findmnt', '-J', '-T', str(mount), '-o',
                                     'TARGET,SOURCE,UUID,FSTYPE,OPTIONS,MAJ:MIN'],
                                    capture_output=True, text=True, check=True, timeout=10)
            dev = mount.stat().st_dev
            rows = [r for r in json.loads(result.stdout)['filesystems']
                    if r['target'] == str(mount) and r['maj:min'] == f'{os.major(dev)}:{os.minor(dev)}']
            if rows:
                actual = rows[-1]
                details = '\n实际挂载：' + json.dumps(actual, ensure_ascii=False)
                flags = set(actual['options'].split(','))
                if flags & {'emergency_ro', 'shutdown'}:
                    details += '\n文件系统处于故障保护状态。先恢复硬盘连接/设备健康，并在卸载后检查文件系统；不要强制恢复写入。'
                elif 'ro' in flags:
                    details += '\n当前挂载或执行环境为只读。请在宿主终端检查 findmnt；确认文件系统健康且恢复可写后再 resume。'
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            pass
        raise StoragePause(str(exc) + details) from exc


def discover_routes(base):
    return sorted(e.name for e in os.scandir(base)
                  if e.is_dir(follow_symlinks=False) and (Path(e.path) / 'anno').is_dir())


def plan(old_config_path, tag, report_dir):
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('Use a simple job tag')
    old, old_state = config_and_state(old_config_path)
    if alive(old_state):
        raise StoragePause('Baseline is still writing; freeze it before allocating remaining budgets')
    old_db = readonly_db(old_state / 'index.sqlite')
    producer = json.loads(old_db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
    old_requested = json.loads(old_db.execute("SELECT value FROM meta WHERE key='requested_routes'").fetchone()[0])
    if old_requested != old['routes'] or digest(producer) != digest(feature_producer(old)):
        raise StoragePause('Baseline route/producer identity differs')
    all_routes = discover_routes(old['dataset_root'])
    pending = sorted(set(all_routes) - set(old_requested))
    if not pending or set(old_requested) - set(all_routes):
        raise StoragePause('No unprocessed routes or baseline source routes are missing')
    registry = ROOT / 'tools/training/diffusiondrive_v1/features_jobs.json'
    if registry.exists():
        raise StoragePause('A job collection already exists; reuse it instead of replanning the same routes')
    new_path = ROOT / 'tools/training/diffusiondrive_v1' / ('features_' + tag + '.json')
    if new_path.exists():
        raise StoragePause('Job tag already planned')
    new = json.loads(json.dumps(old))
    new.update(routes=pending, state_dir='outputs/diffusiondrive_v1/features_jobs/' + tag)
    budget = {}
    for root_id, spec in old['cache_roots'].items():
        mounted = active_mount(spec)  # Writing check is repeated by start in host view.
        old_bytes = CacheStorage.measure(Path(spec['path']))
        # Conservative extra 1 GiB covers directory metadata and bounded repairs
        # in the frozen baseline; never grant the full global quota to both jobs.
        new_quota = math.floor(spec['budget_gib'] - old_bytes / GIB - 1)
        if new_quota <= 0 or mounted['free_bytes'] < spec['minimum_free_gib'] * GIB:
            raise StoragePause('No safe remaining allocation on ' + root_id)
        new_spec = new['cache_roots'][root_id]
        new_spec['path'] = str(Path(spec['path']).with_name(Path(spec['path']).name + '_' + tag))
        new_spec['budget_gib'] = new_quota
        if Path(new_spec['path']).exists():
            raise StoragePause('New cache root already exists: ' + new_spec['path'])
        budget[root_id] = {'global_budget_gib': spec['budget_gib'], 'baseline_allocated_bytes': old_bytes,
                           'new_job_budget_gib': new_quota, 'reserved_extra_gib': 1,
                           'minimum_free_gib': spec['minimum_free_gib'], 'mount': mounted}
    if digest(feature_producer(new)) != digest(producer):
        raise StoragePause('Expansion unexpectedly changed the producer')
    state = ROOT / new['state_dir']
    db = connect(state / 'index.sqlite')
    pin(db, 'producer', producer)
    pin(db, 'dataset_id', new['dataset_id'])
    pin(db, 'requested_routes', pending)
    discover(db, new)
    # Cover progress queries on the large pending queue; the serialized tensor
    # contract and existing cache implementation remain untouched.
    db.execute('CREATE INDEX IF NOT EXISTS progress_roots ON samples(state,root_id,bytes)')
    db.execute('CREATE INDEX IF NOT EXISTS progress_errors ON samples(state,error_code)')
    db.commit()
    summary = index_summary(db)
    db.close()
    durable_json(new_path, new)
    collection = {'schema': 'b2d-v1-feature-jobs-v1', 'producer_sha256': digest(producer),
                  'dataset_id': old['dataset_id'],
                  'jobs': [{'job_id': 'baseline', 'config': str(Path(old_config_path).resolve().relative_to(ROOT))},
                           {'job_id': tag, 'config': str(new_path.relative_to(ROOT))}]}
    durable_json(registry, collection)
    result = {'total_routes': len(all_routes), 'baseline_routes': len(old_requested),
              'unprocessed_routes': len(pending), 'new_candidate_index': summary,
              'route_sets_disjoint': True, 'producer_unchanged': True, 'producer_sha256': digest(producer),
              'new_config': str(new_path), 'new_state': str(state), 'budget': budget,
              'baseline_index': index_summary(old_db), 'collection': str(registry)}
    old_db.close()
    durable_json(report_dir / 'plan.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


class FeatureJobsDataset:
    """One features-only Dataset over job indexes, without moving any tensor files."""
    def __init__(self, collection_path, expected_producer, sample_ids=None):
        collection_path = Path(collection_path).resolve()
        collection = json.loads(collection_path.read_text())
        if collection['schema'] != 'b2d-v1-feature-jobs-v1' or collection['producer_sha256'] != digest(expected_producer):
            raise StoragePause('Collection producer differs')
        self.datasets, self.rows, self.locations, self.sample_ids = [], [], [], []
        seen = set()
        requested = set(sample_ids) if sample_ids is not None else None
        for entry in collection['jobs']:
            cfg, state = config_and_state(ROOT / entry['config'])
            if cfg['dataset_id'] != collection['dataset_id']:
                raise StoragePause('Collection dataset identity differs')
            db = readonly_db(state / 'index.sqlite')
            ids = [r[0] for r in db.execute("SELECT sample_id FROM samples WHERE state='complete' ORDER BY route,frame")]
            db.close()
            if requested is not None:
                ids = [s for s in ids if s in requested]
            dataset = FeatureCacheDataset(state / 'index.sqlite', cfg['cache_roots'], expected_producer, sample_ids=ids)
            owner = len(self.datasets)
            self.datasets.append(dataset)
            for local, row in enumerate(dataset.rows):
                if row['sample_id'] in seen:
                    raise StoragePause('Duplicate sample across jobs: ' + row['sample_id'])
                seen.add(row['sample_id'])
                self.rows.append(dict(row, job_id=entry['job_id']))
                self.locations.append((owner, local))
                self.sample_ids.append(row['sample_id'])
        if requested is not None:
            missing = requested - seen
            if missing:
                raise KeyError('Requested supervised feature sample is missing: ' + str(next(iter(missing))))
            lookup = {s: i for i, s in enumerate(self.sample_ids)}
            ordering = [lookup[s] for s in sample_ids]
            self.rows = [self.rows[i] for i in ordering]
            self.locations = [self.locations[i] for i in ordering]
            self.sample_ids = list(sample_ids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        owner, local = self.locations[i]
        return self.datasets[owner][local]


def export_collection(path):
    collection = json.loads(path.read_text())
    result = {'schema': 'b2d-v1-feature-jobs-manifests-v1', 'producer_sha256': collection['producer_sha256'], 'jobs': []}
    for entry in collection['jobs']:
        cfg, state = config_and_state(ROOT / entry['config'])
        command = [cfg['python'], '-m', 'tools.training.diffusiondrive_v1.feature_service', 'export',
                   '--config', str(ROOT / entry['config'])]
        subprocess.run(command, cwd=ROOT, check=True)
        result['jobs'].append(dict(entry, manifest=str((state / 'features_manifest.json').relative_to(ROOT))))
    dest = path.with_name('features_manifests.json')
    durable_json(dest, result)
    print(json.dumps({'collection_manifest': str(dest)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'start', 'resume', 'status', 'logs', 'stop', 'export', 'verify'])
    p.add_argument('--tag', default='remaining_20261015')
    p.add_argument('--config', type=Path)
    p.add_argument('--baseline-config', type=Path, default=ROOT / 'tools/training/diffusiondrive_v1/features_cache.json')
    p.add_argument('--limit', type=int)
    p.add_argument('--lines', type=int, default=15)
    p.add_argument('--retry-failed', action='store_true')
    args = p.parse_args()
    report = ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/features_expansion'
    report.mkdir(exist_ok=True)
    if args.action == 'plan':
        plan(args.baseline_config, args.tag, report)
    elif args.action == 'export':
        export_collection(ROOT / 'tools/training/diffusiondrive_v1/features_jobs.json')
    else:
        config_path = args.config or ROOT / 'tools/training/diffusiondrive_v1' / ('features_' + args.tag + '.json')
        cfg, state = config_and_state(config_path)
        if args.action in ('start', 'resume'):
            from tools.training.diffusiondrive_v1.feature_storage import validate_extension
            validate_extension(cfg)
            old, old_state = config_and_state(args.baseline_config)
            if alive(old_state):
                raise StoragePause('Baseline writer restarted; suspend expansion until budgets are checked')
            from tools.training.diffusiondrive_v1.feature_shared_storage import validate_registration
            shared = validate_registration(cfg)
            for root_id, spec in cfg['cache_roots'].items():
                require_writable_root(spec)
                if root_id in shared:
                    # Registered roots have one global quota and whole-root
                    # writer lock; do not reserve the same files twice.
                    if old['cache_roots'][root_id] != spec:
                        raise StoragePause('Shared-root specs differ between job configs')
                    continue
                if root_id not in old['cache_roots']:
                    if root_id != 'cache_system' or root_id not in cfg.get('storage_extensions', {}):
                        raise StoragePause('Unregistered additional cache root: ' + root_id)
                    continue
                baseline_bytes = CacheStorage.measure(Path(old['cache_roots'][root_id]['path']))
                if baseline_bytes + (spec['budget_gib'] + 1) * GIB > old['cache_roots'][root_id]['budget_gib'] * GIB:
                    raise StoragePause('Frozen baseline grew beyond quota reservation')
        command = [cfg['python'], '-m', 'tools.training.diffusiondrive_v1.feature_service', args.action,
                   '--config', str(config_path)]
        if args.limit is not None:
            command += ['--limit', str(args.limit)]
        if args.retry_failed:
            command += ['--retry-failed']
        if args.action == 'logs':
            command += ['--lines', str(args.lines)]
        raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == '__main__':
    try:
        main()
    except StoragePause as exc:
        print('Features 任务保持安全暂停：' + str(exc), file=sys.stderr)
        print('本次未启动后台进程，未删除或重新生成缓存。恢复磁盘可读写后使用相同 resume 命令。', file=sys.stderr)
        raise SystemExit(2)
