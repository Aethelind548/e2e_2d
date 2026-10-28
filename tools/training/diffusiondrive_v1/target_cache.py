"""Independent original-seven-class supervision; no feature decoding or rebuild."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1.features_cache import (
    GIB, StoragePause, digest, durable_json, file_digest, safe_path, sample_id, directory_sync)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'b2d-v1-targets-only-v1'
CONTRACT_PATH = Path(__file__).with_name('v1_target_contract.json')


def target_contract(native, model):
    from tools.b2d_adapter.v1_builders import validate_v1_config, BEV_CLASSES
    from tools.b2d_adapter.v1_label_rules import label_rules
    validate_v1_config(native, model)
    contract = json.loads(CONTRACT_PATH.read_text())
    assert list(BEV_CLASSES) == contract['classes']
    assert label_rules().name == contract['label_rules']['name'] and label_rules().version == contract['label_rules']['version']
    return contract


def producer_identity(config, model, native):
    import cv2
    import numpy
    import shapely
    import torch
    from tools.b2d_adapter.v1_label_rules import label_rules
    rules = label_rules(config.get('label_rules'))
    if rules.identity() != label_rules('v1_original_7class').identity():
        raise AdapterError('target_contract_mismatch', 'Only frozen original seven-class targets accepted')
    files = ['v1_builders.py', 'v1_bev.py', 'v1_label_rules.py', 'v1_geometry.py', 'native_builders.py',
             'native_config.py', 'config.py', 'trajectory_target.py', 'temporal_sampler.py',
             'frame_reader.py', 'route_index.py', 'coordinate_transform.py', 'navsim_bridge.py']
    hashes = {p: file_digest(ROOT / 'tools/b2d_adapter' / p) for p in files}
    hashes['target_cache.py'] = file_digest(Path(__file__))
    for p in ['transfuser_config.py', 'transfuser_features.py']:
        hashes['official/' + p] = file_digest(ROOT / 'repositories/DiffusionDrive/navsim/agents/diffusiondrive' / p)
    maps = {town: {'sha256': spec['sha256'], 'version': spec['carla_version']} for town, spec in config['map_paths'].items()}
    return {'schema': SCHEMA, 'contract': target_contract(native, model), 'rules': rules.identity(),
            'geometry': native.geometry.provenance(), 'implementation': hashes, 'maps': maps,
            'libraries': {'torch': str(torch.__version__), 'numpy': numpy.__version__, 'opencv': cv2.__version__, 'shapely': shapely.__version__},
            'map_sampling_m': .5, 'map_height_tolerance_m': 3.}


def validate_targets(values):
    import torch
    contract = json.loads(CONTRACT_PATH.read_text())
    if set(values) != set(contract['tensors']):
        raise AdapterError('invalid_targets_cache', 'Wrong target keys')
    for k, expected in contract['tensors'].items():
        v = values[k]
        if not isinstance(v, torch.Tensor) or tuple(v.shape) != tuple(expected['shape']) or v.dtype != getattr(torch, expected['dtype']):
            raise AdapterError('invalid_targets_cache', k + ': shape/dtype')
        if v.device.type != 'cpu' or not torch.isfinite(v).all():
            raise AdapterError('invalid_targets_cache', k + ': finite CPU values required')
    b = values['bev_semantic_map']
    if not ((b >= 0).all() and (b <= 6).all()):
        raise AdapterError('invalid_targets_cache', 'BEV outside 0..6')


def reference_pose(raw, native):
    from tools.b2d_adapter.coordinate_transform import actor_to_reference
    rear = actor_to_reference(raw.actor_pose, (native.geometry.axle_dx_m, native.geometry.axle_dy_m))
    return {'actor_location_xyz': [float(v) for v in raw.ego_box['location']],
            'actor_pitch_roll_yaw_deg': [float(v) for v in raw.ego_box['rotation']],
            'rear_world_carla_xy_yaw_rad': [rear.x, rear.y, rear.yaw],
            'rear_world_reflected_xy_yaw_rad': [rear.x, -rear.y, -rear.yaw]}


def current_annotation_source(sources, route, frame):
    # Numeric filenames, not directory sorting or an assumed padding width.
    matches = [s for s in sources if Path(s['path']).parent == Path(route) / 'anno'
               and Path(s['path']).name.endswith('.json.gz') and Path(s['path']).name.split('.')[0].isdigit()
               and int(Path(s['path']).name.split('.')[0]) == frame]
    if len(matches) != 1:
        raise AdapterError('pair_source_mismatch', 'Exactly one current annotation SHA required')
    return matches[0]


def feature_records(config, all_completed=False):
    """Read completed Features metadata only; never load Camera/LiDAR tensors."""
    from tools.training.diffusiondrive_v1.feature_jobs import config_and_state, readonly_db
    from tools.training.diffusiondrive_v1.features_cache import feature_producer
    collection = json.loads((ROOT / config['features_collection']).read_text())
    expected = digest(feature_producer(config))
    if expected != collection['producer_sha256']:
        raise AdapterError('pair_feature_producer_mismatch', 'Collection differs from frozen features')
    requested = None if all_completed else {
        sample_id(config['dataset_id'], str(Path(e['route']).resolve().relative_to(Path(config['dataset_root']).resolve())), frame)
        for entries in config['data'].values() for e in entries for frame in e['frames']}
    records = {}
    for entry in collection['jobs']:
        cfg, state = config_and_state(ROOT / entry['config'])
        if cfg['dataset_id'] != config['dataset_id']:
            raise AdapterError('pair_dataset_mismatch', entry['job_id'])
        db = readonly_db(state / 'index.sqlite')
        pinned = json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])
        if digest(pinned) != expected:
            raise AdapterError('pair_feature_producer_mismatch', entry['job_id'])
        for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY route,frame"):
            row = dict(r)
            if requested is not None and row['sample_id'] not in requested:
                continue
            if row['sample_id'] != sample_id(cfg['dataset_id'], row['route'], row['frame']) or row['sample_id'] in records:
                raise AdapterError('pair_sample_id_mismatch', row['sample_id'])
            current_annotation_source(json.loads(row['source_json']), row['route'], row['frame'])
            records[row['sample_id']] = {'job_id': entry['job_id'], 'config': entry['config'],
                                        'producer_sha256': expected, 'dataset_id': cfg['dataset_id'], 'row': row}
        db.close()
    return records


def source_identity(reader, ref, native, provider, root):
    from tools.b2d_adapter.temporal_sampler import TemporalSampler
    window = TemporalSampler(native.geometry).sample(reader.index, ref, full_scene=False, target_only=True)
    sources = [{'path': str(reader.index.path(r, 'anno').relative_to(root)),
                'bytes': reader.index.path(r, 'anno').stat().st_size,
                'sha256': file_digest(reader.index.path(r, 'anno'))} for r in window.continuous]
    town = re.search(r'_(Town\d+(?:HD)?)_', ref.route.name).group(1)
    mapped = provider.map_identity(town)
    return sources, {'town': town, 'sha256': mapped['sha256'], 'version': mapped['carla_version']}, [r.frame_id for r in window.trajectory_future]


def open_index(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS samples(sample_id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL,route TEXT NOT NULL,frame INTEGER NOT NULL,
      state TEXT NOT NULL,current_anno_sha256 TEXT,reference_json TEXT,sources_json TEXT,map_json TEXT,
      relative_path TEXT,sha256 TEXT,bytes INTEGER,error_code TEXT,error TEXT,updated REAL NOT NULL);
      CREATE INDEX IF NOT EXISTS targets_state ON samples(state);''')
    return db


def storage_check(spec, writing=False):
    root = Path(spec['path']).resolve()
    parent = root
    while not parent.exists():
        parent = parent.parent
    try:
        result = subprocess.run(['findmnt', '-J', '-T', str(parent), '-o', 'UUID,FSTYPE,OPTIONS,MAJ:MIN'],
                                capture_output=True, text=True, check=True, timeout=10)
        dev = parent.stat().st_dev
        actual = [r for r in json.loads(result.stdout)['filesystems'] if r['maj:min'] == f'{os.major(dev)}:{os.minor(dev)}'][-1]
        options = set(actual['options'].split(','))
        if actual['uuid'] != spec['uuid'] or actual['fstype'] != spec['fstype'] or options & {'shutdown', 'emergency_ro'}:
            raise StoragePause('Target storage identity/health differs')
        if writing and ('rw' not in options or not os.access(parent, os.W_OK)):
            raise StoragePause('Target storage is not writable')
        return shutil.disk_usage(parent).free
    except (OSError, IndexError, ValueError, subprocess.SubprocessError) as exc:
        raise StoragePause(str(exc)) from exc


def load_target_row(row, spec, producer_sha):
    import torch
    storage_check(spec)
    path = safe_path(spec, row['relative_path'])
    if path.stat().st_size != row['bytes'] or file_digest(path) != row['sha256']:
        raise AdapterError('invalid_targets_cache', 'File checksum/size differs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('schema') != SCHEMA or payload.get('sample_id') != row['sample_id']
        or payload.get('dataset_id') != row['dataset_id'] or payload.get('route') != row['route'] or payload.get('frame') != row['frame']
        or payload.get('target_producer_sha256') != producer_sha
        or digest(payload.get('sources')) != digest(json.loads(row['sources_json']))
        or digest(payload.get('reference_pose')) != digest(json.loads(row['reference_json']))
        or payload.get('current_annotation_sha256') != row['current_anno_sha256']
        or payload.get('coordinate_contract') != json.loads(CONTRACT_PATH.read_text())['coordinates']
        or digest(payload.get('map')) != digest(json.loads(row['map_json']))):
        raise AdapterError('target_contract_mismatch', row['sample_id'])
    validate_targets(payload['targets'])
    return payload


def generate(config, state, all_completed=False, limit=None, retry_failed=False):
    import fcntl
    import signal
    import torch
    from tools.b2d_adapter import FrameReader, RouteIndex
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1TargetBuilder
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration, bev_provider
    from tools.training.diffusiondrive_v1.features_cache import pin
    state = Path(state); state.mkdir(parents=True, exist_ok=True)
    lock = (state / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.monotonic(); counts = {'success': 0, 'failed': 0, 'skipped': 0, 'processed': 0}
    status = {'pid': os.getpid(), 'state': 'initializing', 'counts': counts}; db = None
    try:
        torch.set_num_threads(config.get('threads', 2))
        model, native = model_configuration(config), native_configuration(config)
        producer = producer_identity(config, model, native); sha = digest(producer)
        spec = config['target_root']; storage_check(spec, True)
        root = Path(spec['path']); root.mkdir(parents=True, exist_ok=True)
        db = open_index(state / 'index.sqlite'); pin(db, 'producer', producer); pin(db, 'dataset_id', config['dataset_id'])
        marker = root / 'target_root.json'; identity = {'schema': SCHEMA, 'state_id': str(state.resolve()), 'root_id': spec['root_id']}
        if marker.exists() and json.loads(marker.read_text()) != identity:
            raise StoragePause('Target root belongs to another index')
        if not marker.exists():
            if any(root.iterdir()): raise StoragePause('Unmarked nonempty target root')
            durable_json(marker, identity)
        namespace = root / sha
        # Only this writer owns this namespace; interrupted temps never count as complete.
        if namespace.exists():
            for p in namespace.rglob('*.partial'): p.unlink()
        used = sum(max(p.stat().st_size, p.stat().st_blocks * 512) for p in root.rglob('*') if p.is_file())
        provider_config = dict(config, bev_cache_dir=None)
        provider = bev_provider(provider_config)
        records = feature_records(config, all_completed)
        status.update(state='running', selected_features=len(records), producer_sha256=sha)
        reader = builder = None; current_route = None
        with (state / 'attempts.jsonl').open('a', buffering=1) as journal:
            for sid, feature in records.items():
                if stopped or (state / 'stop.request').exists(): status['state'] = 'stopped'; break
                if limit is not None and counts['processed'] >= limit: status['state'] = 'batch_complete'; break
                row = feature['row']; status['last_sample_id'] = sid
                prior = db.execute('SELECT * FROM samples WHERE sample_id=?', (sid,)).fetchone()
                if prior and prior['state'] == 'unavailable' and not retry_failed: continue
                try:
                    if row['route'] != current_route:
                        reader = FrameReader(RouteIndex.from_dataset_root(config['dataset_root'], row['route']))
                        builder = DiffusionDriveV1TargetBuilder(reader, native, model, provider); current_route = row['route']
                    ref = reader.index.ref(row['frame'])
                    sources, map_identity, future_ids = source_identity(reader, ref, native, provider, Path(config['dataset_root']).resolve())
                    current = current_annotation_source(sources, row['route'], row['frame'])
                    feature_current = current_annotation_source(json.loads(row['source_json']), row['route'], row['frame'])
                    if current['sha256'] != feature_current['sha256']:
                        raise StoragePause('Current annotation changed relative to Features: ' + sid)
                    raw = reader.read(ref); reference = reference_pose(raw, native)
                    if prior and prior['sources_json']:
                        if digest(json.loads(prior['sources_json'])) != digest(sources) or digest(json.loads(prior['map_json'])) != digest(map_identity):
                            raise StoragePause('Pinned target sources changed: ' + sid)
                    relative = str(Path(sha) / sid[:2] / (sid + '.pt')); dest = safe_path(spec, relative)
                    if prior and prior['state'] == 'complete':
                        try: load_target_row(dict(prior), spec, sha); counts['skipped'] += 1; continue
                        except (AdapterError, FileNotFoundError): pass
                    reserve = 1024 ** 2
                    if used + reserve > spec['budget_gib'] * GIB or storage_check(spec, True) - reserve < spec['minimum_free_gib'] * GIB:
                        raise StoragePause('Target cache budget/free-space threshold')
                    result = builder.build_targets(ref); validate_targets(result.tensors)
                    after = source_identity(reader, ref, native, provider, Path(config['dataset_root']).resolve())
                    if digest(after) != digest((sources, map_identity, future_ids)): raise StoragePause('Target sources changed during build')
                    payload = {'schema': SCHEMA, 'sample_id': sid, 'dataset_id': config['dataset_id'], 'route': row['route'], 'frame': row['frame'],
                               'target_producer_sha256': sha, 'reference_frame': row['frame'], 'reference_pose': reference,
                               'coordinate_contract': producer['contract']['coordinates'], 'sources': sources, 'map': map_identity,
                               'current_annotation_sha256': current['sha256'], 'future_frame_ids': future_ids,
                               'targets': result.tensors, 'provenance': result.provenance}
                    stream = io.BytesIO(); torch.save(payload, stream); data = stream.getvalue()
                    if used + len(data) + 65536 > spec['budget_gib'] * GIB or storage_check(spec, True) - len(data) - 65536 < spec['minimum_free_gib'] * GIB:
                        raise StoragePause('Target cache exact-size threshold')
                    dest.parent.mkdir(parents=True, exist_ok=True); temp = dest.with_name(dest.name + f'.{os.getpid()}.partial')
                    old_size = max(dest.stat().st_size, dest.stat().st_blocks * 512) if dest.exists() else 0
                    try:
                        with temp.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
                        storage_check(spec, True); os.replace(temp, dest); directory_sync(dest.parent)
                    finally:
                        storage_check(spec, True); temp.unlink(missing_ok=True)
                    used += max(dest.stat().st_size, dest.stat().st_blocks * 512) - old_size
                    cached = {'sample_id': sid, 'dataset_id': config['dataset_id'], 'route': row['route'], 'frame': row['frame'], 'state': 'complete',
                              'current_anno_sha256': current['sha256'], 'reference_json': json.dumps(reference), 'sources_json': json.dumps(sources),
                              'map_json': json.dumps(map_identity), 'relative_path': relative, 'sha256': hashlib.sha256(data).hexdigest(),
                              'bytes': len(data), 'error_code': None, 'error': None, 'updated': time.time()}
                    restored = load_target_row(cached, spec, sha)
                    assert all(torch.equal(result.tensors[k], restored['targets'][k]) for k in result.tensors)
                    cols = list(cached)
                    db.execute('INSERT OR REPLACE INTO samples(' + ','.join(cols) + ') VALUES(' + ','.join('?' for _ in cols) + ')', list(cached.values()))
                    db.commit(); counts['success'] += 1
                    event = {'sample_id': sid, 'state': 'complete', 'bev_pixels': result.provenance['bev']['class_pixel_counts']}
                except (StoragePause, OSError) as exc: raise StoragePause(str(exc)) from exc
                except AdapterError as exc:
                    if any(s in str(exc) for s in ['Input/output error', 'Read-only file system', 'Permission denied', 'identity_mismatch']): raise StoragePause(str(exc)) from exc
                    db.execute('INSERT OR REPLACE INTO samples(sample_id,dataset_id,route,frame,state,error_code,error,updated) VALUES(?,?,?,?,?,?,?,?)',
                               (sid, config['dataset_id'], row['route'], row['frame'], 'unavailable', exc.code, str(exc), time.time()))
                    db.commit(); counts['failed'] += 1; event = {'sample_id': sid, 'state': 'unavailable', 'error_code': exc.code, 'error': str(exc)}
                    status['last_error'] = event
                counts['processed'] += 1; journal.write(json.dumps(event) + '\n'); print(json.dumps(event), flush=True)
                status.update(updated=time.time(), seconds=time.monotonic() - started, cache_bytes=used)
                durable_json(state / 'status.json', status)
            else: status['state'] = 'complete'
        status['index_counts'] = dict(db.execute('SELECT state,COUNT(*) FROM samples GROUP BY state').fetchall())
        durable_json(state / 'target_manifest.json', {'schema': SCHEMA, 'producer': producer, 'dataset_id': config['dataset_id'],
                     'root_id': spec['root_id'], 'accepted': [dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY route,frame")]})
    except Exception as exc:
        status.update(state='paused', last_error={'code': type(exc).__name__, 'error': str(exc)})
        print(json.dumps(status['last_error']), flush=True)
    finally:
        status.update(updated=time.time(), seconds=time.monotonic() - started)
        durable_json(state / 'status.json', status)
        if db is not None: db.close()
        lock.close()
    return status
