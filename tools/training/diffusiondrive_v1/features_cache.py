"""Versioned features-only cache: UUID guarded roots and a transactional index.

No BEV/trajectory/detection builders are called during generation. Supervision
eligibility is deliberately separate from sensor/input eligibility.
"""
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1.prepared import digest, file_digest

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'b2d-v1-features-only-v1'
GIB = 1024 ** 3
FEATURE_SHAPES = {'camera_feature': (3, 256, 1024), 'lidar_feature': (1, 256, 256), 'status_feature': (8,)}


class StoragePause(RuntimeError):
    """Mount, identity, budget or persistent I/O problem; stop further writes."""


def validate_features(features):
    import torch
    if set(features) != set(FEATURE_SHAPES):
        raise AdapterError('invalid_feature_cache', 'Feature keys differ')
    for name, shape in FEATURE_SHAPES.items():
        t = features[name]
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or t.dtype != torch.float32:
            raise AdapterError('invalid_feature_cache', name + ': shape/dtype')
        if t.device.type != 'cpu' or not torch.isfinite(t).all():
            raise AdapterError('invalid_feature_cache', name + ': finite CPU tensor required')
        if name != 'status_feature' and not ((t >= 0).all() and (t <= 1).all()):
            raise AdapterError('invalid_feature_cache', name + ': outside [0,1]')


def feature_producer(config):
    """Input provenance only; no map/BEV rules or learned weights in identity."""
    import cv2
    import numpy
    import PIL
    import torch
    import torchvision
    from tools.training.diffusiondrive_v1.runtime import native_configuration
    native = native_configuration(config)
    if native.camera_policy != 'b2d-proportional-crop-v1':
        raise AdapterError('feature_camera_policy_mismatch', native.camera_policy)
    files = ['camera_adapter.py', 'native_builders.py', 'native_config.py', 'config.py',
             'v1_builders.py', 'frame_reader.py', 'route_index.py', 'object_adapter.py',
             'coordinate_transform.py', 'navsim_bridge.py', 'native_navigation.py', 'lidar_adapter.py', 'lidar_decode.py']
    implementation = {name: file_digest(ROOT / 'tools/b2d_adapter' / name) for name in files}
    for name in ['transfuser_features.py', 'transfuser_config.py']:
        implementation['official/' + name] = file_digest(ROOT / 'repositories/DiffusionDrive/navsim/agents/diffusiondrive' / name)
    implementation['features_cache.py'] = file_digest(Path(__file__))
    decoder = subprocess.run([config['lidar_decoder_python'], '-c',
        'import json,sys,numpy,importlib.metadata as m; print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,laspy=m.version("laspy"),lazrs=m.version("lazrs"))))'],
        capture_output=True, text=True, check=True, timeout=20)
    return {'schema': SCHEMA, 'native_policy': native.provenance(), 'implementation': implementation,
            'features': {name: {'shape': list(shape), 'dtype': 'float32'} for name, shape in FEATURE_SHAPES.items()},
            'libraries': {'torch': str(torch.__version__), 'numpy': numpy.__version__, 'PIL': PIL.__version__,
                          'opencv': cv2.__version__, 'torchvision': str(torchvision.__version__)},
            'decoder': json.loads(decoder.stdout), 'map_dependency': None,
            'camera_rule': {'top_bottom': 23, 'side_left_right': 346, 'stitch_wh': [3416, 854],
                            'resize_wh': [1024, 256], 'resize': 'cv2.INTER_LINEAR', 'ToTensor': 'RGB /255 once'}}


def source_identity(index, ref, dataset_root):
    """Only three causal annotations and current JPG/LAZ, never future labels."""
    from tools.b2d_adapter.route_index import CAMERAS
    paths = [index.path(index.ref(i), 'anno') for i in range(ref.frame_id - 2, ref.frame_id + 1)]
    paths += [index.path(ref, m) for m in (*CAMERAS, 'lidar')]
    return [{'path': str(p.relative_to(dataset_root)), 'bytes': p.stat().st_size, 'sha256': file_digest(p)} for p in paths]


def sample_id(dataset_id, route_name, frame):
    return digest({'dataset': dataset_id, 'route': route_name, 'frame': frame})


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS samples(
          sample_id TEXT PRIMARY KEY, route TEXT NOT NULL, frame INTEGER NOT NULL,
          state TEXT NOT NULL DEFAULT 'pending', root_id TEXT, relative_path TEXT,
          sha256 TEXT, bytes INTEGER, source_json TEXT, error_code TEXT, error TEXT,
          attempts INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS discoveries(route TEXT PRIMARY KEY, summary TEXT NOT NULL);
    ''')
    return db


def pin(db, key, value):
    encoded = json.dumps(value, sort_keys=True)
    existing = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    if existing and existing[0] != encoded:
        raise StoragePause('Index identity changed: ' + key + '; use a new job namespace')
    db.execute('INSERT OR IGNORE INTO meta VALUES(?,?)', (key, encoded))
    db.commit()


def active_mount(spec, writing=False):
    """Filter overmounted historical entries using the actual mount st_dev."""
    mount = Path(spec['mountpoint'])
    root = Path(spec['path'])
    if not mount.is_dir() or not root.is_relative_to(mount) or root == mount:
        raise StoragePause('Cache root unavailable/outside configured mount: ' + str(root))
    try:
        result = subprocess.run(['findmnt', '-J', '-T', str(mount), '-o', 'TARGET,UUID,FSTYPE,OPTIONS,MAJ:MIN'],
                                capture_output=True, text=True, check=True, timeout=10)
        rows = json.loads(result.stdout)['filesystems']
        device = mount.stat().st_dev
        major_minor = f'{os.major(device)}:{os.minor(device)}'
        matches = [r for r in rows if r['target'] == str(mount) and r['maj:min'] == major_minor]
        if not matches:
            raise StoragePause('Configured mount is absent: ' + str(mount))
        actual = matches[-1]
        options = set(actual['options'].split(','))
        if actual['uuid'] != spec['uuid'] or actual['fstype'] != spec['fstype']:
            raise StoragePause('UUID/filesystem mismatch: ' + str(mount))
        if options & {'shutdown', 'emergency_ro'} or (writing and 'rw' not in options):
            raise StoragePause('Unsafe or read-only mount: ' + str(mount))
        if root.exists() and root.stat().st_dev != device:
            raise StoragePause('Cache root on a different device: ' + str(root))
        if writing and not os.access(root if root.exists() else mount, os.W_OK):
            raise StoragePause('Cache root is not writable: ' + str(root))
        return dict(actual, free_bytes=__import__('shutil').disk_usage(mount).free)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise StoragePause('Mount inspection failed: ' + str(exc)) from exc


def safe_path(spec, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise StoragePause('Invalid relative cache path')
    root = Path(spec['path'])
    path = root / rel
    if root.resolve() != root or not path.resolve().is_relative_to(root):
        raise StoragePause('Symlink or path escapes cache root')
    return path


def directory_sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    try:
        with temp.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        directory_sync(path.parent)
    finally:
        temp.unlink(missing_ok=True)


class CacheStorage:
    def __init__(self, config, producer_sha, index_id):
        self.config = config
        self.namespace = SCHEMA + '/' + producer_sha
        self.roots = config['cache_roots']
        self.used = {}
        self.locks = []
        for root_id, spec in self.roots.items():
            active_mount(spec, writing=True)
            path = Path(spec['path'])
            path.mkdir(exist_ok=True)
            marker = path / 'cache_root.json'
            expected = {'schema': SCHEMA, 'root_id': root_id, 'dataset_id': config['dataset_id'], 'index_id': index_id}
            if marker.exists():
                if json.loads(marker.read_text()) != expected:
                    raise StoragePause('Cache root owner differs: ' + str(marker))
            else:
                # Do not adopt somebody else's directory containing files.
                if any(path.iterdir()):
                    raise StoragePause('Unmarked nonempty cache root: ' + str(path))
                durable_json(marker, expected)
            import fcntl
            lock = (path / '.writer.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                lock.close()
                raise StoragePause('Another writer owns cache root: ' + root_id) from exc
            self.locks.append(lock)
            self.used[root_id] = self.measure(path)

    def close(self):
        for lock in self.locks:
            lock.close()

    @staticmethod
    def measure(path):
        return sum(max(p.stat().st_size, p.stat().st_blocks * 512) for p in path.rglob('*') if p.is_file())

    def available(self, root_id, size):
        spec = self.roots[root_id]
        info = active_mount(spec, writing=True)
        reserve = size + 65536
        return (self.used[root_id] + reserve <= int(spec['budget_gib'] * GIB)
                and info['free_bytes'] - reserve >= int(spec['minimum_free_gib'] * GIB))

    def allocate(self, size, preferred=None):
        order = list(self.roots)
        if preferred:
            order.remove(preferred)
            order.insert(0, preferred)
        for root_id in order:
            if self.available(root_id, size):
                return root_id
        raise StoragePause('Both cache roots reached budget or free-space floor')

    def clean_partials(self):
        count = 0
        for root_id, spec in self.roots.items():
            active_mount(spec, writing=True)
            folder = safe_path(spec, self.namespace)
            if folder.exists():
                for p in folder.rglob('*.partial'):
                    # Single job holds the writer lock; only own namespace temps.
                    self.used[root_id] -= max(p.stat().st_size, p.stat().st_blocks * 512)
                    p.unlink()
                    count += 1
        return count

    def write(self, root_id, relative, data):
        spec = self.roots[root_id]
        if not self.available(root_id, len(data)):
            raise StoragePause('Assigned disk budget/free floor reached: ' + root_id)
        dest = safe_path(spec, relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        active_mount(spec, writing=True)
        temp = dest.with_name(dest.name + f'.{os.getpid()}.partial')
        old_size = max(dest.stat().st_size, dest.stat().st_blocks * 512) if dest.exists() else 0
        try:
            # Verify directory device after opening; fail if a mount changed.
            fd = os.open(dest.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                device = Path(spec['mountpoint']).stat().st_dev
                if os.fstat(fd).st_dev != device:
                    raise StoragePause('Mount changed before write')
                with os.fdopen(os.open(temp.name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                                       0o600, dir_fd=fd), 'wb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                active_mount(spec, writing=True)
                os.replace(temp.name, dest.name, src_dir_fd=fd, dst_dir_fd=fd)
                os.fsync(fd)
            finally:
                os.close(fd)
            self.used[root_id] += max(dest.stat().st_size, dest.stat().st_blocks * 512) - old_size
        except OSError as exc:
            raise StoragePause('Cache write I/O error: ' + str(exc)) from exc
        finally:
            # Never clean through an unmounted path.
            active_mount(spec, writing=True)
            temp.unlink(missing_ok=True)
        return dest


def load_feature_row(row, root_mapping, producer_sha):
    import torch
    spec = root_mapping[row['root_id']]
    active_mount(spec)
    path = safe_path(spec, row['relative_path'])
    if path.stat().st_size != row['bytes'] or file_digest(path) != row['sha256']:
        raise AdapterError('invalid_feature_cache', 'checksum/size: ' + str(path))
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('schema') != SCHEMA or payload.get('sample_id') != row['sample_id']
            or payload.get('producer_sha256') != producer_sha
            or digest(payload.get('sources')) != digest(json.loads(row['source_json']))):
        raise StoragePause('Feature artifact identity mismatch: ' + str(path))
    validate_features(payload['features'])
    return payload['features']


class FeatureCacheDataset:
    """Root-independent unified reader. Returns features only, never fake targets."""
    def __init__(self, index_path, root_mapping, expected_producer, sample_ids=None):
        self.root_mapping = root_mapping
        self.producer_sha = digest(expected_producer)
        db = sqlite3.connect('file:' + str(Path(index_path).resolve()) + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        recorded = db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()
        if not recorded or digest(json.loads(recorded[0])) != self.producer_sha:
            raise AdapterError('feature_contract_mismatch', 'Producer or Camera policy differs')
        rows = [dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY route,frame")]
        db.close()
        if sample_ids is not None:
            lookup = {r['sample_id']: r for r in rows}
            rows = [lookup[s] for s in sample_ids]
        self.rows = rows
        self.sample_ids = [r['sample_id'] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return load_feature_row(self.rows[i], self.root_mapping, self.producer_sha)


class FeatureTargetDataset:
    """Join an explicit supervised selection to a separately validated target reader.

    target_reader receives (dataset-relative route, frame, sample_id). It must
    provide real supervision or raise. Feature success never certifies targets.
    """
    def __init__(self, features, target_reader):
        self.features, self.target_reader = features, target_reader

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        row = self.features.rows[i]
        return self.features[i], self.target_reader(row['route'], row['frame'], row['sample_id'])


def discover(db, config):
    from tools.b2d_adapter import RouteIndex
    from tools.b2d_adapter.route_index import CAMERAS
    base = Path(config['dataset_root']).resolve()
    for route in config['routes']:
        if db.execute('SELECT 1 FROM discoveries WHERE route=?', (route,)).fetchone():
            continue
        index = RouteIndex.from_dataset_root(base, route)
        candidates, missing = 0, 0
        for frame in index.frame_ids:
            if frame < 2 or any(i not in index.files['anno'] for i in range(frame - 2, frame + 1)):
                continue
            if any(frame not in index.files[m] for m in (*CAMERAS, 'lidar')):
                missing += 1
                continue
            db.execute('INSERT OR IGNORE INTO samples(sample_id,route,frame) VALUES(?,?,?)',
                       (sample_id(config['dataset_id'], route, frame), route, frame))
            candidates += 1
        summary = {'route': route, 'input_history_sensor_candidates': candidates,
                   'missing_current_modality': missing, 'supervision_eligibility': 'not evaluated'}
        db.execute('INSERT INTO discoveries VALUES(?,?)', (route, json.dumps(summary)))
        db.commit()
        print(json.dumps(dict(event='discovery', **summary)), flush=True)


def index_summary(db):
    states = dict(db.execute('SELECT state,COUNT(*) FROM samples GROUP BY state').fetchall())
    roots = {r[0]: {'samples': r[1], 'bytes': r[2]} for r in db.execute(
        "SELECT root_id,COUNT(*),SUM(bytes) FROM samples WHERE state='complete' GROUP BY root_id")}
    errors = dict(db.execute("SELECT error_code,COUNT(*) FROM samples WHERE state='failed' GROUP BY error_code").fetchall())
    return {'total': sum(states.values()), 'states': states, 'roots': roots, 'failure_codes': errors}


def run(config, state, limit=None, preferred=None, retry_failed=False):
    import fcntl
    import torch
    from tools.b2d_adapter import FrameReader, RouteIndex
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1FeatureBuilder
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / 'writer.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise StoragePause('Another writer is already running') from exc
    started = time.monotonic()
    counts = {'processed': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'generated_bytes': 0}
    progress = {'pid': os.getpid(), 'state': 'initializing', 'counts': counts, 'last_sample_id': None}
    stop_requested = False
    import signal
    def stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    db = None
    storage = None
    def status():
        seconds = time.monotonic() - started
        progress.update(updated=time.time(), seconds=seconds, rate_samples_per_s=counts['success'] / max(seconds, .001),
                        write_bytes_per_s=counts['generated_bytes'] / max(seconds, .001))
        if db is not None:
            progress['index'] = index_summary(db)
            remaining = progress['index']['states'].get('pending', 0)
            progress['estimated_remaining_seconds'] = remaining / progress['rate_samples_per_s'] if counts['success'] else None
        if storage is not None:
            progress['disk_cache_bytes'] = storage.used
        durable_json(state / 'status.json', progress)
    try:
        torch.set_num_threads(config.get('threads', 2))
        producer = feature_producer(config)
        producer_sha = digest(producer)
        db = connect(state / 'index.sqlite')
        pin(db, 'producer', producer)
        pin(db, 'dataset_id', config['dataset_id'])
        pin(db, 'requested_routes', config['routes'])
        recorded_id = db.execute("SELECT value FROM meta WHERE key='index_id'").fetchone()
        index_id = json.loads(recorded_id[0]) if recorded_id else __import__('uuid').uuid4().hex
        pin(db, 'index_id', index_id)
        storage = CacheStorage(config, producer_sha, index_id)
        progress['cleaned_partial_files'] = storage.clean_partials()
        discover(db, config)
        model, native = model_configuration(config), native_configuration(config)
        progress['state'] = 'running'
        status()
        reader, builder, current_route = None, None, None
        with (state / 'attempts.jsonl').open('a', buffering=1) as journal:
            rows = db.execute('SELECT * FROM samples ORDER BY route,frame').fetchall()
            for original in rows:
                if stop_requested or (state / 'stop.request').exists():
                    progress['state'] = 'stopped'
                    break
                if limit is not None and counts['processed'] >= limit:
                    progress['state'] = 'batch_complete'
                    break
                row = dict(original)
                if row['state'] == 'failed' and not retry_failed:
                    continue
                progress['last_sample_id'] = row['sample_id']
                sample_started = time.monotonic()
                try:
                    if row['route'] != current_route:
                        reader = FrameReader(RouteIndex.from_dataset_root(config['dataset_root'], row['route']))
                        builder = DiffusionDriveV1FeatureBuilder(reader, native, model, config['lidar_decoder_python'])
                        current_route = row['route']
                    ref = reader.index.ref(row['frame'])
                    sources = source_identity(reader.index, ref, Path(config['dataset_root']).resolve())
                    encoded_sources = json.dumps(sources, sort_keys=True)
                    if row['source_json'] and row['source_json'] != encoded_sources:
                        raise StoragePause('Raw data identity changed: ' + row['sample_id'])
                    if row['root_id']:
                        spec = config['cache_roots'][row['root_id']]
                        active_mount(spec)
                        path = safe_path(spec, row['relative_path'])
                        if path.exists():
                            if row['state'] == 'complete':
                                try:
                                    load_feature_row(row, config['cache_roots'], producer_sha)
                                    counts['skipped'] += 1
                                    if counts['skipped'] % 25 == 0:
                                        status()
                                    continue
                                except AdapterError:
                                    pass  # Checksum/shape damaged: rebuild at the same pinned location.
                            else:
                                # Recover a rename committed before the SQLite transaction.
                                try:
                                    payload = torch.load(path, map_location='cpu', weights_only=True)
                                    if payload.get('schema') != SCHEMA or payload.get('sample_id') != row['sample_id'] or payload.get('producer_sha256') != producer_sha or digest(payload.get('sources')) != digest(sources):
                                        raise StoragePause('Orphan artifact identity differs')
                                    validate_features(payload['features'])
                                    db.execute("UPDATE samples SET state='complete',sha256=?,bytes=?,updated=? WHERE sample_id=?",
                                               (file_digest(path), path.stat().st_size, time.time(), row['sample_id']))
                                    db.commit()
                                    counts['skipped'] += 1
                                    continue
                                except (AdapterError, EOFError, RuntimeError, ValueError, KeyError, __import__('pickle').UnpicklingError):
                                    pass
                    # Capacity must be checked before decoding, then again for exact serialized bytes.
                    if row['root_id']:
                        if not storage.available(row['root_id'], 4 * 1024 ** 2):
                            raise StoragePause('Pinned cache disk lacks capacity: ' + row['root_id'])
                    else:
                        storage.allocate(4 * 1024 ** 2, preferred)
                    result = builder.build_features(builder.read_current(ref))
                    validate_features(result.tensors)
                    if sources != source_identity(reader.index, ref, Path(config['dataset_root']).resolve()):
                        raise StoragePause('Source changed during generation: ' + row['sample_id'])
                    payload = {'schema': SCHEMA, 'sample_id': row['sample_id'], 'dataset_id': config['dataset_id'],
                               'route': row['route'], 'frame': row['frame'], 'producer_sha256': producer_sha,
                               'sources': sources, 'features': result.tensors,
                               'provenance': result.provenance, 'map_dependency': None,
                               'supervision_eligibility': 'not evaluated'}
                    stream = io.BytesIO()
                    torch.save(payload, stream)
                    data = stream.getvalue()
                    root_id = row['root_id'] or storage.allocate(len(data), preferred)
                    relative = row['relative_path'] or f'{storage.namespace}/{row["sample_id"][:2]}/{row["sample_id"]}.pt'
                    db.execute("UPDATE samples SET state='pending',root_id=?,relative_path=?,source_json=?,attempts=attempts+1 WHERE sample_id=?",
                               (root_id, relative, encoded_sources, row['sample_id']))
                    db.commit()
                    path = storage.write(root_id, relative, data)
                    sha = hashlib.sha256(data).hexdigest()
                    verified = dict(row, root_id=root_id, relative_path=relative, source_json=encoded_sources, bytes=len(data), sha256=sha)
                    readback = load_feature_row(verified, config['cache_roots'], producer_sha)
                    if not all(torch.equal(readback[k], result.tensors[k]) for k in FEATURE_SHAPES):
                        raise StoragePause('Bitwise roundtrip differs')
                    db.execute("UPDATE samples SET state='complete',sha256=?,bytes=?,error=NULL,error_code=NULL,updated=? WHERE sample_id=?",
                               (sha, len(data), time.time(), row['sample_id']))
                    db.commit()
                    counts['success'] += 1
                    counts['generated_bytes'] += len(data)
                    event = dict(sample_id=row['sample_id'], route=row['route'], frame=row['frame'], status='complete', root_id=root_id, bytes=len(data))
                except (StoragePause, OSError) as exc:
                    raise StoragePause(str(exc)) from exc
                except AdapterError as exc:
                    # Decoder wraps OSError into AdapterError: recognize severe device/source faults.
                    text = str(exc)
                    if any(s in text for s in ['Input/output error', 'No space left', 'Read-only file system', 'Stale file handle', 'Permission denied', 'source_mismatch', 'identity_mismatch']):
                        raise StoragePause(text) from exc
                    if exc.code == 'route_unavailable':
                        raise StoragePause(text) from exc
                    counts['failed'] += 1
                    progress['last_error'] = {'sample_id': row['sample_id'], 'code': exc.code, 'error': text}
                    db.execute("UPDATE samples SET state='failed',error_code=?,error=?,attempts=attempts+1,updated=? WHERE sample_id=?",
                               (exc.code, text, time.time(), row['sample_id']))
                    db.commit()
                    event = dict(sample_id=row['sample_id'], status='failed', error_code=exc.code, error=text)
                counts['processed'] += 1
                event['seconds'] = time.monotonic() - sample_started
                journal.write(json.dumps(event, ensure_ascii=False) + '\n')
                journal.flush()
                print(json.dumps(event, ensure_ascii=False), flush=True)
                status()
            else:
                progress['state'] = 'complete'
    except Exception as exc:
        progress['state'] = 'paused'
        progress['last_error'] = {'code': type(exc).__name__, 'error': str(exc)}
        print(json.dumps({'event': 'paused', 'error': str(exc)}, ensure_ascii=False), flush=True)
    finally:
        status()
        if db is not None:
            db.close()
        if storage is not None:
            storage.close()
        lock.close()
    return progress
