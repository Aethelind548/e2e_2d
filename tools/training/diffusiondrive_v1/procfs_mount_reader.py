"""Optional Linux mount inspection for cache reads, without per-item subprocesses.

Checks live mountinfo, visible st_dev and the expected /dev/disk/by-uuid block
device on every call. No TTL, skipped checksum or changes to numeric producers.
Scopes are restricted to the deterministic Dataset reader in each worker.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import stat
import threading

from tools.training.diffusiondrive_v1.features_cache import StoragePause
from tools.training.diffusiondrive_v1.recovery_paired import RecoveryPairedDataset

_LOCK = threading.RLock()


def _unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), value)


def _inspect(spec, parent, exact_mount=None, writing=False):
    try:
        device = parent.stat().st_dev
        expected = os.stat('/dev/disk/by-uuid/' + spec['uuid'])
        if not stat.S_ISBLK(expected.st_mode) or expected.st_rdev != device:
            raise StoragePause('UUID/device mismatch: ' + str(parent))
        major_minor = f'{os.major(device)}:{os.minor(device)}'
        candidates = []
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            before, after = line.split(' - ', 1)
            fields = before.split(); filesystem = after.split()
            target = Path(_unescape(fields[4]))
            if fields[2] != major_minor:
                continue
            if exact_mount is not None:
                if target != exact_mount: continue
            elif not parent.is_relative_to(target):
                continue
            options = set(fields[5].split(',')) | set(filesystem[2].split(','))
            candidates.append((len(target.parts), target, filesystem[0], options))
        if not candidates:
            raise StoragePause('Expected visible mount is absent: ' + str(parent))
        _, target, fstype, options = max(enumerate(candidates), key=lambda p: (p[1][0], p[0]))[1]
        if fstype != spec['fstype'] or options & {'shutdown', 'emergency_ro'}:
            raise StoragePause('Filesystem/mount health mismatch: ' + str(parent))
        if writing and ('rw' not in options or 'ro' in options or not os.access(parent, os.W_OK)):
            raise StoragePause('Mount is not writable: ' + str(parent))
        # Check visible device again to detect replacement during inspection.
        if parent.stat().st_dev != device:
            raise StoragePause('Mount changed during inspection')
        return {'target': str(target), 'uuid': spec['uuid'], 'fstype': fstype,
                'options': ','.join(sorted(options)), 'maj:min': major_minor,
                'free_bytes': shutil.disk_usage(parent).free}
    except (OSError, ValueError, IndexError) as exc:
        raise StoragePause('Kernel mount inspection failed: ' + str(exc)) from exc


def active_mount(spec, writing=False):
    mount = Path(spec['mountpoint']); root = Path(spec['path'])
    if not mount.is_dir() or root == mount or not root.is_relative_to(mount):
        raise StoragePause('Cache root unavailable/outside configured mount: ' + str(root))
    result = _inspect(spec, mount, exact_mount=mount, writing=False)
    if root.exists() and root.stat().st_dev != mount.stat().st_dev:
        raise StoragePause('Cache root on a different device: ' + str(root))
    if writing:
        _inspect(spec, root if root.exists() else mount, writing=True)
    return result


def storage_check(spec, writing=False):
    parent = Path(spec['path']).resolve()
    while not parent.exists(): parent = parent.parent
    return _inspect(spec, parent, writing=writing)['free_bytes']


@contextmanager
def mount_scope():
    # Existing frozen cache readers use module-global inspectors. As with the
    # existing target_storage.codec_scope, restore every binding even on errors.
    from tools.training.diffusiondrive_v1 import features_cache, target_cache, system_feature_relocation
    with _LOCK:
        original = (features_cache.active_mount, target_cache.storage_check, system_feature_relocation.active_mount)
        features_cache.active_mount = active_mount
        target_cache.storage_check = storage_check
        system_feature_relocation.active_mount = active_mount
        try:
            yield
        finally:
            features_cache.active_mount, target_cache.storage_check, system_feature_relocation.active_mount = original


class ProcfsMountPairedDataset(RecoveryPairedDataset):
    def __getitem__(self, index):
        if self.manifest['target_sources'][self.rows[index]['target_source']].get('kind') == 'paired-gravity6':
            raise StoragePause('procfs reader has not been accepted for gravity6')
        with mount_scope():
            return super().__getitem__(index)
