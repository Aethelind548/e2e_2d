"""Safety boundaries: no missing-mount writes, budget fallback, atomic failure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.training.diffusiondrive_v1.features_cache import CacheStorage, StoragePause, active_mount, safe_path


class StorageTests(unittest.TestCase):
    def test_mount_identity_and_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            mount = Path(tmp)
            dev = mount.stat().st_dev
            import os
            entry = {'target': tmp, 'uuid': 'good', 'fstype': 'ntfs3', 'options': 'rw',
                     'maj:min': f'{os.major(dev)}:{os.minor(dev)}'}
            spec = {'path': tmp + '/cache', 'mountpoint': tmp, 'uuid': 'good', 'fstype': 'ntfs3'}
            def response(e):
                return type('Result', (), {'stdout': json.dumps({'filesystems': [e]})})()
            with patch('subprocess.run', return_value=response(entry)):
                active_mount(spec, True)
                with self.assertRaises(StoragePause):
                    active_mount(dict(spec, uuid='wrong'), True)
            for options in ['ro', 'rw,shutdown', 'rw,emergency_ro']:
                with patch('subprocess.run', return_value=response(dict(entry, options=options))):
                    with self.assertRaises(StoragePause):
                        active_mount(spec, True)
            with patch('subprocess.run', return_value=response(dict(entry, target='/'))):
                with self.assertRaises(StoragePause):
                    active_mount(spec, True)
            self.assertFalse(Path(spec['path']).exists())

    def test_budget_and_free_floor_fallback(self):
        storage = CacheStorage.__new__(CacheStorage)
        storage.roots = {k: {'budget_gib': 1, 'minimum_free_gib': 2} for k in ('a', 'b')}
        storage.used = {'a': 1024 ** 3, 'b': 0}
        with patch('tools.training.diffusiondrive_v1.features_cache.active_mount', return_value={'free_bytes': 4 * 1024 ** 3}):
            self.assertEqual(storage.allocate(1000), 'b')
        with patch('tools.training.diffusiondrive_v1.features_cache.active_mount', return_value={'free_bytes': 2 * 1024 ** 3}):
            with self.assertRaises(StoragePause):
                storage.allocate(1)

    def test_atomic_write_error_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'sample.pt').write_bytes(b'previous')
            storage = CacheStorage.__new__(CacheStorage)
            storage.roots = {'a': {'path': tmp, 'mountpoint': tmp}}
            storage.used = {'a': 4096}
            with patch.object(storage, 'available', return_value=True), patch(
                    'tools.training.diffusiondrive_v1.features_cache.active_mount', return_value={}), patch(
                    'os.replace', side_effect=OSError('Injected I/O failure')):
                with self.assertRaises(StoragePause):
                    storage.write('a', 'sample.pt', b'new data')
            self.assertEqual((folder / 'sample.pt').read_bytes(), b'previous')
            self.assertEqual(list(folder.glob('*.partial')), [])

    def test_relative_paths_cannot_escape(self):
        for path in ['../other.pt', '/other.pt']:
            with self.assertRaises(StoragePause):
                safe_path({'path': '/tmp/cache'}, path)


if __name__ == '__main__':
    unittest.main()
