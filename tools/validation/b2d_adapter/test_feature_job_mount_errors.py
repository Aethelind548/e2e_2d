"""Scheduler diagnostics must preserve the existing producer mount refusal."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.training.diffusiondrive_v1.feature_jobs import require_writable_root
from tools.training.diffusiondrive_v1.features_cache import StoragePause


class MountErrorTests(unittest.TestCase):
    def test_emergency_mount_remains_blocked_with_explanation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp).stat().st_dev
            actual = {'target': tmp, 'source': '/dev/test', 'uuid': 'expected', 'fstype': 'ext4',
                      'options': 'rw,emergency_ro', 'maj:min': f'{os.major(dev)}:{os.minor(dev)}'}
            # A hidden older overmount must not be used to explain the active one.
            hidden = dict(actual, source='/dev/old', options='rw,shutdown', **{'maj:min': '0:0'})
            reply = type('Reply', (), {'stdout': json.dumps({'filesystems': [hidden, actual]})})()
            with patch('tools.training.diffusiondrive_v1.feature_jobs.active_mount', side_effect=StoragePause('Unsafe or read-only mount')) as gate, patch(
                    'tools.training.diffusiondrive_v1.feature_jobs.subprocess.run', return_value=reply):
                with self.assertRaises(StoragePause) as caught:
                    require_writable_root({'mountpoint': tmp})
                gate.assert_called_once_with({'mountpoint': tmp}, writing=True)
                self.assertIn('故障保护', str(caught.exception))
                self.assertIn('/dev/test', str(caught.exception))
                self.assertNotIn('/dev/old', str(caught.exception))

    def test_healthy_mount_uses_unchanged_gate(self):
        with patch('tools.training.diffusiondrive_v1.feature_jobs.active_mount', return_value={'uuid': 'expected'}) as gate:
            self.assertEqual(require_writable_root({'path': '/example'}), {'uuid': 'expected'})
            gate.assert_called_once_with({'path': '/example'}, writing=True)


if __name__ == '__main__':
    unittest.main()
