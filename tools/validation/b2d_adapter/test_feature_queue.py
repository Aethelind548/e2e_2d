"""Resume must exclude completed blobs and keep holes/retries; audit is explicit."""
import errno
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1 import feature_queue as queue


class PendingQueueTests(unittest.TestCase):
    def make_db(self, retry=False):
        db = sqlite3.connect(':memory:', factory=lambda *a, **k: queue.PendingConnection(*a, retry_failed=retry, **k))
        db.execute('CREATE TABLE samples(sample_id TEXT,route TEXT,frame INTEGER,state TEXT)')
        db.executemany('INSERT INTO samples VALUES(?,?,?,?)', [
            ('hole', 'a', 2, 'pending'), ('done', 'a', 10, 'complete'),
            ('rejected', 'a', 3, 'failed'), ('next', 'b', 1, 'pending')])
        self.addCleanup(db.close)
        return db

    def test_pending_hole_before_completed_frame_is_preserved(self):
        db = self.make_db()
        self.assertEqual([r[0] for r in db.execute('SELECT * FROM samples ORDER BY route,frame')], ['hole', 'next'])

    def test_failed_requires_explicit_retry(self):
        db = self.make_db(retry=True)
        self.assertEqual([r[0] for r in db.execute(queue.LEGACY_SWEEP)], ['hole', 'rejected', 'next'])

    def test_other_queries_and_writes_are_not_filtered(self):
        db = self.make_db()
        self.assertEqual(db.execute('SELECT state FROM samples WHERE sample_id=?', ('done',)).fetchone()[0], 'complete')
        db.execute("UPDATE samples SET state='complete' WHERE state='pending'")
        self.assertEqual(db.execute(queue.LEGACY_SWEEP).fetchall(), [])
        self.assertEqual(db.execute('SELECT COUNT(*) FROM samples').fetchone()[0], 4)

    def test_connection_adapter_restored_on_error(self):
        original = queue.core.connect
        with patch.object(queue.core, 'run', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                queue.run_pending({}, '/unused')
        self.assertIs(queue.core.connect, original)


class CompletedAuditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name)
        self.config = {'cache_roots': {}}
        self.producer = {'test': 'pinned-contract'}
        db = queue.core.connect(self.state / 'index.sqlite')
        queue.core.pin(db, 'producer', self.producer)
        db.execute("INSERT INTO samples(sample_id,route,frame,state) VALUES('complete','a',1,'complete')")
        db.execute("INSERT INTO samples(sample_id,route,frame,state) VALUES('pending','a',2,'pending')")
        db.commit(); db.close()
        p = patch.object(queue.core, 'feature_producer', return_value=self.producer)
        p.start(); self.addCleanup(p.stop)

    def test_audit_only_loads_complete_and_does_not_generate(self):
        with patch.object(queue.core, 'load_feature_row', return_value={}) as load:
            result = queue.verify_completed(self.config, self.state, limit=1)
        self.assertEqual(result['valid'], 1)
        self.assertEqual(load.call_args.args[0]['sample_id'], 'complete')

    def test_missing_or_corrupt_are_reported_without_index_changes(self):
        for exc in (FileNotFoundError('missing'), AdapterError('invalid_feature_cache', 'checksum')):
            with patch.object(queue.core, 'load_feature_row', side_effect=exc):
                result = queue.verify_completed(self.config, self.state)
            self.assertEqual(result['invalid'], 1)
            with sqlite3.connect(self.state / 'index.sqlite') as db:
                self.assertEqual(db.execute("SELECT state FROM samples WHERE sample_id='complete'").fetchone()[0], 'complete')

    def test_device_fault_pauses_without_invalidating_samples(self):
        with patch.object(queue.core, 'load_feature_row', side_effect=OSError(errno.EIO, 'device error')):
            with self.assertRaises(queue.core.StoragePause):
                queue.verify_completed(self.config, self.state)
        self.assertFalse((self.state / 'verification_result.json').exists())

    def test_producer_identity_mismatch_stops_before_loading(self):
        with patch.object(queue.core, 'feature_producer', return_value={'test': 'changed'}), patch.object(queue.core, 'load_feature_row') as load:
            with self.assertRaises(queue.core.StoragePause):
                queue.verify_completed(self.config, self.state)
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
