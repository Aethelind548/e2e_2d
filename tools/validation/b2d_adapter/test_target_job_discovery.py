"""Independent supervision discovery and bounded retry queue, without sensor data."""
import errno
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.b2d_adapter.config import CandidateConfig
from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1 import target_jobs as jobs


class TargetDiscoveryTests(unittest.TestCase):
    def test_annotation_only_all_routes_and_future_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for route, count in [('a',45), ('b',43)]:
                folder = root / route / 'anno'; folder.mkdir(parents=True)
                for frame in range(count): (folder / f'{frame:05}.json.gz').touch()
            db = jobs.open_index(root / 'state' / 'index.sqlite')
            cfg = {'dataset_root': str(root), 'dataset_id': 'same-id', 'target_root': {}}
            with patch.object(jobs.core, 'storage_check', return_value=1000):
                self.assertTrue(jobs.discover(db, cfg, SimpleNamespace(geometry=CandidateConfig()), 'frozen'))
                self.assertTrue(jobs.discover(db, cfg, SimpleNamespace(geometry=CandidateConfig()), 'frozen'))
            self.assertEqual(jobs.summary(db)['states'], {'ineligible':80, 'pending':8})
            self.assertEqual(jobs.summary(db)['routes_discovered'], 2)
            row = db.execute("SELECT * FROM samples WHERE route='a' AND frame=44").fetchone()
            self.assertEqual(row['error_code'], 'insufficient_future')
            self.assertEqual(row['sample_id'], jobs.sample_id('same-id','a',44))
            # No sensor folder exists, so discovering all candidates cannot depend on it.
            index = jobs.AnnotationIndex(root / 'a')
            with self.assertRaises(AdapterError): index.path(index.ref(0),'lidar')
            db.close()

    def test_annotation_gap_is_not_filled(self):
        from tools.b2d_adapter.temporal_sampler import TemporalSampler
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'anno';folder.mkdir()
            for frame in range(45):
                if frame != 10: (folder/f'{frame}.json.gz').touch()
            index=jobs.AnnotationIndex(tmp)
            with self.assertRaises(AdapterError) as caught:
                TemporalSampler().sample(index,index.ref(0),full_scene=False,target_only=True)
            self.assertEqual(caught.exception.code,'missing_frame')

    def test_paged_queue_never_retries_same_failure_in_this_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=jobs.open_index(Path(tmp)/'index.sqlite')
            db.executemany('INSERT INTO samples(sample_id,dataset_id,route,frame,state,updated) VALUES(?,?,?,?,?,0)',
                           [(str(i),'d','r',i,'pending') for i in range(600)])
            db.commit(); visited=[]
            for row in jobs.pending_rows(db, retry_failed=True):
                visited.append(row['frame'])
                db.execute("UPDATE samples SET state='failed_retryable' WHERE sample_id=?",(row['sample_id'],))
            self.assertEqual(visited,list(range(600)))
            self.assertEqual(list(jobs.pending_rows(db)),[])
            db.close()

    def test_device_fault_and_retryable_access_are_distinct(self):
        self.assertEqual(jobs.failure_state(OSError(errno.EAGAIN,'again')),'failed_retryable')
        self.assertEqual(jobs.failure_state(AdapterError('invalid_map_polygon','self intersection')),'ineligible')
        for code in (errno.EIO,errno.EROFS,errno.EACCES,errno.ENOENT):
            with self.assertRaises(jobs.StoragePause): jobs.failure_state(OSError(code,'disk'))
        try:
            raise OSError(errno.EIO,'wrapped device fault')
        except OSError as cause:
            error=AdapterError('annotation_decode_failed','read');error.__cause__=cause
        with self.assertRaises(jobs.StoragePause): jobs.failure_state(error)


if __name__ == '__main__': unittest.main()
