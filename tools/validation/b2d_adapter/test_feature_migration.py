"""Relocation safety: resumable copies, guarded cleanup and interrupted cutover."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools.training.diffusiondrive_v1 import feature_migration as m


class MigrationTests(unittest.TestCase):
    def test_copy_replays_rename_without_overwrite_and_deletes_only_verified_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source.pt'; dest=root/'new'/'file.pt'
            content=b'actual serialized bytes\0'*1000; source.write_bytes(content)
            row={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
            stamp=m.copy_verified(source,dest,row)
            # Simulate rename success followed by process death before journal commit.
            self.assertEqual(m.copy_verified(source,dest,row),stamp)
            self.assertEqual(source.read_bytes(),content)
            m.safe_delete(source,dest,dict(row,source_stamp=stamp))
            self.assertFalse(source.exists()); self.assertEqual(dest.read_bytes(),content)
            m.safe_delete(source,dest,dict(row,source_stamp=stamp))

    def test_corrupt_destination_and_changed_source_are_never_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; dest=root/'dest'
            source.write_bytes(b'good')
            row={'bytes':4,'sha256':hashlib.sha256(b'good').hexdigest()}
            stamp=m.copy_verified(source,dest,row)
            dest.write_bytes(b'evil')
            with self.assertRaises(m.StoragePause):m.safe_delete(source,dest,dict(row,source_stamp=stamp))
            self.assertEqual(source.read_bytes(),b'good')
            with self.assertRaises(m.StoragePause):m.copy_verified(source,dest,row)
            self.assertEqual(dest.read_bytes(),b'evil')
            dest.write_bytes(b'good'); source.write_bytes(b'new source')
            with self.assertRaises(m.StoragePause):m.safe_delete(source,dest,dict(row,source_stamp=stamp))
            self.assertTrue(source.exists())

    def test_io_failure_and_wrong_source_checksum_keep_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; dest=root/'new'/'file'
            source.write_bytes(b'original')
            row={'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}
            with patch.object(m.os,'fsync',side_effect=OSError('I/O error')):
                with self.assertRaises(OSError):m.copy_verified(source,dest,row)
            self.assertTrue(source.exists()); self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name+'.external_migration.partial').exists())
            with self.assertRaises(m.StoragePause):m.copy_verified(source,dest,dict(row,sha256='0'*64))
            self.assertTrue(source.exists()); self.assertFalse(dest.exists())

    def test_cutover_fence_blocks_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp); m.migration_preflight(state)
            (state/'external_migration_cutover.json').write_text('{}')
            with self.assertRaises(m.StoragePause):m.migration_preflight(state)

    def test_committed_mapping_rejects_stale_worker_but_allows_quota_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp)
            identity={'path':'/new/cache','mountpoint':'/new','uuid':'id','fstype':'ntfs3'}
            m.durable_json(state/'external_migration_active.json',{'roots':{'cache_a':identity}})
            m.migration_preflight(state,{'cache_roots':{'cache_a':dict(identity,budget_gib=500)}})
            with self.assertRaises(m.StoragePause):
                m.migration_preflight(state,{'cache_roots':{'cache_a':dict(identity,path='/old/cache')}})

    def test_cutover_requires_real_bind_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); physical=root/'physical'; alias=root/'alias'
            physical.mkdir(); alias.mkdir()
            job={'destinations':{'cache_a':{'path':str(physical)}},'aliases':{'cache_a':{'path':str(alias)}}}
            with patch.object(m,'active_mount'),patch.object(m,'assert_owner'):
                with self.assertRaises(m.StoragePause):m.validate_aliases({'jobs':[job]})

    def test_interrupted_multi_config_switch_remains_fenced_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp); jobs=[]
            for name in ['baseline','remaining']:
                folder=state/name; folder.mkdir(); source=folder/'source'; source.mkdir()
                (source/'precious.pt').write_bytes(b'keep original')
                before={'cache_roots':{'cache_a':{'path':str(source)}}}
                after={'cache_roots':{'cache_a':{'path':str(folder/'destination')}}}
                config=folder/'config.json'; config.write_text(json.dumps(before))
                db=sqlite3.connect(folder/'index.sqlite')
                db.execute('CREATE TABLE samples(state TEXT,root_id TEXT)'); db.commit(); db.close()
                jobs.append({'name':name,'state':str(folder),'config':str(config),'before':before,'after':after,'aliases':{}})
            ledger=sqlite3.connect(state/'ledger.sqlite')
            ledger.execute('CREATE TABLE files(phase TEXT)');ledger.commit()
            original=m.durable_json
            failed=False
            def interrupted(path,value):
                nonlocal failed
                original(path,value)
                if str(path)==jobs[0]['config'] and value==jobs[0]['after'] and not failed:
                    failed=True
                    raise OSError('simulated death after first configuration replacement')
            with patch.object(m,'STATE',state),patch.object(m,'validate_aliases'),patch.object(m,'validate_job'),patch.object(m,'refresh'),patch.object(m,'copy_pending'),patch.object(m,'validate_reads'),patch('tools.training.diffusiondrive_v1.feature_service.alive',return_value=False),patch('tools.training.diffusiondrive_v1.feature_jobs.export_collection'):
                with patch.object(m,'durable_json',side_effect=interrupted):
                    with self.assertRaises(OSError):m.cutover({'jobs':jobs},ledger)
                for job in jobs:
                    with self.assertRaises(m.StoragePause):m.migration_preflight(Path(job['state']))
                    self.assertEqual((Path(job['before']['cache_roots']['cache_a']['path'])/'precious.pt').read_bytes(),b'keep original')
                self.assertEqual(m.read_json(jobs[0]['config']),jobs[0]['after'])
                self.assertEqual(m.read_json(jobs[1]['config']),jobs[1]['before'])
                record=m.cutover({'jobs':jobs},ledger)
                self.assertEqual(record['phase'],'committed')
                for job in jobs:
                    m.migration_preflight(Path(job['state']))
                    self.assertEqual(m.read_json(job['config']),job['after'])
            ledger.close()


if __name__=='__main__':unittest.main()
