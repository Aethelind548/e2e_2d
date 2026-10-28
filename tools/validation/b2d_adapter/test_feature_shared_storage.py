"""Shared physical roots retain ownership checks, global quota and exclusivity."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from tools.training.diffusiondrive_v1 import feature_shared_storage as s
from tools.training.diffusiondrive_v1 import feature_flatten as f
from tools.training.diffusiondrive_v1 import feature_migration as m


class SharedRootTests(unittest.TestCase):
    def setup_roots(self,folder):
        cache=folder/'features'; cache.mkdir()
        producer={'implementation':'frozen'}; sha=s.core.digest(producer)
        spec={'path':str(cache),'mountpoint':str(folder),'uuid':'uuid','fstype':'ntfs3','budget_gib':10,'minimum_free_gib':1}
        members=[]; configs=[]; registry_path=folder/'registry.json'
        for index in ['one','two']:
            state=folder/index;state.mkdir()
            db=sqlite3.connect(state/'index.sqlite');db.execute('CREATE TABLE meta(key TEXT,value TEXT)')
            db.executemany('INSERT INTO meta VALUES(?,?)',[('index_id',json.dumps(index)),('producer',json.dumps(producer))]);db.commit();db.close()
            members.append({'index_id':index,'state':str(state)})
            configs.append({'dataset_id':'dataset','state_dir':str(state),'shared_feature_roots':str(registry_path),'cache_roots':{'cache_a':dict(spec)}})
        registry={'schema':s.REGISTRY_SCHEMA,'dataset_id':'dataset','producer_sha256':sha,'roots':{'cache_a':{'spec':spec,'storage_owner_id':'shared-owner','members':members}}}
        registry_path.write_text(json.dumps(registry));(cache/'cache_root.json').write_text(json.dumps(s.marker(registry,'cache_a')))
        (cache/'one.pt').write_bytes(b'first');(cache/'two.pt').write_bytes(b'second')
        return configs,sha,cache

    def test_same_root_measures_both_jobs_and_prevents_simultaneous_writers(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(s.core,'active_mount'):
            configs,sha,cache=self.setup_roots(Path(tmp))
            first=s.SharedCacheStorage(configs[0],sha,'one')
            try:
                self.assertEqual(first.used['cache_a'],s.ORIGINAL_STORAGE.measure(cache))
                with self.assertRaises(s.core.StoragePause):s.SharedCacheStorage(configs[1],sha,'two')
            finally:first.close()
            second=s.SharedCacheStorage(configs[1],sha,'two');second.close()

    def test_unregistered_job_producer_spec_and_marker_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(s.core,'active_mount'):
            configs,sha,cache=self.setup_roots(Path(tmp))
            with self.assertRaises(s.core.StoragePause):s.validate_registration(configs[0],index_id='unknown')
            with self.assertRaises(s.core.StoragePause):s.validate_registration(configs[0],producer_sha='other')
            changed=json.loads(json.dumps(configs[0]));changed['cache_roots']['cache_a']['budget_gib']=999
            with self.assertRaises(s.core.StoragePause):s.validate_registration(changed)
            (cache/'cache_root.json').write_text('{}')
            with self.assertRaises(s.core.StoragePause):s.validate_registration(configs[0])

    def test_scope_restores_original_storage_on_error(self):
        original=s.core.CacheStorage
        with self.assertRaises(RuntimeError):
            with s.storage_scope({'shared_feature_roots':'registry'}):
                self.assertIs(s.core.CacheStorage,s.SharedCacheStorage)
                raise RuntimeError('interruption')
        self.assertIs(s.core.CacheStorage,original)
        with s.storage_scope({}):self.assertIs(s.core.CacheStorage,original)

    def test_flatten_rename_crash_is_recoverable_and_wrong_collision_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old.pt';new=root/'flat'/'sample.pt';content=b'unchanged tensor bytes'
            row={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()};old.write_bytes(content)
            with patch.object(m,'directory_sync',side_effect=OSError('after rename')):
                with self.assertRaises(OSError):f.move_verified_copy(old,new,row)
            self.assertFalse(old.exists());self.assertEqual(new.read_bytes(),content)
            f.move_verified_copy(old,new,row)
            old.write_bytes(content);new.write_bytes(b'wrong existing payload')
            with self.assertRaises(m.StoragePause):f.move_verified_copy(old,new,row)
            self.assertEqual(old.read_bytes(),content)

    def test_namespace_pruning_never_removes_untracked_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'remaining';path.mkdir();expected={'index_id':'old'}
            (path/'cache_root.json').write_text(json.dumps(expected));(path/'precious.pt').write_bytes(b'keep')
            with self.assertRaises(m.StoragePause):f.prune_empty_namespace(path,expected)
            self.assertEqual((path/'precious.pt').read_bytes(),b'keep')
            (path/'precious.pt').unlink();(path/'empty').mkdir();(path/'.writer.lock').touch()
            f.prune_empty_namespace(path,expected);self.assertFalse(path.exists())

    def test_full_flatten_resumes_after_first_rename_without_touching_sources_or_indexes(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as patches:
            root=Path(tmp);state=root/'migration';state.mkdir();registry_path=root/'shared.json'
            for obj,key,value in [(m,'ROOT',root),(m,'STATE',state),(s,'REGISTRY',registry_path)]:
                patches.enter_context(patch.object(obj,key,value))
            patches.enter_context(patch.object(m,'active_mount'))
            patches.enter_context(patch.object(m,'validate_job'))
            patches.enter_context(patch.object(m,'validate_aliases'))
            jobs=[];records=[];originals=[]
            for job_id,name in enumerate(['baseline','remaining']):
                job_state=root/('state_'+name);job_state.mkdir()
                sentinel=job_state/'index.sqlite';sentinel.write_bytes(b'unchanged original index')
                job={'name':name,'state':str(job_state),'index_id':name,'dataset_id':'d',
                     'producer_sha256':'p','config':str(job_state/'config.json'),'destinations':{},'aliases':{},
                     'before':{'cache_roots':{}},'after':{'cache_roots':{}}}
                for disk_id,rid in enumerate(['cache_a','cache_b']):
                    parent=root/rid/'features';folder=parent/name;folder.mkdir(parents=True)
                    spec={'path':str(folder),'mountpoint':str(root/rid),'uuid':rid,'fstype':'ntfs3','budget_gib':450,'minimum_free_gib':100}
                    job['destinations'][rid]=spec;job['aliases'][rid]=dict(spec)
                    job['after']['cache_roots'][rid]=dict(spec)
                    (folder/'cache_root.json').write_text(json.dumps(m.owner(job,rid)))
                    sid='%02d%02d'%(job_id,disk_id);relative='schema/producer/'+sid+'.pt'
                    content=('sample '+sid).encode();source=root/('original_'+sid);source.write_bytes(content)
                    copy=m.safe_path(spec,relative);copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(content)
                    records.append((name,rid,sid,relative,hashlib.sha256(content).hexdigest(),len(content),'copied',m.stamp(source)))
                    originals.append((source,content,folder,parent/relative))
                jobs.append(job)
            plan={'jobs':jobs,'disks':{'cache_a':{'budget_gib':450},'cache_b':{'budget_gib':650}}}
            m.durable_json(state/'plan.json',plan)
            db=m.connection();db.executemany('INSERT INTO files VALUES(?,?,?,?,?,?,?,?)',records);db.commit();db.close()
            real_move=f.move_verified_copy;failed=False
            def fail_after_move(*args):
                nonlocal failed
                real_move(*args)
                if not failed:
                    failed=True;raise OSError('death after first rename')
            with patch.object(f,'move_verified_copy',side_effect=fail_after_move):
                with self.assertRaises(OSError):f.flatten()
            self.assertTrue((state/'flatten_in_progress.json').exists())
            self.assertFalse(m.read_json(state/'plan.json').get('flat_layout',False))
            f.flatten()
            self.assertTrue(m.read_json(state/'plan.json')['flat_layout'])
            self.assertFalse((state/'flatten_in_progress.json').exists())
            for source,content,old,new in originals:
                self.assertEqual(source.read_bytes(),content)
                self.assertEqual(new.read_bytes(),content)
                self.assertFalse(old.exists())
            for job in jobs:self.assertEqual((Path(job['state'])/'index.sqlite').read_bytes(),b'unchanged original index')


if __name__=='__main__':unittest.main()
