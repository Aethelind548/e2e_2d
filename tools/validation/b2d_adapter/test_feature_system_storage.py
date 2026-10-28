"""System quota requires real free space; extensions must keep owner identities."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.training.diffusiondrive_v1 import feature_storage as storage


class SystemStorageTests(unittest.TestCase):
    def test_initial_quota_reserves_system_space(self):
        storage.capacity_check(250,40,0,295*storage.GIB)
        with self.assertRaises(storage.StoragePause):
            storage.capacity_check(300,40,0,295*storage.GIB)

    def test_future_capacity_cannot_be_granted_before_expansion(self):
        with self.assertRaises(storage.StoragePause):
            storage.capacity_check(550,40,250*storage.GIB,45*storage.GIB)
        storage.capacity_check(550,40,250*storage.GIB,345*storage.GIB)

    def test_cannot_shrink_below_existing_usage_or_remove_reserve(self):
        with self.assertRaises(storage.StoragePause): storage.capacity_check(200,40,250*storage.GIB,345*storage.GIB)
        with self.assertRaises(storage.StoragePause): storage.capacity_check(250,0,0,295*storage.GIB)

    def test_registry_and_owner_marker_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'features';folder.mkdir();plan_path=Path(tmp)/'plan.json'
            spec={'path':str(folder),'mountpoint':'/','uuid':storage.SYSTEM_UUID,'fstype':'ext4','budget_gib':250,'minimum_free_gib':40}
            producer={'pinned':'unchanged'};meta={'index_id':'index','producer':producer,'dataset_id':'data'}
            cfg={'dataset_id':'data','state_dir':'state','cache_roots':{storage.ROOT_ID:spec},
                 'storage_extensions':{storage.ROOT_ID:{'plan':str(plan_path)}}}
            plan={'schema':'b2d-v1-system-features-storage-v1','dataset_id':'data','state_dir':'state','spec':spec,
                  'index_id':'index','producer_sha256':storage.digest(producer)}
            plan_path.write_text(json.dumps(plan));marker=folder/'cache_root.json'
            marker.write_text(json.dumps({'schema':storage.SCHEMA,'root_id':storage.ROOT_ID,'dataset_id':'data','index_id':'index'}))
            with patch.object(storage,'SYSTEM_PATH',folder),patch.object(storage,'index_identity',return_value=meta):
                storage.validate_extension(cfg)
                cfg['cache_roots'][storage.ROOT_ID]=dict(spec,budget_gib=550)
                with self.assertRaises(storage.StoragePause):storage.validate_extension(cfg)
                cfg['cache_roots'][storage.ROOT_ID]=spec
                marker.write_text(json.dumps({'schema':storage.SCHEMA,'root_id':storage.ROOT_ID,'dataset_id':'data','index_id':'different'}))
                with self.assertRaises(storage.StoragePause):storage.validate_extension(cfg)

    def test_existing_two_disk_configs_need_no_system_registration(self):
        storage.validate_extension({'cache_roots':{'cache_a':{},'cache_b':{}}})
        with self.assertRaises(storage.StoragePause):storage.validate_extension({'cache_roots':{storage.ROOT_ID:{}}})


if __name__ == '__main__':unittest.main()
