"""Storage-only tests: exact labels, invalid encodings, and rename/commit recovery."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tools.training.diffusiondrive_v1 import target_storage as storage


def payload():
    return {'schema':storage.core.SCHEMA,'sample_id':'s','dataset_id':'d','route':'r','frame':0,
            'target_producer_sha256':'p','sources':[],'reference_pose':{},'current_annotation_sha256':'anno',
            'coordinate_contract':json.loads(storage.core.CONTRACT_PATH.read_text())['coordinates'],'map':{},
            'targets':{'trajectory':torch.arange(24,dtype=torch.float32).reshape(8,3),
                       'agent_states':torch.arange(150,dtype=torch.float32).reshape(30,5),
                       'agent_labels':torch.ones(30,dtype=torch.bool),
                       'bev_semantic_map':(torch.arange(128*256)%7).reshape(128,256)}}


class StorageCodecTests(unittest.TestCase):
    def test_roundtrip_is_lossless_and_original_is_not_mutated(self):
        original=payload();compact=storage.encode(original);restored=storage.decode(compact)
        self.assertEqual(compact['targets']['bev_semantic_map'].dtype,torch.uint8)
        self.assertEqual(original['targets']['bev_semantic_map'].dtype,torch.int64)
        storage.assert_equal(original,restored)
        a=io.BytesIO();b=io.BytesIO();storage.RAW_SAVE(original,a);storage.RAW_SAVE(compact,b)
        self.assertGreater(len(a.getvalue())-len(b.getvalue()),220000)

    def test_unknown_codec_and_out_of_range_refuse(self):
        p=storage.encode(payload());p['storage_encoding']='unknown'
        with self.assertRaises(storage.AdapterError):storage.decode(p)
        p=storage.encode(payload());p['targets']['bev_semantic_map'][0,0]=7
        with self.assertRaises(storage.AdapterError):storage.decode(p)

    def test_scope_restores_torch_methods_and_does_not_modify_other_payloads(self):
        old_save,old_load=torch.save,torch.load
        with self.assertRaises(RuntimeError):
            with storage.codec_scope(write=True):raise RuntimeError('interrupted')
        self.assertIs(torch.save,old_save);self.assertIs(torch.load,old_load)
        self.assertEqual(storage.encode({'features':{}}),{'features':{}})

    def test_recover_after_rename_and_after_commit(self):
        for fault_phase in ('renamed','committed'):
            with self.subTest(phase=fault_phase),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);spec={'path':str(folder)};original=payload()
                oldstream=io.BytesIO();storage.RAW_SAVE(original,oldstream);old=oldstream.getvalue()
                newstream=io.BytesIO();storage.RAW_SAVE(storage.encode(original),newstream);new=newstream.getvalue()
                (folder/'s.pt').write_bytes(old);(folder/'s.pt.uint8.partial').write_bytes(new)
                db=storage.core.open_index(folder/'index.sqlite');db.row_factory=__import__('sqlite3').Row
                row={'sample_id':'s','dataset_id':'d','route':'r','frame':0,'state':'complete','current_anno_sha256':'anno',
                     'reference_json':'{}','sources_json':'[]','map_json':'{}','relative_path':'s.pt',
                     'sha256':hashlib.sha256(old).hexdigest(),'bytes':len(old),'error_code':None,'error':None,'updated':0}
                cols=list(row);db.execute('INSERT INTO samples('+','.join(cols)+') VALUES('+','.join('?' for _ in cols)+')',list(row.values()))
                db.execute('CREATE TABLE codec_migrations(sample_id TEXT PRIMARY KEY,old_sha TEXT,old_bytes INTEGER,new_sha TEXT,new_bytes INTEGER,temp TEXT,backup TEXT,phase TEXT)')
                entry={'sample_id':'s','old_sha':row['sha256'],'old_bytes':len(old),'new_sha':hashlib.sha256(new).hexdigest(),
                       'new_bytes':len(new),'temp':'s.pt.uint8.partial','backup':'s.pt.legacy-backup','phase':'prepared'}
                cols=list(entry);db.execute('INSERT INTO codec_migrations('+','.join(cols)+') VALUES('+','.join('?' for _ in cols)+')',list(entry.values()));db.commit()
                def interrupt(phase):
                    if phase==fault_phase:raise RuntimeError('simulated interruption')
                with patch.object(storage.core,'storage_check',return_value=100*storage.GIB):
                    with self.assertRaises(RuntimeError):storage._finish(db,row,entry,spec,'p',interrupt)
                    current=dict(db.execute('SELECT * FROM samples').fetchone());phase=db.execute('SELECT phase FROM codec_migrations').fetchone()[0]
                    if phase=='done':storage._cleanup_done(current,entry,spec,'p')
                    else:storage._finish(db,current,entry,spec,'p')
                    current=dict(db.execute('SELECT * FROM samples').fetchone())
                    storage.assert_equal(original,storage.load_target_row(current,spec,'p'))
                self.assertFalse((folder/'s.pt.legacy-backup').exists());self.assertEqual(current['bytes'],len(new));db.close()


if __name__=='__main__':unittest.main()
