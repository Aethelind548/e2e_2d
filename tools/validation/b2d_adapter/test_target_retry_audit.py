"""Recovery safety boundaries: independent heads and explicit storage pauses."""
from types import SimpleNamespace
import unittest

import torch

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1.features_cache import StoragePause
from tools.training.diffusiondrive_v1.target_retry_audit import component_checks


class HeadIsolation(unittest.TestCase):
    def builder(self, detection_error=None, bev_error=None):
        trajectory=torch.arange(24,dtype=torch.float32).reshape(8,3)
        def detection(raw):
            if detection_error is not None:raise detection_error
            return SimpleNamespace(tensors={'agent_states':torch.zeros(30,5),'agent_labels':torch.zeros(30,dtype=torch.bool)},provenance={},validation={})
        def bev(*args):
            if bev_error is not None:raise bev_error
            return SimpleNamespace(tensors={'bev_semantic_map':torch.ones(128,256,dtype=torch.int64)},provenance={},validation={})
        return SimpleNamespace(
            trajectory_builder=SimpleNamespace(build_targets=lambda _:SimpleNamespace(tensors={'trajectory':trajectory},provenance={},validation={})),
            reader=SimpleNamespace(read=lambda _:object()),build_agent_targets=detection,
            bev_provider=SimpleNamespace(build_targets=bev),config=object(),model_config=object(),_validate_bev=lambda *_:None)

    def test_actor_error_does_not_discard_valid_trajectory_or_bev(self):
        result,tensors=component_checks(self.builder(detection_error=AdapterError('invalid_vehicle_box','real geometry unavailable')),object())
        self.assertFalse(result['full_training_eligible'])
        self.assertEqual(set(tensors),{'trajectory','bev_semantic_map'})
        self.assertEqual(result['components']['detection']['error_code'],'invalid_vehicle_box')
        self.assertNotIn('agent_labels',tensors)  # No negative labels fabricated for unknown actors.

    def test_map_error_does_not_discard_trajectory_or_detection(self):
        result,tensors=component_checks(self.builder(bev_error=AdapterError('invalid_map_polygon','self-intersection')),object())
        self.assertFalse(result['full_training_eligible'])
        self.assertEqual(set(tensors),{'trajectory','agent_states','agent_labels'})
        self.assertNotIn('bev_semantic_map',tensors)  # No all-background replacement.

    def test_io_failure_is_a_pause_not_a_semantic_reject(self):
        error=AdapterError('annotation_decode_failed','device IO error')
        error.__cause__=OSError(5,'Input/output error')
        with self.assertRaises(StoragePause):component_checks(self.builder(detection_error=error),object())

    def test_contract_fault_pauses_instead_of_partial_success(self):
        with self.assertRaises(StoragePause):
            component_checks(self.builder(bev_error=AdapterError('map_version_mismatch','unknown map SHA')),object())

    def test_complete_contract_is_validated(self):
        result,tensors=component_checks(self.builder(),object())
        self.assertTrue(result['full_training_eligible'])
        self.assertEqual(set(tensors),{'trajectory','agent_states','agent_labels','bev_semantic_map'})


if __name__=='__main__':unittest.main()
