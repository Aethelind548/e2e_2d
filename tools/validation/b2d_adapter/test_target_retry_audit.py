"""Recovery safety boundaries: independent heads and explicit storage pauses."""
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


class InterfaceReportIsolation(unittest.TestCase):
    """Exercise report routing only; this is not a real-network acceptance test."""

    def run_interface(self, directory, output_path=None):
        from tools.b2d_adapter import v1_builders
        from tools.validation.b2d_adapter import audit_v1_target_rejections as audit

        class Agent:
            def train(self):
                pass

            def forward(self, features, targets):
                return {'trajectory': targets['trajectory']}

            def compute_loss(self, features, targets, predictions):
                return {'loss': predictions['trajectory'].sum()}

        pair = ({'status_feature': torch.zeros(8)}, {'trajectory': torch.zeros(8, 3)})
        factory = SimpleNamespace(TransfuserAgent=lambda *args, **kwargs: Agent())
        with patch.object(audit, 'OUT', directory), patch.object(v1_builders, 'official_v1_module', return_value=factory):
            audit.interface([pair, pair], {'learning_rate': 1e-4}, object(), output_path=output_path)

    def test_recovered_report_never_replaces_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            baseline = directory / 'interface_validation.json'
            baseline.write_bytes(b'original frozen evidence')
            recovered = directory / 'recovered.json'
            self.run_interface(directory, recovered)
            self.assertEqual(baseline.read_bytes(), b'original frozen evidence')
            self.assertEqual(json.loads(recovered.read_text())['status'], 'pass')

    def test_existing_explicit_report_is_rejected_before_network_work(self):
        from tools.validation.b2d_adapter import audit_v1_target_rejections as audit

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'existing.json'
            output.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                audit.interface([], {}, None, output_path=output)
            self.assertEqual(output.read_bytes(), b'keep')

    def test_historical_default_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.run_interface(directory)
            self.assertEqual(json.loads((directory / 'interface_validation.json').read_text())['batch_size'], 2)


if __name__=='__main__':unittest.main()
