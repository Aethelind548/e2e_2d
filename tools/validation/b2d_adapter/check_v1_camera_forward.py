"""Real prepared Dataset roundtrip and original initialized V1 loss, no optimizer."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')


def main():
    import torch
    from torch.utils.data import default_collate
    from tools.b2d_adapter import FrameReader, RouteIndex, AdapterError
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1Dataset, official_v1_module
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration, bev_provider, initialization_manifest
    from tools.training.diffusiondrive_v1.prepared import prepared_datasets, atomic_json
    config = json.loads((ROOT / 'tools/training/diffusiondrive_v1/camera_proportional_check.json').read_text())
    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    model = model_configuration(config)
    datasets, identity = prepared_datasets(ROOT / config['prepared_manifest'], config, model)
    native = native_configuration(config)
    assert native.camera_policy == 'b2d-proportional-crop-v1'
    provider = bev_provider(config)
    pairs = [datasets['train'][0], datasets['validation'][0]]
    for dataset, item in [(datasets['train'], pairs[0]), (datasets['validation'], pairs[1])]:
        route, frame = dataset.tokens[0].rsplit('::', 1)
        reader = FrameReader(RouteIndex(route))
        fresh = DiffusionDriveV1Dataset(reader, [reader.index.ref(int(frame))], native, model,
                                       config['lidar_decoder_python'], provider)[0]
        assert all(torch.equal(fresh[j][k], item[j][k]) for j in (0, 1) for k in item[j])
    try:
        prepared_datasets(ROOT / config['prepared_manifest'], dict(config, camera_policy='navsim-pixel-crop-v1'), model)
    except AdapterError as exc:
        assert exc.code == 'prepared_contract_mismatch'
    else:
        raise AssertionError('Wrong Camera policy accepted')
    features, targets = default_collate(pairs)
    agent = official_v1_module('transfuser_agent').TransfuserAgent(model, config['learning_rate'], checkpoint_path=None)
    agent.train()
    started = time.monotonic()
    with torch.no_grad():
        prediction = agent.forward(features, targets=targets)
        losses = agent.compute_loss(features, targets, prediction)
    assert prediction['trajectory'].shape == (2, 8, 3)
    assert prediction['bev_semantic_map'].shape == (2, 7, 128, 256)
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values())
    assert all(torch.isfinite(v).all() for v in prediction.values() if isinstance(v, torch.Tensor))
    report = {'status': 'pass', 'batch_size': 2, 'device': 'cpu', 'optimizer_steps': 0,
              'initialization': initialization_manifest(config, model), 'prepared_dataset': identity,
              'fresh_features_and_all_targets_bit_equal': True, 'wrong_camera_policy_rejected': True,
              'losses': {k: float(v) for k, v in losses.items()}, 'seconds': time.monotonic() - started,
              'prediction_shapes': {k: list(v.shape) for k, v in prediction.items() if isinstance(v, torch.Tensor)}}
    atomic_json(ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/camera_crop_fix/forward_loss.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
