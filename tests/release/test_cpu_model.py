"""No-data CPU smoke of the pinned official V1 architecture and original loss.

Temporary random initialization and synthetic anchors are test inputs, not
official pretrained assets or evidence of R2/real-checkpoint restoration.
"""
import numpy as np
import torch
import timm

from tools.b2d_adapter.v1_builders import official_v1_module


def test_official_v1_cpu_forward_backward_without_external_assets(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            config = official_v1_module('transfuser_config').TransfuserConfig()
            steps = np.arange(1, 9, dtype=np.float32)
            anchors = np.stack([np.stack((steps * (1 + mode / 10),
                                          steps ** 2 * (mode - 10) / 100), axis=-1)
                                for mode in range(20)])
            anchor_path = tmp_path / 'synthetic_anchor.npy'
            np.save(anchor_path, anchors)
            config.plan_anchor_path = str(anchor_path)
            backbone_path = tmp_path / 'synthetic_resnet34.pth'
            factory = timm.create_model
            backbone = factory(config.image_architecture, pretrained=False)
            torch.save(backbone.state_dict(), backbone_path)
            del backbone
            config.bkb_path = str(backbone_path)
            resolutions = []

            def local_initialization(name, *args, **kwargs):
                # Resolve only pretrained input bytes. The real timm factory,
                # official architecture, forward and loss remain unchanged.
                if kwargs.get('pretrained'):
                    kwargs['pretrained_cfg_overlay'] = {'file': str(backbone_path)}
                    resolutions.append(str(backbone_path))
                return factory(name, *args, **kwargs)

            monkeypatch.setattr(timm, 'create_model', local_initialization)
            model = official_v1_module('transfuser_model_v2').V2TransfuserModel(config).cpu().train()
            features = {'camera_feature': torch.rand(1, 3, 256, 1024),
                        'lidar_feature': torch.rand(1, 1, 256, 256),
                        'status_feature': torch.tensor([[1., 0., 0., 0., 2., 0., 0., 0.]])}
            targets = {'trajectory': torch.zeros(1, 8, 3),
                       'agent_states': torch.zeros(1, 30, 5),
                       'agent_labels': torch.zeros(1, 30, dtype=torch.bool),
                       'bev_semantic_map': torch.arange(128 * 256).reshape(1, 128, 256) % 7}
            targets['trajectory'][0, :, 0] = torch.arange(1, 9)
            targets['agent_states'][0, 0] = torch.tensor([10., 1., 0., 2., 4.])
            targets['agent_labels'][0, 0] = True
            predictions = model(features, targets=targets)
            losses = official_v1_module('transfuser_loss').transfuser_loss(targets, predictions, config)
            assert predictions['trajectory'].shape == (1, 8, 3)
            assert predictions['bev_semantic_map'].shape == (1, 7, 128, 256)
            assert torch.isfinite(losses['loss']).all()
            losses['loss'].backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            assert len(gradients) > 500
            assert all(torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g).item() > 0 for g in gradients)
            assert resolutions == [str(backbone_path)]
    finally:
        torch.set_num_threads(previous_threads)
