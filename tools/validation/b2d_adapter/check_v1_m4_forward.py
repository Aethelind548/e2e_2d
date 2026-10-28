"""One real M4 batch through unchanged V1 training forward/loss; no optimizer."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
sys.dont_write_bytecode=True
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')


def main():
    import torch
    from torch.utils.data import default_collate
    from tools.training.diffusiondrive_v1.runtime import model_configuration
    from tools.training.diffusiondrive_v1.prepared import prepared_datasets,file_digest,atomic_json
    from tools.b2d_adapter.v1_builders import official_v1_module
    config=json.loads((ROOT/'tools/training/diffusiondrive_v1/m4_subset.json').read_text())
    model=model_configuration(config)
    datasets,_=prepared_datasets(ROOT/'outputs/diffusiondrive_v1/m4_prepared_final/v1_semantic_enriched/m4-v1/accepted_manifest.json',config,model)
    features,targets=default_collate([datasets['train'][0],datasets['validation'][0]])
    torch.set_num_threads(4);torch.manual_seed(20261014)
    path=ROOT/'outputs/diffusiondrive_v1/m2_fit_checked/checkpoints/last.ckpt'
    before=file_digest(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    agent=official_v1_module('transfuser_agent').TransfuserAgent(model,config['learning_rate'],checkpoint_path=None)
    agent.load_state_dict({k.removeprefix('agent.'):v for k,v in checkpoint['state_dict'].items()},strict=True)
    del checkpoint
    agent.train();started=time.monotonic()
    with torch.no_grad():
        prediction=agent.forward(features,targets=targets)
        losses=agent.compute_loss(features,targets,prediction)
    assert prediction['bev_semantic_map'].shape==(2,7,128,256)
    assert prediction['trajectory'].shape==(2,8,3)
    assert all(torch.isfinite(v).all() if isinstance(v,torch.Tensor) else __import__('math').isfinite(v) for v in losses.values())
    assert file_digest(path)==before
    result={'status':'pass','batch_size':2,'device':'cpu','branch':'official train forward with targets, no_grad',
        'optimizer_steps':0,'checkpoint_sha256_unchanged':before,
        'tokens':[datasets['train'].tokens[0],datasets['validation'].tokens[0]],
        'prediction_shapes':{k:list(v.shape) for k,v in prediction.items() if isinstance(v,torch.Tensor)},
        'weighted_official_losses':{k:float(v) for k,v in losses.items()},'seconds':time.monotonic()-started}
    atomic_json(ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/m4/forward_loss.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
