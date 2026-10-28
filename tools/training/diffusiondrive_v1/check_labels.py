"""One bounded real R0/R2 label check; writes evidence and reusable smoke tensors."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')


def main():
    import torch
    from torch.utils.data import DataLoader
    from tools.b2d_adapter.navsim_bridge import candidate_vehicle_boxes,PARKED_VEHICLE_PREFIX
    from tools.training.diffusiondrive_v1.runtime import configuration,model_configuration,datasets
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('baseline.json'))
    args=parser.parse_args();config=configuration(args.config);model=model_configuration(config)
    output=ROOT/config['label_output_dir'];output.mkdir(parents=True,exist_ok=True)
    result={'status':'running','samples':[]};samples=[];metadata=[]
    try:
        for dataset in datasets(config,model):
            for i,ref in enumerate(dataset.refs):
                sample=dataset.build_sample(i);raw=dataset.reader.read(ref)
                view=candidate_vehicle_boxes(raw,dataset.feature_builder.config.geometry)
                parked=[b for b in raw.annotation['bounding_boxes'] if str(b.get('type_id','')).startswith(PARKED_VEHICLE_PREFIX)]
                assert len(view.boxes)==sum(b['class']=='vehicle' for b in raw.annotation['bounding_boxes'])
                parked_checks=[]
                for box in parked:
                    assert str(box['id']) in view.provenance['source_actor_ids']
                    j=view.provenance['source_actor_ids'].index(str(box['id']))
                    import numpy as np
                    np.testing.assert_allclose(view.boxes[j,3:6],2*np.asarray(box['extent']),rtol=1e-6)
                    # Real source center/yaw are retained by the shared transformation.
                    from tools.b2d_adapter.coordinate_transform import world_to_ego,wrap_angle
                    import math
                    xy=world_to_ego(box['center'][:2],raw.actor_pose)
                    expected=[xy[0]-dataset.feature_builder.config.geometry.axle_dx_m,-xy[1],
                              wrap_angle(raw.actor_pose.yaw-math.radians(box['rotation'][2]))]
                    np.testing.assert_allclose(view.boxes[j,[0,1,6]],expected,atol=1e-5)
                    parked_checks.append({'id':str(box['id']),'transformed_box':view.boxes[j].tolist(),
                                          'source_extent':box['extent'],'geometry_check':'pass'})
                assert set(sample.targets)=={'trajectory','agent_states','agent_labels','bev_semantic_map'}
                bev=sample.targets['bev_semantic_map'];counts=torch.bincount(bev.flatten(),minlength=7).tolist()
                assert tuple(bev.shape)==(128,256) and bev.dtype==torch.int64
                assert counts[1]+counts[3]>0,'real road/centerline coverage missing'
                metadata.append(sample.metadata);samples.append((sample.features,sample.targets))
                result['samples'].append({'token':ref.token,'parked':parked_checks,
                    'valid_agent_count':int(sample.targets['agent_labels'].sum()),'bev_class_counts':counts,
                    'bev_source':sample.metadata['targets']['bev']})
        f,t=next(iter(DataLoader(samples,batch_size=config['batch_size'],num_workers=0)))
        result['batch_shapes']={'features':{k:list(v.shape) for k,v in f.items()},'targets':{k:list(v.shape) for k,v in t.items()}}
        bundle={'samples':samples,'metadata':metadata,'config':config,'model_contract':metadata[0]['contract']}
        torch.save(bundle,output/'real_samples.pt')
        result.update(status='pass',tensor_bundle=str(output/'real_samples.pt'),sample_count=len(samples))
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        (output/'labels.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'sample_count':len(samples),'report':str(output/'labels.json')},ensure_ascii=False))


if __name__=='__main__': main()
