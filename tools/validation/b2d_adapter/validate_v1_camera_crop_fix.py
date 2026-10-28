"""Five real camera pairs, versioned Dataset caches and one full V1 loss forward."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
sys.dont_write_bytecode=True
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image
import torch
from torch.utils.data import default_collate

from tools.b2d_adapter import FrameReader,RouteIndex,AdapterError
from tools.b2d_adapter.native_builders import B2DObservation
from tools.b2d_adapter.v1_builders import DiffusionDriveV1FeatureBuilder,v1_native_config,official_v1_module,validate_v1_config
from tools.b2d_adapter.v1_label_rules import label_rules
from tools.training.diffusiondrive_v1.runtime import fit_configuration,model_configuration,native_configuration,initialization_manifest
from tools.training.diffusiondrive_v1.prepared import prepared_datasets,load_artifact,file_digest,digest,atomic_json,producer_identity,compatible_producer
from tools.validation.b2d_adapter.audit_v1_camera_alignment import geometry,signed_overlap,NAMES,stats,save_image

OUT=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/camera_crop_fix'


def main():
    torch.set_num_threads(4);torch.manual_seed(20261015)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    config,_=fit_configuration(ROOT/'tools/training/diffusiondrive_v1/camera_proportional_check.json')
    oldconfig,_=fit_configuration(ROOT/'tools/training/diffusiondrive_v1/pilot_original.json')
    model=model_configuration(config);native=native_configuration(config);legacy=native_configuration(oldconfig)
    assert native.camera_policy=='b2d-proportional-crop-v1' and legacy.camera_policy=='navsim-pixel-crop-v1'
    assert v1_native_config().camera_policy==native.camera_policy
    datasets,identity=prepared_datasets(ROOT/config['prepared_manifest'],config,model)
    olddatasets,oldidentity=prepared_datasets(ROOT/oldconfig['prepared_manifest'],oldconfig,model)
    oldmanifest=json.loads((ROOT/oldconfig['prepared_manifest']).read_text())
    oldrows={row['token']:row for values in oldmanifest['accepted'].values() for row in values}
    report={'schema':'v1-camera-crop-fix-v1','camera_policy':native.camera_policy,
            'prepared_dataset':identity,'historical_dataset':oldidentity,'frames':[],
            'scope':'compatibility only; no training or metric comparisons for old checkpoint under new camera input'}
    samples=[]
    for dataset in datasets.values():
        for index,token in enumerate(dataset.tokens):
            cached,targets=dataset[index]
            oldfeatures,oldtargets=load_artifact(oldrows[token])
            route,frame=token.rsplit('::',1);reader=FrameReader(RouteIndex(route))
            builder=DiffusionDriveV1FeatureBuilder(reader,native,model,config['lidar_decoder_python'])
            observation=builder.read_current(reader.index.ref(int(frame)))
            actual=builder.build_features(observation).tensors
            assert all(torch.equal(actual[k],cached[k]) for k in cached)
            assert all(torch.equal(actual[k],oldfeatures[k]) for k in ('lidar_feature','status_feature'))
            assert all(torch.equal(targets[k],oldtargets[k]) for k in targets)
            oldbuilder=DiffusionDriveV1FeatureBuilder(reader,legacy,model,config['lidar_decoder_python'])
            oldactual=oldbuilder.build_features(B2DObservation(observation.raw,observation.motion,legacy.provenance()['policy_sha256'])).tensors
            assert all(torch.equal(oldactual[k],oldfeatures[k]) for k in oldfeatures)
            # A second real JPG decode, not a tensor/decoded-image memory cache.
            second=reader.read(reader.index.ref(int(frame)),cameras=True)
            repeated=builder.build_camera(B2DObservation(second,None,native.provenance()['policy_sha256'])).tensors['camera_feature']
            assert torch.equal(repeated,actual['camera_feature'])
            images=[second.cameras[name] for name in NAMES]
            oldparts=[a[28:-28,416:-416] if i!=1 else a[28:-28] for i,a in enumerate(images)]
            newparts=[a[23:-23,346:-346] if i!=1 else a[23:-23,:] for i,a in enumerate(images)]
            oldstitched=np.concatenate(oldparts,axis=1);newstitched=np.concatenate(newparts,axis=1)
            assert [list(a.shape[:2][::-1]) for a in newparts]==[[908,854],[1600,854],[908,854]]
            assert newstitched.shape==(854,3416,3) and newstitched.shape[1]==4*newstitched.shape[0]
            oldrgb=cv2.resize(oldstitched,(1024,256));newrgb=cv2.resize(newstitched,(1024,256),interpolation=cv2.INTER_LINEAR)
            expected=official_v1_module('transfuser_features').transforms.ToTensor()(newrgb)
            assert torch.equal(expected,actual['camera_feature'])
            assert not torch.equal(actual['camera_feature'],oldfeatures['camera_feature'])
            assert np.array_equal(images[0],cv2.imread(str(reader.index.path(second.ref,NAMES[0])))[:,:,::-1])
            folder=OUT/(Path(route).name.split('_Weather')[0]+'_frame'+f'{int(frame):05d}');folder.mkdir(exist_ok=True)
            for key,value in [('old_stitched',oldstitched),('new_stitched',newstitched),('old_input',oldrgb),('new_input',newrgb)]:
                save_image(folder/(key+'.png'),value)
            for name,a in zip(NAMES,newparts):save_image(folder/(name+'_new_crop.png'),a)
            np.savez_compressed(folder/'camera_comparison.npz',old_camera_feature=oldfeatures['camera_feature'].numpy(),
                                new_camera_feature=actual['camera_feature'].numpy())
            calibrated={}
            for name in NAMES:
                a=geometry(second.cameras,second.sensors[name],second.ego_box,name)
                b=geometry(second.cameras,second.sensors[name],second.ego_box,name,346,23)
                calibrated[name]={'intrinsic':b['intrinsic'],'camera_to_ego_carla':b['camera_to_ego_carla'],
                    'relative_target_bearing_deg':b['relative_target_bearing_deg'],
                    'source_roi_uv':b['source_roi_uv'],'old_angle_interval':a['current_angle_interval_target_deg'],
                    'new_angle_interval':b['current_angle_interval_target_deg'],'source_path':str(reader.index.path(second.ref,name))}
            overlaps={policy:[signed_overlap(calibrated[a][policy+'_angle_interval'],calibrated[b][policy+'_angle_interval'])
                              for a,b in zip(NAMES[:-1],NAMES[1:])] for policy in ('old','new')}
            assert all(x>1.6 for x in overlaps['new']) and all(x<0 for x in overlaps['old'])
            fig,axes=plt.subplots(2,2,figsize=(16,8),constrained_layout=True)
            for ax,image,title in zip(axes.flat,[oldstitched,newstitched,oldrgb,newrgb],
                    ['旧拼接 3136×844 · 比例3.7156','新版拼接 3416×854 · 比例4:1',
                     '旧输入 1024×256 · 相对横向拉伸7.65%','新版输入 1024×256 · 横纵等比例缩放']):
                ax.imshow(image);ax.set_title(title);ax.axis('off')
            fig.suptitle(Path(route).name+f' · Frame {frame}\n真实JPG直接裁剪，非几何重投影；近场视差保留',fontsize=12)
            fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
            record={'token':token,'source_wh':[[a.shape[1],a.shape[0]] for a in images],
                    'new_crop_wh':[[a.shape[1],a.shape[0]] for a in newparts],
                    'new_stitched_wh':[3416,854],'new_aspect':4.,
                    'new_scale_x':1024/3416,'new_scale_y':256/854,
                    'tensor':dict(stats(actual['camera_feature'].permute(1,2,0).numpy()),training_shape=[3,256,1024]),
                    'calibration':calibrated,'adjacent_overlap_deg':overlaps,
                    'fresh_all_features_equal_new_cache':True,'second_jpg_decode_camera_equal':True,
                    'legacy_all_features_equal_historical_cache':True,'lidar_status_all_targets_equal_historical':True,
                    'artifact_directory':folder.name}
            atomic_json(folder/'frame_check.json',record);report['frames'].append(record)
            samples.append((cached,targets))
    # Wrong-policy manifests must fail even when shape/data/label classes match.
    errors={}
    for key,path,c in [('old_manifest_new_policy',ROOT/oldconfig['prepared_manifest'],dict(oldconfig,camera_policy=native.camera_policy)),
                       ('new_manifest_old_policy',ROOT/config['prepared_manifest'],dict(config,camera_policy=legacy.camera_policy))]:
        try:prepared_datasets(path,c,model)
        except AdapterError as e:
            assert e.code=='prepared_contract_mismatch';errors[key]=str(e)
        else:raise AssertionError('mixed camera policies accepted')
    # Verify exact old prepared-resume semantics, without executing resumed training.
    from tools.training.diffusiondrive_v1.sampling import CommittedSampler
    historical_result=json.loads((ROOT/'outputs/diffusiondrive_v1/pilot_original_resume/fit_result.json').read_text())
    semantic={'data':oldconfig['data'],'map_paths':oldconfig['map_paths'],'bev_producer_sha256':None,
              'prepared_data':oldidentity,'sampler_schema':CommittedSampler.schema,
              'adapter':validate_v1_config(legacy,model),'initialization':initialization_manifest(oldconfig,model),
              'seed':oldconfig['seed'],'batch_size':oldconfig['batch_size'],'lr':oldconfig['learning_rate'],
              'precision':oldconfig['training']['precision'],'evaluation_noise_seed':oldconfig['evaluation_noise_seed']}
    legacy_hash=hashlib.sha256(json.dumps(semantic,sort_keys=True).encode()).hexdigest()
    assert legacy_hash==historical_result['contract_sha256']
    new_semantic=dict(semantic,adapter=validate_v1_config(native,model))
    assert hashlib.sha256(json.dumps(new_semantic,sort_keys=True).encode()).hexdigest()!=legacy_hash
    checkpoint_path=ROOT/'outputs/diffusiondrive_v1/pilot_original_resume/checkpoints/last.ckpt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    agent=official_v1_module('transfuser_agent').TransfuserAgent(model,config['learning_rate'],checkpoint_path=None)
    agent.load_state_dict({k.removeprefix('agent.'):v for k,v in checkpoint['state_dict'].items()},strict=True)
    del checkpoint
    features,targets=default_collate(samples[:2]);agent.train()
    with torch.no_grad():prediction=agent.forward(features,targets=targets);losses=agent.compute_loss(features,targets,prediction)
    assert prediction['trajectory'].shape==(2,8,3) and prediction['bev_semantic_map'].shape==(2,7,128,256)
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values())
    assert all(torch.isfinite(v).all() for v in prediction.values() if isinstance(v,torch.Tensor))
    report['forward']={'batch_size':2,'device':'cpu','optimizer_steps':0,'checkpoint_strict_load':str(checkpoint_path),
                       'losses_finite':True,'weighted_official_losses':{k:float(v) for k,v in losses.items()},
                       'prediction_shapes':{k:list(v.shape) for k,v in prediction.items() if isinstance(v,torch.Tensor)},
                       'scope':'legacy-camera checkpoint under new camera inputs: compatibility only, not evaluation metrics'}
    report['isolation']={'wrong_policy_rejections':errors,'historical_prepared_resume_contract_unchanged':legacy_hash,
                         'new_camera_contract_differs':True,'cache_namespace':native.camera_policy}
    protected=json.loads((OUT/'protected_before.json').read_text())
    differences=[p for p,h in protected.items() if file_digest(p)!=h]
    assert not differences,differences
    report.update(status='pass',protected_file_count=len(protected),protected_sha_changes=differences)
    atomic_json(OUT/'results.json',report)
    print(json.dumps({'status':'pass','frames':len(report['frames']),
                      'new_overlap_deg':[f['adjacent_overlap_deg']['new'] for f in report['frames']],
                      'forward_losses_finite':True,'protected_sha_changes':differences},ensure_ascii=False))


if __name__=='__main__':main()
