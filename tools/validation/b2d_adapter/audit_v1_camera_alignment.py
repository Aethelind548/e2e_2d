"""Read-only, five-real-frame camera audit. Candidates never enter training caches."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'repositories/DiffusionDrive'))
sys.dont_write_bytecode = True
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from tools.b2d_adapter import FrameReader, RouteIndex
from tools.b2d_adapter.route_index import CAMERAS
from tools.b2d_adapter.v1_builders import DiffusionDriveV1FeatureBuilder, official_v1_module, v1_native_config
from tools.b2d_adapter.v1_geometry import carla_rotation
from tools.training.diffusiondrive_v1.runtime import fit_configuration, model_configuration, native_configuration
from tools.training.diffusiondrive_v1.prepared import atomic_json, file_digest, load_artifact

NAMES = tuple(CAMERAS)
CASES = [('DynamicObjectCrossing_Town01', '日间直道'),
         ('HazardAtSideLane_Town03', '弯道 / 道路边缘车辆'),
         ('BlockedIntersection_Town04', '复杂路口 / 邻近车辆'),
         ('ParkingCrossingPedestrian_Town05', '近直道 / 停放车辆与行人场景'),
         ('InvadingTurn_Town10HD', '转弯 / 邻近车辆')]


def stats(value):
    a = np.asarray(value)
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'min': float(a.min()),
            'max': float(a.max()), 'mean': float(a.mean()), 'std': float(a.std()),
            'rgb_mean': a.reshape(-1,3).mean(0).tolist(), 'rgb_std': a.reshape(-1,3).std(0).tolist()}


def crops(images):
    return [images[0][28:-28,416:-416], images[1][28:-28], images[2][28:-28,416:-416]]


def rotation_angles(r):
    # Inverse of existing CARLA pitch/roll/yaw rotation, away from gimbal lock.
    return np.degrees([np.arcsin(np.clip(r[2,0],-1,1)),
                       np.arctan2(-r[2,1],r[2,2]), np.arctan2(r[1,0],r[0,0])]).tolist()


def angle_interval(rotation, k, u_bounds):
    # CARLA [forward,right,up] -> optical [right,down,forward].
    # Continuous u=[0,W] edge approximation; v=cy, pinhole, no translation/distortion.
    directions = np.array([[1.,(u-k[0,2])/k[0,0],0.] for u in u_bounds]) @ rotation.T
    angles = np.degrees(np.arctan2(-directions[:,1],directions[:,0]))
    return sorted(angles.tolist())


def signed_overlap(a,b):
    return min(a[1],b[1])-max(a[0],b[0])


def geometry(images, sensor, ego, name, side_margin=416, vertical_margin=28):
    im = images[name]
    h,w = im.shape[:2]
    k = np.asarray(sensor['intrinsic'],dtype=float)
    world_to_cam = np.asarray(sensor['world2cam'],dtype=float)
    ego_to_world = np.linalg.inv(np.asarray(ego['world2ego'],dtype=float))
    camera_to_ego = np.asarray(ego['world2ego'],dtype=float) @ np.linalg.inv(world_to_cam)
    r = camera_to_ego[:3,:3]
    camera_to_world = np.linalg.inv(world_to_cam)
    assert np.max(np.abs(camera_to_world[:3,3]-sensor['location'])) < 1e-3
    assert np.max(np.abs(camera_to_world[:3,:3]-carla_rotation(sensor['rotation']))) < 1e-5
    assert np.max(np.abs(ego_to_world[:3,:3]-carla_rotation(ego['rotation']))) < 1e-5
    assert [w,h] == [sensor['image_size_x'],sensor['image_size_y']]
    side = name != 'CAM_FRONT'
    bounds = [side_margin,w-side_margin] if side else [0,w]
    fov = np.degrees(np.arctan((w-k[0,2])/k[0,0])-np.arctan(-k[0,2]/k[0,0]))
    heading = np.degrees(np.arctan2(-r[1,0],r[0,0]))
    # Two-times sampling does not certify a new calibration. Candidate source
    # crop corresponds approximately to the scaled official pixel coordinates.
    reference_sx = 1920/w
    scaled_bounds = [416/reference_sx, w-416/reference_sx] if side else [0,w]
    return {'source_calibration': sensor, 'intrinsic': k.tolist(),
            'annotated_fov_deg': sensor['fov'], 'intrinsic_pinhole_fov_deg': float(fov),
            'camera_to_ego_carla': camera_to_ego.tolist(),
            'relative_carla_pitch_roll_yaw_deg': rotation_angles(r),
            'relative_target_bearing_deg': float(heading),
            'translation_target_forward_left_up_m': [float(camera_to_ego[0,3]),float(-camera_to_ego[1,3]),float(camera_to_ego[2,3])],
            'rotation_matrix_orthogonality_error': float(np.max(np.abs(r.T@r-np.eye(3)))),
            'source_roi_uv': [bounds[0],vertical_margin,bounds[1],h-vertical_margin],
            'source_cropped_wh': [bounds[1]-bounds[0],h-2*vertical_margin],
            'width_retained_fraction': (bounds[1]-bounds[0])/w,
            'height_retained_fraction': (h-2*vertical_margin)/h,
            'area_retained_fraction': (bounds[1]-bounds[0])*(h-2*vertical_margin)/(w*h),
            'original_angle_interval_target_deg': angle_interval(r,k,[0,w]),
            'current_angle_interval_target_deg': angle_interval(r,k,bounds),
            'diagnostic_preresize_angle_interval_target_deg': angle_interval(r,k,scaled_bounds),
            'distortion': {'status':'not present in supplied camera annotation; pinhole intervals are approximate',
                           'fields_present':[key for key in sensor if 'distort' in key or key.startswith('lens_')]}}


def save_image(path, a):
    Image.fromarray(a).save(path)


def montage(path, token, case, images, sliced, stitched, final, candidate):
    fig = plt.figure(figsize=(18,15),constrained_layout=True)
    gs = fig.add_gridspec(5,3,height_ratios=[1.1,1.05,.75,.7,.7])
    for i,name in enumerate(NAMES):
        ax=fig.add_subplot(gs[0,i]);ax.imshow(images[i]);ax.axis('off')
        h,w=images[i].shape[:2];x=416 if i!=1 else 0
        ax.add_patch(Rectangle((x,28),w-2*x,h-56,fill=False,edgecolor='#ffbb00',lw=2))
        ax.set_title(f'原始 {name}\n{w}×{h} RGB · 黄色框为实际保留区域',fontsize=12)
        ax=fig.add_subplot(gs[1,i]);ax.imshow(sliced[i]);ax.axis('off')
        ax.set_title(f'实际固定裁剪后 · {sliced[i].shape[1]}×{sliced[i].shape[0]}',fontsize=12)
    for row,a,title in [(2,stitched,'横向拼接，Resize 前 · 3136×844 · 宽高比 3.7156'),
                        (3,final,'当前训练输入 · 1024×256 · 横向相对纵向拉伸约 7.65%'),
                        (4,candidate,'诊断候选：原图先缩放至1920×1080，再执行官方裁剪 · 不写入训练缓存')]:
        ax=fig.add_subplot(gs[row,:]);ax.imshow(a);ax.axis('off');ax.set_title(title,fontsize=13)
    fig.suptitle(f'{case}\n{token.split("/")[-1]}\n真实 JPG → 当前预处理；候选只用于检查，不表示 NAVSIM 几何等效',fontsize=15)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def coverage_plot(path, records):
    fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True,constrained_layout=True)
    colors=['#3b78b5','#26946a','#d78834']
    for ax,key,title in zip(axes,['original_angle_interval_target_deg','current_angle_interval_target_deg','diagnostic_preresize_angle_interval_target_deg'],
                            ['源图全水平视场','当前固定 416px 裁剪：约 1.42° 远场角隙','预缩放诊断候选：更宽的源图裁剪范围，仍非 NAVSIM 等效']):
        for i,(name,color) in enumerate(zip(NAMES,colors)):
            lo,hi=records[name][key]
            ax.plot([lo,hi],[i,i],lw=12,alpha=.75,color=color,solid_capstyle='butt')
            ax.text(lo,i+.18,f'{lo:.2f}°',fontsize=10);ax.text(hi,i+.18,f'{hi:.2f}°',ha='right',fontsize=10)
        ax.set_yticks(range(3),['左前','正前','右前']);ax.set_ylim(-.5,2.6)
        ax.set_title(title);ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('自车水平角 / °：y 向左为正；忽略平移、畸变，按标注 K 针孔近似')
    fig.suptitle('真实帧 Camera-to-Ego 矩阵与内参推算；角覆盖不是近距离拼接连续性证明',fontsize=13)
    fig.savefig(path,dpi=160);plt.close(fig)


def overview(path, out, frames):
    fig,axes=plt.subplots(len(frames),3,figsize=(19,2.1*len(frames)),constrained_layout=True)
    for row,frame in enumerate(frames):
        folder=out/frame['artifact_directory']
        for col,(filename,title) in enumerate([
                ('current_input.png','当前固定像素裁剪'),
                ('diagnostic_preresize_input.png','预缩放 + 官方裁剪（诊断）'),
                ('diagnostic_full_view_letterbox.png','保留完整视场 + 等比例缩放/填充（诊断）')]):
            with Image.open(folder/filename) as im:a=np.array(im)
            axes[row,col].imshow(a);axes[row,col].axis('off')
            axes[row,col].set_title(f"{frame['case']} · {frame['token'].rsplit('::',1)[1]}帧\n{title}",fontsize=11)
    fig.suptitle('真实 Pilot 图像对照：全部 1024×256，候选只用于检查、未接入训练\n图像拼接不代表连续几何；黑边是候选等比例缩放填充',fontsize=15)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/camera_alignment')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    torch.set_num_threads(4)
    config_path=ROOT/'tools/training/diffusiondrive_v1/pilot_original.json'
    config,_=fit_configuration(config_path);model=model_configuration(config);native=native_configuration(config)
    manifest_path=ROOT/config['prepared_manifest'];manifest=json.loads(manifest_path.read_text())
    rows=[next(r for values in manifest['accepted'].values() for r in values
               if marker in r['token'] and r['token'].endswith('::80')) for marker,_ in CASES]
    protected={str(config_path):file_digest(config_path),str(manifest_path):file_digest(manifest_path)}
    for p in (ROOT/'tools/b2d_adapter').rglob('*.py'):protected[str(p)]=file_digest(p)
    for repo in ('DiffusionDrive','DiffusionDriveV2'):
        for p in (ROOT/'repositories'/repo).rglob('*.py'):protected[str(p)]=file_digest(p)
    for row in rows:
        protected[row['tensor_path']]=file_digest(row['tensor_path'])
        for source in row['identity']['sources']:protected[source['path']]=source['sha256']
    atomic_json(out/'protected_before.json',protected)
    official=official_v1_module('transfuser_features');original_camera=official.TransfuserFeatureBuilder._get_camera_feature
    original_resize=cv2.resize
    report={'schema':'v1-camera-alignment-audit-v1','config':str(config_path),
            'camera_policy':native.camera_policy,'manifest':str(manifest_path),
            'scope':'five real Pilot frames; no training, production policy edits or cache regeneration',
            'navsim_reference':{'source':'analysis/data_interface_analysis/navsim_sample_validation.md:39',
                'evidence':'inherited prior real sample; source mount unavailable in current session',
                'image_wh':[1920,1080],'fx_px':1545.,'pinhole_fov_deg_approx':63.71,
                'distortion_approx':[-.3561,.1725,-.0021,.0005,-.0523],
                'official_stitched_wh':[4096,1024],'aspect':4.},
            'angular_method':'T_ego_camera = world2ego @ inv(world2cam); ray=[1,(u-cx)/fx,0]; rotate to ego; atan2(-y,x); continuous crop edge approximation; no translations/distortion',
            'references':{'carla_projection':'https://carla.readthedocs.io/en/0.9.15/tuto_G_bounding_boxes/',
                          'carla_rgb_sensor':'https://carla.readthedocs.io/en/0.9.15/ref_sensors/'},
            'frames':[]}
    first_features=None
    for row,(_,case) in zip(rows,CASES):
        route,frame=row['token'].rsplit('::',1);reader=FrameReader(RouteIndex(route))
        builder=DiffusionDriveV1FeatureBuilder(reader,native,model,config['lidar_decoder_python'])
        trace=[]
        def camera_call(self,agent_input):
            trace.append({'event':'official_v1_camera_call','class_file':inspect.getfile(type(self)),
                          'camera_wh':[[getattr(agent_input.cameras[-1],key).image.shape[1],
                                        getattr(agent_input.cameras[-1],key).image.shape[0]] for key in ('cam_l0','cam_f0','cam_r0')]})
            return original_camera(self,agent_input)
        def resize_call(a,dsize,*args,**kwargs):
            trace.append({'event':'cv2.resize','source_shape':list(a.shape),'dsize':list(dsize),
                          'explicit_interpolation':kwargs.get('interpolation'),'default':'INTER_LINEAR'})
            return original_resize(a,dsize,*args,**kwargs)
        with patch.object(official.TransfuserFeatureBuilder,'_get_camera_feature',camera_call),patch.object(cv2,'resize',resize_call):
            obs=builder.read_current(reader.index.ref(int(frame)))
            actual=builder.build_features(obs).tensors
        cached,_=load_artifact(row)
        assert all(torch.equal(actual[key],cached[key]) for key in actual),row['token']
        if first_features is None:first_features=actual
        im=[obs.raw.cameras[name] for name in NAMES];sliced=crops(im);stitched=np.concatenate(sliced,axis=1)
        final=cv2.resize(stitched,(1024,256));expected=official.transforms.ToTensor()(final)
        assert torch.equal(expected,actual['camera_feature'])
        assert np.array_equal(final,cv2.resize(stitched,(1024,256),interpolation=cv2.INTER_LINEAR))
        candidate_images=[cv2.resize(a,(1920,1080),interpolation=cv2.INTER_LINEAR) for a in im]
        candidate_stitched=np.concatenate(crops(candidate_images),axis=1)
        candidate=cv2.resize(candidate_stitched,(1024,256),interpolation=cv2.INTER_LINEAR)
        # Diagnostic full-view letterbox reveals the content discarded by cropping.
        full=np.concatenate(im,axis=1);scale=min(1024/full.shape[1],256/full.shape[0])
        resized=cv2.resize(full,(round(full.shape[1]*scale),round(full.shape[0]*scale)),interpolation=cv2.INTER_LINEAR)
        letterbox=np.zeros((256,1024,3),dtype=np.uint8)
        y=(256-resized.shape[0])//2;x=(1024-resized.shape[1])//2
        letterbox[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        tag=route.split('/')[-1].split('_Weather')[0]+f'_frame{int(frame):05d}'
        folder=out/tag;folder.mkdir(exist_ok=True)
        camera_records={}
        for i,name in enumerate(NAMES):
            record=geometry(obs.raw.cameras,obs.raw.sensors[name],obs.raw.ego_box,name)
            source=reader.index.path(obs.raw.ref,name)
            record.update(path=str(source),frame_id=int(frame),source_sha256=file_digest(source),decoded=stats(im[i]))
            bgr=cv2.imread(str(source),cv2.IMREAD_COLOR)
            assert bgr is not None and bgr.shape==im[i].shape
            record['pillow_rgb_vs_opencv_bgr_to_rgb_max_abs_error']=int(np.abs(im[i].astype(int)-bgr[:,:,::-1].astype(int)).max())
            camera_records[name]=record
            save_image(folder/(name+'_original.png'),im[i]);save_image(folder/(name+'_current_crop.png'),sliced[i])
        for key,a in [('current_stitched',stitched),('current_input',final),
                      ('diagnostic_preresize_input',candidate),('diagnostic_full_view_letterbox',letterbox)]:
            save_image(folder/(key+'.png'),a)
        np.savez_compressed(folder/'camera_tensor_evidence.npz',
                            current_camera_feature=actual['camera_feature'].numpy(),
                            current_input_rgb=final,diagnostic_preresize_input_rgb=candidate,
                            diagnostic_full_view_letterbox_rgb=letterbox)
        overlap={}
        for key in ('original_angle_interval_target_deg','current_angle_interval_target_deg','diagnostic_preresize_angle_interval_target_deg'):
            overlap[key]=[signed_overlap(camera_records[a][key],camera_records[b][key])
                          for a,b in zip(NAMES[:-1],NAMES[1:])]
        assert camera_records[NAMES[0]]['relative_target_bearing_deg']>45
        assert abs(camera_records[NAMES[1]]['relative_target_bearing_deg'])<1
        assert camera_records[NAMES[2]]['relative_target_bearing_deg']<-45
        _,cached_targets=load_artifact(row)
        item={'token':row['token'],'case':case,'trace':trace,'cameras':camera_records,
              'weather':obs.raw.annotation.get('weather'),
              'scene_evidence':{'future_trajectory_max_abs_relative_yaw_deg':float(cached_targets['trajectory'][:,2].abs().max()*180/np.pi),
                                'valid_vehicle_targets':int(cached_targets['agent_labels'].sum()),
                                'scope':'context only; future trajectory not used to construct camera input'},
              'same_frame_binding':{'frame_id':int(frame),'indexed_image_stems':[Path(camera_records[n]['path']).stem for n in NAMES],
                                    'annotation':str(reader.index.path(obs.raw.ref,'anno')),
                                    'scope':'same indexed frame/annotation confirmed; independent capture timestamps unavailable'},
              'geometry':{'crop_wh':[[a.shape[1],a.shape[0]] for a in sliced],
                          'stitched_wh':[stitched.shape[1],stitched.shape[0]],
                          'stitched_aspect':stitched.shape[1]/stitched.shape[0],'output_wh':[1024,256],
                          'scale_x':1024/stitched.shape[1],'scale_y':256/stitched.shape[0],
                          'relative_horizontal_stretch':(1024/stitched.shape[1])/(256/stitched.shape[0]),
                          'adjacent_signed_overlap_deg':overlap,
                          'diagnostic_preresize_stitched_wh':[candidate_stitched.shape[1],candidate_stitched.shape[0]],
                          'diagnostic_preresize_two_resize_passes':True,
                          'full_view_letterbox_content_wh':[resized.shape[1],resized.shape[0]]},
              'tensor':{},
              'all_three_features_equal_pilot_cache':True,'manual_camera_equal_official':True,
              'artifact_directory':str(folder.relative_to(out))}
        item['tensor']=dict(stats(actual['camera_feature'].permute(1,2,0).numpy()),
                            training_shape=list(actual['camera_feature'].shape),cached_camera_feature_equal=True)
        atomic_json(folder/'frame_audit.json',item)
        montage(folder/'pipeline_comparison.png',row['token'],case,im,sliced,stitched,final,candidate)
        report['frames'].append(item)
        print(json.dumps({'token':row['token'],'cache_equal':True,'tensor_stats':item['tensor'],
                          'current_overlap_deg':overlap['current_angle_interval_target_deg']},ensure_ascii=False),flush=True)
    coverage_plot(out/'horizontal_coverage.png',report['frames'][0]['cameras'])
    overview(out/'overview_current_vs_diagnostic.png',out,report['frames'])
    # Observe the actual network's first image convolution, stopping BEFORE any
    # convolution/trajectory rollout. No predictions or training claims made.
    agent=official_v1_module('transfuser_agent').TransfuserAgent(model,config['learning_rate'],checkpoint_path=None)
    agent.eval();captured={}
    class StopAtImageConv(Exception):pass
    def hook(module,args):
        captured['tensor']=args[0].detach().cpu().clone()
        raise StopAtImageConv()
    first_conv=agent._transfuser_model._backbone.image_encoder.conv1
    handle=first_conv.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            agent.forward({key:value.unsqueeze(0) for key,value in first_features.items()})
    except StopAtImageConv:
        pass
    finally:
        handle.remove()
    assert torch.equal(captured['tensor'],first_features['camera_feature'].unsqueeze(0))
    report['numeric_pipeline']={'decode':'Pillow Image.open: RGB uint8; no BGR conversion in actual adapter',
        'resize':'cv2.resize default INTER_LINEAR on RGB uint8; no color conversion',
        'tensor':'torchvision ToTensor: HWC->CHW, float32 /255 exactly once',
        'imagenet_normalize_in_adapter':False,'imagenet_normalize_before_backbone_first_conv':False,
        'actual_first_image_conv_input_equal_camera_feature':True,
        'hook_scope':'eval, no_grad; stopped at conv1 pre-hook, no model output, optimizer steps=0',
        'pretrained_cfg_mean_std':'timm initialization metadata, not automatically applied to forward tensors',
        'libraries':{'torch':torch.__version__,'opencv':cv2.__version__, 'pillow':Image.__version__},
        'to_tensor_source':inspect.getfile(official.transforms.ToTensor)}
    differences=[path for path,digest in protected.items() if file_digest(path)!=digest]
    assert not differences,differences
    report.update(status='audit_completed_geometry_alignment_not_certified',protected_sha_changes=differences,
                  production_code_changes=False,existing_cache_changes=False,training_steps=0)
    atomic_json(out/'results.json',report)
    print(json.dumps({'status':report['status'],'frames':len(report['frames']),
                      'first_conv_input_equal':True,'protected_sha_changes':differences},ensure_ascii=False))


if __name__=='__main__':main()
