"""Export two same-frame real sensor illustrations using existing V1 renderers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')


def main():
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tools.training.diffusiondrive_v1.runtime import fit_configuration,model_configuration
    from tools.training.diffusiondrive_v1.paired_dataset import paired_datasets
    from tools.b2d_adapter.v1_builders import official_v1_module
    from tools.validation.b2d_adapter.audit_v1_camera_alignment import save_image
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--route-token',default='BlockedIntersection_Town04_Route27419')
    p.add_argument('--frame',type=int,default=80)
    p.add_argument('--existing-camera',type=Path,default=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/camera_crop_fix/BlockedIntersection_Town04_Route27419_frame00080/new_stitched.png')
    args=p.parse_args();torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError('Use a new report output directory')
    cfg,_=fit_configuration(args.config)
    cfg.update(cache_read_mode='cache-only',mount_check_mode='procfs',cache_file_read_mode='verified-single-pass')
    mc=model_configuration(cfg);data,identity=paired_datasets(cfg['paired_manifest'],cfg,mc)
    matches=[(split,ds,i,row) for split,ds in data.items() for i,row in enumerate(ds.rows)
             if args.route_token in row['route'] and row['frame']==args.frame]
    if len(matches)!=1:raise ValueError('Expected one exact paired frame: '+str(len(matches)))
    split,ds,i,row=matches[0];features,_=ds[i]
    with Image.open(args.existing_camera) as image:camera=np.array(image.convert('RGB'))
    expected=features['camera_feature'].permute(1,2,0).numpy()
    reconstructed=cv2.resize(camera,(expected.shape[1],expected.shape[0]),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
    if not np.array_equal(reconstructed,expected):raise ValueError('Existing camera image differs from this real cached frame')
    callback=official_v1_module('transfuser_callback')
    empty_boxes=np.empty((0,5),dtype=np.float32);empty_traj=np.empty((0,3),dtype=np.float32)
    lidar=features['lidar_feature'][0].numpy()
    rgb=callback.lidar_map_to_rgb(lidar,empty_boxes,empty_boxes,empty_traj,empty_traj,mc)
    args.output.mkdir(parents=True,exist_ok=False)
    save_image(args.output/'camera.png',camera)
    nearest=Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST
    Image.fromarray(rgb).resize((1024,1024),resample=nearest).save(args.output/'lidar_bev.png',dpi=(300,300))
    meta={'route':row['route'],'frame':row['frame'],'split':split,'sample_id':row['sample_id'],
        'dataset_identity':identity,'feature_sha256':row['feature_sha'],
        'camera':{'file':'camera.png','size_wh':[camera.shape[1],camera.shape[0]],
            'source_existing_png':str(args.existing_camera.resolve()),'policy':cfg['camera_policy'],
            'meaning':'Real left/front/right camera crops stitched before resize; resize exactly reproduces cached (3,256,1024) RGB input.'},
        'lidar':{'file':'lidar_bev.png','native_shape':list(lidar.shape),'display_wh':[1024,1024],
            'renderer':str(Path(callback.__file__).resolve())+':lidar_map_to_rgb',
            'meaning':'Real cached height-filtered point-count histogram; official grayscale, no boxes/trajectory overlay, nearest-neighbor display enlargement.',
            'x_limits_m':[mc.lidar_min_x,mc.lidar_max_x],'y_limits_m':[mc.lidar_min_y,mc.lidar_max_y],
            'axes':'forward x points up; left y points left','nonzero_pixels':int(np.count_nonzero(lidar)),
            'value_range':[float(lidar.min()),float(lidar.max())]},
        'images_sha256':{name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ['camera.png','lidar_bev.png']}}
    (args.output/'source.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    for ds in data.values():ds.close()
    text=f'''# 报告用 Camera / LiDAR 图像

同一真实样本：`{row['route']}`，frame **{row['frame']}**。

- [camera.png](camera.png)：三路前视 RGB 按左前、正前、右前排列，采用现有 B2D 裁剪；3416×854。复用已有 Camera 审计图，并确认缩放后与本帧真实训练缓存逐位一致。
- [lidar_bev.png](lidar_bev.png)：同帧 LiDAR 点云计数 BEV；复用官方 `transfuser_callback.py:lidar_map_to_rgb`，去掉框/轨迹叠加。原栅格 256×256，最近邻放大为 1024×1024；前方朝上、左侧朝左，范围 x/y 均 {mc.lidar_min_x:g} 至 {mc.lidar_max_x:g} m。

可用图注：

> Bench2Drive 真实样本的三路前视相机拼接图，展示 Bench2Drive Camera Adapter 裁剪与拼接后的视觉输入。

> 同一帧 LiDAR 的鸟瞰计数栅格。灰度表示经过高度筛选与计数裁剪的点云密度，深色代表更高计数。

数值来源、sample_id、缓存 SHA、图片 SHA 见 [source.json](source.json)。PNG 仅为展示，训练张量未改动。

复现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/e2e_v1_mpl /home/aethelind/miniconda3/envs/navsim_gpu/bin/python -m tools.validation.b2d_adapter.render_sensor_report --config tools/training/diffusiondrive_v1/baseline_original_7class_bs8_bf16_20261020.json --output /tmp/v1_sensor_report_reproduction
```
'''
    (args.output/'README.md').write_text(text)
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
