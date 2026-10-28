"""Six finite M5 sites, numerical caches/plots and unchanged M2/M4 contracts."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')
os.environ.setdefault('HF_HUB_OFFLINE','1')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    from tools.b2d_adapter import FrameReader,AdapterError
    from tools.b2d_adapter.coordinate_transform import ActorPose,actor_to_reference
    from tools.b2d_adapter.route_index import FrameRef
    from tools.b2d_adapter.v1_bev import OpenDriveLocalMap,reference_line
    from tools.b2d_adapter.v1_builders import v1_native_config,official_v1_module
    from tools.b2d_adapter.static_layers import MapStaticLayers,ROADMARK_TYPES,load_static_query
    from tools.b2d_adapter.static_layers.opendrive import read_npz,digest
    from tools.validation.b2d_adapter.validate_v1_m4 import AnnotationIndex
    from tools.training.diffusiondrive_v1.runtime import model_configuration
    from tools.training.diffusiondrive_v1.prepared import prepared_datasets,producer_identity,validate_tensors,atomic_json

    out=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/m5';out.mkdir(exist_ok=True)
    cfg_path=ROOT/'tools/training/diffusiondrive_v1/m2_subset.json';config=json.loads(cfg_path.read_text())
    manifest_path=ROOT/config['prepared_manifest'];manifest=json.loads(manifest_path.read_text());rows=[r for rs in manifest['accepted'].values() for r in rs]
    m4_config_path=ROOT/'tools/training/diffusiondrive_v1/m4_subset.json';m4_config=json.loads(m4_config_path.read_text());m4_manifest_path=ROOT/m4_config['prepared_manifest'];m4_manifest=json.loads(m4_manifest_path.read_text())
    protected={p:h for p,h in json.loads((out.parent/'m4/results.json').read_text())['protected_files_unchanged'].items()}
    for p in [cfg_path,m4_config_path,m4_manifest_path,ROOT/'tools/b2d_adapter/v1_bev.py',ROOT/'tools/b2d_adapter/v1_geometry.py',ROOT/'tools/b2d_adapter/v1_label_rules.py',ROOT/'tools/training/diffusiondrive_v1/fit.py',ROOT/'tools/training/diffusiondrive_v1/prepared.py',ROOT/'tools/training/diffusiondrive_v1/runtime.py',ROOT/'tools/training/diffusiondrive_v1/prepare.py',ROOT/'tools/training/diffusiondrive_v1/m4_original_compatibility.json']:
        protected[str(p)]=sha(p)
    for rs in m4_manifest['accepted'].values():
        for row in rs:
            p=Path(row['tensor_path']);protected[str(p)]=sha(p)
    for cache_root in (ROOT/config['bev_cache_dir'],ROOT/m4_config['bev_cache_dir']):
        for p in cache_root.rglob('*.npz'):protected[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in protected.items())
    model=model_configuration(config);before_producer=digest(producer_identity(model))
    maps={};layers={}; reader=FrameReader(AnnotationIndex());geom=v1_native_config().geometry

    def load_town(town):
        if town not in maps:
            spec=config['map_paths'][town];maps[town]=OpenDriveLocalMap(spec['path'],town)
            layers[town]=MapStaticLayers(maps[town],spec['carla_version'],ROOT/'outputs/diffusiondrive_v1/m5_static_map_cache',expected_sha256=spec['sha256'])
        return maps[town],layers[town]

    def recorded(town,scenario,frame):
        row=next(row for row in rows if f'/{scenario}_{town}_' in row['token'] and row['token'].endswith(f'::{frame}'))
        route,frame=row['token'].rsplit('::',1);raw=reader.read(FrameRef(Path(route),int(frame)))
        with np.load(ROOT/config['bev_cache_dir']/(row['provenance']['targets']['bev']['cache_key']+'.npz'),allow_pickle=False) as archive:labels=archive['labels'].copy()
        _,layer=load_town(town);result=layer.query_from_raw(raw,geom)
        return result,labels,{'source':'recorded B2D annotation, existing M2 labels','token':row['token'],'ego_road_id':str(raw.ego_box['road_id']),'ego_lane_id':raw.ego_box['lane_id']}

    def map_site(town,rid,lid,s,section=0):
        m,layer=load_town(town);arrays,metadata=layer.section(rid,section)
        lane=next(l for l in metadata['lanes'] if l['lane_id']==lid)
        if lane['status']!='supported':raise AdapterError('unsupported_validation_site',lane['reason'])
        xyz=arrays[lane['center_array']];point=np.array([np.interp(s,arrays['s'],xyz[:,i]) for i in range(3)])
        heading=float(reference_line(m.roads[rid],np.array([s]))[0,2]);direction=next((d['travelDir'] for d in reversed(lane['direction_records']) if d['sOffset']<=s-lane['section_s']),None)
        if direction=='backward':heading+=math.pi
        pose=(point[0],point[1],heading);result=layer.query(pose,float(point[2]))
        return result,None,{'source':'real OpenDRIVE geometry; constructed rear-reference query pose','road_id':rid,'lane_id':lid,'section_index':section,'s_m':s,'orientation_source':direction or 'geometric increasing s',
                            'lane_width_range_m':[float(arrays[lane['width_array']].min()),float(arrays[lane['width_array']].max())]}

    cases=[('straight_double_solid','Town03',lambda:map_site('Town03','17',-1,5)),
           ('recorded_curve','Town03',lambda:recorded('Town03','HazardAtSideLane',80)),
           ('recorded_junction','Town04',lambda:recorded('Town04','BlockedIntersection',80)),
           ('roadmark_transition','Town04',lambda:map_site('Town04','2',1,2)),
           ('crosswalk','Town05',lambda:map_site('Town05','10',-1,0)),
           ('curve_variable_width','Town15',lambda:map_site('Town15','3',1,1))]
    results=[];plot_cases=[];known_rejects=[]
    for name,town,build in cases:
        result,original,source=build();m,layer=load_town(town)
        assert all(v.dtype==np.uint8 and v.shape[-2:]==(128,256) for v in result['masks'].values())
        assert set(np.unique(result['masks']['roadmark_type_mask'])) <= {0,255,*ROADMARK_TYPES.values()}
        assert result['masks']['lane_boundary_mask'].sum()>0
        path=layer.save_query(result,out/'local_masks')
        archive_arrays,_=read_npz(path,{'schema':'b2d-m5-static-map-v1','map_cache_key':layer.cache_key,'query':result['provenance']['rear_pose_map_se2'],'world_z_reference_m':result['provenance']['world_z_reference_m'],'requested_road_ids':None})
        for key,value in result['masks'].items():np.testing.assert_array_equal(value,archive_arrays[key])
        standalone=load_static_query(path,expected_map_sha256=m.sha256,expected_cache_key=layer.cache_key)
        for key,value in result['masks'].items():np.testing.assert_array_equal(value,standalone['masks'][key])
        # Fresh module instance consumes map-level section/object caches, no labels/actors/sensors.
        fresh=MapStaticLayers(m,config['map_paths'][town]['carla_version'],ROOT/'outputs/diffusiondrive_v1/m5_static_map_cache')
        again=fresh.query(tuple(result['provenance']['rear_pose_map_se2']),result['provenance']['world_z_reference_m'])
        for key,value in result['masks'].items():np.testing.assert_array_equal(value,again['masks'][key])
        record={'name':name,'town':town,'source':source,'numerical_path':str(path),'map_cache_key':layer.cache_key,
                'mask_nonzero_pixels':{k:int(np.count_nonzero(v)) for k,v in result['masks'].items()},
                'roadmark_types':sorted({r['type'] for r in result['provenance']['roadmarks'] if r['type']}),
                'crosswalk_object_ids':[r['object_id'] for r in result['provenance']['crosswalks']],
                'source_index':result['provenance'],'cache_roundtrip':True,'fresh_cache_stats':fresh.cache_stats}
        results.append(record);plot_cases.append((record,result,original));print(name,town,record['mask_nonzero_pixels'],flush=True)
    assert any(r['mask_nonzero_pixels']['crosswalk_mask']>0 for r in results)
    assert {typ for r in results for typ in r['roadmark_types']} >= set(ROADMARK_TYPES)
    transition=next(r for r in results if r['name']=='roadmark_transition')
    transition_records=[r for r in transition['source_index']['roadmarks'] if r['road_id']=='2' and r['lane_id']==1]
    assert {'none','solid'}<={r['type'] for r in transition_records}
    varying=next(r for r in results if r['name']=='curve_variable_width');assert varying['source']['lane_width_range_m'][1]-varying['source']['lane_width_range_m'][0]>.2

    # Keep the known folded coordinates rejected, and record finite newly encountered limitations.
    for town,rid,lid,section in [('Town01','307',1,0),('Town02','329',1,0),('Town15','63',-1,2)]:
        _,layer=load_town(town);a,meta=layer.section(rid,section);lane=next(l for l in meta['lanes'] if l['lane_id']==lid)
        assert lane['status']=='unsupported'
        known_rejects.append({'town':town,'road_id':rid,'section_index':section,'lane_id':lid,'reason':lane['reason'],'offset_geometry':lane['offset_geometry']})
    try:recorded('Town05','PedestrianCrossing',40)
    except AdapterError as exc:known_rejects.append({'town':'Town05','position':'existing PedestrianCrossing frame40','reason':str(exc)})
    else:known_rejects.append({'town':'Town05','position':'existing PedestrianCrossing frame40','status':'supported after numeric duplicate handling'})

    # Chinese readable figures; exact masks, nearest-neighbor, no image-derived labels.
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),'axes.unicode_minus':False,'font.size':9})
    mark_colors=['#f6f6f6','#bbc1c8','#925c39','#e74c3c','#387ad4','#ac36bd','#e89324','#269d88']
    mark_cmap=ListedColormap(mark_colors+['#141414']);mark_norm=BoundaryNorm(np.arange(-.5,9.5),9)
    legend=[Patch(facecolor=mark_colors[k],label=f'{k} {typ}') for typ,k in ROADMARK_TYPES.items()]+[Patch(facecolor='#141414',label='255 未知/重叠')]
    for record,result,original in plot_cases:
        fig,axes=plt.subplots(1,4,figsize=(19,5.3));masks=result['masks'];provenance=result['provenance']
        base=np.zeros((128,256,3),dtype=float)+.96
        if original is not None:
            base[original==1]=.62;base[original==2]=np.array([.63,.82,.90]);base[original==5]=np.array([.35,.42,.85])
        for ax in axes:
            ax.imshow(base[:,::-1],origin='lower',extent=(32,-32,0,32),interpolation='nearest',aspect='equal')
            ax.set_xlim(32,-32);ax.set_ylim(0,32);ax.set_xlabel('横向 / m（左为正）');ax.set_ylabel('前向 / m')
        center=(original==3).astype(np.uint8) if original is not None else masks['centerline_geometry_mask']
        overlay=np.zeros((128,256,4));overlay[center>0]=[1,.77,0,1]
        axes[0].imshow(overlay[:,::-1],origin='lower',extent=(32,-32,0,32),interpolation='nearest');axes[0].set_title('V1 原版类别 3 中心线' if original is not None else '中心线几何（独立查询）')
        byid={b['id']:b for b in provenance['boundaries']};left_ids={l['left_boundary_id'] for l in provenance['lanes'] if l['lane_type'] in ('driving','bidirectional')};right_ids={l['right_boundary_id'] for l in provenance['lanes'] if l['lane_type'] in ('driving','bidirectional')}
        for ids,color in [(left_ids-right_ids,'#087dbe'),(right_ids-left_ids,'#eb6b26'),(left_ids&right_ids,'#8e5ea2')]:
            lines=[np.array(line)[:,::-1] for bid in ids if bid in byid for line in byid[bid]['local_lines']]
            if lines:axes[1].add_collection(LineCollection(lines,colors=color,linewidths=1.3))
        axes[1].set_title('左边界蓝 / 右边界橙 / 共享紫\n左右按参考 s；交通方向另附属性')
        types=masks['roadmark_type_mask'].copy();display=types.copy();display[types==255]=8
        axes[2].imshow(display[:,::-1],origin='lower',extent=(32,-32,0,32),interpolation='nearest',cmap=mark_cmap,norm=mark_norm);axes[2].set_title('RoadMark 区间类型\n不是虚线逐段油漆占用')
        overlay=np.zeros((128,256,4));overlay[masks['lane_boundary_mask']>0]=[.3,.4,.6,.4];overlay[masks['crosswalk_mask']>0]=[.1,.75,.3,1]
        axes[3].imshow(overlay[:,::-1],origin='lower',extent=(32,-32,0,32),interpolation='nearest');axes[3].set_title(f"独立 Crosswalk（绿）\n{int(masks['crosswalk_mask'].sum())} 像素")
        fig.suptitle(f"{record['town']} · {record['name']} · {record['source']['source']}",fontsize=12,y=.96)
        fig.legend(handles=legend,loc='lower center',ncol=8,bbox_to_anchor=(.5,.035),fontsize=8)
        fig.text(.5,.015,'世界几何和数值 Mask 独立保存；V1 七类训练标签未修改。',ha='center',fontsize=9)
        fig.subplots_adjust(left=.04,right=.99,bottom=.22,top=.79,wspace=.25)
        fig.savefig(out/(record['name']+'.png'),dpi=150);plt.close(fig)
    # Width variation is small in this real short arc: include a metric profile, not enlarged false masks.
    m,layer=load_town('Town15');a,meta=layer.section('3');lane=next(l for l in meta['lanes'] if l['lane_id']==1)
    fig,axes=plt.subplots(1,2,figsize=(11,4));center=a[lane['center_array']];by={b['id']:b for b in meta['boundaries']}
    for key,color in [('left_boundary_id','#087dbe'),('right_boundary_id','#eb6b26')]:
        line=a[by[lane[key]]['array']];axes[0].plot(line[:,0],line[:,1],color=color,label=key)
    axes[0].plot(center[:,0],center[:,1],color='#e3bb00',label='centerline');axes[0].set_aspect('equal');axes[0].set_title('Town15 road3 lane1 · 真实世界 XY');axes[0].set_xlabel('世界 X / m');axes[0].set_ylabel('世界 Y / m');axes[0].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1.))
    axes[1].plot(a['s'],a[lane['width_array']],marker='o');axes[1].set_xlabel('OpenDRIVE s / m');axes[1].set_ylabel('车道宽度 / m');axes[1].set_title('真实短弯道宽度变化');fig.tight_layout();fig.savefig(out/'curve_width_profile.png',dpi=160);plt.close(fig)

    old_ds,old_identity=prepared_datasets(manifest_path,config,model)
    m4_ds,m4_identity=prepared_datasets(m4_manifest_path,m4_config,model_configuration(m4_config))
    for datasets in (old_ds,m4_ds):
        for split in ('train','validation'):validate_tensors(*datasets[split][0])
    assert digest(producer_identity(model))==before_producer
    assert all(sha(p)==h for p,h in protected.items())
    counts={}
    for town,layer in layers.items():
        _,meta=layer.crosswalks();counts[town]={'objects':len(meta['crosswalks']),'supported':sum(r['status']=='supported' for r in meta['crosswalks']),
            'unsupported':[r for r in meta['crosswalks'] if r['status']!='supported'],'cache':layer.cache_stats,
            'map_sha256':layer.map.sha256,'map_cache_key':layer.cache_key,'cache_dir':str(layer.cache_dir)}
    report={'schema':'b2d-m5-finite-validation-v1','status':'pass','sites':results,'known_rejections':known_rejects,
            'crosswalk_catalog_scope':counts,'original_producer_sha256_unchanged':before_producer,
            'protected_files_unchanged':protected,'m2_prepared_read':old_identity,'m4_prepared_read':m4_identity,
            'training_executed':False,'whole_dataset_scanned':False,'map_geometry_not_training_tensor_cache':True}
    artifact_index={'schema':'b2d-m5-artifact-index-v1','label_schema':'b2d-m5-static-map-v1','rule_version':'m5-v1',
        'sites':[{'name':r['name'],'town':r['town'],'mask_path':r['numerical_path'],'map_cache_key':r['map_cache_key'],
                  'map_sha256':r['source_index']['map_identity']['map_sha256'],'source':r['source']} for r in results],
        'map_cache_scope':counts,'experiment_design':str(ROOT/'tools/training/diffusiondrive_v1/experiments/m5_abc_design.json'),
        'training_executed':False}
    atomic_json(out/'artifact_index.json',artifact_index)
    atomic_json(out/'results.json',report)
    print(json.dumps({'status':'pass','sites':len(results),'roadmark_types':sorted({v for r in results for v in r['roadmark_types']}),'protected_files':len(protected)},ensure_ascii=False))


if __name__=='__main__':main()
