"""A few real recovery comparisons; refused maps are never drawn as background."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def main():
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from tools.b2d_adapter import RouteIndex,AdapterError
    from tools.training.diffusiondrive_v1.full_recovery_rules import MemoReader,RecoveryProvider,RecoveryBuilder
    from tools.training.diffusiondrive_v1.runtime import model_configuration,native_configuration,bev_provider
    from tools.training.diffusiondrive_v1.target_retry_audit import readonly
    from tools.training.diffusiondrive_v1.features_cache import durable_json
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text());model,native=model_configuration(cfg),native_configuration(cfg)
    state=ROOT/'outputs/diffusiondrive_v1/targets_full_recovery_20261017_r2';out=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/full_target_recovery'
    torch.set_num_threads(1);choices=[]
    with readonly(state/'recovery.sqlite') as db:
        for sql,params,label in [
            ("original_code='invalid_map_polygon' AND profile='projected'",(), 'map_projected'),
            ("original_code='invalid_map_polygon' AND profile='compatible'",(), 'map_roi_exclusion'),
            ("original_code='map_route_mismatch' AND profile='compatible'",(), 'parking'),
            ("route LIKE 'AccidentTwoWays_Town13_Route30356_%' AND frame=50",(), 'police_projected')]:
            r=db.execute('SELECT * FROM tasks WHERE '+sql+' ORDER BY route,frame LIMIT 1',params).fetchone()
            if r:choices.append((dict(r),label))
    result=[];colors=['#eeeeee','#666666','#66c2a5','#ffcc44','#aa77bb','#4477dd','#ee5566']
    for row,label in choices:
        ix=RouteIndex.from_dataset_root(cfg['dataset_root'],row['route']);reader=MemoReader(ix);ref=ix.ref(row['frame']);raw=reader.read(ref)
        profile=row['profile'] or 'projected';provider=RecoveryProvider(cfg,profile)
        targets=RecoveryBuilder(reader,native,model,provider,policy=profile).compute_targets(ref)
        old=None;old_error=None
        try:old=bev_provider(cfg).build_targets(ref,reader,native,model).tensors['bev_semantic_map'].numpy()
        except AdapterError as exc:old_error={'code':exc.code,'error':str(exc)}
        fig,axes=plt.subplots(2,2,figsize=(13,11),constrained_layout=True)
        with Image.open(ix.path(ref,'CAM_FRONT')) as image:axes[0,0].imshow(image)
        axes[0,0].set_title('Same-frame real CAM_FRONT');axes[0,0].axis('off')
        def bev(ax,values,title):
            ax.imshow(values.T,origin='lower',extent=(0,32,-32,32),cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,7.5),7),interpolation='nearest')
            ax.set_xlim(0,32);ax.set_ylim(-32,32);ax.set_xlabel('forward x (m)');ax.set_ylabel('left y (m)');ax.set_title(title)
        new=targets['bev_semantic_map'].numpy();bev(axes[0,1],new,'New validated '+profile+' BEV')
        trajectory=targets['trajectory'].numpy();axes[0,1].plot(trajectory[:,0],trajectory[:,1],'w.-')
        if old is not None:bev(axes[1,0],old,'Old planar BEV diagnostic (not a complete accepted Target)')
        else:
            axes[1,0].text(.05,.55,'Original map refused:\n'+old_error['code']+'\nNo synthetic old label is displayed.',transform=axes[1,0].transAxes,fontsize=12)
            axes[1,0].axis('off')
        mapped,_,_=provider.get_map(ref);roi=mapped.roi(raw,native.geometry);ax=axes[1,1]
        detail=None;diag=json.loads(row['diagnosis']);map_errors=[e for e in diag['bev_errors'] if e['code']=='invalid_map_polygon']
        if map_errors:
            target=map_errors[0]['detail'];origin=np.array([raw.actor_pose.x,-raw.actor_pose.y])
            for rec in mapped.compile_road(mapped.roads[target['road']]):
                if rec.get('lane_id')!=target['lane']:continue
                near=(np.linalg.norm(rec['center']-origin,axis=1)<80)&(np.abs(rec['z']-float(raw.ego_box['location'][2]))<3)&(rec['width']>1e-4)
                changes=np.flatnonzero(np.r_[False,near,False][1:]!=np.r_[False,near,False][:-1])
                for a,b in changes.reshape(-1,2):
                    if b-a<2:continue
                    from shapely.geometry import Polygon,MultiPoint
                    coords=np.concatenate((rec['inner'][a:b],rec['outer'][a:b][::-1]));polygon=Polygon(coords)
                    if polygon.is_valid:continue
                    ax.plot(*np.r_[coords,coords[:1]].T,'r-',lw=1,label='invalid source boundary')
                    try:
                        pieces,proof=mapped.strip_geometry(rec,a,b)
                        for j,p in enumerate(pieces):ax.plot(*p.exterior.xy,'g-',lw=.8,label='source triangle surface' if j==0 else None)
                    except AdapterError:proof={}
                    ax.plot(*roi.exterior.xy,'b--',label='BEV ROI + 0.5m')
                    detail={'source_invalid_signed_ring_area_m2':float(polygon.area),'source_boundary_world_bounds':list(polygon.bounds),
                        'inflated_roi_bounds':list(roi.bounds),'complete_boundary_hull_outside_roi':not MultiPoint(coords).convex_hull.intersects(roi),**proof}
                    break
                if detail:break
            ax.set_title('Original boundary vs source-mesh footprint');ax.set_aspect('equal');ax.legend(fontsize=8)
        else:
            from tools.b2d_adapter.v1_geometry import actor_local_hull
            for box in raw.annotation['bounding_boxes']:
                if box.get('class')=='vehicle' and np.max(np.abs(box['rotation'][:2]))>5:
                    hull,_=actor_local_hull(box,raw,native.geometry);ax.plot(*hull.exterior.xy,'r-',label='true 3D XY hull')
            for v in targets['agent_states'][targets['agent_labels']].numpy():
                x,y,h,l,w=v;c,s=np.cos(h),np.sin(h);p=np.array([[-l/2,-w/2],[l/2,-w/2],[l/2,w/2],[-l/2,w/2],[-l/2,-w/2]])@np.array([[c,s],[-s,c]])+[x,y]
                ax.plot(p[:,0],p[:,1],'b-',lw=.6)
            ax.set_xlim(0,40);ax.set_ylim(-20,25);ax.set_aspect('equal');ax.set_title('New detection enclosure / true 3D footprint')
        fig.suptitle(row['route'].split('_Weather')[0]+' frame '+str(row['frame']),fontsize=11)
        fig.savefig(out/(label+'.png'),dpi=130);plt.close(fig)
        np.savez_compressed(out/(label+'.npz'),bev=new.astype(np.uint8),trajectory=trajectory,agent_states=targets['agent_states'].numpy(),agent_labels=targets['agent_labels'].numpy())
        result.append({'sample_id':row['sample_id'],'route':row['route'],'frame':row['frame'],'image':label+'.png','profile':profile,
            'old_bev_error':old_error,'old_new_pixel_difference':int(np.count_nonzero(old!=new)) if old is not None else None,
            'new_bev_class_pixels':np.bincount(new.ravel(),minlength=7).tolist(),'geometry_comparison':detail})
    durable_json(out/'visual_checks.json',result)


if __name__=='__main__':main()
