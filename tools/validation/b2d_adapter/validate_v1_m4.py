"""Bounded M4 checks: only the 48 historical frames and two actual map positions."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')
os.environ.setdefault('HF_HUB_OFFLINE','1')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AnnotationIndex:
    def path(self,ref,modality):
        assert modality=='anno'
        return ref.route/'anno'/('%05d.json.gz'%ref.frame_id)
    def check_ref(self,ref):assert self.path(ref,'anno').exists()


def main():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from nuplan.common.actor_state.state_representation import StateSE2
    from tools.b2d_adapter import FrameReader, RouteIndex, AdapterError
    from tools.b2d_adapter.route_index import FrameRef
    from tools.b2d_adapter.coordinate_transform import ActorPose
    from tools.b2d_adapter.v1_bev import (B2DOpenDriveBEVBuilder,OpenDriveLocalMap,OfficialMapView,
        reference_line,polynomial)
    from tools.b2d_adapter.v1_geometry import flat_guardrail_surface,enriched_labels,actor_local_hull
    from tools.b2d_adapter.v1_builders import (official_v1_module,v1_native_config,
        DiffusionDriveV1Dataset,DiffusionDriveV1TargetBuilder)
    from tools.b2d_adapter.v1_label_rules import label_rules
    from tools.training.diffusiondrive_v1.runtime import model_configuration,bev_provider
    from tools.training.diffusiondrive_v1.prepared import (prepared_datasets,producer_identity,digest,atomic_json,
        validate_tensors,compatible_producer)
    torch.set_num_threads(2)
    out=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/m4';out.mkdir(exist_ok=True)
    config=json.loads((ROOT/'tools/training/diffusiondrive_v1/m2_subset.json').read_text())
    from tools.training.diffusiondrive_v1.runtime import native_configuration
    model=model_configuration(config);native=native_configuration(config);geom=native.geometry
    manifest_path=ROOT/config['prepared_manifest'];manifest=json.loads(manifest_path.read_text())
    rows=[r for split in manifest['accepted'].values() for r in split]
    protected=[manifest_path,ROOT/'tools/b2d_adapter/v1_builders.py']
    protected += [ROOT/'repositories/DiffusionDrive/navsim/agents/diffusiondrive'/name for name in
        ('transfuser_config.py','transfuser_features.py','transfuser_loss.py','transfuser_model_v2.py')]
    protected += [Path(r['tensor_path']) for r in rows]
    protected += [ROOT/config['bev_cache_dir']/(r['provenance']['targets']['bev']['cache_key']+'.npz') for r in rows]
    before={str(p):sha(p) for p in protected}
    baseline=B2DOpenDriveBEVBuilder(config['map_paths']);reader=FrameReader(AnnotationIndex())
    checks=[]
    for row in rows:
        route,frame=row['token'].rsplit('::',1);ref=FrameRef(Path(route),int(frame))
        result=baseline.build_targets(ref,reader,native,model)
        path=ROOT/config['bev_cache_dir']/(row['provenance']['targets']['bev']['cache_key']+'.npz')
        with np.load(path,allow_pickle=False) as archive:prior=archive['labels']
        different=int(np.count_nonzero(prior!=result.tensors['bev_semantic_map'].numpy()))
        assert different==0,row['token']
        checks.append({'token':ref.token,'different_pixels':different})
        print('original pixels verified:',ref.token,flush=True)
    atomic_json(out/'original_pixel_checks.json',{'producer_sha256':digest(producer_identity(model)),'checks':checks})
    enriched= B2DOpenDriveBEVBuilder(config['map_paths'],cache_dir=out/'bev_cache',label_config='v1_semantic_enriched')
    failures=json.loads((manifest_path.parent/'preparation_result.json').read_text())['failures'];failure_checks=[]
    for row in failures:
        route,frame=row['token'].rsplit('::',1);ref=FrameRef(Path(route),int(frame));raw=reader.read(ref)
        record={'token':ref.token}
        try:
            result=enriched.build_targets(ref,reader,native,model)
            tensor=result.tensors['bev_semantic_map'];assert tensor.shape==(128,256) and tensor.dtype==torch.int64
            cached=enriched.build_targets(ref,reader,native,model)
            assert torch.equal(tensor,cached.tensors['bev_semantic_map'])
            record.update(bev='accepted',cache_roundtrip=True,actor_diagnostics=result.provenance['actor_floor_diagnostics'])
            np.savez_compressed(out/('Town07_'+frame+'_labels.npz'),labels=tensor.numpy().astype(np.uint8))
            tilted=next(b for b in raw.annotation['bounding_boxes'] if b['class']=='vehicle' and max(abs(x) for x in b['rotation'][:2])>5)
            hull,vertices=actor_local_hull(tilted,raw,geom)
            np.savez_compressed(out/('Town07_'+frame+'_box_geometry.npz'),world_vertices=vertices,local_hull=np.array(hull.exterior.coords))
            record['tilted_actor']={k:tilted[k] for k in ('id','center','location','extent','rotation')}
            record['projected_hull_area_m2']=hull.area
        except AdapterError as exc:
            record.update(bev='rejected',code=exc.code,reason=str(exc))
            assert exc.code=='invalid_map_polygon'
        try:DiffusionDriveV1TargetBuilder(reader,native,model,enriched).build_agent_targets(raw)
        except AdapterError as exc:record['detection_rejection']=str(exc)
        failure_checks.append(record);print('known failure checked:',ref.token,record['bev'],flush=True)
    # Actual map coordinates, without pretending these positions are recorded B2D frames.
    official=official_v1_module('transfuser_features').TransfuserTargetBuilder(model)
    map_positions=[];plots=[]
    for town,rid,oid in [('Town12','712','30003'),('Town13','121','32194')]:
        local=OpenDriveLocalMap(config['map_paths'][town]['path'],town,rules=label_rules('v1_semantic_enriched').identity())
        road=local.roads[rid];obj=next(o for o in road.findall('objects/object') if o.get('id')==oid)
        s=np.array([float(obj.get('s'))]);refline=reference_line(road,s)[0];section=road.find('lanes/laneSection')
        lane_offset=polynomial(road.findall('lanes/laneOffset'),s)[0]
        boundary=lane_offset
        for lane in sorted(section.findall('right/lane'),key=lambda l:abs(int(l.get('id')))):
            width=polynomial(lane.findall('width'),s-float(section.get('s')),key='sOffset')[0]
            if lane.get('type')=='driving':break
            boundary-=width
        t=boundary-width/2;xy=refline[:2]+t*np.array([-np.sin(refline[2]),np.cos(refline[2])]);z=float(polynomial(road.findall('elevationProfile/elevation'),s)[0])
        raw=SimpleNamespace(actor_pose=ActorPose(xy[0],-xy[1],-refline[2]),
            ego_box={'location':[xy[0],-xy[1],z+.02],'road_id':rid,'lane_id':int(lane.get('id'))})
        lanes,coverage=local.local_lanes(raw);lines,diagnostic=local.local_guardrails(raw,lanes)
        pose=StateSE2(xy[0],xy[1],refline[2]);annotations=SimpleNamespace(boxes=np.empty((0,7)),names=[])
        original=official._compute_bev_semantic_map(annotations,OfficialMapView(lanes),pose).to(torch.int64).numpy()
        enhanced=enriched_labels(official,annotations,OfficialMapView(lanes),pose,[],lines).numpy()
        surface=flat_guardrail_surface(road,obj,reference_line,polynomial)
        pixels=int(np.count_nonzero(enhanced==4));assert pixels>0,(town,diagnostic)
        record={'town':town,'road_id':rid,'object_id':oid,'position_source':'real OpenDRIVE map location, constructed ego pose; no synthetic rail geometry',
            'pose':[pose.x,pose.y,pose.heading],'world_road_z_m':z,'source_surface':surface.tolist(),
            'class4_pixels':pixels,'changed_pixels':int(np.count_nonzero(original!=enhanced)),
            'coverage':coverage,'guardrail':diagnostic}
        map_positions.append(record);plots.append((town,original,enhanced,lines,pose))
        np.savez_compressed(out/(town+'_GuardRail_labels.npz'),original=original.astype(np.uint8),enriched=enhanced.astype(np.uint8),source_surface=surface)
        print('actual GuardRail map position:',town,pixels,flush=True)
    # Compare lossless storage on these exact arrays, using the same archive metadata.
    storage=[]
    for town,original,enhanced,_,_ in plots:
        row={'town':town}
        for dtype in ('uint8','int64'):
            stream=io.BytesIO();np.savez_compressed(stream,labels=enhanced.astype(dtype));row[dtype+'_compressed_bytes']=len(stream.getvalue())
            row[dtype+'_raw_bytes']=enhanced.astype(dtype).nbytes
        storage.append(row)
    cmap=ListedColormap(['#eeeeee','#666666','#89cce0','#ffdf00','#eb5757','#5268f5','#40c97b'])
    fig,axes=plt.subplots(len(plots),3,figsize=(15,7))
    for i,(town,a,b,lines,pose) in enumerate(plots):
        for j,labels in enumerate((a,b)):
            axes[i,j].imshow(labels,cmap=cmap,vmin=0,vmax=6);axes[i,j].set_title(town+(' original' if j==0 else ' enriched (surface=4)'))
        for line in lines:
            line=official._geometry_local_coords(line,pose);xyline=np.array(line.coords)
            axes[i,2].plot(xyline[:,0],xyline[:,1],color='#eb5757')
        axes[i,2].set_xlim(0,32);axes[i,2].set_ylim(-32,32);axes[i,2].set_aspect('equal');axes[i,2].set_title('Source segments in local meters')
    fig.tight_layout();fig.savefig(out/'GuardRail_comparison.png',dpi=140);plt.close(fig)
    # Full Dataset integration, numeric dtype/shape and actual cache hit; no training.
    row=next(r for r in rows if 'Town12_' in r['token']);route,frame=row['token'].rsplit('::',1)
    actual_reader=FrameReader(RouteIndex(route));actual_ref=actual_reader.index.ref(int(frame))
    dataset=DiffusionDriveV1Dataset(actual_reader,[actual_ref],native,model,config['lidar_decoder_python'],enriched)
    sample=dataset.build_sample(0);validate_tensors(sample.features,sample.targets)
    repeat=dataset.build_sample(0);assert torch.equal(sample.targets['bev_semantic_map'],repeat.targets['bev_semantic_map'])
    # Existing prepared M2 manifest remains readable only under original rules.
    bridge={'schema':'m4-reviewed-original-producer-bridge-v1',
        'historical_producer_sha256':digest(manifest['producer']),
        'reviewed_current_producer_sha256':digest(producer_identity(model)),
        'evidence':str(out/'results.json'),
        'scope':'M2 producer only; 42 original pixel comparisons, immutable numeric artifacts; never enriched'}
    atomic_json(ROOT/'tools/training/diffusiondrive_v1/m4_original_compatibility.json',bridge)
    prepared,identity=prepared_datasets(manifest_path,config,model)
    for split in ('train','validation'):validate_tensors(*prepared[split][0])
    try:prepared_datasets(manifest_path,dict(config,label_rules='v1_semantic_enriched'),model)
    except AdapterError as exc:assert exc.code=='prepared_contract_mismatch'
    else:raise AssertionError('enriched rules accepted original manifest')
    changed_producer=json.loads(json.dumps(producer_identity(model)));changed_producer['libraries']['numpy']='future'
    assert not compatible_producer(manifest['producer'],changed_producer,label_rules())
    after={str(p):sha(p) for p in protected};assert before==after
    result={'status':'pass','whole_dataset_scanned':False,'training_executed':False,
        'original_pixel_comparisons':checks,'known_failures':failure_checks,'map_positions':map_positions,
        'lossless_storage':storage,'dataset_integration':{'token':row['token'],'cache_repeat_identical':True,
            'keys':list(sample.targets),'bev_dtype':str(sample.targets['bev_semantic_map'].dtype)},
        'historical_prepared_read':identity,'enriched_rejects_historical_manifest':True,
        'protected_files_unchanged':before,'enriched_cache_stats':enriched.cache_stats}
    atomic_json(out/'results.json',result)
    print(json.dumps({'status':'pass','original_frames':len(checks),'known_failures':len(failure_checks),'guardrail_positions':len(map_positions)},ensure_ascii=False))


if __name__=='__main__':main()
