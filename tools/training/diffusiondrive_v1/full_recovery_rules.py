"""Versioned recovery rules over frozen B2D/V1 builders, not a second service.

Strict trajectory semantics are preserved. Map/actor projected geometry has
an explicit new contract, while source/Feature reference axes stay unchanged.
"""
import copy
import math
import re
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiPoint, box as rectangle
from shapely.ops import unary_union

from tools.b2d_adapter import FrameReader
from tools.b2d_adapter.errors import AdapterError
from tools.b2d_adapter.coordinate_transform import actor_to_reference, wrap_angle
from tools.b2d_adapter.native_builders import NativeBuildResult
from tools.b2d_adapter.v1_bev import (OpenDriveLocalMap, B2DOpenDriveBEVBuilder, polynomial,
    reference_line, OfficialMapView, bev_annotations, bev_actor_name)
from tools.b2d_adapter.v1_builders import DiffusionDriveV1TargetBuilder, BEV_CLASSES, official_v1_module
from tools.b2d_adapter.v1_geometry import actor_local_hull, actor_vertices, enriched_labels, carla_rotation, offset_geometry_diagnostic
from tools.b2d_adapter.navsim_bridge import candidate_vehicle_boxes, is_b2d_vehicle
from tools.training.diffusiondrive_v1.target_recovery import IrrelevantActorRecoveryBuilder, irrelevant_tilt
from tools.training.diffusiondrive_v1.features_cache import StoragePause

PROFILES = ('compatible', 'projected')
GEOMETRY_CONTRACT = {
    'version': 'b2d-v1-projected-geometry-v1',
    'reference': 'frozen yaw-only current rear reference; unchanged cached Features coordinates',
    'trajectory': 'frozen real t+5..t+40; all 41 frames checked; no angle/flag relaxation',
    'actor_bev': 'convex hull of true eight 3D box vertices projected into frozen XY',
    'detection': 'annotation-yaw-aligned minimal enclosing rectangle of the same projected vertices',
    'actor_scope': 'only vehicles exceeding frozen 5deg gate; others retain original targets',
    'map': 'valid original polygons retained; invalid ROI-intersecting ribbons use source boundary triangle strip',
    'map_scope': 'only zero-bank source ribbons without extra lateral surface profiles',
    'ignore_mask': False, 'classes': list(BEV_CLASSES), 'feature_regeneration_required': False,
}


def error(code, detail): return {'code': code, 'detail': detail}


class MemoReader(FrameReader):
    """Bounded annotation memo; changed files pause, never return stale frames."""
    def __init__(self,index):
        super().__init__(index); self.memo=OrderedDict()

    def read(self,ref,cameras=False,lidar=False):
        if cameras or lidar: return super().read(ref,cameras,lidar)
        path=self.index.path(ref,'anno');s=path.stat();finger=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if ref.frame_id in self.memo:
            prior,raw=self.memo[ref.frame_id]
            if prior != finger: raise StoragePause('Annotation changed after memoization: '+str(path))
            self.memo.move_to_end(ref.frame_id);return raw
        raw=super().read(ref);after=path.stat()
        if finger != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise StoragePause('Annotation changed during read')
        self.memo[ref.frame_id]=(finger,raw)
        while len(self.memo)>128:self.memo.popitem(last=False)
        return raw


def trajectory_diagnosis(reader,ref,geometry):
    """Collect all 41-frame issues, not just the first frozen builder exception."""
    failures=[];frames=[];metrics={'max_xyz_speed_mps':0.,'max_xy_speed_mps':0.,'max_yaw_rate_radps':0.,
        'max_pitch_roll_deg':0.,'flag_frames':[],'nonplanar_frames':[],'physical_clock_certified':False}
    for frame in range(ref.frame_id,ref.frame_id+41):
        try:raw=reader.read(reader.index.ref(frame));frames.append(raw)
        except AdapterError as exc:
            failures.append(error(exc.code,{'frame':frame,'error':str(exc)}));continue
        rotation=np.asarray(raw.ego_box['rotation'],float)
        if raw.ego_box.get('type_id')!=geometry.axle_vehicle_type:failures.append(error('reference_vehicle_mismatch',frame))
        tilt=float(np.max(np.abs(rotation[:2])));metrics['max_pitch_roll_deg']=max(metrics['max_pitch_roll_deg'],tilt)
        if tilt>geometry.max_planar_tilt_deg:metrics['nonplanar_frames'].append(frame)
        if raw.state.get('teleport_flag',False):metrics['flag_frames'].append(frame)
        if len(frames)>1 and frames[-2].ref.frame_id==frame-1:
            previous=frames[-2];d=np.asarray(raw.ego_box['location'])-np.asarray(previous.ego_box['location'])
            xy=float(np.linalg.norm(d[:2])/geometry.source_dt_s);xyz=float(np.linalg.norm(d)/geometry.source_dt_s)
            yaw=abs(wrap_angle(raw.actor_pose.yaw-previous.actor_pose.yaw))/geometry.source_dt_s
            metrics['max_xyz_speed_mps']=max(metrics['max_xyz_speed_mps'],xyz)
            metrics['max_xy_speed_mps']=max(metrics['max_xy_speed_mps'],xy)
            metrics['max_yaw_rate_radps']=max(metrics['max_yaw_rate_radps'],yaw)
            if xy>geometry.max_speed_mps or yaw>geometry.max_yaw_rate_radps:
                failures.append(error('teleport_or_discontinuity',{'frame':frame,'xy_speed':xy,'yaw_rate':yaw}))
    if metrics['nonplanar_frames']:failures.append(error('nonplanar_pose',metrics['nonplanar_frames']))
    if metrics['flag_frames']:failures.append(error('teleport_flag',{'frames':metrics['flag_frames'],
        'blocker':'version-matched collector flag/reset semantics unavailable; continuous XY alone is insufficient'}))
    return failures,metrics


def verified_actor(box,raw,geometry):
    """Check independent annotation world corners before accepting new geometry."""
    hull,vertices=actor_local_hull(box,raw,geometry)
    observed=np.asarray(box.get('world_cord'),float)
    if observed.shape!=(8,3) or not np.isfinite(observed).all():
        raise AdapterError('unverified_3d_actor','eight real annotation world_cord vertices required')
    distance=np.linalg.norm(vertices[:,None,:]-observed[None,:,:],axis=-1)
    residual=float(max(distance.min(0).max(),distance.min(1).max()))
    if residual>.01:raise AdapterError('unverified_3d_actor',{'actor':str(box['id']),'corner_residual_m':residual})
    heading=wrap_angle(raw.actor_pose.yaw-math.radians(float(box['rotation'][2])))
    xy=np.array(hull.exterior.coords)[:-1];c,s=math.cos(heading),math.sin(heading)
    basis=np.array([[c,-s],[s,c]])
    aligned=xy@basis;low,high=aligned.min(0),aligned.max(0)
    center=((low+high)/2)@basis.T;size=high-low
    if (size<=0).any():raise AdapterError('degenerate_projected_actor',box['id'])
    state=np.array([*center,heading,*size],dtype=np.float32)
    rect=Polygon((np.array([[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]])@basis.T))
    if not rect.buffer(1e-7).covers(hull):raise AdapterError('projected_actor_enclosure_failed',box['id'])
    return hull,state,{'actor_id':str(box['id']),'world_corner_residual_m':residual,
        'hull_area_m2':float(hull.area),'detection_enclosure_area_m2':float(rect.area),
        'detection_state':state.tolist(),'max_pitch_roll_deg':float(np.max(np.abs(box['rotation'][:2])))}


def detection_diagnosis(raw,native,official):
    failures=[];actors=[];seen=set()
    for box in raw.annotation['bounding_boxes']:
        if not isinstance(box,dict) or not isinstance(box.get('class'),str):failures.append(error('invalid_annotation','class missing'));continue
        if box['class']!='vehicle':continue
        aid=str(box.get('id'))
        if not is_b2d_vehicle(box) or 'id' not in box or aid in seen:failures.append(error('invalid_vehicle_box',{'actor':aid,'reason':'type/id/duplicate'}));continue
        seen.add(aid)
        try:
            rotation=np.asarray(box['rotation'],float)
            if rotation.shape!=(3,) or not np.isfinite(rotation).all():raise AdapterError('invalid_vehicle_box','finite rotation required')
            if np.max(np.abs(rotation[:2]))>native.geometry.max_planar_tilt_deg:
                proof=irrelevant_tilt(box,raw,native.geometry,official)
                rec={'actor_id':aid,'irrelevance':proof}
                try:_,_,projected=verified_actor(box,raw,native.geometry);rec['projected_geometry']=projected
                except AdapterError as exc:rec['projected_error']=error(exc.code,str(exc))
                actors.append(rec);failures.append(error('invalid_vehicle_box',{'actor':aid,'reason':'tilt','safely_irrelevant':proof['safely_irrelevant']}))
        except (KeyError,TypeError,ValueError,AdapterError) as exc:
            failures.append(error(exc.code if isinstance(exc,AdapterError) else 'invalid_vehicle_box',{'actor':aid,'error':str(exc)}))
    # Retain all finite bbox/type validation independently of the tilt precheck.
    try:candidate_vehicle_boxes(raw,native.geometry)
    except AdapterError as exc:failures.append(error(exc.code,str(exc)))
    return failures,actors


class RecoveryMap(OpenDriveLocalMap):
    """Cache exact frozen samples; diagnose all map pieces and prove ROI exclusion.

    No buffer(0), make_valid, invented surfaces or name-only type remapping.
    """
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.compiled={};self.mesh_cache={}

    def compile_road(self,road):
        rid=road.get('id')
        if rid in self.compiled:return self.compiled[rid]
        records=[];sections=road.findall('lanes/laneSection');length=float(road.get('length'))
        for section_idx,section in enumerate(sections):
            start=float(section.get('s'));stop=float(sections[section_idx+1].get('s')) if section_idx+1<len(sections) else length
            if stop<=start:continue
            s=np.linspace(start,stop,max(2,int(math.ceil((stop-start)/self.spacing))+1))
            ref=reference_line(road,s);normal=np.column_stack((-np.sin(ref[:,2]),np.cos(ref[:,2])))
            elevation=polynomial(road.findall('elevationProfile/elevation'),s)
            roll=polynomial(road.findall('lateralProfile/superelevation'),s)
            offset=polynomial(road.findall('lanes/laneOffset'),s)
            for side,sign in [('left',1),('right',-1)]:
                boundary=offset.copy()
                for lane in sorted(section.findall(side+'/lane'),key=lambda l:abs(int(l.get('id')))):
                    widths=lane.findall('width');lane_type=lane.get('type')
                    if not widths:
                        if lane_type=='none':continue
                        records.append({'error':error('unsupported_lane_width',{'road':rid,'lane':lane.get('id')})});break
                    width=polynomial(widths,s-start,key='sOffset')
                    if (width < -1e-5).any() or not np.isfinite(width).all():
                        records.append({'error':error('invalid_lane_width',rid)});break
                    width=np.maximum(width,0);inner=boundary.copy();outer=inner+sign*width;boundary=outer
                    center=(inner+outer)/2
                    records.append({'road':road,'section':section,'road_id':rid,'lane_id':int(lane.get('id')),'type':lane_type,
                        's':s,'inner_offset':inner,'outer_offset':outer,'roll':roll,'width':width,
                        'inner':ref[:,:2]+normal*(inner*np.cos(roll))[:,None],
                        'outer':ref[:,:2]+normal*(outer*np.cos(roll))[:,None],
                        'center':ref[:,:2]+normal*(center*np.cos(roll))[:,None],
                        'z':elevation+center*np.sin(roll),'junction':road.get('junction')!='-1'})
        self.compiled[rid]=records;return records

    @staticmethod
    def roi(raw,geometry,margin=.5):
        rear=actor_to_reference(raw.actor_pose,(geometry.axle_dx_m,geometry.axle_dy_m))
        h=-rear.yaw;c,s=math.cos(h),math.sin(h)
        corners=np.array([[-margin,-32-margin],[32+margin,-32-margin],[32+margin,32+margin],[-margin,32+margin]])
        return Polygon(corners@np.array([[c,s],[-s,c]])+np.array([rear.x,-rear.y]))

    def parking_proof(self,raw,reader):
        if not raw.ref.route.name.startswith('ParkingExit_'):return None
        supports=[]
        for frame in range(raw.ref.frame_id,raw.ref.frame_id+41):
            f=reader.read(reader.index.ref(frame));rid=str(f.ego_box['road_id']);lid=int(f.ego_box['lane_id'])
            road=self.roads.get(rid)
            if road is None:return None
            point=Point(f.actor_pose.x,-f.actor_pose.y);matched=[]
            for rec in self.compile_road(road):
                if 'error' in rec or rec['type'] not in ('parking','driving','bidirectional'):continue
                if rec['type']=='parking' and not (rec['lane_id']*lid>0 and abs(rec['lane_id'])==abs(lid)+1):continue
                if rec['type']!='parking' and rec['lane_id']!=lid:continue
                polygon=Polygon(np.concatenate((rec['inner'],rec['outer'][::-1])))
                if not polygon.is_valid or not polygon.covers(point):continue
                k=int(np.linalg.norm(rec['center']-np.array(point.coords[0]),axis=1).argmin())
                dz=abs(float(f.ego_box['location'][2])-float(rec['z'][k]))
                if dz>float(f.ego_box['extent'][2]):continue  # Per-ego physical half-height, not a global distance relaxation.
                matched.append({'road':rid,'lane':rec['lane_id'],'type':rec['type'],'road_z':float(rec['z'][k]),'ego_z_gap_m':dz})
            if not matched:return None
            supports.append({'frame':frame,'supports':matched})
        if not any(s['type']=='parking' for s in supports[0]['supports']):return None
        return {'rule':'parking-exit-contained-adjacent-lane-same-deck-v1','frames':supports,
            'distance_threshold_relaxed':False,'road_and_lane_identity_checked':True}

    def strip_geometry(self,rec,a,b):
        key=(rec['road_id'],id(rec['section']),rec['lane_id'],a,b)
        if key in self.mesh_cache:return self.mesh_cache[key]
        road=rec['road']
        extra=road.findall('lateralProfile/shape')+road.findall('lateralProfile/crossfall')
        coefficients=np.array([float(r.get(k,0)) for r in extra for k in ('a','b','c','d')])
        # RoadRunner exports many explicit identically-zero shape records. Their
        # presence is not an additional surface; inspect the actual polynomial.
        if np.max(np.abs(rec['roll'][a:b]))>1e-10 or not np.isfinite(coefficients).all() or (np.abs(coefficients)>1e-12).any():
            raise AdapterError('uncertified_map_surface','banked/extra lateral profile triangle strip not verified')
        vertices=np.stack((rec['inner'][a:b],rec['outer'][a:b]),axis=1)
        triangles=[]
        for left,right in zip(vertices[:-1],vertices[1:]):
            for coords in ([left[0],left[1],right[0]],[left[1],right[0],right[1]]):
                triangle=Polygon(coords)
                if triangle.area>1e-12:triangles.append(triangle)
        if not triangles:raise AdapterError('invalid_map_polygon','empty source triangle strip')
        surface=unary_union(triangles)
        pieces=list(surface.geoms) if surface.geom_type=='MultiPolygon' else [surface]
        if not all(isinstance(p,Polygon) and p.is_valid and p.area>0 for p in pieces):raise AdapterError('invalid_map_polygon','source strip union invalid')
        # No new vertex is invented before union; union computes intersections of real triangles.
        detail={'rule':'opendrive-boundary-triangle-strip-v1','triangles':len(triangles),
            'triangle_area_sum_m2':float(sum(t.area for t in triangles)),'union_area_m2':float(surface.area),
            'components':len(pieces),'holes':sum(len(p.interiors) for p in pieces),'source_spacing_m':self.spacing,
            'source_offset':offset_geometry_diagnostic(road,rec['s'][a:b],rec['inner_offset'][a:b],rec['outer_offset'][a:b])}
        # Official map rasterizer only handles exteriors. Preserve holes by returning the
        # original triangles instead of silently filling holes in a union polygon.
        if detail['holes']:pieces=triangles;detail['raster_representation']='individual source triangles preserve holes'
        self.mesh_cache[key]=(pieces,detail);return pieces,detail

    def query(self,raw,geometry,reader,policy='strict'):
        origin=np.array([raw.actor_pose.x,-raw.actor_pose.y]);z=float(raw.ego_box['location'][2])
        expected_road=str(raw.ego_box['road_id']);expected_lane=int(raw.ego_box['lane_id'])
        roi=self.roi(raw,geometry);lanes=[];failures=[];events=[];ego_matches=[];types=set()
        if expected_road not in self.roads:failures.append(error('map_route_mismatch','ego road absent'))
        for road,low,high in self.bounds:
            if ((origin<low-48)|(origin>high+48)).any():continue
            for rec in self.compile_road(road):
                if 'error' in rec:failures.append(rec['error']);continue
                near=(np.linalg.norm(rec['center']-origin,axis=1)<80)&(np.abs(rec['z']-z)<self.height_tolerance)&(rec['width']>1e-4)
                if not near.any():continue
                lane_type=rec['type'];types.add(lane_type)
                if rec['road_id']==expected_road and rec['lane_id']==expected_lane:
                    ego_matches.append(float(np.min(np.linalg.norm(rec['center'][near]-origin,axis=1)-rec['width'][near]/2)))
                allowed=lane_type in ('driving','bidirectional','sidewalk','none','shoulder','parking','biking','median')
                changes=np.flatnonzero(np.r_[False,near,False][1:]!=np.r_[False,near,False][:-1])
                for a,b in changes.reshape(-1,2):
                    if b-a<2:continue
                    coords=np.concatenate((rec['inner'][a:b],rec['outer'][a:b][::-1]))
                    polygon=Polygon(coords);detail={'road':rec['road_id'],'lane':rec['lane_id'],'lane_type':lane_type}
                    if not allowed:
                        if policy!='strict' and not MultiPoint(coords).convex_hull.intersects(roi):
                            events.append(dict(detail,rule='outside-inflated-roi-unknown-lane'));continue
                        failures.append(error('unmapped_lane_type',dict(detail,type=lane_type)));continue
                    if lane_type not in ('driving','bidirectional','sidewalk'):continue
                    if not polygon.is_valid:
                        from shapely.validation import explain_validity
                        detail.update(reason=explain_validity(polygon),offset=offset_geometry_diagnostic(road,rec['s'][a:b],rec['inner_offset'][a:b],rec['outer_offset'][a:b]))
                        if policy!='strict' and not MultiPoint(coords).convex_hull.intersects(roi):
                            events.append(dict(detail,rule='outside-inflated-roi-invalid-polygon'));continue
                        if policy=='projected':
                            try:
                                pieces,proof=self.strip_geometry(rec,a,b);events.append(dict(detail,**proof))
                            except AdapterError as exc:failures.append(error(exc.code,dict(detail,error=str(exc))));continue
                        else:failures.append(error('invalid_map_polygon',detail));continue
                    else:
                        if polygon.area<1e-6:continue
                        pieces=[polygon]
                    for piece in pieces:lanes.append({'polygon':piece,'baseline':LineString(rec['center'][a:b]),'type':lane_type,'junction':rec['junction']})
        if not ego_matches or min(ego_matches)>1.:
            proof=self.parking_proof(raw,reader) if policy!='strict' and ego_matches else None
            if proof:events.append(proof)
            else:failures.append(error('map_route_mismatch',{'ego_road':expected_road,'ego_lane':expected_lane,'lane_edge_distance':ego_matches}))
        if not any(l['type'] in ('driving','bidirectional') for l in lanes):failures.append(error('missing_map_coverage','no local driving polygon'))
        return lanes,{'ego_lane_edge_distance_m':min(ego_matches) if ego_matches else None,'local_lane_types':sorted(types),
            'lane_polygon_count':len(lanes),'height_tolerance_m':self.height_tolerance,'sample_spacing_m':self.spacing,
            'recovery_events':events},failures


class RecoveryProvider(B2DOpenDriveBEVBuilder):
    def __init__(self,config,policy='compatible'):
        paths={k:dict(v,path=str(v['path'])) for k,v in config['map_paths'].items()}
        super().__init__(paths,cache_dir=None,max_maps=1);self.policy=policy

    def get_map(self,ref):
        town=re.search(r'_(Town\d+(?:HD)?)_',ref.route.name).group(1)
        identity=self.map_identity(town)
        if town not in self.maps:
            self.maps[town]=RecoveryMap(identity['path'],town,self.spacing,self.height_tolerance)
            while len(self.maps)>self.max_maps:self.maps.popitem(last=False)
        return self.maps[town],town,identity

    def build_targets(self,current,reader,native_config,model_config):
        from nuplan.common.actor_state.state_representation import StateSE2
        from tools.b2d_adapter.v1_builders import validate_v1_config
        raw=reader.read(current);mapped,town,identity=self.get_map(current)
        lanes,coverage,failures=mapped.query(raw,native_config.geometry,reader,self.policy)
        if failures:raise AdapterError(failures[0]['code'],failures)
        projected=[];actors=[];keep=[]
        for box in raw.annotation['bounding_boxes']:
            name=bev_actor_name(box)
            if self.policy=='projected' and name=='vehicle' and np.max(np.abs(box['rotation'][:2]))>native_config.geometry.max_planar_tilt_deg:
                hull,_,proof=verified_actor(box,raw,native_config.geometry);projected.append((name,hull));actors.append(proof)
            else:keep.append(box)
        filtered=copy.copy(raw);filtered.annotation=dict(raw.annotation,bounding_boxes=keep)
        annotations,ignored=bev_annotations(filtered,native_config.geometry)
        rear=actor_to_reference(raw.actor_pose,(native_config.geometry.axle_dx_m,native_config.geometry.axle_dy_m))
        pose=StateSE2(rear.x,-rear.y,-rear.yaw)
        official=official_v1_module('transfuser_features').TransfuserTargetBuilder(model_config)
        labels=enriched_labels(official,annotations,OfficialMapView(lanes),pose,projected,[])
        return NativeBuildResult({'bev_semantic_map':labels},{
            'current_token':current.token,'contract_sha256':validate_v1_config(native_config,model_config)['contract_sha256'],
            'class_names':BEV_CLASSES,'map_source':str(mapped.path),'map_sha256':mapped.sha256,'map_version':identity['carla_version'],
            'map_town':town,'annotation_source':str(reader.index.path(current,'anno')),'coverage':coverage,
            'class_pixel_counts':__import__('torch').bincount(labels.flatten(),minlength=7).tolist(),
            'ignored_official_non_targets':ignored,'projected_actors':actors,'recovery_policy':self.policy},
            {'real_labels_validated':True,'map_ego_alignment_pass':True,'new_geometry_contract':self.policy=='projected'})


class RecoveryBuilder(IrrelevantActorRecoveryBuilder):
    def __init__(self,*a,policy='compatible',**kw):super().__init__(*a,**kw);self.policy=policy

    def build_agent_targets(self,raw):
        if self.policy!='projected':return super().build_agent_targets(raw)
        # Identity and finite geometry checks are retained before projecting tilt.
        vehicles=[b for b in raw.annotation['bounding_boxes'] if isinstance(b,dict) and b.get('class')=='vehicle']
        if any(not is_b2d_vehicle(b) or 'id' not in b for b in vehicles) or len({str(b['id']) for b in vehicles})!=len(vehicles):
            raise AdapterError('invalid_vehicle_box','type/id/duplicate')
        if raw.ego_box.get('type_id')!=self.config.geometry.axle_vehicle_type:raise AdapterError('reference_vehicle_mismatch',raw.ref.token)
        view=candidate_vehicle_boxes(raw,self.config.geometry);proof=[]
        for i,box in enumerate(vehicles):
            rotation=np.asarray(box['rotation'],float)
            if rotation.shape!=(3,) or not np.isfinite(rotation).all():raise AdapterError('invalid_vehicle_box','finite rotation required')
            if np.max(np.abs(rotation[:2]))>self.config.geometry.max_planar_tilt_deg:
                _,state,evidence=verified_actor(box,raw,self.config.geometry)
                view.boxes[i,[0,1,6,3,4]]=state;proof.append(evidence)
        states,labels=self.official_builder._compute_agent_targets(view)
        return NativeBuildResult({'agent_states':states,'agent_labels':labels},
            dict(view.provenance,current_token=raw.ref.token,projected_actors=proof,new_geometry_contract=GEOMETRY_CONTRACT),
            {'engineering':'pass','new_geometry_contract':True})


def rear6(raw,geometry):
    return np.asarray(raw.ego_box['location'],float)+carla_rotation(raw.ego_box['rotation'])@np.array([geometry.axle_dx_m,geometry.axle_dy_m,0.])


def world_plane6(points,raw,geometry):
    points=np.asarray(points,float);rear=rear6(raw,geometry);yaw=raw.actor_pose.yaw;c,s=math.cos(yaw),math.sin(yaw)
    return (points-rear)@np.array([[c,s,0.],[s,-c,0.],[0.,0.,1.]])


def plane6_world(points,raw,geometry):
    yaw=raw.actor_pose.yaw;c,s=math.cos(yaw),math.sin(yaw)
    return np.asarray(points,float)@np.array([[c,s,0.],[s,-c,0.],[0.,0.,1.]])+rear6(raw,geometry)


def ego6_recipe(reader,ref,geometry):
    """Explicit paired regeneration recipe, never applied to old cached Features."""
    source=[reader.read(reader.index.ref(i)) for i in range(max(0,ref.frame_id-2),ref.frame_id+41)]
    differences=[];inverse=[]
    for raw in source:
        new=rear6(raw,geometry);old=actor_to_reference(raw.actor_pose,(geometry.axle_dx_m,geometry.axle_dy_m))
        differences.append(float(np.linalg.norm(new[:2]-[old.x,old.y])))
        probe=actor_vertices(raw.ego_box);inverse.append(float(np.max(np.abs(plane6_world(world_plane6(probe,raw,geometry),raw,geometry)-probe))))
    return {'version':'b2d-gravity-plane-6dof-rear-v1-proposal','validated_world_inverse_max_m':max(inverse),
        'old_new_rear_xy_max_m':max(differences),'feature_regeneration_required':True,
        'feature_changes':['LiDAR: raw Actor XYZ -> world -> new gravity plane before histogram; cannot recover from histogram',
            'Status: trailing rear6 XY velocity/acceleration and near node bearing at rear6',
            'Camera: unchanged crop, but new paired producer/reference identity'],
        'target_changes':['same real future frame IDs; rear6 future XY in current gravity plane',
            'detection/BEV world geometry projected at rear6'],
        'status':'not eligible with existing frozen Feature reference; no batch regeneration started'}
