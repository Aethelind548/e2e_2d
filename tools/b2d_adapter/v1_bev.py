"""Real OpenDRIVE lanes + B2D actor boxes, rasterized by the official V1 code.

Supported planView primitives are line/arc/spiral (including local Town15). Unsupported
local geometry or annotation classes fail rather than become background. This
is a data-domain map bridge, not a replacement for the model or its BEV loss.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import os
import tempfile
from collections import OrderedDict
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import LineString, Polygon, Point

from .coordinate_transform import actor_to_reference, world_to_ego, wrap_angle
from .errors import AdapterError
from .native_builders import NativeBuildResult
from .navsim_bridge import candidate_vehicle_boxes, is_b2d_vehicle
from .v1_builders import BEV_CLASSES, official_v1_module, validate_v1_config
from .v1_label_rules import label_rules
from .v1_geometry import (actor_local_hull, actor_floor_diagnostic, carla_rotation,
    flat_guardrail_surface, road_support, clip_vertical_segment, enriched_labels, offset_geometry_diagnostic)


def polynomial(records, s, key="s", default=0.):
    """OpenDRIVE piecewise cubic; missing elevation/offset means spec zero."""
    if not records:
        return np.full_like(s, default)
    starts = np.array([float(r.get(key)) for r in records])
    idx = np.searchsorted(starts, s, side="right") - 1
    if (idx < 0).any():
        raise AdapterError("incomplete_opendrive", "polynomial interval has no covering record")
    coefficients = np.array([[float(r.get(k,0)) for k in ("a","b","c","d")] for r in records])
    d = s - starts[idx]
    a,b,c,e = coefficients[idx].T
    return a + d*(b+d*(c+d*e))


def reference_line(road, s):
    geometries = road.findall("planView/geometry")
    starts = np.array([float(g.get("s")) for g in geometries])
    idx = np.searchsorted(starts,s,side="right")-1
    if (idx<0).any():
        raise AdapterError("incomplete_opendrive", road.get("id"))
    result = np.empty((len(s),3))
    for i in np.unique(idx):
        g=geometries[i];mask=idx==i;d=s[mask]-starts[i]
        x,y,h = (float(g.get(k)) for k in ("x","y","hdg"))
        primitive=list(g)[0]
        if primitive.tag=="line":
            xx=x+d*np.cos(h);yy=y+d*np.sin(h);hh=np.full_like(d,h)
        elif primitive.tag=="arc":
            k=float(primitive.get("curvature"));hh=h+k*d
            if abs(k)<1e-12:
                xx=x+d*np.cos(h);yy=y+d*np.sin(h)
            else:
                xx=x+(np.sin(hh)-np.sin(h))/k;yy=y-(np.cos(hh)-np.cos(h))/k
        elif primitive.tag=="spiral":
            # OpenDRIVE clothoid: curvature changes linearly with arc length.
            # Fresnel integration retains the real start/end curvature and heading.
            from scipy.special import fresnel
            length=float(g.get('length'))
            if length<=0:raise AdapterError('invalid_opendrive_geometry','nonpositive spiral length')
            k0=float(primitive.get('curvStart'));k1=float(primitive.get('curvEnd'))
            rate=(k1-k0)/length
            hh=h+k0*d+.5*rate*d*d
            if abs(rate)<1e-14:
                if abs(k0)<1e-12:
                    integral=d.astype(complex)
                else:
                    integral=np.expm1(1j*k0*d)/(1j*k0)
            else:
                scale=np.sqrt(abs(rate)/np.pi)
                t0=scale*k0/rate;t1=t0+scale*d
                s0,c0=fresnel(t0);s1,c1=fresnel(t1)
                integral=np.sqrt(np.pi/abs(rate))*np.exp(-.5j*k0*k0/rate)*(
                    c1-c0+1j*np.sign(rate)*(s1-s0))
            rotated=np.exp(1j*h)*integral
            xx=x+rotated.real;yy=y+rotated.imag
        else:
            raise AdapterError("unsupported_opendrive_geometry", {"road":road.get("id"),"primitive":primitive.tag})
        result[mask]=np.column_stack((xx,yy,hh))
    return result


class OpenDriveLocalMap:
    def __init__(self,path,town,sample_spacing_m=.5,height_tolerance_m=3., rules=None):
        self.path=Path(path).resolve();self.town=town
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.spacing=sample_spacing_m;self.height_tolerance=height_tolerance_m
        self.rules=label_rules(rules);self._guardrails=None
        if self.spacing<=0 or self.spacing>.5 or self.height_tolerance<=0:
            raise AdapterError("invalid_bev_config", "map spacing <= .5m and positive height tolerance required")
        tree=ET.parse(self.path)
        self.header=dict(tree.getroot().find("header").attrib)
        self.roads={r.get("id"):r for r in tree.getroot().findall("road")}
        # Map-only coarse spatial index; no dataset scan or external server.
        self.bounds=[]
        for road in self.roads.values():
            length=float(road.get("length"))
            s=np.linspace(0,length,max(2,int(math.ceil(length/5))+1))
            xyz=reference_line(road,s)[:,:2]
            self.bounds.append((road,xyz.min(0)-40,xyz.max(0)+40))

    def local_lanes(self, raw):
        """Map XY is OpenDRIVE right-handed: (CARLA x, -CARLA y)."""
        origin=np.array([raw.actor_pose.x,-raw.actor_pose.y])
        ego_z=float(raw.ego_box["location"][2])
        expected_road=str(raw.ego_box["road_id"]);expected_lane=int(raw.ego_box["lane_id"])
        if expected_road not in self.roads:
            raise AdapterError("map_route_mismatch", "ego road id absent from configured OpenDRIVE")
        lanes=[];ego_matches=[];local_types=set()
        for road,low,high in self.bounds:
            if ((origin<low-48)|(origin>high+48)).any():
                continue
            sections=road.findall("lanes/laneSection")
            length=float(road.get("length"))
            for section_idx,section in enumerate(sections):
                start=float(section.get("s"));stop=float(sections[section_idx+1].get("s")) if section_idx+1<len(sections) else length
                if stop<=start:
                    continue
                s=np.linspace(start,stop,max(2,int(math.ceil((stop-start)/self.spacing))+1))
                ref=reference_line(road,s);xy=ref[:,:2];h=ref[:,2]
                elevation=polynomial(road.findall("elevationProfile/elevation"),s)
                roll=polynomial(road.findall("lateralProfile/superelevation"),s)
                lane_offset=polynomial(road.findall("lanes/laneOffset"),s)
                for side,sign in (("left",1),("right",-1)):
                    boundary=lane_offset.copy()
                    side_lanes=sorted(section.findall(side+"/lane"),key=lambda l:abs(int(l.get("id"))))
                    for lane in side_lanes:
                        widths=lane.findall("width")
                        if not widths:
                            if lane.get("type")=="none":
                                continue
                            raise AdapterError("unsupported_lane_width", {"road":road.get("id"),"lane":lane.get("id")})
                        width=polynomial(widths,s-start,key="sOffset")
                        if (width < -1e-5).any() or not np.isfinite(width).all():
                            raise AdapterError("invalid_lane_width", road.get("id"))
                        width=np.maximum(width,0)
                        inner=boundary.copy();outer=inner+sign*width;boundary=outer
                        center=(inner+outer)/2
                        normal=np.column_stack((-np.sin(h),np.cos(h)))
                        inner_xy=xy+normal*(inner*np.cos(roll))[:,None]
                        outer_xy=xy+normal*(outer*np.cos(roll))[:,None]
                        center_xy=xy+normal*(center*np.cos(roll))[:,None]
                        lane_z=elevation+center*np.sin(roll)
                        near=(np.linalg.norm(center_xy-origin,axis=1)<80)&(np.abs(lane_z-ego_z)<self.height_tolerance)&(width>1e-4)
                        if not near.any():
                            continue
                        lane_type=lane.get("type");local_types.add(lane_type)
                        # OpenDRIVE bidirectional is a drivable lane. Median is the
                        # explicit separator surface, outside official road/walkway layers.
                        if lane_type not in ("driving","bidirectional","sidewalk","none","shoulder","parking","biking","median"):
                            raise AdapterError("unmapped_lane_type",lane_type)
                        if road.get("id")==expected_road and int(lane.get("id"))==expected_lane:
                            ego_matches.append(float(np.min(np.linalg.norm(center_xy[near]-origin,axis=1)-width[near]/2)))
                        if lane_type not in ("driving","bidirectional","sidewalk"):
                            continue  # Specified non-road/non-walkway classes, not missing geometry.
                        # Split around other elevations/remote intervals; no bridges across gaps.
                        padded=np.r_[False,near,False];changes=np.flatnonzero(padded[1:]!=padded[:-1])
                        for a,b in changes.reshape(-1,2):
                            if b-a<2: continue
                            polygon=Polygon(np.concatenate((inner_xy[a:b],outer_xy[a:b][::-1])))
                            if not polygon.is_valid:
                                from shapely.validation import explain_validity
                                detail={"road":road.get("id"),"lane":lane.get("id"),"lane_type":lane_type,
                                    "reason":explain_validity(polygon),"map_source":str(self.path)}
                                if self.rules.enriched:
                                    detail["offset_geometry"]=offset_geometry_diagnostic(road,s[a:b],inner[a:b],outer[a:b])
                                raise AdapterError("invalid_map_polygon",detail)
                            if polygon.area<1e-6: continue
                            lanes.append({"polygon":polygon,"baseline":LineString(center_xy[a:b]),
                                          "type":lane_type,"junction":road.get("junction")!="-1"})
                            if self.rules.enriched:
                                # Sample both road edges and center for roadside height support.
                                lanes[-1]["surface_xyz"]=np.concatenate([np.column_stack((points[a:b],
                                    (elevation+offset*np.sin(roll))[a:b]))
                                    for points,offset in ((inner_xy,inner),(center_xy,center),(outer_xy,outer))])
        if not ego_matches or min(ego_matches)>1.:
            raise AdapterError("map_route_mismatch", {"ego_road":expected_road,"ego_lane":expected_lane,"lane_edge_distance":ego_matches})
        if not any(l["type"] in ("driving","bidirectional") for l in lanes):
            raise AdapterError("missing_map_coverage","no driving polygon near real ego")
        return lanes,{"ego_lane_edge_distance_m":min(ego_matches),"local_lane_types":sorted(local_types),
                      "lane_polygon_count":len(lanes),"height_tolerance_m":self.height_tolerance,"sample_spacing_m":self.spacing}


    def local_guardrails(self, raw, lanes):
        """Supported open vertical faces; z support is local along each segment."""
        if self._guardrails is None:
            supported=[];unsupported=[]
            for road in self.roads.values():
                for obj in road.findall('objects/object'):
                    if obj.get('type')!='barrier' or obj.get('name')!='GuardRail': continue
                    source={'road_id':road.get('id'),'object_id':obj.get('id')}
                    try:
                        surface=flat_guardrail_surface(road,obj,reference_line,polynomial)
                        supported.append((surface,source))
                    except AdapterError as exc:
                        unsupported.append(dict(source,reason=str(exc)))
            self._guardrails=supported,unsupported
        origin=np.array([raw.actor_pose.x,-raw.actor_pose.y])
        lines=[];sources=[];stats={'unsupported_sources':self._guardrails[1],
            'supported_source_count':len(self._guardrails[0]),'local_source_count':0,
            'segments_without_road_support':0,'other_layer_segments':0,
            'representation':'open vertical face; one-pixel line quantization; no inferred material thickness'}
        for surface,source in self._guardrails[0]:
            if ((origin<surface[:,:2].min(0)-48)|(origin>surface[:,:2].max(0)+48)).any():continue
            accepted=0
            for a,b in zip(surface[:-1],surface[1:]):
                if LineString([a[:2],b[:2]]).distance(Point(origin))>48:continue
                n=max(1,int(math.ceil(np.linalg.norm(b[:2]-a[:2])/self.spacing)))
                for j in range(n):
                    aa=a+(b-a)*j/n;bb=a+(b-a)*(j+1)/n
                    support=road_support(lanes,(aa[:2]+bb[:2])/2,self.rules)
                    if support['status']!='supported':
                        stats['segments_without_road_support']+=1;continue
                    lo,hi=support['road_z_range_m'];margin=self.rules.guardrail_deck_clearance_m
                    clipped=clip_vertical_segment(aa,bb,lo-margin,hi+margin)
                    if clipped is None:stats['other_layer_segments']+=1;continue
                    if np.linalg.norm(clipped[0]-clipped[1])<1e-9:continue
                    lines.append(LineString(clipped));accepted+=1
            if accepted:
                sources.append(dict(source,accepted_segments=accepted,world_z_range_m=[float(surface[:,2].min()),float(surface[:,3].max())]))
        stats['local_source_count']=len(sources);stats['sources']=sources
        return lines,stats


class OfficialMapView:
    """Only the proximal query used by the official target builder, with real geometry."""
    def __init__(self,lanes): self.lanes=lanes

    def get_proximal_map_objects(self,point,radius,layers):
        from nuplan.common.maps.abstract_map import SemanticMapLayer as S
        from shapely.geometry import Point
        origin=Point(point.x,point.y);result={layer:[] for layer in layers}
        for lane in self.lanes:
            if lane["polygon"].distance(origin)>radius: continue
            layer=S.WALKWAYS if lane["type"]=="sidewalk" else (S.INTERSECTION if lane["junction"] else S.LANE)
            if layer in result:
                result[layer].append(SimpleNamespace(polygon=lane["polygon"],baseline_path=SimpleNamespace(linestring=lane["baseline"])))
            # Junction connector baselines use the same centerline class and official width.
            baseline_layer=S.LANE_CONNECTOR if lane["junction"] else S.LANE
            if lane["type"] in ("driving","bidirectional") and baseline_layer in result and baseline_layer!=layer:
                result[baseline_layer].append(SimpleNamespace(polygon=lane["polygon"],baseline_path=SimpleNamespace(linestring=lane["baseline"])))
        return result


def bev_actor_name(box):
    cls=box.get("class");type_id=str(box.get("type_id",""))
    if cls=="ego_vehicle": return None
    if cls=="vehicle":
        if not is_b2d_vehicle(box): raise AdapterError("invalid_vehicle_box",type_id)
        return "vehicle"
    if cls in ("walker","pedestrian") and type_id.startswith("walker."): return "pedestrian"
    if cls=="traffic_light" and type_id.startswith("traffic.traffic_light"): return None
    if cls=="traffic_sign":
        if type_id.startswith(("traffic.speed_limit.","traffic.stop","traffic.yield")): return None
        lower=type_id.lower()
        if "cone" in lower: return "traffic_cone"
        if "warning" in lower: return "czone_sign"
        if "dirtdebris" in lower: return "generic_object"
    if cls in ("traffic_cone","barrier","czone_sign","generic_object"): return cls
    raise AdapterError("unmapped_bev_actor", {"class":cls,"type_id":type_id})


def bev_annotations(raw,geometry):
    # Same real center/dimensions/yaw transformation as the detection target.
    boxes=[];names=[];ignored=[]
    for box in raw.annotation["bounding_boxes"]:
        name=bev_actor_name(box)
        if name is None:
            ignored.append({"class":box.get("class"),"type_id":box.get("type_id")});continue
        try:
            center=np.asarray(box["center"],dtype=np.float64);extent=np.asarray(box["extent"],dtype=np.float64)
            rotation=np.asarray(box["rotation"],dtype=np.float64)
            if center.shape!=(3,) or extent.shape!=(3,) or rotation.shape!=(3,) or not np.isfinite(np.r_[center,extent,rotation]).all() or (extent<=0).any():
                raise ValueError("finite positive real box geometry required")
            xy=world_to_ego(center[:2],raw.actor_pose)
            z=(np.asarray(raw.ego_box["world2ego"])@np.r_[center,1])[2]+geometry.lidar_z_shift_m
            boxes.append([xy[0]-geometry.axle_dx_m,-(xy[1]-geometry.axle_dy_m),z,
                          *(2*extent),wrap_angle(raw.actor_pose.yaw-math.radians(rotation[2]))])
            names.append(name)
        except (KeyError,ValueError,TypeError) as exc:
            raise AdapterError("invalid_bev_actor",str(exc)) from exc
    return SimpleNamespace(boxes=np.array(boxes,dtype=np.float32).reshape(-1,7),names=names),ignored


class B2DOpenDriveBEVBuilder:
    def __init__(self,map_paths,sample_spacing_m=.5,height_tolerance_m=3.,cache_dir=None,max_maps=2,label_config=None):
        self.rules=label_rules(label_config)
        self.paths=dict(map_paths);self.maps=OrderedDict();self.spacing=sample_spacing_m;self.height_tolerance=height_tolerance_m
        self.cache_dir=Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir and self.rules.enriched:
            self.cache_dir=self.cache_dir/self.rules.name/self.rules.version
        if self.cache_dir: self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.max_maps=max_maps;self.identities={}
        self.cache_stats={"hits":0,"generated":0}
        self.producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def map_identity(self,town):
        spec=self.paths[town]
        spec={"path":spec,"carla_version":"unspecified legacy configuration"} if isinstance(spec,str) else spec
        path=Path(spec['path']).resolve();stat=path.stat();fingerprint=(stat.st_size,stat.st_mtime_ns)
        prior=self.identities.get(town)
        if prior and prior['stat']==fingerprint: return prior
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if spec.get('sha256') and spec['sha256']!=sha:
            raise AdapterError('map_version_mismatch',{'town':town,'path':str(path),'expected':spec['sha256'],'actual':sha})
        identity={'path':str(path),'sha256':sha,'carla_version':spec['carla_version'],'stat':fingerprint}
        self.identities[town]=identity
        self.maps.pop(town,None)
        return identity

    def build_targets(self,current,reader,native_config,model_config):
        import torch
        from nuplan.common.actor_state.state_representation import StateSE2
        reader.index.check_ref(current)
        match=re.search(r"_(Town\d+(?:HD)?)_",current.route.name)
        if not match or match.group(1) not in self.paths:
            raise AdapterError("missing_route_map",current.route.name)
        town=match.group(1)
        identity=self.map_identity(town)
        contract=validate_v1_config(native_config,model_config)
        annotation_path=reader.index.path(current,'anno')
        cache_identity={'schema':'b2d-v1-bev-cache-v1','current_token':current.token,
            'annotation_sha256':hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
            'map_sha256':identity['sha256'],'map_version':identity['carla_version'],
            'contract_sha256':contract['contract_sha256'],'classes':BEV_CLASSES,
            'sample_spacing_m':self.spacing,'height_tolerance_m':self.height_tolerance,
            'producer_sha256':self.producer_sha256,'label_rules':self.rules.identity(),
            'label_rules_sha256':self.rules.sha256}
        if self.rules.enriched:
            cache_identity.update(schema='b2d-v1-bev-cache-m4-v1',label_rules=self.rules.identity(),
                                  label_rules_sha256=self.rules.sha256,storage_dtype='uint8',
                                  geometry_sha256=hashlib.sha256(Path(__file__).with_name('v1_geometry.py').read_bytes()).hexdigest())
        cache_key=hashlib.sha256(json.dumps(cache_identity,sort_keys=True).encode()).hexdigest()
        cache_path=self.cache_dir/(cache_key+'.npz') if self.cache_dir else None
        if cache_path and cache_path.exists():
            try:
                with np.load(cache_path,allow_pickle=False) as archive:
                    labels=archive['labels'].copy();metadata=json.loads(str(archive['metadata'].item()))
                expected_dtype=np.uint8 if self.rules.enriched else np.int64
                if labels.dtype!=expected_dtype or labels.shape!=(128,256) or labels.min()<0 or labels.max()>6:
                    raise ValueError('invalid cached labels')
                if (metadata['provenance']['cache_key']!=cache_key or
                        json.dumps(metadata['provenance']['cache_identity'],sort_keys=True)!=json.dumps(cache_identity,sort_keys=True)):
                    raise ValueError('cache identity mismatch')
                if hashlib.sha256(labels.tobytes()).hexdigest()!=metadata['labels_sha256']:
                    raise ValueError('cached labels checksum mismatch')
            except (ValueError,OSError,KeyError) as exc:
                raise AdapterError('invalid_bev_cache',str(exc)) from exc
            metadata['provenance']['cache_read']='hit';self.cache_stats['hits']+=1
            return NativeBuildResult({'bev_semantic_map':torch.from_numpy(labels.astype(np.int64,copy=False))},metadata['provenance'],metadata['validation'])
        if town not in self.maps:
            self.maps[town]=OpenDriveLocalMap(identity['path'],town,self.spacing,self.height_tolerance,self.rules.identity())
            while len(self.maps)>self.max_maps: self.maps.popitem(last=False)
        self.maps.move_to_end(town)
        map_data=self.maps[town];raw=reader.read(current)
        lanes,coverage=map_data.local_lanes(raw)
        actor_polygons=[];floor_diagnostics=[];rail_lines=[];rail_diagnostics={}
        annotation_raw=raw
        if self.rules.enriched:
            # Exclude projected/filtered actors from the official planar-mask input.
            keep=[]
            for box in raw.annotation["bounding_boxes"]:
                name=bev_actor_name(box)
                if name is None:keep.append(box);continue
                hull,vertices=actor_local_hull(box,raw,native_config.geometry)
                diagnostic=actor_floor_diagnostic(box,vertices,lanes,self.rules)
                diagnostic["projected_3d"]=bool(np.max(np.abs(box["rotation"][:2]))>native_config.geometry.max_planar_tilt_deg)
                floor_diagnostics.append(diagnostic)
                if diagnostic["excluded"]:continue
                if diagnostic["projected_3d"]:actor_polygons.append((name,hull))
                else:keep.append(box)
            annotation_raw=SimpleNamespace(annotation={"bounding_boxes":keep},actor_pose=raw.actor_pose,ego_box=raw.ego_box)
            rail_lines,rail_diagnostics=map_data.local_guardrails(raw,lanes)
        annotations,ignored=bev_annotations(annotation_raw,native_config.geometry)
        rear=actor_to_reference(raw.actor_pose,(native_config.geometry.axle_dx_m,native_config.geometry.axle_dy_m))
        pose=StateSE2(rear.x,-rear.y,-rear.yaw)
        official=official_v1_module("transfuser_features").TransfuserTargetBuilder(model_config)
        # Retain official projection, class ordering/overwrites, and line width exactly.
        if self.rules.enriched:
            labels=enriched_labels(official,annotations,OfficialMapView(lanes),pose,actor_polygons,rail_lines)
        else:
            labels=official._compute_bev_semantic_map(annotations,OfficialMapView(lanes),pose).to(torch.int64)
        result=NativeBuildResult({"bev_semantic_map":labels},
            {"current_token":current.token,"contract_sha256":contract["contract_sha256"],"class_names":BEV_CLASSES,
             "map_source":str(map_data.path),"map_sha256":map_data.sha256,"map_town":town,"map_header":map_data.header,
             "map_version":identity['carla_version'],'cache_key':cache_key,'cache_identity':cache_identity,
             'cache_read':'generated','producer_sha256':self.producer_sha256,
             "label_rules":self.rules.identity(),"label_rules_sha256":self.rules.sha256,
             "annotation_source":str(reader.index.path(current,"anno")),"coverage":coverage,"ignored_official_non_targets":ignored,
             "class_pixel_counts":torch.bincount(labels.flatten(),minlength=7).tolist(),
             "actor_names":annotations.names,"geometry_policy":"b2d-opendrive-line-arc-spiral-v2",
             "source_policy":"complete supplied OpenDRIVE lane surface + published current B2D annotations; no RGB projection or fake label fill"},
            {"real_labels_validated":True,"map_ego_alignment_pass":True,"formal_semantics":"B2D class mapping; not NuPlan map equivalence"})
        if self.rules.enriched:
            result.provenance.update(label_rules=self.rules.identity(),label_rules_sha256=self.rules.sha256,
                guardrail=rail_diagnostics,actor_floor_diagnostics=floor_diagnostics,
                actor_names=annotations.names+[name for name,_ in actor_polygons],
                rear_reference_world_z_m=float(raw.ego_box['location'][2]+
                    (carla_rotation(raw.ego_box['rotation'])@np.array([native_config.geometry.axle_dx_m,native_config.geometry.axle_dy_m,0.]))[2]),
                geometry_policy='m4-world-xy-hull-and-supported-vertical-surfaces-v1')
        self.cache_stats['generated']+=1
        if cache_path:
            values=labels.numpy().astype(np.uint8) if self.rules.enriched else labels.numpy();metadata={'provenance':result.provenance,'validation':result.validation,
                                           'labels_sha256':hashlib.sha256(values.tobytes()).hexdigest()}
            fd,tmp=tempfile.mkstemp(prefix='bev-',suffix='.npz',dir=self.cache_dir)
            try:
                with os.fdopen(fd,'wb') as stream:
                    np.savez_compressed(stream,labels=values,metadata=np.array(json.dumps(metadata)))
                os.replace(tmp,cache_path)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        return result
