"""M5 map/section caches and local multilayer queries, separate from V1 targets.

Uses the existing OpenDriveLocalMap XML/spatial index, reference_line and
polynomial functions. Boundaries are geometry; RoadMark masks classify source
intervals, not individual painted dashes or invented double-line offsets.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import shapely
import scipy
from shapely.geometry import LineString, Polygon, Point, box as rectangle
from shapely.validation import explain_validity

from ..errors import AdapterError
from ..coordinate_transform import actor_to_reference
from ..v1_bev import reference_line, polynomial
from ..v1_geometry import carla_rotation, offset_geometry_diagnostic

SCHEMA = 'b2d-m5-static-map-v1'
ROADMARK_TYPES = {'none': 1, 'curb': 2, 'solid': 3, 'broken': 4,
                  'solid solid': 5, 'solid broken': 6, 'broken solid': 7}
UNKNOWN = 255
DRIVABLE = ('driving', 'bidirectional')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_npz(path, arrays, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, array_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
                    array_contract={k:{'dtype':v.dtype.str,'shape':list(v.shape)} for k,v in arrays.items()})
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays, metadata=np.array(json.dumps(metadata, ensure_ascii=False)))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def read_npz(path, identity):
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata'].item()))
            arrays = {k: archive[k].copy() for k in archive.files if k != 'metadata'}
        if metadata['identity'] != identity or set(arrays) != set(metadata['array_sha256']):
            raise ValueError('identity/array key mismatch')
        if any(hashlib.sha256(v.tobytes()).hexdigest() != metadata['array_sha256'][k] or not np.isfinite(v).all()
               or metadata['array_contract'][k]!={'dtype':v.dtype.str,'shape':list(v.shape)} for k,v in arrays.items()):
            raise ValueError('nonfinite/checksum mismatch')
        return arrays, metadata
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise AdapterError('invalid_static_map_cache', f'{path}: {exc}') from exc


def load_static_query(path, expected_map_sha256=None, expected_cache_key=None):
    """Read numerical layers/provenance without XML, sensors or training tensors."""
    try:
        with np.load(path,allow_pickle=False) as archive:
            identity=json.loads(str(archive['metadata'].item()))['identity']
        arrays,metadata=read_npz(path,identity)
        provenance=metadata['provenance']
        if identity['schema']!=SCHEMA or provenance['schema']!=SCHEMA:
            raise ValueError('unknown static layer schema')
        if expected_map_sha256 and provenance['map_identity']['map_sha256']!=expected_map_sha256:
            raise ValueError('different map SHA')
        if expected_cache_key and identity['map_cache_key']!=expected_cache_key:
            raise ValueError('different map/rule/converter cache identity')
        for name,values in arrays.items():
            expected=(len(provenance['roadmark_layer_type_ids']),128,256) if name=='roadmark_type_layers' else (128,256)
            if values.dtype!=np.uint8 or values.shape!=expected:
                raise ValueError('invalid mask dtype/shape')
            if name=='roadmark_type_mask':
                if not set(np.unique(values))<={0,UNKNOWN,*provenance['roadmark_layer_type_ids']}:
                    raise ValueError('unknown RoadMark label')
            elif not ((values==0)|(values==1)).all():
                raise ValueError('invalid binary layer')
        return {'masks':arrays,'provenance':provenance}
    except (ValueError,OSError,KeyError,TypeError) as exc:
        raise AdapterError('invalid_static_query_cache',str(exc)) from exc


@dataclass(frozen=True)
class StaticLayerRules:
    version: str = 'm5-v1'
    sample_spacing_m: float = .5
    map_height_tolerance_m: float = 3.
    boundary_thickness_px: int = 2
    roadmark_interval_thickness_px: int = 1
    continuity_tolerance_m: float = 1e-4
    pixel_size_m: float = .25
    shape: tuple = (128, 256)

    def __post_init__(self):
        if (self.version != 'm5-v1' or not 0 < self.sample_spacing_m <= .5
                or not 0 < self.map_height_tolerance_m < float('inf')
                or not 0 < self.continuity_tolerance_m <= .001
                or self.boundary_thickness_px not in (1, 2) or self.roadmark_interval_thickness_px != 1
                or self.pixel_size_m != .25 or tuple(self.shape) != (128, 256)):
            raise AdapterError('invalid_static_map_rules', asdict(self))

    def identity(self):
        return json.loads(json.dumps(asdict(self)))


def _clip_height(a, b, low, high):
    begin, end = 0., 1.
    delta = b[2]-a[2]
    if abs(delta) < 1e-12:
        return (a, b) if low <= a[2] <= high else None
    u, v = sorted(((low-a[2])/delta, (high-a[2])/delta))
    begin, end = max(begin, u), min(end, v)
    if end <= begin: return None
    return a+(b-a)*begin, a+(b-a)*end


def _local_xy(xyz, pose):
    xy = np.asarray(xyz)[:, :2]-np.array(pose[:2])
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return xy @ np.array([[c, -s], [s, c]])


def _pixels(xy):
    # Exact coordinate/transpose convention of official V1 target rasterizer.
    return (np.asarray(xy)/.25+np.array([0., 128.])).astype(np.int32)


def _line_mask(lines, thickness):
    canvas = np.zeros((256, 128), dtype=np.uint8)
    for line in lines:
        cv2.polylines(canvas, [_pixels(np.asarray(line.coords)).reshape(-1, 1, 2)], False, 1, thickness=thickness)
    return np.rot90(canvas)[::-1].copy()


def _parts(geometry):
    if geometry.is_empty: return []
    if geometry.geom_type in ('LineString', 'Polygon'): return [geometry]
    return [item for item in getattr(geometry, 'geoms', []) if item.geom_type in ('LineString', 'Polygon')]


def _profile_guard(road):
    for record in road.findall('lateralProfile/shape')+road.findall('lateralProfile/crossfall'):
        coefficients = np.array([float(record.get(k, 0)) for k in ('a', 'b', 'c', 'd')])
        if not np.isfinite(coefficients).all() or np.max(np.abs(coefficients)) > 1e-8:
            raise AdapterError('unsupported_static_map_geometry', 'nonzero lateral shape/crossfall not in existing lane converter')


class MapStaticLayers:
    """Wrap a reused OpenDriveLocalMap; lazily persist each world-coordinate section.

    Cached arrays stay map-relative XYZ and are reused for any ego query. This
    module lives in a subpackage so M2/M4's top-level producer file list is unchanged.
    """
    def __init__(self, map_data, carla_version, cache_dir=None, rules=None, expected_sha256=None):
        self.map = map_data
        if hashlib.sha256(map_data.path.read_bytes()).hexdigest()!=map_data.sha256:
            raise AdapterError('map_version_mismatch','map file changed after XML parsing')
        if not isinstance(carla_version,str) or not carla_version.strip():
            raise AdapterError('invalid_static_map_rules','explicit map/CARLA version required')
        self.rules = rules or StaticLayerRules(sample_spacing_m=map_data.spacing,
                                               map_height_tolerance_m=map_data.height_tolerance)
        if expected_sha256 and expected_sha256 != map_data.sha256:
            raise AdapterError('map_version_mismatch', map_data.town)
        implementation = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                          (__file__, Path(__file__).parents[1]/'v1_bev.py', Path(__file__).parents[1]/'v1_geometry.py')}
        self.identity = {'schema': SCHEMA, 'town': map_data.town, 'map_sha256': map_data.sha256,
                         'carla_version': str(carla_version), 'opendrive_header':map_data.header, 'rules': self.rules.identity(),
                         'implementation': implementation, 'libraries':{'numpy':np.__version__,'opencv':cv2.__version__,'shapely':shapely.__version__,'scipy':scipy.__version__},
                         'coordinate_frame': 'CARLA x, -CARLA y, CARLA world z',
                         'roadmark_types': ROADMARK_TYPES}
        self.cache_key = digest(self.identity)
        self.cache_dir = Path(cache_dir)/map_data.sha256/self.cache_key if cache_dir else None
        self.sections = {}; self.objects = None
        self.cache_stats = {'section_hits': 0, 'section_generated': 0, 'object_hits': 0, 'object_generated': 0}

    def _section_identity(self, road_id, section_index):
        return json.loads(json.dumps(dict(self.identity, road_id=str(road_id), section_index=int(section_index))))

    def section(self, road_id, section_index=0):
        key = str(road_id), int(section_index)
        if key in self.sections: return self.sections[key]
        identity = self._section_identity(*key)
        path = self.cache_dir/'sections'/(digest(identity)+'.npz') if self.cache_dir else None
        if path and path.exists():
            arrays, metadata = read_npz(path, identity); self.cache_stats['section_hits'] += 1
        else:
            arrays, records = self._build_section(*key)
            metadata = {'identity': identity, **records}
            if path: atomic_npz(path, arrays, metadata)
            self.cache_stats['section_generated'] += 1
        self.sections[key] = arrays, metadata
        return arrays, metadata

    def _build_section(self, road_id, section_index):
        road = self.map.roads[road_id]; sections = road.findall('lanes/laneSection')
        section = sections[section_index]
        start = float(section.get('s')); stop = float(sections[section_index+1].get('s')) if section_index+1 < len(sections) else float(road.get('length'))
        if stop <= start: raise AdapterError('unsupported_static_map_geometry', 'nonpositive laneSection length')
        _profile_guard(road)
        all_lanes = section.findall('left/lane')+section.findall('right/lane')
        breaks = [start, stop]
        for query, attr, base in [('planView/geometry','s',0.), ('elevationProfile/elevation','s',0.),
                                  ('lateralProfile/superelevation','s',0.), ('lanes/laneOffset','s',0.)]:
            breaks += [float(r.get(attr))+base for r in road.findall(query)]
        for lane in all_lanes+section.findall('center/lane'):
            for child in lane.findall('width')+lane.findall('roadMark')+lane.findall('userData/vectorLane'):
                breaks.append(start+float(child.get('sOffset', 0)))
        breaks = sorted(set(v for v in breaks if start <= v <= stop))
        s = np.unique(np.r_[np.linspace(start, stop, max(2, int(math.ceil((stop-start)/self.rules.sample_spacing_m))+1)), breaks,
                            [np.nextafter(v,start) for v in breaks[1:-1]]])
        ref = reference_line(road, s); elevation = polynomial(road.findall('elevationProfile/elevation'), s)
        roll = polynomial(road.findall('lateralProfile/superelevation'), s)
        offset = polynomial(road.findall('lanes/laneOffset'), s)
        if not np.isfinite(np.r_[ref.ravel(), elevation, roll, offset]).all():
            raise AdapterError('unsupported_static_map_geometry', 'nonfinite reference profiles')
        normal = np.column_stack((-np.sin(ref[:, 2]), np.cos(ref[:, 2])))

        def xyz(t):
            return np.column_stack((ref[:, :2]+normal*(t*np.cos(roll))[:, None], elevation+t*np.sin(roll)))

        def edge_valid(points):
            return (~((np.diff(s)<1e-8)&(np.linalg.norm(np.diff(points,axis=0),axis=1)>self.rules.continuity_tolerance_m))).astype(np.uint8)

        arrays = {'s': s, 'boundary_0_xyz': xyz(offset), 'boundary_0_valid': edge_valid(xyz(offset))}
        lanes = []; boundaries = {'0': {'id': f'{road_id}:{section_index}:b0', 'array': 'boundary_0_xyz', 'valid_segments_array':'boundary_0_valid', 'adjacent_lanes': []}}
        lane_xml = {int(l.get('id')): l for l in all_lanes}
        for side, sign in (('left', 1), ('right', -1)):
            inner = offset.copy(); previous_boundary = '0'
            for lane in sorted(section.findall(side+'/lane'), key=lambda l: abs(int(l.get('id')))):
                lid = int(lane.get('id')); widths = lane.findall('width')
                if not widths:
                    if lane.get('type') == 'none': continue
                    raise AdapterError('unsupported_static_map_geometry', {'road': road_id, 'lane': lid, 'reason': 'missing width/border-only unsupported'})
                width = polynomial(widths, s-start, key='sOffset')
                if not np.isfinite(width).all() or (width < -1e-5).any():
                    raise AdapterError('unsupported_static_map_geometry', {'road': road_id, 'lane': lid, 'reason': 'invalid width'})
                width = np.maximum(width, 0); outer = inner+sign*width; center = (inner+outer)/2
                tag = str(lid); boundary_key = f'boundary_{lid}_xyz'; center_key = f'lane_{lid}_center_xyz'; width_key = f'lane_{lid}_width'
                arrays[boundary_key] = xyz(outer); arrays[center_key] = xyz(center); arrays[width_key] = width
                arrays[f'boundary_{lid}_valid']=edge_valid(arrays[boundary_key]);arrays[f'lane_{lid}_center_valid']=edge_valid(arrays[center_key])
                boundary = {'id': f'{road_id}:{section_index}:b{lid}', 'array': boundary_key, 'valid_segments_array':f'boundary_{lid}_valid', 'adjacent_lanes': [lid]}
                boundaries[tag] = boundary; boundaries[previous_boundary]['adjacent_lanes'].append(lid)
                inner_xyz,outer_xyz=xyz(inner),xyz(outer)
                valid_edges=edge_valid(inner_xyz)&edge_valid(outer_xyz)
                runs=np.flatnonzero(np.r_[False,valid_edges>0,False][1:]!=np.r_[False,valid_edges>0,False][:-1]).reshape(-1,2)
                polygons=[];duplicates_removed=0
                for a,b in runs:
                    ring=np.r_[inner_xyz[a:b+1,:2],outer_xyz[a:b+1,:2][::-1]]
                    # nextafter limits can produce consecutive coincident vertices.
                    # Drop only sub-nanometer numerical duplicates for validity checks;
                    # cached world coordinates and actual segments remain untouched.
                    keep=np.r_[True,np.linalg.norm(np.diff(ring,axis=0),axis=1)>1e-10]
                    duplicates_removed+=int((~keep).sum())
                    polygons.append(Polygon(ring[keep]) if keep.sum()>=3 else Polygon())
                polygon=next((p for p in polygons if not p.is_valid),polygons[0] if polygons else Polygon())
                diagnostics = offset_geometry_diagnostic(road, s, inner, outer)
                reason = None
                if diagnostics['folded_source_coordinates']: reason = 'folded lane offset coordinates'
                elif not polygon.is_valid: reason = explain_validity(polygon)
                elif not polygons or sum(p.area for p in polygons) < 1e-6: reason = 'zero-width/degenerate lane area'
                elif lane.findall('height'): reason = 'lane-specific height not handled by existing converter'
                elif lane.get('level') == 'true' and np.max(np.abs(roll)) > 1e-8: reason = 'level lane with banked reference unsupported'
                higher, lower = (tag, previous_boundary) if sign > 0 else (previous_boundary, tag)
                direction = [{'sOffset': float(v.get('sOffset', 0)), 'travelDir': v.get('travelDir'), 'laneId': v.get('laneId'),
                              'source': 'lane/userData/vectorLane'} for v in lane.findall('userData/vectorLane')]
                raw_links = {kind: dict(node.attrib) for kind in ('predecessor', 'successor') if (node := lane.find('link/'+kind)) is not None}
                lanes.append({'id': f'{road_id}:{section_index}:l{lid}', 'road_id': road_id, 'lane_id': lid,
                              'section_index': section_index, 'section_s': start, 's_interval_m': [start, stop],
                              'lane_type': lane.get('type'), 'width_profiles_raw':[dict(w.attrib) for w in widths], 'height_profiles_raw':[dict(h.attrib) for h in lane.findall('height')], 'junction_id': road.get('junction'), 'center_array': center_key,
                              'width_array': width_key, 'center_valid_segments_array':f'lane_{lid}_center_valid', 'left_boundary_id': boundaries[higher]['id'], 'right_boundary_id': boundaries[lower]['id'],
                              'left_right_frame': 'increasing OpenDRIVE s; not inferred traffic direction',
                              'inner_boundary_id': boundaries[previous_boundary]['id'], 'outer_boundary_id': boundary['id'],
                              'direction_records': direction, 'lane_direction_raw': lane.get('direction'), 'lane_links_raw': raw_links,
                              'status': 'supported' if reason is None else 'unsupported', 'reason': reason, 'numerical_duplicate_vertices_removed_for_validation':duplicates_removed, 'offset_geometry': diagnostics})
                inner = outer; previous_boundary = tag
        by_id = {l['lane_id']: l for l in lanes}
        for boundary in boundaries.values():
            owner_types = [by_id[l]['lane_type'] for l in boundary['adjacent_lanes']]
            if len(owner_types) < 2: role = 'section_outer_edge'
            elif all(t in DRIVABLE for t in owner_types): role = 'drivable_lane_divider'
            elif any(t in DRIVABLE for t in owner_types): role = 'drivable_non_drivable_transition'
            else: role = 'non_drivable_lane_divider'
            boundary['role'] = role; boundary['adjacent_lane_types'] = owner_types
            rejected=[{'lane_id':l,'reason':by_id[l]['reason']} for l in boundary['adjacent_lanes'] if by_id[l]['status']!='supported']
            boundary['status']='unsupported' if rejected else 'supported'
            boundary['unsupported_owners']=rejected

        # Keep both real limits at declared starts, exclude only the connecting
        # jump, and record it. No guessed connector joins width/elevation steps.
        discontinuities=[]
        for boundary in boundaries.values():
            points=arrays[boundary['array']]
            for i in np.flatnonzero(arrays[boundary['valid_segments_array']]==0):
                discontinuities.append({'boundary_id':boundary['id'],'s_m':float(s[i+1]),
                    'gap_m':float(np.linalg.norm(points[i+1]-points[i])),
                    'world_endpoints':points[i:i+2].tolist(),
                    'status':'unsupported_connector','reason':'declared geometry/profile discontinuity; real segments retained separately'})
        marks = []
        for lane in all_lanes+section.findall('center/lane'):
            lid = int(lane.get('id')); entries = lane.findall('roadMark'); boundary_id = f'{road_id}:{section_index}:b{lid}'
            starts = [float(e.get('sOffset')) for e in entries]
            if any(not np.isfinite(v) or v < 0 or v > stop-start for v in starts) or any(b <= a for a,b in zip(starts, starts[1:])):
                raise AdapterError('unsupported_static_map_geometry', {'road':road_id,'lane':lid,'reason':'invalid RoadMark intervals'})
            if not entries or starts[0] > 0:
                end = start+starts[0] if entries else stop
                marks.append({'id':f'{road_id}:{section_index}:mark:{lid}:unknown', 'boundary_id':boundary_id, 'lane_id':lid,
                              's_interval_m':[start,end], 'type':None, 'status':'unknown', 'reason':'RoadMark record absent', 'type_id':UNKNOWN})
            for i, mark in enumerate(entries):
                end = start+starts[i+1] if i+1<len(entries) else stop
                raw_width = mark.get('width'); mark_width = float(raw_width) if raw_width is not None else None
                if mark_width is not None and (not np.isfinite(mark_width) or mark_width < 0):
                    raise AdapterError('unsupported_static_map_geometry', 'invalid RoadMark width')
                typ = mark.get('type'); known = typ in ROADMARK_TYPES
                marks.append({'id':f'{road_id}:{section_index}:mark:{lid}:{i}', 'road_id':road_id, 'section_index':section_index,
                              'lane_id':lid, 'boundary_id':boundary_id, 'sOffset':starts[i], 's_interval_m':[start+starts[i],end],
                              'type':typ, 'type_id':ROADMARK_TYPES.get(typ, UNKNOWN), 'color':mark.get('color'),
                              'width_m':mark_width, 'width_known':raw_width is not None, 'laneChange':mark.get('laneChange'),
                              'attributes_raw':dict(mark.attrib), 'detailed_lines_raw': [dict(c.attrib) for c in mark.findall('type/line')+mark.findall('explicit/line')],
                              'status':'supported' if known else 'unsupported', 'reason':None if known else 'unrecognized source RoadMark type',
                              'geometry_meaning':'outer boundary interval; painted dash phase/stripe offsets not synthesized'})
        return arrays, {'lanes':lanes, 'boundaries':list(boundaries.values()), 'roadmarks':marks, 'discontinuities':discontinuities,
                        'road_links_raw':{kind:dict(node.attrib) for kind in ('predecessor','successor') if (node:=road.find('link/'+kind)) is not None},
                        'junction_id':road.get('junction'), 'road_rule_raw':road.get('rule')}

    def crosswalks(self):
        if self.objects is not None: return self.objects
        identity = json.loads(json.dumps(dict(self.identity, content='crosswalks')))
        path = self.cache_dir/'crosswalks.npz' if self.cache_dir else None
        if path and path.exists():
            arrays, metadata = read_npz(path, identity); self.cache_stats['object_hits'] += 1
        else:
            arrays = {}; records = []
            for road in self.map.roads.values():
                for obj in road.findall('objects/object'):
                    if obj.get('type') != 'crosswalk': continue
                    record = {'id': f"{road.get('id')}:crosswalk:{obj.get('id')}", 'road_id':road.get('id'), 'object_id':obj.get('id'),
                              'attributes_raw':dict(obj.attrib), 'source':'road/objects/object[@type=crosswalk]'}
                    try:
                        xyz, source = self._crosswalk_geometry(road, obj)
                        key = 'crosswalk_'+str(len(arrays)); arrays[key] = xyz
                        record.update(status='supported', array=key, geometry_source=source,
                                      bounds_xy=[xyz[:,:2].min(0).tolist(), xyz[:,:2].max(0).tolist()])
                    except (AdapterError, ValueError, TypeError, KeyError) as exc:
                        record.update(status='unsupported', reason=str(exc))
                        # A reliable declared box is used only to localize unknown coverage, never to create labels.
                        try:
                            pivot = self._object_pivot(road, obj); width, length = float(obj.get('width')), float(obj.get('length'))
                            radius = math.hypot(width, length)/2
                            if radius <= 0 or not np.isfinite(radius): raise ValueError('unknown box')
                            record['bounds_xy']=[(pivot[:2]-radius).tolist(),(pivot[:2]+radius).tolist()]
                        except (ValueError, TypeError, KeyError, AdapterError): record['bounds_xy']=None
                    records.append(record)
            metadata = {'identity':identity, 'crosswalks':records}
            if path: atomic_npz(path, arrays, metadata)
            self.cache_stats['object_generated'] += 1
        self.objects = arrays, metadata
        return self.objects

    @staticmethod
    def _object_pivot(road, obj):
        s, t, dz = (float(obj.get(k)) for k in ('s','t','zOffset'))
        if not np.isfinite([s,t,dz]).all() or not 0 <= s <= float(road.get('length')):
            raise AdapterError('unsupported_crosswalk', 'invalid object insertion')
        ref = reference_line(road, np.array([s]))[0]
        return np.array([ref[0]-t*math.sin(ref[2]), ref[1]+t*math.cos(ref[2]),
                         float(polynomial(road.findall('elevationProfile/elevation'), np.array([s]))[0])+dz, ref[2]])

    def _crosswalk_geometry(self, road, obj):
        if obj.find('repeat') is not None: raise AdapterError('unsupported_crosswalk','repeated object')
        pivot = self._object_pivot(road,obj)
        hdg, pitch, roll = (float(obj.get(k,0)) for k in ('hdg','pitch','roll'))
        if not np.isfinite([hdg,pitch,roll]).all(): raise AdapterError('unsupported_crosswalk','invalid object rotation')
        outlines = obj.findall('outline')+obj.findall('outlines/outline')
        if outlines:
            if len(outlines)!=1 or outlines[0].get('closed')=='false' or outlines[0].get('outer')=='false':
                raise AdapterError('unsupported_crosswalk','only one closed outer outline supported')
            outline = outlines[0]; corners = list(outline)
            if len(corners)<3 or any(c.tag!='cornerLocal' for c in corners):
                raise AdapterError('unsupported_crosswalk','cornerRoad/mixed/insufficient outline unsupported')
            local = np.array([[float(c.get(k)) for k in ('u','v','z')] for c in corners])
            if outline.get('closed')!='true' and not np.allclose(local[0],local[-1],rtol=0,atol=1e-6):
                raise AdapterError('unsupported_crosswalk','outline closure not specified or observed')
            rotation = carla_rotation([-math.degrees(pitch),-math.degrees(roll),math.degrees(pivot[3]+hdg)])
            xyz = local@rotation.T+pivot[:3]; source='closed cornerLocal outline with own object rotation'
        else:
            length,width = float(obj.get('length')),float(obj.get('width'))
            if not np.isfinite([length,width]).all() or min(length,width)<=0:
                raise AdapterError('unsupported_crosswalk','no valid outline or rectangular dimensions')
            local=np.array([[-length/2,-width/2,0],[length/2,-width/2,0],[length/2,width/2,0],[-length/2,width/2,0],[-length/2,-width/2,0]])
            rotation=carla_rotation([-math.degrees(pitch),-math.degrees(roll),math.degrees(pivot[3]+hdg)])
            xyz=local@rotation.T+pivot[:3];source='rectangular object dimensions, no outline present'
        polygon=Polygon(xyz[:,:2])
        if not np.isfinite(xyz).all() or not polygon.is_valid or polygon.area<1e-6:
            raise AdapterError('unsupported_crosswalk','invalid/degenerate source footprint: '+explain_validity(polygon))
        return xyz,source

    def query_from_raw(self, raw, geometry):
        rear=actor_to_reference(raw.actor_pose,(geometry.axle_dx_m,geometry.axle_dy_m))
        rear_z=float(raw.ego_box['location'][2]+(carla_rotation(raw.ego_box['rotation'])@np.array([geometry.axle_dx_m,geometry.axle_dy_m,0.]))[2])
        result=self.query((rear.x,-rear.y,-rear.yaw),rear_z)
        result['provenance'].update(current_token=raw.ref.token,rear_world_z_m=rear_z)
        return result

    def query(self, rear_pose, world_z_reference, road_ids=None):
        pose=tuple(float(v) for v in rear_pose); z=float(world_z_reference)
        if len(pose)!=3 or not np.isfinite([*pose,z]).all(): raise AdapterError('invalid_static_query','finite rear SE2/world z required')
        origin=np.array(pose[:2]); radius=math.hypot(32,32)
        low,high=z-self.rules.map_height_tolerance_m,z+self.rules.map_height_tolerance_m
        roi=rectangle(0,-32,32,32); lane_records=[]; boundary_records=[]; mark_records=[]; unsupported=[]
        centers=[]; boundaries=[]; gap_lines=[]; topology_refs=[]; mark_lines={k:[] for k in ROADMARK_TYPES.values()}; unknown_lines=[]
        selected_boundaries=set(); local_boundary_geometry={}

        def visible_lines(xyz, s=None, interval=None, valid=None):
            pieces=[]
            for i,(a,b) in enumerate(zip(xyz[:-1],xyz[1:])):
                if valid is not None and not valid[i]:continue
                aa,bb=a,b
                if s is not None and interval is not None:
                    begin,end=max(float(s[i]),interval[0]),min(float(s[i+1]),interval[1])
                    if end<=begin:continue
                    aa=a+(b-a)*(begin-s[i])/(s[i+1]-s[i]);bb=a+(b-a)*(end-s[i])/(s[i+1]-s[i])
                clipped=_clip_height(aa,bb,low,high)
                if clipped is None:continue
                local=_local_xy(np.array(clipped),pose)
                if np.linalg.norm(local[1]-local[0])<1e-9:continue
                pieces += [p for p in _parts(LineString(local).intersection(roi)) if p.geom_type=='LineString']
            return pieces

        for road,bbox_low,bbox_high in self.map.bounds:
            rid=road.get('id')
            if road_ids is not None and rid not in {str(r) for r in road_ids}:continue
            if ((origin<bbox_low-radius)|(origin>bbox_high+radius)).any():continue
            for section_index,_ in enumerate(road.findall('lanes/laneSection')):
                try: arrays,metadata=self.section(rid,section_index)
                except AdapterError as exc:
                    raise AdapterError('unsupported_static_map_geometry',{'road':rid,'section':section_index,'detail':str(exc)}) from exc
                boundary_by_id={b['id']:b for b in metadata['boundaries']}
                topology_refs.append({'road_id':rid,'section_index':section_index,'road_links_raw':metadata['road_links_raw'],'junction_id':metadata['junction_id'],'road_rule_raw':metadata['road_rule_raw']})
                for gap in metadata['discontinuities']:
                    pieces=visible_lines(np.array(gap['world_endpoints']))
                    if pieces:
                        gap_lines+=pieces;unsupported.append(dict(gap,road_id=rid,section_index=section_index))
                active=[]
                for lane in metadata['lanes']:
                    xyz=arrays[lane['center_array']]
                    left_boundary=boundary_by_id[lane['left_boundary_id']];right_boundary=boundary_by_id[lane['right_boundary_id']]
                    points=np.concatenate((xyz,arrays[left_boundary['array']],arrays[right_boundary['array']]))
                    if ((origin<points[:,:2].min(0)-radius)|(origin>points[:,:2].max(0)+radius)).any():continue
                    if lane['status']=='supported' and (points[:,2].max()<low or points[:,2].min()>high):continue
                    visible=visible_lines(xyz,valid=arrays[lane['center_valid_segments_array']])
                    left_boundary=boundary_by_id[lane['left_boundary_id']]
                    left=visible_lines(arrays[left_boundary['array']],valid=arrays[left_boundary['valid_segments_array']])
                    right_boundary=boundary_by_id[lane['right_boundary_id']]
                    right=visible_lines(arrays[right_boundary['array']],valid=arrays[right_boundary['valid_segments_array']])
                    if not (visible or left or right):continue
                    if lane['status']!='supported':
                        raise AdapterError('unsupported_static_map_geometry',{'lane':lane['id'],'reason':lane['reason'],'source':str(self.map.path)})
                    directions=[]
                    for i,entry in enumerate(lane['direction_records']):
                        a=lane['section_s']+entry['sOffset'];b=lane['section_s']+lane['direction_records'][i+1]['sOffset'] if i+1<len(lane['direction_records']) else lane['s_interval_m'][1]
                        direction=entry['travelDir'];left_id,right_id=lane['left_boundary_id'],lane['right_boundary_id']
                        directions.append(dict(entry,s_interval_m=[a,b],travel_left_boundary_id=left_id if direction=='forward' else right_id if direction=='backward' else None,
                                               travel_right_boundary_id=right_id if direction=='forward' else left_id if direction=='backward' else None,
                                               lane_change_decision_normalized=False))
                    record=dict(lane,direction_intervals=directions,local_center_lines=[np.array(p.coords).tolist() for p in visible])
                    lane_records.append(record);active.append(lane)
                    if lane['lane_type'] in DRIVABLE:
                        centers += visible
                        selected_boundaries.update((lane['left_boundary_id'],lane['right_boundary_id']))
                for boundary in metadata['boundaries']:
                    lines=visible_lines(arrays[boundary['array']],valid=arrays[boundary['valid_segments_array']])
                    if not lines:continue
                    if boundary['status']!='supported':
                        raise AdapterError('unsupported_static_map_geometry',{'boundary':boundary['id'],'unsupported_owners':boundary['unsupported_owners'],'source':str(self.map.path)})
                    local_boundary_geometry[boundary['id']]=(arrays[boundary['array']],arrays['s'],arrays[boundary['valid_segments_array']])
                    boundary_records.append(dict(boundary,road_id=rid,section_index=section_index,local_lines=[np.array(p.coords).tolist() for p in lines]))
                    if boundary['id'] in selected_boundaries:boundaries += lines
                for mark in metadata['roadmarks']:
                    if mark['boundary_id'] not in local_boundary_geometry:continue
                    xyz,s,valid=local_boundary_geometry[mark['boundary_id']];lines=visible_lines(xyz,s,mark['s_interval_m'],valid)
                    if not lines:continue
                    record=dict(mark,road_id=rid,section_index=section_index,local_lines=[np.array(p.coords).tolist() for p in lines])
                    mark_records.append(record)
                    if mark['type_id'] in mark_lines:mark_lines[mark['type_id']]+=lines
                    else:unknown_lines+=lines;unsupported.append(record)

        layers=np.stack([_line_mask(mark_lines[k],self.rules.roadmark_interval_thickness_px) for k in ROADMARK_TYPES.values()])
        unknown_mask=_line_mask(unknown_lines,1); count=layers.sum(0)
        type_mask=np.zeros((128,256),dtype=np.uint8)
        for i,k in enumerate(ROADMARK_TYPES.values()):type_mask[layers[i]>0]=k
        type_mask[(count>1)|(unknown_mask>0)]=UNKNOWN
        validity=((count==1)&(unknown_mask==0)).astype(np.uint8)
        crosswalk_mask=np.zeros((128,256),dtype=np.uint8); crosswalk_validity=np.ones((128,256),dtype=np.uint8)
        cross_records=[];object_arrays,object_metadata=self.crosswalks()
        for record in object_metadata['crosswalks']:
            if road_ids is not None and record['road_id'] not in {str(r) for r in road_ids}:continue
            bbox=record.get('bounds_xy')
            if bbox and ((origin<np.array(bbox[0])-radius)|(origin>np.array(bbox[1])+radius)).any():continue
            if record['status']!='supported':
                unsupported.append(record);crosswalk_validity[:]=0;continue
            xyz=object_arrays[record['array']]
            if xyz[:,2].min()>high or xyz[:,2].max()<low:continue
            polygon=Polygon(_local_xy(xyz,pose)).intersection(roi)
            if polygon.is_empty:continue
            if xyz[:,2].min()<low or xyz[:,2].max()>high:
                unsupported.append(dict(record,reason='outline crosses height selection band; partial 3D surface not inferred'));crosswalk_validity[:]=0;continue
            canvas=np.zeros((256,128),dtype=np.uint8)
            for part in _parts(polygon):
                if part.geom_type=='Polygon':cv2.fillPoly(canvas,[_pixels(np.array(part.exterior.coords)).reshape(-1,1,2)],1)
            crosswalk_mask |= np.rot90(canvas)[::-1].copy()
            cross_records.append(dict(record,world_z_range_m=[float(xyz[:,2].min()),float(xyz[:,2].max())]))
        masks={'centerline_geometry_mask':_line_mask(centers,self.rules.boundary_thickness_px),
               'lane_boundary_mask':_line_mask(boundaries,self.rules.boundary_thickness_px),
               'lane_boundary_validity_mask':1-_line_mask(gap_lines,self.rules.boundary_thickness_px),
               'roadmark_type_mask':type_mask,'roadmark_type_layers':layers,'roadmark_validity_mask':validity,
               'roadmark_overlap_mask':(count>1).astype(np.uint8),'roadmark_unknown_mask':unknown_mask,
               'crosswalk_mask':crosswalk_mask,'crosswalk_validity_mask':crosswalk_validity}
        provenance={'schema':SCHEMA,'map_identity':self.identity,'map_source':str(self.map.path),'map_cache_key':self.cache_key,
                    'rear_pose_map_se2':list(pose),'requested_road_ids':None if road_ids is None else sorted(str(v) for v in road_ids),'world_z_reference_m':z,'height_tolerance_m':self.rules.map_height_tolerance_m,
                    'roi':{'forward_m':[0,32],'left_m':[-32,32],'pixel_size_m':.25,'shape':[128,256]},
                    'roadmark_layer_type_ids':list(ROADMARK_TYPES.values()),
                    'query_scope':'complete local neighborhood' if road_ids is None else 'focused road geometry; not full-ROI negative supervision',
                    'roadmark_semantics':'boundary interval classification; not exact paint occupancy; none != unknown; overlapping types kept in independent layers',
                    'left_right_semantics':'geometric increasing-s frame; traffic-relative sides only with source forward/backward direction',
                    'topology_refs':topology_refs,'lanes':lane_records,'boundaries':boundary_records,'roadmarks':mark_records,'crosswalks':cross_records,
                    'unsupported':unsupported,'complete_all_map_semantics_certified':False}
        return {'masks':masks,'provenance':provenance}

    def save_query(self, result, output_dir):
        """Optional compact numerical masks; map geometry remains in map-level caches."""
        identity={'schema':SCHEMA,'map_cache_key':self.cache_key,'query':result['provenance']['rear_pose_map_se2'],
                  'world_z_reference_m':result['provenance']['world_z_reference_m'],'requested_road_ids':result['provenance']['requested_road_ids']}
        path=Path(output_dir)/self.map.sha256/(digest(identity)+'.npz')
        atomic_npz(path,result['masks'],{'identity':identity,'provenance':result['provenance']})
        return path
