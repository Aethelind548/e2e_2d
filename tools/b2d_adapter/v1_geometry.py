"""Bounded M4 geometry: CARLA box vertices, supported road heights, vertical rails."""
import itertools
import math

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon, LineString

from .coordinate_transform import world_to_ego
from .errors import AdapterError


def carla_rotation(rotation):
    """CARLA 0.9.15 Transform::GetMatrix, input B2D [pitch, roll, yaw] degrees."""
    p, r, y = np.radians(rotation)
    cp, sp, cr, sr, cy, sy = np.cos(p), np.sin(p), np.cos(r), np.sin(r), np.cos(y), np.sin(y)
    return np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                     [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                     [sp, -cp*sr, cp*cr]])


def actor_vertices(box):
    try:
        center, extent, rotation = (np.asarray(box[k], dtype=float) for k in ('center', 'extent', 'rotation'))
        if (any(v.shape != (3,) for v in (center, extent, rotation))
                or not np.isfinite(np.r_[center, extent, rotation]).all() or (extent <= 0).any()):
            raise ValueError('finite center/rotation and positive extent3 required')
        corners = np.array(list(itertools.product((-1, 1), repeat=3))) * extent
        return corners @ carla_rotation(rotation).T + center
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterError('invalid_bev_actor', str(exc)) from exc


def actor_local_hull(box, raw, geometry):
    vertices = actor_vertices(box)
    xy = np.array([world_to_ego(p[:2], raw.actor_pose) for p in vertices])
    xy[:, 0] -= geometry.axle_dx_m
    xy[:, 1] = -(xy[:, 1] - geometry.axle_dy_m)
    hull = MultiPoint(xy).convex_hull
    if not isinstance(hull, Polygon) or not hull.is_valid or hull.area <= 0:
        raise AdapterError('invalid_bev_actor', 'degenerate projected 3D box')
    return hull, vertices


def road_support(lanes, xy, rules):
    """Heights only from already selected same-ego-layer driving lane samples.

    Ambiguous junction/grade matches keep actors; no extrapolation from ego z.
    """
    candidates = []
    for lane in lanes:
        if lane['type'] not in ('driving', 'bidirectional') or 'surface_xyz' not in lane:
            continue
        xyz = lane['surface_xyz']
        distance = np.linalg.norm(xyz[:, :2] - xy, axis=1)
        k = int(distance.argmin())
        if distance[k] <= rules.road_support_distance_m:
            candidates.append((float(distance[k]), float(xyz[k, 2])))
    if not candidates:
        return {'status': 'no_nearby_selected_road'}
    zs = [v[1] for v in candidates]
    if max(zs) - min(zs) > rules.road_support_z_spread_m:
        return {'status': 'ambiguous_road_height', 'road_z_range_m': [min(zs), max(zs)]}
    return {'status': 'supported', 'road_z_range_m': [min(zs), max(zs)],
            'nearest_xy_distance_m': min(v[0] for v in candidates)}


def actor_floor_diagnostic(box, vertices, lanes, rules):
    # OpenDRIVE XY reflects CARLA y; height is world z, never LiDAR-shifted z.
    support = road_support(lanes, (float(box['center'][0]), -float(box['center'][1])), rules)
    lo, hi = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    result = dict(support, actor_id=str(box.get('id')), world_z_interval_m=[lo, hi], excluded=False)
    if support['status'] == 'supported':
        a, b = support['road_z_range_m']
        separated = lo > b + rules.actor_deck_clearance_m or hi < a - rules.actor_deck_clearance_m
        result['separated_from_selected_road'] = separated
        result['excluded'] = separated and rules.actor_floor == 'filter-supported-road'
    return result


def flat_guardrail_surface(road, obj, reference_line, polynomial):
    """Convert only unbanked, zero-grade insertion frames and zero object tilt.

    Observed CARLA GuardRail outlines describe open longitudinal vertical faces.
    Their length/width are envelope dimensions, not material thickness. Never
    close the point chain, fill its interior, or buffer it into an invented solid.
    """
    if obj.get('type') != 'barrier' or obj.get('name') != 'GuardRail':
        raise AdapterError('unsupported_guardrail', 'requires barrier type and GuardRail source family')
    try:
        s, t, zoff, heading = (float(obj.get(k)) for k in ('s', 't', 'zOffset', 'hdg'))
        tilt = [float(obj.get(k, 0)) for k in ('pitch', 'roll')]
        corners = obj.findall('outline/cornerLocal')
        if obj.find('repeat') is not None or len(obj.findall('outline')) != 1 or len(corners) < 2:
            raise ValueError('only one non-repeated cornerLocal chain supported')
        if any(c.tag != 'cornerLocal' for c in obj.find('outline')):
            raise ValueError('mixed/unsupported outline corner coordinate systems')
        extra_profiles=road.findall('lateralProfile/shape')+road.findall('lateralProfile/crossfall')
        extra_coefficients=[float(record.get(k,0)) for record in extra_profiles for k in ('a','b','c','d')]
        if any(not math.isfinite(v) or abs(v)>1e-8 for v in extra_coefficients):
            raise ValueError('nonzero additional lateral road surface profiles not verified')
        local = np.array([[float(c.get(k)) for k in ('u', 'v', 'z', 'height')] for c in corners])
        if not np.isfinite(np.r_[s, t, zoff, heading, tilt, local.ravel()]).all() or (local[:, 3] <= 0).any():
            raise ValueError('finite coordinates and positive per-corner heights required')
        recs = road.findall('elevationProfile/elevation')
        grade = 0.
        if recs:
            covering = [r for r in recs if float(r.get('s')) <= s]
            if not covering: raise ValueError('missing insertion elevation interval')
            rec = covering[-1]; d = s - float(rec.get('s'))
            grade = float(rec.get('b', 0)) + 2*float(rec.get('c', 0))*d + 3*float(rec.get('d', 0))*d*d
        roll = float(polynomial(road.findall('lateralProfile/superelevation'), np.array([s]))[0])
        if max(abs(grade), abs(roll), *map(abs, tilt)) > 1e-8:
            raise ValueError('nonflat insertion frame/object tilt: exporter frame equivalence not verified')
        if not 0 <= s <= float(road.get('length')):
            raise ValueError('object insertion outside road length')
        ref = reference_line(road, np.array([s]))[0]
        pivot = ref[:2] + t*np.array([-math.sin(ref[2]), math.cos(ref[2])])
        h = ref[2] + heading
        rotation = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
        xy = local[:, :2] @ rotation.T + pivot
        z = float(polynomial(recs, np.array([s]))[0]) + zoff + local[:, 2]
        return np.column_stack((xy, z, z + local[:, 3]))
    except (ValueError, TypeError) as exc:
        raise AdapterError('unsupported_guardrail', str(exc)) from exc


def clip_vertical_segment(a, b, low, high):
    """Clip segment parameter where vertical surface [bottom,top] meets z band."""
    begin, end = 0., 1.
    for value, delta, limit, greater in ((a[2], b[2]-a[2], high, False),
                                         (a[3], b[3]-a[3], low, True)):
        if abs(delta) < 1e-12:
            if (value < limit if greater else value > limit): return None
            continue
        cross = (limit-value)/delta
        lower_bound = (delta > 0) == greater
        if lower_bound: begin = max(begin, cross)
        else: end = min(end, cross)
    if end < begin: return None
    return a[:2] + begin*(b[:2]-a[:2]), a[:2] + end*(b[:2]-a[:2])


def polygon_mask(official, polygons):
    import cv2
    mask = np.zeros(official._config.bev_semantic_frame[::-1], dtype=np.uint8)
    for polygon in polygons:
        coords = np.array(polygon.exterior.coords).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [official._coords_to_pixel(coords)], 255)
    return np.rot90(mask)[::-1] > 0


def enriched_labels(official, annotations, map_view, pose, actor_polygons, rail_lines):
    """Official masks/order; replace only tilted actor geometry, add class-4 surfaces."""
    import torch
    import cv2
    from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
    labels = np.zeros(official._config.bev_semantic_frame, dtype=np.int64)
    for label, (kind, layers) in official._config.bev_semantic_classes.items():
        if kind == 'polygon': mask = official._compute_map_polygon_mask(map_view, pose, layers)
        elif kind == 'linestring': mask = official._compute_map_linestring_mask(map_view, pose, layers)
        else:
            mask = official._compute_box_mask(annotations, layers)
            mask |= polygon_mask(official, [p for name, p in actor_polygons if tracked_object_types[name] in layers])
        if label == 4:
            surface_mask = np.zeros(official._config.bev_semantic_frame[::-1], dtype=np.uint8)
            for line in rail_lines:
                local = official._geometry_local_coords(line, pose)
                pixels = official._coords_to_pixel(np.array(local.coords).reshape(-1, 1, 2))
                cv2.polylines(surface_mask, [pixels], False, 255, thickness=1)
            mask |= np.rot90(surface_mask)[::-1] > 0
        labels[mask] = label
    return torch.from_numpy(labels)


def offset_geometry_diagnostic(road, s, inner, outer):
    """Planar offset Jacobian 1-curvature*t exposes folded lane coordinates."""
    geometries = road.findall('planView/geometry')
    starts = np.array([float(g.get('s')) for g in geometries])
    indices = np.searchsorted(starts, s, side='right')-1
    curvature = np.zeros_like(s)
    for i in np.unique(indices):
        primitive = list(geometries[i])[0]; mask = indices == i
        if primitive.tag == 'arc': curvature[mask] = float(primitive.get('curvature'))
        elif primitive.tag == 'spiral':
            k0, k1 = float(primitive.get('curvStart')), float(primitive.get('curvEnd'))
            curvature[mask] = k0 + (k1-k0)*(s[mask]-starts[i])/float(geometries[i].get('length'))
    jacobian = np.minimum(1-curvature*inner, 1-curvature*outer)
    k = int(jacobian.argmin())
    return {'minimum_offset_jacobian': float(jacobian[k]), 'at_s_m': float(s[k]),
            'folded_source_coordinates': bool(jacobian[k] <= 0),
            'policy': 'reject invalid polygon; no buffer repair or region removal'}
