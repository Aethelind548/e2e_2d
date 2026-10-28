"""Isolated route-preview geometry for Adapter diagnosis, not production input.

The implementation projects the current rear axle onto an already ordered route
and takes a target by arc length. It deliberately does not infer a route from
future expert trajectories, sparse near/far points or raw RoadOption integers.
The current data set has very limited exact route coverage; callers must report
that provenance rather than treating this helper as a certified navigation
producer.
"""
from dataclasses import dataclass
import math

import numpy as np

from .coordinate_transform import rotate_xy


COMMAND_ORDER = ('Left', 'Straight', 'Right', 'Unknown')


@dataclass(frozen=True)
class RoutePolyline:
    points_xy: np.ndarray
    source: str
    dense: bool = False

    def __post_init__(self):
        p = np.asarray(self.points_xy, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError('route points must be finite (N>=2, 2)')
        if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) <= 1e-9):
            raise ValueError('route contains duplicate adjacent points')
        object.__setattr__(self, 'points_xy', p)

    @classmethod
    def from_waypoints(cls, points, source, dense=False):
        return cls(points, source=source, dense=dense)

    @property
    def segment_lengths(self):
        return np.linalg.norm(np.diff(self.points_xy, axis=0), axis=1)

    @property
    def cumulative_s(self):
        return np.concatenate(([0.0], np.cumsum(self.segment_lengths)))

    @property
    def length_m(self):
        return float(self.cumulative_s[-1])

    def heading_rad(self, index):
        d = self.points_xy[index + 1] - self.points_xy[index]
        return math.atan2(float(d[1]), float(d[0]))

    def project(self, point_xy):
        """Return (arc_length, distance, projected_xy, segment_index).

        Projection is vectorized over the finite set of route segments.  The
        previous Python edge loop made long but valid centerline references
        dominate generation time.
        """
        point = np.asarray(point_xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('finite 2-D point required')
        starts = self.points_xy[:-1]
        vectors = self.points_xy[1:] - starts
        length_squared = np.einsum('ij,ij->i', vectors, vectors)
        factors = np.einsum('ij,ij->i', point - starts, vectors) / length_squared
        factors = np.clip(factors, 0.0, 1.0)
        projected = starts + factors[:, None] * vectors
        distances = np.linalg.norm(projected - point, axis=1)
        index = int(np.argmin(distances))
        cumulative = self.cumulative_s
        arc_length = float(cumulative[index] +
                           factors[index] * math.sqrt(length_squared[index]))
        return arc_length, float(distances[index]), projected[index], index

    def point_at_distance(self, distance_m):
        if not (math.isfinite(distance_m) and distance_m >= 0):
            raise ValueError('finite nonnegative distance required')
        if distance_m > self.length_m:
            return None
        i = int(np.searchsorted(self.cumulative_s, distance_m, side='right') - 1)
        i = min(i, len(self.points_xy) - 2)
        segment = self.segment_lengths[i]
        t = (distance_m - self.cumulative_s[i]) / segment
        return self.points_xy[i] + t * (self.points_xy[i + 1] - self.points_xy[i])

    def self_intersections(self):
        """Count non-adjacent segment intersections; useful for route audits."""
        segments = list(zip(self.points_xy[:-1], self.points_xy[1:]))
        count = 0
        for i in range(len(segments)):
            for j in range(i + 2, len(segments)):
                if i == 0 and j == len(segments) - 1:
                    continue  # closed loop endpoint is not a self-crossing
                if _segments_intersect(segments[i][0], segments[i][1],
                                       segments[j][0], segments[j][1]):
                    count += 1
        return count

    def summary(self):
        turn = []
        for i in range(len(self.points_xy) - 2):
            h1 = self.heading_rad(i)
            h2 = self.heading_rad(i + 1)
            turn.append(abs(math.degrees((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)))
        return {
            'source': self.source,
            'dense': self.dense,
            'point_count': int(len(self.points_xy)),
            'length_m': round(self.length_m, 3),
            'mean_spacing_m': round(float(self.segment_lengths.mean()), 4),
            'max_spacing_m': round(float(self.segment_lengths.max()), 4),
            'max_turn_deg': round(max(turn), 3) if turn else 0.0,
            'turns_over_5deg': int(sum(v > 5 for v in turn)),
            'self_intersections': int(self.self_intersections()),
        }


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _segments_intersect(a, b, c, d):
    r = b - a
    s = d - c
    denom = _cross(r, s)
    if abs(denom) < 1e-12:
        return False
    t = _cross(c - a, s) / denom
    u = _cross(c - a, r) / denom
    return 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9


def world_to_rear_left(point_xy, rear_xy, yaw):
    x, right_y = rotate_xy((float(point_xy[0]) - float(rear_xy[0]),
                            float(point_xy[1]) - float(rear_xy[1])), -float(yaw))
    return np.asarray([x, -right_y], dtype=np.float64)


def bearing_command(point_xy, rear_xy, yaw, threshold_deg=10.0):
    local = world_to_rear_left(point_xy, rear_xy, yaw)
    distance = float(np.linalg.norm(local))
    bearing = math.degrees(math.atan2(float(local[1]), float(local[0])))
    if distance <= 1e-9 or local[0] <= 0:
        category = 3
    else:
        category = 0 if bearing > threshold_deg else 2 if bearing < -threshold_deg else 1
    return {
        'category': category,
        'category_name': COMMAND_ORDER[category],
        'local_xy_m': local.tolist(),
        'distance_m': distance,
        'bearing_deg': bearing,
    }


def preview_command(route, rear_xy, yaw, preview_m=20.0, lateral_threshold_m=2.0):
    """Project, take an arc-length lookahead target, and classify the command."""
    if not (math.isfinite(preview_m) and preview_m > 0):
        raise ValueError('positive finite preview_m required')
    if not (math.isfinite(lateral_threshold_m) and lateral_threshold_m > 0):
        raise ValueError('positive finite lateral_threshold_m required')
    s0, projection_distance, projected, segment = route.project(rear_xy)
    s_target = s0 + float(preview_m)
    target = route.point_at_distance(s_target)
    if target is None:
        return {
            'category': 3, 'category_name': COMMAND_ORDER[3], 'reason': 'route_end_before_preview',
            's0_m': s0, 'projection_distance_m': projection_distance, 'projected_xy': projected.tolist(),
            'segment_index': int(segment), 'preview_m': float(preview_m), 'target_xy': None,
        }
    local = world_to_rear_left(target, rear_xy, yaw)
    lateral = float(local[1])
    longitudinal = float(local[0])
    if longitudinal <= 0:
        category = 3
        reason = 'target_not_forward'
    elif lateral > lateral_threshold_m:
        category = 0
        reason = 'lateral_left_of_threshold'
    elif lateral < -lateral_threshold_m:
        category = 2
        reason = 'lateral_right_of_threshold'
    else:
        category = 1
        reason = 'inside_lateral_threshold'
    return {
        'category': category, 'category_name': COMMAND_ORDER[category], 'reason': reason,
        's0_m': s0, 'projection_distance_m': projection_distance, 'projected_xy': projected.tolist(),
        'segment_index': int(segment), 'preview_m': float(preview_m),
        'target_xy': np.asarray(target).tolist(), 'local_xy_m': local.tolist(),
        'lateral_m': lateral, 'longitudinal_m': longitudinal,
        'lateral_threshold_m': float(lateral_threshold_m),
    }
