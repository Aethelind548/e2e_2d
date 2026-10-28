"""Shared Route Preview -> V1 Status contract.

The same pure functions are intended for:
  * offline training-data generation from a reconstructed ordered route;
  * CARLA Agent evaluation from an ordered ``set_global_plan`` route.

The calculator uses the rear-axle reference frame, an arc-length lookahead of
20 m, and the project's left-positive convention.  It never reads a future ego
pose, a near/far annotation, or a raw CARLA RoadOption integer.

V1 command order is fixed:
    [Left, Straight, Right, Unknown]

A U-Turn is reported as ``UNSUPPORTED_UTURN``.  It is not silently mapped to a
certified category; the V1 vector keeps an explicit Unknown slot only so that
the row remains schema-valid, while ``preview_valid`` and ``status8_valid`` are
false.
"""
from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np

from .experimental_route_preview import RoutePolyline
from .experimental_route_preview import world_to_rear_left


COMMAND_ORDER = ('Left', 'Straight', 'Right', 'Unknown')
COMMAND4 = {
    'Left': (1.0, 0.0, 0.0, 0.0),
    'Straight': (0.0, 1.0, 0.0, 0.0),
    'Right': (0.0, 0.0, 1.0, 0.0),
    'Unknown': (0.0, 0.0, 0.0, 1.0),
}
UNSUPPORTED_UTURN = 'UNSUPPORTED_UTURN'


@dataclass(frozen=True)
class RouteProgress:
    status: str
    valid: bool
    reason: Optional[str]
    route_progress_m: float
    progress_fraction: float
    projection_distance_m: float
    projected_xy: Optional[tuple]
    segment_index: Optional[int]
    route_length_m: float

    def as_dict(self):
        return {
            'status': self.status,
            'valid': self.valid,
            'reason': self.reason,
            'route_progress_m': self.route_progress_m,
            'progress_fraction': self.progress_fraction,
            'projection_distance_m': self.projection_distance_m,
            'projected_xy': list(self.projected_xy) if self.projected_xy is not None else None,
            'segment_index': self.segment_index,
            'route_length_m': self.route_length_m,
        }


def _route(route):
    if isinstance(route, RoutePolyline):
        return route
    return RoutePolyline.from_waypoints(route, source='route_preview_contract', dense=False)


def route_progress(route, rear_xy, previous_progress_m=None,
                   max_projection_m=3.0, max_backtrack_m=1.0):
    """Project the rear axle and return an explicit route-progress state.

    ``previous_progress_m`` is optional and is intended for the online Agent.
    It rejects a large backward jump caused by a self-intersection or a route
    replacement; it does not smooth or alter the measured projection.
    """
    if not (math.isfinite(max_projection_m) and max_projection_m > 0):
        raise ValueError('positive finite max_projection_m required')
    if not (math.isfinite(max_backtrack_m) and max_backtrack_m >= 0):
        raise ValueError('finite nonnegative max_backtrack_m required')
    route = _route(route)
    s0, projection, projected, segment = route.project(rear_xy)
    valid = True
    reason = None
    if not math.isfinite(projection) or projection > max_projection_m:
        valid = False
        reason = 'projection_distance_exceeded'
    elif previous_progress_m is not None and s0 + max_backtrack_m < float(previous_progress_m):
        valid = False
        reason = 'progress_regression'
    return RouteProgress(
        status='PASS' if valid else 'FAIL', valid=valid, reason=reason,
        route_progress_m=float(s0),
        progress_fraction=float(s0 / route.length_m) if route.length_m > 0 else 0.0,
        projection_distance_m=float(projection),
        projected_xy=tuple(float(v) for v in projected),
        segment_index=int(segment), route_length_m=float(route.length_m),
    )


def _heading_at(route, s_m, epsilon_m=0.5):
    p0 = route.point_at_distance(s_m)
    if p0 is None:
        return None
    p1 = route.point_at_distance(min(route.length_m, s_m + epsilon_m))
    if p1 is None or float(np.linalg.norm(p1 - p0)) <= 1e-9:
        return None
    return math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))


def _failure(reason, progress, preview_m, category='Unknown', uturn=False):
    return {
        'status': 'FAIL' if not uturn else UNSUPPORTED_UTURN,
        'preview_valid': False,
        'reason': reason,
        'category': category,
        'command4': list(COMMAND4['Unknown']),
        'command4_order': list(COMMAND_ORDER),
        'v1_compatible': False,
        'uturn_status': UNSUPPORTED_UTURN if uturn else None,
        'preview_point_xy': None,
        'local_xy_m': None,
        'lateral_m': None,
        'longitudinal_m': None,
        'heading_delta_deg': None,
        'preview_m': float(preview_m),
        'route_progress': progress.as_dict(),
        'route_progress_m': progress.route_progress_m,
        'progress_fraction': progress.progress_fraction,
        'route_length_m': progress.route_length_m,
        'projection_distance_m': progress.projection_distance_m,
        'segment_index': progress.segment_index,
        'source_status': 'INVALID',
    }


def preview_command4(route, rear_xy, yaw, preview_m=20.0,
                     lateral_threshold_m=2.0, previous_progress_m=None,
                     max_projection_m=3.0):
    """Return a 20 m preview point and a V1 command4.

    The output separates geometric validity from the command value.  Invalid
    rows still carry the Unknown vector for schema stability but must not be
    counted as valid coverage or used as a training Status.
    """
    if not (math.isfinite(preview_m) and preview_m > 0):
        raise ValueError('positive finite preview_m required')
    if not (math.isfinite(lateral_threshold_m) and lateral_threshold_m > 0):
        raise ValueError('positive finite lateral_threshold_m required')
    route = _route(route)
    progress = route_progress(route, rear_xy, previous_progress_m=previous_progress_m,
                              max_projection_m=max_projection_m)
    if not progress.valid:
        return _failure(progress.reason or 'projection_invalid', progress, preview_m)
    s0 = progress.route_progress_m
    target = route.point_at_distance(s0 + float(preview_m))
    if target is None:
        return _failure('route_end_before_preview', progress, preview_m)
    current_heading = _heading_at(route, s0)
    target_heading = _heading_at(route, s0 + float(preview_m))
    if current_heading is None or target_heading is None:
        return _failure('insufficient_tangent', progress, preview_m)
    delta = math.degrees((target_heading - current_heading + math.pi) % (2 * math.pi) - math.pi)
    local = world_to_rear_left(target, rear_xy, yaw)
    lateral = float(local[1])
    longitudinal = float(local[0])
    if abs(delta) >= 135.0:
        result = _failure('uturn_heading', progress, preview_m,
                          category='UTurn', uturn=True)
        result.update({
            'preview_point_xy': [float(v) for v in target],
            'local_xy_m': [float(v) for v in local],
            'lateral_m': lateral, 'longitudinal_m': longitudinal,
            'heading_delta_deg': delta, 'uturn': True,
            'source_status': 'UNSUPPORTED_UTURN',
        })
        return result
    if longitudinal <= 0.0:
        category, reason = 'Unknown', 'target_not_forward'
    elif lateral > float(lateral_threshold_m):
        category, reason = 'Left', 'lateral_left_of_threshold'
    elif lateral < -float(lateral_threshold_m):
        category, reason = 'Right', 'lateral_right_of_threshold'
    else:
        category, reason = 'Straight', 'inside_lateral_threshold'
    valid = category != 'Unknown'
    return {
        'status': 'PASS' if valid else 'FAIL',
        'preview_valid': valid,
        'reason': reason,
        'category': category,
        'command4': list(COMMAND4[category]),
        'command4_order': list(COMMAND_ORDER),
        'v1_compatible': valid,
        'uturn_status': None,
        'uturn': False,
        'preview_point_xy': [float(v) for v in target],
        'local_xy_m': [float(v) for v in local],
        'lateral_m': lateral,
        'longitudinal_m': longitudinal,
        'heading_delta_deg': delta,
        'preview_m': float(preview_m),
        'lateral_threshold_m': float(lateral_threshold_m),
        'route_progress': progress.as_dict(),
        'route_progress_m': progress.route_progress_m,
        'progress_fraction': progress.progress_fraction,
        'route_length_m': progress.route_length_m,
        'projection_distance_m': progress.projection_distance_m,
        'segment_index': progress.segment_index,
        'source_status': 'VALID' if valid else 'INVALID',
    }


DEFAULT_RECOVERY_HORIZONS = (20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0)


def stable_preview_command4(route, rear_xy, yaw, *,
                            horizons=DEFAULT_RECOVERY_HORIZONS,
                            min_preview_m=5.0, lateral_threshold_m=2.0,
                            max_projection_m=3.0):
    """Evaluate a route at several lookaheads and require a stable command.

    This is the V3 navigation-recovery primitive.  It does not invent a route,
    change the 20 m contract, or turn an invalid/U-turn preview into a command.
    The longest valid horizon is selected only when every valid horizon at or
    below it agrees on the same non-Unknown command and at least one shorter
    horizon corroborates it.  Callers decide whether a shorter horizon is
    admissible (only real route-end recovery may use it).
    """
    values = tuple(float(h) for h in horizons)
    if not values or any(not math.isfinite(h) or h <= 0 for h in values):
        raise ValueError('positive finite horizons required')
    if len(set(values)) != len(values):
        raise ValueError('horizons must be unique')
    values = tuple(sorted(values, reverse=True))
    if not (math.isfinite(min_preview_m) and min_preview_m > 0):
        raise ValueError('positive finite min_preview_m required')
    results = []
    for horizon in values:
        preview = preview_command4(
            route, rear_xy, yaw, preview_m=horizon,
            lateral_threshold_m=lateral_threshold_m,
            max_projection_m=max_projection_m)
        results.append(preview)
    valid = [item for item in results
             if item.get('preview_valid')
             and item.get('category') in ('Left', 'Straight', 'Right')
             and float(item.get('projection_distance_m', math.inf)) <= max_projection_m]
    if not valid:
        return {
            'status': 'FAIL', 'preview_valid': False,
            'reason': 'no_valid_horizon', 'stable': False,
            'selected': None, 'horizons': results,
            'valid_horizons': [],
            'min_preview_m': float(min_preview_m),
        }
    selected = max(valid, key=lambda item: float(item['preview_m']))
    category = selected['category']
    lower = [item for item in valid
             if float(item['preview_m']) <= float(selected['preview_m'])]
    disagree = [item for item in lower if item['category'] != category]
    same = [item for item in lower if item['category'] == category]
    # A shorter valid preview may not hide a longer-horizon U-turn, an
    # unknown/backward target, or a projection failure.  The only admissible
    # longer-horizon failure is a genuine route end, which is the one case
    # where shortening is allowed by the V3 contract.
    hidden_longer = [
        item for item in results
        if float(item['preview_m']) > float(selected['preview_m'])
        and item.get('reason') != 'route_end_before_preview'
    ]
    stable = bool(
        float(selected['preview_m']) >= float(min_preview_m)
        and len(same) >= 2
        and min(float(item['preview_m']) for item in same) < float(selected['preview_m'])
        and not disagree
        and not hidden_longer)
    if not stable:
        if float(selected['preview_m']) < float(min_preview_m):
            reason = 'route_remaining_too_short'
        elif hidden_longer:
            reason = 'longer_horizon_not_route_end'
        else:
            reason = 'command_not_stable_across_horizons'
    else:
        reason = 'stable_command'
    return {
        'status': 'PASS' if stable else 'FAIL',
        'preview_valid': stable,
        'reason': reason,
        'stable': stable,
        'selected': selected,
        'horizons': results,
        'valid_horizons': [float(item['preview_m']) for item in valid],
        'same_category_horizons': [float(item['preview_m']) for item in same],
        'disagreeing_horizons': [float(item['preview_m']) for item in disagree],
        'hidden_longer_horizons': [float(item['preview_m']) for item in hidden_longer],
        'hidden_longer_reasons': [item.get('reason') for item in hidden_longer],
        'min_preview_m': float(min_preview_m),
    }


def validate_command4(command4):
    if len(command4) != 4:
        raise ValueError('command4 must have four elements')
    values = [float(v) for v in command4]
    if not all(math.isfinite(v) and v in (0.0, 1.0) for v in values):
        raise ValueError('command4 must be finite binary values')
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError('command4 must be one-hot')
    return tuple(values)


def status8(command4, motion4):
    """Return ``concat(command4, original_motion4)``.

    ``motion4`` is the existing A0 baseline [vx, vy, ax, ay].  This function
    does not alter or re-estimate it.
    """
    command = validate_command4(command4)
    motion = tuple(float(v) for v in motion4)
    if len(motion) != 4 or not all(math.isfinite(v) for v in motion):
        raise ValueError('motion4 must be four finite values')
    return command + motion


def status8_from_preview(preview, motion4):
    """Build status8 only when the preview is a valid V1 command."""
    if not isinstance(preview, dict) or not preview.get('preview_valid'):
        raise ValueError('valid preview required for status8')
    if preview.get('uturn_status') == UNSUPPORTED_UTURN:
        raise ValueError('UNSUPPORTED_UTURN cannot be converted to status8')
    return status8(preview.get('command4'), motion4)
