"""Causal, history-constrained route progress and isolated NAVSIM preview.

Only the current rear-axle position and actual body yaw are consumed. Route
reconstruction may be offline; that provenance is independent of geometric
validity. This module does not certify reconstructed routes for training.
Disconnected reference segments share an ordering coordinate, not a fictitious
straight-line bridge. A certified directed link permits tracking transitions;
preview may cross it only when the supplied geometry meets at the same point.
"""
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .experimental_route_preview import RoutePolyline, world_to_rear_left
from .route_preview_contract import COMMAND4, COMMAND_ORDER, UNSUPPORTED_UTURN


END_KINDS = ('TASK_END', 'LOCAL_SEGMENT_END', 'MAP_MISSING', 'ROUTE_GAP')


@dataclass(frozen=True)
class ReferenceSegment:
    segment_id: object
    route: RoutePolyline
    route_progress_start_m: float = 0.0
    navigation_source: str = 'RECONSTRUCTED_ROUTE_OFFLINE'
    source_status: str = 'NOT_VERIFIED'
    end_kind: str = 'LOCAL_SEGMENT_END'
    break_after: Optional[str] = None
    directed_next_segment_ids: tuple = ()

    def __post_init__(self):
        if not isinstance(self.route, RoutePolyline):
            raise ValueError('segment route must be a RoutePolyline')
        if not math.isfinite(self.route_progress_start_m) or self.route_progress_start_m < 0:
            raise ValueError('segment progress offset must be finite and nonnegative')
        if self.end_kind not in END_KINDS:
            raise ValueError('end_kind must explicitly describe the segment boundary')
        # Freeze geometry as well as metadata; callers cannot silently replace
        # an expected route while a tracker retains history for the old route.
        frozen = np.array(self.route.points_xy, dtype=np.float64, copy=True)
        frozen.flags.writeable = False
        object.__setattr__(self, 'route', RoutePolyline(frozen, self.route.source, self.route.dense))
        object.__setattr__(self, 'directed_next_segment_ids', tuple(self.directed_next_segment_ids))

    @classmethod
    def from_mapping(cls, row, *, navigation_source=None, source_status=None):
        """Accept an existing ``build_segments`` row without reading frame poses.

        No task endpoint is inferred from the last reconstructed sample. The
        caller must supply ``end_kind='TASK_END'`` using task-route evidence.
        ``trust='HIGH'`` in an offline map reconstruction is not certification.
        """
        source = navigation_source or row.get('navigation_source', 'RECONSTRUCTED_ROUTE_OFFLINE')
        if source == 'RECONSTRUCTED_ROUTE':
            source = 'RECONSTRUCTED_ROUTE_OFFLINE'
        boundary = row.get('break_after')
        kind = row.get('end_kind') or ('ROUTE_GAP' if boundary else 'LOCAL_SEGMENT_END')
        return cls(
            segment_id=row.get('segment_id', 0),
            route=RoutePolyline(row['points_xy'], source=source, dense=True),
            route_progress_start_m=float(row.get('route_progress_start_m', 0.0)),
            navigation_source=source,
            source_status=source_status or row.get('source_status', 'NOT_VERIFIED'),
            end_kind=kind, break_after=boundary,
            directed_next_segment_ids=tuple(row.get('directed_next_segment_ids', ())),
        )


class RouteProgressTracker:
    """Track an ordered route causally, retaining only the last accepted match.

    Distances and body heading gate every candidate, then history restricts arc
    progress to a window derived from observed displacement. Small movements
    are never clamped to zero. Real backwards displacement permits regression;
    ``reverse=True`` also allows a body facing opposite the route direction.
    ``relocalized=True`` explicitly reacquires a match after a relocation. Replanning uses ``set_route``
    and a new revision rather than silently carrying over the old progress.

    ``timestamp`` is optional and only labels motion diagnostics; no assumed
    0.1 s interval or velocity integration affects geometric route progress.
    """

    def __init__(self, segments, *, route_revision='initial', max_projection_m=3.0,
                 max_heading_deg=80.0, continuity_slack_m=1.0,
                 ambiguity_distance_m=0.2, ambiguity_progress_m=2.0):
        values = (max_projection_m, max_heading_deg, continuity_slack_m,
                  ambiguity_distance_m, ambiguity_progress_m)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('finite positive tracker limits required')
        if max_heading_deg > 180:
            raise ValueError('max_heading_deg must not exceed 180')
        self.max_projection_m = float(max_projection_m)
        self.max_heading_deg = float(max_heading_deg)
        self.continuity_slack_m = float(continuity_slack_m)
        self.ambiguity_distance_m = float(ambiguity_distance_m)
        self.ambiguity_progress_m = float(ambiguity_progress_m)
        self._history = None
        self._reset_reason = None
        self.set_route(segments, route_revision=route_revision)

    def set_route(self, segments, *, route_revision):
        """Replace the frozen expected route and explicitly reset match history."""
        if route_revision is None:
            raise ValueError('explicit route_revision required')
        frozen = tuple(item if isinstance(item, ReferenceSegment)
                       else ReferenceSegment.from_mapping(item) for item in segments)
        if len({item.segment_id for item in frozen}) != len(frozen):
            raise ValueError('reference segment ids must be unique and hashable')
        by_id = {item.segment_id: item for item in frozen}
        for left in frozen:
            if any(value not in by_id for value in left.directed_next_segment_ids):
                raise ValueError('directed connection references an absent segment')
        for left, right in zip(frozen, frozen[1:]):
            if right.route_progress_start_m < left.route_progress_start_m + left.route.length_m - 1e-6:
                raise ValueError('ordered segment progress intervals overlap')
        self._reset_reason = 'route_replanned' if self._history is not None else None
        self.segments = frozen
        self._by_id = by_id
        self._geometry = []
        for segment in frozen:
            points = segment.route.points_xy
            vectors = np.diff(points, axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            self._geometry.append((segment, points[:-1], vectors, lengths,
                                   np.concatenate(([0.0], np.cumsum(lengths))),
                                   np.arctan2(vectors[:, 1], vectors[:, 0])))
        self.route_revision = str(route_revision)
        self._history = None

    def _candidates(self, rear, yaw, reverse):
        candidates = []
        for segment, starts, vectors, lengths, cumulative, headings in self._geometry:
            factors = np.clip(np.einsum('ij,ij->i', rear - starts, vectors) / lengths ** 2, 0, 1)
            projected_points = starts + factors[:, None] * vectors
            distances = np.linalg.norm(projected_points - rear, axis=1)
            heading_errors = np.abs(np.degrees((yaw - headings + math.pi) % (2 * math.pi) - math.pi))
            for index in np.flatnonzero(distances <= self.max_projection_m):
                projected = projected_points[index]
                distance = float(distances[index])
                heading_error = float(heading_errors[index])
                matching_heading_error = min(heading_error, 180.0 - heading_error) if reverse else heading_error
                local_s = float(cumulative[index] + factors[index] * lengths[index])
                candidates.append({
                    'segment': segment, 'segment_index': int(index), 'local_s_m': local_s,
                    'route_progress_m': segment.route_progress_start_m + local_s,
                    'projected_xy': projected.tolist(), 'projection_distance_m': distance,
                    'heading_error_deg': heading_error,
                    'matching_heading_error_deg': matching_heading_error,
                    'score': distance + .01 * matching_heading_error,
                })
        return candidates

    def _connected(self, previous, current, backwards):
        if previous == current:
            return True
        # A directed topology certificate, not endpoint proximity, authorizes
        # moving between independently reconstructed reference segments.
        source, target = (current, previous) if backwards else (previous, current)
        pending, visited = [source], set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in visited:
                visited.add(node)
                pending.extend(self._by_id[node].directed_next_segment_ids)
        return False

    def _target(self, segment, local_s, distance):
        """Follow only certified, geometrically complete directed route links."""
        visited = set()
        traversed = []
        while True:
            traversed.append(segment)
            route = segment.route
            remaining = route.length_m - local_s
            if distance <= remaining + 1e-9:
                return route.point_at_distance(min(route.length_m, local_s + distance)), segment, None, traversed
            if segment.segment_id in visited:
                return None, segment, 'AMBIGUOUS_ROUTE_CONNECTION', traversed
            visited.add(segment.segment_id)
            links = segment.directed_next_segment_ids
            if len(links) > 1:
                return None, segment, 'AMBIGUOUS_ROUTE_CONNECTION', traversed
            if not links:
                return None, segment, segment.end_kind, traversed
            following = self._by_id[links[0]]
            # A graph edge alone cannot supply missing centerline geometry.
            if np.linalg.norm(route.points_xy[-1] - following.route.points_xy[0]) > 1e-6:
                return None, segment, 'ROUTE_GAP', traversed
            if abs(following.route_progress_start_m -
                   (segment.route_progress_start_m + route.length_m)) > 1e-6:
                return None, segment, 'ROUTE_GAP', traversed
            distance -= remaining
            segment, local_s = following, 0.0

    def _base(self, validity, reason, match=None):
        result = {
            'status': 'FAIL', 'preview_valid': False, 'preview_validity': validity,
            'failure_reason': reason, 'reason': reason, 'category': None,
            'command4': None, 'command4_order': list(COMMAND_ORDER),
            'v1_compatible': False, 'uturn_status': None,
            'route_progress_valid': match is not None,
            'route_progress_m': None, 'reference_segment_id': None,
            'preview_point_xy': None, 'preview_xy_rear': None, 'local_xy_m': None,
            'preview_lateral_y': None, 'lateral_m': None, 'longitudinal_m': None,
            'projected_xy': None, 'projection_distance_m': None,
            'heading_error_deg': None, 'segment_index': None,
            'preview_m': 20.0, 'lateral_threshold_m': 2.0,
            'navigation_source': None, 'source_status': 'NOT_VERIFIED',
            'route_revision': self.route_revision, 'recovery_reason': None,
            'boundary_margin_m': None, 'boundary_near': False,
        }
        if match is not None:
            segment = match['segment']
            result.update({key: match[key] for key in (
                'route_progress_m', 'segment_index', 'projected_xy',
                'projection_distance_m', 'heading_error_deg')})
            result.update(reference_segment_id=segment.segment_id,
                          navigation_source=segment.navigation_source,
                          source_status=segment.source_status)
        return result

    def update(self, rear_xy, yaw, *, timestamp=None, reverse=False,
               relocalized=False, route_revision=None):
        """Return one preview row from one current pose; never accepts future poses.

        Failed navigation has ``command4=None``. A schema-valid Unknown is a
        separate upstream policy decision, never an implicit geometry fallback.
        """
        rear = np.asarray(rear_xy, dtype=np.float64)
        if rear.shape != (2,) or not np.isfinite(rear).all() or not math.isfinite(yaw):
            return self._base('INVALID_POSE', 'invalid_current_rear_pose')
        if timestamp is not None and not math.isfinite(timestamp):
            return self._base('INVALID_TIME', 'nonfinite_timestamp')
        if route_revision is not None and str(route_revision) != self.route_revision:
            return self._base('ROUTE_REVISION_MISMATCH', 'set_route_required_for_replan')
        if not self.segments:
            return self._base('MAP_MISSING', 'reference_path_missing')
        previous = None if relocalized else self._history
        if previous is not None and timestamp is not None and previous['timestamp'] is not None:
            if timestamp < previous['timestamp']:
                return self._base('INVALID_TIME', 'timestamp_regression')
        displacement = 0.0 if previous is None else float(np.linalg.norm(rear - previous['rear']))
        backwards = bool(reverse)
        if previous is not None:
            forward = np.array([math.cos(yaw), math.sin(yaw)])
            backwards |= float(np.dot(rear - previous['rear'], forward)) < -.05
        candidates = self._candidates(rear, yaw, bool(reverse))
        nearby = [c for c in candidates if c['projection_distance_m'] <= self.max_projection_m]
        if not nearby:
            return self._base('PROJECTION_INVALID', 'projection_distance_exceeded')
        aligned = [c for c in nearby if c['matching_heading_error_deg'] <= self.max_heading_deg]
        if not aligned:
            return self._base('HEADING_INVALID', 'body_heading_mismatch')
        if previous is not None:
            bound = 2.0 * displacement + self.continuity_slack_m
            aligned = [c for c in aligned
                       if abs(c['route_progress_m'] - previous['match']['route_progress_m']) <= bound
                       and self._connected(previous['match']['segment'].segment_id,
                                           c['segment'].segment_id, backwards)]
            if not aligned:
                return self._base('CONTINUITY_REJECTED', 'history_or_directed_topology_conflict')
        aligned.sort(key=lambda c: (c['score'],
                                   abs(c['route_progress_m'] - previous['match']['route_progress_m'])
                                   if previous else c['route_progress_m']))
        match = aligned[0]
        for alternative in aligned[1:]:
            if alternative['score'] - match['score'] > self.ambiguity_distance_m:
                break
            same_boundary = (
                abs(alternative['route_progress_m'] - match['route_progress_m']) <= 1e-6
                and np.linalg.norm(np.asarray(alternative['projected_xy']) - match['projected_xy']) <= 1e-6
                and (self._connected(match['segment'].segment_id, alternative['segment'].segment_id, False)
                     or self._connected(match['segment'].segment_id, alternative['segment'].segment_id, True)))
            if not same_boundary and (
                    abs(alternative['route_progress_m'] - match['route_progress_m']) > self.ambiguity_progress_m
                    or alternative['segment'].segment_id != match['segment'].segment_id):
                # History narrows candidates, but does not certify two still
                # plausible route branches merely because one is centimetres closer.
                return self._base('AMBIGUOUS_MATCH', 'multiple_plausible_route_positions')
        recovery = 'relocalized' if relocalized else self._reset_reason
        self._history = {'rear': rear.copy(), 'match': match, 'timestamp': timestamp}
        self._reset_reason = None
        target, target_segment, failure, traversed = self._target(match['segment'], match['local_s_m'], 20.0)
        result = self._base(failure or 'VALID', failure.lower() if failure else None, match)
        statuses = {segment.source_status for segment in traversed}
        sources = list(dict.fromkeys(segment.navigation_source for segment in traversed))
        result['source_status'] = next(iter(statuses)) if len(statuses) == 1 else 'NOT_VERIFIED'
        result['navigation_sources'] = sources
        if len(sources) > 1:
            result['navigation_source'] = 'MIXED_ROUTE_SOURCES'
        result.update(recovery_reason=recovery, reverse_motion=backwards,
                      displacement_m=displacement,
                      time_source='OBSERVED_TIMESTAMP' if timestamp is not None else 'UNAVAILABLE')
        if failure:
            return result
        local = world_to_rear_left(target, rear, yaw)
        lateral, longitudinal = float(local[1]), float(local[0])
        current_heading = match['segment'].route.heading_rad(match['segment_index'])
        target_route = target_segment.route
        # Resolve a self-intersection by the requested target arc, not a fresh
        # global projection of that target point.
        if target_segment.segment_id == match['segment'].segment_id:
            target_s = min(target_route.length_m, match['local_s_m'] + 20.0)
        else:
            target_s = match['route_progress_m'] + 20.0 - target_segment.route_progress_start_m
        target_index = min(len(target_route.points_xy) - 2,
                           int(np.searchsorted(target_route.cumulative_s, target_s, side='right') - 1))
        delta = (math.degrees(target_route.heading_rad(target_index) - current_heading) + 180) % 360 - 180
        result.update(preview_point_xy=target.tolist(), preview_xy_rear=local.tolist(),
                      local_xy_m=local.tolist(), preview_lateral_y=lateral,
                      lateral_m=lateral, longitudinal_m=longitudinal,
                      heading_delta_deg=delta,
                      boundary_margin_m=abs(abs(lateral) - 2.0),
                      boundary_near=abs(abs(lateral) - 2.0) <= .25)
        if abs(delta) >= 135.0:
            result.update(status=UNSUPPORTED_UTURN, preview_validity=UNSUPPORTED_UTURN,
                          failure_reason='uturn_heading', reason='uturn_heading',
                          uturn_status=UNSUPPORTED_UTURN, category='UTurn')
            return result
        if longitudinal <= 0:
            result.update(preview_validity='TARGET_NOT_FORWARD',
                          failure_reason='target_not_forward', reason='target_not_forward')
            return result
        category = 'Left' if lateral > 2.0 else 'Right' if lateral < -2.0 else 'Straight'
        result.update(status='PASS', preview_valid=True, preview_validity='VALID',
                      category=category, command4=list(COMMAND4[category]), v1_compatible=True,
                      reason={'Left': 'lateral_left_of_threshold', 'Right': 'lateral_right_of_threshold',
                              'Straight': 'inside_lateral_threshold'}[category])
        return result
