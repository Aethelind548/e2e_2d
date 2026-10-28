"""Online navigation: dense ``set_global_plan`` + shared 20 m Route Preview.

The retired near-bearing policy is gone.  ``RouteManager`` now owns exactly one
status builder; ``update`` runs the frozen calculator on the dense world plan and
returns a ``NavigationState`` carrying the explicit Status8 validity.

The leaderboard still calls ``set_global_plan(gps_route, route)``; the signature
and lifecycle are unchanged, but only the dense world route is used for
navigation.  GPS points are validated for alignment and recorded for provenance,
never used for command generation.
"""
import math
from typing import Any, Optional, Sequence

from agent.common.errors import NavigationError
from agent.common.types import CoordinateFrame, EgoState, GlobalRoute, NavigationState, Pose2D, Timestamp
from agent.config.settings import NavigationConfig
from .status_builder import STATUS_POLICY, OnlineStatusBuilder


class RouteManager:
    def __init__(self, config: NavigationConfig):
        self.config = config
        self.route: Optional[GlobalRoute] = None
        self.generation = 0
        self.builder = OnlineStatusBuilder(config)

    def set_global_plan(self, gps_plan: Sequence[Any], world_plan: Sequence[Any]) -> None:
        if not gps_plan or not world_plan or len(gps_plan) != len(world_plan):
            raise NavigationError("Nonempty aligned GPS and world routes required")
        try:
            for gps_pair in gps_plan:
                gps, _ = gps_pair
                if not all(math.isfinite(gps[k]) for k in ('lat', 'lon')):
                    raise ValueError('finite GPS route points required')
                if not -90 < gps['lat'] < 90 or not -180 <= gps['lon'] <= 180:
                    raise ValueError('GPS route coordinates out of range')
        except (KeyError, TypeError, ValueError) as exc:
            raise NavigationError(f"Invalid global route: {exc}") from exc
        # Validate and commit the dense route atomically; a bad replacement
        # leaves the previous plan and motion history untouched.
        self.builder.set_global_plan(world_plan, gps_plan)
        self.route = GlobalRoute(tuple(gps_plan), tuple(world_plan))
        self.generation = self.builder.generation

    def update(self, ego: EgoState, timestamp: Timestamp) -> NavigationState:
        if self.route is None:
            raise NavigationError("Global plan has not been supplied")
        status = self.builder.update(ego, timestamp)
        target = None
        point = status.preview.get('preview_point_xy')
        if status.valid and point is not None:
            target = Pose2D(float(point[0]), float(point[1]), ego.pose.heading_rad,
                            CoordinateFrame.CARLA_WORLD, 'route_preview_20m_point')
        return NavigationState(
            timestamp, target, raw_command=None, command4=status.command4,
            policy=STATUS_POLICY, near_index=None, route_generation=status.route_generation,
            status8=status.status8, status_validity=status.validity,
            status_reason=status.reason, motion4=status.motion4,
            route_sha256=status.route_sha256,
            provenance={
                'navigation_policy': STATUS_POLICY,
                'route_source': 'dense_set_global_plan_world_coords',
                'sparse_plan_used': False,
                'future_gt_used': False,
                'raw_command_used_for_encoding': False,
                'road_option_available': False,
                'route_points': status.route_point_count,
                'route_arc_m': status.route_arc_m,
                'preview': {k: status.preview.get(k) for k in
                            ('status', 'reason', 'category', 'preview_point_xy',
                             'route_progress_m', 'projection_distance_m',
                             'lateral_m', 'longitudinal_m', 'heading_delta_deg')},
                'motion4': status.motion4,
                'motion_reason': status.motion_reason,
                'motion_meta': status.provenance.get('motion_meta'),
                'status8_validity': status.validity,
            })

    def preview(self, ego: EgoState, previous_progress_m: Optional[float] = None):
        """Preview-only helper for audits; it never feeds the model Status."""
        if self.route is None:
            raise NavigationError("Global plan has not been supplied")
        rear = ego.pose
        if (rear is None or rear.frame != CoordinateFrame.CARLA_WORLD
                or not all(math.isfinite(v) for v in (rear.x_m, rear.y_m, rear.heading_rad))):
            raise NavigationError("Current world rear pose required for Route Preview")
        from .route_preview import route_preview_from_world_nodes
        return route_preview_from_world_nodes(
            self.builder.route['points'], (rear.x_m, rear.y_m), rear.heading_rad,
            route_generation=self.generation, previous_progress_m=previous_progress_m,
            preview_m=self.config.preview_m,
            lateral_threshold_m=self.config.lateral_threshold_m,
            max_projection_m=self.config.max_projection_m)

    def reset(self) -> None:
        self.route = None
        self.generation = 0
        self.builder.reset()
