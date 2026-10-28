"""Reuse offline planar math only when source frame and origin are explicit."""

import math
from typing import Tuple

from agent.common.errors import TrajectoryError
from agent.common.types import CoordinateFrame, Pose2D


def carla_world_to_model_xy(world_xy: Tuple[float, float], rear_origin: Pose2D
                            ) -> Tuple[float, float]:
    """World metres -> rear-relative x forward/y left; no inferred axle offset."""
    if rear_origin.frame != CoordinateFrame.CARLA_WORLD or not rear_origin.reference:
        raise TrajectoryError("An explicit rear origin in CARLA world coordinates is required")
    if not all(math.isfinite(v) for v in (*world_xy, rear_origin.x_m, rear_origin.y_m, rear_origin.heading_rad)):
        raise TrajectoryError("Nonfinite coordinate transform input")
    from tools.b2d_adapter.coordinate_transform import ActorPose, world_to_ego
    origin = ActorPose(rear_origin.x_m, rear_origin.y_m, rear_origin.heading_rad)
    x, right_y = world_to_ego(world_xy, origin)
    return x, -right_y


def model_xy_to_carla_world(local_xy: Tuple[float, float], rear_origin: Pose2D
                            ) -> Tuple[float, float]:
    """Inverse of carla_world_to_model_xy with the same supplied rear origin."""
    if rear_origin.frame != CoordinateFrame.CARLA_WORLD or not rear_origin.reference:
        raise TrajectoryError("An explicit rear origin in CARLA world coordinates is required")
    if not all(math.isfinite(v) for v in (*local_xy, rear_origin.x_m, rear_origin.y_m, rear_origin.heading_rad)):
        raise TrajectoryError("Nonfinite coordinate transform input")
    from tools.b2d_adapter.coordinate_transform import ActorPose, ego_to_world
    return ego_to_world((local_xy[0], -local_xy[1]),
                        ActorPose(rear_origin.x_m, rear_origin.y_m, rear_origin.heading_rad))
