"""Column-vector planar transforms from coordinate_validation.md §1.

CARLA world/Actor: x forward, y right, positive yaw turns right.
R(yaw) maps Actor local XY to CARLA world XY. Target local x forward,
y left, heading positive left. This is an axis reflection, not a yaw offset.
No use of top-level x/y/theta or bounding-box center for pose origins.
"""

from dataclasses import dataclass
import math

from .errors import AdapterError


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rotate_xy(xy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = xy
    return (c * x - s * y, s * x + c * y)


@dataclass(frozen=True)
class ActorPose:
    x: float
    y: float
    yaw: float  # radians, from ego bounding box rotation[2] degrees

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.yaw)):
            raise AdapterError("invalid_pose", "pose contains nonfinite values")


def actor_pose(ego_box):
    try:
        location, rotation = ego_box["location"], ego_box["rotation"]
        if len(location) != 3 or len(rotation) != 3:
            raise ValueError("location/rotation must each contain 3 values")
        if not all(math.isfinite(float(v)) for v in (*location, *rotation)):
            raise ValueError("nonfinite 3D pose")
        return ActorPose(float(location[0]), float(location[1]), math.radians(float(rotation[2])))
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterError("invalid_pose", exc) from exc


def world_to_ego(world_xy, origin):
    """World → CARLA local (+y right), R(-yaw) (p - origin)."""
    return rotate_xy((world_xy[0] - origin.x, world_xy[1] - origin.y), -origin.yaw)


def ego_to_world(ego_xy, origin):
    x, y = rotate_xy(ego_xy, origin.yaw)
    return (origin.x + x, origin.y + y)


def actor_to_reference(actor, delta):
    """Explicit Actor-local offset; δ=0 is Actor reference, NOT a measured rear axle."""
    x, y = ego_to_world(delta, actor)
    return ActorPose(x, y, actor.yaw)


def target_relative(current_reference, future_reference):
    x, y = world_to_ego((future_reference.x, future_reference.y), current_reference)
    return (x, -y, wrap_angle(current_reference.yaw - future_reference.yaw))


def target_to_actor_world(target_pose, current_actor, delta):
    """Inverse target → future Actor world; undo axis reflection and both offsets."""
    reference = actor_to_reference(current_actor, delta)
    x, y = ego_to_world((target_pose[0], -target_pose[1]), reference)
    yaw = wrap_angle(current_actor.yaw - target_pose[2])
    dx, dy = rotate_xy(delta, yaw)
    return ActorPose(x - dx, y - dy, yaw)
