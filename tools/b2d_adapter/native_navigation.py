"""Current expected-route near waypoint bearing, NOT NAVSIM 20m command.

Inference contract: provide a real planner near node in CARLA world XY metres,
as RoutePlanner.run_step does in Bench2DriveZoo/team_code/{planner,vad_b2d_agent}.
Offline source: anno x/y_command_near. No integer command, far/target point,
expert future or top-level noisy GNSS position is used. Missing/degenerate route
information is an error, never an Unknown fallback. v1 emits only L/S/R.
"""

import math
import numpy as np

from .coordinate_transform import actor_to_reference, world_to_ego
from .errors import AdapterError


def near_route_command(raw, config):
    if config.mode != "engineering" or config.navigation_policy != "b2d-near-bearing-v1":
        raise AdapterError("unresolved_navigation", "explicit engineering b2d-near-bearing-v1 required")
    keys = ("x_command_near", "y_command_near")
    if any(k not in raw.navigation for k in keys):
        raise AdapterError("missing_navigation", "current expected-route near waypoint XY required")
    try:
        point = tuple(float(raw.navigation[k]) for k in keys)
    except (ValueError, TypeError) as exc:
        raise AdapterError("invalid_navigation", "numeric near XY required") from exc
    if not all(math.isfinite(v) for v in point):
        raise AdapterError("invalid_navigation", "nonfinite near XY")
    c = config.geometry
    origin = actor_to_reference(raw.actor_pose, (c.axle_dx_m, c.axle_dy_m))
    x, right_y = world_to_ego(point, origin)
    y = -right_y
    distance, bearing = math.hypot(x, y), math.degrees(math.atan2(y, x))
    if x <= 0 or not config.navigation_min_distance_m <= distance <= config.navigation_max_distance_m:
        raise AdapterError("invalid_navigation", "near waypoint must be forward and inside configured distance range")
    threshold = config.navigation_bearing_threshold_deg
    category = 0 if bearing > threshold else 2 if bearing < -threshold else 1
    command = np.eye(4, dtype=np.float32)[category]
    return command, {"policy": config.navigation_policy, "source_frame_id": raw.ref.frame_id,
                     "source_annotation": raw.provenance["source_files"]["anno"], "source_fields": list(keys),
                     "world_xy_m": list(point), "rear_local_xy_m": [x, y], "bearing_deg": bearing,
                     "threshold_deg": threshold, "command_order": ["Left", "Straight", "Right", "Unknown"],
                     "unknown_policy": "never emitted; missing route fails", "future_gt_used": False,
                     "inference_source": "expected global plan -> RoutePlanner.run_step(current position) near XY",
                     "navsim_20m_equivalent": False, "collection_producer_certified": False}
