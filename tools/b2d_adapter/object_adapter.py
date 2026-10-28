"""Candidate intermediate objects only; no NAVSIM, Torch, maps or navigation imports."""

from dataclasses import dataclass
import math

import numpy as np

from .config import CandidateConfig
from .coordinate_transform import actor_to_reference, rotate_xy, wrap_angle
from .errors import AdapterError
from .lidar_decode import decode_lidar_in_process
from .trajectory_target import CandidateTrajectoryTargetBuilder


@dataclass
class CandidateState:
    velocity: object
    acceleration: object
    provenance: dict


@dataclass
class CandidateFrameObject:
    raw: object
    rear_world_carla: object
    global_pose_nav: object
    causal_state: object
    relative_time_s: float


@dataclass
class CandidateObjectBundle:
    window: object
    frames: tuple
    current_sensors: object
    target: object
    provenance: dict


def candidate_causal_state(reader, ref, config):
    """Trailing differences, not central future differences or certified dynamic state.

    Velocity has midpoint lag dt/2; backward second position difference has lag
    dt. Express both at current body axes as diagnostics. Do not hide those lags.
    """
    if ref.frame_id < 2:
        return CandidateState(None, None, {"status": "missing", "reason": "insufficient_past"})
    frames = [reader.read(reader.index.ref(i)) for i in (ref.frame_id-2, ref.frame_id-1, ref.frame_id)]
    delta = (config.axle_dx_m, config.axle_dy_m)
    for frame in frames:
        if frame.ego_box.get("type_id") != config.axle_vehicle_type:
            raise AdapterError("reference_vehicle_mismatch", frame.ref.token)
    poses = [actor_to_reference(f.actor_pose, delta) for f in frames]
    points = np.array([[p.x, p.y] for p in poses], dtype=np.float64)
    velocity_world = (points[2] - points[1]) / config.source_dt_s
    acceleration_world = (points[2] - 2*points[1] + points[0]) / config.source_dt_s**2
    def local(vector):
        x, y = rotate_xy(vector, -poses[2].yaw)
        return np.array([x, -y], dtype=np.float32)
    values = (local(velocity_world), local(acceleration_world))
    if not all(np.isfinite(v).all() for v in values):
        raise AdapterError("invalid_state", ref.token)
    return CandidateState(*values, provenance={"status": "candidate", "causal": True,
                          "source_frame_ids": [f.ref.frame_id for f in frames],
                          "source_files": [f.provenance["source_files"]["anno"] for f in frames],
                          "clock_source": config.clock_source, "reference_source": config.axle_source,
                          "velocity_lag_s": config.source_dt_s/2,
                          "acceleration_lag_s": config.source_dt_s,
                          "axes": "current candidate rear frame x forward y left; m/s, m/s^2"})


class B2DObjectAdapter:
    def __init__(self, reader, config=None, lidar_decoder_python=None):
        self.reader = reader
        self.config = config or CandidateConfig()
        self.lidar_decoder_python = lidar_decoder_python

    def build(self, current):
        c = self.config
        # Reuse all existing integrity, continuity and trajectory checks.
        target = CandidateTrajectoryTargetBuilder(self.reader, c).build(current, full_scene=True)
        frames = []
        for ref in (*target.window.history, *target.window.scene_future):
            raw = self.reader.read(ref)
            rear = actor_to_reference(raw.actor_pose, (c.axle_dx_m, c.axle_dy_m))
            global_pose = np.array([rear.x, -rear.y, wrap_angle(-rear.yaw)], dtype=np.float64)
            state = candidate_causal_state(self.reader, ref, c) if ref.frame_id <= current.frame_id else None
            frames.append(CandidateFrameObject(raw, rear, global_pose, state,
                                               (ref.frame_id-current.frame_id)*c.source_dt_s))
        sensors = self.reader.read(current, cameras=True, lidar=self.lidar_decoder_python is None)
        if self.lidar_decoder_python is not None:
            sensors.lidar_xyz, sensors.lidar_metadata = decode_lidar_in_process(current, self.lidar_decoder_python)
            sensors.provenance["source_files"]["lidar"] = str(self.reader.index.path(current, "lidar"))
        return CandidateObjectBundle(target.window, tuple(frames), sensors, target,
                                     dict(c.provenance(), navigation_status="missing: scheme C",
                                          absolute_timestamp_status="missing; relative candidate times only",
                                          world_embedding="CARLA world mirrored Y and yaw, NOT NuPlan georeferenced map",
                                          raw_actor_pose_source="ego bounding box location/rotation",
                                          future_state_estimation=False))
