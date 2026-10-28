"""Only candidate_trajectory: never exposes a certified `trajectory` training key."""

from dataclasses import dataclass
import math

from .config import CandidateConfig
from .coordinate_transform import actor_to_reference, target_relative, target_to_actor_world, wrap_angle
from .errors import AdapterError
from .temporal_sampler import TemporalSampler


@dataclass
class CandidateTrajectory:
    candidate_trajectory: object  # numpy float32 [N,3], diagnostic only
    current: object
    window: object
    provenance: dict
    validation: dict


class CandidateTrajectoryTargetBuilder:
    def __init__(self, reader, config=None):
        self.reader = reader
        self.config = config or CandidateConfig()
        self.sampler = TemporalSampler(self.config)

    def build(self, current, full_scene=True, target_only=False, terminal_padding=False):
        try:
            import numpy as np
        except ImportError as exc:
            raise AdapterError("dependency_missing", "candidate target requires numpy") from exc
        c = self.config
        terminal = None
        if terminal_padding:
            if full_scene or not target_only:
                raise AdapterError('invalid_terminal_window', 'padding only applies to trajectory supervision')
            from .terminal_policy import require_terminal_padding
            terminal = require_terminal_padding(self.reader, c)
        endpoint = self.reader.index.ref(self.reader.index.frame_ids[-1]) if terminal else None
        window = self.sampler.sample(self.reader.index, current, full_scene=full_scene,
                                     target_only=target_only, terminal_endpoint=endpoint)
        # Decode bounded annotations only, including unsampled frames for continuity.
        frames = {ref.frame_id: self.reader.read(ref) for ref in window.continuous}
        max_speed, max_yaw_rate = 0.0, 0.0
        previous = None
        for ref in window.continuous:
            frame = frames[ref.frame_id]
            if frame.ego_box.get("type_id") != c.axle_vehicle_type:
                raise AdapterError("reference_vehicle_mismatch", "{}: {}".format(ref.token, frame.ego_box.get("type_id")))
            if not np.isfinite(frame.ego_box["rotation"]).all():
                raise AdapterError("invalid_pose", ref.token)
            if max(abs(float(v)) for v in frame.ego_box["rotation"][:2]) > c.max_planar_tilt_deg:
                raise AdapterError("nonplanar_pose", ref.token)
            if frame.state.get("teleport_flag", False):
                raise AdapterError("teleport_flag", ref.token)
            if previous is not None:
                p, q = previous.actor_pose, frame.actor_pose
                speed = math.hypot(q.x - p.x, q.y - p.y) / c.source_dt_s
                rate = abs(wrap_angle(q.yaw - p.yaw)) / c.source_dt_s
                max_speed, max_yaw_rate = max(max_speed, speed), max(max_yaw_rate, rate)
                if speed > c.max_speed_mps or rate > c.max_yaw_rate_radps:
                    raise AdapterError("teleport_or_discontinuity", ref.token)
            previous = frame
        delta = (c.axle_dx_m, c.axle_dy_m)
        origin = frames[current.frame_id].actor_pose
        reference = actor_to_reference(origin, delta)
        values, errors = [], []
        for ref in window.trajectory_future:
            future = frames[ref.frame_id].actor_pose
            target = target_relative(reference, actor_to_reference(future, delta))
            inverse = target_to_actor_world(target, origin, delta)
            errors.append(math.hypot(inverse.x - future.x, inverse.y - future.y))
            values.append(target)
        poses = np.asarray(values, dtype=np.float32)
        if poses.shape != (c.trajectory_count, 3) or not np.isfinite(poses).all():
            raise AdapterError("invalid_candidate_target", "shape or finite check failed")
        # Report float32 reconstruction separately from pre-cast double algebra.
        cast_errors = []
        for row, ref in zip(poses, window.trajectory_future):
            inverse = target_to_actor_world(row, origin, delta)
            future = frames[ref.frame_id].actor_pose
            cast_errors.append(math.hypot(inverse.x - future.x, inverse.y - future.y))
        provenance = dict(c.provenance(), pose_source=frames[current.frame_id].provenance['pose_source'],
                          source_annotations=[frames[r.frame_id].provenance['source_files']['anno'] for r in window.continuous])
        if terminal:
            real_count = sum(current.frame_id+n*c.stride <= endpoint.frame_id for n in range(1, c.trajectory_count+1))
            provenance['terminal_padding'] = dict(
                policy=terminal['policy'], category='A', real_point_count=real_count,
                padded_point_count=c.trajectory_count-real_count,
                source='repeat_true_terminal_rear_axle_pose', endpoint_frame=endpoint.frame_id,
                result_sha256=terminal['result_sha256'], qualification=terminal['metrics'],
                complete_training_targets=False)
        return CandidateTrajectory(poses, current, window, provenance,
                                   {"double_roundtrip_max_m": max(errors), "float32_roundtrip_max_m": max(cast_errors),
                                    "continuous_actor_speed_max_mps": max_speed,
                                    "continuous_yaw_rate_max_radps": max_yaw_rate,
                                    "gate_status": "project diagnostic thresholds, not certified physical clock"})
