"""Isolated, causal-only ego-state prototypes for Adapter diagnosis.

This module is deliberately not imported by any Feature Producer. It provides
three candidates over trailing rear-axle poses:

* A0: the current two-point/backward-difference baseline.
* A1: a four-point endpoint polynomial derivative at the current sample.
* A2: A1 velocity with the current scalar ``speed`` used only as a magnitude
  correction; it never invents a full 2-D vector from the scalar.
* A2V: the speed-fused velocity above plus the A0 acceleration, retained only
  as a partial candidate after the full A2 acceleration failed the panel test.

No function reads a pose after the current sample. Central differences live in
``reference_state`` and are for offline evaluation only.
"""
from dataclasses import dataclass
import math

import numpy as np

from .coordinate_transform import rotate_xy


A0 = 'a0-backward-difference'
A1 = 'a1-causal-poly4'
A2 = 'a2-causal-poly4-speed-magnitude'
A2V = 'a2v-speed-fused-velocity-a0-acceleration'
METHODS = (A0, A1, A2, A2V)


@dataclass(frozen=True)
class StateEstimate:
    velocity: np.ndarray
    acceleration: np.ndarray
    method: str
    source_frame_count: int
    used_speed: bool

    def as_dict(self):
        return {
            'method': self.method,
            'velocity_local_xy_mps': self.velocity.tolist(),
            'acceleration_local_xy_mps2': self.acceleration.tolist(),
            'source_frame_count': self.source_frame_count,
            'used_speed_magnitude_only': self.used_speed,
        }


def _to_current_local(vector_world, current_yaw):
    x, right_y = rotate_xy(tuple(float(v) for v in vector_world), -float(current_yaw))
    return np.asarray([x, -right_y], dtype=np.float64)


def _check_poses(poses):
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise ValueError('poses must have shape (N>=3, 3): x,y,yaw')
    if not np.isfinite(p).all():
        raise ValueError('poses must be finite')
    # Yaw may wrap across +/-pi; positions drive the derivatives and only the
    # current yaw is used for the local-axis rotation.
    return p


def causal_state(poses, method=A1, *, speed=None, dt=0.1):
    """Estimate current rear-frame velocity/acceleration from trailing poses.

    ``poses`` is ordered oldest to newest and ends at the current sample. All
    vectors use the project convention ``[forward, left]`` in metres/metres per
    second. ``speed`` is optional and is used only as a magnitude, never as a
    substitute for the 2-D direction.
    """
    if method not in METHODS:
        raise ValueError('unknown method: ' + str(method))
    p = _check_poses(poses)
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError('positive finite dt required')
    h = float(dt)
    current_yaw = float(p[-1, 2])
    if method == A0:
        if len(p) < 3:
            raise ValueError('A0 requires t-2..t')
        v_world = (p[-1, :2] - p[-2, :2]) / h
        a_world = (p[-1, :2] - 2 * p[-2, :2] + p[-3, :2]) / (h * h)
        used_speed = False
    else:
        if len(p) < 4:
            raise ValueError('A1/A2 requires t-3..t')
        x0, x1, x2, x3 = p[-4:, :2]
        # Four-point causal quadratic least-squares fit over t-3..t. This is
        # less noise-amplifying than exact cubic interpolation for acceleration.
        v_world = (0.45 * x0 - 0.85 * x1 - 0.65 * x2 + 1.05 * x3) / h
        if method == A2V:
            # The full A2 acceleration did not improve on A0; keep the
            # lower-MAE backward acceleration for this partial candidate.
            a_world = (x3 - 2 * x2 + x1) / (h * h)
        else:
            a_world = (0.5 * x0 - 0.5 * x1 - 0.5 * x2 + 0.5 * x3) / (h * h)
        used_speed = method in (A2, A2V) and speed is not None
    velocity = _to_current_local(v_world, current_yaw)
    acceleration = _to_current_local(a_world, current_yaw)
    if method in (A2, A2V):
        if speed is None:
            raise ValueError('A2 requires current scalar speed')
        speed = float(speed)
        if not math.isfinite(speed):
            raise ValueError('finite speed required')
        # B2D stores a signed scalar and can report a tiny negative value at
        # rest. Only its magnitude is used here; the 2-D direction still comes
        # from the causal position fit.
        speed = abs(speed)
        magnitude = float(np.linalg.norm(velocity))
        # Do not fabricate a direction when the polynomial fit is effectively
        # stationary. Otherwise preserve the fitted direction and use speed
        # only to correct its magnitude.
        if magnitude > 0.02:
            velocity = velocity * (speed / magnitude)
        else:
            used_speed = False
    return StateEstimate(velocity, acceleration, method, min(len(p), 4), used_speed)


def reference_state(poses, dt=0.1):
    """Offline-only central reference; requires a pose at t+1.

    This must never be used as an online input. It is a smoothness reference,
    not physical ground truth.
    """
    p = _check_poses(poses)
    if len(p) < 3:
        raise ValueError('reference requires t-1..t+1')
    h = float(dt)
    v_world = (p[-1, :2] - p[-3, :2]) / (2 * h)
    a_world = (p[-1, :2] - 2 * p[-2, :2] + p[-3, :2]) / (h * h)
    return StateEstimate(_to_current_local(v_world, float(p[-2, 2])),
                         _to_current_local(a_world, float(p[-2, 2])),
                         'offline-central-reference', 3, False)


def central_reference_from_window(poses, dt=0.1):
    """Convenience wrapper: window is t-1..t+1 and returns the t state."""
    return reference_state(poses, dt=dt)
