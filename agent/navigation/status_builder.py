"""Single online Status8 builder: dense global plan -> Navigation4 + Motion4.

This replaces the retired near-bearing policy.  One ordered dense world route is
captured from ``set_global_plan`` and never re-planned per frame; every tick runs
the shared offline calculator (``route_preview_contract.preview_command4``) on
that route plus the current verified rear-axle pose, and derives Motion4 from the
last three 10 Hz rear poses using the frozen A0 definition.

Contract highlights:

* route: dense ``[(Transform, RoadOption), ...]`` converted to world ``(x, y)``,
  consecutive duplicates removed, arc length and content SHA recorded once;
* navigation: 20.0 m arc preview, 2.0 m lateral threshold, 5.0 m rear projection
  gate, order ``[Left, Straight, Right, Unknown]``;
* motion: A0 backward differences at exactly 10 Hz in the current rear frame
  ``[forward, left]`` (velocity m/s, acceleration m/s^2);
* invalid preview or invalid motion yields ``status8=None`` and an explicit reason
  code.  There is no Near fallback and no silent ``Unknown`` padding.
"""
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from agent.common.errors import NavigationError
from agent.common.types import CoordinateFrame, EgoState, Timestamp
from agent.config.settings import NavigationConfig
from agent.navigation.route_preview import route_preview_from_world_nodes
from tools.b2d_adapter.experimental_causal_state import A0, causal_state

STATUS_POLICY = 'b2d-route-preview-20m-v1'
STATUS_ORDER = ('Left', 'Straight', 'Right', 'Unknown', 'vx', 'vy', 'ax', 'ay')
COMMAND_ORDER = ('Left', 'Straight', 'Right', 'Unknown')
MOTION_METHOD = A0
REAR_POSE_POLICY = 'lincoln-10hz-rear-v1'


def validate_command4(command) -> None:
    """Finite four-element one-hot required by the V1 navigation head."""
    if (command is None or len(command) != 4
            or not all(isinstance(v, float) or isinstance(v, int) for v in command)
            or not all(math.isfinite(float(v)) and float(v) in (0.0, 1.0) for v in command)
            or sum(float(v) for v in command) != 1.0):
        raise NavigationError('Model navigation requires a finite four-element one-hot command')


def _validate_config(config: NavigationConfig) -> None:
    if config.status_policy != STATUS_POLICY:
        raise NavigationError(f'Only {STATUS_POLICY} is supported; no legacy policy switch')
    if config.preview_m != 20.0 or config.lateral_threshold_m != 2.0:
        raise NavigationError('Frozen preview rule requires 20.0 m and 2.0 m')
    if config.max_projection_m != 5.0:
        raise NavigationError('Frozen preview rule requires a 5.0 m rear projection gate')
    if not (math.isfinite(config.motion_dt_s) and config.motion_dt_s > 0
            and math.isfinite(config.motion_dt_tolerance_s)
            and 0 < config.motion_dt_tolerance_s < config.motion_dt_s):
        raise NavigationError('Finite positive Motion4 sampling interval required')
    if not (math.isfinite(config.motion_max_gap_s)
            and config.motion_max_gap_s >= config.motion_dt_s):
        raise NavigationError('Motion history max gap must be >= sampling interval')


def build_dense_route(world_plan: Sequence[Any]) -> Tuple[Tuple[Tuple[float, float], ...], float, str]:
    """Validate the dense plan, drop only consecutive duplicates, hash the result."""
    points, duplicates = [], 0
    for entry in world_plan:
        try:
            transform = entry[0]
            location = transform.location
            x, y = float(location.x), float(location.y)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise NavigationError(f'Dense plan requires (Transform, RoadOption): {exc}') from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise NavigationError('Dense plan requires finite world coordinates')
        if points and points[-1] == (x, y):
            duplicates += 1
            continue
        points.append((x, y))
    if len(points) < 2:
        raise NavigationError('Dense plan requires at least two distinct points')
    arc = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    digest = hashlib.sha256()
    for x, y in points:
        digest.update(f'{x:.6f},{y:.6f}\n'.encode())
    return tuple(points), arc, digest.hexdigest()


class MotionHistory:
    """A0 motion over trailing 10 Hz rear poses (leaderboard runs at 20 Hz)."""

    def __init__(self, config: NavigationConfig):
        self.expected_dt = float(config.motion_dt_s)
        self.tolerance = float(config.motion_dt_tolerance_s)
        self.max_gap = float(config.motion_max_gap_s)
        self._samples = []

    def reset(self) -> None:
        self._samples.clear()

    def history_size(self) -> int:
        return len(self._samples)

    def update(self, ego: EgoState, timestamp: Timestamp):
        rear = ego.pose
        if ego.timestamp != timestamp:
            raise NavigationError('Motion4 requires the current simulation timestamp')
        if (rear is None or rear.frame != CoordinateFrame.CARLA_WORLD
                or not all(math.isfinite(v) for v in (rear.x_m, rear.y_m, rear.heading_rad))):
            raise NavigationError('Motion4 requires a finite CARLA world rear pose')
        now = float(timestamp.seconds)
        if not math.isfinite(now):
            raise NavigationError('Motion4 requires a finite simulation timestamp')
        if self._samples and now <= self._samples[-1][0]:
            raise NavigationError('Ego timestamps must strictly increase for Motion4')
        if self._samples and now - self._samples[-1][0] > self.max_gap:
            self._samples.clear()
        interval = None if not self._samples else now - self._samples[-1][0]
        if interval is None or interval >= self.expected_dt - self.tolerance:
            self._samples.append((now, rear.x_m, rear.y_m, rear.heading_rad))
            del self._samples[:-4]
        if len(self._samples) < 3:
            return None, 'MOTION_HISTORY_INSUFFICIENT', {'history': len(self._samples)}
        window = self._samples[-3:]
        intervals = [b[0] - a[0] for a, b in zip(window, window[1:])]
        if any(abs(value - self.expected_dt) > self.tolerance for value in intervals):
            return None, 'MOTION_SAMPLING_INTERVAL', {'intervals_s': intervals,
                                                     'expected_dt_s': self.expected_dt}
        estimate = causal_state([(s[1], s[2], s[3]) for s in window],
                                method=MOTION_METHOD, dt=self.expected_dt)
        motion = (float(estimate.velocity[0]), float(estimate.velocity[1]),
                  float(estimate.acceleration[0]), float(estimate.acceleration[1]))
        return motion, None, {'method': estimate.method,
                              'intervals_s': intervals,
                              'history': len(self._samples),
                              'source_frame_count': estimate.source_frame_count}


@dataclass(frozen=True)
class OnlineStatus:
    timestamp: Timestamp
    command4: Optional[Tuple[float, float, float, float]]
    status8: Optional[Tuple[float, ...]]
    validity: str
    reason: Optional[str]
    preview: Mapping[str, Any]
    motion4: Optional[Tuple[float, float, float, float]]
    motion_reason: Optional[str]
    route_generation: int
    route_sha256: str
    route_point_count: int
    route_arc_m: float
    policy: str = STATUS_POLICY
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.validity == 'VALID'


class OnlineStatusBuilder:
    """One dense route, one shared calculator, explicit invalid states."""

    def __init__(self, config: NavigationConfig):
        _validate_config(config)
        self.config = config
        self.motion = MotionHistory(config)
        self._points: Tuple[Tuple[float, float], ...] = ()
        self._route_sha256: Optional[str] = None
        self._route_arc_m = 0.0
        self._duplicates_removed = 0
        self._generation = 0
        self._last_progress_m: Optional[float] = None
        self._last_status: Optional[OnlineStatus] = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def route(self):
        return {'points': self._points, 'sha256': self._route_sha256,
                'arc_length_m': self._route_arc_m, 'point_count': len(self._points),
                'generation': self._generation,
                'duplicates_removed': self._duplicates_removed,
                'policy': STATUS_POLICY, 'frame': 'carla_world_xy'}

    def set_global_plan(self, world_plan: Sequence[Any], gps_plan: Optional[Sequence[Any]] = None) -> None:
        if gps_plan is not None and len(gps_plan) != len(world_plan):
            raise NavigationError('Aligned GPS and world plans are required')
        points, arc, sha = build_dense_route(world_plan)
        self._points = points
        self._route_arc_m = arc
        self._route_sha256 = sha
        self._duplicates_removed = len(world_plan) - len(points)
        self._generation += 1
        self._last_progress_m = None

    def reset(self) -> None:
        self._points = ()
        self._route_sha256 = None
        self._route_arc_m = 0.0
        self._duplicates_removed = 0
        self._generation = 0
        self._last_progress_m = None
        self._last_status = None
        self.motion.reset()

    def preview_only(self, ego: EgoState):
        return route_preview_from_world_nodes(
            self._points, (ego.pose.x_m, ego.pose.y_m), ego.pose.heading_rad,
            route_generation=self._generation,
            previous_progress_m=self._last_progress_m,
            preview_m=self.config.preview_m,
            lateral_threshold_m=self.config.lateral_threshold_m,
            max_projection_m=self.config.max_projection_m)

    def update(self, ego: EgoState, timestamp: Timestamp) -> OnlineStatus:
        if not self._points or self._route_sha256 is None:
            raise NavigationError('Global plan has not been supplied')
        rear = ego.pose
        if (rear is None or rear.frame != CoordinateFrame.CARLA_WORLD
                or not all(math.isfinite(v) for v in (rear.x_m, rear.y_m, rear.heading_rad))):
            raise NavigationError('Online Status requires a finite CARLA world rear pose')
        if self.config.require_rear_pose_policy and \
                ego.provenance.get('geometry_policy') != REAR_POSE_POLICY:
            raise NavigationError('Online Status requires the verified rear-axle geometry policy')
        resolved = route_preview_from_world_nodes(
            self._points, (rear.x_m, rear.y_m), rear.heading_rad,
            route_generation=self._generation,
            previous_progress_m=self._last_progress_m,
            preview_m=self.config.preview_m,
            lateral_threshold_m=self.config.lateral_threshold_m,
            max_projection_m=self.config.max_projection_m)
        preview = resolved.preview
        motion4, motion_reason, motion_meta = self.motion.update(ego, timestamp)

        reason = None
        command4 = None
        if preview.get('preview_valid'):
            command4 = tuple(float(v) for v in preview['command4'])
            self._last_progress_m = preview.get('route_progress_m')
        else:
            reason = str(preview.get('reason') or 'preview_invalid')
        if motion_reason is not None:
            reason = motion_reason if reason is None else f'{reason}+{motion_reason}'

        status8 = None
        validity = 'INVALID'
        if command4 is not None and motion4 is not None:
            status8 = tuple(command4) + tuple(motion4)
            validity = 'VALID'

        record = OnlineStatus(
            timestamp=timestamp,
            command4=command4,
            status8=status8,
            validity=validity,
            reason=reason,
            preview=dict(preview),
            motion4=motion4,
            motion_reason=motion_reason,
            route_generation=self._generation,
            route_sha256=self._route_sha256,
            route_point_count=len(self._points),
            route_arc_m=round(self._route_arc_m, 3),
            provenance={
                'navigation_policy': STATUS_POLICY,
                'route_source': 'dense_set_global_plan_world_coords',
                'sparse_plan_used': False,
                'future_gt_used': False,
                'road_option_mapped': False,
                'motion_method': MOTION_METHOD,
                'motion_meta': motion_meta,
                'preview_parameters': {
                    'preview_m': self.config.preview_m,
                    'lateral_threshold_m': self.config.lateral_threshold_m,
                    'max_projection_m': self.config.max_projection_m},
            })
        self._last_status = record
        return record

    @property
    def last_status(self) -> Optional[OnlineStatus]:
        return self._last_status
