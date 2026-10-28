"""Single definitions for all eight module boundaries. Units are SI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

XY = Tuple[float, float]


class CoordinateFrame(str, Enum):
    UNKNOWN = "unknown"
    CARLA_WORLD = "carla_world_xy_yaw_right"
    CARLA_ACTOR = "carla_actor_x_forward_y_right"
    MODEL_REAR = "model_rear_x_forward_y_left_yaw_left"


@dataclass(frozen=True)
class Timestamp:
    seconds: float  # Simulation seconds, NOT a sensor frame number or wall time.
    source: str = "leaderboard_game_time"


@dataclass(frozen=True)
class SimulationTick:
    """Current external GameTime anchor; absolute CARLA frame is independent."""
    frame_id: int
    timestamp: Timestamp
    delta_seconds: float


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    heading_rad: float
    frame: CoordinateFrame
    reference: str  # E.g. measured Actor origin or verified rear axle policy.


@dataclass(frozen=True)
class SensorSample:
    sensor_id: str
    frame_id: int
    payload: Any  # Raw Leaderboard payload; see README for layouts.
    timestamp: Optional[Timestamp] = None  # Unknown until frame-clock alignment.


@dataclass(frozen=True)
class SensorObservation:
    timestamp: Timestamp
    samples: Mapping[str, SensorSample]


@dataclass(frozen=True)
class EgoState:
    timestamp: Timestamp
    pose: Optional[Pose2D] = None
    speed_mps: Optional[float] = None
    velocity_xy_mps: Optional[XY] = None
    acceleration_xy_mps2: Optional[XY] = None
    dynamics_frame: CoordinateFrame = CoordinateFrame.UNKNOWN
    provenance: Mapping[str, Any] = field(default_factory=dict)
    actor_pose: Optional[Pose2D] = None
    navigation_world_xy_m: Optional[XY] = None  # Measured GNSS installation point.


@dataclass(frozen=True)
class GlobalRoute:
    gps_plan: Tuple[Any, ...]  # Original (GPS dict, RoadOption) pairs, unchanged.
    world_plan: Tuple[Any, ...]  # Original (CARLA Transform, RoadOption) pairs.
    frame: CoordinateFrame = CoordinateFrame.CARLA_WORLD


@dataclass(frozen=True)
class NavigationState:
    timestamp: Timestamp
    local_target: Optional[Pose2D] = None
    raw_command: Any = None  # Never implicitly interpreted as a model index.
    command4: Optional[Tuple[float, float, float, float]] = None
    policy: Optional[str] = None
    near_index: Optional[int] = None
    route_generation: int = 0
    status8: Optional[Tuple[float, ...]] = None  # [nav4, vx, vy, ax, ay] or None.
    status_validity: str = 'INVALID'
    status_reason: Optional[str] = None
    motion4: Optional[Tuple[float, float, float, float]] = None
    route_sha256: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelFeatures:
    """Single sample, unbatched; predictor owns batching/device placement."""
    timestamp: Timestamp
    camera_feature: Tensor
    lidar_feature: Tensor
    status_feature: Tensor
    frame: CoordinateFrame
    provenance: Mapping[str, Any] = field(default_factory=dict)
    origin: Optional[Pose2D] = None  # Verified world pose of local model origin.

    def as_dict(self) -> Mapping[str, Tensor]:
        return {"camera_feature": self.camera_feature,
                "lidar_feature": self.lidar_feature,
                "status_feature": self.status_feature}


@dataclass(frozen=True)
class TrajectoryPoint:
    x_m: float
    y_m: float
    heading_rad: float
    time_from_start_s: float  # Explicit future offset from trajectory timestamp.


@dataclass(frozen=True)
class TrajectoryPrediction:
    timestamp: Timestamp
    points: Tuple[TrajectoryPoint, ...]
    frame: CoordinateFrame
    source: str  # Real model identity or explicit mock identifier.
    origin: Optional[Pose2D] = None  # Local reference pose at prediction time.


@dataclass(frozen=True)
class ReferenceTrajectory:
    timestamp: Timestamp
    points: Tuple[TrajectoryPoint, ...]
    frame: CoordinateFrame
    source: str
    origin: Optional[Pose2D] = None


@dataclass(frozen=True)
class ControlCommand:
    timestamp: Timestamp
    steer: float  # CARLA normalized sign/range [-1, 1], not a wheel angle.
    throttle: float  # [0, 1]
    brake: float  # [0, 1]
    hand_brake: bool = False
    reverse: bool = False


class RuntimeStatus(str, Enum):
    CREATED = "created"
    CONFIGURED = "configured"  # Configuration only; does not mean driving-ready.
    WARMING_UP = "warming_up"
    RUNNING = "running"
    FAILED = "failed"
    DESTROYED = "destroyed"


@dataclass
class AgentRuntimeState:
    status: RuntimeStatus = RuntimeStatus.CREATED
    completed_steps: int = 0
    timestamp: Optional[Timestamp] = None
    active_module: Optional[str] = None
    module_status: dict = field(default_factory=dict)
    last_error_type: Optional[str] = None
    last_error_message: Optional[str] = None
    warmup_steps: int = 0
    readiness_message: Optional[str] = None
