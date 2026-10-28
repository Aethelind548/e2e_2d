"""Configuration defaults are structural; unverified physical values stay None."""

from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from agent.common.errors import ConfigurationError
from agent.common.types import CoordinateFrame


@dataclass(frozen=True)
class SensorConfig:
    specifications: Tuple[Mapping[str, Any], ...] = ()
    required_ids: Tuple[str, ...] = ()
    buffer_size: int = 3  # Storage capacity only, not a sampling/physics assumption.
    alignment_policy: Optional[str] = None


@dataclass(frozen=True)
class EgoStateConfig:
    policy: Optional[str] = None
    gnss_id: Optional[str] = None
    imu_id: Optional[str] = None
    speed_id: Optional[str] = None
    lat_ref_deg: Optional[float] = None
    lon_ref_deg: Optional[float] = None
    vehicle_type: Optional[str] = None
    geometry_policy: Optional[str] = None
    planar_assumption: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    camera_policy: Optional[str] = None
    lidar_policy: Optional[str] = None
    state_policy: Optional[str] = None


@dataclass(frozen=True)
class NavigationConfig:
    """Single online Status policy: dense plan + shared 20 m preview + A0 motion."""
    status_policy: str = 'b2d-route-preview-20m-v1'
    preview_m: float = 20.0
    lateral_threshold_m: float = 2.0
    max_projection_m: float = 5.0
    motion_dt_s: float = 0.1  # Bench2Drive 10 Hz; leaderboard ticks at 20 Hz.
    motion_dt_tolerance_s: float = 0.01
    motion_max_gap_s: float = 0.5
    require_rear_pose_policy: bool = True


@dataclass(frozen=True)
class PlannerConfig:
    kind: str = "diffusiondrive_v1"
    device: Optional[str] = None
    checkpoint_path: Optional[str] = None
    plan_anchor_path: Optional[str] = None
    backbone_path: Optional[str] = None
    # TODO: resolve from the selected official model config and checkpoint.
    num_poses: Optional[int] = None
    interval_s: Optional[float] = None
    output_frame: CoordinateFrame = CoordinateFrame.UNKNOWN
    mock_speed_mps: Optional[float] = None


@dataclass(frozen=True)
class PIDConfig:
    kp: Optional[float] = None
    ki: Optional[float] = None
    kd: Optional[float] = None
    output_min: Optional[float] = None
    output_max: Optional[float] = None
    integral_limit: Optional[float] = None


@dataclass(frozen=True)
class ControllerConfig:
    longitudinal_pid: PIDConfig = field(default_factory=PIDConfig)
    lateral_method: Optional[str] = None
    wheelbase_m: Optional[float] = None
    target_speed_policy: Optional[str] = None


@dataclass(frozen=True)
class RuntimeConfig:
    logger_name: str = "diffusiondrive.agent"
    log_level: str = "INFO"


@dataclass(frozen=True)
class AgentConfig:
    sensors: SensorConfig = field(default_factory=SensorConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    ego_state: EgoStateConfig = field(default_factory=EgoStateConfig)


def _section(cls, value):
    if not isinstance(value, dict):
        raise ConfigurationError(f"{cls.__name__} must be a JSON object")
    unknown = set(value) - {item.name for item in fields(cls)}
    if unknown:
        raise ConfigurationError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**value)


def _positive(value, name, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            or (integer and not isinstance(value, int))):
        raise ConfigurationError(f"{name} must be finite and positive")


def _object(data, name):
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a JSON object")
    return dict(value)


def load_config(path=None) -> AgentConfig:
    """Load JSON; relative asset paths resolve against the config file directory.

    No environment variable overrides, model imports or asset reads occur here.
    None selects dataclass defaults, which intentionally cannot drive a vehicle.
    """
    try:
        data = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigurationError("Agent config must be a JSON object")
        unknown = set(data) - {item.name for item in fields(AgentConfig)}
        if unknown:
            raise ConfigurationError(f"Unknown AgentConfig keys: {sorted(unknown)}")
        sensors = _object(data, "sensors")
        specs, required = sensors.get("specifications", []), sensors.get("required_ids", [])
        if (not isinstance(specs, list) or not all(isinstance(s, dict) for s in specs)
                or not isinstance(required, list) or not all(isinstance(s, str) and s for s in required)):
            raise ConfigurationError("Sensor specifications must be objects; required_ids must be strings")
        ids = [s.get("id") for s in specs]
        if (any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids))
                or any(not isinstance(s.get("type"), str) or not s["type"] for s in specs)
                or not set(required).issubset(ids) or len(required) != len(set(required))):
            raise ConfigurationError("Unique sensor ids/types and declared required_ids are required")
        sensors.update(specifications=tuple(specs), required_ids=tuple(required))
        sensor_config = _section(SensorConfig, sensors)
        _positive(sensor_config.buffer_size, "buffer_size", integer=True)
        planner = _object(data, "planner")
        planner["output_frame"] = CoordinateFrame(planner.get("output_frame", "unknown"))
        base = Path(path).resolve().parent if path is not None else Path.cwd()
        for key in ("checkpoint_path", "plan_anchor_path", "backbone_path"):
            if planner.get(key) is not None:
                if not isinstance(planner[key], str) or not planner[key]:
                    raise ConfigurationError(f"{key} must be a nonempty path or null")
                planner[key] = str((base / planner[key]).resolve())
        planner_config = _section(PlannerConfig, planner)
        if planner_config.kind not in ("diffusiondrive_v1", "mock"):
            raise ConfigurationError("planner.kind must be diffusiondrive_v1 or explicit mock")
        for key in ("num_poses", "interval_s"):
            value = getattr(planner_config, key)
            if value is not None:
                _positive(value, key, integer=key == "num_poses")
        if planner_config.mock_speed_mps is not None:
            v = planner_config.mock_speed_mps
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ConfigurationError("mock_speed_mps must be finite and nonnegative")
        controller = _object(data, "controller")
        controller["longitudinal_pid"] = _section(PIDConfig, controller.get("longitudinal_pid", {}))
        return AgentConfig(sensor_config, _section(FeatureConfig, data.get("features", {})),
                           _section(NavigationConfig, data.get("navigation", {})), planner_config,
                           _section(ControllerConfig, controller), _section(RuntimeConfig, data.get("runtime", {})),
                           _section(EgoStateConfig, data.get("ego_state", {})))
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigurationError(f"Cannot load Agent configuration: {exc}") from exc
