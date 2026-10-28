"""Real NAVSIM diagnostic objects with explicit missing fields, never training-ready.

This module does not instantiate Agents/models or clone/mock NAVSIM classes.
Factories are imported lazily in the intended model environment.
"""

from dataclasses import dataclass
import math
import re

import numpy as np

from .config import CandidateConfig
from .coordinate_transform import target_relative, world_to_ego, wrap_angle
from .errors import AdapterError
from .lidar_adapter import candidate_lidar_xyz


@dataclass
class VehicleBoxesView:
    """Only the real box/name subset consumed by _compute_agent_targets.

    NOT full NAVSIM Annotations: velocity_3d and certified persistent tokens are
    unavailable. Other object classes are explicitly omitted, not padded as data.
    """
    boxes: object
    names: list
    provenance: dict


@dataclass
class PartialNavsimObjects:
    agent_input: object
    scene: object
    source: object
    missing_fields: dict

    def require_complete(self):
        raise AdapterError("incomplete_navsim_contract", self.missing_fields)


PARKED_VEHICLE_PREFIX = "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/"


def is_b2d_vehicle(box):
    """Collector dynamic actors and parked-car mesh actors share vehicle GT."""
    type_id = str(box.get("type_id", ""))
    return box.get("class") == "vehicle" and (type_id.startswith("vehicle.") or type_id.startswith(PARKED_VEHICLE_PREFIX))


def candidate_vehicle_boxes(raw, config):
    boxes, sources = [], []
    skipped = {}
    for box in raw.annotation["bounding_boxes"]:
        if not is_b2d_vehicle(box):
            name = str(box.get("class"))
            skipped[name] = skipped.get(name, 0) + 1
            continue
        try:
            center = np.asarray(box["center"], dtype=np.float64)
            size = 2*np.asarray(box["extent"], dtype=np.float64)
            yaw = math.radians(float(box["rotation"][2]))
            if center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any() or not math.isfinite(yaw):
                raise ValueError("invalid vehicle center/size/yaw")
            xy = world_to_ego(center[:2], raw.actor_pose)
            z = (np.asarray(raw.ego_box["world2ego"]) @ np.r_[center, 1.0])[2] + config.lidar_z_shift_m
            boxes.append([xy[0]-config.axle_dx_m, -(xy[1]-config.axle_dy_m), z,
                          *size, wrap_angle(raw.actor_pose.yaw-yaw)])
            sources.append(str(box["id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError("invalid_vehicle_box", str(exc)) from exc
    return VehicleBoxesView(np.asarray(boxes, dtype=np.float32).reshape(-1, 7), ["vehicle"]*len(boxes),
                            {"candidate_only": True, "source_actor_ids": sources,
                             "source": raw.provenance["source_files"]["anno"],
                             "box_origin": "real bounding-box center, not Actor location",
                             "dimensions": "2*extent; local x/y/z,length,width,height,heading",
                             "omitted_class_counts": skipped, "missing_full_annotations": ["velocity_3d", "instance_tokens", "track_tokens"],
                             "reference_status": config.axle_source, "height_status": "candidate Actor z origin"})


def candidate_camera_calibration(raw, sensor_name, config):
    """Column vectors: target_from_optical = F_delta * Ego_from_CARLAcam * A.

    A maps optical (x right,y down,z forward) → CARLAcam (x forward,y right,z up).
    F_delta subtracts Actor-local candidate delta and flips Y. Height origin and
    cross-domain camera geometry remain unverified; missing distortion stays None.
    """
    sensor = raw.sensors[sensor_name]
    ego_from_cam = np.asarray(raw.ego_box["world2ego"], dtype=np.float64) @ np.linalg.inv(np.asarray(sensor["world2cam"], dtype=np.float64))
    optical_to_carla = np.eye(4)
    optical_to_carla[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, -1, 0]]
    target_from_actor = np.diag([1., -1., 1., 1.])
    target_from_actor[:3, 3] = [-config.axle_dx_m, config.axle_dy_m, config.lidar_z_shift_m]
    transform = target_from_actor @ ego_from_cam @ optical_to_carla
    k = np.asarray(sensor["intrinsic"], dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise AdapterError("invalid_calibration", sensor_name)
    r = transform[:3, :3]
    orthogonal_error = float(np.max(np.abs(r.T @ r - np.eye(3))))
    if orthogonal_error > 1e-5 or abs(np.linalg.det(r)-1) > 1e-5:
        raise AdapterError("invalid_calibration", "candidate optical rotation not proper orthogonal")
    return {"rotation": r, "translation": transform[:3, 3], "intrinsics": k,
            "distortion": None, "transform": transform,
            "provenance": {"candidate_only": True, "orthogonality_max": orthogonal_error,
                           "determinant": float(np.linalg.det(r)), "raw_calibration": sensor,
                           "source": raw.provenance["source_files"]["anno"],
                           "optical_axis_status": "explicit optical convention, not pixel-level equivalence certification",
                           "distortion_status": "missing source coefficients; NOT fabricated zeros"}}


def to_partial_navsim(bundle, config=None):
    from navsim.common.dataclasses import (AgentInput, Camera, Cameras, EgoStatus, Frame,
                                          Lidar, Scene, SceneMetadata)
    c = config or CandidateConfig()
    if not math.isclose(c.target_dt_s, 0.5, abs_tol=1e-12, rel_tol=0):
        raise AdapterError("navsim_interval_mismatch", "original Scene fixes NAVSIM_INTERVAL_LENGTH=0.5s")
    if c.provenance()["policy_sha256"] != bundle.provenance["policy_sha256"]:
        raise AdapterError("policy_mismatch", "object bundle and NAVSIM conversion policies differ")
    camera_map = {"cam_l0": "CAM_FRONT_LEFT", "cam_f0": "CAM_FRONT", "cam_r0": "CAM_FRONT_RIGHT"}
    names = ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
    raw = bundle.current_sensors
    loaded, calibration = {}, {}
    for target, name in camera_map.items():
        cal = candidate_camera_calibration(raw, name, c)
        calibration[target] = cal["provenance"]
        loaded[target] = Camera(image=raw.cameras[name], sensor2lidar_rotation=cal["rotation"],
                                sensor2lidar_translation=cal["translation"], intrinsics=cal["intrinsics"], distortion=None)
    cameras = Cameras(**{name: loaded.get(name, Camera()) for name in names})
    lidar = Lidar(lidar_pc=candidate_lidar_xyz(raw, c).T)  # Only real XYZ: 3xN, no false padding.
    annotation_view = candidate_vehicle_boxes(raw, c)
    current_ref_pose = bundle.frames[len(bundle.window.history)-1].rear_world_carla
    frames = []
    history_statuses, history_cameras, history_lidars = [], [], []
    for i, item in enumerate(bundle.frames):
        state = item.causal_state
        status = EgoStatus(ego_pose=item.global_pose_nav,
                           ego_velocity=state.velocity if state else None,
                           ego_acceleration=state.acceleration if state else None,
                           driving_command=None, in_global_frame=True)
        is_current = item.raw.ref == bundle.window.current
        frame_cameras = cameras if is_current else Cameras(**{name: Camera() for name in names})
        frame_lidar = lidar if is_current else Lidar()
        frames.append(Frame(token=item.raw.ref.token, timestamp=None, roadblock_ids=None,
                            traffic_lights=None, annotations=annotation_view if is_current else None,
                            ego_status=status, cameras=frame_cameras, lidar=frame_lidar))
        if i < len(bundle.window.history):
            local_pose = np.asarray(target_relative(current_ref_pose, item.rear_world_carla), dtype=np.float64)
            history_statuses.append(EgoStatus(ego_pose=local_pose, ego_velocity=status.ego_velocity,
                                             ego_acceleration=status.ego_acceleration,
                                             driving_command=None, in_global_frame=False))
            history_cameras.append(frame_cameras)
            history_lidars.append(frame_lidar)
    match = re.search(r"_(Town\d+)_", bundle.window.current.route.name)
    map_name = "carla:" + (match.group(1) if match else "unresolved-town")
    metadata = SceneMetadata(log_name=bundle.window.current.route.name,
                             scene_token=bundle.window.current.token, map_name=map_name,
                             initial_token=bundle.window.current.token,
                             num_history_frames=len(bundle.window.history), num_future_frames=len(bundle.window.scene_future))
    scene = Scene(metadata, None, frames)
    agent_input = AgentInput(history_statuses, history_cameras, history_lidars)
    missing = {"navigation": "None; no certified NAVSIM command4 mapping",
               "map_api": "None; CARLA Town is not NuPlan map name/API",
               "absolute_timestamp": "None; candidate relative times only",
               "route_roadblock_ids_and_traffic_lights": "None; CARLA IDs not NuPlan IDs",
               "full_annotations": "vehicle box view only; no velocity_3d/certified persistent tokens/all classes",
               "lidar_extra_rows": "intensity/ring/lidar_id not extracted; real XYZ only",
               "camera_distortion": "None; source coefficients missing",
               "unused_requested_cameras": "original Agents request all current cameras; this diagnostic bridge only maps Builder-consumed L0/F0/R0",
               "physics_and_clock": "candidate; collection asset identity/time unverified"}
    bundle.provenance["camera_calibration"] = calibration
    return PartialNavsimObjects(agent_input, scene, bundle, missing)
