#!/usr/bin/env python3
"""Inspect one NAVSIM log/PCD and one short Bench2Drive annotation window.

This tool only reads explicitly supplied files. It does not build training data.
"""

import argparse
import collections
import gzip
import json
import math
import pickle
from pathlib import Path

import numpy as np


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Permit only NumPy constructors used by the inspected OpenScene logs."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError("Disallowed pickle global: {}.{}".format(module, name))
        return super().find_class(module, name)


def load_log(path):
    with path.open("rb") as stream:
        frames = NumpyOnlyUnpickler(stream).load()
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], dict):
        raise ValueError("Expected a nonempty OpenScene list of frame dictionaries")
    return frames


def local_future_pose(frame, future):
    start = np.asarray(frame["ego2global"])
    end = np.asarray(future["ego2global"])
    xy = start[:2, :2].T.dot(end[:2, 3] - start[:2, 3])
    start_yaw = math.atan2(start[1, 0], start[0, 0])
    end_yaw = math.atan2(end[1, 0], end[0, 0])
    yaw = (end_yaw - start_yaw + math.pi) % (2 * math.pi) - math.pi
    return [float(xy[0]), float(xy[1]), math.degrees(yaw)]


def summarize_status(frames):
    velocity_errors = []
    acceleration_errors = []
    for index in range(1, len(frames) - 1):
        prev, current, future = frames[index - 1:index + 2]
        p0 = np.asarray(prev["ego2global"])
        p1 = np.asarray(current["ego2global"])
        p2 = np.asarray(future["ego2global"])
        seconds = (future["timestamp"] - prev["timestamp"]) * 1e-6
        if seconds <= 0:
            continue
        rotation = p1[:2, :2]
        state = np.asarray(current["ego_dynamic_state"])
        estimated_velocity = rotation.T.dot((p2[:2, 3] - p0[:2, 3]) / seconds)
        velocity_errors.append(estimated_velocity - state[:2])
        world_v0 = p0[:2, :2].dot(np.asarray(prev["ego_dynamic_state"])[:2])
        world_v2 = p2[:2, :2].dot(np.asarray(future["ego_dynamic_state"])[:2])
        estimated_acceleration = rotation.T.dot((world_v2 - world_v0) / seconds)
        acceleration_errors.append(estimated_acceleration - state[2:4])
    return {
        "velocity_mae_mps_xy": np.mean(np.abs(velocity_errors), axis=0).tolist(),
        "acceleration_mae_mps2_xy": np.mean(np.abs(acceleration_errors), axis=0).tolist(),
        "finite_difference_checks": len(velocity_errors),
    }


def read_pcd(path):
    fields = {}
    with path.open("rb") as stream:
        for _ in range(20):
            line = stream.readline().decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            fields[key] = values
            if key == "DATA":
                break
        if fields.get("FIELDS") != ["x", "y", "z", "intensity", "lidar_info", "ring"]:
            raise ValueError("Unexpected PCD fields: {}".format(fields.get("FIELDS")))
        if (
            fields.get("SIZE") != ["4", "4", "4", "1", "1", "1"]
            or fields.get("TYPE") != ["F", "F", "F", "U", "U", "U"]
            or fields.get("DATA") != ["binary"]
        ):
            raise ValueError("Unexpected PCD binary layout")
        point_count = int(fields["POINTS"][0])
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("intensity", "u1"), ("lidar_info", "u1"), ("ring", "u1"),
        ])
        points = np.fromfile(stream, dtype=dtype, count=point_count)
    if len(points) != point_count:
        raise ValueError("PCD truncated: {} of {} points".format(len(points), point_count))
    return fields, np.column_stack([points["x"], points["y"], points["z"]])


def box_alignment(point_xyz, annotations):
    boxes = np.asarray(annotations["gt_boxes"])[np.asarray(annotations["gt_names"]) == "vehicle"]
    results = {}
    for label, xy in (
        ("raw", point_xyz[:, :2]),
        ("flip_y", point_xyz[:, :2] * [1, -1]),
        ("swap_xy", point_xyz[:, [1, 0]]),
    ):
        counts = []
        for box in boxes:
            delta = xy - box[:2]
            cosine, sine = math.cos(box[6]), math.sin(box[6])
            length_axis = cosine * delta[:, 0] + sine * delta[:, 1]
            width_axis = -sine * delta[:, 0] + cosine * delta[:, 1]
            inside = (
                (np.abs(length_axis) < box[3] / 2 + 0.2)
                & (np.abs(width_axis) < box[4] / 2 + 0.2)
                & (np.abs(point_xyz[:, 2] - box[2]) < box[5] / 2 + 0.3)
            )
            counts.append(int(inside.sum()))
        results[label] = {
            "vehicle_boxes_with_at_least_10_points": sum(count >= 10 for count in counts),
            "total_points_in_vehicle_boxes": sum(counts),
        }
    results["vehicle_box_count"] = len(boxes)
    return results


def summarize_navsim(frames, log_path, nav_root, frame_index, future_frames):
    if not (0 <= frame_index < len(frames) - future_frames):
        raise ValueError("NAVSIM frame index does not leave enough future frames")
    selected = frames[frame_index]
    commands = collections.defaultdict(list)
    for index, frame in enumerate(frames[:-future_frames]):
        command = tuple(int(value) for value in frame["driving_command"])
        commands[command].append(local_future_pose(frame, frames[index + future_frames]))
    sample_deltas = np.diff([frame["timestamp"] for frame in frames[:41]]) * 1e-6
    lidar2ego_errors = [float(np.max(np.abs(np.asarray(frame["lidar2ego"]) - np.eye(4))))
                         for frame in frames[:40]]
    sensor_root = nav_root / "sensor_blobs" / log_path.parent.name
    lidar_path = sensor_root / selected["lidar_path"]
    pcd_fields, point_xyz = read_pcd(lidar_path)
    cameras = {}
    for name in ("CAM_L0", "CAM_F0", "CAM_R0"):
        camera = selected["cams"][name]
        optical_axis = np.asarray(camera["sensor2lidar_rotation"])[:, 2]
        cameras[name] = {
            "image_exists": (sensor_root / camera["data_path"]).is_file(),
            "fx_px": float(camera["cam_intrinsic"][0, 0]),
            "optical_axis_yaw_deg": math.degrees(math.atan2(optical_axis[1], optical_axis[0])),
            "distortion": np.asarray(camera["distortion"]).tolist(),
        }
    return {
        "log": str(log_path),
        "frames_in_one_log": len(frames),
        "selected_frame": frame_index,
        "timestamp_interval_s_first_40": {
            "median": float(np.median(sample_deltas)),
            "min": float(np.min(sample_deltas)),
            "max": float(np.max(sample_deltas)),
        },
        "commands": {
            "".join(map(str, command)): {
                "count": len(poses),
                "future_local_pose_m_m_deg_median": np.median(poses, axis=0).tolist(),
            } for command, poses in sorted(commands.items())
        },
        "future_frame_offset": future_frames,
        "status_finite_difference": summarize_status(frames),
        "lidar2ego_max_abs_error_first_40": max(lidar2ego_errors),
        "pcd": {
            "path": str(lidar_path),
            "fields": pcd_fields["FIELDS"],
            "point_count": len(point_xyz),
            "box_alignment": box_alignment(point_xyz, selected["anns"]),
        },
        "cameras": cameras,
    }


def summarize_b2d(route, start, count, rear_offset_m):
    top_to_rear = []
    navigation = collections.defaultdict(list)
    commands = {key: collections.Counter() for key in ("near", "far", "next")}
    for index in range(start, start + count):
        path = route / "anno" / "{:05d}.json.gz".format(index)
        with gzip.open(path, "rt") as stream:
            annotation = json.load(stream)
        ego = next(box for box in annotation["bounding_boxes"] if box.get("class") == "ego_vehicle")
        yaw = math.radians(ego["rotation"][2])
        rear_x = ego["location"][0] - rear_offset_m * math.cos(yaw)
        rear_y = ego["location"][1] - rear_offset_m * math.sin(yaw)
        top_to_rear.append(math.hypot(annotation["x"] - rear_x, annotation["y"] - rear_y))
        for name in ("near", "far"):
            target_x = annotation["x_command_" + name]
            target_y = annotation["y_command_" + name]
            navigation[name].append(math.hypot(target_x - rear_x, target_y - rear_y))
            commands[name][annotation["command_" + name]] += 1
        commands["next"][annotation["next_command"]] += 1
    return {
        "route": str(route),
        "frames": "{}..{}".format(start, start + count - 1),
        "top_xy_to_measured_rear_median_m": float(np.median(top_to_rear)),
        "top_xy_to_measured_rear_min_max_m": [min(top_to_rear), max(top_to_rear)],
        "navigation_point_distance_median_m": {
            name: float(np.median(values)) for name, values in navigation.items()
        },
        "command_counts": {name: dict(values) for name, values in commands.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nav_log", type=Path)
    parser.add_argument("nav_root", type=Path)
    parser.add_argument("b2d_route", type=Path)
    parser.add_argument("--nav-frame", type=int, default=20)
    parser.add_argument("--future-frames", type=int, default=8)
    parser.add_argument("--b2d-start", type=int, default=80)
    parser.add_argument("--b2d-count", type=int, default=50)
    parser.add_argument("--rear-offset-m", type=float, default=1.388632877)
    args = parser.parse_args()
    if args.b2d_count < 1 or args.future_frames < 1:
        parser.error("Frame counts must be positive")
    frames = load_log(args.nav_log)
    result = {
        "navsim": summarize_navsim(frames, args.nav_log, args.nav_root, args.nav_frame, args.future_frames),
        "bench2drive": summarize_b2d(args.b2d_route, args.b2d_start, args.b2d_count, args.rear_offset_m),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
