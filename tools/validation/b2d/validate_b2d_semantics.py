#!/usr/bin/env python3
"""Read-only, single-route checks for Bench2Drive pose, timing and LiDAR frames.

Requires an existing environment with numpy and laspy. Never writes to the route.
"""

import argparse
import gzip
import importlib.util
import json
import math
from pathlib import Path

import laspy
import numpy as np


def annotation(route, frame):
    with gzip.open(route / "anno" / f"{frame:05d}.json.gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


def ego_box(anno):
    return next(box for box in anno["bounding_boxes"] if box["class"] == "ego_vehicle")


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def ids(route, subdir, suffix):
    return sorted(int(path.name[: -len(suffix)]) for path in (route / subdir).glob("*" + suffix)
                  if path.name[: -len(suffix)].lstrip("-").isdigit())


def pose_and_time(route, start, count):
    records = []
    for frame in range(start, start + count):
        anno = annotation(route, frame)
        box = ego_box(anno)
        records.append({"frame": frame, "theta": anno["theta"], "yaw_deg": box["rotation"][2],
                        "gps_xy": np.asarray([anno["x"], anno["y"]]),
                        "actor_xy": np.asarray(box["location"][:2]), "speed": anno["speed"],
                        "timestamp_keys": [key for key in anno if "time" in key.lower()]})
    angle_errors = [wrap(rec["theta"] - math.radians(rec["yaw_deg"]) - math.pi / 2)
                    for rec in records]
    gps_errors = [np.linalg.norm(rec["gps_xy"] - rec["actor_xy"]) for rec in records]
    intervals = []
    for left, right in zip(records, records[1:]):
        mean_speed = (left["speed"] + right["speed"]) / 2
        if mean_speed >= 2:
            intervals.append(float(np.linalg.norm(right["actor_xy"] - left["actor_xy"]) / mean_speed))
    return {
        "window": [start, start + count - 1], "frames": len(records),
        "yaw_deg_range": [min(r["yaw_deg"] for r in records), max(r["yaw_deg"] for r in records)],
        "theta_rad_range": [min(r["theta"] for r in records), max(r["theta"] for r in records)],
        "theta_relation_max_abs_residual_rad": max(map(abs, angle_errors)),
        "gps_actor_xy_error_m_median_max": [float(np.median(gps_errors)), max(gps_errors)],
        "motion_dt_estimate_s_n_median_min_max": [len(intervals), float(np.median(intervals)) if intervals else None,
                                                   min(intervals) if intervals else None,
                                                   max(intervals) if intervals else None],
        "timestamp_keys": sorted(set(key for rec in records for key in rec["timestamp_keys"])),
    }


def matrix_and_lidar(route, frame):
    anno = annotation(route, frame)
    ego = ego_box(anno)
    matrices = {"ego": np.asarray(ego["world2ego"]),
                "cam": np.asarray(anno["sensors"]["CAM_FRONT"]["world2cam"]),
                "lidar": np.asarray(anno["sensors"]["LIDAR_TOP"]["world2lidar"])}
    origins = {"ego": np.asarray(ego["location"]),
               "cam": np.asarray(anno["sensors"]["CAM_FRONT"]["location"]),
               "lidar": np.asarray(anno["sensors"]["LIDAR_TOP"]["location"])}
    matrix_checks = {}
    for name, mat in matrices.items():
        origin = np.r_[origins[name], 1.0]
        matrix_checks[name] = {
            "origin_to_zero_m": float(np.linalg.norm((mat @ origin)[:3])),
            "rotation_orthogonality_max_abs": float(np.max(np.abs(mat[:3, :3].T @ mat[:3, :3] - np.eye(3)))),
            "rotation_determinant": float(np.linalg.det(mat[:3, :3])),
            "inverse_roundtrip_max_abs": float(np.max(np.abs(np.linalg.inv(mat) @ mat - np.eye(4)))),
        }
    ego_to_cam = matrices["cam"] @ np.linalg.inv(matrices["ego"])
    ego_to_lidar = matrices["lidar"] @ np.linalg.inv(matrices["ego"])
    with laspy.open(route / "lidar" / f"{frame:05d}.laz") as reader:
        las = reader.read()
    laz_xyz = np.column_stack((las.x, las.y, las.z))
    box_results = {}
    with np.load(route / "lidar_seg" / f"{frame:05d}.npz") as seg:
        xyz, object_ids = seg["xyz"], seg["obj_idx"]
        for name in ("ego", "lidar"):
            shares = []
            actor_count = 0
            for box in anno["bounding_boxes"]:
                if box["class"] != "vehicle" or "world2vehicle" not in box:
                    continue
                cloud = xyz[object_ids == int(box["id"])]
                if len(cloud) < 10:
                    continue
                world = np.column_stack((cloud, np.ones(len(cloud)))) @ np.linalg.inv(matrices[name]).T
                local = world @ np.asarray(box["world2vehicle"]).T
                corners = np.column_stack((box["world_cord"], np.ones(8))) @ np.asarray(box["world2vehicle"]).T
                # 0.2 m tolerance absorbs rounding and near-surface semantic returns.
                lo, hi = corners[:, :3].min(axis=0) - 0.2, corners[:, :3].max(axis=0) + 0.2
                shares.append(float(np.mean(np.all((local[:, :3] >= lo) & (local[:, :3] <= hi), axis=1))))
                actor_count += 1
            box_results[name] = {"matched_vehicles": actor_count,
                                 "median_point_share_inside_3d_box": float(np.median(shares)) if shares else None}
        laz_result = {"points": len(laz_xyz), "seg_points": len(xyz),
                      "same_order_max_abs_xyz_m": float(np.max(np.abs(laz_xyz - xyz))) if len(laz_xyz) == len(xyz) else None}
    return {"frame": frame, "matrix_checks": matrix_checks,
            "ego_to_cam_translation_m": ego_to_cam[:3, 3].tolist(),
            "ego_to_lidar_translation_m": ego_to_lidar[:3, 3].tolist(),
            "laz": laz_result, "actor_id_box_test": box_results}


def modal_indices(route):
    out = {}
    for name, subdir, suffix in [("anno", "anno", ".json.gz"), ("occ", "occ", ".npz"),
                                 ("expert_assessment", "expert_assessment", ".npz"),
                                 ("radar", "radar", ".h5")]:
        frame_ids = ids(route, subdir, suffix)
        out[name] = {"count": len(frame_ids), "first": frame_ids[0] if frame_ids else None,
                     "last": frame_ids[-1] if frame_ids else None,
                     "has_minus_one": -1 in frame_ids,
                     "gaps_nonnegative": [value for value in range(0, max(frame_ids) + 1)
                                          if value not in set(frame_ids)] if frame_ids else []}
    with (route / "radar" / "00000.h5").open("rb") as stream:
        out["radar"]["first_eight_bytes_hex"] = stream.read(8).hex()
    out["radar"]["h5py_available"] = importlib.util.find_spec("h5py") is not None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    parser.add_argument("--turn-start", type=int, default=90)
    parser.add_argument("--turn-count", type=int, default=53)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 108, 132])
    args = parser.parse_args()
    route = args.route.resolve()
    result = {"route": str(route), "straight_window": pose_and_time(route, 10, 10),
              "turn_window": pose_and_time(route, args.turn_start, args.turn_count),
              "frames": [matrix_and_lidar(route, frame) for frame in args.frames],
              "modal_indices": modal_indices(route)}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
