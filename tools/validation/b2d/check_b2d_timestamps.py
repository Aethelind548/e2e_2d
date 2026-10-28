#!/usr/bin/env python3
"""Bounded, read-only clock/ego metadata inspection for explicit B2D routes.

Usage: python check_b2d_timestamps.py ROUTE [ROUTE ...] --start 80 --count 50
The motion-derived interval is diagnostic, not a recorded timestamp.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import laspy
import numpy as np


def load_anno(route, frame):
    with gzip.open(route / "anno" / f"{frame:05d}.json.gz", "rt") as stream:
        return json.load(stream)


def time_key_paths(value, prefix="", depth=0):
    if depth > 4:
        return []
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if any(term in key.lower() for term in ("time", "frame", "tick")):
                out.append(path)
            out.extend(time_key_paths(item, path, depth + 1))
        return out
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return time_key_paths(value[0], prefix + "[0]", depth + 1)
    return []


def inspect(route, start, count):
    if not route.is_dir():
        raise FileNotFoundError(route)
    ids = sorted(int(p.name[:5]) for p in (route / "anno").glob("[0-9][0-9][0-9][0-9][0-9].json.gz"))
    if len(ids) < 2 or ids != list(range(ids[0], ids[-1] + 1)):
        raise ValueError(f"Annotation numbering is not contiguous: {route}")
    chosen = list(range(start, start + count))
    if not chosen or chosen[0] < ids[0] or chosen[-1] > ids[-1]:
        raise ValueError(f"Window outside Route: {route}")
    annos = [load_anno(route, frame) for frame in chosen]
    egos = [next(box for box in anno["bounding_boxes"] if box.get("class") == "ego_vehicle") for anno in annos]
    xy = np.array([ego["location"][:2] for ego in egos], dtype=float)
    speed = np.array([anno["speed"] for anno in annos], dtype=float)
    mid_speed = (speed[1:] + speed[:-1]) / 2
    diagnostic = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(mid_speed, 1e-8)
    diagnostic = diagnostic[mid_speed >= 2]
    duration = json.load((route / "result.json").open())["_checkpoint"]["records"][0]["meta"]["duration_game"]
    inspected_annos = [load_anno(route, ids[0]), *annos, load_anno(route, ids[-1])]
    sample_frame = chosen[0]
    jpeg = route / "camera" / "rgb_front" / f"{sample_frame:05d}.jpg"
    laz = route / "lidar" / f"{sample_frame:05d}.laz"
    radar = route / "radar" / f"{sample_frame:05d}.h5"
    expert = route / "expert_assessment" / f"{sample_frame - 1:05d}.npz"
    with laspy.open(laz) as reader:
        point_fields = list(reader.header.point_format.dimension_names)
    with np.load(expert) as archive:
        expert_keys = {key: list(archive[key].shape) for key in archive.files}
    with jpeg.open("rb") as stream:
        jpeg_prefix = stream.read(65536)
    with radar.open("rb") as stream:
        radar_signature = stream.read(8).hex()
    top_xy = np.array([[anno["x"], anno["y"]] for anno in annos])
    yaw = np.deg2rad([ego["rotation"][2] for ego in egos])
    measured_rear_xy = xy - 1.388632877 * np.column_stack((np.cos(yaw), np.sin(yaw)))
    theta_error = [math.atan2(math.sin(anno["theta"] - angle - math.pi / 2),
                              math.cos(anno["theta"] - angle - math.pi / 2))
                   for anno, angle in zip(annos, yaw)]
    return {
        "route": str(route),
        "annotation_ids_first_last_count": [ids[0], ids[-1], len(ids)],
        "window": [chosen[0], chosen[-1]],
        "timestamp_like_anno_paths_first_window_last": sorted(set(
            path for anno in inspected_annos for path in time_key_paths(anno)
        )),
        "duration_game_s": duration,
        "duration_minus_0p1_times_nplus1_s": duration - (len(ids) + 1) * 0.1,
        "motion_div_speed_diagnostic_s": {
            "count": len(diagnostic),
            "mean": float(np.mean(diagnostic)) if len(diagnostic) else None,
            "median": float(np.median(diagnostic)) if len(diagnostic) else None,
            "min": float(np.min(diagnostic)) if len(diagnostic) else None,
            "max": float(np.max(diagnostic)) if len(diagnostic) else None,
            "p95": float(np.percentile(diagnostic, 95)) if len(diagnostic) else None,
            "outside_0p09_to_0p11_count": int(((diagnostic < 0.09) | (diagnostic > 0.11)).sum()),
        },
        "ego_type_ids": sorted(set(ego["type_id"] for ego in egos)),
        "ego_bbox_local_and_extent_first": [egos[0]["bbx_loc"], egos[0]["extent"]],
        "top_xy_minus_actor_xy_m_median": float(np.median(np.linalg.norm(top_xy - xy, axis=1))),
        "top_xy_minus_candidate_rear_xy_m_median": float(np.median(np.linalg.norm(top_xy - measured_rear_xy, axis=1))),
        "theta_vs_actor_yaw_plus_pi2_max_abs_rad": float(np.max(np.abs(theta_error))),
        "sample_sensor_files": {
            "jpeg": {"exists": jpeg.is_file(), "has_exif_marker_first_64k": b"Exif\x00\x00" in jpeg_prefix},
            "laz": {"exists": laz.is_file(), "point_fields": point_fields},
            "radar": {"exists": radar.is_file(), "signature_hex": radar_signature},
            "expert": {"exists": expert.is_file(), "arrays": expert_keys},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("routes", type=Path, nargs="+")
    parser.add_argument("--start", type=int, default=80)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    if args.count < 2:
        parser.error("--count must be at least 2")
    print(json.dumps([inspect(route, args.start, args.count) for route in args.routes], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
