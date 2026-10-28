#!/usr/bin/env python3
"""Read a small, explicit Bench2Drive route window and print a JSON summary.

This script does not write to the dataset and never traverses other routes.
"""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path


def read_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def names(path, suffix):
    if not path.is_dir():
        return []
    return sorted(int(p.name[: -len(suffix)]) for p in path.glob("*" + suffix)
                  if p.name[: -len(suffix)].isdigit())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    route = args.route.resolve()
    assert args.count > 0 and args.start >= 0
    frame_ids = range(args.start, args.start + args.count)
    modalities = {
        "anno": (route / "anno", ".json.gz"),
        "rgb_front": (route / "camera/rgb_front", ".jpg"),
        "lidar": (route / "lidar", ".laz"),
        "radar": (route / "radar", ".h5"),
        "lidar_seg": (route / "lidar_seg", ".npz"),
        "occ": (route / "occ", ".npz"),
    }
    coverage = {}
    for modality, (folder, suffix) in modalities.items():
        ids = names(folder, suffix)
        present = set(ids)
        coverage[modality] = {
            "total": len(ids),
            "first": ids[0] if ids else None,
            "last": ids[-1] if ids else None,
            "missing_in_window": [i for i in frame_ids if i not in present],
            "gaps_within_route": sum(b - a - 1 for a, b in zip(ids, ids[1:])),
        }

    frames = []
    obj_classes = Counter()
    keysets = Counter()
    sensor_keysets = Counter()
    for frame_id in frame_ids:
        path = route / "anno" / f"{frame_id:05d}.json.gz"
        if not path.is_file():
            continue
        anno = read_json(path)
        ego = next((obj for obj in anno.get("bounding_boxes", [])
                    if obj.get("class") == "ego_vehicle"), None)
        obj_classes.update(obj.get("class", "<missing>")
                           for obj in anno.get("bounding_boxes", []))
        keysets[tuple(sorted(anno))] += 1
        sensor_keysets[tuple(sorted(anno.get("sensors", {})))] += 1
        frames.append({
            "id": frame_id,
            "x": anno.get("x"), "y": anno.get("y"),
            "theta": anno.get("theta"), "speed": anno.get("speed"),
            "ego_location": ego.get("location") if ego else None,
            "ego_rotation": ego.get("rotation") if ego else None,
            "ego_speed": ego.get("speed") if ego else None,
            "command_near": anno.get("command_near"),
            "command_far": anno.get("command_far"),
            "next_command": anno.get("next_command"),
            "object_count": len(anno.get("bounding_boxes", [])),
            "timestamp_keys": [key for key in anno if "time" in key.lower()],
        })

    steps = []
    for left, right in zip(frames, frames[1:]):
        dx = right["x"] - left["x"]
        dy = right["y"] - left["y"]
        dtheta = math.atan2(math.sin(right["theta"] - left["theta"]),
                            math.cos(right["theta"] - left["theta"]))
        steps.append({"from": left["id"], "to": right["id"],
                      "distance_xy_m": round(math.hypot(dx, dy), 6),
                      "delta_theta_rad": round(dtheta, 6)})

    result = {
        "route": str(route), "window": [args.start, args.start + args.count - 1],
        "coverage": coverage,
        "camera_folders": sorted(p.name for p in (route / "camera").iterdir()
                                 if p.is_dir()) if (route / "camera").is_dir() else [],
        "top_level_keysets": [{"frames": count, "keys": list(keys)}
                              for keys, count in keysets.items()],
        "sensor_keysets": [{"frames": count, "keys": list(keys)}
                           for keys, count in sensor_keysets.items()],
        "object_classes_in_window": dict(obj_classes),
        "frames": frames, "steps": steps,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
