#!/usr/bin/env python3
"""Read-only two-frame pose/sign check; delta=0 is Actor diagnostic, not rear-axle GT."""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


def ego(route, frame):
    with gzip.open(route / "anno" / f"{frame:05d}.json.gz", "rt", encoding="utf-8") as stream:
        anno = json.load(stream)
    return next(box for box in anno["bounding_boxes"] if box["class"] == "ego_vehicle")


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("--rear-axle-dx", type=float, default=0.0,
                        help="Actor-local candidate offset; 0 is diagnostic, NOT a measured axle offset")
    parser.add_argument("--rear-axle-dy", type=float, default=0.0)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("end frame must exceed start frame")
    route = args.route.resolve()
    boxes = [ego(route, frame) for frame in (args.start, args.end)]
    yaw = [math.radians(box["rotation"][2]) for box in boxes]
    actor = [np.asarray(box["location"][:2], dtype=float) for box in boxes]
    delta = np.array([args.rear_axle_dx, args.rear_axle_dy], dtype=float)
    rear = [p + rotation(psi) @ delta for p, psi in zip(actor, yaw)]
    q_car = rotation(-yaw[0]) @ (rear[1] - rear[0])
    q_target = np.array([q_car[0], -q_car[1]])
    heading = math.atan2(math.sin(-(yaw[1] - yaw[0])), math.cos(-(yaw[1] - yaw[0])))
    roundtrip = rear[0] + rotation(yaw[0]) @ np.array([q_target[0], -q_target[1]])
    matrix_q = None
    if np.allclose(delta, 0):
        world2ego = np.asarray(boxes[0]["world2ego"], dtype=float)
        h1 = np.r_[boxes[1]["location"], 1.0]
        h0 = np.r_[boxes[0]["location"], 1.0]
        matrix_q = (world2ego @ h1 - world2ego @ h0)[:2].tolist()
    print(json.dumps({"route": str(route), "frames": [args.start, args.end],
                      "actor_to_rear_axle_delta_candidate_m": delta.tolist(),
                      "candidate_only": True, "carla_local_xy_m": q_car.tolist(),
                      "target_local_xy_m": q_target.tolist(), "target_heading_deg": math.degrees(heading),
                      "rear_axle_world_roundtrip_error_m": float(np.linalg.norm(roundtrip - rear[1])),
                      "world2ego_matrix_actor_displacement_xy_m": matrix_q},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
