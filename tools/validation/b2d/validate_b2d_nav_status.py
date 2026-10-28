#!/usr/bin/env python3
"""Read-only route-level checks for navigation and ego-state semantics."""

import argparse
import collections
import gzip
import json
import math
from pathlib import Path

import numpy as np


def load_route(route):
    records = []
    for path in sorted((route / "anno").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            anno = json.load(stream)
        ego = next(box for box in anno["bounding_boxes"] if box["class"] == "ego_vehicle")
        records.append((int(path.name[:5]), anno, ego))
    if any(b[0] != a[0] + 1 for a, b in zip(records, records[1:])):
        raise ValueError(f"nonconsecutive annotations in {route}")
    return records


def report(route, dt):
    records = load_route(route)
    xyz = np.array([item[2]["location"][:2] for item in records], dtype=float)
    yaw = np.deg2rad(np.array([item[2]["rotation"][2] for item in records], dtype=float))
    speed = np.array([item[1]["speed"] for item in records], dtype=float)
    imu = np.array([item[1]["acceleration"] for item in records], dtype=float)
    counts = {key: dict(sorted(collections.Counter(item[1][key] for item in records).items()))
              for key in ("command_near", "command_far", "next_command")}
    # Central differences omit route boundaries; no implicit cross-route samples.
    v_world = (xyz[2:] - xyz[:-2]) / (2 * dt)
    a_world = (xyz[2:] - 2 * xyz[1:-1] + xyz[:-2]) / (dt * dt)
    c, s = np.cos(yaw[1:-1]), np.sin(yaw[1:-1])
    v_ego = np.column_stack((c * v_world[:, 0] + s * v_world[:, 1],
                             -s * v_world[:, 0] + c * v_world[:, 1]))
    a_ego = np.column_stack((c * a_world[:, 0] + s * a_world[:, 1],
                             -s * a_world[:, 0] + c * a_world[:, 1]))
    moving = speed[1:-1] > 2
    valid = np.isfinite(imu[1:-1, :2]).all(axis=1) & moving
    def corr(left, right):
        return float(np.corrcoef(left, right)[0, 1]) if len(left) >= 3 else None
    return {
        "route": str(route), "frames": len(records), "frame_first_last": [records[0][0], records[-1][0]],
        "assumed_dt_s_for_diagnostics_only": dt,
        "yaw_unwrapped_change_deg": float(np.rad2deg(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0])),
        "command_counts": counts,
        "moving_centered_samples": int(moving.sum()),
        "median_abs_forward_velocity_minus_speed_mps": float(np.median(np.abs(v_ego[moving, 0] - speed[1:-1][moving]))),
        "median_abs_lateral_velocity_mps": float(np.median(np.abs(v_ego[moving, 1]))),
        "p95_abs_lateral_velocity_mps": float(np.percentile(np.abs(v_ego[moving, 1]), 95)),
        "imu_z_median_mps2": float(np.median(imu[:, 2])),
        "imu_xy_vs_position_second_difference": {
            axis: {"correlation": corr(a_ego[valid, i], imu[1:-1][valid, i]),
                   "mae_mps2": float(np.mean(np.abs(a_ego[valid, i] - imu[1:-1][valid, i])))}
            for i, axis in enumerate(("x", "y"))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("routes", type=Path, nargs="+")
    parser.add_argument("--diagnostic-dt", type=float, default=0.1,
                        help="Candidate interval for kinematic diagnostics, NOT asserted as dataset clock")
    args = parser.parse_args()
    if not math.isfinite(args.diagnostic_dt) or args.diagnostic_dt <= 0:
        parser.error("--diagnostic-dt must be finite and positive")
    print(json.dumps([report(route.resolve(), args.diagnostic_dt) for route in args.routes],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
