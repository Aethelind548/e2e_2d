#!/usr/bin/env python3
"""Bounded real-route checks. Prints JSON; never writes source data or runs a model."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.b2d_adapter import (AdapterError, CandidateConfig, FrameReader, RouteIndex,
                          CandidateTrajectoryTargetBuilder)
from tools.b2d_adapter.camera_adapter import camera_metadata, candidate_camera_feature
from tools.b2d_adapter.lidar_adapter import candidate_lidar_feature
from tools.b2d_adapter.config import DEFAULT_B2D_DATA_ROOT


def validate(route, start, count, config, full_scene):
    index = RouteIndex(route)
    reader = FrameReader(index)
    # Read a small current continuous interval, not the whole route.
    frames = [reader.read(index.ref(i)) for i in range(start, start + count)]
    target = CandidateTrajectoryTargetBuilder(reader, config).build(index.ref(start), full_scene)
    sensor_frame = reader.read(index.ref(start), cameras=True, lidar=True)
    camera = candidate_camera_feature(sensor_frame)
    lidar = candidate_lidar_feature(sensor_frame, config)
    arrays = {"candidate_trajectory": target.candidate_trajectory,
              "candidate_camera_feature": camera["candidate_camera_feature"],
              "candidate_lidar_feature": lidar["candidate_lidar_feature"]}
    import numpy as np

    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise AdapterError("nonfinite_output", key)
    if target.validation["double_roundtrip_max_m"] > 1e-5:
        raise AdapterError("coordinate_roundtrip_failed", target.validation)
    # Fail closed: this validator cannot promote diagnostic data to training GT.
    return {"status": "diagnostic_pass", "basic_real_sample_check": True, "formal_level_2": False,
            "route_index": index.summary(), "raw_frames_read": [f.ref.frame_id for f in frames],
            "history": [r.frame_id for r in target.window.history],
            "scene_future": [r.frame_id for r in target.window.scene_future],
            "trajectory_future": [r.frame_id for r in target.window.trajectory_future],
            "outputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                            "min": float(v.min()), "max": float(v.max())} for k, v in arrays.items()},
            "candidate_trajectory": target.candidate_trajectory.tolist(),
            "coordinate_validation": target.validation,
            "camera_metadata": camera_metadata(sensor_frame), "camera_policy": camera["policy"],
            "lidar_metadata": sensor_frame.lidar_metadata, "lidar_height_status": lidar["height_status"],
            "raw_navigation": sensor_frame.navigation, "status_feature": "NOT built: navigation/state unresolved",
            "provenance": config.provenance()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", type=Path, nargs="?", help="explicit absolute or cwd-relative Route path")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_B2D_DATA_ROOT)
    parser.add_argument("--route-name", help="explicit path relative to data root, including v1/; no dataset scan")
    parser.add_argument("--start", type=int, default=80)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--source-dt", type=float, default=0.1)
    parser.add_argument("--axle-dx", type=float, default=-1.388632877)
    parser.add_argument("--axle-dy", type=float, default=0.0)
    parser.add_argument("--axle-vehicle-type", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--trajectory-only-window", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error("count must be 1..20 (bounded validation)")
    if bool(args.route) == bool(args.route_name):
        parser.error("provide either route or --route-name, not both")
    try:
        route = args.route
        if args.route_name:
            route = RouteIndex.from_dataset_root(args.data_root, args.route_name).route
        config = replace(CandidateConfig(), source_dt_s=args.source_dt, axle_dx_m=args.axle_dx,
                         axle_dy_m=args.axle_dy, axle_vehicle_type=args.axle_vehicle_type,
                         clock_source="CLI candidate dt {}; collection clock unverified".format(args.source_dt),
                         axle_source="CLI candidate delta; local 0.9.15 baseline, V0.0.4 identity unverified")
        result = validate(route, args.start, args.count, config, not args.trajectory_only_window)
        code = 0
    except AdapterError as exc:
        result = {"status": "blocked_or_invalid", "reason": exc.code, "detail": exc.detail,
                  "formal_level_2": False}
        code = 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
