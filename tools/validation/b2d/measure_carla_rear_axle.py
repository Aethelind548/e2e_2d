#!/usr/bin/env python3
"""Measure Actor-origin-to-rear-axle geometry in a disposable CARLA test server.

The script spawns only its own test vehicle and destroys it before returning.
It never reads or writes Bench2Drive data. Client/server versions must match.
"""

import argparse
import json
import math
import statistics

import carla


def vec3(value):
    return [float(value.x), float(value.y), float(value.z)]


def local_point(inverse_matrix, world_point):
    point = list(world_point) + [1.0]
    return [sum(inverse_matrix[i][j] * point[j] for j in range(4)) for i in range(3)]


def measure_vehicle(vehicle, wheel_position_scale):
    wheels = vehicle.get_physics_control().wheels
    if len(wheels) != 4:
        raise RuntimeError("Expected four wheels in FL, FR, RL, RR order")
    transform = vehicle.get_transform()
    inverse = transform.get_inverse_matrix()
    wheel_world_raw = [vec3(wheel.position) for wheel in wheels]
    wheel_local = [local_point(inverse, [value * wheel_position_scale for value in raw])
                   for raw in wheel_world_raw]
    rear = [(wheel_local[2][i] + wheel_local[3][i]) / 2.0 for i in range(3)]
    front = [(wheel_local[0][i] + wheel_local[1][i]) / 2.0 for i in range(3)]
    box = vehicle.bounding_box
    return {
        "vehicle_type_id": vehicle.type_id,
        "actor_world_location_m": vec3(transform.location),
        "actor_yaw_deg": float(transform.rotation.yaw),
        "wheel_world_xyz_raw_FL_FR_RL_RR": wheel_world_raw,
        "wheel_local_xyz_m_FL_FR_RL_RR": wheel_local,
        "actor_to_rear_axle_local_xyz_m": rear,
        "wheelbase_m": front[0] - rear[0],
        "rear_left_right_x_difference_m": wheel_local[2][0] - wheel_local[3][0],
        "rear_axle_y_symmetry_error_m": rear[1],
        "bounding_box_location_local_m": vec3(box.location),
        "bounding_box_extent_m": vec3(box.extent),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--vehicle", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--wheel-position-scale", type=float, default=0.01,
                        help="multiply raw CARLA wheel.position by this factor to obtain metres; "
                             "0.01 is measured for the local CARLA 0.9.15 and 0.9.16 builds")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.wheel_position_scale <= 0:
        parser.error("--wheel-position-scale must be positive")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    client_version = client.get_client_version()
    server_version = client.get_server_version()
    if client_version != server_version:
        raise RuntimeError("Refusing mixed CARLA versions: client={} server={}".format(
            client_version, server_version))
    world = client.get_world()
    blueprint = world.get_blueprint_library().find(args.vehicle)
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No map spawn points available")

    samples = []
    for spawn in spawn_points:
        if samples:
            yaw_diff = abs((spawn.rotation.yaw - samples[0]["actor_yaw_deg"] + 180.0) % 360.0 - 180.0)
            start = samples[0]["actor_world_location_m"]
            separation = math.hypot(spawn.location.x - start[0], spawn.location.y - start[1])
            if yaw_diff < 30.0 or separation < 10.0:
                continue
        vehicle = world.try_spawn_actor(blueprint, spawn)
        if vehicle is None:
            continue
        try:
            world.wait_for_tick(10.0)
            sample = measure_vehicle(vehicle, args.wheel_position_scale)
            if not (1.5 < sample["wheelbase_m"] < 4.5
                    and abs(sample["rear_axle_y_symmetry_error_m"]) < 0.2
                    and abs(sample["rear_left_right_x_difference_m"]) < 0.2):
                raise RuntimeError("Implausible wheel geometry: check wheel.position units, "
                                   "coordinate frame, and --wheel-position-scale")
            samples.append(sample)
        finally:
            vehicle.destroy()
        if len(samples) >= args.samples:
            break
    if len(samples) < args.samples:
        raise RuntimeError("Only {} of {} requested measurements succeeded".format(len(samples), args.samples))

    rear = [sample["actor_to_rear_axle_local_xyz_m"] for sample in samples]
    result = {
        "client_version": client_version,
        "server_version": server_version,
        "map": world.get_map().name,
        "blueprint": args.vehicle,
        "wheel_position_scale_to_m": args.wheel_position_scale,
        "samples": samples,
        "actor_to_rear_axle_local_xyz_mean_m": [statistics.mean(p[i] for p in rear) for i in range(3)],
        "actor_to_rear_axle_local_xyz_range_m": [max(p[i] for p in rear) - min(p[i] for p in rear) for i in range(3)],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
