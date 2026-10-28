"""Read raw B2D data only; no rear-axle, navigation or normalization policy."""

from dataclasses import dataclass
import gzip
import json

from .coordinate_transform import ActorPose, actor_pose
from .errors import AdapterError
from .route_index import CAMERAS, FrameRef


@dataclass
class RawFrame:
    ref: FrameRef
    annotation: dict
    ego_box: dict
    actor_pose: ActorPose
    state: dict
    navigation: dict
    sensors: dict
    cameras: dict
    lidar_xyz: object
    lidar_metadata: dict
    provenance: dict


class FrameReader:
    def __init__(self, index):
        self.index = index

    def read(self, ref, cameras=False, lidar=False):
        path = self.index.path(ref, "anno")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                anno = json.load(stream)
        except (OSError, ValueError, EOFError) as exc:
            raise AdapterError("annotation_decode_failed", "{}: {}".format(path, exc)) from exc
        if not isinstance(anno, dict) or not isinstance(anno.get("bounding_boxes"), list):
            raise AdapterError("invalid_annotation", "bounding_boxes list required")
        ego = [b for b in anno["bounding_boxes"] if isinstance(b, dict) and b.get("class") == "ego_vehicle"]
        if len(ego) != 1:
            raise AdapterError("invalid_ego_count", "{} ego boxes".format(len(ego)))
        pose = actor_pose(ego[0])
        images, xyz, lidar_meta = {}, None, {}
        paths = {"anno": str(path)}
        if cameras:
            try:
                import numpy as np
                from PIL import Image
            except (ImportError, AttributeError) as exc:
                raise AdapterError("dependency_missing", "camera requires numpy/Pillow") from exc
            for name in CAMERAS:
                image_path = self.index.path(ref, name)
                try:
                    with Image.open(image_path) as im:
                        im.load()
                        if im.mode != "RGB":
                            raise AdapterError("invalid_camera_mode", "{} {}".format(image_path, im.mode))
                        images[name] = np.array(im, dtype=np.uint8)
                except OSError as exc:
                    raise AdapterError("camera_decode_failed", image_path) from exc
                paths[name] = str(image_path)
        if lidar:
            try:
                import numpy as np
                import laspy
            except ImportError as exc:
                raise AdapterError("dependency_missing", "LiDAR requires numpy/laspy and LAZ backend") from exc
            lidar_path = self.index.path(ref, "lidar")
            try:
                las = laspy.read(lidar_path)
                xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
                if not np.isfinite(xyz).all():
                    raise AdapterError("invalid_lidar", "nonfinite XYZ")
                lidar_meta = {"dimensions": list(las.point_format.dimension_names),
                              "scales": las.header.scales.tolist(), "offsets": las.header.offsets.tolist(),
                              "point_count": len(xyz), "shape": list(xyz.shape),
                              "dtype": str(xyz.dtype), "coordinate_status": "Actor ego; prior sample evidence"}
            except AdapterError:
                raise
            except Exception as exc:
                raise AdapterError("lidar_decode_failed", "{}: {}".format(lidar_path, exc)) from exc
            paths["lidar"] = str(lidar_path)
        sensors = anno.get("sensors", {})
        if not isinstance(sensors, dict):
            raise AdapterError("invalid_annotation", "sensors must be a dictionary")
        state = {k: anno[k] for k in ("x", "y", "theta", "speed", "acceleration", "angular_velocity", "teleport_flag") if k in anno}
        # Preserve unknown navigation fields verbatim, never encode from future motion.
        nav = {k: v for k, v in anno.items() if "command" in k or k in ("x_target", "y_target")}
        return RawFrame(ref, anno, ego[0], pose, state, nav, sensors, images, xyz, lidar_meta,
                        {"source_files": paths, "pose_source": "bounding_boxes[class=ego_vehicle].location/rotation[2]",
                         "frame_id_is_timestamp": False, "timestamp_s": None,
                         "timestamp_status": "not extracted; version-matched clock required",
                         "implicit_transforms": False})
