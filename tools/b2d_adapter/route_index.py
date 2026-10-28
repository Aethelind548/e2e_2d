"""Filename-only indexing of one explicit Route. No recursive dataset scan."""

from dataclasses import dataclass
from pathlib import Path
import re

from .errors import AdapterError


CAMERAS = {
    "CAM_FRONT_LEFT": "camera/rgb_front_left",
    "CAM_FRONT": "camera/rgb_front",
    "CAM_FRONT_RIGHT": "camera/rgb_front_right",
}
MODALITIES = dict(CAMERAS, anno="anno", lidar="lidar")
SUFFIXES = dict.fromkeys(CAMERAS, ".jpg")
SUFFIXES.update(anno=".json.gz", lidar=".laz")


@dataclass(frozen=True)
class FrameRef:
    route: Path
    frame_id: int

    def __post_init__(self):
        object.__setattr__(self, "route", Path(self.route).resolve())
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise AdapterError("invalid_frame_id", self.frame_id)

    @property
    def token(self):
        # Absolute Route path prevents collisions among roots with the same basename.
        return "{}::{}".format(self.route, self.frame_id)


class RouteIndex:
    @classmethod
    def from_dataset_root(cls, data_root, route_name):
        """Resolve one explicit dataset-relative Route; never search all Routes."""
        root = Path(data_root).resolve()
        name = Path(route_name)
        if name.is_absolute():
            raise AdapterError("invalid_route_name", "use a dataset-relative Route name")
        route = (root / name).resolve()
        try:
            route.relative_to(root)
        except ValueError as exc:
            raise AdapterError("invalid_route_name", "Route escapes dataset root") from exc
        return cls(route)

    def __init__(self, route):
        self.route = Path(route).resolve()
        if not self.route.is_dir():
            raise AdapterError("route_unavailable", self.route)
        self.files = {}
        for modality, folder in MODALITIES.items():
            suffix = SUFFIXES[modality]
            pattern = re.compile(r"^(\d+)" + re.escape(suffix) + r"$")
            found = {}
            for path in (self.route / folder).glob("*" + suffix):
                match = pattern.fullmatch(path.name)
                if match and path.is_file():
                    frame = int(match[1])
                    if frame in found:
                        raise AdapterError("duplicate_frame_id", "{} {}".format(modality, frame))
                    found[frame] = path
            self.files[modality] = found
        self.frame_ids = tuple(sorted(self.files["anno"]))
        if not self.frame_ids:
            raise AdapterError("empty_route", "no nonnegative annotation frame IDs")

    def ref(self, frame_id):
        return FrameRef(self.route, frame_id)

    def check_ref(self, ref):
        if ref.route != self.route:
            raise AdapterError("route_mismatch", ref.token)

    def path(self, ref, modality):
        self.check_ref(ref)
        if modality not in self.files:
            raise AdapterError("unknown_modality", modality)
        path = self.files[modality].get(ref.frame_id)
        if path is None or not path.is_file():
            raise AdapterError("missing_frame" if modality == "anno" else "missing_modality",
                               "{} {}".format(ref.token, modality))
        return path

    def require(self, ref, modalities=("anno",)):
        for modality in modalities:
            self.path(ref, modality)

    def summary(self):
        return {"route": str(self.route), "first": self.frame_ids[0],
                "last": self.frame_ids[-1], "counts": {k: len(v) for k, v in self.files.items()},
                "annotation_gaps": [i for i in range(self.frame_ids[0], self.frame_ids[-1] + 1)
                                    if i not in self.files["anno"]]}
