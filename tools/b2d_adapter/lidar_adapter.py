"""Actor-ego LAZ → candidate rear/left axes → V2-style histogram.

No world2lidar inversion: prior samples prove LAZ is already Actor ego.
Output row indexes x, column indexes y, with increasing bin edges. Height origin
is unverified. See transfuser_features.py:77-118 and coordinate_validation.md §3.
"""

from .config import CandidateConfig
from .errors import AdapterError


def candidate_lidar_xyz(frame, config=None):
    """Shared candidate reference/axis conversion; no binning or input mutation."""
    import numpy as np

    c = config or CandidateConfig()
    if frame.ego_box.get("type_id") != c.axle_vehicle_type:
        raise AdapterError("reference_vehicle_mismatch", frame.ego_box.get("type_id"))
    if frame.lidar_xyz is None:
        raise AdapterError("missing_modality", "decoded LAZ XYZ required")
    xyz = np.asarray(frame.lidar_xyz, dtype=np.float64).copy()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise AdapterError("invalid_lidar", "finite N x 3 XYZ required")
    xyz[:, 0] -= c.axle_dx_m
    xyz[:, 1] = -(xyz[:, 1] - c.axle_dy_m)
    xyz[:, 2] += c.lidar_z_shift_m
    return xyz


def candidate_lidar_feature(frame, config=None):
    import numpy as np

    c = config or CandidateConfig()
    xyz = candidate_lidar_xyz(frame, c)
    keep = (xyz[:, 2] < c.lidar_max_height_m) & (xyz[:, 2] > c.lidar_split_height_m)
    bins = int(round((c.lidar_max_xy_m - c.lidar_min_xy_m) * c.lidar_pixels_per_m))
    edges = np.linspace(c.lidar_min_xy_m, c.lidar_max_xy_m, bins + 1)
    hist = np.histogramdd(xyz[keep, :2], bins=(edges, edges))[0]
    value = (np.minimum(hist, c.lidar_count_clip) / c.lidar_count_clip).astype(np.float32)[None]
    return {"candidate_lidar_feature": value, "candidate_only": True,
            "height_status": "candidate raw Actor z plus explicit shift; NAVSIM height origin unverified",
            "pixel_axes": "[channel,x_bin_increasing,y_bin_increasing]; final edge inclusive",
            "height_kept_count": int(keep.sum()), "in_grid_count": int(hist.sum()),
            "provenance": c.provenance()}
