"""Explicit process boundary for an existing LAZ decoder environment.

Only real XYZ plus JSON metadata cross the boundary as an allow_pickle=False
NPZ byte stream. No foreign site-packages are inserted in the caller process.
Run worker from the project root with python -m tools.b2d_adapter.lidar_decode.
"""

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys

from .errors import AdapterError


def decode_lidar_in_process(ref, decoder_python):
    import numpy as np

    executable = Path(decoder_python)
    if not executable.is_file():
        raise AdapterError("decoder_unavailable", executable)
    root = Path(__file__).resolve().parents[2]
    command = [str(executable), "-m", "tools.b2d_adapter.lidar_decode",
               "--route", str(ref.route), "--frame", str(ref.frame_id)]
    try:
        result = subprocess.run(command, cwd=str(root), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdapterError("decoder_failed", str(exc)) from exc
    if result.returncode:
        raise AdapterError("decoder_failed", result.stderr.decode("utf-8", errors="replace"))
    try:
        with np.load(io.BytesIO(result.stdout), allow_pickle=False) as archive:
            xyz = archive["xyz"].copy()
            metadata = json.loads(str(archive["metadata"].item()))
    except (OSError, KeyError, ValueError) as exc:
        raise AdapterError("decoder_protocol_failed", exc) from exc
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise AdapterError("invalid_lidar", "decoder must supply finite Nx3 real XYZ")
    if metadata.get("frame_id") != ref.frame_id or metadata.get("route") != str(ref.route):
        raise AdapterError("decoder_identity_mismatch", ref.token)
    metadata["decoder_boundary"] = {"python": str(executable), "transport": "NPZ bytes, allow_pickle=False",
                                    "caller_site_packages_modified": False}
    return xyz, metadata


def main():
    # Reader performs the existing LAZ parsing; no duplicate decoder implementation.
    import numpy as np
    from .frame_reader import FrameReader
    from .route_index import RouteIndex

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    args = parser.parse_args()
    try:
        index = RouteIndex(args.route)
        frame = FrameReader(index).read(index.ref(args.frame), lidar=True)
        metadata = dict(frame.lidar_metadata, route=str(index.route), frame_id=args.frame,
                        source_files=frame.provenance["source_files"])
        stream = io.BytesIO()
        np.savez_compressed(stream, xyz=frame.lidar_xyz, metadata=np.array(json.dumps(metadata)))
        sys.stdout.buffer.write(stream.getvalue())
        return 0
    except AdapterError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
