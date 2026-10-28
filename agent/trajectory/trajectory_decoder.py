"""Decode explicit poses/time metadata without guessing sampling or axes."""

from typing import Optional, Sequence

from agent.common.errors import TrajectoryError
from agent.common.types import CoordinateFrame, Pose2D, Timestamp, TrajectoryPoint, TrajectoryPrediction


class TrajectoryDecoder:
    def decode_poses(self, poses: Sequence[Sequence[float]], sample_times_s: Sequence[float],
                     timestamp: Timestamp, frame: CoordinateFrame, source: str,
                     origin: Optional[Pose2D] = None) -> TrajectoryPrediction:
        """Consume one unbatched sequence of [x metres, y metres, heading rad].

        Caller owns Tensor->CPU conversion and batch removal. Sample times and
        coordinate frame must come from the actual selected model contract.
        """
        if len(poses) != len(sample_times_s) or not len(poses):
            raise TrajectoryError("Nonempty pose/time sequences of equal length required")
        if frame == CoordinateFrame.UNKNOWN or not source:
            raise TrajectoryError("Explicit trajectory coordinate frame and source required")
        points = []
        for pose, time_s in zip(poses, sample_times_s):
            if len(pose) != 3:
                raise TrajectoryError("Each model trajectory pose must have x/y/heading")
            try:
                points.append(TrajectoryPoint(float(pose[0]), float(pose[1]), float(pose[2]), float(time_s)))
            except (TypeError, ValueError) as exc:
                raise TrajectoryError("Numeric trajectory values required") from exc
        prediction = TrajectoryPrediction(timestamp, tuple(points), frame, source, origin)
        from .trajectory_processor import TrajectoryProcessor
        TrajectoryProcessor.validate(prediction)
        return prediction
