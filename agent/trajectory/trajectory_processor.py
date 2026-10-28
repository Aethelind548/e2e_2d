"""Structural validation only; no claims of dynamic or collision feasibility."""

import math
from typing import Optional, Sequence, Union

from agent.common.errors import PendingImplementationError, TrajectoryError
from agent.common.types import CoordinateFrame, EgoState, ReferenceTrajectory, TrajectoryPoint, TrajectoryPrediction


class TrajectoryProcessor:
    def __init__(self):
        self.last_reference: Optional[ReferenceTrajectory] = None

    @staticmethod
    def validate(trajectory: Union[TrajectoryPrediction, ReferenceTrajectory]) -> None:
        if not trajectory.points:
            raise TrajectoryError("Empty trajectory")
        if (trajectory.frame == CoordinateFrame.UNKNOWN or not trajectory.source
                or not math.isfinite(trajectory.timestamp.seconds)):
            raise TrajectoryError("Trajectory needs a known frame, source and finite timestamp")
        previous_time = 0.0
        for point in trajectory.points:
            values = (point.x_m, point.y_m, point.heading_rad, point.time_from_start_s)
            if not all(math.isfinite(v) for v in values):
                raise TrajectoryError("Trajectory contains NaN or Inf")
            if point.time_from_start_s <= previous_time:
                raise TrajectoryError("Future trajectory offsets must be positive and strictly increasing")
            previous_time = point.time_from_start_s

    def process(self, prediction: TrajectoryPrediction, ego: EgoState) -> ReferenceTrajectory:
        self.validate(prediction)
        if prediction.timestamp != ego.timestamp:
            raise TrajectoryError("Current V0.1 reference requires matching prediction/ego timestamps")
        # Pass through explicit coordinates/times. Controllers must request any
        # additional frame conversion, resampling or reference-point policy.
        reference = ReferenceTrajectory(prediction.timestamp, prediction.points,
                                        prediction.frame, prediction.source, prediction.origin)
        self.last_reference = reference
        return reference

    def transform(self, trajectory: ReferenceTrajectory, target_frame: CoordinateFrame
                  ) -> ReferenceTrajectory:
        self.validate(trajectory)
        if trajectory.frame == target_frame:
            return trajectory
        # TODO: bind trajectory origin, verified rear offset and heading sign;
        # reuse adapters.coordinate_transform instead of duplicating planar math.
        raise PendingImplementationError("Trajectory frame conversion requires verified origin/reference policy")

    def resample(self, trajectory: ReferenceTrajectory, sample_times_s: Sequence[float]) -> ReferenceTrajectory:
        # TODO: define interpolation, angle unwrapping and horizon boundary policy.
        raise PendingImplementationError("Trajectory resampling policy is not established")

    def select_control_point(self, trajectory: ReferenceTrajectory, ego: EgoState) -> TrajectoryPoint:
        # TODO: choose lookahead/time policy together with controller dynamics.
        raise PendingImplementationError("Control reference-point policy is not established")

    def reset(self) -> None:
        self.last_reference = None
