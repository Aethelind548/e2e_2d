"""Candidate index sampling; endpoint repetition is explicit and target-only."""

from dataclasses import dataclass

from .config import CandidateConfig
from .errors import AdapterError
from .route_index import CAMERAS, FrameRef


@dataclass(frozen=True)
class CandidateWindow:
    current: FrameRef
    history: tuple
    scene_future: tuple
    trajectory_future: tuple
    continuous: tuple
    provenance: dict


class TemporalSampler:
    def __init__(self, config=None):
        self.config = config or CandidateConfig()

    def sample(self, index, current, full_scene=True, timestamps=None, target_only=False,
               terminal_endpoint=None):
        index.check_ref(current)
        if timestamps is not None:
            # Future interface: no pretending frame IDs are physical timestamps.
            raise AdapterError("timestamp_sampling_unimplemented", "implement authenticated clock/tolerance first")
        c, t = self.config, current.frame_id
        if target_only and full_scene:
            raise AdapterError("invalid_window_mode", "target-only does not build a full Scene")
        if terminal_endpoint is not None:
            index.check_ref(terminal_endpoint)
            if not target_only or full_scene or terminal_endpoint.frame_id != index.frame_ids[-1]:
                raise AdapterError("invalid_terminal_window", "only real endpoint in target-only mode")
            index.require(current)
        history = (current,) if target_only else tuple(index.ref(t - n * c.stride) for n in reversed(range(c.history_count)) if t - n * c.stride >= 0)
        if not target_only and (len(history) != c.history_count or history[0].frame_id < index.frame_ids[0]):
            raise AdapterError("insufficient_past", current.token)
        count = c.scene_future_count if full_scene else c.trajectory_count
        end = t + count * c.stride
        if end > index.frame_ids[-1] and terminal_endpoint is None:
            raise AdapterError("insufficient_future", current.token)
        future = tuple(index.ref(min(t + n * c.stride, terminal_endpoint.frame_id)
                                 if terminal_endpoint is not None else t + n * c.stride)
                       for n in range(1, count + 1))
        end = min(end, terminal_endpoint.frame_id) if terminal_endpoint is not None else end
        continuous = tuple(index.ref(i) for i in range(history[0].frame_id, end + 1))
        # Even an unsampled intermediate missing annotation invalidates the window.
        for ref in continuous:
            index.require(ref)
        # Sensor inputs are current/history only; future labels need annotation only.
        if not target_only:
            for ref in history:
                index.require(ref, ("anno", "lidar", *CAMERAS))
        return CandidateWindow(current, history, future if full_scene else (),
                               future[:c.trajectory_count], continuous,
                               dict(c.provenance(), full_scene=full_scene, target_only=target_only,
                                    future_times_s=[n * c.target_dt_s for n in range(1, c.trajectory_count + 1)]))
