"""Online Route Preview adapter for the CARLA Agent.

This adapter is the only bridge from the validated ordered ``set_global_plan``
nodes to the shared calculator used by offline data generation.  The online
Status builder (``status_builder.py``) owns it; there is no second navigation
rule and no legacy near-bearing branch.
"""
from dataclasses import dataclass
import math
from typing import Optional, Sequence

from tools.b2d_adapter.route_preview_contract import preview_command4


@dataclass(frozen=True)
class AgentRoutePreview:
    preview: dict
    route_generation: int
    source: str = 'CARLA_SET_GLOBAL_PLAN'

    @property
    def command4(self):
        return self.preview.get('command4')

    @property
    def valid(self):
        return bool(self.preview.get('preview_valid'))


def route_preview_from_world_nodes(nodes, rear_xy, yaw, route_generation=0,
                                   previous_progress_m=None, preview_m=20.0,
                                   lateral_threshold_m=2.0, max_projection_m=5.0):
    """Run the shared calculator on validated ordered CARLA world nodes.

    ``nodes`` is a sequence of ``(x_m, y_m)`` pairs derived from
    ``set_global_plan`` in the same order as the original global plan.
    """
    points = []
    for node in nodes:
        try:
            x, y = float(node[0]), float(node[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError('ordered world nodes require finite x,y') from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError('ordered world nodes require finite x,y')
        if not points or points[-1] != (x, y):
            points.append((x, y))
    if len(points) < 2:
        raise ValueError('at least two distinct ordered route points required')
    result = preview_command4(points, rear_xy, yaw, preview_m=preview_m,
                              lateral_threshold_m=lateral_threshold_m,
                              max_projection_m=max_projection_m,
                              previous_progress_m=previous_progress_m)
    return AgentRoutePreview(preview=result, route_generation=int(route_generation))
