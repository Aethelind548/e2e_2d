"""M1: version-specific wrapper for the inspected local Bench2Drive checkout.

Imported only by get_entry_point(). Constructing the external base Agent calls
get_hero() and needs an existing world; no constructor is invoked in V0.1 work.
"""

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from agent.bench2drive_agent import Bench2DriveAgent
from agent.common.errors import ConfigurationError, SensorError
from agent.common.types import SimulationTick, Timestamp


class DiffusionDriveLeaderboardAgent(Bench2DriveAgent, AutonomousAgent):
    """Retain runner __call__, sensor interface and metric hooks from its base."""

    def __init__(self, carla_host, carla_port, debug=False):
        AutonomousAgent.__init__(self, carla_host, carla_port, debug)
        Bench2DriveAgent.__init__(self)
        self.track = Track.SENSORS
        self.run_label = None

    def setup(self, path_to_conf_file: str) -> None:
        """Local evaluator appends '+save_name' to the configured JSON path."""
        if not isinstance(path_to_conf_file, str) or not path_to_conf_file:
            raise ConfigurationError("Expected config JSON path, optionally followed by +save_name")
        config_path, separator, label = path_to_conf_file.rpartition("+")
        if separator:
            if not config_path or not label:
                raise ConfigurationError("Both config JSON path and appended save_name must be nonempty")
            self.run_label = label
        else:
            config_path = path_to_conf_file
        Bench2DriveAgent.setup(self, config_path)
        if self.config.ego_state.policy is not None:
            actor = getattr(self, 'hero_actor', None)
            if actor is None or actor.type_id != self.config.ego_state.vehicle_type:
                error = ConfigurationError("Live hero vehicle does not match the declared engineering geometry")
                self.supervisor.handle_exception(error)
                raise error

    def run_step(self, input_data, timestamp):
        """Read current tick metadata only; do not use privileged Actor motion."""
        from srunner.scenariomanager.timer import GameTime
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        import math
        try:
            snapshot = CarlaDataProvider.get_world().get_snapshot()
            frame = GameTime.get_frame()
            if snapshot.frame != frame or not math.isclose(GameTime.get_time(), timestamp, abs_tol=1e-7, rel_tol=0):
                raise SensorError("World snapshot and Agent GameTime do not describe the same tick")
            self.set_simulation_tick(SimulationTick(frame, Timestamp(timestamp), snapshot.timestamp.delta_seconds))
        except Exception as exc:
            self.supervisor.enter_module('M1.clock')
            self.supervisor.handle_exception(exc)
            raise
        return Bench2DriveAgent.run_step(self, input_data, timestamp)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord) -> None:
        # Preserve the runner's own downsampled private fields/metric behavior,
        # while M3 receives the original complete expected route.
        AutonomousAgent.set_global_plan(self, global_plan_gps, global_plan_world_coord)
        Bench2DriveAgent.set_global_plan(self, global_plan_gps, global_plan_world_coord)
