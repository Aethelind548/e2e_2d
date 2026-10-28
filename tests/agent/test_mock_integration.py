"""L2 attempts the REAL main pipeline; only predictor and CARLA object are doubles.

No replacement of SensorManager, RouteManager, Adapter, processor or controller.
Current expected result: real M2/M3 complete after warmup; Blocked at M4.
"""

from dataclasses import asdict, replace
import math

import pytest

from agent.bench2drive_agent import Bench2DriveAgent
from agent.common.errors import PendingImplementationError, PredictionError, SensorError, WarmupRequiredError
from agent.common.types import CoordinateFrame, RuntimeStatus, TrajectoryPoint
from agent.config.settings import PlannerConfig
from agent.planning.mock_predictor import MockTrajectoryPredictor


class ScenarioPredictor(MockTrajectoryPredictor):
    """Only the allowed inference boundary varies deterministic model outputs."""
    def __init__(self, scenario):
        super().__init__(PlannerConfig(kind='mock', num_poses=8, interval_s=.5,
                                       output_frame=CoordinateFrame.MODEL_REAR, mock_speed_mps=2.))
        self.scenario = scenario
        self.calls = []

    def predict(self, features):
        prediction = super().predict(features)
        self.calls.append(features)
        if self.scenario == 'inference_exception':
            raise PredictionError('deterministic injected inference failure')
        if self.scenario == 'invalid_prediction':
            return replace(prediction, points=())
        sign = 1 if self.scenario == 'left' else -1 if self.scenario == 'right' else 0
        points = []
        for point in prediction.points:
            t = point.time_from_start_s
            x = 0. if self.scenario == 'stop' else 2. * (1. - math.exp(-t)) if self.scenario == 'decelerate' else 2. * t
            points.append(TrajectoryPoint(x, sign * .1 * t*t, sign * .1 * t, t))
        return replace(prediction, points=tuple(points))


@pytest.mark.xfail(strict=True, raises=PendingImplementationError,
                   reason='BLOCKED L2: real M2/M3 complete after four warmup ticks; stops at M4 online features')
@pytest.mark.parametrize('scenario,cycles', [('straight', 1), ('left', 1), ('right', 1),
    ('decelerate', 1), ('stop', 1), ('invalid_prediction', 1), ('inference_exception', 1), ('continuous_30', 30)])
def test_real_agent_mock_scenarios(scenario, cycles, config_file, m2m3_config, world_route, sensor_stream, fake_carla):
    agent = Bench2DriveAgent()
    lateral = -5. if scenario == 'left' else 5. if scenario == 'right' else 0.
    agent.set_global_plan(*world_route(((0., 0.), (10., lateral), (20., lateral), (30., lateral), (40., lateral))))
    agent.setup(config_file(m2m3_config))
    predictor = ScenarioPredictor(scenario)
    predictor.initialize()
    agent.predictor = predictor  # Dependency injection limited to inference.
    completed = 0
    try:
        for cycle in range(4):
            observations, tick = sensor_stream(cycle)
            agent.set_simulation_tick(tick)
            with pytest.raises(WarmupRequiredError):
                agent.run_step(observations, tick.timestamp.seconds)
            assert agent.supervisor.state.status == RuntimeStatus.WARMING_UP
        for cycle in range(cycles):
            observations, tick = sensor_stream(4+cycle)
            agent.set_simulation_tick(tick)
            try:
                output = agent.run_step(observations, tick.timestamp.seconds)
            except PendingImplementationError:
                assert agent.supervisor.state.status == RuntimeStatus.FAILED
                assert agent.supervisor.state.active_module == 'M4.features'
                assert agent.supervisor.state.module_status == {
                    'M2.sensors': 'completed', 'M3.navigation': 'completed', 'M4.features': 'failed'}
                assert agent.sensor_manager.buffer.latest.samples['rgb'].payload is observations['rgb'][1]
                assert agent.last_navigation_state.command4 is not None
                assert agent.last_ego_state.timestamp == agent.last_navigation_state.timestamp == tick.timestamp
                assert agent.supervisor.state.completed_steps == completed == 0
                assert predictor.calls == []
                assert fake_carla.created == []
                raise
            if scenario == 'invalid_prediction':
                pytest.fail('Invalid prediction must be rejected before a control is returned')
            if scenario == 'inference_exception':
                pytest.fail('Inference exception must propagate with a failed runtime state')
            assert -1. <= output.steer <= 1.
            assert 0. <= output.throttle <= 1. and 0. <= output.brake <= 1.
            assert not (output.throttle > 0 and output.brake > 0)
            completed += 1
        assert completed == cycles
        assert len(predictor.calls) == cycles
        assert agent.trajectory_processor.last_reference is not None
    finally:
        agent.destroy()
        assert agent.supervisor.state.status == RuntimeStatus.DESTROYED
        assert not predictor.initialized


def test_real_pipeline_missing_input_fails_before_any_control(config_file, sensor_config, fake_carla):
    agent = Bench2DriveAgent()
    agent.setup(config_file({'sensors': asdict(sensor_config)}))
    try:
        with pytest.raises(SensorError, match='Nonempty'):
            agent.run_step({}, 1.)
        assert agent.supervisor.state.status == RuntimeStatus.FAILED
        assert agent.supervisor.state.active_module == 'M2.sensors'
        assert agent.supervisor.state.completed_steps == 0
        assert fake_carla.created == []
    finally:
        agent.destroy()
