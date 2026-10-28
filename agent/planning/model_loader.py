"""Reference the original V1 namespace; do not copy neural network code."""

from agent.common.errors import PendingImplementationError
from agent.config.settings import PlannerConfig


class V1ModelLoader:
    MODEL_MODULE = "transfuser_model_v2"
    MODEL_CLASS = "V2TransfuserModel"  # Official V1 name, despite the V2 prefix.

    def __init__(self, config: PlannerConfig):
        self.config = config

    def resolve_official_interfaces(self):
        """Explicit future hook: source guard rejects a shadowing V2 namespace.

        Requires repositories/DiffusionDrive first in PYTHONPATH, as in training.
        This imports class definitions only; Agent setup never calls this hook.
        """
        from tools.b2d_adapter.v1_builders import official_v1_module
        config_type = official_v1_module("transfuser_config").TransfuserConfig
        model_type = getattr(official_v1_module(self.MODEL_MODULE), self.MODEL_CLASS)
        return config_type, model_type

    def build_model(self):
        # TODO: use resolve_official_interfaces + validate_v1_config once the
        # selected training config, local backbone and plan-anchor hash are
        # supplied. Construction itself reads anchor/backbone assets and must
        # remain an explicit action; no download or random-weight substitute.
        raise PendingImplementationError("V1 model construction awaits verified local assets/config")

    def load_checkpoint(self, model) -> None:
        # TODO: verify fit.py checkpoint schema, key prefixes and strict loading
        # against the actual baseline artifact, including config provenance.
        raise PendingImplementationError("Checkpoint loading is not implemented in Agent V0.1")
