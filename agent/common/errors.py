"""Errors at module boundaries; unimplemented paths never return fake data."""


class AgentError(Exception):
    """Base error for the Agent pipeline."""


class ConfigurationError(AgentError):
    """Invalid or incomplete configuration."""


class SensorError(AgentError):
    """Missing, malformed or unsynchronized observation."""


class WarmupRequiredError(SensorError):
    """Valid observations exist, but causal history is not ready yet."""


class NavigationError(AgentError):
    """Unavailable route or navigation condition."""


class FeatureError(AgentError):
    """Invalid model features."""


class PredictionError(AgentError):
    """Model initialization or inference failure."""


class TrajectoryError(AgentError):
    """Invalid trajectory or coordinate/time contract."""


class ControlError(AgentError):
    """Invalid controller state or command."""


class LifecycleError(AgentError):
    """Operation inconsistent with Agent lifecycle."""


class PendingImplementationError(AgentError, NotImplementedError):
    """Verified implementation prerequisites are still missing."""
