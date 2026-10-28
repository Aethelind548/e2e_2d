"""Library logging without global basicConfig or file side effects."""

import logging
from typing import Optional

from agent.common.errors import ConfigurationError
from agent.common.types import AgentRuntimeState
from agent.config.settings import RuntimeConfig


def get_logger(config: Optional[RuntimeConfig] = None) -> logging.Logger:
    config = config if config is not None else RuntimeConfig()
    if not isinstance(config.logger_name, str) or not config.logger_name:
        raise ConfigurationError("Logger name must be a nonempty string")
    if not isinstance(config.log_level, str):
        raise ConfigurationError("Log level must be a string")
    level = getattr(logging, config.log_level.upper(), None)
    if not isinstance(level, int):
        raise ConfigurationError(f"Unknown log level: {config.log_level}")
    logger = logging.getLogger(config.logger_name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def runtime_snapshot(state: AgentRuntimeState) -> dict:
    """Small serializable diagnostic record, without raw sensor payloads."""
    return {"status": state.status.value, "completed_steps": state.completed_steps,
            "warmup_steps": state.warmup_steps, "readiness_message": state.readiness_message,
            "timestamp_s": state.timestamp.seconds if state.timestamp else None,
            "timestamp_source": state.timestamp.source if state.timestamp else None,
            "active_module": state.active_module, "modules": dict(state.module_status),
            "last_error_type": state.last_error_type, "last_error_message": state.last_error_message}
