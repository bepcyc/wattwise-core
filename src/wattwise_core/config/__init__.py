"""Configuration package: layered, fail-closed settings (CFG-R*, RUN-R4.1)."""

from __future__ import annotations

from wattwise_core.config.settings import (
    ConfigError,
    Environment,
    Settings,
    get_settings,
    load_eval_budget,
    load_settings,
)

__all__ = [
    "ConfigError",
    "Environment",
    "Settings",
    "get_settings",
    "load_eval_budget",
    "load_settings",
]
