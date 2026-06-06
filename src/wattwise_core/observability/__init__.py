"""Observability layer: structured logging with mandatory central redaction.

Exposes the logging entry points (LOG-R*, PRIV-R5 / OBS-R2): structured JSON
events to stdout only, redacted at the emit boundary by the single central
redactor.
"""

from __future__ import annotations

from wattwise_core.observability.logging import (
    configure_logging,
    get_logger,
    redact_processor,
)

__all__ = ["configure_logging", "get_logger", "redact_processor"]
