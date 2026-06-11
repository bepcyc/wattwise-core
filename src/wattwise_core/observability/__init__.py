"""Observability layer: structured logging with mandatory central redaction.

Exposes the logging entry points (LOG-R*, PRIV-R5 / OBS-R2): structured JSON
events to stdout only, redacted at the emit boundary by the single central
redactor.
"""

from __future__ import annotations

from wattwise_core.observability.logging import (
    audit_redact_processor,
    configure_logging,
    get_audit_logger,
    get_logger,
    redact_processor,
)

__all__ = [
    "audit_redact_processor",
    "configure_logging",
    "get_audit_logger",
    "get_logger",
    "redact_processor",
]
