"""Structured logging to stdout with mandatory central redaction (LOG-R*, PRIV-R5).

Contract realized here:

- **LOG-R1** Emit a structured event stream to **stdout ONLY**. This module opens,
  writes, rotates, and retains NO log files; the container/platform captures the
  stream. :func:`configure_logging` installs a single
  :class:`logging.StreamHandler` to ``sys.stdout`` and nothing else.
- **LOG-R2** One JSON object per line, carrying at least ``timestamp`` (UTC,
  ISO-8601), ``level``, ``logger``, ``message`` plus any bound correlation context
  (LOG-R3: ``trace_id``/``span_id``/``request_id``/``athlete_id``/``run_id``/
  ``thread_id``).
- **LOG-R5 / PRIV-R5** A SINGLE central redactor (:func:`redact_processor`) runs
  on EVERY event before it leaves the process, across all log streams (LOG-R6). It
  combines three defenses: (1) **key-name** redaction — any event key whose name
  matches the sensitive denylist is dropped/masked; (2) **value-pattern**
  redaction — substrings inside string values that look like secrets/PII (bearer
  tokens, API keys, emails, credential refs, base64 envelope material) are masked;
  (3) **prompt/health containment** — known prompt/health-bearing keys are dropped.
  Redaction is allowlist-friendly: known-safe correlation/operational keys pass;
  everything unrecognized that *looks* sensitive is masked. Redaction is
  unconditional — it is NEVER relaxed by debug/verbose mode (LOG-R4 / RUN-R4.2).

The redactor is exported so the audit-log and agent/eval-trace streams (LOG-R6.2 /
LOG-R6.3) reuse the SAME function — there is exactly one redactor in the system.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger

# --- redaction policy --------------------------------------------------------

_MASK = "[REDACTED]"

# (1) Key-name denylist (LOG-R5): any event key whose lowercased name contains one
# of these tokens is masked outright. Covers secrets, credentials, tokens, raw
# prompt/response/health payloads, and envelope ciphertext fields. This is a
# denylist of DANGEROUS key *names*; the value-pattern pass (below) is the
# allowlist-style net that catches sensitive values under innocuous-looking keys.
_DENY_KEY_SUBSTRINGS: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",  # access_token, refresh_token, signing_token, ...
        "api_key",
        "apikey",
        "authorization",
        "auth_header",
        "credential",  # credential, credentials, credential_ref, raw_credential
        "cred_ref",
        "wrapped_key",
        "wrapped_token",
        "data_key",
        "root_key",
        "encryption_key",
        "private_key",
        "session_key",
        "cookie",
        "set-cookie",
        "dsn",  # database DSN may embed a password
        "prompt",  # raw model prompt content (PRIV-R5 / LOG-R5)
        "completion",  # raw model response content
        "response_text",
        "messages",  # chat message arrays carry prompt/health content
        "hrv",  # special-category health (GDPR Art. 9)
        "heart_rate",
        "heartrate",
        "weight",
        "health",
        "payload",  # raw source payload may carry health/PII
        "raw_body",
        "request_body",
        "response_body",
    }
)

# Keys that are explicitly SAFE to emit verbatim even though a substring above
# might otherwise match (e.g. "logger"). Keeps the correlation context intact.
_SAFE_KEYS: frozenset[str] = frozenset(
    {
        "timestamp",
        "level",
        "logger",
        "logger_name",
        "event",
        "message",
        "trace_id",
        "span_id",
        "request_id",
        "athlete_id",  # opaque internal id, never PII (LOG-R3 / OBS-R2)
        "run_id",
        "thread_id",
        "source",  # opaque source name (e.g. "intervals_icu"), not a secret
        "status",
        "duration_ms",
        "outcome",
    }
)

# (2) Value-pattern denylist (LOG-R5): masks sensitive substrings inside string
# values regardless of the key they appear under (e.g. a token pasted into a free
# "message"). Ordered most-specific-first.
_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Bearer / authorization header values.
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    # Opaque credential refs minted by the credential store (cred_<token>).
    re.compile(r"\bcred_[A-Za-z0-9_\-]{16,}"),
    # JWT-shaped tokens: three base64url segments separated by dots.
    re.compile(r"\b[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
    # Email addresses (PII, LOG-R3/PRIV-R5).
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # Common provider API-key prefixes (OpenAI/OpenRouter/etc.).
    re.compile(r"\b(?:sk|rk|pk|or)[-_][A-Za-z0-9]{16,}\b"),
    # Long opaque high-entropy blobs (envelope wire material, raw tokens):
    # a base64/base64url run of 40+ chars.
    re.compile(r"\b[A-Za-z0-9+/_\-]{40,}={0,2}\b"),
)


def _mask_value(value: str) -> str:
    """Mask sensitive substrings inside a single string value (LOG-R5 pass 2)."""
    masked = value
    for pat in _VALUE_PATTERNS:
        masked = pat.sub(_MASK, masked)
    return masked


def _redact_obj(value: Any) -> Any:
    """Recursively redact a log value: mask strings, recurse into maps/sequences."""
    if isinstance(value, str):
        return _mask_value(value)
    if isinstance(value, MutableMapping):
        return {k: _redact_one(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_obj(v) for v in value)
    return value


def _redact_one(key: str, value: Any) -> Any:
    """Redact a single (key, value) pair by key-name then value-pattern."""
    lkey = key.lower()
    if lkey not in _SAFE_KEYS and any(token in lkey for token in _DENY_KEY_SUBSTRINGS):
        return _MASK
    return _redact_obj(value)


def redact_processor(
    _logger: WrappedLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Central emit-boundary redactor (LOG-R5 / PRIV-R5).

    Runs on EVERY event of EVERY stream before emission: key-name redaction +
    value-pattern redaction + recursive scrub of nested structures. Never
    relaxed by debug mode. The single source of truth for what may leave the
    process — the audit and agent/eval streams (LOG-R6.2/R6.3) reuse this exact
    function so there is only one redactor.
    """
    return {key: _redact_one(key, value) for key, value in event_dict.items()}


# --- configuration -----------------------------------------------------------

_DEFAULT_LEVEL = logging.INFO
# Single-element mutable holder so the idempotency flag can be set without a
# module-level ``global`` statement (PLW0603).
_state: dict[str, bool] = {"configured": False}


def configure_logging(level: int | str = _DEFAULT_LEVEL) -> None:
    """Configure structlog: JSON events to stdout ONLY, with central redaction.

    Idempotent. Installs a single stdout :class:`logging.StreamHandler` (LOG-R1 —
    no files, no rotation) and the processor chain ending in
    :func:`redact_processor` then a JSON renderer (LOG-R2). The redactor sits
    LAST among the mutating processors so nothing added later can slip an
    unredacted value into the emitted line.
    """
    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.upper(), _DEFAULT_LEVEL)

    # LOG-R1: exactly one stdout handler; no FileHandler / RotatingFileHandler.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stdout)
    root.addHandler(handler)
    root.setLevel(level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # LOG-R3 correlation context
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),  # LOG-R2 UTC ISO-8601
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            redact_processor,  # LOG-R5 — last mutating step before render
            structlog.processors.JSONRenderer(),  # LOG-R2 — one JSON object per line
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    _state["configured"] = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structured logger (LOG-R2/R3).

    Configures logging on first use (idempotent) so callers never get an
    un-redacted logger by forgetting to call :func:`configure_logging`.
    """
    if not _state["configured"]:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
