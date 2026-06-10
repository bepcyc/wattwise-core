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
  on EVERY event before it leaves the process, across all log streams (LOG-R6).
  Redaction is **ALLOWLIST-based, never blocklist-based** (LOG-R5 / PRIV-R5 verbatim):
  ONLY explicitly enumerated, known-safe operational/correlation fields are emitted
  verbatim; EVERY other key — and EVERY non-string value, even under an allowed key —
  is redacted/dropped by default. There is no deny-substring list to bypass: an
  unknown key (``home_address``, ``athlete_name``, ``start_lat``) or a non-string
  value (a raw ``hrv`` float, a health int) is masked because it is NOT on the
  allowlist, not because it matched a known-bad name. As a defence-in-depth second
  layer, the values of allowlisted *string* fields are additionally scrubbed for any
  secret/PII substring (bearer tokens, API keys, emails, credential refs, base64
  envelope material) so a token pasted into ``event``/``message`` is still masked.
  Redaction is unconditional — it is NEVER relaxed by debug/verbose mode (LOG-R4 /
  RUN-R4.2).

The redactor is exported so the audit-log and agent/eval-trace streams (LOG-R6.2 /
LOG-R6.3) reuse the SAME function — there is exactly one redactor in the system.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger

# --- redaction policy --------------------------------------------------------

_MASK = "[REDACTED]"

# (1) Field ALLOWLIST (LOG-R5 / PRIV-R5): the COMPLETE set of event keys that may be
# emitted. This is the redaction policy's single gate — it is allowlist-based, NOT
# blocklist-based. A key NOT in this set is redacted by default; there is no
# deny-substring escape hatch, so an arbitrary unknown PII/health/GPS key
# (``athlete_name``, ``home_address``, ``start_lat``, ``resting_bpm``, …) is masked
# precisely because it is unknown, never because it matched a known-bad name.
#
# Membership is restricted to fields that are operationally necessary AND structurally
# incapable of carrying special-category (health) data, secrets, PII, raw
# prompt/response content, or full request/response bodies (PRIV-R5):
#   - structlog/render-injected envelope: timestamp, level, logger(_name), event;
#   - LOG-R3 correlation ids (all opaque, never PII): trace_id, span_id, request_id,
#     athlete_id (opaque internal id), run_id, thread_id;
#   - bounded operational descriptors emitted by production call sites: status,
#     outcome, duration_ms / latency_ms, path (URL path, no query/body), error_type
#     (an exception class name, never the message), source / source_key /
#     connection_id (opaque source + connection identifiers), schema (schema name),
#     attempt / max_attempts (small ints — but see below: non-strings are still
#     dropped unless the key is numeric-safe), requested_at (an ISO-8601 instant).
# Every other key is redacted. This list is a security invariant (like the SEC-R2.1
# signing-algorithm allowlist), not an operator-tunable, so it lives in code.
_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        # render/structlog envelope
        "timestamp",
        "level",
        "logger",
        "logger_name",
        "event",
        # LOG-R3 correlation ids (opaque, never PII)
        "trace_id",
        "span_id",
        "request_id",
        "athlete_id",
        "run_id",
        "thread_id",
        # bounded operational descriptors (no health/PII/secret/body)
        "status",
        "outcome",
        "path",
        "error_type",
        "source",
        "source_key",
        "connection_id",
        "schema",
        "requested_at",
        # AGT-OBS-R5 / MODEL-R2: a tier-escalation decision is logged explicitly with its
        # node, the chosen tier/effort labels, and the policy-recorded reason. LANG-R4: a
        # language-fallback event is logged with the requested + resolved language tags.
        # All are bounded operational descriptors (enum labels / policy-authored reason /
        # BCP-47-ish tags) — never athlete content or PII.
        "node",
        "tier",
        "reasoning_effort",
        "reason",
        "requested_language",
        "resolved_language",
    }
)

# Allowlisted keys whose value is permitted to be a bounded NON-string scalar
# (small bools/ints/floats that carry no health/PII — e.g. a latency in ms, a retry
# attempt count). For every OTHER allowed key, only ``str`` values pass; any
# non-string value (a raw ``hrv`` float, a health int, a nested object) is dropped,
# so a sensitive value can never ride through under an allowed key as a number.
_ALLOWED_NUMERIC_KEYS: frozenset[str] = frozenset(
    {
        "duration_ms",
        "latency_ms",
        "attempt",
        "max_attempts",
        "status_code",
        # doc 30 ING-OBS-R1 per-run sync trace: per-phase timings + record counts
        # (bounded operational integers/floats; no health values, no PII).
        "authorize_ms",
        "discover_ms",
        "fetch_ms",
        "map_ms",
        "upsert_ms",
        "refs_discovered",
        "refs_skipped",
        "records_fetched",
        "records_failed",
        "candidates_mapped",
        "activities_written",
        "wellness_written",
        "gaps_opened",
        "gaps_closed",
        "watermarks_advanced",
        "retries",
        "rate_limit_wait_ms",
        "untrusted_content",
    }
)
_ALLOWED_KEYS = _ALLOWED_KEYS | _ALLOWED_NUMERIC_KEYS

# (2) Value-pattern scrub (LOG-R5 defence-in-depth): even under an ALLOWLISTED string
# key (e.g. a token pasted into a free-text ``event``/``message``), mask sensitive
# substrings. This is a second layer BEHIND the allowlist, never a substitute for it.
# Ordered most-specific-first.
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
    """Mask sensitive substrings inside an allowlisted string value (LOG-R5 layer 2).

    Defence-in-depth behind the field allowlist: a token/email/credential-ref pasted
    into a permitted free-text field (``event``/``message``) is still masked.
    """
    masked = value
    for pat in _VALUE_PATTERNS:
        masked = pat.sub(_MASK, masked)
    return masked


def _redact_one(key: str, value: Any) -> Any:
    """Resolve one (key, value) pair against the ALLOWLIST (LOG-R5 / PRIV-R5).

    Allowlist-based, never blocklist-based:

    * a key NOT in :data:`_ALLOWED_KEYS` is masked outright — including arbitrary
      unknown PII/health/GPS keys, because the policy emits ONLY known-safe fields;
    * an allowlisted *string* value passes after the value-pattern scrub;
    * an allowlisted *numeric-safe* key (``_ALLOWED_NUMERIC_KEYS``) passes a bounded
      ``bool``/``int``/``float`` scalar verbatim;
    * any OTHER value type under an allowed key (a float under ``hrv``-via-``status``,
      a nested object, a list) is masked — a non-string can never ride through.
    """
    lkey = key.lower()
    if lkey not in _ALLOWED_KEYS:
        return _MASK
    if isinstance(value, str):
        return _mask_value(value)
    if lkey in _ALLOWED_NUMERIC_KEYS and isinstance(value, (bool, int, float)):
        return value
    # Allowed key, but a non-string / non-whitelisted-numeric value: drop it. This
    # catches a raw health number, a nested mapping, or a sequence smuggled under an
    # otherwise-safe key name.
    return _MASK


def redact_processor(
    _logger: WrappedLogger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Central emit-boundary redactor (LOG-R5 / PRIV-R5).

    Runs on EVERY event of EVERY stream before emission. ALLOWLIST-based: emits
    ONLY the explicitly enumerated known-safe fields (:data:`_ALLOWED_KEYS`) and
    redacts/drops EVERYTHING else — every unknown key AND every non-string value —
    by default. Never relaxed by debug mode. The single source of truth for what
    may leave the process — the audit and agent/eval streams (LOG-R6.2/R6.3) reuse
    this exact function so there is only one redactor.
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
