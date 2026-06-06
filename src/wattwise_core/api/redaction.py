"""PII redaction for athlete-facing problem documents and trace payloads (API-R19).

Problem documents (RFC 9457 ``detail`` / ``errors[].message``), and any structured
payload correlated by ``trace_id``, **SHALL** have PII redacted before they leave the
process (API-R19), and a problem ``detail`` **SHALL NOT** leak secrets, tokens, or
internal identifiers (ERR-R5). This module is the single, reusable redactor the API
layer applies to the free-text it surfaces in a problem body and to any structured
detail it records on a trace.

It masks the high-confidence PII / secret classes — email addresses, bearer tokens
and common API-key shapes, long opaque high-entropy blobs (envelope/token wire
material), and bare phone-number runs — replacing each match with a fixed
:data:`MASK` token. It is deliberately conservative (it masks, it never rewrites
semantics) so a redacted message stays human-readable coach copy (API-R21) while
carrying none of the sensitive substring. Structured payloads are scrubbed key-first
(a key whose name signals a secret is masked wholesale) then value-pattern-masked,
recursively, mirroring the log-emit redactor (LOG-R5) so problem-doc redaction and
log redaction agree on what may leave the process.

Requirement IDs: API-R19 (PII redaction for problem docs/logs/traces), ERR-R5 (no
secret/token/internal leakage in ``detail``/``errors[].message``), API-R21 (the
redacted human copy stays warm and jargon-free).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

#: The fixed replacement token a redacted span/value collapses to (API-R19).
MASK: Final = "[redacted]"

#: Substrings in a STRUCTURED key name that mark its value as a secret to mask
#: wholesale (API-R19 / ERR-R5), regardless of the value's own shape. Lower-cased
#: comparison; a key containing any of these is masked before value scanning.
_SECRET_KEY_SUBSTRINGS: Final[tuple[str, ...]] = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "cookie",
    "private_key",
    "access_key",
    "session",
)

#: Value-level patterns masked inside ANY free-text string (API-R19 / ERR-R5). Order
#: matters: longer/structured shapes (emails, key-prefixes) run before the generic
#: high-entropy blob so a structured secret is masked as one unit.
_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    # Email addresses.
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # Common provider API-key / token prefixes (OpenAI/OpenRouter/Stripe/etc.).
    re.compile(r"\b(?:sk|rk|pk|or|tok|key)[-_][A-Za-z0-9]{12,}\b"),
    # Bearer credential carried inline in text.
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
    # JWT-shaped triple (header.payload.signature).
    re.compile(r"\b[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
    # Long opaque high-entropy blobs (envelope wire material / raw tokens):
    # a base64/base64url run of 40+ chars.
    re.compile(r"\b[A-Za-z0-9+/_\-]{40,}={0,2}\b"),
    # Bare phone-number runs (>= 9 digits, optional leading +, separators).
    re.compile(r"(?<!\w)\+?\d[\d\s().\-]{8,}\d(?!\w)"),
)


def redact_text(text: str) -> str:
    """Return ``text`` with every PII / secret span masked (API-R19 / ERR-R5).

    Masks emails, API-key/token shapes, inline bearer credentials, JWTs, long
    high-entropy blobs, and bare phone runs. Conservative: it only substitutes the
    matched spans, so the surrounding human message stays readable coach copy
    (API-R21). Idempotent — re-redacting a redacted string changes nothing.
    """
    masked = text
    for pattern in _VALUE_PATTERNS:
        masked = pattern.sub(MASK, masked)
    return masked


def _is_secret_key(key: str) -> bool:
    """True iff a structured key name signals a secret value (mask wholesale)."""
    lowered = key.lower()
    return any(token in lowered for token in _SECRET_KEY_SUBSTRINGS)


def redact_payload(value: Any) -> Any:
    """Recursively redact a structured trace/problem payload (API-R19).

    Masks a value wholesale when its KEY name signals a secret; otherwise masks PII
    spans inside any string and recurses into mappings/sequences. Non-text scalars
    (ints, floats, bools, ``None``) pass through unchanged — they carry no free-text
    PII. Mirrors the log-emit redactor (LOG-R5) so the two never disagree.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            key: MASK if _is_secret_key(str(key)) else redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    return value


def contains_pii(text: str) -> bool:
    """True iff ``text`` still carries an unmasked PII / secret span (test helper).

    Used by the redaction contract test (API-R19) to assert a surfaced problem
    ``detail`` is clean. It is NOT the redactor; it only detects residual matches.
    """
    return any(pattern.search(text) for pattern in _VALUE_PATTERNS)


__all__ = [
    "MASK",
    "contains_pii",
    "redact_payload",
    "redact_text",
]
