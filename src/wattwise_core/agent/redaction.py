"""PII redaction for durable agent state and provider sends (AGT-SEC-R4, CKPT-R8).

AGT-SEC-R4 requires that PII in messages and checkpoints be redacted per policy BEFORE
persistence AND before being sent to any third-party model provider where policy
requires. The checkpointed agent state holds the athlete's own words (``messages``,
``request_text``) and composed prose (``draft``/``grounded_text``) — health-adjacent
special-category content (MEM-R3) — and the model seam sends the system instruction plus
the untrusted-data region to a third-party provider. Neither path was masked.

This module is the agent-side adapter over the SINGLE central redactor
(:func:`wattwise_core.api.redaction.redact_text`), so checkpoint/provider redaction agrees
byte-for-byte with the problem-doc and log-emit redactors on what counts as a secret/PII
span (API-R19 / LOG-R5). It only MASKS the high-confidence PII/secret spans the central
redactor recognizes; it never rewrites semantics, so a redacted checkpoint still
deserializes and a redacted prompt still reads as coach copy.

Unlike :func:`wattwise_core.api.redaction.redact_payload` (which lowers tuples to lists for
a JSON problem body), :func:`redact_state_payload` is TYPE-PRESERVING: a durable checkpoint
blob round-trips its container types (dict/list/tuple/set) unchanged so the serialized state
deserializes identically — only the string LEAVES are masked. Non-text scalars pass through
untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import Any, Final

from langgraph.checkpoint.base import Checkpoint

from wattwise_core.api.redaction import redact_text

# Server-derived IDENTITY / control channels that MUST NOT be masked: they are opaque internal
# identifiers (doc 70 — ``athlete_id`` is "an opaque internal identifier, never PII"), not PII,
# and masking them would corrupt the durable thread/athlete scoping that resume + cross-identity
# refusal depend on (CKPT-R3). The athlete UUID and the ``{athlete}:{conversation}`` thread id
# can incidentally match the redactor's bare-digit-run shape, so they are preserved by key.
IDENTITY_CHANNELS: Final[frozenset[str]] = frozenset(
    {"athlete_id", "thread_id", "idempotency_key", "turn_id", "run_epoch", "trigger"}
)


def redact_state_payload(value: Any) -> Any:
    """Recursively mask PII/secret spans in a checkpoint/state value, preserving types.

    String leaves are masked via the central :func:`redact_text`; mappings, lists, tuples,
    and sets are recursed into and rebuilt with their ORIGINAL container type so a durable
    checkpoint blob deserializes identically (only the sensitive substrings change). Other
    scalars (ints, floats, bools, ``None``, bytes) carry no free-text PII and pass through.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {key: redact_state_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(redact_state_payload(item) for item in value)
    if isinstance(value, Set):
        return {redact_state_payload(item) for item in value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_state_payload(item) for item in value]
    return value


def redact_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    """Return a copy of ``checkpoint`` whose channel values have PII masked (AGT-SEC-R4).

    The PII-bearing durable state lives under ``channel_values`` (the athlete's
    ``messages``/``request_text`` and the composed ``draft``/``grounded_text`` — MEM-R3
    special-category content). This masks those string leaves through the central redactor
    BEFORE the saver serializes the blob, so the persisted bytes carry no unmasked PII
    (AGT-SEC-R4 "redacted ... before persistence", CKPT-R8 §10). The server-derived IDENTITY
    channels (:data:`IDENTITY_CHANNELS`) are preserved verbatim — they are opaque internal
    identifiers, not PII, and masking them would corrupt the durable thread/athlete scoping
    resume depends on (CKPT-R3). The checkpoint's structural fields (``id``/``ts``/
    ``versions_seen``) are preserved so the blob still deserializes and resume is identical
    (CKPT-R2).
    """
    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return checkpoint
    redacted: Checkpoint = dict(checkpoint)  # type: ignore[assignment]
    redacted["channel_values"] = {
        key: value if key in IDENTITY_CHANNELS else redact_state_payload(value)
        for key, value in channel_values.items()
    }
    return redacted


__all__ = ["IDENTITY_CHANNELS", "redact_checkpoint", "redact_state_payload"]
