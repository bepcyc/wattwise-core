"""Tamper-evident audit log stream (LOG-R6 / LOG-R6.2 / LOG-R8).

LOG-R6 mandates three correlated but DISTINCT log streams; this module is the second:
the tamper-evident AUDIT stream carrying authentication events, entitlement/plan
decisions, source connect/disconnect, admin/operator actions, and data export/erasure
(PRIV-R8/R9). Every audit event's PAYLOAD passes the same central allowlist redaction
policy as the application stream (LOG-R5 — redaction runs on EVERY stream), applied
BEFORE hashing and again (idempotently) by the dedicated audit pipeline at emission,
onto stdout (LOG-R1: the platform ships/retains it, with the audit stream's own, longer
retention window — LOG-R9), and is distinguishable by the constant ``stream="audit"``
field — shipped VERBATIM by the dedicated pipeline — so the shipper can route it to its
dedicated, longer-retention sink.

Tamper evidence (LOG-R6.2) is a per-process SHA-256 hash chain: each event carries
``audit_seq`` (monotonic), ``prev_hash`` (the previous event's ``entry_hash``), and
``entry_hash`` = SHA-256 over the canonicalized event payload **including the
``stream`` discriminator and ``audit_seq``** + ``prev_hash``. The hashed payload is the
FINAL, redacted, correlation-context-bearing form — exactly what ships — so the chain
is verifiable from the SHIPPED lines alone, not just from in-process records. Removing,
reordering, renumbering (``audit_seq`` tamper), stream-swapping, or editing any emitted
event breaks verification, so the stream is verifiable append-only from the records
alone (:func:`verify_chain`).

LOG-R8's per-athlete erasure hook is :func:`record_erasure_hook`: a PRIV-R8 erasure
emits an audit event naming the erased athlete so the platform's log retention layer
purges/anonymizes that athlete's log-borne identifiers within the stated SLA (the
streams themselves carry only the opaque ``athlete_id``, never direct PII — PRIV-R5).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from typing import Any

import structlog

from wattwise_core.observability.logging import audit_redact_processor, get_audit_logger

#: The constant stream discriminator carried on every audit event (LOG-R6.2).
AUDIT_STREAM = "audit"

#: The DEDICATED audit-stream logger (LOG-R6.2): its pipeline applies the central PII
#: allowlist to the event PAYLOAD but ships the four structural chain-descriptor keys
#: (stream/audit_seq/prev_hash/entry_hash) VERBATIM, so the EMITTED line — not just the
#: in-process record — round-trips through :func:`verify_chain`. Emitting through the
#: shared ``get_logger`` instead would mask those four keys (allowlist drop + high-entropy
#: value scrub), leaving the shipped chain unverifiable.
_logger = get_audit_logger()

#: The chain genesis: the first event's ``prev_hash`` (a fixed, documented sentinel).
_GENESIS = "0" * 64


class _AuditChain:
    """The process-local hash chain state (one per worker, like the metrics registry)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._prev_hash = _GENESIS

    @staticmethod
    def _entry_hash(payload: dict[str, Any], prev_hash: str, seq: int, stream: str) -> str:
        """SHA-256 over the canonicalized payload + ``stream`` + ``audit_seq`` + prev hash.

        ``audit_seq`` and the ``stream`` discriminator are INSIDE the hash input, so
        renumbering a stored record or swapping it onto another stream breaks
        verification — not only deleting/reordering/editing the payload (LOG-R6.2).
        """
        canonical = json.dumps(
            {**payload, "stream": stream, "audit_seq": seq},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Append one event under the lock; return the chained record."""
        with self._lock:
            self._seq += 1
            record = {
                **payload,
                "stream": AUDIT_STREAM,
                "audit_seq": self._seq,
                "prev_hash": self._prev_hash,
            }
            record["entry_hash"] = self._entry_hash(
                payload, self._prev_hash, self._seq, AUDIT_STREAM
            )
            self._prev_hash = record["entry_hash"]
        return record

    def reset(self) -> None:
        """Reset to genesis (test isolation only)."""
        with self._lock:
            self._seq = 0
            self._prev_hash = _GENESIS


_chain = _AuditChain()


def audit_event(event: str, **fields: Any) -> dict[str, Any]:
    """Append ONE event to the tamper-evident audit stream (LOG-R6.2).

    The hash must commit to the SHIPPED bytes (LOG-R6.2 shipped-line verifiability),
    so the payload is finalized BEFORE chaining: the bound LOG-R3 correlation context
    (``request_id``/``trace_id``/…) is merged in, the reserved envelope keys are
    dropped, and the central allowlist redaction (LOG-R5/PRIV-R5) is applied via
    :func:`audit_redact_processor`. Only THEN is the hash-chain triplet
    (``audit_seq``/``prev_hash``/``entry_hash``) computed under a process lock (so
    concurrent emitters cannot fork the chain) and the record emitted through the
    DEDICATED audit-stream logger, whose pipeline re-runs the same idempotent
    redaction as defence in depth and ships the chain-descriptor keys verbatim.
    Hash-THEN-redact is the trap this ordering avoids: a payload field masked only at
    emission (e.g. a non-allowlisted ``rows_deleted`` int), or a correlation key
    merged only at emission, makes the emitted line differ from the hashed payload —
    so the shipped line could never verify. Returns the chained record in its
    redacted, shipped form (the receipt equals the emitted line minus the render
    envelope). Never raises into the caller's request path: a logging error must not
    turn a successful operation into a 500 (the platform's stream monitor owns
    emit-failure alerting).
    """
    context = structlog.contextvars.get_contextvars()
    raw = {**context, **fields, "event": event}
    payload = dict(
        audit_redact_processor(
            None, "info", {k: v for k, v in raw.items() if k not in _ENVELOPE_KEYS}
        )
    )
    record = _chain.append(payload)
    with contextlib.suppress(Exception):
        _logger.info(str(record["event"]), **{k: v for k, v in record.items() if k != "event"})
    return record


def record_erasure_hook(athlete_id: str) -> dict[str, Any]:
    """Emit the LOG-R8 per-athlete log-PII erasure hook event.

    A PRIV-R8 erasure calls this so the platform retention layer (which owns the shipped
    log streams, OBS-R8/LOG-R9) purges or anonymizes the erased athlete's log-borne
    identifiers within the stated SLA. The streams carry only the opaque ``athlete_id``
    (PRIV-R5), so this names the identifier to purge.
    """
    return audit_event("log_pii_erasure_hook", athlete_id=athlete_id)


#: The chain-envelope keys a record carries OUTSIDE its event payload: the three
#: hash-linkage descriptors (used directly for verification) plus the ``stream``
#: discriminator (a hash input via the canonical form, never a payload field).
_CHAIN_ENVELOPE_KEYS = frozenset({"stream", "audit_seq", "prev_hash", "entry_hash"})

#: The structlog RENDER-envelope keys the dedicated audit logger adds to the EMITTED
#: line (LOG-R2): a UTC ``timestamp``, the log ``level``, and (if ever bound) the
#: ``logger`` name. They are renderer additions, NOT part of the hashed audit payload,
#: so :func:`verify_chain` ignores them — this is what lets it run on a line parsed
#: straight back from the shipped JSON (the LOG-R6.2 shipped-line verifiability
#: invariant) without the caller having to pre-strip the renderer's own fields.
_RENDER_ENVELOPE_KEYS = frozenset({"timestamp", "level", "logger", "logger_name"})

#: Everything excluded from the recomputed payload: chain descriptors + render envelope.
_ENVELOPE_KEYS = _CHAIN_ENVELOPE_KEYS | _RENDER_ENVELOPE_KEYS


def verify_chain(records: list[dict[str, Any]]) -> bool:
    """Verify a contiguous run of audit records from the records alone (LOG-R6.2).

    For every record, recomputes ``entry_hash`` from its own payload fields + its
    stored ``stream`` + ``audit_seq`` + ``prev_hash``, and checks that ``prev_hash``
    links to the prior record's ``entry_hash`` and that ``audit_seq`` is contiguous.
    Because ``audit_seq`` and ``stream`` are hash inputs, a renumbered or
    stream-swapped record fails even when its payload is untouched. An empty run
    verifies trivially; the first record anchors the run (its ``prev_hash`` is the
    genesis sentinel for a full-process run, or the preceding segment's head).

    Verifies equally over an in-process ``record`` and over a line parsed straight
    back from the SHIPPED JSON: the dedicated audit logger ships the four chain keys
    verbatim, and the structlog render-envelope additions (``timestamp``/``level``)
    are ignored here (:data:`_RENDER_ENVELOPE_KEYS`) since they are not hashed.
    """
    prev_hash: str | None = None
    prev_seq: int | None = None
    for record in records:
        payload = {k: v for k, v in record.items() if k not in _ENVELOPE_KEYS}
        seq = record.get("audit_seq")
        if not isinstance(seq, int):
            return False
        if prev_hash is not None and record.get("prev_hash") != prev_hash:
            return False
        if prev_seq is not None and seq != prev_seq + 1:
            return False
        expected = _AuditChain._entry_hash(
            payload, str(record.get("prev_hash")), seq, str(record.get("stream"))
        )
        if record.get("entry_hash") != expected:
            return False
        prev_hash = expected
        prev_seq = seq
    return True


def reset_chain_for_tests() -> None:
    """Reset the chain to genesis (test isolation only; never called in production)."""
    _chain.reset()


__all__ = [
    "AUDIT_STREAM",
    "audit_event",
    "record_erasure_hook",
    "reset_chain_for_tests",
    "verify_chain",
]
