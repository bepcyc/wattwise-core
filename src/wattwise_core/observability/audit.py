"""Tamper-evident audit log stream (LOG-R6 / LOG-R6.2 / LOG-R8).

LOG-R6 mandates three correlated but DISTINCT log streams; this module is the second:
the tamper-evident AUDIT stream carrying authentication events, entitlement/plan
decisions, source connect/disconnect, admin/operator actions, and data export/erasure
(PRIV-R8/R9). Every audit event is emitted through the same central allowlist redactor
as the application stream (LOG-R5 — the redactor runs on EVERY stream) onto
stdout (LOG-R1: the platform ships/retains it, with the audit stream's own, longer
retention window — LOG-R9), and is distinguishable by the constant ``stream="audit"``
field so the shipper can route it to its dedicated, longer-retention sink.

Tamper evidence (LOG-R6.2) is a per-process SHA-256 hash chain: each event carries
``audit_seq`` (monotonic), ``prev_hash`` (the previous event's ``entry_hash``), and
``entry_hash`` = SHA-256 over the canonicalized event payload + ``prev_hash``. Removing,
reordering, or editing any emitted event breaks the chain for every later event, so the
stream is verifiable append-only from the records alone.

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

from wattwise_core.observability.logging import get_logger

#: The constant stream discriminator carried on every audit event (LOG-R6.2).
AUDIT_STREAM = "audit"

_logger = get_logger("wattwise_core.audit")

#: The chain genesis: the first event's ``prev_hash`` (a fixed, documented sentinel).
_GENESIS = "0" * 64


class _AuditChain:
    """The process-local hash chain state (one per worker, like the metrics registry)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._prev_hash = _GENESIS

    @staticmethod
    def _entry_hash(payload: dict[str, Any], prev_hash: str) -> str:
        """The SHA-256 chain hash over the canonicalized payload + previous hash."""
        canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
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
            record["entry_hash"] = self._entry_hash(payload, self._prev_hash)
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

    Computes the hash-chain triplet (``audit_seq``/``prev_hash``/``entry_hash``) under a
    process lock so concurrent emitters cannot fork the chain, then emits through the
    central structured logger — which applies the mandatory allowlist redaction
    (LOG-R5/PRIV-R5) before the event leaves the process. Returns the chained record
    (useful to tests and to callers that persist a receipt). Never raises into the
    caller's request path: a logging error must not turn a successful operation into a
    500 (the platform's stream monitor owns emit-failure alerting).
    """
    record = _chain.append({"event": event, **fields})
    with contextlib.suppress(Exception):
        _logger.info(event, **{k: v for k, v in record.items() if k != "event"})
    return record


def record_erasure_hook(athlete_id: str) -> dict[str, Any]:
    """Emit the LOG-R8 per-athlete log-PII erasure hook event.

    A PRIV-R8 erasure calls this so the platform retention layer (which owns the shipped
    log streams, OBS-R8/LOG-R9) purges or anonymizes the erased athlete's log-borne
    identifiers within the stated SLA. The streams carry only the opaque ``athlete_id``
    (PRIV-R5), so this names the identifier to purge.
    """
    return audit_event("log_pii_erasure_hook", athlete_id=athlete_id)


def reset_chain_for_tests() -> None:
    """Reset the chain to genesis (test isolation only; never called in production)."""
    _chain.reset()


__all__ = ["AUDIT_STREAM", "audit_event", "record_erasure_hook", "reset_chain_for_tests"]
