"""The EMITTED audit line — not just the in-process record — must verify (LOG-R6.2).

The tamper-evidence guarantee is only real if an independent auditor can re-derive the
hash chain **from the shipped bytes alone** (Crosby & Wallach, USENIX Security 2009):
``verify_chain`` running in-process is exactly where an attacker who can edit the store
also runs. So the load-bearing property is that the JSON the handler actually WROTE —
parsed back — round-trips through :func:`verify_chain`.

These tests pin that property and guard the two redaction traps that the obvious
"just allowlist the four keys" fix leaves open:

* the four structural chain-descriptor keys (``stream``/``audit_seq``/``prev_hash``/
  ``entry_hash``) ship VERBATIM — ``stream`` is an unknown key to the central
  allowlist, ``audit_seq`` is a non-string, and ``prev_hash``/``entry_hash`` are the
  exact "long opaque high-entropy blob" shape the central layer-2 value scrub destroys
  even when allowlisted — so they must flow through the DEDICATED audit logger that
  exempts them (:func:`get_audit_logger`);
* a PII / non-allowlisted payload field is STILL redacted on the emitted line — the
  exemption is scoped to the four structural keys, never to the event payload, so
  ``athlete_id`` stays opaque and a stray PII key is masked (LOG-R5 / PRIV-R5).

This is the gap the prior in-process-only test (``test_doc70_units.py``) could not see:
it verified the returned ``record`` dict, which is intact regardless of redaction.
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import Any

import pytest

from wattwise_core.observability.audit import (
    audit_event,
    reset_chain_for_tests,
    verify_chain,
)
from wattwise_core.observability.logging import configure_logging

pytestmark = pytest.mark.logging


def _emit_and_capture(events: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Emit each ``(event, fields)`` through the real audit path; return the PARSED lines.

    Captures the dedicated audit logger's stdout (it writes via ``PrintLogger`` to
    ``sys.stdout``), so the returned dicts are exactly what an external auditor would
    read off the shipped stream — not the in-process record.
    """
    configure_logging()
    reset_chain_for_tests()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for name, fields in events:
            audit_event(name, **fields)
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def test_emitted_audit_line_round_trips_through_verify_chain() -> None:
    """The audit line parsed back from the SHIPPED JSON verifies (LOG-R6.2 shipped-line).

    Each emitted line carries the four structural chain keys verbatim, so ``verify_chain``
    re-derives the chain from the bytes alone — the in-process record is never consulted.
    """
    emitted = _emit_and_capture(
        [
            ("auth_token_issued", {"athlete_id": "a-1"}),
            ("source_connected", {"athlete_id": "a-1"}),
            ("data_export_started", {"athlete_id": "a-1"}),
        ]
    )
    assert len(emitted) == 3
    # The four structural chain-descriptor keys are present and UNMASKED on every line.
    for line in emitted:
        assert line["stream"] == "audit"
        assert isinstance(line["audit_seq"], int)
        assert line["prev_hash"] != "[REDACTED]"
        assert line["entry_hash"] != "[REDACTED]"
        # 64-char SHA-256 hex survived the high-entropy value scrub.
        assert len(line["entry_hash"]) == 64
    # The security invariant: the SHIPPED bytes verify on their own.
    assert verify_chain(emitted) is True


def test_remasking_a_shipped_chain_key_breaks_verification() -> None:
    """Mutation-proof: re-masking any one chain key on the shipped line fails verify (LOG-R6.2).

    This is the regression the bug caused (the redactor masked the keys) and the exact
    state the fix prevents — proving the round-trip assertion above is non-vacuous.
    """
    emitted = _emit_and_capture(
        [
            ("auth_token_issued", {"athlete_id": "a-1"}),
            ("source_connected", {"athlete_id": "a-1"}),
        ]
    )
    assert verify_chain(emitted) is True
    # Re-mask EACH chain key in turn (the exact damage redact_processor did): each
    # independently makes the shipped chain unverifiable.
    for key in ("stream", "audit_seq", "prev_hash", "entry_hash"):
        broken = [dict(line) for line in emitted]
        broken[1][key] = "[REDACTED]"
        assert verify_chain(broken) is False, f"masking {key!r} should break verification"


def test_pii_in_an_audit_payload_is_still_redacted_on_the_emitted_line() -> None:
    """A non-allowlisted / PII payload field IS masked on the shipped line (LOG-R5 / PRIV-R5).

    The chain-key exemption is scoped to the four structural descriptors ONLY; the event
    payload still flows through the central allowlist, so ``athlete_id`` stays opaque and
    a stray PII key (an email, a home address) is masked — the dedicated audit logger is
    not a redaction bypass.
    """
    emitted = _emit_and_capture(
        [
            (
                "data_export_started",
                {
                    "athlete_id": "a-1",
                    "home_address": "221B Baker Street",
                    "athlete_email": "rider@example.com",
                },
            ),
        ]
    )
    line = emitted[0]
    # Structural chain keys still ship intact.
    assert line["stream"] == "audit"
    assert isinstance(line["audit_seq"], int)
    # Opaque correlation id passes the allowlist verbatim.
    assert line["athlete_id"] == "a-1"
    # The PII fields are masked — neither the value nor any fragment leaks.
    assert line["home_address"] == "[REDACTED]"
    assert line["athlete_email"] == "[REDACTED]"
    serialized = json.dumps(line)
    assert "Baker Street" not in serialized
    assert "rider@example.com" not in serialized
