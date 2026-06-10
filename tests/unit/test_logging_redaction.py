"""Unit tests for the central log redactor (LOG-R5 / PRIV-R5).

The redactor is ALLOWLIST-based, never blocklist-based (LOG-R5 / PRIV-R5 verbatim):
ONLY explicitly enumerated known-safe fields are emitted; EVERY other key and EVERY
non-string value is redacted by default. The cardinal non-vacuous test plants an
ARBITRARY non-denylisted PII/health/GPS bundle (athlete_name, home_address,
resting_bpm, start_lat) and asserts ALL are redacted — none of these would have been
caught by the old deny-substring list, so this only passes under a true allowlist
(mutation-proof: revert to a blocklist and it fails). Other tests assert the safe
operational fields still flow, the value-pattern scrub still masks secrets pasted
into an allowlisted free-text field, and redaction holds in BOTH INFO and
verbose-debug modes (LOG-R5-AC) with stdout-only emission (LOG-R1).
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from contextlib import redirect_stdout

import pytest

from wattwise_core.observability.logging import (
    configure_logging,
    get_logger,
    redact_processor,
)
from wattwise_core.security.credentials import InMemoryCredentialStore
from wattwise_core.security.crypto import EnvelopeCipher

# QA-LOG-R1: the central-redaction contract is part of the gated `logging` tier
# (CI-R1 item 11 / `just test-logging`), not an unmarked stray.
pytestmark = pytest.mark.logging


def _redact(event: dict[str, object]) -> dict[str, object]:
    """Run the central redactor exactly as the emit boundary does."""
    return dict(redact_processor(None, "info", dict(event)))


# --- direct redactor (processor-level) tests --------------------------------


def test_arbitrary_non_denylisted_pii_all_redacted() -> None:
    """LOG-R5/PRIV-R5 allowlist gate (THE non-vacuous test, mutation-proof).

    Plant an ARBITRARY bundle of PII / special-category-health / GPS keys, NONE of
    which appear on any deny-substring list (``athlete_name``, ``home_address``,
    ``resting_bpm``, ``start_lat``). A blocklist redactor would leak ALL of them.
    The allowlist redactor masks ALL of them because none is on the known-safe set;
    revert the fix to a blocklist and this assertion fails (mutation-proof).
    """
    planted = {
        "athlete_name": "Jane Doe",
        "home_address": "12 Privet Drive, Little Whinging",
        "resting_bpm": 41,
        "start_lat": 48.1351,
        "start_lng": 11.5820,
        "email_addr": "jane@example.com",
        "phone": "+49 170 1234567",
        # plus a known-safe correlation id that MUST survive
        "athlete_id": "ath_42",
    }
    out = _redact(planted)
    blob = json.dumps(out)
    for key, value in planted.items():
        if key == "athlete_id":
            continue
        assert out[key] == "[REDACTED]", f"{key} leaked"
        assert str(value) not in blob, f"value of {key} leaked: {value!r}"
    # The single allowlisted correlation id survives verbatim (LOG-R3).
    assert out["athlete_id"] == "ath_42"


def test_unknown_string_key_redacted() -> None:
    """A plain, innocuous-looking unknown string key is redacted (allowlist default)."""
    out = _redact({"notes": "this looks harmless but is not allowlisted"})
    assert out["notes"] == "[REDACTED]"


def test_token_key_redacted_because_unknown() -> None:
    """``access_token`` is masked because it is not on the allowlist, not by name."""
    out = _redact({"access_token": "abc123secrettoken", "athlete_id": "ath_42"})
    assert out["access_token"] == "[REDACTED]"
    assert out["athlete_id"] == "ath_42"  # safe correlation id preserved


def test_email_in_allowlisted_event_redacted_by_value_pattern() -> None:
    """Layer-2 scrub masks an email pasted into the allowlisted ``event`` field."""
    out = _redact({"event": "user athlete@example.com connected source"})
    assert "athlete@example.com" not in json.dumps(out)


def test_hrv_health_value_redacted() -> None:
    """Raw health keys (not allowlisted) are masked regardless of value type."""
    out = _redact({"hrv": 58.3, "heart_rate": 172, "event": "ok"})
    assert out["hrv"] == "[REDACTED]"
    assert out["heart_rate"] == "[REDACTED]"
    assert out["event"] == "ok"  # allowlisted operational field flows


def test_raw_prompt_redacted() -> None:
    """A raw model prompt key is masked; ``event`` (allowlisted) flows."""
    out = _redact({"prompt": "You are a coach. Athlete HRV=58 weight=72kg ...", "event": "llm"})
    assert out["prompt"] == "[REDACTED]"
    assert out["event"] == "llm"


def test_bearer_token_in_free_text_redacted() -> None:
    """Layer-2 scrub masks a bearer token pasted into the allowlisted ``event``."""
    out = _redact({"event": "Authorization: Bearer eyJabc.def.ghi token used"})
    blob = json.dumps(out)
    assert "Bearer eyJabc" not in blob


def test_credential_ref_redacted_in_value() -> None:
    """Layer-2 scrub masks a real credential ref pasted into ``event``."""
    cipher = EnvelopeCipher(EnvelopeCipher.generate_root_key())
    store = InMemoryCredentialStore(cipher)
    ref = store.store("secret")
    out = _redact({"event": f"resolving {ref} for sync"})
    assert ref not in json.dumps(out)


def test_envelope_wire_material_redacted() -> None:
    """Layer-2 scrub masks real envelope wire material pasted into ``event``."""
    cipher = EnvelopeCipher(EnvelopeCipher.generate_root_key())
    wire = cipher.encrypt(b"a-source-secret").to_wire()
    out = _redact({"event": f"stored {wire}"})
    assert wire not in json.dumps(out)


def test_nested_object_dropped_wholesale() -> None:
    """A nested mapping / list under any key is dropped (allowlist + non-string rule).

    Even a sibling that happens to be a safe-looking string inside the nested object
    does NOT survive — the entire non-string value is masked, so a secret can never
    ride through inside a nested structure under an allowed-or-unknown key.
    """
    out = _redact(
        {
            "context": {"password": "hunter2", "ok": "value"},
            "items": [{"api_key": "sk-livesecretkeyABCDEFGHIJ012345"}],
            "status": "ok",  # allowlisted string sibling survives
        }
    )
    blob = json.dumps(out)
    assert "hunter2" not in blob
    assert "sk-livesecretkeyABCDEFGHIJ012345" not in blob
    assert "value" not in blob  # nested innocuous sibling is dropped too
    assert out["context"] == "[REDACTED]"
    assert out["items"] == "[REDACTED]"
    assert out["status"] == "ok"  # allowlisted operational field still flows


def test_non_string_value_under_allowlisted_key_dropped() -> None:
    """A non-string value smuggled under an allowlisted string key is dropped.

    ``status``/``outcome`` are allowlisted as STRING fields. A float/int placed under
    them (e.g. a raw health number) is masked, not emitted — only the numeric-safe
    keys (latency/attempt/status_code) may carry a bounded scalar.
    """
    out = _redact({"status": 58.3, "outcome": 172, "latency_ms": 12})
    assert out["status"] == "[REDACTED]"
    assert out["outcome"] == "[REDACTED]"
    assert out["latency_ms"] == 12  # numeric-safe key carries the bounded scalar


def test_dsn_redacted() -> None:
    """A DSN key is masked because it is not allowlisted."""
    out = _redact({"database_dsn": "postgresql://u:p@host/db"})
    assert out["database_dsn"] == "[REDACTED]"


def test_production_operational_fields_flow() -> None:
    """Every kwarg emitted by a real production call site survives the allowlist.

    Mirrors the exact kwargs of the production log sites (api/errors.py,
    api/app.py, ingestion/sync.py, agent/structured.py) so the redactor never
    blackholes a legitimate operational field.
    """
    out = _redact(
        {
            "event": "sync.source_degraded",
            "level": "warning",
            "trace_id": "tr_1",
            "span_id": "sp_1",
            "request_id": "rq_1",
            "athlete_id": "ath_1",
            "run_id": "run_1",
            "thread_id": "th_1",
            "path": "/agent/ask",
            "error_type": "TimeoutError",
            "source": "intervals_icu",
            "source_key": "intervals_icu",
            "connection_id": "conn_1",
            "schema": "RetrievalPlan",
            "attempt": 2,
            "max_attempts": 3,
            "requested_at": "2026-06-09T00:00:00+00:00",
            "status": "pending_deletion",
            "outcome": "degraded",
            "duration_ms": 42,
        }
    )
    # None of these is redacted: they are the allowlisted operational surface.
    assert "[REDACTED]" not in json.dumps(out)
    assert out["error_type"] == "TimeoutError"
    assert out["attempt"] == 2
    assert out["athlete_id"] == "ath_1"


# --- end-to-end stream tests (LOG-R1 stdout-only, LOG-R5-AC both modes) ------


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    # Restore default config so test order does not leak handler/level state.
    configure_logging(logging.INFO)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_planted_secrets_redacted_in_emitted_stream(level: int) -> None:
    """LOG-R5-AC through the REAL processor chain, every mode (non-vacuous).

    Drives the production path: ``configure_logging`` installs the real structlog
    chain ending in :func:`redact_processor`, and ``get_logger().info(...)`` emits
    to stdout. The planted bundle mixes deny-substring secrets (token/prompt) with
    ARBITRARY non-denylisted PII/health/GPS keys (``athlete_name``, ``home_address``,
    ``hrv``, ``start_lat``) — none of which a blocklist would catch. ALL must be
    absent from the emitted stream; only the allowlisted ``athlete_id`` survives.
    Revert the redactor to a blocklist and the arbitrary-PII assertions fail.
    """
    configure_logging(level)
    buf = io.StringIO()
    with redirect_stdout(buf):
        log = get_logger("test")
        log.info(
            "source_connected",
            access_token="live-token-shouldnotappear",
            prompt="raw model prompt with athlete health",
            hrv=58.3,
            # arbitrary, NON-denylisted PII / GPS that only an allowlist stops:
            athlete_name="Jane Doe",
            home_address="12 Privet Drive",
            start_lat=48.1351,
            email_addr="jane@example.com",
            athlete_id="ath_99",
        )
    emitted = buf.getvalue()
    assert emitted  # something was written to stdout (LOG-R1)
    assert "live-token-shouldnotappear" not in emitted
    assert "raw model prompt with athlete health" not in emitted
    assert "58.3" not in emitted
    # arbitrary PII/GPS: present ONLY if the redactor is a blocklist (mutation-proof)
    assert "Jane Doe" not in emitted
    assert "12 Privet Drive" not in emitted
    assert "48.1351" not in emitted
    assert "jane@example.com" not in emitted
    # Safe correlation id survives so the line is still reconstructable (LOG-R3).
    assert "ath_99" in emitted
    # It is one JSON object per line (LOG-R2).
    line = emitted.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["athlete_id"] == "ath_99"
    # The unknown keys remain present but with the value masked (never the PII).
    assert parsed["athlete_name"] == "[REDACTED]"
    assert parsed["home_address"] == "[REDACTED]"


def test_emits_one_json_object_per_line() -> None:
    """LOG-R2: each emitted line is one JSON object carrying timestamp + level."""
    configure_logging(logging.INFO)
    buf = io.StringIO()
    with redirect_stdout(buf):
        log = get_logger("t")
        log.info("e1", athlete_id="a")
        log.info("e2", athlete_id="a")
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 2
    for ln in lines:
        obj = json.loads(ln)
        assert "timestamp" in obj
        assert "level" in obj
