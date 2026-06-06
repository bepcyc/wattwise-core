"""Unit tests for the central log redactor (LOG-R5 / PRIV-R5).

RED test for the redactor: a planted secret/PII/health/prompt value in a log
event MUST be redacted/omitted in the emitted stream — including via key-name
redaction, value-pattern redaction, nested structures, and credential refs'
wrapped material. Asserts redaction holds in BOTH INFO and verbose-debug modes
(LOG-R5-AC) and that the app emits to stdout only (LOG-R1).
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


def _redact(event: dict[str, object]) -> dict[str, object]:
    """Run the central redactor exactly as the emit boundary does."""
    return dict(redact_processor(None, "info", dict(event)))  # type: ignore[arg-type]


# --- direct redactor (processor-level) tests --------------------------------


def test_token_redacted_by_key_name() -> None:
    out = _redact({"access_token": "abc123secrettoken", "athlete_id": "ath_42"})
    assert out["access_token"] == "[REDACTED]"
    assert out["athlete_id"] == "ath_42"  # safe correlation id preserved


def test_email_redacted_by_value_pattern() -> None:
    out = _redact({"message": "user athlete@example.com connected source"})
    assert "athlete@example.com" not in json.dumps(out)


def test_hrv_health_value_redacted() -> None:
    out = _redact({"hrv": 58.3, "heart_rate": 172, "message": "ok"})
    assert out["hrv"] == "[REDACTED]"
    assert out["heart_rate"] == "[REDACTED]"


def test_raw_prompt_redacted() -> None:
    out = _redact({"prompt": "You are a coach. Athlete HRV=58 weight=72kg ...", "event": "llm"})
    assert out["prompt"] == "[REDACTED]"
    assert out["event"] == "llm"


def test_bearer_token_in_free_text_redacted() -> None:
    out = _redact({"message": "Authorization: Bearer eyJabc.def.ghi token used"})
    blob = json.dumps(out)
    assert "Bearer eyJabc" not in blob


def test_credential_ref_redacted_in_value() -> None:
    cipher = EnvelopeCipher(EnvelopeCipher.generate_root_key())
    store = InMemoryCredentialStore(cipher)
    ref = store.store("secret")
    # The opaque ref pasted into free text is masked by the cred_ value pattern.
    out = _redact({"message": f"resolving {ref} for sync"})
    assert ref not in json.dumps(out)


def test_envelope_wire_material_redacted() -> None:
    cipher = EnvelopeCipher(EnvelopeCipher.generate_root_key())
    wire = cipher.encrypt(b"a-source-secret").to_wire()
    out = _redact({"message": f"stored {wire}"})
    assert wire not in json.dumps(out)


def test_nested_secret_redacted() -> None:
    out = _redact(
        {
            "context": {"password": "hunter2", "ok": "value"},
            "items": [{"api_key": "sk-livesecretkeyABCDEFGHIJ012345"}],
        }
    )
    blob = json.dumps(out)
    assert "hunter2" not in blob
    assert "sk-livesecretkeyABCDEFGHIJ012345" not in blob
    assert "value" in blob  # innocuous sibling preserved


def test_dsn_redacted() -> None:
    out = _redact({"database_dsn": "postgresql://u:p@host/db"})
    assert out["database_dsn"] == "[REDACTED]"


# --- end-to-end stream tests (LOG-R1 stdout-only, LOG-R5-AC both modes) ------


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    yield
    # Restore default config so test order does not leak handler/level state.
    configure_logging(logging.INFO)


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_planted_secrets_redacted_in_emitted_stream(level: int) -> None:
    """LOG-R5-AC: token + email + HRV + raw prompt all redacted, every mode."""
    configure_logging(level)
    buf = io.StringIO()
    with redirect_stdout(buf):
        log = get_logger("test")
        log.info(
            "source_connected",
            access_token="live-token-shouldnotappear",
            prompt="raw model prompt with athlete health",
            hrv=58.3,
            message="contact athlete@example.com",
            athlete_id="ath_99",
        )
    emitted = buf.getvalue()
    assert emitted  # something was written to stdout (LOG-R1)
    assert "live-token-shouldnotappear" not in emitted
    assert "raw model prompt with athlete health" not in emitted
    assert "58.3" not in emitted
    assert "athlete@example.com" not in emitted
    # Safe correlation id survives so the line is still reconstructable (LOG-R3).
    assert "ath_99" in emitted
    # It is one JSON object per line (LOG-R2).
    line = emitted.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["athlete_id"] == "ath_99"


def test_emits_one_json_object_per_line() -> None:
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
