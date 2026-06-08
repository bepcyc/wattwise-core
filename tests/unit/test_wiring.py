"""Unit tests for the ingestion/credential seam composition root (``api.wiring``).

The app factory binds the connect/import/sync router seams to these providers; the
load-bearing contracts are: the import processor and sync orchestrator are ALWAYS wired
(so the file-upload + sync-trigger journey works on the built stack), while the credential
sink is present ONLY when an envelope key is configured — without one the crypto layer
refuses to seal a secret, so the connect-``complete`` sink stays fail-closed rather than
storing a credential in the clear (AUTH-R16 / BOOT-R4).
"""

from __future__ import annotations

from wattwise_core.api.routers.connections import CredentialSink
from wattwise_core.api.wiring import build_ingestion_seams
from wattwise_core.config import Settings, load_settings
from wattwise_core.persistence import Database
from wattwise_core.security.crypto import EnvelopeCipher


def _settings(*, with_key: bool) -> Settings:
    extra = {"encryption_root_key": EnvelopeCipher.generate_root_key()} if with_key else {}
    return load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="k" * 32,
        **extra,
    )


def test_import_and_sync_seams_are_always_wired() -> None:
    """The import processor + sync orchestrator are wired regardless of the envelope key."""
    settings = _settings(with_key=False)
    seams = build_ingestion_seams(Database(settings), settings)
    assert callable(seams.import_processor)
    assert callable(seams.sync_orchestrator)


def test_credential_sink_present_only_with_an_envelope_key() -> None:
    """No envelope key → no credential sink (fail-closed); a key → a real sink (AUTH-R16)."""
    keyless = _settings(with_key=False)
    assert build_ingestion_seams(Database(keyless), keyless).credential_sink is None

    keyed = _settings(with_key=True)
    sink = build_ingestion_seams(Database(keyed), keyed).credential_sink
    assert isinstance(sink, CredentialSink)
    # The sink stores a secret behind an opaque ref and never returns the raw value.
    ref = sink.store("super-secret-key")
    assert ref and ref != "super-secret-key"
