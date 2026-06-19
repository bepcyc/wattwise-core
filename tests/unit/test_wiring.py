"""Unit tests for the ingestion/credential seam composition root (``api.wiring``).

The app factory binds the connect/import/sync router seams to these providers; the
load-bearing contracts are: the import processor and sync orchestrator are ALWAYS wired
(so the file-upload + sync-trigger journey works on the built stack), while the credential
sink is present ONLY when an envelope key is configured — without one the crypto layer
refuses to seal a secret, so the connect-``complete`` sink stays fail-closed rather than
storing a credential in the clear (AUTH-R16 / BOOT-R4).
"""

from __future__ import annotations

import uuid

import pytest

from wattwise_core.api.errors import ProblemError
from wattwise_core.api.routers.connections import CredentialSink
from wattwise_core.api.wiring import build_ingestion_seams
from wattwise_core.config import Settings, load_settings
from wattwise_core.persistence import Database
from wattwise_core.persistence.models import Base
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


async def test_import_processor_reraises_operator_fault_not_failed_job(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A missing seed descriptor (operator fault) re-raises as 5xx — NOT a per-upload ``failed``.

    API-R33a scopes the terminal ``failed`` job to a POST-acceptance INGEST failure. The seeded
    ``file_import`` descriptor being absent is a pre-ingest operator/config fault: the real
    processor must let that :class:`ProblemError` (``internal-error``) propagate to a generic 5xx
    rather than swallow it into a ``failed`` job the athlete can never clear by retrying — masking
    a deploy bug behind dishonest "try again" copy. A file DSN gives a schema-only (descriptor
    table EMPTY) DB so the descriptor lookup fails closed BEFORE any decode.
    """
    settings = load_settings(
        app__environment="development",
        database_dsn=f"sqlite+aiosqlite:///{tmp_path / 'wiring.db'}",
        token_signing_key="k" * 32,
    )
    database = Database(settings)
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # schema only — no descriptor seeded
    processor = build_ingestion_seams(database, settings).import_processor
    with pytest.raises(ProblemError) as excinfo:
        await processor(str(uuid.uuid4()), b"any-bytes", "ride.fit")
    assert excinfo.value.problem_type.slug == "internal-error"
    await database.engine.dispose()
