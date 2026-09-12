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
from wattwise_core.api.routers.sync import SyncTarget
from wattwise_core.api.wiring import build_ingestion_seams
from wattwise_core.config import Settings, load_settings
from wattwise_core.domain.enums import AuthArchetype, ConnectionStatus
from wattwise_core.persistence import Database
from wattwise_core.persistence.models import Athlete, Base, Connection, SourceDescriptor
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


async def _wired_sync_db(tmp_path) -> tuple[Database, Settings, str]:  # type: ignore[no-untyped-def]
    """A real (file-backed) DB with the schema created, its settings, and a seeded athlete id.

    A FILE-backed SQLite DSN (not ``:memory:``) so the rows the test seeds survive the
    per-subject sessions the wired ``EngineSessionProvider`` opens (skill §7).
    """
    settings = load_settings(
        app__environment="development",
        database_dsn=f"sqlite+aiosqlite:///{tmp_path / 'sync.db'}",
        token_signing_key="k" * 32,
    )
    database = Database(settings)
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with database.session() as session:
        athlete = Athlete(reference_timezone="UTC")
        session.add(athlete)
        await session.flush()
        athlete_id = str(athlete.athlete_id)
    return database, settings, athlete_id


async def test_sync_seam_no_connected_source_is_honest_nothing_to_sync(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The wired sync seam tells the truth when the owner has NO connected source (API-R46c).

    Issue #118: with zero connections the run touches nothing, so the ``202`` handle MUST
    carry the distinct ``nothing_to_sync`` status — NOT the falsely-reassuring "we're
    pulling your latest training now." copy that implies a sync is happening.
    """
    database, settings, athlete_id = await _wired_sync_db(tmp_path)
    try:
        seam = build_ingestion_seams(database, settings).sync_orchestrator
        result = await seam(SyncTarget(athlete_id=athlete_id, connection_id=None))
        assert result.status == "nothing_to_sync"
        assert result.status_text != "We're pulling in your latest training now."
        assert "nothing to bring in" in result.status_text.lower()
    finally:
        await database.engine.dispose()


async def test_sync_seam_with_a_connected_source_starts_accepted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The wired sync seam keeps the started-run handle when a SOURCE is connected (API-R46c).

    The honesty boundary is whether a connected source was reached — not whether data came
    back. A real connected source whose on-demand fetch finds nothing / degrades still
    counts as a started run (``accepted``); only the genuine no-connected-source case is
    ``nothing_to_sync``.
    """
    database, settings, athlete_id = await _wired_sync_db(tmp_path)
    try:
        async with database.session() as session:
            descriptor = SourceDescriptor(
                source_key="intervals_icu", display_name="Intervals.icu", kind="oauth_api"
            )
            session.add(descriptor)
            await session.flush()
            session.add(
                Connection(
                    athlete_id=uuid.UUID(athlete_id),
                    source_descriptor_id=descriptor.source_descriptor_id,
                    status=ConnectionStatus.CONNECTED,
                    auth_archetype=AuthArchetype.API_KEY,
                    credential_ref=None,
                )
            )
        seam = build_ingestion_seams(database, settings).sync_orchestrator
        result = await seam(SyncTarget(athlete_id=athlete_id, connection_id=None))
        assert result.status == "accepted"
        assert result.status_text == "We're pulling in your latest training now."
    finally:
        await database.engine.dispose()


async def test_sync_seam_only_excluded_connection_is_nothing_to_sync(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A connection that EXISTS but is in an excluded state is honest no-sync too (API-R46c).

    The honesty key is whether a *connected/syncable* source was reached, not whether a
    connection row merely exists. A sole connection in ``reauth_required`` is excluded by
    the orchestrator's ``_select_connections`` (AUT-R4), so no source is reached and the
    run touches nothing — it MUST read ``nothing_to_sync``, never the reassuring copy.
    """
    database, settings, athlete_id = await _wired_sync_db(tmp_path)
    try:
        async with database.session() as session:
            descriptor = SourceDescriptor(
                source_key="intervals_icu", display_name="Intervals.icu", kind="oauth_api"
            )
            session.add(descriptor)
            await session.flush()
            session.add(
                Connection(
                    athlete_id=uuid.UUID(athlete_id),
                    source_descriptor_id=descriptor.source_descriptor_id,
                    status=ConnectionStatus.REAUTH_REQUIRED,
                    auth_archetype=AuthArchetype.API_KEY,
                    credential_ref=None,
                )
            )
        seam = build_ingestion_seams(database, settings).sync_orchestrator
        result = await seam(SyncTarget(athlete_id=athlete_id, connection_id=None))
        assert result.status == "nothing_to_sync"
        assert result.status_text != "We're pulling in your latest training now."
    finally:
        await database.engine.dispose()
