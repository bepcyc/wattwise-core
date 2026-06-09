"""Orchestrator-boundary fault-injection suite (TST-R5; CLI-R5, AUT-R4, ONB-R5).

The TST-R5 fault categories that live at the SYNC-ORCHESTRATOR / multi-source boundary
(as opposed to the single-client boundary covered offline in
``tests/contract/test_outbound_resilience.py``):

  * CLI-R5 — a cancellation mid-run propagates cleanly: an already-committed source's
    batch SURVIVES (ING-UPS-R3 batch granularity), the cancelled source leaves NO
    half-applied upsert, and no Connection's auth/status row is corrupted.
  * AUT-R4 + multi-source isolation — in a two-source run where one source's credential
    is revoked (401/403 -> typed ``AuthError``), that source flips to ``reauth_required``
    and stops, while the OTHER (healthy) source still completes and lands its data. This
    is the TST-R5 "other sources keep working" assertion for the auth-break category.
  * ONB-R5 / source-removed isolation — a connection whose source is no longer installed
    in the registry DEGRADES in isolation (UnknownSourceError) while the other source
    still completes; prior canonical data is never deleted (ING-R4). (The full SCH-R6
    circuit-breaker — pause-scheduling + auto-probe — is COMMERCIAL orchestration, §8.4 /
    COMM-R19, not shipped in OSS; only the OSS source-removed isolation path is exercised
    here.)

In every case NO value is fabricated and the failing source never crashes the others.

Real-pool data safety: a throwaway FILE-sqlite engine on a real ``QueuePool`` (WAL +
``busy_timeout``) — never ``:memory:`` / ``StaticPool``, never a host/live DB — so a
committed-vs-cancelled write and a persisted status are read back through a real second
connection.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar

import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import AuthArchetype, ConnectionStatus, Fidelity, SourceKind
from wattwise_core.ingestion.base import (
    AuthError,
    FetchContext,
    FetchErrorKind,
    SourceDescriptorRef,
)
from wattwise_core.ingestion.registry import registry_from_adapters
from wattwise_core.ingestion.sync import SyncOrchestrator, SyncOutcome, SyncWindow
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    Base,
    Connection,
    SourceDescriptor,
    Sport,
)
from wattwise_core.security.credentials import InMemoryCredentialStore
from wattwise_core.security.crypto import EnvelopeCipher
from wattwise_core.storage import content_hash

UTC = _dt.UTC
_FIXED_NOW = _dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- fakes


class _Ride:
    """A trivial source-shaped object the healthy adapter's ``map`` consumes."""

    def __init__(self, native_id: str) -> None:
        self.native_id = native_id


class HealthyAdapter:
    """An api-key adapter that fetches one ride and pure-maps it to a canonical activity."""

    source_key: ClassVar[str] = "healthy_api"
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.API_KEY
    kind: ClassVar[SourceKind] = SourceKind.OAUTH_API
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"

    def __init__(self) -> None:
        self.fetched = False

    async def fetch(
        self, *, api_key: str | None, athlete_native_id: str | None, window: SyncWindow
    ) -> Iterable[Any]:
        self.fetched = True
        return [_Ride("ride-1")]

    def map(
        self, asbo: Any, source_descriptor: SourceDescriptorRef, fetch_context: FetchContext
    ) -> list[GboCandidate]:
        if not isinstance(asbo, _Ride):
            return []
        start = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
        payload: dict[str, Any] = {
            "start_time": start,
            "sport": "cycling",
            "elapsed_time_s": 3600,
            "moving_time_s": 3600,
            "avg_power_w": 250.0,
            "streams": {
                "power_w": {
                    "values": [250.0] * 3600,
                    "sample_basis": "time",
                    "sample_rate_hz": 1.0,
                }
            },
        }
        return [
            GboCandidate(
                gbo_type="activity",
                source_descriptor_id=source_descriptor.source_descriptor_id,
                source_native_id=asbo.native_id,
                content_hash=content_hash(asbo.native_id.encode()),
                payload=payload,
                observed_at=start,
                fetched_at=fetch_context.fetched_at,
                trust_tier=Fidelity.RAW_STREAM,
                connection_id=fetch_context.connection_id,
                adapter_version=self.adapter_version,
                mapping_version=self.mapping_version,
            )
        ]


class RevokedAdapter:
    """An api-key adapter whose impure ``fetch`` raises the typed auth break (AUT-R4)."""

    source_key: ClassVar[str] = "revoked_api"
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.API_KEY
    kind: ClassVar[SourceKind] = SourceKind.OAUTH_API
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"

    async def fetch(
        self, *, api_key: str | None, athlete_native_id: str | None, window: SyncWindow
    ) -> Iterable[Any]:
        raise AuthError(kind=FetchErrorKind.AUTH_REVOKED)

    def map(
        self, asbo: Any, source_descriptor: SourceDescriptorRef, fetch_context: FetchContext
    ) -> list[GboCandidate]:
        return []


class CancellingAdapter:
    """An api-key adapter whose ``fetch`` raises ``CancelledError`` (a mid-run cancel; CLI-R5)."""

    source_key: ClassVar[str] = "cancel_api"
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.API_KEY
    kind: ClassVar[SourceKind] = SourceKind.OAUTH_API
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"

    async def fetch(
        self, *, api_key: str | None, athlete_native_id: str | None, window: SyncWindow
    ) -> Iterable[Any]:
        raise asyncio.CancelledError

    def map(
        self, asbo: Any, source_descriptor: SourceDescriptorRef, fetch_context: FetchContext
    ) -> list[GboCandidate]:
        return []


# --------------------------------------------------------------------------- harness


@pytest_asyncio.fixture
async def session_factory(tmp_path: Path) -> AsyncIterator[Any]:
    """A transactional session-factory over a file-sqlite engine on a REAL QueuePool."""
    db = tmp_path / "faults.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db}",
        poolclass=AsyncAdaptedQueuePool,
        pool_size=5,
        connect_args={"timeout": 30},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _wal(dbapi_conn: Any, _rec: Any) -> None:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    @asynccontextmanager
    async def factory() -> AsyncIterator[AsyncSession]:
        async with sessionmaker() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    yield factory
    await engine.dispose()


async def _seed_athlete(factory: Any) -> str:
    """Seed the athlete + cycling sport; return the athlete id."""
    async with factory() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        athlete = Athlete(sex="male", reference_timezone="UTC")
        session.add(athlete)
        await session.flush()
        return str(athlete.athlete_id)


async def _add_connection(
    factory: Any, *, athlete_id: str, source_key: str, credential_ref: str | None
) -> str:
    """Add a CONNECTED api-key connection for ``source_key``; return its connection id."""
    async with factory() as session:
        descriptor = SourceDescriptor(
            source_key=source_key, display_name=source_key, kind="oauth_api"
        )
        session.add(descriptor)
        await session.flush()
        conn = Connection(
            athlete_id=uuid.UUID(athlete_id),
            source_descriptor_id=descriptor.source_descriptor_id,
            status=ConnectionStatus.CONNECTED,
            credential_ref=credential_ref,
            auth_archetype=AuthArchetype.API_KEY,
        )
        session.add(conn)
        await session.flush()
        return str(conn.connection_id)


def _cred_store() -> tuple[InMemoryCredentialStore, str]:
    cipher = EnvelopeCipher(EnvelopeCipher.generate_root_key())
    store = InMemoryCredentialStore(cipher)
    return store, store.store("api-key")


# -------------------------------------------------------- AUT-R4 multi-source isolation


async def test_revoked_source_isolates_from_a_healthy_source(session_factory: Any) -> None:
    """One source's auth break flips it to reauth while the OTHER source still lands (TST-R5)."""
    store, ref = _cred_store()
    athlete_id = await _seed_athlete(session_factory)
    healthy_conn = await _add_connection(
        session_factory, athlete_id=athlete_id, source_key="healthy_api", credential_ref=ref
    )
    revoked_conn = await _add_connection(
        session_factory, athlete_id=athlete_id, source_key="revoked_api", credential_ref=ref
    )
    orch = SyncOrchestrator(
        session_factory,
        registry=registry_from_adapters([HealthyAdapter(), RevokedAdapter()]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    run = await orch.run(athlete_id)  # no source filter -> both connections

    outcomes = {r.source_key: r.outcome for r in run.results}
    assert outcomes["healthy_api"] is SyncOutcome.OK  # the other source kept working
    assert outcomes["revoked_api"] is SyncOutcome.REAUTH_REQUIRED  # the broken one stopped
    assert run.activities_written == 1  # exactly the healthy source's ride
    # The healthy connection stays connected; only the revoked one flips (read back fresh).
    async with session_factory() as session:
        statuses = {
            str(c.connection_id): c.status
            for c in (await session.execute(select(Connection))).scalars().all()
        }
    assert statuses[healthy_conn] is ConnectionStatus.CONNECTED
    assert statuses[revoked_conn] is ConnectionStatus.REAUTH_REQUIRED
    # The healthy source's activity actually landed (no fabrication for the broken one).
    async with session_factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
    assert len(acts) == 1


async def test_reauth_required_source_is_not_resynced(session_factory: Any) -> None:
    """After flipping to reauth_required, a re-run SKIPS the source entirely (AUT-R4 gating)."""
    store, ref = _cred_store()
    athlete_id = await _seed_athlete(session_factory)
    await _add_connection(
        session_factory, athlete_id=athlete_id, source_key="revoked_api", credential_ref=ref
    )
    revoked = RevokedAdapter()
    orch = SyncOrchestrator(
        session_factory,
        registry=registry_from_adapters([revoked]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    first = await orch.run(athlete_id, source="revoked_api")
    assert first.results[0].outcome is SyncOutcome.REAUTH_REQUIRED

    # A SECOND run must not even select the connection — scheduling stops until reauth
    # (AUT-R4): no fetch is attempted, so no second 401 is hit and no result is produced.
    second = await orch.run(athlete_id, source="revoked_api")
    assert second.results == []


# ------------------------------------------------------------------ CLI-R5 cancellation


async def test_cancellation_mid_fetch_leaves_no_partial_and_no_auth_corruption(
    session_factory: Any,
) -> None:
    """A cancel mid-fetch propagates cleanly: no partial upsert, no auth corruption (CLI-R5)."""
    store, ref = _cred_store()
    athlete_id = await _seed_athlete(session_factory)
    cancel_conn = await _add_connection(
        session_factory, athlete_id=athlete_id, source_key="cancel_api", credential_ref=ref
    )
    # A pre-existing canonical activity (from an earlier healthy sync) so we can prove the
    # cancel neither WROTE a partial row nor DELETED prior data.
    async with session_factory() as session:
        session.add(
            Activity(
                athlete_id=uuid.UUID(athlete_id),
                sport="cycling",
                start_time=_dt.datetime(2026, 5, 20, 7, 0, tzinfo=UTC),
                elapsed_time_s=1200,
            )
        )
    orch = SyncOrchestrator(
        session_factory,
        registry=registry_from_adapters([CancellingAdapter()]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    # The cancellation propagates cleanly out of run() — it is NOT swallowed into a
    # DEGRADED/REAUTH result (CancelledError is a BaseException, not caught by the
    # source's graceful-degradation Exception guard, nor by the AuthError guard).
    with pytest.raises(asyncio.CancelledError):
        await orch.run(athlete_id, source="cancel_api")

    async with session_factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
        conn = (
            await session.execute(
                select(Connection).where(Connection.connection_id == uuid.UUID(cancel_conn))
            )
        ).scalar_one()
    # No half-applied upsert for the cancelled source — exactly the ONE prior activity
    # remains (nothing written, nothing deleted; ING-UPS-R3 / ING-R4).
    assert len(acts) == 1
    assert acts[0].elapsed_time_s == 1200  # the prior row, untouched
    # The cancelled source did NOT corrupt its auth/status row — it remains CONNECTED
    # (no spurious reauth/error flip), so a later run can retry it (CLI-R5).
    assert conn.status is ConnectionStatus.CONNECTED


async def test_cancellation_preserves_a_committed_source(session_factory: Any) -> None:
    """A committed source's data survives a later source's cancel (CLI-R5/ING-UPS-R3).

    Driven purely through the public ``run`` API in a deterministic order: the healthy
    source is synced (and COMMITS in its own transaction) first; a subsequent run of the
    cancelling source raises ``CancelledError`` — proving the already-committed data
    survives a later cancel, with no duplicate and no fabricated row for the cancel.
    """
    store, ref = _cred_store()
    athlete_id = await _seed_athlete(session_factory)
    await _add_connection(
        session_factory, athlete_id=athlete_id, source_key="healthy_api", credential_ref=ref
    )
    await _add_connection(
        session_factory, athlete_id=athlete_id, source_key="cancel_api", credential_ref=ref
    )
    orch = SyncOrchestrator(
        session_factory,
        registry=registry_from_adapters([HealthyAdapter(), CancellingAdapter()]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    # Healthy source first: it lands and COMMITS its batch.
    healthy_run = await orch.run(athlete_id, source="healthy_api")
    assert healthy_run.results[0].outcome is SyncOutcome.OK
    # Then the cancelling source: the cancel propagates cleanly (not swallowed).
    with pytest.raises(asyncio.CancelledError):
        await orch.run(athlete_id, source="cancel_api")

    async with session_factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
    # The healthy source's committed activity SURVIVES the later cancel (ING-UPS-R3),
    # with no fabricated row for the cancelled source.
    assert len(acts) == 1
    assert float(acts[0].avg_power_w) == pytest.approx(250.0)


# ------------------------------------------------------ ONB-R5 source-removed isolation


async def test_removed_source_degrades_in_isolation(session_factory: Any) -> None:
    """A connection whose source is no longer installed degrades; the other still lands (ONB-R5)."""
    store, ref = _cred_store()
    athlete_id = await _seed_athlete(session_factory)
    await _add_connection(
        session_factory, athlete_id=athlete_id, source_key="healthy_api", credential_ref=ref
    )
    # A connection for a source that is NOT in the registry (uninstalled/removed).
    await _add_connection(
        session_factory, athlete_id=athlete_id, source_key="gone_api", credential_ref=ref
    )
    # Pre-existing canonical data from the removed source's earlier healthy sync.
    async with session_factory() as session:
        session.add(
            Activity(
                athlete_id=uuid.UUID(athlete_id),
                sport="cycling",
                start_time=_dt.datetime(2026, 5, 1, 7, 0, tzinfo=UTC),
                elapsed_time_s=1800,
            )
        )
    orch = SyncOrchestrator(
        session_factory,
        registry=registry_from_adapters([HealthyAdapter()]),  # gone_api NOT registered
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    run = await orch.run(athlete_id)

    outcomes = {r.source_key: r.outcome for r in run.results}
    assert outcomes["healthy_api"] is SyncOutcome.OK  # the installed source kept working
    assert outcomes["gone_api"] is SyncOutcome.DEGRADED  # removed source degrades, not crashes
    # Prior canonical data from the removed source is NOT deleted (ING-R4); plus the new
    # healthy ride -> two activities total.
    async with session_factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
    assert len(acts) == 2
