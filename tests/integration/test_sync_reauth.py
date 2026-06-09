"""Sync-engine credential-revocation handling (AUT-R4, TST-R5).

When a source's typed client raises an auth error (a revoked/expired ``api_key`` -> 401/403,
the OSS Intervals path) the orchestrator MUST NOT silently degrade while leaving the
Connection ``connected``. Per AUT-R4 it MUST:

  (1) set the Connection ``status`` to ``reauth_required`` (the canonical enum, doc 20),
  (2) emit a typed gap (§7) explaining degraded coverage, and
  (3) stop scheduling that source's syncs until the athlete re-authorizes,

without deleting previously-ingested canonical data (ING-R4). This is the fail-closed
behavior the clause forbids skipping. The §4.2 transient path stays DEGRADED-and-retry;
only a non-transient auth break flips the status.

Real-pool data safety: a throwaway FILE-sqlite engine on a real ``QueuePool`` (WAL +
``busy_timeout``) — never ``:memory:`` / ``StaticPool``, never a host/live DB — so the
persisted status transition is read back through a real second connection.
"""

from __future__ import annotations

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

from tests.integration._session_provider import FactorySessionProvider
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import (
    AuthArchetype,
    ConnectionStatus,
    GapReason,
    GapState,
    SourceKind,
)
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
    IngestionGap,
    SourceDescriptor,
    Sport,
)
from wattwise_core.security.credentials import InMemoryCredentialStore
from wattwise_core.security.crypto import EnvelopeCipher

UTC = _dt.UTC
_FIXED_NOW = _dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC)

pytestmark = pytest.mark.integration


class RevokedKeyAdapter:
    """An api-key adapter whose impure ``fetch`` raises the typed auth error (AUT-R4).

    Models a revoked/expired Intervals ``api_key``: the client probe/GET would return
    401/403, which the typed client converts to ``AuthError(kind=AUTH_REVOKED)`` rather
    than a raw ``httpx.HTTPStatusError`` (CLI-R7).
    """

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


@pytest_asyncio.fixture
async def session_factory(tmp_path: Path) -> AsyncIterator[Any]:
    """A transactional session-factory over a file-sqlite engine on a REAL QueuePool."""
    db = tmp_path / "reauth.db"
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


async def _seed(factory: Any, *, ref: str) -> tuple[str, str]:
    async with factory() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        athlete = Athlete(sex="male", reference_timezone="UTC")
        session.add(athlete)
        descriptor = SourceDescriptor(
            source_key="revoked_api", display_name="revoked_api", kind="oauth_api"
        )
        session.add(descriptor)
        await session.flush()
        conn = Connection(
            athlete_id=athlete.athlete_id,
            source_descriptor_id=descriptor.source_descriptor_id,
            status=ConnectionStatus.CONNECTED,
            credential_ref=ref,
            auth_archetype=AuthArchetype.API_KEY,
        )
        session.add(conn)
        await session.flush()
        return str(athlete.athlete_id), str(conn.connection_id)


def _cred_store() -> tuple[InMemoryCredentialStore, str]:
    cipher = EnvelopeCipher(EnvelopeCipher.generate_root_key())
    store = InMemoryCredentialStore(cipher)
    return store, store.store("revoked-key")


async def test_revoked_key_sets_reauth_required_and_emits_gap(session_factory: Any) -> None:
    """A revoked api_key flips status to reauth_required + emits a typed gap (AUT-R4)."""
    store, ref = _cred_store()
    athlete_id, connection_id = await _seed(session_factory, ref=ref)
    orch = SyncOrchestrator(
        FactorySessionProvider(session_factory),
        registry=registry_from_adapters([RevokedKeyAdapter()]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    run = await orch.run(athlete_id, source="revoked_api")

    # (3) The source stops with a distinct REAUTH_REQUIRED outcome — NOT generic DEGRADED.
    assert len(run.results) == 1
    result = run.results[0]
    assert result.outcome is SyncOutcome.REAUTH_REQUIRED
    # (2) A typed gap (§7) is emitted carrying the canonical reauth reason.
    assert result.gap is not None
    assert result.gap.reason in {GapReason.NEEDS_REAUTH, GapReason.AUTH_REVOKED}
    # (1) The persisted Connection status flipped — read back via a real second connection.
    async with session_factory() as session:
        conn = (
            await session.execute(
                select(Connection).where(Connection.connection_id == uuid.UUID(connection_id))
            )
        ).scalar_one()
        assert conn.status is ConnectionStatus.REAUTH_REQUIRED


async def test_revoked_key_persists_a_queryable_terminal_gap(session_factory: Any) -> None:
    """The reauth gap is PERSISTED + queryable, not just an in-memory signal (AUT-R4/ING-GAP-R1).

    AUT-R4 mandates the adapter "MUST emit a typed gap (§7)" and ING-GAP-R1 mandates every
    partial failure be "recorded as a typed gap … queryable by downstream consumers" — never
    swallowed nor logged-only. The in-memory ``result.gap`` SIGNAL is not enough: a downstream
    consumer (analytics gates on it; the agent grounds on it; the §9 data-health surface renders
    it) reads the PERSISTED ``ingestion_gap`` row. So we read the gap back through a real second
    connection and assert it is the canonical terminal reauth gap, scoped to the right athlete +
    source. The assertion is non-vacuous: it would FAIL on the pre-fix code, which set the
    Connection status but never persisted any gap (the query would return zero rows).
    """
    store, ref = _cred_store()
    athlete_id, _ = await _seed(session_factory, ref=ref)
    orch = SyncOrchestrator(
        FactorySessionProvider(session_factory),
        registry=registry_from_adapters([RevokedKeyAdapter()]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    await orch.run(athlete_id, source="revoked_api")

    # Read the PERSISTED gap back via a real second connection (not the in-memory signal).
    async with session_factory() as session:
        stmt = select(IngestionGap).where(IngestionGap.athlete_id == uuid.UUID(athlete_id))
        gaps = (await session.execute(stmt)).scalars().all()
    assert len(gaps) == 1, "AUT-R4/ING-GAP-R1: a queryable typed gap row MUST be persisted"
    gap = gaps[0]
    assert gap.reason is GapReason.NEEDS_REAUTH  # the canonical AUT-R4 reauth reason (§7)
    assert gap.transient is False  # terminal: a revoked credential never self-heals (ING-GAP-R4)
    assert gap.state is GapState.OPEN  # open until the athlete re-authorizes
    assert gap.source_descriptor_id is not None  # scoped to the failing source (ING-GAP-R2)


async def test_revoked_key_does_not_delete_prior_data(session_factory: Any) -> None:
    """Setting reauth_required MUST NOT delete previously-ingested data (AUT-R4/ING-R4)."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed(session_factory, ref=ref)
    # Pre-existing canonical activity for the athlete (from an earlier healthy sync).
    async with session_factory() as session:
        session.add(
            Activity(
                athlete_id=uuid.UUID(athlete_id),
                sport="cycling",
                start_time=_dt.datetime(2026, 5, 30, 8, 0, tzinfo=UTC),
                elapsed_time_s=3600,
            )
        )
    orch = SyncOrchestrator(
        FactorySessionProvider(session_factory),
        registry=registry_from_adapters([RevokedKeyAdapter()]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    await orch.run(athlete_id, source="revoked_api")

    async with session_factory() as session:
        remaining = (await session.execute(select(Activity))).scalars().all()
    assert len(remaining) == 1  # prior data preserved (ING-R4)
