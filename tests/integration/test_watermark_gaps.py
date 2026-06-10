"""Watermark + typed-gap ingestion journey (SYN-R2/R3, ING-GAP-R2..R5, ADP-R6).

Exercises the trustworthy-ingestion cluster end to end on a REAL multi-connection pool
(file-SQLite + WAL + busy_timeout — NEVER ``:memory:``/StaticPool, which hides the
crash-resume cursor behaviour and write serialisation, per the data-safety rule):

* SYN-R2 — a watermark is persisted per ``(athlete, source, gbo_type[, stream])`` with a
  high-water cursor + content hint.
* SYN-R3 — the watermark advances ONLY after the batch is durably committed, and a fresh
  orchestrator over the SAME file DB resumes from the committed cursor (crash-resume).
* ADP-R6 — an incremental fetch skips the already-watermarked range: the next run's fetch
  window starts at the watermark, not the fixed lookback.
* ING-GAP-R2/R3/R4 — a typed gap carries the mandated identity fields + the 10-member
  ``GapReason`` taxonomy + open/closed state; a transient time-range gap self-heals on a
  later sync that covers its range.
* ING-GAP-R5 / ING-UPS-R3 — a single record whose map raises is gap-marked to exactly that
  record range while every good record in the same batch still commits.
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

from tests.integration._fake_capability import fake_capability
from tests.integration._session_provider import FactorySessionProvider
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import (
    AuthArchetype,
    ConnectionStatus,
    Fidelity,
    GapReason,
    GapState,
    GboType,
    SourceKind,
)
from wattwise_core.ingestion._sync_records import incremental_floor_date, watermark_floor
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef
from wattwise_core.ingestion.capability import CapabilityDescriptor
from wattwise_core.ingestion.registry import registry_from_adapters
from wattwise_core.ingestion.sync import SyncOrchestrator, SyncOutcome, SyncWindow
from wattwise_core.ingestion.watermark import (
    advance_watermark,
    close_covering_gaps,
    open_gap,
    watermark_for,
)
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    Base,
    Connection,
    IngestionGap,
    IngestionWatermark,
    SourceDescriptor,
    Sport,
)
from wattwise_core.security.credentials import InMemoryCredentialStore
from wattwise_core.security.crypto import EnvelopeCipher
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.integration

UTC = _dt.UTC
_FIXED_NOW = _dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + long busy_timeout per SQLite connection so the real pool serialises writers."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


# --------------------------------------------------------------------------- fakes


class _RideAsbo:
    """A trivial source-shaped object the fake adapter's ``map`` consumes."""

    def __init__(self, native_id: str, watts: float, day: int) -> None:
        self.native_id = native_id
        self.watts = watts
        self.day = day


class _PoisonAsbo:
    """A source-shaped object whose pure map raises (per-record failure, ING-GAP-R5)."""

    def __init__(self, native_id: str) -> None:
        self.native_id = native_id


class FakeApiAdapter:
    """An in-process api-key adapter: impure ``fetch`` + pure ``map`` (ADP-R*)."""

    source_key: ClassVar[str] = "fake_api"
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.API_KEY
    kind: ClassVar[SourceKind] = SourceKind.OAUTH_API
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"
    capability: ClassVar[CapabilityDescriptor] = fake_capability("fake_api")

    def __init__(self, asbos: list[Any]) -> None:
        self._asbos = asbos
        self.seen_window: SyncWindow | None = None

    async def fetch(
        self, *, api_key: str | None, athlete_native_id: str | None, window: SyncWindow
    ) -> Iterable[Any]:
        self.seen_window = window
        return list(self._asbos)

    def map(
        self, asbo: Any, source_descriptor: SourceDescriptorRef, fetch_context: FetchContext
    ) -> list[GboCandidate]:
        if isinstance(asbo, _PoisonAsbo):
            raise ValueError("unmappable record: required canonical field missing")
        if not isinstance(asbo, _RideAsbo):
            return []
        start = _dt.datetime(2026, 6, asbo.day, 8, 0, tzinfo=UTC)
        payload: dict[str, Any] = {
            "start_time": start,
            "sport": "cycling",
            "elapsed_time_s": 3600,
            "moving_time_s": 3600,
            "avg_power_w": asbo.watts,
        }
        return [
            GboCandidate(
                gbo_type="activity",
                source_descriptor_id=source_descriptor.source_descriptor_id,
                source_native_id=asbo.native_id,
                content_hash=content_hash(f"{asbo.native_id}:{asbo.watts}".encode()),
                payload=payload,
                observed_at=start,
                fetched_at=fetch_context.fetched_at,
                trust_tier=Fidelity.RAW_STREAM,
                connection_id=fetch_context.connection_id,
                adapter_version=self.adapter_version,
                mapping_version=self.mapping_version,
            )
        ]


# --------------------------------------------------------------------------- harness


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[tuple[Any, Any]]:
    """A REAL file-SQLite pool (WAL + busy_timeout) + a transactional session factory.

    Deliberately NOT ``:memory:``/StaticPool: a file-backed real pool is what lets a
    SECOND orchestrator see the FIRST run's committed watermark (crash-resume, SYN-R3).
    """
    dsn = f"sqlite+aiosqlite:///{tmp_path}/canon.sqlite"
    engine = create_async_engine(dsn)
    event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
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

    yield factory, sessionmaker
    await engine.dispose()


async def _seed(factory: Any, *, ref: str | None) -> tuple[str, str]:
    """Seed the athlete, the cycling sport, a source descriptor, and an api-key connection."""
    async with factory() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        athlete = Athlete(sex="male", reference_timezone="UTC")
        session.add(athlete)
        descriptor = SourceDescriptor(
            source_key="fake_api", display_name="fake_api", kind="oauth_api"
        )
        session.add(descriptor)
        await session.flush()
        session.add(
            Connection(
                athlete_id=athlete.athlete_id,
                source_descriptor_id=descriptor.source_descriptor_id,
                status=ConnectionStatus.CONNECTED,
                credential_ref=ref,
                auth_archetype=AuthArchetype.API_KEY,
            )
        )
        await session.flush()
        return str(athlete.athlete_id), str(descriptor.source_descriptor_id)


def _cred_store() -> tuple[InMemoryCredentialStore, str]:
    cipher = EnvelopeCipher(EnvelopeCipher.generate_root_key())
    store = InMemoryCredentialStore(cipher)
    return store, store.store("secret-api-key-123")


def _orch(factory: Any, asbos: list[Any], store: InMemoryCredentialStore) -> SyncOrchestrator:
    return SyncOrchestrator(
        FactorySessionProvider(factory),
        registry=registry_from_adapters([FakeApiAdapter(asbos)]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )


# --------------------------------------------------------------------------- tests


async def test_watermark_persisted_per_scope_with_cursor_and_hint(db: Any) -> None:
    """SYN-R2: a sync writes ONE watermark per (athlete, source, gbo_type) with hint."""
    factory, _ = db
    store, ref = _cred_store()
    athlete_id, _descriptor_id = await _seed(factory, ref=ref)
    await _orch(factory, [_RideAsbo("ride-1", 250.0, day=1)], store).run(
        athlete_id, source="fake_api", window=SyncWindow("2026-05-20", "2026-06-01")
    )

    async with factory() as session:
        rows = (await session.execute(select(IngestionWatermark))).scalars().all()
    assert len(rows) == 1
    wm = rows[0]
    assert wm.gbo_type == GboType.ACTIVITY
    assert wm.high_water_at == _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    # content hint (SYN-R2): the last record's content_hash so a changed record re-fetches.
    assert wm.content_hint == content_hash(b"ride-1:250.0")


async def test_watermark_advances_only_after_commit_and_resumes(db: Any) -> None:
    """SYN-R3: the committed cursor survives; a fresh orchestrator resumes from it."""
    factory, sessionmaker = db
    store, ref = _cred_store()
    athlete_id, descriptor_id = await _seed(factory, ref=ref)
    await _orch(factory, [_RideAsbo("ride-1", 250.0, day=1)], store).run(
        athlete_id, source="fake_api", window=SyncWindow("2026-05-20", "2026-06-01")
    )

    # A SECOND orchestrator over the SAME committed file DB reads the prior cursor — the
    # crash-resume guarantee: the watermark advanced only because the batch durably landed.
    async with sessionmaker() as s:
        wm = await watermark_for(
            s, uuid.UUID(athlete_id), uuid.UUID(descriptor_id), GboType.ACTIVITY
        )
    assert wm is not None
    assert wm.high_water_at == _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)


async def test_incremental_fetch_skips_watermarked_range(db: Any) -> None:
    """ADP-R6: the second incremental run's fetch window starts at the watermark."""
    factory, _ = db
    store, ref = _cred_store()
    athlete_id, _ = await _seed(factory, ref=ref)
    # First run lands a ride with high-water = 2026-06-01 08:00.
    await _orch(factory, [_RideAsbo("ride-1", 250.0, day=1)], store).run(
        athlete_id, source="fake_api", window=SyncWindow("2026-05-20", "2026-06-01")
    )
    # Second run is INCREMENTAL (no explicit window): its window must start at the
    # watermark date (2026-06-01), NOT the fixed 42-day lookback.
    adapter = FakeApiAdapter([_RideAsbo("ride-2", 260.0, day=2)])
    orch = SyncOrchestrator(
        FactorySessionProvider(factory),
        registry=registry_from_adapters([adapter]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )
    await orch.run(athlete_id, source="fake_api")  # incremental

    assert adapter.seen_window is not None
    assert adapter.seen_window.oldest == "2026-06-01"  # watermark floor, not lookback


async def test_per_record_isolation_commits_good_and_gaps_failed(db: Any) -> None:
    """ING-GAP-R5/ING-UPS-R3: a bad record gap-marks ONLY itself; good records commit."""
    factory, _ = db
    store, ref = _cred_store()
    athlete_id, descriptor_id = await _seed(factory, ref=ref)
    asbos: list[Any] = [
        _RideAsbo("good-1", 250.0, day=1),
        _PoisonAsbo("bad-1"),
        _RideAsbo("good-2", 260.0, day=2),
    ]
    run = await _orch(factory, asbos, store).run(
        athlete_id, source="fake_api", window=SyncWindow("2026-05-20", "2026-06-03")
    )

    assert run.results[0].outcome is SyncOutcome.DEGRADED  # partial failure
    assert run.activities_written == 2  # BOTH good records committed
    async with factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
        gaps = (await session.execute(select(IngestionGap))).scalars().all()
    assert len(acts) == 2
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.reason == GapReason.MAPPING_FIELD_MISSING
    assert gap.state == GapState.OPEN
    # TERMINAL, not transient (ING-GAP-R2/R4): a map raising is a code/schema defect — a
    # deterministic pure map (MAP-R1) re-fails on re-run, so it needs an operator fix and
    # MUST NOT be classed auto-retryable (which would also leave it stuck-open forever,
    # since close_covering_gaps only heals TIME-range gaps, never token-only ones).
    assert gap.transient is False
    assert str(gap.athlete_id) == athlete_id
    assert str(gap.source_descriptor_id) == descriptor_id
    # Range-precise (ING-GAP-R5): the gap covers EXACTLY the failed record token.
    assert gap.range_start_token == "bad-1"
    assert gap.range_end_token == "bad-1"
    assert gap.ingest_run_id is not None
    assert gap.first_seen_at is not None and gap.last_seen_at is not None


async def test_transient_time_gap_self_heals_on_resync(db: Any) -> None:
    """ING-GAP-R4: a later sync covering a transient gap's range closes it + stamps closure."""
    factory, _ = db
    store, ref = _cred_store()
    athlete_id, descriptor_id = await _seed(factory, ref=ref)
    athlete = uuid.UUID(athlete_id)
    descriptor = uuid.UUID(descriptor_id)
    # Pre-open a TRANSIENT time-range gap (a prior fetch over this range failed).
    async with factory() as session:
        await open_gap(
            session,
            athlete,
            descriptor,
            GboType.ACTIVITY,
            reason=GapReason.FETCH_FAILED,
            seen_at=_dt.datetime(2026, 5, 25, tzinfo=UTC),
            transient=True,
            range_start_at=_dt.datetime(2026, 5, 25, tzinfo=UTC),
            range_end_at=_dt.datetime(2026, 5, 26, tzinfo=UTC),
        )

    # A successful sync whose window FULLY covers [05-25, 05-26] self-heals the gap.
    await _orch(factory, [_RideAsbo("ride-1", 250.0, day=25)], store).run(
        athlete_id, source="fake_api", window=SyncWindow("2026-05-24", "2026-05-27")
    )

    async with factory() as session:
        gaps = (await session.execute(select(IngestionGap))).scalars().all()
    assert len(gaps) == 1
    assert gaps[0].state == GapState.CLOSED  # self-healed (ING-GAP-R4)
    assert gaps[0].closed_at is not None  # closure time recorded


async def test_watermark_advance_is_forward_only(db: Any) -> None:
    """SYN-R3/SYN-R4: an earlier instant never regresses the cursor (monotonic)."""
    factory, _ = db
    athlete_id, descriptor_id = await _seed(factory, ref=None)
    athlete = uuid.UUID(athlete_id)
    descriptor = uuid.UUID(descriptor_id)
    late = _dt.datetime(2026, 6, 10, tzinfo=UTC)
    early = _dt.datetime(2026, 6, 1, tzinfo=UTC)
    async with factory() as session:
        await advance_watermark(
            session,
            athlete,
            descriptor,
            GboType.ACTIVITY,
            high_water_at=late,
            content_hint="h1",
        )
    async with factory() as session:
        await advance_watermark(
            session,
            athlete,
            descriptor,
            GboType.ACTIVITY,
            high_water_at=early,
            content_hint="h2",
        )
        wm = await watermark_for(session, athlete, descriptor, GboType.ACTIVITY)
    assert wm is not None
    assert wm.high_water_at == late  # forward-only: the earlier instant did NOT regress it
    assert wm.content_hint == "h2"  # but the hint always refreshes (SYN-R2)


async def test_close_covering_gaps_skips_partial_overlap(db: Any) -> None:
    """ING-GAP-R4: a gap only PARTIALLY covered by the synced range stays open."""
    factory, _ = db
    athlete_id, descriptor_id = await _seed(factory, ref=None)
    athlete = uuid.UUID(athlete_id)
    descriptor = uuid.UUID(descriptor_id)
    async with factory() as session:
        await open_gap(
            session,
            athlete,
            descriptor,
            GboType.ACTIVITY,
            reason=GapReason.FETCH_FAILED,
            seen_at=_dt.datetime(2026, 5, 25, tzinfo=UTC),
            transient=True,
            range_start_at=_dt.datetime(2026, 5, 20, tzinfo=UTC),
            range_end_at=_dt.datetime(2026, 5, 30, tzinfo=UTC),
        )
    async with factory() as session:
        closed = await close_covering_gaps(
            session,
            athlete,
            descriptor,
            GboType.ACTIVITY,
            range_start_at=_dt.datetime(2026, 5, 22, tzinfo=UTC),
            range_end_at=_dt.datetime(2026, 5, 28, tzinfo=UTC),  # does NOT fully cover
            closed_at=_FIXED_NOW,
        )
    assert closed == 0
    async with factory() as session:
        gaps = (await session.execute(select(IngestionGap))).scalars().all()
    assert gaps[0].state == GapState.OPEN  # still open — only fully-covered gaps heal


async def test_terminal_mapping_gap_is_not_auto_closed_by_covering_sync(db: Any) -> None:
    """ING-GAP-R4/R2: a terminal mapping_field_missing gap is NEVER auto-healed.

    A map raising is a code/schema defect (deterministic re-failure, MAP-R1), so the gap is
    TERMINAL — a later successful sync that fully covers the SAME range and gbo_type MUST
    leave it open (only operator action closes it); only TRANSIENT gaps self-heal.
    """
    factory, _ = db
    store, ref = _cred_store()
    athlete_id, _descriptor_id = await _seed(factory, ref=ref)
    # First run: a poison record opens a TERMINAL mapping gap; a good record commits + advances
    # the activity watermark to 2026-06-01 08:00.
    await _orch(factory, [_RideAsbo("good-1", 250.0, day=1), _PoisonAsbo("bad-1")], store).run(
        athlete_id, source="fake_api", window=SyncWindow("2026-05-20", "2026-06-03")
    )

    # A later sync whose window FULLY covers the activity range must NOT close the terminal
    # mapping gap (contrast test_transient_time_gap_self_heals_on_resync, which DOES heal).
    adapter = FakeApiAdapter([_RideAsbo("good-2", 260.0, day=2)])
    await SyncOrchestrator(
        FactorySessionProvider(factory),
        registry=registry_from_adapters([adapter]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    ).run(athlete_id, source="fake_api", window=SyncWindow("2026-05-20", "2026-06-03"))

    async with factory() as session:
        gaps = (await session.execute(select(IngestionGap))).scalars().all()
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.reason == GapReason.MAPPING_FIELD_MISSING
    assert gap.transient is False
    assert gap.state == GapState.OPEN  # terminal gap stays open through a covering resync
    assert gap.closed_at is None


async def test_floor_is_min_across_gbo_types_so_lagging_scope_not_skipped(db: Any) -> None:
    """ADP-R6/SYN-R2: the incremental floor is the LEAST-advanced cursor, not the max.

    Two gbo_types under one source carry independent cursors (SYN-R2). With activity at a
    LATER high-water than daily_wellness, the floor MUST be the EARLIER (wellness) cursor —
    a max floor would over-advance past wellness's un-fetched range and silently skip it.
    """
    factory, sessionmaker = db
    athlete_id, descriptor_id = await _seed(factory, ref=None)
    athlete = uuid.UUID(athlete_id)
    descriptor = uuid.UUID(descriptor_id)
    activity_hw = _dt.datetime(2026, 6, 10, 8, 0, tzinfo=UTC)  # most-advanced
    wellness_hw = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)  # LAGGING — the conservative floor
    async with factory() as session:
        await advance_watermark(
            session,
            athlete,
            descriptor,
            GboType.ACTIVITY,
            high_water_at=activity_hw,
            content_hint="a",
        )
        await advance_watermark(
            session,
            athlete,
            descriptor,
            GboType.DAILY_WELLNESS,
            high_water_at=wellness_hw,
            content_hint="w",
        )

    async with sessionmaker() as s:
        floor = await watermark_floor(s, athlete, descriptor)
    assert floor == wellness_hw  # MIN, not the activity max — the lagging scope wins

    # And the ISO incremental floor honours it: it advances forward to the lagging cursor's
    # date, never to the activity max (which would skip 06-02..06-09 of wellness).
    iso = await incremental_floor_date(factory, athlete, descriptor, oldest_iso="2026-05-01")
    assert iso == "2026-06-01"
    assert iso != activity_hw.date().isoformat()  # mutation guard: NOT the max-floor bug


async def test_floor_uses_only_advanced_scopes(db: Any) -> None:
    """ADP-R6: a never-advanced gbo_type scope contributes no floor (returns None if none)."""
    factory, sessionmaker = db
    athlete_id, descriptor_id = await _seed(factory, ref=None)
    athlete = uuid.UUID(athlete_id)
    descriptor = uuid.UUID(descriptor_id)
    async with sessionmaker() as s:
        assert await watermark_floor(s, athlete, descriptor) is None  # no scope advanced yet
    only_hw = _dt.datetime(2026, 6, 5, 8, 0, tzinfo=UTC)
    async with factory() as session:
        await advance_watermark(
            session,
            athlete,
            descriptor,
            GboType.ACTIVITY,
            high_water_at=only_hw,
            content_hint="a",
        )
    async with sessionmaker() as s:
        assert await watermark_floor(s, athlete, descriptor) == only_hw  # the one advanced scope
