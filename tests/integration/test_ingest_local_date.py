"""Ingest write-path local-date projection — GBO-R33/R34/R35, CFG-R1a fail-closed.

The canonical write path MUST project the stored UTC ``start_time`` into the athlete's
reference timezone (effective as-of that instant) and persist BOTH the display
``start_time_local`` (local wall-clock) and the reproducible ``local_date`` bucket
(GBO-R35), using stdlib ``zoneinfo``. The UTC instant stays the source of truth: the same
instant + tz history re-projects to the same ``local_date`` (GBO-R33/R34). A non-UTC
athlete whose activity at a UTC instant lands on a DIFFERENT local calendar day than the
UTC day is the decisive case. A missing reference timezone fails closed (CFG-R1a/R6) — the
record is isolated, never silently bucketed under a code-baked UTC default.

Real file-SQLite pool (WAL + busy_timeout), never ``:memory:``/StaticPool.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    Base,
    SourceDescriptor,
    Sport,
)
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.integration

UTC = _dt.UTC


def _enable_wal(dbapi_conn: object, _record: object) -> None:
    """WAL + busy_timeout per connection so this is a REAL pool, not StaticPool."""
    cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


@pytest_asyncio.fixture
async def pool(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A REAL file-SQLite QueuePool (WAL + busy_timeout) — never :memory:/StaticPool."""
    dsn = f"sqlite+aiosqlite:///{tmp_path}/localdate_{uuid.uuid4().hex}.sqlite"
    engine = create_async_engine(dsn)
    event.listen(engine.sync_engine, "connect", _enable_wal)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield factory
    finally:
        await engine.dispose()


async def _seed(
    factory: async_sessionmaker[AsyncSession],
    *,
    tz: str = "America/New_York",
    eff: _dt.datetime | None = None,
) -> tuple[str, str]:
    """Seed cycling, a non-UTC athlete, and one source descriptor."""
    async with factory() as s:
        s.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        athlete = Athlete(sex="male", reference_timezone=tz, reference_timezone_effective_from=eff)
        descriptor = SourceDescriptor(
            source_key="other_src", display_name="Other", kind="oauth_api"
        )
        s.add_all([athlete, descriptor])
        await s.flush()
        ids = (str(athlete.athlete_id), str(descriptor.source_descriptor_id))
        await s.commit()
    return ids


def _ride(native_id: str, start: _dt.datetime) -> GboCandidate:
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(f"{native_id}:{start.isoformat()}".encode()),
        payload={
            "start_time": start,
            "sport": "cycling",
            "elapsed_time_s": 1800,
            "avg_power_w": 200.0,
        },
        trust_tier=Fidelity.RAW_STREAM,
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )


async def _only_activity(factory: async_sessionmaker[AsyncSession]) -> Activity:
    async with factory() as s:
        return (await s.execute(select(Activity))).scalars().one()


async def test_write_path_assigns_local_date_across_midnight(
    pool: async_sessionmaker[AsyncSession],
) -> None:
    """A non-UTC activity gets local_date = the LOCAL calendar day, not the UTC day (GBO-R35).

    2026-06-02 03:00Z for an America/New_York athlete is 2026-06-01 23:00 local. The stored
    ``local_date`` MUST be 2026-06-01 (local), NOT the 2026-06-02 UTC date a ``.date()`` on
    ``start_time`` would give — that mutation is what breaks this assertion.
    """
    athlete_id, descriptor = await _seed(pool)
    instant = _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    async with pool() as session:
        await IngestService(session).ingest(athlete_id, descriptor, [_ride("r1", instant)])
        await session.commit()
    act = await _only_activity(pool)
    assert act.start_time.astimezone(UTC) == instant  # the UTC instant stays the source of truth
    assert act.local_date == _dt.date(2026, 6, 1)  # LOCAL bucket, not the UTC 06-02
    # start_time_local carries the LOCAL wall-clock for display (GBO-R13/§3.8).
    assert act.start_time_local is not None
    assert (act.start_time_local.year, act.start_time_local.month, act.start_time_local.day) == (
        2026,
        6,
        1,
    )
    assert (act.start_time_local.hour, act.start_time_local.minute) == (23, 0)


async def test_local_date_is_reproducible_from_utc_instant(
    pool: async_sessionmaker[AsyncSession],
) -> None:
    """Re-ingesting the same instant re-projects to the same local_date (GBO-R34)."""
    athlete_id, descriptor = await _seed(pool)
    instant = _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    async with pool() as session:
        await IngestService(session).ingest(athlete_id, descriptor, [_ride("r1", instant)])
        await session.commit()
    async with pool() as session:
        await IngestService(session).ingest(athlete_id, descriptor, [_ride("r1", instant)])
        await session.commit()
    async with pool() as session:
        n = (await session.execute(select(func.count()).select_from(Activity))).scalar_one()
        act = (await session.execute(select(Activity))).scalars().one()
    assert n == 1  # idempotent
    assert act.local_date == _dt.date(2026, 6, 1)  # stable bucket


async def test_dst_transition_activity_buckets_to_correct_local_day(
    pool: async_sessionmaker[AsyncSession],
) -> None:
    """A DST-eve instant buckets to the correct local day on the write path (GBO-R33, DST).

    2026-03-08 04:30Z (the morning US clocks spring forward) is 2026-03-07 23:30 EST
    (offset -5, pre-jump) → local_date 2026-03-07, not the UTC 2026-03-08.
    """
    athlete_id, descriptor = await _seed(pool)
    instant = _dt.datetime(2026, 3, 8, 4, 30, tzinfo=UTC)
    async with pool() as session:
        await IngestService(session).ingest(athlete_id, descriptor, [_ride("dst", instant)])
        await session.commit()
    act = await _only_activity(pool)
    assert act.local_date == _dt.date(2026, 3, 7)


async def test_missing_reference_timezone_fails_closed_on_write_path(
    pool: async_sessionmaker[AsyncSession],
) -> None:
    """A blank reference timezone isolates the record — never a code-baked UTC (CFG-R1a/R6).

    The activity cannot be bucketed without an authoritative tz, so the write-path record is
    rejected (fault-isolated, counted) rather than silently attributed to a UTC default.
    """
    athlete_id, descriptor = await _seed(pool, tz="")  # no reference timezone configured
    instant = _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC)
    async with pool() as session:
        result = await IngestService(session).ingest(athlete_id, descriptor, [_ride("r1", instant)])
        await session.commit()
    assert result.candidates_failed == 1  # fail-closed: isolated, not bucketed under UTC
    async with pool() as session:
        n = (await session.execute(select(func.count()).select_from(Activity))).scalar_one()
    assert n == 0  # no canonical activity written without an authoritative tz
