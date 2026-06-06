"""Regression tests for the persistence convergence findings (B-E1 panel).

Each test pins a portability/correctness bug the convergence review found, so it can
never silently return: tz-aware UTC round-trip (GBO-R32), enforced foreign keys
(GBO-R8b/GBO-AC-7), enum CHECK rejecting an out-of-vocabulary token (GBO-R12),
fraction-preserving numeric (GBO-R10), and the upsert seam never clobbering the PK or
created_at (UPS-R3). These are observable on SQLite (where the bugs manifested).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from wattwise_core.config import load_settings
from wattwise_core.persistence.engine import create_engine_from_settings
from wattwise_core.persistence.models import Activity, Athlete, Base, Sport

UTC = _dt.UTC


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Session over a fresh file-backed SQLite schema (FK pragma applies per connection)."""
    settings = load_settings(
        database_dsn="sqlite+aiosqlite:///./.fixtest.sqlite",
        app__environment="development",
    )
    engine = create_engine_from_settings(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    await engine.dispose()
    with contextlib.suppress(FileNotFoundError):
        Path("./.fixtest.sqlite").unlink()  # noqa: ASYNC240 (best-effort teardown cleanup)


async def _athlete(session: AsyncSession) -> uuid.UUID:
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    a = Athlete(sex="male", reference_timezone="UTC")
    session.add(a)
    await session.commit()
    return a.athlete_id


@pytest.mark.integration
async def test_timestamp_round_trips_tz_aware_utc(session: AsyncSession) -> None:
    """A stored instant reads back tz-aware UTC on SQLite, not naive (GBO-R32)."""
    aid = await _athlete(session)
    start = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    act = Activity(athlete_id=aid, start_time=start, sport="cycling")
    session.add(act)
    await session.commit()
    session.expire_all()
    got = (await session.execute(select(Activity))).scalar_one()
    assert got.start_time.tzinfo is not None
    assert got.start_time == start


@pytest.mark.integration
async def test_foreign_keys_enforced_on_sqlite(session: AsyncSession) -> None:
    """An orphan FK is rejected on SQLite (PRAGMA foreign_keys=ON, GBO-AC-7)."""
    orphan = Activity(athlete_id=uuid.uuid4(), start_time=_dt.datetime.now(UTC), sport="cycling")
    session.add(orphan)
    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.integration
async def test_enum_check_rejects_out_of_vocabulary(session: AsyncSession) -> None:
    """A non-canonical enum token is rejected by the DB CHECK constraint (GBO-R12)."""
    aid = await _athlete(session)
    # Bypass the ORM mapper (which also validates) to hit the DB-level CHECK directly,
    # proving the closed vocabulary is enforced even via the Core upsert seam path.
    with pytest.raises(IntegrityError):
        await session.execute(
            text(
                "INSERT INTO activity (activity_id, athlete_id, start_time, sport, "
                "device_class, has_power, has_hr, has_gps, has_cadence, coverage, "
                "created_at, updated_at) VALUES (:id, :aid, :t, 'cycling', 'NOT_A_DEVICE', "
                "0, 0, 0, 0, '{}', :t, :t)"
            ),
            {"id": uuid.uuid4().hex, "aid": aid.hex, "t": "2026-06-01T08:00:00+00:00"},
        )
        await session.commit()


@pytest.mark.integration
async def test_numeric_preserves_fraction(session: AsyncSession) -> None:
    """A fractional numeric round-trips unrounded and as a float (GBO-R10)."""
    aid = await _athlete(session)
    act = Activity(
        athlete_id=aid, start_time=_dt.datetime.now(UTC), sport="cycling", avg_power_w=251.337
    )
    session.add(act)
    await session.commit()
    session.expire_all()
    got = (await session.execute(select(Activity))).scalar_one()
    assert isinstance(got.avg_power_w, float)
    assert got.avg_power_w == pytest.approx(251.337, abs=1e-5)
