"""Analytics day-attribution buckets on athlete local_date, not UTC date — GBO-R33/R35.

The canonical analytics day-buckets (the daily-load series feeding PMC/CTL/ATL, and the
date range window ``_day_bounds``) MUST attribute an activity to the athlete's LOCAL
calendar day, not ``start_time.date()`` (the UTC date). The decisive fixture is a non-UTC
athlete with an activity whose UTC instant lands on a DIFFERENT local calendar day than
the UTC day: its load must land on the LOCAL day's bucket. A mutation reverting the bucket
to ``start_time.date()`` re-attributes the load to the UTC day and breaks these tests.

Buckets read the persisted ``activity.local_date`` (assigned at ingest, GBO-R34
reproducible); when absent it is recomputed from the UTC instant + the athlete's as-of
reference timezone, so the same instant always reproduces the same bucket.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.analytics._service_loaders import _day_bounds_for_tz
from wattwise_core.analytics.result import is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.domain.enums import (
    SampleBasis,
    SignatureOrigin,
    StreamChannelName,
    StreamSetKind,
)
from wattwise_core.persistence.models import (
    Activity,
    ActivityStreamSet,
    Athlete,
    Base,
    FitnessSignature,
    Sport,
    StreamChannel,
)

pytestmark = pytest.mark.integration

UTC = _dt.UTC


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A session over a fresh in-memory schema (no concurrency here; pure read math)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed_ny_athlete(session: AsyncSession) -> uuid.UUID:
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="America/New_York")
    session.add(athlete)
    await session.flush()
    session.add(
        FitnessSignature(
            athlete_id=athlete.athlete_id,
            signature_type="cycling",
            effective_date=_dt.date(2026, 1, 1),
            ftp_w=250.0,
            origin=SignatureOrigin.MEASURED,
        )
    )
    await session.flush()
    return athlete.athlete_id


async def _add_ride(
    session: AsyncSession, aid: uuid.UUID, instant: _dt.datetime, local_date: _dt.date
) -> None:
    """Add a 1-hour constant-FTP power ride carrying the ingest-projected local_date.

    A real power stream (constant FTP for an hour ≈ 100 TSS) is what makes the daily-load /
    PMC paths compute a non-zero load — so the assertion that the load lands on the LOCAL day
    is non-vacuous (an empty-stream activity would be load-None regardless of bucketing).
    """
    seconds = 3600
    activity = Activity(
        athlete_id=aid,
        start_time=instant,
        start_time_local=instant.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=UTC),
        local_date=local_date,
        sport="cycling",
        elapsed_time_s=seconds,
        moving_time_s=seconds,
        avg_power_w=250.0,
        has_power=True,
    )
    session.add(activity)
    await session.flush()
    stream_set = ActivityStreamSet(
        activity_id=activity.activity_id,
        sample_basis=SampleBasis.TIME,
        sample_rate_hz=1.0,
        sample_count=seconds,
        t0=instant,
    )
    session.add(stream_set)
    await session.flush()
    session.add(
        StreamChannel(
            stream_set_id=stream_set.stream_set_id,
            set_kind=StreamSetKind.ACTIVITY,
            channel=StreamChannelName.POWER_W,
            sample_basis=SampleBasis.TIME,
            values=[250.0] * seconds,
            coverage={},
        )
    )
    await session.flush()


async def test_daily_load_lands_on_local_day_not_utc_day(session: AsyncSession) -> None:
    """A late-UTC ride buckets its load to the LOCAL day (GBO-R35), not the UTC day.

    2026-06-02 03:00Z for an America/New_York athlete is 2026-06-01 23:00 local. The day's
    summed load must appear on 2026-06-01, and 2026-06-02 must be a real 0 rest day. The
    UTC-bucket mutation would put the load on 2026-06-02 and make 06-01 the rest day.
    """
    aid = await _seed_ny_athlete(session)
    await _add_ride(session, aid, _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC), _dt.date(2026, 6, 1))
    svc = AnalyticsService(session)
    loads = await svc.daily_load_series(str(aid), _dt.date(2026, 6, 1), _dt.date(2026, 6, 2))
    assert loads[_dt.date(2026, 6, 1)] is not None  # load landed on the LOCAL day
    assert loads[_dt.date(2026, 6, 1)] > 0
    assert loads[_dt.date(2026, 6, 2)] == 0.0  # the UTC day is a real rest day


async def test_day_bounds_window_includes_local_day_activity(session: AsyncSession) -> None:
    """The local-date range window includes an activity whose UTC date is outside it (GBO-R35).

    A query for the single LOCAL day 2026-06-01 must include the 2026-06-02 03:00Z ride
    (local 2026-06-01 23:00) — a UTC-midnight window would exclude it.
    """
    aid = await _seed_ny_athlete(session)
    await _add_ride(session, aid, _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC), _dt.date(2026, 6, 1))
    svc = AnalyticsService(session)
    acts = await svc._activities_in_range(str(aid), _dt.date(2026, 6, 1), _dt.date(2026, 6, 1))
    assert len(acts) == 1  # the local-06-01 ride is in the 06-01 window despite UTC 06-02


def test_day_bounds_for_tz_uses_local_midnight() -> None:
    """``_day_bounds`` resolves local-midnight→UTC instants for the tz (GBO-R35 window)."""
    lo, hi = _day_bounds_for_tz(_dt.date(2026, 6, 1), _dt.date(2026, 6, 1), "America/New_York")
    # Local 2026-06-01 00:00 EDT (-4) = 2026-06-01 04:00Z; the next local midnight = 06-02 04:00Z.
    assert lo == _dt.datetime(2026, 6, 1, 4, 0, tzinfo=UTC)
    assert hi == _dt.datetime(2026, 6, 2, 4, 0, tzinfo=UTC)


async def test_pmc_attributes_load_to_local_day(session: AsyncSession) -> None:
    """The PMC series rises on the LOCAL day the load was earned (GBO-R35 + PMC-R1)."""
    aid = await _seed_ny_athlete(session)
    await _add_ride(session, aid, _dt.datetime(2026, 6, 2, 3, 0, tzinfo=UTC), _dt.date(2026, 6, 1))
    svc = AnalyticsService(session)
    series = await svc.pmc(str(aid), _dt.date(2026, 6, 1), _dt.date(2026, 6, 2))
    day0 = series[0]
    assert is_computed(day0)
    # CTL is the EWMA impulse on the local-06-01 day; a UTC-bucket mutation would put the
    # impulse on 06-02 instead, leaving 06-01's CTL at ~0.
    assert day0.value.ctl > 0.0
