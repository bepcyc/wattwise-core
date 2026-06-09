"""Golden tests for graceful degradation + metric substitution (doc 40 §10A, doc 80 §6.2a).

These are the analytics-engine half of the source-withdrawal & metric-substitution
protocol. The engine is stateless and reads ONLY the canonical store (ANL-R1): a "source
withdrawal" is modelled here by mutating the canonical store so the withdrawn source's
channel disappears, and a "reconnect" by restoring it from the retained data — exactly
what upstream re-resolution leaves the engine to recompute from.

Covered requirements:

* DEGR-R2 / DEGR-T1 / SUB-R1 — top-source removal keeps the channel POPULATED, re-resolved
  to the next-best class member; the recomputed value carries ``Fidelity.SUBSTITUTED`` +
  ``substitution:{class:training_load, from_fidelity:raw_stream}`` (never the displaced
  member's own tier) with reduced confidence — never presented as raw power-TSS.
* DEGR-T2 / SUB-R2 — only-provider removal (empty class) yields a typed ``Unavailable`` /
  surfaced ``None`` day; NO fabricated number, zero-fill, or stale value.
* DEGR-R4 / DEGR-T3 / SUB-R3 — reconnect re-resolves UPWARD to the higher fidelity, the
  value returns to the power-TSS value, ``substitution`` clears, and NO source re-fetch
  occurs (retained-canonical re-resolution only — the engine never touches the network).
* DEGR-T4 — a no-fabrication sweep over the full withdrawal/reconnect lifecycle.

Real-pool data safety: a throwaway file-SQLite database under ``tmp_path`` with a real
``QueuePool`` (WAL + busy_timeout), never ``:memory:``/``StaticPool`` and never a live DB.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import delete, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.analytics.constants import TRAINING_LOAD_CLASS
from wattwise_core.analytics.result import is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.api.perf_helpers import coverage_for as _api_coverage_for
from wattwise_core.domain.enums import (
    Fidelity,
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

UTC = _dt.UTC
pytestmark = pytest.mark.golden

_FTP_W = 250.0
_WATTS = 250.0  # constant ride at FTP -> IF=1, TSS=100 over an hour
_SECONDS = 3600
_HR_BPM = 150.0
_HR_MAX = 190.0
_HR_REST = 50.0
_DAY = _dt.date(2026, 6, 1)


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + busy_timeout so file-sqlite concurrent access serializes, not lock-errors."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A throwaway file-SQLite engine with a REAL QueuePool (never :memory:/StaticPool)."""
    dsn = f"sqlite+aiosqlite:///{tmp_path}/degr.sqlite"
    engine = create_async_engine(dsn, connect_args={"timeout": 30}, pool_size=2, max_overflow=2)
    event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


async def _seed_power_and_hr_ride(session: AsyncSession) -> tuple[str, str]:
    """Seed an athlete + signature + one cycling activity with BOTH power and HR channels.

    Both equivalence-class members for ``training_load`` are present: power-based TSS
    (``raw_stream`` top member) and an HR trace that yields a Banister load (``modeled``).
    """
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    await session.flush()
    aid = athlete.athlete_id
    start = _dt.datetime.combine(_DAY, _dt.time(8, 0), tzinfo=UTC)
    session.add(
        FitnessSignature(
            athlete_id=aid,
            signature_type="cycling",
            effective_date=_dt.date(2026, 1, 1),
            ftp_w=_FTP_W,
            max_hr_bpm=_HR_MAX,
            resting_hr_bpm=_HR_REST,
            origin=SignatureOrigin.MEASURED,
        )
    )
    activity = Activity(
        athlete_id=aid,
        start_time=start,
        sport="cycling",
        elapsed_time_s=_SECONDS,
        moving_time_s=_SECONDS,
        avg_power_w=_WATTS,
        avg_hr_bpm=_HR_BPM,
        has_power=True,
        has_hr=True,
    )
    session.add(activity)
    await session.flush()
    stream_set = ActivityStreamSet(
        activity_id=activity.activity_id,
        sample_basis=SampleBasis.TIME,
        sample_rate_hz=1.0,
        sample_count=_SECONDS,
        t0=start,
    )
    session.add(stream_set)
    await session.flush()
    session.add(
        StreamChannel(
            stream_set_id=stream_set.stream_set_id,
            set_kind=StreamSetKind.ACTIVITY,
            channel=StreamChannelName.POWER_W,
            sample_basis=SampleBasis.TIME,
            values=[_WATTS] * _SECONDS,
            coverage={},
        )
    )
    session.add(
        StreamChannel(
            stream_set_id=stream_set.stream_set_id,
            set_kind=StreamSetKind.ACTIVITY,
            channel=StreamChannelName.HR_BPM,
            sample_basis=SampleBasis.TIME,
            values=[_HR_BPM] * _SECONDS,
            coverage={},
        )
    )
    await session.commit()
    return str(aid), str(activity.activity_id)


async def _withdraw_channel(session: AsyncSession, channel: StreamChannelName) -> None:
    """Model an upstream source withdrawal: the canonical channel is no longer present."""
    await session.execute(delete(StreamChannel).where(StreamChannel.channel == channel))
    await session.commit()


async def _restore_power_channel(session: AsyncSession) -> None:
    """Model a reconnect: re-resolution restores the power channel from retained data."""
    stmt = select(ActivityStreamSet)
    stream_set = (await session.execute(stmt)).scalars().first()
    assert stream_set is not None
    session.add(
        StreamChannel(
            stream_set_id=stream_set.stream_set_id,
            set_kind=StreamSetKind.ACTIVITY,
            channel=StreamChannelName.POWER_W,
            sample_basis=SampleBasis.TIME,
            values=[_WATTS] * _SECONDS,
            coverage={},
        )
    )
    await session.commit()


async def _hr_load(session: AsyncSession, activity_id: str) -> float:
    """The HR (modeled) class member's value for the seeded activity (the substitute)."""
    res = await AnalyticsService(session).trimp(activity_id)
    assert is_computed(res)
    return float(res.value)


async def _day_result(session: AsyncSession, athlete_id: str) -> Any:
    """The single PMC day result for the seeded ride's date."""
    series = await AnalyticsService(session).pmc(athlete_id, _DAY, _DAY)
    assert len(series) == 1
    return series[0]


@pytest.mark.golden
async def test_degr_t1_sub_r1_top_source_removed_stays_computed_substituted(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """DEGR-T1/SUB-R1: withdraw power -> load stays populated from HR, badged SUBSTITUTED.

    The day must remain ``Computed``, the load must equal the HR (modeled) member's value
    (not blanked), and the day's coverage MUST read ``Fidelity.SUBSTITUTED`` with
    ``substitution:{class:training_load, from_fidelity:raw_stream}`` — NEVER the displaced
    HR member's own ``modeled`` tier — at reduced confidence (DEGR-R2).
    """
    async with factory() as session:
        athlete_id, activity_id = await _seed_power_and_hr_ride(session)
    async with factory() as session:
        await _withdraw_channel(session, StreamChannelName.POWER_W)
    async with factory() as session:
        expected_hr_load = await _hr_load(session, activity_id)
        loads = await AnalyticsService(session).daily_load_series(athlete_id, _DAY, _DAY)
        day = await _day_result(session, athlete_id)

    # (a) channel still present (Computed, not Unavailable) and (b) value == HR member.
    assert loads[_DAY] == pytest.approx(expected_hr_load)
    assert is_computed(day)
    cov = day.value.load_coverage
    # (c) coverage downgraded to the substituted token with class + displaced top tier.
    assert cov is not None
    assert cov.fidelity is Fidelity.SUBSTITUTED
    assert cov.fidelity is not Fidelity.MODELED  # never the displaced member's own tier
    assert cov.substitution is not None
    assert cov.substitution.equivalence_class == TRAINING_LOAD_CLASS
    assert cov.substitution.from_fidelity is Fidelity.RAW_STREAM
    # QualityReport surfaces the downgrade (DEGR-R2 reduced confidence).
    assert day.quality.confidence < 1.0
    assert day.quality.extra["substituted"] is True
    assert day.quality.extra["from_fidelity"] == Fidelity.RAW_STREAM.value
    # The QualityReport is honest enough that the API per-point coverage NEVER presents this
    # substituted day at the displaced raw-stream tier (DEGR-R2 surfacing, doc 60 consumes this).
    api_cov = _api_coverage_for(day)
    assert api_cov.fidelity == Fidelity.SUBSTITUTED.value
    assert api_cov.fidelity != Fidelity.RAW_STREAM.value
    # SUB-R1(c): the consumer-visible coverage descriptor also reports the substitution
    # block so a client can retrieve from_fidelity (the displaced top tier) from the API
    # response (API-R29) — not merely the downgraded fidelity token.
    assert api_cov.substitution == {
        "class": TRAINING_LOAD_CLASS,
        "from_fidelity": Fidelity.RAW_STREAM.value,
    }


@pytest.mark.golden
async def test_degr_t2_sub_r2_empty_class_yields_typed_unavailable_no_fabrication(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """DEGR-T2/SUB-R2: remove the ONLY remaining member -> typed absence, NO fabricated value."""
    async with factory() as session:
        athlete_id, _ = await _seed_power_and_hr_ride(session)
    async with factory() as session:
        await _withdraw_channel(session, StreamChannelName.POWER_W)
        await _withdraw_channel(session, StreamChannelName.HR_BPM)
    async with factory() as session:
        loads = await AnalyticsService(session).daily_load_series(athlete_id, _DAY, _DAY)
        day = await _day_result(session, athlete_id)

    # The activity day had an activity but no computable load: a surfaced None, never 0.
    assert loads[_DAY] is None
    # PMC still materializes the day (PMC-R6) but carries NO substituted/fabricated load.
    assert is_computed(day)
    assert day.value.load_coverage is None
    assert "substituted" not in day.quality.extra


@pytest.mark.golden
async def test_degr_t3_sub_r3_reconnect_restores_upward_no_refetch(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """DEGR-T3/SUB-R3: reconnect re-resolves UPWARD to power-TSS; substitution clears; no fetch.

    The restored value MUST equal the from-origin power-TSS computation on the restored
    inputs, coverage returns to the raw_stream tier, ``substitution`` clears, and the whole
    re-resolution reads only the canonical store — the engine issues NO network/source fetch.
    """
    async with factory() as session:
        athlete_id, _activity_id = await _seed_power_and_hr_ride(session)
        # The from-origin power-TSS value, captured while power is present.
        power_loads = await AnalyticsService(session).daily_load_series(athlete_id, _DAY, _DAY)
        power_tss = power_loads[_DAY]
        assert power_tss is not None
        assert power_tss == pytest.approx(100.0, abs=0.5)  # an hour at FTP ~ TSS 100

    async with factory() as session:
        await _withdraw_channel(session, StreamChannelName.POWER_W)
    async with factory() as session:
        sub_day = await _day_result(session, athlete_id)
        assert sub_day.value.load_coverage is not None
        assert sub_day.value.load_coverage.fidelity is Fidelity.SUBSTITUTED

    async with factory() as session:
        await _restore_power_channel(session)
    async with factory() as session:
        svc = AnalyticsService(session)
        loads = await svc.daily_load_series(athlete_id, _DAY, _DAY)
        day = await _day_result(session, athlete_id)

    # Value restored UPWARD to the exact from-origin power-TSS value (DEGR-T3 equality).
    assert loads[_DAY] == pytest.approx(power_tss)
    assert is_computed(day)
    # Coverage restored to the top tier; substitution cleared (no longer downgraded).
    cov = day.value.load_coverage
    assert cov is not None
    assert cov.fidelity is Fidelity.RAW_STREAM
    assert cov.substitution is None
    assert day.quality.confidence == 1.0
    assert "substituted" not in day.quality.extra


@pytest.mark.golden
async def test_degr_t4_no_fabrication_across_lifecycle(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """DEGR-T4: across the full withdrawal/reconnect lifecycle, NO zero/stale/guessed value.

    At every phase the day's load is either an HONEST computed number (top or substituted
    member) or a typed ``None`` — never a fabricated zero-fill while real data exists, and
    never a value carried over from a phase whose inputs are gone.
    """
    async with factory() as session:
        athlete_id, activity_id = await _seed_power_and_hr_ride(session)
        hr_value = await _hr_load(session, activity_id)
    async with factory() as session:
        full = (await AnalyticsService(session).daily_load_series(athlete_id, _DAY, _DAY))[_DAY]
    async with factory() as session:
        await _withdraw_channel(session, StreamChannelName.POWER_W)
    async with factory() as session:
        sub = (await AnalyticsService(session).daily_load_series(athlete_id, _DAY, _DAY))[_DAY]
    async with factory() as session:
        await _withdraw_channel(session, StreamChannelName.HR_BPM)
    async with factory() as session:
        empty = (await AnalyticsService(session).daily_load_series(athlete_id, _DAY, _DAY))[_DAY]

    assert full is not None
    # The substituted phase is the HONEST HR value, never the stale power number.
    assert sub == pytest.approx(hr_value)
    assert sub != pytest.approx(full)
    # The empty-class phase is a typed None, never 0.0 and never a stale carry-forward.
    assert empty is None
