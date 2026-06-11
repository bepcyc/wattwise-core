"""Golden tests for the session-RPE last-resort load member (SRPE-R1, LOAD-R3, DEGR-R2).

The strength-session blind spot: a session with neither a power nor an HR channel
previously contributed NOTHING to the daily load / PMC — an invisible zero the coach
read as rest. With the athlete-reported exertion captured, the LOAD-R3 fallback now
resolves ``power_tss -> hr_load -> srpe_load`` and a power-less, HR-less session enters
the day honestly:

* the day stays ``Computed`` with the session-RPE value (golden: RPE 7 over one hour
  = (0.7)^2 * 100 = 49.0), badged ``Fidelity.SUBSTITUTED`` +
  ``substitution:{class:training_load, from_fidelity:raw_stream}`` at reduced
  confidence — never presented as power-TSS (DEGR-R2);
* an UNREPORTED session stays a surfaced ``None`` day — no default RPE, no zero-fill
  (ANL-R4);
* the priority is preserved: when the top (power) member exists, it wins and the day
  is NOT substituted, regardless of any RPE report (LOAD-R3).

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
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.analytics.constants import TRAINING_LOAD_CLASS
from wattwise_core.analytics.result import is_computed
from wattwise_core.analytics.service import AnalyticsService
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
_SECONDS = 3600
_RPE = 7.0
_SRPE_GOLDEN = 49.0  # (7/10)^2 * (3600/3600) * 100 — the declared RPE-as-intensity mapping
_DAY = _dt.date(2026, 6, 1)
_START = _dt.datetime.combine(_DAY, _dt.time(8, 0), tzinfo=UTC)


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + busy_timeout so file-sqlite concurrent access serializes, not lock-errors."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A throwaway file-SQLite engine with a REAL QueuePool (never :memory:/StaticPool)."""
    dsn = f"sqlite+aiosqlite:///{tmp_path}/srpe.sqlite"
    engine = create_async_engine(dsn, connect_args={"timeout": 30}, pool_size=2, max_overflow=2)
    event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


async def _seed_athlete(session: AsyncSession) -> Athlete:
    """Seed the sport registry rows + one athlete with a UTC reference timezone."""
    session.add(Sport(sport_code="strength", display_name="Strength", has_mechanical_power=False))
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    await session.flush()
    return athlete


async def _seed_strength_session(
    session: AsyncSession, *, perceived_exertion: float | None
) -> tuple[str, str]:
    """Seed one strength session with NO streams — only the athlete's exertion report."""
    athlete = await _seed_athlete(session)
    activity = Activity(
        athlete_id=athlete.athlete_id,
        start_time=_START,
        sport="strength",
        elapsed_time_s=_SECONDS,
        moving_time_s=_SECONDS,
        perceived_exertion=perceived_exertion,
    )
    session.add(activity)
    await session.commit()
    return str(athlete.athlete_id), str(activity.activity_id)


async def _seed_powered_ride_with_rpe(session: AsyncSession) -> tuple[str, str]:
    """Seed a cycling hour at FTP (power channel + signature) that ALSO carries an RPE."""
    athlete = await _seed_athlete(session)
    session.add(
        FitnessSignature(
            athlete_id=athlete.athlete_id,
            signature_type="cycling",
            effective_date=_dt.date(2026, 1, 1),
            ftp_w=_FTP_W,
            origin=SignatureOrigin.MEASURED,
        )
    )
    activity = Activity(
        athlete_id=athlete.athlete_id,
        start_time=_START,
        sport="cycling",
        elapsed_time_s=_SECONDS,
        moving_time_s=_SECONDS,
        avg_power_w=_FTP_W,
        has_power=True,
        perceived_exertion=_RPE,
    )
    session.add(activity)
    await session.flush()
    stream_set = ActivityStreamSet(
        activity_id=activity.activity_id,
        sample_basis=SampleBasis.TIME,
        sample_rate_hz=1.0,
        sample_count=_SECONDS,
        t0=_START,
    )
    session.add(stream_set)
    await session.flush()
    session.add(
        StreamChannel(
            stream_set_id=stream_set.stream_set_id,
            set_kind=StreamSetKind.ACTIVITY,
            channel=StreamChannelName.POWER_W,
            sample_basis=SampleBasis.TIME,
            values=[_FTP_W] * _SECONDS,
            coverage={},
        )
    )
    await session.commit()
    return str(athlete.athlete_id), str(activity.activity_id)


async def test_srpe_golden_value_and_label(factory: async_sessionmaker[AsyncSession]) -> None:
    """SRPE-R1 golden: RPE 7 over one hour reads 49.0 under the ``srpe_load`` label."""
    async with factory() as session:
        _, activity_id = await _seed_strength_session(session, perceived_exertion=_RPE)
    async with factory() as session:
        result = await AnalyticsService(session).srpe(activity_id)
    assert is_computed(result)
    assert result.value == pytest.approx(_SRPE_GOLDEN)
    assert result.quality.extra["load_model"] == "srpe_load"
    assert result.quality.extra["foster_au"] == pytest.approx(_RPE * 60.0)
    assert result.provenance.sport == "strength"


async def test_strength_session_enters_day_substituted(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """LOAD-R3/DEGR-R2: a power-less, HR-less session contributes its session-RPE load.

    The day must be ``Computed`` at the golden value and badged
    ``substitution:{class:training_load, from_fidelity:raw_stream}`` at reduced
    confidence — never the member's own tier, never full fidelity.
    """
    async with factory() as session:
        athlete_id, _ = await _seed_strength_session(session, perceived_exertion=_RPE)
    async with factory() as session:
        service = AnalyticsService(session)
        loads = await service.daily_load_series(athlete_id, _DAY, _DAY)
        series = await service.pmc(athlete_id, _DAY, _DAY)
    assert loads[_DAY] == pytest.approx(_SRPE_GOLDEN)
    assert len(series) == 1
    day = series[0]
    assert is_computed(day)
    cov = day.value.load_coverage
    assert cov is not None
    assert cov.fidelity is Fidelity.SUBSTITUTED
    assert cov.substitution is not None
    assert cov.substitution.equivalence_class == TRAINING_LOAD_CLASS
    assert cov.substitution.from_fidelity is Fidelity.RAW_STREAM
    assert day.quality.confidence < 1.0


async def test_unreported_session_stays_surfaced_none(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """ANL-R4: with no exertion report the day stays a surfaced None — never a default RPE."""
    async with factory() as session:
        athlete_id, activity_id = await _seed_strength_session(session, perceived_exertion=None)
    async with factory() as session:
        service = AnalyticsService(session)
        loads = await service.daily_load_series(athlete_id, _DAY, _DAY)
        result = await service.srpe(activity_id)
    assert loads[_DAY] is None
    assert not is_computed(result)


async def test_power_member_still_wins_over_reported_rpe(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """LOAD-R3 priority: the top (power-TSS) member wins; the RPE report never shadows it.

    An hour at FTP is the TSS=100 golden; the day must NOT carry a substitution badge.
    """
    async with factory() as session:
        athlete_id, _ = await _seed_powered_ride_with_rpe(session)
    async with factory() as session:
        service = AnalyticsService(session)
        loads = await service.daily_load_series(athlete_id, _DAY, _DAY)
        series = await service.pmc(athlete_id, _DAY, _DAY)
    assert loads[_DAY] == pytest.approx(100.0)
    day = series[0]
    assert is_computed(day)
    cov = day.value.load_coverage
    assert cov is not None
    assert cov.fidelity is not Fidelity.SUBSTITUTED
    assert cov.substitution is None


async def test_srpe_duration_falls_back_to_elapsed(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """SRPE-R1: with no moving time the session duration is the elapsed time (whole-session).

    Foster's method prices the whole session; ``moving_time_s`` is preferred when
    present, ``elapsed_time_s`` is the documented fallback — never a fabricated duration.
    """
    async with factory() as session:
        athlete = await _seed_athlete(session)
        activity = Activity(
            athlete_id=athlete.athlete_id,
            start_time=_START,
            sport="strength",
            elapsed_time_s=_SECONDS,
            moving_time_s=None,
            perceived_exertion=_RPE,
        )
        session.add(activity)
        await session.commit()
        activity_id = str(activity.activity_id)
    async with factory() as session:
        result = await AnalyticsService(session).srpe(activity_id)
    assert is_computed(result)
    assert result.value == pytest.approx(_SRPE_GOLDEN)
