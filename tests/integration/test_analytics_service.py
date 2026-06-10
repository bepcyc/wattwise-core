"""Integration tests for the canonical analytics service (B-E3-T6, ANL-R1, PMC, LM).

Seeds the canonical store via the ORM and asserts the service computes headline
analytics from real persisted records — the single consumer surface the API and
agent share. Runs on in-memory SQLite (the portable substrate, GBO-R8b).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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

UTC = _dt.UTC


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A session over a fresh in-memory schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed_constant_power_ride(
    session: AsyncSession, *, watts: float, seconds: int, ftp_w: float
) -> tuple[str, str]:
    """Seed one athlete + signature + a constant-power cycling activity."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    await session.flush()
    aid = athlete.athlete_id
    start = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    session.add(
        FitnessSignature(
            athlete_id=aid,
            signature_type="cycling",
            effective_date=_dt.date(2026, 1, 1),
            ftp_w=ftp_w,
            origin=SignatureOrigin.MEASURED,
        )
    )
    activity = Activity(
        athlete_id=aid,
        start_time=start,
        sport="cycling",
        elapsed_time_s=seconds,
        moving_time_s=seconds,
        avg_power_w=watts,
        has_power=True,
    )
    session.add(activity)
    await session.flush()
    stream_set = ActivityStreamSet(
        activity_id=activity.activity_id,
        sample_basis=SampleBasis.TIME,
        sample_rate_hz=1.0,
        sample_count=seconds,
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
            values=[watts] * seconds,
            coverage={},
        )
    )
    await session.commit()
    return str(aid), str(activity.activity_id)


@pytest.mark.integration
async def test_coggan_bundle_from_canonical_store(session: AsyncSession) -> None:
    """A 1-hour constant-FTP ride yields NP≈FTP, IF≈1, TSS≈100 from persisted streams."""
    _, activity_id = await _seed_constant_power_ride(
        session, watts=250.0, seconds=3600, ftp_w=250.0
    )
    svc = AnalyticsService(session)
    result = await svc.coggan(activity_id)
    assert is_computed(result)
    bundle = result.value  # type: ignore[union-attr]
    assert is_computed(bundle.np)
    assert bundle.np.value.np_w == pytest.approx(250.0, abs=1e-6)  # type: ignore[union-attr]
    assert is_computed(bundle.if_)
    assert bundle.if_.value == pytest.approx(1.0, abs=1e-6)  # type: ignore[union-attr]
    assert is_computed(bundle.tss)
    assert bundle.tss.value == pytest.approx(100.0, abs=1e-4)  # type: ignore[union-attr]


@pytest.mark.integration
async def test_coggan_unavailable_without_power(session: AsyncSession) -> None:
    """An unknown activity fails closed with a typed Unavailable, never a number (ANL-R4)."""
    svc = AnalyticsService(session)
    result = await svc.coggan(str(uuid.uuid4()))
    assert not is_computed(result)


@pytest.mark.integration
async def test_pmc_series_over_range(session: AsyncSession) -> None:
    """PMC produces a CTL/ATL/TSB entry for every day in the range (PMC-R6)."""
    aid, _ = await _seed_constant_power_ride(session, watts=250.0, seconds=3600, ftp_w=250.0)
    svc = AnalyticsService(session)
    series = await svc.pmc(aid, _dt.date(2026, 6, 1), _dt.date(2026, 6, 7))
    assert len(series) == 7
    # Day 1 had a ~100 TSS ride; CTL must rise above 0 on the activity day.
    first = series[0]
    assert is_computed(first)
    assert first.value.ctl > 0  # type: ignore[union-attr]


async def _set_current_sport(session: AsyncSession, athlete_id: str, sport: str | None) -> None:
    """Set the athlete's canonical current_sport (the ES power-component partition key)."""
    athlete = await session.get(Athlete, uuid.UUID(athlete_id))
    assert athlete is not None
    athlete.current_sport = sport
    await session.commit()


@pytest.mark.integration
async def test_endurance_score_composes_from_canonical_store(session: AsyncSession) -> None:
    """ES-R1/ES-R2: the score composes CTL + power-curve durability from persisted records.

    The seeded constant-power hour yields a computed CTL and a flat power curve, so
    the durability ratio MMP(1200)/MMP(300) == 1.0 (present); no HR channel exists,
    so decoupling is missing and the configured partial policy composes with reduced
    confidence and the components recorded in QualityReport (ES-R2b).
    """
    aid, _ = await _seed_constant_power_ride(session, watts=250.0, seconds=3600, ftp_w=250.0)
    await _set_current_sport(session, aid, "cycling")
    svc = AnalyticsService(session)
    result = await svc.endurance_score(aid, _dt.date(2026, 6, 1))
    assert is_computed(result)
    assert 0.0 <= result.value <= 100.0  # type: ignore[union-attr]
    quality = result.quality  # type: ignore[union-attr]
    assert quality.extra["components_present"] == ("ctl", "durability")
    assert quality.extra["components_missing"] == ("decoupling",)
    assert quality.confidence < 1.0
    assert result.provenance.sport == "cycling"  # type: ignore[union-attr]


@pytest.mark.integration
async def test_endurance_score_without_current_sport_degrades_to_ctl(
    session: AsyncSession,
) -> None:
    """ES-R2: no canonical current_sport ⇒ power components fail closed; CTL-only partial.

    The power components cannot be sport-partitioned without a sport (never a
    hardcoded one), so durability AND decoupling are missing and the declared-valid
    CTL-only subset composes with reduced confidence — never a silent 0 (ANL-R4).
    """
    aid, _ = await _seed_constant_power_ride(session, watts=250.0, seconds=3600, ftp_w=250.0)
    svc = AnalyticsService(session)
    result = await svc.endurance_score(aid, _dt.date(2026, 6, 1))
    assert is_computed(result)
    quality = result.quality  # type: ignore[union-attr]
    assert quality.extra["components_present"] == ("ctl",)
    assert quality.extra["components_missing"] == ("decoupling", "durability")
    assert quality.confidence < 1.0


@pytest.mark.integration
async def test_endurance_score_unknown_athlete_fails_closed(session: AsyncSession) -> None:
    """ES-R2(a): no athlete/PMC day ⇒ Unavailable(MISSING_REQUIRED_INPUT), never a number."""
    svc = AnalyticsService(session)
    result = await svc.endurance_score(str(uuid.uuid4()), _dt.date(2026, 6, 1))
    assert not is_computed(result)


async def _add_hr_channel(session: AsyncSession, activity_id: str, bpm: float) -> None:
    """Attach a constant-HR channel to the activity's existing stream set."""
    stmt = select(ActivityStreamSet).where(ActivityStreamSet.activity_id == uuid.UUID(activity_id))
    stream_set = (await session.execute(stmt)).scalar_one()
    session.add(
        StreamChannel(
            stream_set_id=stream_set.stream_set_id,
            set_kind=StreamSetKind.ACTIVITY,
            channel=StreamChannelName.HR_BPM,
            sample_basis=SampleBasis.TIME,
            values=[bpm] * stream_set.sample_count,
            coverage={},
        )
    )
    await session.commit()


@pytest.mark.integration
async def test_endurance_score_all_components_present(session: AsyncSession) -> None:
    """ES-R1: with power AND HR streams all three components compose at full confidence.

    The constant-power hour gives a flat curve (durability ratio 1.0) and, paired with
    a constant HR, a 0% aerobic decoupling — so no component is missing and the
    QualityReport carries full confidence (no ES-R2 degradation).
    """
    aid, activity_id = await _seed_constant_power_ride(
        session, watts=250.0, seconds=3600, ftp_w=250.0
    )
    await _add_hr_channel(session, activity_id, bpm=140.0)
    await _set_current_sport(session, aid, "cycling")
    svc = AnalyticsService(session)
    result = await svc.endurance_score(aid, _dt.date(2026, 6, 1))
    assert is_computed(result)
    quality = result.quality  # type: ignore[union-attr]
    assert quality.extra["components_present"] == ("ctl", "decoupling", "durability")
    assert quality.extra["components_missing"] == ()
    assert quality.confidence == 1.0


@pytest.mark.integration
async def test_endurance_score_short_ride_lacks_durability_point(
    session: AsyncSession,
) -> None:
    """ES-R2: a curve without the configured long-duration point ⇒ durability missing.

    A 600 s ride yields no MMP(1200 s) point, so the durability ratio fails closed
    (typed Unavailable from the curve-point read) and the partial policy composes on
    the remaining declared-valid subset — never a fabricated ratio.
    """
    aid, _ = await _seed_constant_power_ride(session, watts=250.0, seconds=600, ftp_w=250.0)
    await _set_current_sport(session, aid, "cycling")
    svc = AnalyticsService(session)
    result = await svc.endurance_score(aid, _dt.date(2026, 6, 1))
    assert is_computed(result)
    missing = result.quality.extra["components_missing"]  # type: ignore[union-attr]
    assert "durability" in missing
