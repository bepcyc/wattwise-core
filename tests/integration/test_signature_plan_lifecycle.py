"""GBO-R27/R28/R29/R31 prescription + signature lifecycle proof suite.

Proves against a real schema that: the signature write seam closes the prior open
interval and refuses overlapping/out-of-order writes (GBO-R27); resolution honors
``effective_to`` and refuses a modeled signature below the stated fit-quality floor
(GBO-R28, fail-closed); the workout step schema is enforced on EVERY ORM write
(GBO-R29/R29a); and a generated plan lands ONLY with reproducible lineage and is
immutable afterwards — changes go through schedule adjustments (GBO-R31).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.domain.enums import PlanDayIntent, PlanStatus, SignatureOrigin
from wattwise_core.domain.workout_steps import WorkoutStepError
from wattwise_core.persistence.models import (
    Athlete,
    Base,
    FitnessSignature,
    PlanDay,
    Sport,
    Workout,
)
from wattwise_core.persistence.planning_writes import (
    PlanDaySpec,
    PlanLineageError,
    create_plan,
)
from wattwise_core.persistence.signatures import SignatureIntervalError, record_signature

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A session over a fresh canonical schema (offline, no network)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed(session: AsyncSession) -> uuid.UUID:
    """Seed the athlete + cycling sport registry row."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    await session.flush()
    return athlete.athlete_id


async def test_record_signature_closes_prior_open_interval(session: AsyncSession) -> None:
    """GBO-R27: recording a new signature CLOSES the prior open interval (never
    overwrites it) and leaves exactly one open row per scope; resolution honors the
    closed interval so the OLD thresholds resolve for an as-of inside it."""
    athlete = await _seed(session)
    first = await record_signature(
        session, athlete_id=athlete, signature_type="cycling",
        effective_date=_dt.date(2026, 1, 1), origin=SignatureOrigin.USER_ENTERED,
        ftp_w=250.0,
    )
    await record_signature(
        session, athlete_id=athlete, signature_type="cycling",
        effective_date=_dt.date(2026, 6, 1), origin=SignatureOrigin.USER_ENTERED,
        ftp_w=270.0,
    )
    await session.refresh(first)
    assert first.effective_to is not None  # closed, not overwritten
    svc = AnalyticsService(session)
    past = await svc.resolve_signature(str(athlete), "cycling", _dt.date(2026, 3, 1))
    assert past.ftp_w == 250.0  # the closed interval still owns its as-of range
    now = await svc.resolve_signature(str(athlete), "cycling", _dt.date(2026, 6, 2))
    assert now.ftp_w == 270.0  # the open interval owns the present


async def test_out_of_order_signature_write_is_refused(session: AsyncSession) -> None:
    """GBO-R27: a signature dated at-or-before an existing interval in its scope would
    overlap — the write seam refuses it loudly instead of silently reordering."""
    athlete = await _seed(session)
    await record_signature(
        session, athlete_id=athlete, signature_type="cycling",
        effective_date=_dt.date(2026, 6, 1), origin=SignatureOrigin.USER_ENTERED,
        ftp_w=270.0,
    )
    with pytest.raises(SignatureIntervalError):
        await record_signature(
            session, athlete_id=athlete, signature_type="cycling",
            effective_date=_dt.date(2026, 3, 1), origin=SignatureOrigin.USER_ENTERED,
            ftp_w=260.0,
        )


async def test_modeled_signature_below_fit_floor_is_refused(session: AsyncSession) -> None:
    """GBO-R28: resolution REFUSES a modeled signature whose stored fit quality sits
    below the configured floor — empty params (a typed gap) instead of thresholds from
    a bad fit; a modeled signature with a good fit resolves normally."""
    athlete = await _seed(session)
    session.add(
        FitnessSignature(
            athlete_id=athlete, signature_type="cycling",
            effective_date=_dt.date(2026, 1, 1), origin=SignatureOrigin.MODELED,
            cp_w=260.0, fit_quality={"r_squared": 0.41, "n_points": 3},
        )
    )
    await session.flush()
    svc = AnalyticsService(session)
    refused = await svc.resolve_signature(str(athlete), "cycling", _dt.date(2026, 2, 1))
    assert refused.cp_w is None  # fail-closed: the bad fit never becomes a threshold
    sig = (await session.execute(select(FitnessSignature))).scalar_one()
    sig.fit_quality = {"r_squared": 0.97, "n_points": 6}
    await session.flush()
    accepted = await svc.resolve_signature(str(athlete), "cycling", _dt.date(2026, 2, 1))
    assert accepted.cp_w == 260.0


async def test_modeled_signature_without_fit_quality_is_refused_at_write(
    session: AsyncSession,
) -> None:
    """GBO-R28: the write seam refuses a MODELED signature carrying no fit_quality —
    one the analytics layer could never fit-gate must not exist."""
    athlete = await _seed(session)
    with pytest.raises(SignatureIntervalError):
        await record_signature(
            session, athlete_id=athlete, signature_type="cycling",
            effective_date=_dt.date(2026, 1, 1), origin=SignatureOrigin.MODELED,
            cp_w=260.0,
        )


async def test_workout_step_schema_enforced_on_orm_write(session: AsyncSession) -> None:
    """GBO-R29/R29a: an untyped/invalid step array refuses the ORM write; a
    schema-conforming step array lands."""
    athlete = await _seed(session)
    with pytest.raises(WorkoutStepError):
        Workout(
            athlete_id=athlete, name="bad", sport="cycling",
            steps=[{"intent": "work", "duration_s": 600, "source_field": "x"}],
        )
    workout = Workout(
        athlete_id=athlete, name="good", sport="cycling",
        steps=[{
            "target_type": "power_pct_cp", "intent": "work",
            "target_low": 88.0, "target_high": 94.0, "duration_s": 1200,
        }],
    )
    session.add(workout)
    await session.flush()
    assert workout.workout_id is not None


async def test_plan_requires_lineage_and_is_immutable_after_generation(
    session: AsyncSession,
) -> None:
    """GBO-R31: a plan without reproducible lineage is refused; a generated plan_day
    refuses in-place mutation (changes must be schedule adjustments); the plan's
    lifecycle status MAY still transition."""
    athlete = await _seed(session)
    days = [PlanDaySpec(plan_date=_dt.date(2026, 7, 1), intent=PlanDayIntent.EASY)]
    with pytest.raises(PlanLineageError):
        await create_plan(
            session, athlete_id=athlete, start_date=_dt.date(2026, 7, 1),
            end_date=_dt.date(2026, 7, 7), days=days, lineage={"engine_version": "1.2.3"},
        )
    plan = await create_plan(
        session, athlete_id=athlete, start_date=_dt.date(2026, 7, 1),
        end_date=_dt.date(2026, 7, 7), days=days,
        lineage={"engine_version": "1.2.3", "input_snapshot_ids": {"signature": "sig-1"}},
    )
    day = (
        await session.execute(select(PlanDay).where(PlanDay.plan_id == plan.plan_id))
    ).scalar_one()
    day.intent = PlanDayIntent.HARD
    with pytest.raises(ValueError, match="immutable"):
        await session.flush()
    await session.rollback()
    plan = await session.merge(plan)
    plan.status = PlanStatus.COMPLETED  # the lifecycle transition stays allowed
    await session.flush()
