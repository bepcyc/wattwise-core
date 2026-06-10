"""LOAD-R4 + COACH-R6 over the canonical store (file-SQLite + a real QueuePool).

- **LOAD-R4** — the athlete's stored ``default_training_load_model`` preference is
  actually CONSUMED by the engine: when the preferred model is ``hr_load_zonal`` but that
  variant is not applicable for the activity (no declared HR-zone boundaries+weights in
  the canonical store), the engine falls back to the automatic ``hr_load`` and RECORDS the
  substitution in provenance/quality (never fabricating the preferred model's inputs).
- **COACH-R6** — the gather resolvers consume the athlete's CURRENT sport for the
  sport-parameterized analytics (critical power / power curve) instead of a hardwired
  cycling default; with no current sport set the resolver records a typed coverage gap.

A file-SQLite DSN yields a real (non-Static) connection pool, exercising the canonical
store the API and agent share — never an in-memory StaticPool, never a host/live DB.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from wattwise_core.agent.capabilities import gather
from wattwise_core.agent.contracts import RetrievalRequest
from wattwise_core.analytics.result import Unavailable, UnavailableReason, is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.analytics.trimp import LOAD_MODEL_HR_LOAD, LOAD_MODEL_HR_LOAD_ZONAL
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
async def session(tmp_path: Path) -> AsyncIterator[AsyncSession]:
    """A session over a FILE-SQLite schema with a real (non-Static) connection pool.

    File-SQLite (not ``:memory:``) yields a real QueuePool, so this is a genuine
    multi-connection store rather than a single-connection StaticPool. WAL + a busy
    timeout keep concurrent reads/writes honest. The DB is a throwaway under ``tmp_path``,
    never a host/live database.
    """
    dsn = f"sqlite+aiosqlite:///{tmp_path}/canonical.sqlite"
    engine = create_async_engine(dsn, connect_args={"timeout": 30})
    # A real pool, never a StaticPool single connection (data-safety / race-honesty).
    assert not isinstance(engine.pool, StaticPool)
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.exec_driver_sql("PRAGMA busy_timeout=30000")
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed_hr_only_activity(
    session: AsyncSession,
    *,
    default_load_model: str | None,
    current_sport: str | None = "cycling",
) -> tuple[str, str]:
    """Seed one athlete (with a load-model default) + an HR-ONLY cycling activity."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    session.add(Sport(sport_code="running", display_name="Running", has_mechanical_power=False))
    athlete = Athlete(
        sex="male",
        reference_timezone="UTC",
        default_training_load_model=default_load_model,
        current_sport=current_sport,
    )
    session.add(athlete)
    await session.flush()
    aid = athlete.athlete_id
    start = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    session.add(
        FitnessSignature(
            athlete_id=aid,
            signature_type="cycling",
            effective_date=_dt.date(2026, 1, 1),
            max_hr_bpm=190,
            resting_hr_bpm=50,
            origin=SignatureOrigin.MEASURED,
        )
    )
    seconds = 600
    activity = Activity(
        athlete_id=aid,
        start_time=start,
        sport="cycling",
        elapsed_time_s=seconds,
        moving_time_s=seconds,
        avg_hr_bpm=140.0,
        has_power=False,
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
            channel=StreamChannelName.HR_BPM,
            sample_basis=SampleBasis.TIME,
            values=[140.0] * seconds,
            coverage={},
        )
    )
    await session.commit()
    return str(aid), str(activity.activity_id)


def _cp_fittable_power(seconds: int) -> list[float]:
    """A decaying 1 Hz power trace whose MMP curve fits a real CP/W' (CP-R1..R4).

    ``P(t) = CP + W'/(t+1)`` is monotonically decreasing, so the best d-second window is
    the leading one and the MMP curve closely follows the hyperbolic work-time model —
    yielding a ``Computed`` CP across the in-domain durations (verified ≈ R² 0.998). This
    is the contrast trace: if the resolver hardwired ``cycling`` it WOULD surface this
    real fit, so the running gate flipping it to ``Unavailable`` is load-bearing.
    """
    cp_w, w_prime_j = 240.0, 20000.0
    return [cp_w + w_prime_j / (t + 1.0) for t in range(seconds)]


async def _seed_running_pod_and_cycling_power(session: AsyncSession) -> str:
    """Seed a CURRENT-sport=running athlete with BOTH a running-pod-power activity and a
    cycling activity carrying a CP-fittable power curve, in the same date range.

    On the gated (running) path ``power_curve`` filters to the running activity, whose
    ``mmp(sport='running')`` is per-duration ``NOT_APPLICABLE_FOR_SPORT`` ⇒ an empty curve
    ⇒ CP ``Unavailable``. On the (mutation) un-gated cycling path the cycling activity's
    power curve fits a real ``Computed`` CP — so the assertions distinguish the two paths.
    """
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    session.add(Sport(sport_code="running", display_name="Running", has_mechanical_power=False))
    athlete = Athlete(
        sex="male",
        reference_timezone="UTC",
        default_training_load_model=None,
        current_sport="running",
    )
    session.add(athlete)
    await session.flush()
    aid = athlete.athlete_id

    async def _add_power_activity(sport_code: str, start: _dt.datetime, power: list[float]) -> None:
        activity = Activity(
            athlete_id=aid,
            start_time=start,
            sport=sport_code,
            elapsed_time_s=len(power),
            moving_time_s=len(power),
            has_power=True,
        )
        session.add(activity)
        await session.flush()
        stream_set = ActivityStreamSet(
            activity_id=activity.activity_id,
            sample_basis=SampleBasis.TIME,
            sample_rate_hz=1.0,
            sample_count=len(power),
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
                values=power,
                coverage={},
            )
        )

    secs = 1300  # spans the in-domain CP durations (60..1200 s)
    # A running-pod power channel exists, proving the gate fires on SPORT, not on data.
    await _add_power_activity(
        "running", _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC), _cp_fittable_power(secs)
    )
    await _add_power_activity(
        "cycling", _dt.datetime(2026, 6, 1, 12, 0, tzinfo=UTC), _cp_fittable_power(secs)
    )
    await session.commit()
    return str(aid)


async def test_default_zonal_preference_falls_back_to_hr_load_and_records_it(
    session: AsyncSession,
) -> None:
    """LOAD-R4: a stored hr_load_zonal default with no zone weights ⇒ hr_load + recorded.

    The preference is CONSUMED (not orphaned): because the zonal variant is not applicable
    (no declared HR-zone boundaries+weights in the canonical store), the engine falls back
    to the automatic Banister hr_load AND records the substitution in QualityReport.
    """
    _, activity_id = await _seed_hr_only_activity(
        session, default_load_model=LOAD_MODEL_HR_LOAD_ZONAL
    )
    svc = AnalyticsService(session)
    result = await svc.coggan(activity_id)
    assert is_computed(result)
    bundle = result.value
    assert bundle.load_model == LOAD_MODEL_HR_LOAD
    assert is_computed(bundle.hr_load)
    extra = bundle.hr_load.quality.extra
    assert extra.get("requested_load_model") == LOAD_MODEL_HR_LOAD_ZONAL
    assert extra.get("load_model_substituted") is True


async def test_no_preference_uses_automatic_hr_load_without_substitution(
    session: AsyncSession,
) -> None:
    """LOAD-R4: no stored preference ⇒ the automatic hr_load, no substitution flag."""
    _, activity_id = await _seed_hr_only_activity(session, default_load_model=None)
    svc = AnalyticsService(session)
    result = await svc.coggan(activity_id)
    assert is_computed(result)
    bundle = result.value
    assert bundle.load_model == LOAD_MODEL_HR_LOAD
    assert is_computed(bundle.hr_load)
    assert "load_model_substituted" not in bundle.hr_load.quality.extra


async def test_hr_only_activity_contributes_load_to_daily_series(
    session: AsyncSession,
) -> None:
    """LM-R2/LOAD-R4: an HR-only activity contributes its HR load to the daily series."""
    aid, _ = await _seed_hr_only_activity(session, default_load_model=None)
    svc = AnalyticsService(session)
    series = await svc.daily_load_series(aid, _dt.date(2026, 6, 1), _dt.date(2026, 6, 1))
    assert series[_dt.date(2026, 6, 1)] is not None
    assert series[_dt.date(2026, 6, 1)] > 0.0


async def test_resolver_threads_current_sport_into_critical_power(
    session: AsyncSession,
) -> None:
    """COACH-R6: the critical-power resolver consumes the athlete's CURRENT sport.

    Non-vacuity (mutation-mind): the athlete's CURRENT sport is ``running`` but carries a
    running-pod power channel AND a separate cycling activity whose power curve fits a real
    CP. The resolver threads the current sport, so it scopes CP to running — gated to an
    empty curve ⇒ ``Unavailable``. Deleting the ``current_sport`` thread and hardwiring
    ``cycling`` would instead consume the cycling activity's curve and return a ``Computed``
    CP, FAILING the ``not is_computed`` assertion. The contrast is proven directly: the
    cycling-scoped curve is non-empty (a Computed CP) while the running-scoped curve is
    empty — the gate fires on SPORT, not on absent data.
    """
    aid = await _seed_running_pod_and_cycling_power(session)
    svc = AnalyticsService(session)
    day = _dt.date(2026, 6, 1)
    rng = {"from_date": "2026-06-01", "to_date": "2026-06-01"}

    # Contrast (the mutation target): scoping to cycling DOES yield a Computed power curve
    # and a Computed CP — so a hardwired-cycling resolver would NOT abstain here.
    cycling_curve = await svc.power_curve(aid, day, day, sport="cycling")
    assert any(is_computed(r) for r in cycling_curve.values())
    cycling_cp = await svc.critical_power(aid, day, day, sport="cycling")
    assert is_computed(cycling_cp)
    # And running gates the same power data to an EMPTY curve (NOT_APPLICABLE per duration).
    running_curve = await svc.power_curve(aid, day, day, sport="running")
    assert not any(is_computed(r) for r in running_curve.values())

    # The resolver follows current_sport='running' ⇒ Unavailable, NOT the cycling fit.
    out = (await gather(svc, aid, [RetrievalRequest("critical_power", dict(rng))])).records
    cp = out["critical_power"]
    assert not is_computed(cp)
    assert isinstance(cp, Unavailable)
    # Non-vacuous (ANL-R12 / TEST-R3): a sport with no mechanical-power curve fails with
    # the typed sport-applicability reason — NEVER an INSUFFICIENT_DATA on a surrogate
    # channel. Pinning the reason makes the assertion fail if the sport gate is deleted
    # (the empty curve would then surface as INSUFFICIENT_DATA from the min-points check).
    assert cp.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


async def test_resolver_records_gap_when_no_current_sport(
    session: AsyncSession,
) -> None:
    """COACH-R6: with no current sport set, the resolver records a typed gap (fail-closed)."""
    aid, _ = await _seed_hr_only_activity(session, default_load_model=None, current_sport=None)
    svc = AnalyticsService(session)
    req = RetrievalRequest("power_curve", {"from_date": "2026-06-01", "to_date": "2026-06-01"})
    out = (await gather(svc, aid, [req])).records
    rec = out["power_curve"]
    assert isinstance(rec, dict)
    assert rec.get("available") is False
    assert rec.get("reason") == "no_current_sport"
