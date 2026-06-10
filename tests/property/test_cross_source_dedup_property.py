"""Property: adding duplicate overlapping sources never changes the counted load.

Companion to ``tests/golden/test_cross_source_single_count.py`` and the doc 80
DEDUP-R1 / DEDUP-R4 single-count proof. Where the golden pins exact numbers for a
fixed ride, this asserts the *invariant* over generated inputs:

    For a generated base ride and a generated number ``k`` (1..4) of DUPLICATE
    overlapping sources for that SAME ride, the canonical activity COUNT is 1 and
    the day's aggregate totals (PMC CTL/ATL/TSB, the daily load roll-up, the
    per-activity TSS/NP) are INVARIANT to ``k`` — equal to the single-source
    (``k == 1``) baseline. Extra duplicate sources change coverage/fidelity, NEVER
    the counted load.

All reads are over RESOLVED CANONICAL entities (DEDUP-R4): the COUNT is read from
the ``Activity`` table and every total is read through :class:`AnalyticsService`,
which reads resolved canonical activities only. Summing the ``SourceCandidate``
rows would multiply the load by ``k`` and fail this property — which is the point.

Determinism / cost (ANL-R30, TEST-R2)
-------------------------------------
Each example does real DB round-trips (a fresh in-memory SQLite store, ``k``
ingests, then an analytics read), so ``max_examples`` is small and an explicit
``deadline`` is set. The async ingest/analytics flow is driven from the sync
hypothesis body via ``asyncio.run`` over a brand-new engine per example, so every
example is fully isolated and reproducible.
"""

from __future__ import annotations

import asyncio
import datetime as _dt

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.analytics.result import is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, SignatureOrigin
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.persistence.models import (
    Activity,
    ActivityStreamSet,
    Athlete,
    Base,
    FitnessSignature,
    SourceDescriptor,
    Sport,
    StreamChannel,
)
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.property

UTC = _dt.UTC

_RIDE_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
_RIDE_DAY = _dt.date(2026, 6, 1)
_TOL = 1e-9

# The TRUTH source carries the highest fidelity (raw_stream); it MUST win every
# contested field. Every duplicate but the FIRST is an identical truth copy.
_TRUTH_TIER = Fidelity.RAW_STREAM
# The FIRST duplicate is a DECOY at a strictly-LOWER trust tier carrying a DISTINCT
# (wrong) value. resolve_field must ACTIVELY reject it on trust rank; if it leaked
# through, avg_power_w/the power stream — and therefore TSS/NP/load — would move.
#
# Crucially the decoy is created FIRST, so (PKs are UUIDv7 minted in creation order,
# time-ordered + monotonic) it gets the SMALLEST source_descriptor_id and therefore
# the FAVORABLE step-5 stable tiebreak in dedup._sort_key
# ``(trust_rank, -confidence, -recency, -completeness, source_descriptor_id)``. So the
# high-trust truth source (created last, largest id) LOSES the id tiebreak and can win
# the contested field ONLY via step-1 trust rank — making this a genuine trust-resolution
# assertion (CONF-R2), not a tautology over value-identical copies and not an artifact of
# the id ordering (the orthogonal trust order is also pinned by the golden and
# tests/integration/test_ingestion_findings.py).
_DECOY_TIER = Fidelity.SUMMARY_ONLY


def _decoy_watts(watts: float) -> float:
    """A power clearly DISTINCT from ``watts`` (≥100 W apart, still a plausible ride).

    Different enough that, were the decoy ever to win avg_power_w or the power
    stream, the resolved TSS/NP/daily-load would visibly diverge from the baseline.
    """
    return watts + 150.0 if watts <= 240.0 else watts - 150.0


def _ride_candidate(
    *, native_id: str, watts: float, seconds: int, tier: Fidelity = _TRUTH_TIER
) -> GboCandidate:
    """A constant-power cycling candidate; same start/sport/duration ⇒ same identity.

    ``watts`` drives BOTH the scalar ``avg_power_w`` and every sample of the
    ``power_w`` stream, so a wrongly-chosen value would move the aggregate either
    through the scalar field or through the per-channel stream resolution.
    """
    payload = {
        "start_time": _RIDE_START,
        "sport": "cycling",
        "elapsed_time_s": seconds,
        "moving_time_s": seconds,
        "avg_power_w": watts,
        "streams": {
            "power_w": {"values": [watts] * seconds, "sample_basis": "time", "sample_rate_hz": 1.0}
        },
    }
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(f"{native_id}:{watts}:{seconds}".encode()),
        payload=payload,
        trust_tier=tier,
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )


async def _seed(session: AsyncSession, *, n_sources: int) -> tuple[str, list[str]]:
    """Seed the athlete, the cycling sport, an FTP signature, and ``n_sources`` sources."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    descriptors = [
        SourceDescriptor(source_key=f"src_{i}", display_name=f"S{i}", kind="oauth_api")
        for i in range(n_sources)
    ]
    session.add_all(descriptors)
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
    await session.commit()
    return str(athlete.athlete_id), [str(d.source_descriptor_id) for d in descriptors]


async def _run_one(
    watts: float, seconds: int, k: int
) -> tuple[int, dict[str, float], dict[str, float]]:
    """Ingest the ride from ``k`` sources; return (count, rollups, resolved_power).

    A fresh in-memory store per call (full isolation/determinism). The rollups are
    read over RESOLVED CANONICAL entities only (DEDUP-R4) — never a candidate sum.
    ``resolved_power`` is the directly-read canonical contested value (the resolved
    ``Activity.avg_power_w`` scalar and a sample of the resolved ``power_w`` stream),
    so a trust inversion that let the lower-trust decoy through is caught directly —
    not only via the downstream load-invariant.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with factory() as session:
            athlete_id, descriptor_ids = await _seed(session, n_sources=k)
            ingest = IngestService(session)
            # k overlapping observations of ONE real ride (distinct native ids, same
            # start/sport/duration ⇒ identity resolution collapses them to one).
            # With k >= 2 the FIRST observation is a lower-trust DECOY carrying a
            # distinct (wrong) power: the resolver must reject it on trust rank, so
            # the canonical fields stay equal to the high-trust truth sources.
            # The decoy is FIRST on purpose: its descriptor (created first) gets the
            # SMALLEST source_descriptor_id, so it WINS the step-5 ascending-id stable
            # tiebreak in _sort_key — the high-trust truth (created later, larger ids)
            # can therefore beat it ONLY on trust rank, never on the id tiebreak. With
            # k == 1 there is no decoy, so the lone source is always the high-trust truth.
            for i, descriptor_id in enumerate(descriptor_ids):
                is_decoy = k > 1 and i == 0
                cand = _ride_candidate(
                    native_id=f"obs-{i}",
                    watts=_decoy_watts(watts) if is_decoy else watts,
                    seconds=seconds,
                    tier=_DECOY_TIER if is_decoy else _TRUTH_TIER,
                )
                await ingest.ingest(athlete_id, descriptor_id, [cand])
            await session.commit()

            count = len((await session.execute(select(Activity))).scalars().all())
            rollups = await _rollups(session, athlete_id)
            power = await _resolved_power(session)
            return count, rollups, power
    finally:
        await engine.dispose()


async def _resolved_power(session: AsyncSession) -> dict[str, float]:
    """The RESOLVED canonical contested value: the ``avg_power_w`` scalar + a stream sample.

    Reads the single resolved canonical ``Activity`` and a sample of its resolved
    ``power_w`` ``StreamChannel`` (via the activity's ``ActivityStreamSet``). Mirrors the
    query idiom in ``tests/integration/test_ingestion_findings.py`` so a trust inversion
    that let the lower-trust decoy value win EITHER the scalar field or the per-channel
    stream is caught directly, ordering-independently — not only through the load roll-up.
    """
    act = (await session.execute(select(Activity))).scalars().one()
    ss = (await session.execute(select(ActivityStreamSet))).scalars().one()
    power = (
        (
            await session.execute(
                select(StreamChannel).where(
                    StreamChannel.stream_set_id == ss.stream_set_id,
                    StreamChannel.channel == "power_w",
                )
            )
        )
        .scalars()
        .one()
    )
    return {"avg_power_w": float(act.avg_power_w), "stream_sample": float(power.values[0])}


async def _rollups(session: AsyncSession, athlete_id: str) -> dict[str, float]:
    """The day's aggregate totals, read through AnalyticsService (resolved canonical)."""
    svc = AnalyticsService(session)
    activities = await svc._activities_in_range(athlete_id, _RIDE_DAY, _RIDE_DAY)
    assert len(activities) == 1
    activity_id = str(activities[0].activity_id)

    bundle = await svc.coggan(activity_id)
    assert is_computed(bundle)
    assert is_computed(bundle.value.tss) and is_computed(bundle.value.np)

    daily = await svc.daily_load_series(athlete_id, _RIDE_DAY, _RIDE_DAY)
    pmc = await svc.pmc(athlete_id, _RIDE_DAY, _RIDE_DAY)
    assert is_computed(pmc[0])

    load = daily[_RIDE_DAY]
    assert load is not None
    return {
        "tss": bundle.value.tss.value,
        "np_w": bundle.value.np.value.np_w,
        "daily_load": load,
        "ctl": pmc[0].value.ctl,
        "atl": pmc[0].value.atl,
        "tsb": pmc[0].value.tsb,
    }


def _assert_close(baseline: dict[str, float], other: dict[str, float]) -> None:
    assert baseline.keys() == other.keys()
    for key, b in baseline.items():
        assert other[key] == pytest.approx(b, abs=_TOL * max(1.0, abs(b))), key


@settings(
    max_examples=15,
    deadline=_dt.timedelta(seconds=20),  # explicit: DB round-trips per example are slow
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    # A real, computable ride: enough seconds for NP to seed (>=30) and positive power.
    watts=st.floats(min_value=80.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    seconds=st.integers(min_value=60, max_value=1200),
    k=st.integers(min_value=2, max_value=4),  # 2..4 DUPLICATE overlapping sources
)
def test_duplicate_sources_do_not_change_any_aggregate(watts: float, seconds: int, k: int) -> None:
    """Adding ``k`` duplicate sources keeps COUNT == 1 and every total == baseline.

    The DEDUP-R1 property: a duplicate source for an existing activity changes no
    aggregate total. The single-source (``k == 1``) run is the oracle; the
    ``k``-source run must match it exactly (within the analytics tolerance) AND
    resolve to a single canonical activity (DEDUP-R4 read path).

    Crucially the ``k``-source run includes a LOWER-trust decoy whose power differs
    from the truth by ≥100 W, and the decoy is created FIRST so it holds the
    SMALLEST ``source_descriptor_id`` — i.e. it is handed the FAVORABLE step-5
    ascending-id stable tiebreak in ``dedup._sort_key``. The high-trust truth source
    (created last, larger id) therefore LOSES that tiebreak and can win the contested
    field ONLY on trust rank. So for the totals to stay equal to the baseline,
    ``resolve_field`` must ACTIVELY pick the high-trust value on trust rank alone (on
    both the scalar ``avg_power_w`` and the per-channel power stream) — a genuine
    trust-resolution assertion (CONF-R2), not a tautology over value-identical copies
    and not an artifact of the id ordering. (Neutralizing trust rank in ``_sort_key``
    flips the winner to the decoy and fails the DIRECT value assertion below.)
    """
    base_count, baseline, base_power = asyncio.run(_run_one(watts, seconds, 1))
    assert base_count == 1  # one source ⇒ one canonical activity
    # The lone source IS the high-trust truth, so the baseline resolves to the truth power.
    assert base_power["avg_power_w"] == pytest.approx(watts)
    assert base_power["stream_sample"] == pytest.approx(watts)

    multi_count, multi, multi_power = asyncio.run(_run_one(watts, seconds, k))
    # COUNT invariant: k duplicate sources collapse to ONE canonical activity.
    assert multi_count == 1, (
        f"{k} duplicate sources must resolve to ONE activity, got {multi_count}"
    )
    # DIRECT trust assertion (ordering-independent): the RESOLVED canonical contested
    # value is the HIGH-trust truth ``watts``, NOT the lower-trust decoy value — even
    # though the decoy holds the favorable smallest-id tiebreak. Read straight off the
    # resolved ``Activity.avg_power_w`` and a ``power_w`` stream sample, so a trust
    # inversion that let the decoy through is caught here directly, not only downstream.
    assert multi_power["avg_power_w"] == pytest.approx(watts), (
        "resolved avg_power_w must be the high-trust truth, not the lower-trust decoy"
    )
    assert multi_power["stream_sample"] == pytest.approx(watts), (
        "resolved power_w stream must be the high-trust truth, not the lower-trust decoy"
    )
    assert multi_power["avg_power_w"] != pytest.approx(_decoy_watts(watts))
    # LOAD invariant: the lower-trust decoy was rejected, so NO aggregate total moved
    # — every resolved total equals the single high-trust truth baseline.
    _assert_close(baseline, multi)
