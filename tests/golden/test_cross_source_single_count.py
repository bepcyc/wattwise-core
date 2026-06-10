"""GOLD-R5 single-count proof for cross-source dedup (DEDUP-R1 / DEDUP-R4).

THE invariant under proof (doc 80 GOLD-R5 / DEDUP-R1):

    One real ride ingested from N overlapping sources yields the SAME canonical
    activity COUNT and the SAME daily/weekly load TOTALS (PMC CTL/ATL/TSB, the
    TSS / load roll-up, the per-activity Coggan bundle, the best-effort power
    curve — every aggregate the analytics surface exposes for that day) as the
    identical ride ingested from ONE source. Extra sources change fidelity /
    coverage, they NEVER change the COUNTED load.

Every assertion below reads RESOLVED CANONICAL entities only (DEDUP-R4): the
canonical-activity ``COUNT`` is read from the ``Activity`` table the ingest write
path resolves into, and every roll-up is read through :class:`AnalyticsService`,
which itself reads only resolved canonical activities (``daily_load_series`` /
``pmc`` iterate ``Activity`` rows, ``coggan`` reads one resolved ``Activity`` +
its resolved ``ActivityStreamSet``). NOTHING here sums per-source candidates — a
test that summed the ``SourceCandidate`` rows would (by construction) count the
ride N times and FAIL this proof, which is exactly DEDUP-R4's point.

Construction that makes the invariant testable
----------------------------------------------
The single-source baseline ingests ONLY the highest-trust source (file import,
``RAW_STREAM``, 250 W / 3600 s). The multi-source runs ingest that SAME
highest-trust source PLUS one or two lower-trust overlapping sources for the same
real ride (same start / duration / sport, so identity resolution collapses them
to one canonical activity; DIFFERENT power so they are genuinely extra *fidelity*,
not byte clones, and a leaked lower-trust value would visibly move the resolved
field). Because field resolution is trust-ranked (CONF-R2), the higher-trust file
source wins every counted field — so the resolved canonical load is byte-identical
to the baseline, and the only thing the extra sources can change is
coverage/disputed flags, never the counted total.

Crucially the lower-trust extras are deliberately given the FAVORABLE stable
tiebreak: ``_ingest_ride`` zips the specs with ``reversed(descriptor_ids)`` so the
lower-trust sources take the SMALLEST ``source_descriptor_id``s and the high-trust
truth takes the LARGEST. Step 5 of ``dedup._sort_key`` is the ascending-id stable
tiebreak, so the high-trust source LOSES it and can win each contested field ONLY
on trust rank (step 1). This makes the proof a genuine CONF-R2 / PRV-R6 trust test
rather than an artifact of id ordering, and a DIRECT value/stream assertion pins the
resolved ``avg_power_w`` + ``power_w`` sample to the high-trust truth so a trust
inversion is caught straight off the canonical entity. That is the whole claim: more
sources ⇒ same count, same load, with the high-trust value winning purely on trust.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
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

pytestmark = pytest.mark.golden

UTC = _dt.UTC

# The single calendar day the ride lands on, and the PMC read window around it.
_RIDE_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
_RIDE_DAY = _dt.date(2026, 6, 1)
_PMC_FROM = _dt.date(2026, 6, 1)
_PMC_TO = _dt.date(2026, 6, 3)

# Analytics comparison tolerance. The resolved canonical inputs are byte-identical
# across runs (the higher-trust source wins every counted field), so equality is
# in fact exact; we still assert through the documented PMC-R4 relative floor so a
# future legitimate sub-ulp difference would not be a flake.
_TOL = 1e-9


@pytest_asyncio.fixture
async def make_session() -> AsyncIterator[object]:
    """A factory yielding independent fresh in-memory canonical stores.

    Each call returns a brand-new engine + session so the single-source baseline
    and the multi-source runs are computed in ISOLATED stores (no cross-talk):
    the only difference between the stores is how many sources were ingested.
    """
    engines = []

    async def _make() -> AsyncSession:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        engines.append(engine)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        return factory()

    yield _make
    for engine in engines:
        await engine.dispose()


async def _seed(session: AsyncSession, *, n_sources: int) -> tuple[str, list[str]]:
    """Seed the athlete, the cycling sport, an FTP signature, and ``n_sources`` sources."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    descriptors = [
        SourceDescriptor(source_key=f"src_{i}", display_name=f"Source {i}", kind="oauth_api")
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


def _ride_candidate(*, native_id: str, watts: float, seconds: int, tier: Fidelity) -> GboCandidate:
    """A constant-power cycling activity candidate with a 1 Hz power stream.

    All same-ride candidates share ``_RIDE_START`` / ``sport`` and a duration within
    the identity tolerance so the resolver collapses them to ONE canonical activity.
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
        "laps": [
            {"lap_index": 0, "start_offset_s": 0, "duration_s": seconds, "avg_power_w": watts}
        ],
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


# The same real ride, observed by up to three overlapping sources. The FIRST entry
# is the highest-trust (RAW_STREAM file) source — it alone IS the single-source
# baseline, and it wins every counted field when the lower-trust sources are added.
_HI_TRUST = ("hi-file", 250.0, 3600, Fidelity.RAW_STREAM)
_EXTRA_SOURCES = (
    ("mid-api", 251.0, 3605, Fidelity.PLATFORM_COMPUTED),
    ("lo-api", 248.0, 3598, Fidelity.SUMMARY_ONLY),
)
# The high-trust truth power a trust-correct resolver MUST pick for the contested field
# (avg_power_w + the power_w stream). The lower-trust extras carry DIFFERENT powers, so
# a trust inversion that let one through would resolve to a value != this.
_TRUTH_WATTS = _HI_TRUST[1]


async def _ingest_ride(session: AsyncSession, n_sources: int) -> str:
    """Ingest the ride from ``n_sources`` overlapping sources; return the athlete id.

    ``n_sources == 1`` ingests ONLY the highest-trust source (the baseline). Higher
    counts add the lower-trust overlapping observations of the SAME ride.

    The spec→descriptor mapping is REVERSED on purpose: ``descriptor_ids`` are minted
    in creation order (UUIDv7, time-ordered + monotonic), so the FIRST is the smallest.
    Zipping ``specs`` with ``reversed(descriptor_ids)`` hands the HIGH-trust ``_HI_TRUST``
    spec the LARGEST id and the lower-trust extras the SMALLEST ids. Step 5 of
    ``dedup._sort_key`` is the ASCENDING ``source_descriptor_id`` stable tiebreak, so the
    high-trust source now LOSES that tiebreak — it can win each contested field ONLY via
    step-1 trust rank, making the single-count/load proof a genuine trust-resolution test
    (CONF-R2/PRV-R6), not an artifact of the id ordering. (With ``n_sources == 1`` the
    reverse is a no-op, so the baseline still ingests the high-trust source.)
    """
    athlete_id, descriptor_ids = await _seed(session, n_sources=n_sources)
    ingest = IngestService(session)
    specs = [_HI_TRUST, *_EXTRA_SOURCES][:n_sources]
    for descriptor_id, (native, watts, seconds, tier) in zip(
        reversed(descriptor_ids), specs, strict=True
    ):
        cand = _ride_candidate(native_id=native, watts=watts, seconds=seconds, tier=tier)
        await ingest.ingest(athlete_id, descriptor_id, [cand])
    await session.commit()
    return athlete_id


async def _canonical_activity_count(session: AsyncSession) -> int:
    """COUNT of RESOLVED canonical activities (DEDUP-R4 read path — Activity, not candidates)."""
    return len((await session.execute(select(Activity))).scalars().all())


async def _resolved_power(session: AsyncSession) -> dict[str, float]:
    """The RESOLVED canonical contested value: the ``avg_power_w`` scalar + a stream sample.

    Reads the single resolved canonical ``Activity`` and a sample of its resolved
    ``power_w`` ``StreamChannel`` (via the activity's ``ActivityStreamSet``). Mirrors the
    query idiom in ``tests/integration/test_ingestion_findings.py`` so a trust inversion
    that let a lower-trust value win EITHER the scalar field or the per-channel stream is
    caught directly, ordering-independently — not only through the load roll-up.
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


async def _rollups(session: AsyncSession, athlete_id: str) -> dict[str, object]:
    """Every aggregate the analytics surface exposes for the ride's day.

    Read EXCLUSIVELY through :class:`AnalyticsService`, which reads resolved
    canonical activities only (DEDUP-R4): never a candidate sum.
    """
    svc = AnalyticsService(session)
    out: dict[str, object] = {}

    # The single resolved canonical activity for the day (DEDUP-R4: from Activity).
    activities = await svc._activities_in_range(athlete_id, _RIDE_DAY, _RIDE_DAY)
    assert len(activities) == 1, "the day resolved to a single canonical activity"
    activity_id = str(activities[0].activity_id)

    # Per-activity Coggan load bundle (TSS / NP / IF / duration / VI / tss_per_hour).
    bundle = await svc.coggan(activity_id)
    assert is_computed(bundle)
    b = bundle.value
    assert is_computed(b.tss) and is_computed(b.np) and is_computed(b.if_)
    assert is_computed(b.duration_valid_s) and is_computed(b.tss_per_hour)
    assert is_computed(b.variability_index)
    out["tss"] = b.tss.value
    out["np_w"] = b.np.value.np_w
    out["if"] = b.if_.value
    out["duration_valid_s"] = b.duration_valid_s.value
    out["tss_per_hour"] = b.tss_per_hour.value
    out["variability_index"] = b.variability_index.value

    # Daily summed load roll-up over resolved canonical activities (LOAD-R1).
    daily = await svc.daily_load_series(athlete_id, _RIDE_DAY, _RIDE_DAY)
    out["daily_load"] = daily[_RIDE_DAY]

    # PMC CTL / ATL / TSB for the ride's day (the weekly/daily load chart, PMC-R1).
    pmc = await svc.pmc(athlete_id, _PMC_FROM, _PMC_TO)
    assert is_computed(pmc[0])
    out["ctl"] = pmc[0].value.ctl
    out["atl"] = pmc[0].value.atl
    out["tsb"] = pmc[0].value.tsb

    # Best-effort power curve — the peak mean power at each duration (MMP-R4).
    curve = await svc.power_curve(athlete_id, _RIDE_DAY, _RIDE_DAY)
    out["best_efforts"] = tuple(
        sorted((d, res.value.mean_power_w) for d, res in curve.items() if is_computed(res))
    )
    return out


def _assert_rollups_identical(baseline: dict[str, object], other: dict[str, object]) -> None:
    """Assert EVERY roll-up total is identical within the analytics tolerance."""
    assert baseline.keys() == other.keys()
    for key, b in baseline.items():
        o = other[key]
        if key == "best_efforts":
            # Same set of durations, each peak mean power equal within tolerance.
            assert isinstance(b, tuple) and isinstance(o, tuple)
            assert [d for d, _ in b] == [d for d, _ in o], f"{key}: durations differ"
            for (_, bp), (_, op) in zip(b, o, strict=True):
                assert bp == pytest.approx(op, abs=_TOL * max(1.0, abs(bp))), key
        else:
            assert isinstance(b, int | float) and isinstance(o, int | float)
            assert o == pytest.approx(b, abs=_TOL * max(1.0, abs(b))), key


async def test_single_source_baseline_is_one_activity_with_computed_load(
    make_session: object,
) -> None:
    """The one-source baseline lands exactly ONE canonical activity with real load."""
    make = make_session  # the fixture yields an async store factory
    session: AsyncSession = await make()  # type: ignore[operator]
    athlete_id = await _ingest_ride(session, n_sources=1)
    assert await _canonical_activity_count(session) == 1
    # The lone source IS the high-trust truth, so the resolved contested value is _TRUTH_WATTS.
    base_power = await _resolved_power(session)
    assert base_power["avg_power_w"] == pytest.approx(_TRUTH_WATTS)
    assert base_power["stream_sample"] == pytest.approx(_TRUTH_WATTS)
    rollups = await _rollups(session, athlete_id)
    # The baseline is a real, computed load (not a vacuous all-zero pass).
    assert rollups["tss"] == pytest.approx(100.0, abs=1e-3)  # 250 W @ FTP 250, 3600 s
    assert isinstance(rollups["daily_load"], float) and rollups["daily_load"] > 0.0
    assert isinstance(rollups["ctl"], float) and rollups["ctl"] > 0.0
    await session.close()


@pytest.mark.parametrize("n_sources", [2, 3])
async def test_multi_source_yields_same_count_and_same_load_totals(
    make_session: object, n_sources: int
) -> None:
    """N overlapping sources ⇒ COUNT == 1 and EVERY roll-up == the one-source baseline.

    This is the GOLD-R5 / DEDUP-R1 proof: extra sources add coverage/fidelity but
    NEVER change the counted activity total or any load aggregate. Computed over
    resolved canonical entities (DEDUP-R4), never by summing candidates.

    The lower-trust extra sources are handed the SMALLEST ``source_descriptor_id``s
    (``_ingest_ride`` zips the specs with ``reversed(descriptor_ids)``), so the
    high-trust truth source holds the LARGEST id and LOSES the step-5 ascending-id
    stable tiebreak in ``dedup._sort_key``. It can therefore win each contested field
    ONLY on trust rank — making this a genuine CONF-R2 / PRV-R6 trust-resolution proof,
    not an artifact of the id ordering. The DIRECT value/stream assertion below would
    fail if trust rank were neutralized (the lower-trust value would win the id tiebreak).
    """
    make = make_session

    # Baseline: the SAME ride from ONE source, in its own isolated store.
    base_session: AsyncSession = await make()  # type: ignore[operator]
    base_athlete = await _ingest_ride(base_session, n_sources=1)
    assert await _canonical_activity_count(base_session) == 1
    baseline = await _rollups(base_session, base_athlete)

    # Same ride from N overlapping sources, in a fresh isolated store.
    multi_session: AsyncSession = await make()  # type: ignore[operator]
    multi_athlete = await _ingest_ride(multi_session, n_sources=n_sources)

    # COUNT invariant: N sources collapse to ONE canonical activity (DEDUP-R1).
    assert await _canonical_activity_count(multi_session) == 1, (
        f"{n_sources} sources for one ride must resolve to ONE canonical activity"
    )

    # DIRECT trust assertion (ordering-independent): the RESOLVED canonical contested
    # value is the HIGH-trust truth (_TRUTH_WATTS), NOT any lower-trust value — even
    # though the lower-trust sources hold the favorable smallest-id tiebreak. Read off
    # the resolved Activity.avg_power_w and a power_w stream sample, so a trust inversion
    # that let a lower-trust value through is caught here directly, not only downstream.
    multi_power = await _resolved_power(multi_session)
    assert multi_power["avg_power_w"] == pytest.approx(_TRUTH_WATTS), (
        "resolved avg_power_w must be the high-trust truth, not a lower-trust source"
    )
    assert multi_power["stream_sample"] == pytest.approx(_TRUTH_WATTS), (
        "resolved power_w stream must be the high-trust truth, not a lower-trust source"
    )

    # LOAD invariant: every aggregate total is identical to the one-source baseline.
    multi = await _rollups(multi_session, multi_athlete)
    _assert_rollups_identical(baseline, multi)

    await base_session.close()
    await multi_session.close()
