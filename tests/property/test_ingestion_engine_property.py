"""Engine-invariant property suite (doc 30 TST-R4) over generated inputs.

Property-based assertions for the sync-engine invariants TST-R4 names:

* **idempotency** (ING-R6/SYN-R4): re-ingesting the same generated candidate batch
  converges to the SAME canonical state — one activity, identical resolved scalars;
* **watermark monotonicity** (SYN-R3): for any generated sequence of advance calls,
  the high-water cursor equals the running maximum — it NEVER regresses;
* **range-precise gaps** (ING-GAP-R5): for any generated failing subset of records,
  the opened gap tokens are EXACTLY the failed ids and the mapped survivors are
  exactly the complement;
* **conflict-resolution determinism** (PRV-R4): the resolved canonical value is
  invariant to the ingest ORDER of competing candidates — the trust order decides;
* **no-raw-JSON read path** (ING-R9): for any generated source activity (including
  arbitrary unknown source keys), the mapped canonical payload carries only named
  typed fields — no source-shaped key, no passed-through object blob.

Each DB-bound example runs over a brand-new single-connection SQLite engine driven
via ``asyncio.run`` (the established pattern of
``test_cross_source_dedup_property.py``) — these are value-semantics properties,
not concurrency/pool tests. ``max_examples`` is small with an explicit deadline so
the suite stays deterministic and cheap in the offline gate (TIER-R1).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid as _uuid
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, GboType
from wattwise_core.ingestion._sync_records import map_records_isolated
from wattwise_core.ingestion.adapters.intervals_icu import (
    IntervalsActivityAsbo,
    IntervalsIcuAdapter,
)
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.ingestion.watermark import advance_watermark, watermark_for
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    Base,
    SourceDescriptor,
    Sport,
)
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.property

UTC = _dt.UTC
_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
_FETCHED = _dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
_SETTINGS = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _candidate(
    native_id: str, watts: float, *, tier: Fidelity = Fidelity.RAW_STREAM
) -> GboCandidate:
    payload: dict[str, Any] = {
        "start_time": _START,
        "sport": "cycling",
        "elapsed_time_s": 600,
        "avg_power_w": watts,
    }
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(f"{native_id}:{watts}:{tier}".encode()),
        payload=payload,
        observed_at=_START,
        fetched_at=_FETCHED,
        trust_tier=tier,
    )


async def _fresh_store() -> tuple[Any, Any, _uuid.UUID, _uuid.UUID]:
    """A brand-new schema + seeded athlete/sport/descriptor for one example."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        athlete = Athlete(sex="male", reference_timezone="UTC")
        session.add(athlete)
        descriptor = SourceDescriptor(source_key="prop_src", display_name="p", kind="oauth_api")
        session.add(descriptor)
        await session.flush()
        await session.commit()
        return engine, maker, athlete.athlete_id, descriptor.source_descriptor_id


@_SETTINGS
@given(
    watts=st.floats(min_value=80, max_value=400),
    repeats=st.integers(min_value=2, max_value=4),
)
def test_reingest_converges_to_identical_canonical_state(watts: float, repeats: int) -> None:
    """ING-R6/SYN-R4: k ingests of the same batch == one ingest (count AND values)."""

    async def run() -> tuple[int, float]:
        engine, maker, athlete, descriptor = await _fresh_store()
        try:
            for _ in range(repeats):
                async with maker() as session:
                    await IngestService(session).ingest(
                        str(athlete), str(descriptor), [_candidate("ride-x", watts)]
                    )
            async with maker() as session:
                acts = (await session.execute(select(Activity))).scalars().all()
                return len(acts), float(acts[0].avg_power_w)
        finally:
            await engine.dispose()

    count, resolved = asyncio.run(run())
    assert count == 1
    assert resolved == pytest.approx(watts)


@_SETTINGS
@given(offsets=st.lists(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=8))
def test_watermark_high_water_is_monotonic(offsets: list[int]) -> None:
    """SYN-R3: the high-water cursor equals the running max — a re-run never regresses it."""

    async def run() -> list[_dt.datetime]:
        engine, maker, athlete, descriptor = await _fresh_store()
        observed: list[_dt.datetime] = []
        try:
            async with maker() as session:
                for off in offsets:
                    instant = _START + _dt.timedelta(seconds=off)
                    await advance_watermark(
                        session,
                        athlete,
                        descriptor,
                        GboType.ACTIVITY,
                        high_water_at=instant,
                        content_hint=None,
                    )
                    row = await watermark_for(session, athlete, descriptor, GboType.ACTIVITY)
                    assert row is not None and row.high_water_at is not None
                    observed.append(row.high_water_at.replace(tzinfo=UTC))
                await session.commit()
            return observed
        finally:
            await engine.dispose()

    observed = asyncio.run(run())
    running_max = _START
    for off, seen in zip(offsets, observed, strict=True):
        running_max = max(running_max, _START + _dt.timedelta(seconds=off))
        assert seen == running_max


@_SETTINGS
@given(data=st.data())
def test_map_failures_are_exactly_token_precise(data: st.DataObject) -> None:
    """ING-GAP-R5: the failed-record set is EXACTLY the failing ids; survivors land."""
    ids = data.draw(
        st.lists(
            st.text(alphabet="abcdef0123456789", min_size=4, max_size=8),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    failing = set(data.draw(st.sets(st.sampled_from(ids))))

    class _Asbo:
        def __init__(self, native_id: str) -> None:
            self.native_id = native_id
            self.gbo_type = "activity"

    class _Adapter:
        adapter_version = "1"
        mapping_version = "1"

        def map(self, asbo: Any, ref: Any, ctx: Any) -> list[GboCandidate]:
            if asbo.native_id in failing:
                raise ValueError("unmappable record")
            return [_candidate(asbo.native_id, 200.0)]

    ref = SourceDescriptorRef("sd-1", "prop_src", "oauth_api")  # type: ignore[arg-type]
    ctx = FetchContext(ingest_run_id="run-1", fetched_at=_FETCHED)
    batch = map_records_isolated(_Adapter(), [_Asbo(i) for i in ids], ref, ctx, source_key="p")

    assert {f.source_native_id for f in batch.failed} == failing
    assert {c.source_native_id for c in batch.candidates} == set(ids) - failing


@_SETTINGS
@given(
    truth_watts=st.floats(min_value=100, max_value=400),
    order=st.permutations([0, 1, 2]),
)
def test_conflict_resolution_is_order_invariant(truth_watts: float, order: list[int]) -> None:
    """PRV-R4: the resolved value is decided by trust, never by ingest order."""
    decoy = truth_watts + 50.0
    cands = [
        _candidate("same-ride", truth_watts, tier=Fidelity.RAW_STREAM),
        _candidate("same-ride-2", decoy, tier=Fidelity.SUMMARY_ONLY),
        _candidate("same-ride-3", decoy, tier=Fidelity.MODELED),
    ]

    async def run() -> float:
        engine, maker, athlete, descriptor = await _fresh_store()
        try:
            for idx in order:
                async with maker() as session:
                    await IngestService(session).ingest(str(athlete), str(descriptor), [cands[idx]])
            async with maker() as session:
                acts = (await session.execute(select(Activity))).scalars().all()
                assert len(acts) == 1  # same start/duration/sport -> ONE resolved entity
                return float(acts[0].avg_power_w)
        finally:
            await engine.dispose()

    assert asyncio.run(run()) == pytest.approx(truth_watts)


_CANONICAL_ACTIVITY_KEYS = {
    "start_time",
    "sport",
    "sub_sport",
    "elapsed_time_s",
    "moving_time_s",
    "distance_m",
    "total_work_j",
    "energy_kj",
    "avg_power_w",
    "max_power_w",
    "avg_hr_bpm",
    "max_hr_bpm",
    "avg_cadence_rpm",
    "avg_speed_mps",
    "elevation_gain_m",
    "avg_temp_c",
    "device_class",
    "has_power",
    "has_hr",
    "has_gps",
    "has_cadence",
    "streams",
    "laps",
}


@_SETTINGS
@given(
    watts=st.one_of(st.none(), st.floats(min_value=50, max_value=500)),
    extra_key=st.text(alphabet="abcdefgh_", min_size=3, max_size=10),
)
def test_mapped_payload_is_typed_canonical_never_a_blob(
    watts: float | None, extra_key: str
) -> None:
    """ING-R9: mapping emits ONLY named canonical fields; unknown source keys never leak."""
    raw = {
        "id": "i12345",
        "start_date": "2026-06-01T08:00:00Z",
        "type": "Ride",
        "moving_time": 600,
        "elapsed_time": 600,
        "icu_average_watts": watts,
        extra_key: {"nested": "source blob"},  # arbitrary unknown source key
    }
    asbo = IntervalsActivityAsbo.model_validate(raw)
    ref = SourceDescriptorRef("sd-1", "intervals_icu", "oauth_api")  # type: ignore[arg-type]
    ctx = FetchContext(ingest_run_id="run-1", fetched_at=_FETCHED)
    cands = IntervalsIcuAdapter().map(asbo, ref, ctx)
    assert len(cands) == 1
    payload = cands[0].payload
    assert set(payload) <= _CANONICAL_ACTIVITY_KEYS
    # absence stays absence — never a defaulted zero (ADP-R12)
    if watts is None:
        assert payload["avg_power_w"] is None
