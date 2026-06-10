"""MAP-R2/MAP-R6 candidate validation-gate proof suite.

Proves through the REAL ingest write path that a candidate carrying a non-canonical
(source-named) payload key, or violating a canonical invariant (implausible physical
range, non-monotonic time base, non-contiguous laps), is QUARANTINED: persisted with
its lineage + the failing rule id, EXCLUDED from every resolution set, and never
partially written into the canonical store — while valid candidates in the same
batch still land normally.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.ingestion.validation import validate_candidate
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    Base,
    SourceCandidate,
    SourceDescriptor,
    Sport,
)
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.integration

UTC = _dt.UTC
_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)


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


async def _seed(session: AsyncSession) -> tuple[uuid.UUID, uuid.UUID]:
    """Seed the athlete, cycling, and one source descriptor."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    src = SourceDescriptor(source_key="src_q", display_name="Q", kind="oauth_api")
    session.add_all([athlete, src])
    await session.flush()
    ids = (athlete.athlete_id, src.source_descriptor_id)
    await session.commit()
    return ids


def _cand(sid: uuid.UUID, native: str, payload: dict[str, Any]) -> GboCandidate:
    base: dict[str, Any] = {"start_time": _START, "sport": "cycling", "elapsed_time_s": 3600}
    base.update(payload)
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id=str(sid),
        source_native_id=native,
        content_hash=content_hash(f"{native}|{sorted(base.items())!r}".encode()),
        payload=base,
        observed_at=_START,
        fetched_at=_START,
        trust_tier=Fidelity.PLATFORM_COMPUTED,
    )


async def test_non_canonical_key_is_quarantined_not_resolved(session: AsyncSession) -> None:
    """MAP-R2: a payload carrying a source-named key is rejected by the gate — the row
    is retained with the failing rule id, contributes to NO canonical record, and the
    valid candidate in the same batch still resolves normally."""
    athlete, src = await _seed(session)
    svc = IngestService(session)
    result = await svc.ingest(
        athlete,
        src,
        [
            _cand(src, "bad-1", {"icu_average_watts": 250.0}),
            _cand(src, "good-1", {"avg_power_w": 240.0}),
        ],
    )
    assert result.candidates_quarantined == 1
    assert result.candidates_persisted == 1
    bad = (
        await session.execute(
            select(SourceCandidate).where(SourceCandidate.source_native_id == "bad-1")
        )
    ).scalar_one()
    assert bad.quarantine_rule_id == "MAP-R2:non-canonical-key:icu_average_watts"
    assert bad.resolved_activity_id is None  # excluded from resolution entirely
    act = (await session.execute(select(Activity))).scalar_one()
    assert float(act.avg_power_w) == 240.0  # only the valid candidate contributed


async def test_implausible_range_is_quarantined_with_rule_id(session: AsyncSession) -> None:
    """MAP-R6: a value outside its plausible physical range quarantines the candidate
    with the range rule id; nothing reaches the canonical store."""
    athlete, src = await _seed(session)
    svc = IngestService(session)
    result = await svc.ingest(athlete, src, [_cand(src, "hot-1", {"avg_hr_bpm": 999.0})])
    assert result.candidates_quarantined == 1
    row = (await session.execute(select(SourceCandidate))).scalar_one()
    assert row.quarantine_rule_id == "MAP-R6:range:avg_hr_bpm"
    assert (await session.execute(select(Activity))).scalar_one_or_none() is None


def test_monotonic_time_base_and_lap_contiguity_rules() -> None:
    """MAP-R6: a non-monotonic stream time base and a non-contiguous lap index set each
    fail validation with their own named rule (pure-gate check)."""
    bad_time = GboCandidate(
        gbo_type="activity",
        source_descriptor_id="s",
        source_native_id="n1",
        content_hash="h1",
        payload={
            "start_time": _START,
            "streams": {"time_s": {"values": [0, 1, 5, 3]}},
        },
    )
    assert validate_candidate(bad_time) == "MAP-R6:time-base-not-monotonic"
    bad_laps = GboCandidate(
        gbo_type="activity",
        source_descriptor_id="s",
        source_native_id="n2",
        content_hash="h2",
        payload={
            "start_time": _START,
            "laps": [{"lap_index": 0}, {"lap_index": 2}],
        },
    )
    assert validate_candidate(bad_laps) == "MAP-R6:lap-contiguity"


def test_valid_candidate_passes_the_gate() -> None:
    """The gate is not a tautology: a canonical, in-range candidate with monotonic
    streams and contiguous laps passes with no rule fired."""
    good = GboCandidate(
        gbo_type="activity",
        source_descriptor_id="s",
        source_native_id="n3",
        content_hash="h3",
        payload={
            "start_time": _START,
            "sport": "cycling",
            "avg_power_w": 250.0,
            "streams": {"time_s": {"values": [0, 1, 2]}},
            "laps": [{"lap_index": 0, "start_offset_s": 0}, {"lap_index": 1, "start_offset_s": 60}],
        },
    )
    assert validate_candidate(good) is None
