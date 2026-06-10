"""MAP-R10/MAP-R12 identity-resolution proof suite: strong fingerprint + split.

Proves through the REAL ingest write path that a typed ``strong_fingerprint`` merges
the same real-world session ACROSS the ±2h identity window (MAP-R10 "strong signals
MUST be sufficient to match regardless of the time window"), that every identity
decision is RECORDED on the candidate row (rule fired, match score, matched ids —
MAP-R12), and that a mistaken merge can be undone by the explicit, recorded
``split_activity`` operation with no destruction of contributing source values.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.ingestion.reresolve import split_activity
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


async def _seed(session: AsyncSession) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """Seed the athlete, two sports, and TWO source descriptors."""
    session.add_all(
        [
            Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True),
            Sport(sport_code="running", display_name="Running", has_mechanical_power=False),
        ]
    )
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    a = SourceDescriptor(source_key="src_a", display_name="A", kind="file_upload")
    b = SourceDescriptor(source_key="src_b", display_name="B", kind="oauth_api")
    session.add_all([a, b])
    await session.flush()
    ids = (athlete.athlete_id, a.source_descriptor_id, b.source_descriptor_id)
    await session.commit()
    return ids


def _ride(
    sid: uuid.UUID,
    native: str,
    *,
    start: _dt.datetime = _START,
    sport: str = "cycling",
    fingerprint: str | None = None,
    power: float = 200.0,
) -> GboCandidate:
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id=str(sid),
        source_native_id=native,
        content_hash=content_hash(f"{native}|{start}|{power}".encode()),
        payload={
            "start_time": start,
            "sport": sport,
            "elapsed_time_s": 3600,
            "avg_power_w": power,
        },
        observed_at=start,
        fetched_at=start + _dt.timedelta(hours=1),
        trust_tier=Fidelity.RAW_STREAM,
        strong_fingerprint=fingerprint,
    )


async def test_strong_fingerprint_merges_across_the_window(session: AsyncSession) -> None:
    """MAP-R10: two sources reporting the same session with start times 3h apart (outside
    the ±2h window) but sharing a typed device fingerprint resolve to ONE canonical
    activity, and the merge decision is recorded with the fingerprint rule (MAP-R12)."""
    athlete, src_a, src_b = await _seed(session)
    svc = IngestService(session)
    await svc.ingest(athlete, src_a, [_ride(src_a, "a-1", fingerprint="garmin|1|42|t0")])
    shifted = _START + _dt.timedelta(hours=3)
    await svc.ingest(
        athlete, src_b, [_ride(src_b, "b-1", start=shifted, fingerprint="garmin|1|42|t0")]
    )
    acts = (await session.execute(select(Activity))).scalars().all()
    assert len(acts) == 1  # cross-window merge via the strong signal
    row_b = (
        await session.execute(
            select(SourceCandidate).where(SourceCandidate.source_native_id == "b-1")
        )
    ).scalar_one()
    assert row_b.identity_resolution is not None
    assert row_b.identity_resolution["rule"] == "strong_fingerprint"
    assert row_b.identity_resolution["match_score"] == 1.0
    assert row_b.identity_resolution["matched_activity_id"] == str(acts[0].activity_id)


async def test_shared_fingerprint_never_merges_incompatible_sports(
    session: AsyncSession,
) -> None:
    """MAP-R10 (conservative): a colliding fingerprint on an INCOMPATIBLE sport must not
    merge — the sport gate precedes the fingerprint short-circuit."""
    athlete, src_a, src_b = await _seed(session)
    svc = IngestService(session)
    await svc.ingest(athlete, src_a, [_ride(src_a, "a-1", fingerprint="fp-x")])
    await svc.ingest(
        athlete, src_b, [_ride(src_b, "b-1", sport="running", fingerprint="fp-x")]
    )
    acts = (await session.execute(select(Activity))).scalars().all()
    assert len(acts) == 2  # distinct sports stay separate sessions


async def test_windowed_and_new_record_decisions_are_recorded(
    session: AsyncSession,
) -> None:
    """MAP-R12: a windowed fuzzy merge records rule + score + matched activity id, and a
    no-match candidate records the explicit new-record decision."""
    athlete, src_a, src_b = await _seed(session)
    svc = IngestService(session)
    await svc.ingest(athlete, src_a, [_ride(src_a, "a-1")])
    await svc.ingest(
        athlete, src_b, [_ride(src_b, "b-1", start=_START + _dt.timedelta(seconds=60))]
    )
    row_a = (
        await session.execute(
            select(SourceCandidate).where(SourceCandidate.source_native_id == "a-1")
        )
    ).scalar_one()
    row_b = (
        await session.execute(
            select(SourceCandidate).where(SourceCandidate.source_native_id == "b-1")
        )
    ).scalar_one()
    assert row_a.identity_resolution is not None
    assert row_a.identity_resolution["rule"] == "no_match_new_record"
    assert row_b.identity_resolution is not None
    assert row_b.identity_resolution["rule"] == "windowed_fuzzy"
    assert 0.0 < row_b.identity_resolution["match_score"] <= 1.0
    assert "matched_activity_id" in row_b.identity_resolution


async def test_split_undoes_a_merge_without_losing_source_values(
    session: AsyncSession,
) -> None:
    """MAP-R12: the explicit split re-points one contributing candidate at a fresh
    canonical activity, records the split decision, and re-resolves BOTH records —
    every source's contribution stays intact (reversible at the record level)."""
    athlete, src_a, src_b = await _seed(session)
    svc = IngestService(session)
    await svc.ingest(athlete, src_a, [_ride(src_a, "a-1", power=250.0)])
    await svc.ingest(
        athlete,
        src_b,
        [_ride(src_b, "b-1", start=_START + _dt.timedelta(seconds=30), power=240.0)],
    )
    assert len((await session.execute(select(Activity))).scalars().all()) == 1
    row_b = (
        await session.execute(
            select(SourceCandidate).where(SourceCandidate.source_native_id == "b-1")
        )
    ).scalar_one()
    cand_id = row_b.source_candidate_id  # captured: split expires ORM state
    new_id = await split_activity(session, cand_id)
    acts = {a.activity_id: a for a in (await session.execute(select(Activity))).scalars()}
    assert len(acts) == 2  # the merge is undone: two canonical records again
    assert float(acts[new_id].avg_power_w) == 240.0  # the split candidate's own value
    old = next(a for aid, a in acts.items() if aid != new_id)
    assert float(old.avg_power_w) == 250.0  # the remaining source re-resolved intact
    refreshed = await session.get(SourceCandidate, cand_id)
    assert refreshed is not None
    assert refreshed.identity_resolution is not None
    assert refreshed.identity_resolution["rule"] == "explicit_split"
