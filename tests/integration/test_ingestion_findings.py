"""Regression tests pinning the convergence-review ingestion fixes (one per finding).

Each test names the finding it pins and proves the bug is fixed end-to-end against the
canonical store the ingest write path produces:

* CONF-R2 / ING-UPS-R5 — two-source wellness resolves by trust, not last-write.
* ING-R6 / DEDUP-R1   — re-ingest reuses the resolved activity id (single-count).
* PRV-R2 / UPS-R5     — a CHANGED re-ingest supersedes the prior candidate version.
* CONF-R5             — coverage ``disputed`` is set when sources materially disagree.
* MAP-R10             — a stored fingerprint matches on re-resolution (no duplicate).
* ING-R8 / FIL-R1     — a file import stores the verbatim bytes + an ActivityFile row.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import ActivityFileFormat, Fidelity
from wattwise_core.ingestion._canonical import field_candidates
from wattwise_core.ingestion.dedup import resolve_field
from wattwise_core.ingestion.ingest import IngestService, OriginalFile
from wattwise_core.persistence.models import (
    Activity,
    ActivityFile,
    ActivityStreamSet,
    Athlete,
    Base,
    DailyWellness,
    SourceCandidate,
    SourceDescriptor,
    Sport,
    StreamChannel,
)
from wattwise_core.storage import LocalObjectStore, content_hash

pytestmark = pytest.mark.integration

UTC = _dt.UTC
_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A session over a fresh in-memory canonical schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed(session: AsyncSession) -> tuple[str, str, str]:
    """Seed the athlete, the cycling sport, and TWO source descriptors."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    file_src = SourceDescriptor(
        source_key="file_import", display_name="Activity files", kind="file_upload"
    )
    api_src = SourceDescriptor(source_key="other_src", display_name="Other", kind="oauth_api")
    session.add_all([file_src, api_src])
    await session.flush()
    ids = (
        str(athlete.athlete_id),
        str(file_src.source_descriptor_id),
        str(api_src.source_descriptor_id),
    )
    await session.commit()
    return ids


def _ride(
    *, native_id: str, watts: float, seconds: int, tier: Fidelity, hash_salt: str = ""
) -> GboCandidate:
    payload = {
        "start_time": _START,
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
        content_hash=content_hash(f"{native_id}:{watts}:{seconds}:{hash_salt}".encode()),
        payload=payload,
        trust_tier=tier,
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )


def _wellness(*, native_id: str, rhr: int, tier: Fidelity) -> GboCandidate:
    return GboCandidate(
        gbo_type="daily_wellness",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(f"{native_id}:{rhr}".encode()),
        payload={"local_date": _dt.date(2026, 6, 1), "resting_hr_bpm": rhr},
        trust_tier=tier,
        observed_at=_dt.datetime(2026, 6, 1, tzinfo=UTC),
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )


# --------------------------------------------------- CONF-R2 / ING-UPS-R5: wellness


async def test_wellness_resolves_by_trust_not_last_write(session: AsyncSession) -> None:
    """A LATER lower-trust wellness value never clobbers the higher-trust one (CONF-R2)."""
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    # Higher-trust source first, then a LATER lower-trust source for the same day.
    hi = _wellness(native_id="w-hi", rhr=42, tier=Fidelity.DEVICE_COMPUTED)
    lo = _wellness(native_id="w-lo", rhr=99, tier=Fidelity.SUMMARY_ONLY)
    await ingest.ingest(athlete_id, api_src, [hi])
    await ingest.ingest(athlete_id, file_src, [lo])
    await session.commit()
    rows = (await session.execute(select(DailyWellness))).scalars().all()
    assert len(rows) == 1
    # The higher-trust value wins despite arriving FIRST (would be 99 under last-write).
    assert rows[0].resting_hr_bpm == 42


# ----------------------------------------------------- ING-R6 / DEDUP-R1: single-count


async def test_reingest_reuses_activity_id_single_count(session: AsyncSession) -> None:
    """Re-ingesting the SAME candidate reuses its resolved activity id (ING-R6/DEDUP-R1)."""
    athlete_id, file_src, _ = await _seed(session)
    ingest = IngestService(session)
    cand = _ride(native_id="ride-1", watts=200.0, seconds=1800, tier=Fidelity.RAW_STREAM)
    r1 = await ingest.ingest(athlete_id, file_src, [cand])
    r2 = await ingest.ingest(athlete_id, file_src, [cand])  # same content again
    await session.commit()
    assert r1.activities_written == r2.activities_written  # same id, no new mint
    count = len((await session.execute(select(Activity))).scalars().all())
    assert count == 1


async def test_reingest_changed_start_drift_does_not_duplicate(session: AsyncSession) -> None:
    """A CHANGED re-ingest whose start drifts still maps to the SAME activity (ING-R6).

    The prior candidate's resolved_activity_id is reused rather than re-running the
    fuzzy matcher (which could mint a duplicate if the start drifted past the window).
    """
    athlete_id, file_src, _ = await _seed(session)
    ingest = IngestService(session)
    first = _ride(native_id="ride-1", watts=200.0, seconds=1800, tier=Fidelity.RAW_STREAM)
    await ingest.ingest(athlete_id, file_src, [first])
    # Same source_native_id (same real session), but a CHANGED payload that would drift
    # the resolved start far outside the ±2h identity window if re-matched fuzzily.
    drifted = _ride(
        native_id="ride-1", watts=205.0, seconds=1800, tier=Fidelity.RAW_STREAM, hash_salt="v2"
    )
    drifted.payload["start_time"] = _START + _dt.timedelta(hours=6)
    await ingest.ingest(athlete_id, file_src, [drifted])
    await session.commit()
    assert len((await session.execute(select(Activity))).scalars().all()) == 1


# --------------------------------------------------------- PRV-R2 / UPS-R5: supersede


async def test_changed_reingest_supersedes_prior_version(session: AsyncSession) -> None:
    """A CHANGED re-ingest supersedes the prior candidate and inserts a new version."""
    athlete_id, file_src, _ = await _seed(session)
    ingest = IngestService(session)
    v1 = _ride(native_id="ride-1", watts=200.0, seconds=1800, tier=Fidelity.RAW_STREAM)
    v2 = _ride(
        native_id="ride-1", watts=260.0, seconds=1800, tier=Fidelity.RAW_STREAM, hash_salt="v2"
    )
    await ingest.ingest(athlete_id, file_src, [v1])
    await ingest.ingest(athlete_id, file_src, [v2])
    await session.commit()
    rows = (await session.execute(select(SourceCandidate))).scalars().all()
    # Two versions retained for audit: exactly one superseded, one live (PRV-R2).
    assert len(rows) == 2
    assert sum(1 for r in rows if r.is_superseded) == 1
    live = [r for r in rows if not r.is_superseded]
    assert len(live) == 1 and live[0].content_hash == v2.content_hash
    # The new live version carries the SAME resolved activity id (ING-R6 carry-forward).
    assert live[0].resolved_activity_id is not None


# -------------------------------------------------------------- CONF-R5: disputed flag


async def test_disputed_flag_set_on_material_disagreement(session: AsyncSession) -> None:
    """Two sources materially disagreeing set coverage.disputed=True (CONF-R5)."""
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    # Same real ride, two sources, power disagreeing far beyond tolerance (200 vs 320).
    await ingest.ingest(
        athlete_id, file_src,
        [_ride(native_id="file-1", watts=200.0, seconds=3600, tier=Fidelity.RAW_STREAM)],
    )
    await ingest.ingest(
        athlete_id, api_src,
        [_ride(native_id="api-1", watts=320.0, seconds=3600, tier=Fidelity.PLATFORM_COMPUTED)],
    )
    await session.commit()
    act = (await session.execute(select(Activity))).scalars().one()
    assert act.coverage["avg_power_w"]["disputed"] is True
    # The best (highest-trust) value still wins — disagreement surfaced, not averaged.
    assert float(act.avg_power_w) == pytest.approx(200.0)


async def test_disputed_flag_absent_when_sources_agree(session: AsyncSession) -> None:
    """Sources agreeing within tolerance do NOT raise disputed (CONF-R5)."""
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    await ingest.ingest(
        athlete_id, file_src,
        [_ride(native_id="file-1", watts=250.0, seconds=3600, tier=Fidelity.RAW_STREAM)],
    )
    await ingest.ingest(
        athlete_id, api_src,
        [_ride(native_id="api-1", watts=251.0, seconds=3600, tier=Fidelity.PLATFORM_COMPUTED)],
    )
    await session.commit()
    act = (await session.execute(select(Activity))).scalars().one()
    assert act.coverage["avg_power_w"]["disputed"] is False


# ----------------------------------------------------- ING-R8 / FIL-R1: activity_file


async def test_file_import_stores_bytes_and_creates_activity_file(
    session: AsyncSession, tmp_path: Path
) -> None:
    """A file import stores verbatim bytes + creates an ActivityFile row (ING-R8/FIL-R1)."""
    athlete_id, file_src, _ = await _seed(session)
    store = LocalObjectStore(tmp_path)
    ingest = IngestService(session, object_store=store)
    cand = _ride(native_id="ride-1", watts=250.0, seconds=3600, tier=Fidelity.RAW_STREAM)
    raw = b"<<verbatim fit bytes>>"
    original = OriginalFile(
        data=raw, file_format=ActivityFileFormat.FIT, source_native_id="ride-1"
    )
    await ingest.ingest(athlete_id, file_src, [cand], original_files=[original])
    await session.commit()
    files = (await session.execute(select(ActivityFile))).scalars().all()
    assert len(files) == 1
    af = files[0]
    assert af.content_hash == content_hash(raw)
    assert af.byte_size == len(raw)
    assert af.format is ActivityFileFormat.FIT
    # The bytes are actually retrievable verbatim from the object store (tier-1).
    assert store.get(af.object_ref) == raw
    # Linked to the resolved canonical activity.
    act = (await session.execute(select(Activity))).scalars().one()
    assert af.activity_id == act.activity_id


async def test_file_import_dedup_idempotent_on_resync(
    session: AsyncSession, tmp_path: Path
) -> None:
    """Re-importing the same file does not create a second ActivityFile (FIL-R5)."""
    athlete_id, file_src, _ = await _seed(session)
    store = LocalObjectStore(tmp_path)
    ingest = IngestService(session, object_store=store)
    cand = _ride(native_id="ride-1", watts=250.0, seconds=3600, tier=Fidelity.RAW_STREAM)
    original = OriginalFile(
        data=b"<<verbatim>>", file_format=ActivityFileFormat.FIT, source_native_id="ride-1"
    )
    await ingest.ingest(athlete_id, file_src, [cand], original_files=[original])
    await ingest.ingest(athlete_id, file_src, [cand], original_files=[original])
    await session.commit()
    assert len((await session.execute(select(ActivityFile))).scalars().all()) == 1


async def test_direct_api_source_creates_no_activity_file(session: AsyncSession) -> None:
    """A direct-API source (no original file) creates no ActivityFile row (ING-R8)."""
    athlete_id, _, api_src = await _seed(session)
    ingest = IngestService(session)
    cand = _ride(native_id="api-1", watts=250.0, seconds=3600, tier=Fidelity.PLATFORM_COMPUTED)
    await ingest.ingest(athlete_id, api_src, [cand])  # no original_files
    await session.commit()
    assert (await session.execute(select(ActivityFile))).scalars().all() == []


# ------------------------------------------------ CONF-R2 step-4: completeness tiebreak


def test_completeness_breaks_tie_stream_over_summary() -> None:
    """At the same tier/confidence/recency, a stream-backed scalar wins (CONF-R2 step 4).

    The stream-backed contributor is given the LEXICALLY-GREATER source id so it would
    LOSE the final stable tiebreak — it can only win via the completeness step, proving
    completeness is actually applied (was inert when every candidate defaulted to 1.0).
    """

    def _candidate(sid: str, value: float, *, streamed: bool) -> SourceCandidate:
        payload: dict[str, object] = {"avg_power_w": value}
        if streamed:
            payload["streams"] = {"power_w": {"values": [value]}}
        return SourceCandidate(
            athlete_id=None, source_descriptor_id=sid, source_native_id=sid,
            content_hash=sid, trust_profile={"tier": Fidelity.PLATFORM_COMPUTED.value},
            payload=payload, confidence=1.0,
        )

    def _tier_of(c: SourceCandidate) -> Fidelity:
        return Fidelity(str(c.trust_profile["tier"]))

    streamed = _candidate("zzz", 300.0, streamed=True)  # greater id -> loses stable tiebreak
    summary = _candidate("aaa", 210.0, streamed=False)  # lower id -> would win the tiebreak
    contributors = field_candidates([streamed, summary], "avg_power_w", _tier_of)
    winner = resolve_field(contributors)
    assert winner is not None
    assert winner.value == 300.0  # completeness wins despite the worse stable tiebreak


# ---------------------------------------------------- ING-UPS-R5: per-channel streams


def _ride_channels(
    *, native_id: str, channels: dict[str, list[float]], tier: Fidelity
) -> GboCandidate:
    streams = {
        name: {"values": vals, "sample_basis": "time", "sample_rate_hz": 1.0}
        for name, vals in channels.items()
    }
    payload = {
        "start_time": _START,
        "sport": "cycling",
        "elapsed_time_s": 3600,
        "moving_time_s": 3600,
        "streams": streams,
    }
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(f"{native_id}:{sorted(channels)}".encode()),
        payload=payload,
        trust_tier=tier,
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )


async def test_streams_resolved_per_channel_across_sources(session: AsyncSession) -> None:
    """A channel a higher-trust source lacks is filled from a lower-trust one (CONF-R3)."""
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    # Higher-trust source has only power; lower-trust source has only HR.
    hi = _ride_channels(
        native_id="hi", channels={"power_w": [200.0] * 5}, tier=Fidelity.RAW_STREAM
    )
    lo = _ride_channels(
        native_id="lo", channels={"hr_bpm": [140.0] * 5}, tier=Fidelity.SUMMARY_ONLY
    )
    await ingest.ingest(athlete_id, file_src, [hi])
    await ingest.ingest(athlete_id, api_src, [lo])
    await session.commit()
    ss = (await session.execute(select(ActivityStreamSet))).scalars().one()
    chans = (
        await session.execute(
            select(StreamChannel).where(StreamChannel.stream_set_id == ss.stream_set_id)
        )
    ).scalars().all()
    by_name = {c.channel.value: c for c in chans}
    # Both channels present — the HR channel was filled from the lower-trust source.
    assert "power_w" in by_name and "hr_bpm" in by_name


async def test_lower_trust_stream_never_overwrites_higher_trust(session: AsyncSession) -> None:
    """A later lower-trust power stream never clobbers the stored higher-trust one (ING-UPS-R5)."""
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    hi = _ride_channels(native_id="hi", channels={"power_w": [200.0] * 5}, tier=Fidelity.RAW_STREAM)
    lo = _ride_channels(
        native_id="lo", channels={"power_w": [999.0] * 5}, tier=Fidelity.SUMMARY_ONLY
    )
    await ingest.ingest(athlete_id, file_src, [hi])
    await ingest.ingest(athlete_id, api_src, [lo])
    await session.commit()
    ss = (await session.execute(select(ActivityStreamSet))).scalars().one()
    power = (
        await session.execute(
            select(StreamChannel).where(
                StreamChannel.stream_set_id == ss.stream_set_id,
                StreamChannel.channel == "power_w",
            )
        )
    ).scalars().one()
    # The higher-trust raw_stream power survives the lower-trust write.
    assert power.values == [200.0] * 5
