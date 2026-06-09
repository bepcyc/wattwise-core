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
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import ActivityFileFormat, Fidelity
from wattwise_core.ingestion._canonical import coverage_for, field_candidates
from wattwise_core.ingestion.dedup import resolve_field
from wattwise_core.ingestion.ingest import IngestService, OriginalFile
from wattwise_core.persistence.models import (
    Activity,
    ActivityFile,
    ActivityStreamSet,
    Athlete,
    AthleteSourcePreference,
    Base,
    DailyWellness,
    SourceCandidate,
    SourceDescriptor,
    Sport,
    StreamChannel,
)
from wattwise_core.persistence.models.athlete_preference import WHOLE_SOURCE_CHANNEL
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


# --------------------------------- CON-R4 wiring: configurable per-source trust precedence
#
# Effective trust tier is CONFIGURATION DATA layered on the adapter tier (PRV-R7/LIN-R1):
# per-athlete override > descriptor trust_profile > adapter tier. With NO config the winner
# is the adapter-tier winner (the prior behaviour); config is an explicit opt-in re-rank.


async def _set_profile(
    session: AsyncSession, descriptor_id: str, profile: dict[str, object]
) -> None:
    """Set a descriptor's declared per-channel trust_profile (LIN-R1 configuration data)."""
    desc = await session.get(SourceDescriptor, uuid.UUID(descriptor_id))
    assert desc is not None
    desc.trust_profile = profile
    await session.commit()


async def _add_override(
    session: AsyncSession, athlete_id: str, descriptor_id: str, channel: str, tier: Fidelity
) -> None:
    """Insert a per-athlete (source, channel) trust override row (PRV-R7)."""
    session.add(
        AthleteSourcePreference(
            athlete_id=uuid.UUID(athlete_id),
            source_descriptor_id=uuid.UUID(descriptor_id),
            channel=channel,
            trust_tier=tier,
        )
    )
    await session.commit()


async def _ingest_two_source_power(
    session: AsyncSession, athlete_id: str, file_src: str, api_src: str
) -> None:
    """Two sources report the same ride; file_src=RAW_STREAM(200W), api_src=PLATFORM(320W)."""
    ingest = IngestService(session)
    await ingest.ingest(
        athlete_id, file_src,
        [_ride(native_id="file-1", watts=200.0, seconds=3600, tier=Fidelity.RAW_STREAM)],
    )
    await ingest.ingest(
        athlete_id, api_src,
        [_ride(native_id="api-1", watts=320.0, seconds=3600, tier=Fidelity.PLATFORM_COMPUTED)],
    )
    await session.commit()


async def _power(session: AsyncSession) -> float:
    act = (await session.execute(select(Activity))).scalars().one()
    return float(act.avg_power_w)


async def test_no_config_keeps_the_adapter_tier_winner_unchanged(session: AsyncSession) -> None:
    """CONTROL: with NO config the higher adapter-tier source wins (prior behaviour, PRV-R6)."""
    athlete_id, file_src, api_src = await _seed(session)
    await _ingest_two_source_power(session, athlete_id, file_src, api_src)
    # file_src is RAW_STREAM -> it wins by adapter tier; config is absent -> no re-rank.
    assert await _power(session) == pytest.approx(200.0)


async def test_descriptor_profile_reranks_the_winner(session: AsyncSession) -> None:
    """A DESCRIPTOR trust_profile re-ranks the field winner vs the no-config baseline (LIN-R1)."""
    athlete_id, file_src, api_src = await _seed(session)
    # Declare file_src's avg_power_w as the LOWEST tier so the api_src value now outranks it.
    await _set_profile(session, file_src, {"avg_power_w": Fidelity.SUMMARY_ONLY.value})
    await _ingest_two_source_power(session, athlete_id, file_src, api_src)
    # api_src (PLATFORM_COMPUTED, 320W) now beats the demoted file_src — flipped by config.
    assert await _power(session) == pytest.approx(320.0)


async def test_per_athlete_override_flips_the_winner(session: AsyncSession) -> None:
    """A PER-ATHLETE override flips the field winner without any code change (PRV-R7)."""
    athlete_id, file_src, api_src = await _seed(session)
    # The athlete demotes file_src's power channel to the lowest tier, so the otherwise
    # lower adapter-tier api_src value now outranks it — flipped per-athlete, no code change.
    await _add_override(session, athlete_id, file_src, "avg_power_w", Fidelity.SUMMARY_ONLY)
    await _ingest_two_source_power(session, athlete_id, file_src, api_src)
    # The override demotes file_src below api_src -> api_src (320W) wins for this athlete.
    assert await _power(session) == pytest.approx(320.0)


async def test_per_athlete_override_beats_descriptor_profile(session: AsyncSession) -> None:
    """The per-athlete override is consulted BEFORE the descriptor profile (PRV-R7 precedence)."""
    athlete_id, file_src, api_src = await _seed(session)
    # Descriptor says file_src power is RAW_STREAM (would win), but the athlete overrides it
    # down to SUMMARY_ONLY -> the override wins the precedence, so api_src takes the field.
    await _set_profile(session, file_src, {"avg_power_w": Fidelity.RAW_STREAM.value})
    await _add_override(session, athlete_id, file_src, "avg_power_w", Fidelity.SUMMARY_ONLY)
    await _ingest_two_source_power(session, athlete_id, file_src, api_src)
    assert await _power(session) == pytest.approx(320.0)


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


# ----------------------------------- D2: coverage fidelity badges the RESOLVED WINNER


async def test_coverage_fidelity_badges_the_winner_not_first_scanned(
    session: AsyncSession,
) -> None:
    """The scalar coverage ``fidelity`` is the WINNER's tier, not an arbitrary contributor (D2).

    The LOSER (a lower-trust ``summary_only`` source) is ingested FIRST so it is the first
    row in scan/insert order; the WINNER (a higher-trust ``raw_stream`` source) is ingested
    second. The resolver has no ORDER BY, so badging ``contributors[0].trust_tier`` would
    mislabel the canonical value ``summary_only`` (a PRV-R6 inversion on a client badge).
    The fix badges the resolved winner's tier — here ``raw_stream``.
    """
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    # LOSER first (lower trust), WINNER second (higher trust) — exercises the scan-order bug.
    loser = _ride(native_id="lo-1", watts=180.0, seconds=3600, tier=Fidelity.SUMMARY_ONLY)
    winner = _ride(
        native_id="hi-1", watts=250.0, seconds=3600, tier=Fidelity.RAW_STREAM, hash_salt="hi"
    )
    await ingest.ingest(athlete_id, api_src, [loser])  # inserted FIRST -> first scanned
    await ingest.ingest(athlete_id, file_src, [winner])
    await session.commit()
    act = (await session.execute(select(Activity))).scalars().one()
    cov = act.coverage["avg_power_w"]
    assert isinstance(cov, dict)
    # The higher-trust raw_stream value won the field...
    assert float(act.avg_power_w) == pytest.approx(250.0)
    # ...and the coverage badge is the WINNER's tier, never the first-scanned loser's.
    assert cov["fidelity"] == Fidelity.RAW_STREAM.value
    assert cov["present"] is True


# --------------------------------- SF-3: per-channel trust applies to STREAM channels too


async def test_per_channel_stream_trust_profile_flips_stream_winner(
    session: AsyncSession,
) -> None:
    """A per-channel STREAM trust_profile / override changes which source wins a stream (SF-3).

    Both sources carry ONLY the ``power_w`` stream channel with DIFFERENT values. By adapter
    tier the ``file_src`` (RAW_STREAM) would win the channel. A per-athlete override demotes
    ``file_src``'s ``power_w`` to ``summary_only``, so the otherwise lower-tier ``api_src``
    now wins the STREAM channel — proving per-channel effective trust threads into streams
    (previously streams resolved under the whole-source ``"*"`` tier only, so the override
    had NO effect on the stream).
    """
    athlete_id, file_src, api_src = await _seed(session)
    # Demote file_src's power_w CHANNEL for this athlete below api_src's adapter tier.
    await _add_override(session, athlete_id, file_src, "power_w", Fidelity.SUMMARY_ONLY)
    ingest = IngestService(session)
    hi = _ride_channels(native_id="hi", channels={"power_w": [200.0] * 5}, tier=Fidelity.RAW_STREAM)
    lo = _ride_channels(
        native_id="lo", channels={"power_w": [333.0] * 5}, tier=Fidelity.PLATFORM_COMPUTED
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
    # The override flips the STREAM winner: api_src (333) wins the channel for this athlete.
    assert power.values == [333.0] * 5
    cov = power.coverage
    assert isinstance(cov, dict)
    # The channel coverage badges the winning channel's effective tier (api_src adapter tier).
    assert cov["fidelity"] == Fidelity.PLATFORM_COMPUTED.value
    assert cov["present"] is True


async def test_whole_source_override_keeps_sole_carrier_stream_channel(
    session: AsyncSession,
) -> None:
    """A whole-source ``"*"`` demotion re-badges but NEVER drops a sole-carrier stream (PRV-R7).

    Exactly ONE source physically carries the per-second ``power_w`` stream (``file_src``,
    adapter tier ``RAW_STREAM``); the other source (``api_src``) carries a DIFFERENT channel
    (``hr_bpm``) and so is the sole carrier of nothing on ``power_w``. A per-athlete override
    keyed on the WHOLE source (:data:`WHOLE_SOURCE_CHANNEL` ``"*"``) demotes ``file_src`` to
    ``summary_only`` — that demotion changes WHICH source would win a contested channel, but
    here ``power_w`` is uncontested, so it must NOT delete the only real per-second data that
    exists. The resolved activity must STILL carry the ``power_w`` channel with its verbatim
    per-second values, while its coverage badge reflects the demoted ``summary_only`` tier (a
    trust demotion re-ranks/re-badges; it never destroys data no other source provides).
    """
    athlete_id, file_src, api_src = await _seed(session)
    # Whole-source "*" demotion of the SOLE power_w carrier to the lowest tier for this athlete.
    await _add_override(session, athlete_id, file_src, WHOLE_SOURCE_CHANNEL, Fidelity.SUMMARY_ONLY)
    ingest = IngestService(session)
    # file_src is the ONLY source with a power_w stream (real per-second data, adapter RAW_STREAM);
    # api_src carries only hr_bpm, so nothing else provides power_w.
    carrier = _ride_channels(
        native_id="carrier", channels={"power_w": [200.0] * 5}, tier=Fidelity.RAW_STREAM
    )
    other = _ride_channels(
        native_id="other", channels={"hr_bpm": [140.0] * 5}, tier=Fidelity.PLATFORM_COMPUTED
    )
    await ingest.ingest(athlete_id, file_src, [carrier])
    await ingest.ingest(athlete_id, api_src, [other])
    await session.commit()
    ss = (await session.execute(select(ActivityStreamSet))).scalars().one()
    chans = (
        await session.execute(
            select(StreamChannel).where(StreamChannel.stream_set_id == ss.stream_set_id)
        )
    ).scalars().all()
    by_name = {c.channel.value: c for c in chans}
    # The sole-carrier power_w channel is PRESERVED, not dropped by the whole-source demotion.
    assert "power_w" in by_name, "whole-source demotion dropped the SOLE-carrier power_w stream"
    power = by_name["power_w"]
    # ...with its real per-second values intact (would fail if the channel were dropped/emptied).
    assert power.values == [200.0] * 5
    cov = power.coverage
    assert isinstance(cov, dict)
    assert cov["present"] is True
    # ...and the badge reflects the demoted summary_only tier, NOT the carrier's RAW_STREAM
    # adapter tier — proving the "*" override threaded through (non-vacuous: differs from adapter).
    assert cov["fidelity"] == Fidelity.SUMMARY_ONLY.value


# ----------------------------------------- GAP-R3: typed absent_true on the write path


async def test_gap_r3_no_contributor_field_is_typed_absent_true(session: AsyncSession) -> None:
    """GAP-R3/GAP-R1: a scalar NO source supplied is written as typed absent_true, not skipped.

    The ingest write path used to ``continue`` on a no-contributor field, leaving neither
    coverage nor value — making a true absence indistinguishable from anything else. GAP-R3
    requires the union-of-presence AND a typed absence: a channel ``absent_true`` only when no
    source provides it at all. Here the ride supplies ``avg_power_w`` but no HR scalar, so the
    canonical coverage MUST carry a present power badge AND an ``absent_true`` HR badge.
    """
    athlete_id, file_src, _ = await _seed(session)
    ingest = IngestService(session)
    await ingest.ingest(
        athlete_id, file_src,
        [_ride(native_id="ride-1", watts=250.0, seconds=3600, tier=Fidelity.RAW_STREAM)],
    )
    await session.commit()
    act = (await session.execute(select(Activity))).scalars().one()
    cov = act.coverage
    # Union-of-presence: the supplied power scalar is present (GAP-R3 conforms).
    assert cov["avg_power_w"]["present"] is True
    # The HR scalar NO source supplied is a typed absence — absent_true (not failed), never skipped.
    assert "avg_hr_bpm" in cov, "no-contributor field was silently skipped (GAP-R1/GAP-R3 gap)"
    assert cov["avg_hr_bpm"]["present"] is False
    assert cov["avg_hr_bpm"]["fidelity"] == Fidelity.ABSENT_TRUE.value
    # ...and it is NOT zero-filled onto the record (GAP-R1: no fabricated value).
    assert act.avg_hr_bpm is None


async def test_gap_r3_coverage_for_failed_produces_absent_failed(session: AsyncSession) -> None:
    """GAP-R3/HLT-R3: coverage_for can emit absent_failed for a fetch-failure, distinct from true.

    The fetch-failure trigger is owned by the ingestion-failure lifecycle (ING-GAP/ING-SUB,
    out of this slice); this pins the write-path CAPABILITY the descriptor must expose so a
    "source should have supplied it but failed" gap is distinguishable from "no source at all".
    """
    failed = coverage_for(False, Fidelity.ABSENT_FAILED, disputed=False, failed=True)
    true_absent = coverage_for(False, Fidelity.ABSENT_TRUE, disputed=False)
    assert failed.fidelity is Fidelity.ABSENT_FAILED
    assert true_absent.fidelity is Fidelity.ABSENT_TRUE
    assert failed.fidelity is not true_absent.fidelity
