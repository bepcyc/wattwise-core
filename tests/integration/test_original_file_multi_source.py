"""Multiple original files per session resolve onto ONE canonical activity (RAW-T-R2(c)).

The same real-world ride observed by two sources — each carrying its OWN verbatim
original recording file — must collapse to a single canonical activity through the
trust/fidelity conflict policy, while BOTH originals are retained with their own
provenance (tier-1 ``activity_file`` rows + object-store bytes). The retained bytes
round-trip (the stored object's hash equals the recorded ``content_hash``) and the
relational store never holds the bytes — only the typed reference (RAW-T-R2(a)/(b)).
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
from wattwise_core.ingestion.ingest import IngestService, OriginalFile
from wattwise_core.persistence.models import (
    Activity,
    ActivityFile,
    Athlete,
    Base,
    SourceDescriptor,
    Sport,
)
from wattwise_core.storage import LocalObjectStore, content_hash

pytestmark = pytest.mark.integration

UTC = _dt.UTC
_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
_FIT_BYTES = b"verbatim-fit-bytes-from-the-head-unit" * 8
_GPX_BYTES = b"<gpx>verbatim-gpx-export-of-the-same-ride</gpx>" * 8


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A session over a fresh in-memory schema (single-connection resolve math)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    await engine.dispose()


def _ride_candidate(*, native_id: str, watts: float, tier: Fidelity) -> GboCandidate:
    """A same-ride candidate (shared start/sport/duration -> one canonical identity)."""
    payload = {
        "start_time": _START,
        "sport": "cycling",
        "elapsed_time_s": 3600,
        "moving_time_s": 3600,
        "avg_power_w": watts,
    }
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(f"{native_id}:{watts}".encode()),
        payload=payload,
        trust_tier=tier,
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )


async def test_two_source_originals_one_canonical_activity(
    session: AsyncSession, tmp_path: Path
) -> None:
    """Two originals from two sources for the SAME session: one canonical activity,
    TWO retained tier-1 files each with its own provenance, bytes round-tripping, and
    the contested field resolved by trust — the file never bypasses the policy."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    fit_src = SourceDescriptor(source_key="src_fit", display_name="FIT", kind="file_upload")
    gpx_src = SourceDescriptor(source_key="src_gpx", display_name="GPX", kind="oauth_api")
    session.add_all([fit_src, gpx_src])
    await session.flush()
    athlete_id = str(athlete.athlete_id)

    store = LocalObjectStore(tmp_path)
    ingest = IngestService(session, object_store=store)
    await ingest.ingest(
        athlete_id,
        str(fit_src.source_descriptor_id),
        [_ride_candidate(native_id="ride-1", watts=250.0, tier=Fidelity.RAW_STREAM)],
        original_files=[
            OriginalFile(
                data=_FIT_BYTES,
                file_format=ActivityFileFormat.FIT,
                source_native_id="ride-1",
            )
        ],
    )
    await ingest.ingest(
        athlete_id,
        str(gpx_src.source_descriptor_id),
        [_ride_candidate(native_id="ride-1-gpx", watts=247.0, tier=Fidelity.SUMMARY_ONLY)],
        original_files=[
            OriginalFile(
                data=_GPX_BYTES,
                file_format=ActivityFileFormat.GPX,
                source_native_id="ride-1-gpx",
            )
        ],
    )
    await session.commit()

    # ONE canonical activity, resolved by trust (the RAW_STREAM value wins).
    activities = (await session.execute(select(Activity))).scalars().all()
    assert len(activities) == 1
    assert float(activities[0].avg_power_w) == pytest.approx(250.0)

    # TWO retained originals, each with its OWN provenance, linked to that one activity.
    files = (await session.execute(select(ActivityFile))).scalars().all()
    assert len(files) == 2
    assert {f.activity_id for f in files} == {activities[0].activity_id}
    assert len({f.source_descriptor_id for f in files}) == 2
    by_format = {f.format: f for f in files}
    assert set(by_format) == {ActivityFileFormat.FIT, ActivityFileFormat.GPX}

    # The relational rows hold ONLY the typed reference; the bytes round-trip from the
    # object store and hash to the recorded content_hash (RAW-T-R2(a)/(b)).
    expected = ((ActivityFileFormat.FIT, _FIT_BYTES), (ActivityFileFormat.GPX, _GPX_BYTES))
    for fmt, original in expected:
        row = by_format[fmt]
        stored = store.get(row.object_ref)
        assert stored == original
        assert content_hash(stored) == row.content_hash
        assert row.byte_size == len(original)
