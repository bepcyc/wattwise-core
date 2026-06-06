"""End-to-end ingest -> canonical -> analytics journey (E2E-R1: connect->sync->PMC).

Exercises the ingestion write path (candidates -> identity resolution -> resolved
canonical) and the analytics service over the resulting canonical store, on the
portable substrate. Proves single-count (DEDUP-R1), idempotency (UPS-R3), and that
PMC reads the canonical activities the ingest path wrote.
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
    Athlete,
    Base,
    FitnessSignature,
    SourceDescriptor,
    Sport,
)
from wattwise_core.storage import content_hash

UTC = _dt.UTC


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


async def _seed(session: AsyncSession) -> tuple[str, str]:
    """Seed the single athlete, the cycling sport, an FTP signature, and a source."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    descriptor = SourceDescriptor(
        source_key="file_import", display_name="Activity files", kind="file_upload"
    )
    session.add(descriptor)
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
    return str(athlete.athlete_id), str(descriptor.source_descriptor_id)


def _ride_candidate(*, native_id: str, watts: float, seconds: int, tier: Fidelity) -> GboCandidate:
    """A constant-power cycling activity candidate with a 1 Hz power stream."""
    payload = {
        "start_time": _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
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


@pytest.mark.integration
async def test_connect_sync_pmc_journey(session: AsyncSession) -> None:
    """A synced ride lands one canonical activity whose TSS feeds the PMC series."""
    athlete_id, descriptor_id = await _seed(session)
    ingest = IngestService(session)
    cand = _ride_candidate(native_id="ride-1", watts=250.0, seconds=3600, tier=Fidelity.RAW_STREAM)
    result = await ingest.ingest(athlete_id, descriptor_id, [cand])
    await session.commit()
    assert len(result.activities_written) == 1

    svc = AnalyticsService(session)
    activity_id = next(iter(result.activities_written))
    bundle = await svc.coggan(activity_id)
    assert is_computed(bundle)
    assert is_computed(bundle.value.tss)
    assert bundle.value.tss.value == pytest.approx(100.0, abs=1e-3)

    pmc = await svc.pmc(athlete_id, _dt.date(2026, 6, 1), _dt.date(2026, 6, 3))
    assert len(pmc) == 3
    assert is_computed(pmc[0])
    assert pmc[0].value.ctl > 0


@pytest.mark.integration
async def test_resync_is_idempotent_single_count(session: AsyncSession) -> None:
    """Re-syncing the same ride does not create a second activity (UPS-R3, DEDUP-R1)."""
    athlete_id, descriptor_id = await _seed(session)
    ingest = IngestService(session)
    cand = _ride_candidate(native_id="ride-1", watts=200.0, seconds=1800, tier=Fidelity.RAW_STREAM)
    await ingest.ingest(athlete_id, descriptor_id, [cand])
    await ingest.ingest(athlete_id, descriptor_id, [cand])  # same content again
    await session.commit()
    count = len((await session.execute(select(Activity))).scalars().all())
    assert count == 1


@pytest.mark.integration
async def test_two_sources_same_session_resolve_to_one_activity(session: AsyncSession) -> None:
    """An uploaded file and a second source for the same ride resolve to ONE activity (DEDUP-R1)."""
    athlete_id, descriptor_id = await _seed(session)
    # A second registered source (a different api-key source) for the same real ride.
    other = SourceDescriptor(source_key="other_src", display_name="Other", kind="oauth_api")
    session.add(other)
    await session.flush()
    ingest = IngestService(session)
    file_cand = _ride_candidate(
        native_id="file-abc", watts=250.0, seconds=3600, tier=Fidelity.RAW_STREAM
    )
    api_cand = _ride_candidate(
        native_id="api-xyz", watts=251.0, seconds=3605, tier=Fidelity.PLATFORM_COMPUTED
    )
    await ingest.ingest(athlete_id, descriptor_id, [file_cand])
    await ingest.ingest(athlete_id, str(other.source_descriptor_id), [api_cand])
    await session.commit()
    activities = (await session.execute(select(Activity))).scalars().all()
    assert len(activities) == 1  # single-count: one real session -> one canonical activity
    # The higher-trust (raw_stream file) avg_power wins the field resolution (CONF-R2).
    assert float(activities[0].avg_power_w) == pytest.approx(250.0)

