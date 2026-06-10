"""E3 multi-source re-resolution lifecycle proof suite.

Proves, through the REAL ingest write path against the canonical store, the doc-20
re-resolution lifecycle: disabling a source is a configuration action whose affected
records re-resolve to the next-best provider from RETAINED candidates with no
re-fetch (EVOL-R2, DM-SUB-R5, doc 30 ING-SUB-R3/R7); the policy version that produced
every resolved record is recorded (CONF-R6); a trust-profile change re-resolves
without re-fetch under the NEW recorded version (CONF-R6); the per-field resolution
record carries winner/considered candidate pointers + the deciding rule (LIN-R3);
and a source-side deletion is a typed tombstone that never cascade-deletes a
multi-source canonical record (UPS-R5).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain import equivalence as eq
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, GboType
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.ingestion.reresolve import (
    deactivate_source,
    ingest_tombstone,
    re_resolve_activity,
    reactivate_source,
)
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
    """A session over a fresh file-less canonical schema (offline, no network)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed(session: AsyncSession) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """Seed the athlete, cycling, and TWO descriptors (high-trust file + lower api)."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    file_src = SourceDescriptor(
        source_key="file_import", display_name="Files", kind="file_upload"
    )
    api_src = SourceDescriptor(source_key="other_api", display_name="Api", kind="oauth_api")
    session.add_all([file_src, api_src])
    await session.flush()
    ids = (athlete.athlete_id, file_src.source_descriptor_id, api_src.source_descriptor_id)
    await session.commit()
    return ids


def _ride(
    sid: uuid.UUID, native: str, power: float, tier: Fidelity, salt: str = ""
) -> GboCandidate:
    payload: dict[str, object] = {
        "start_time": _START,
        "sport": "cycling",
        "elapsed_time_s": 3600,
        "avg_power_w": power,
    }
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id=str(sid),
        source_native_id=native,
        content_hash=content_hash(f"{native}|{power}|{salt}".encode()),
        payload=payload,
        observed_at=_START,
        fetched_at=_START + _dt.timedelta(hours=1),
        trust_tier=tier,
    )


async def _ingest_two_sources(
    session: AsyncSession,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID]:
    """Land the same ride from two sources (file=raw 250 W beats api=platform 240 W)."""
    athlete, file_src, api_src = await _seed(session)
    svc = IngestService(session)
    await svc.ingest(athlete, file_src, [_ride(file_src, "f-1", 250.0, Fidelity.RAW_STREAM)])
    await svc.ingest(
        athlete, api_src, [_ride(api_src, "a-1", 240.0, Fidelity.PLATFORM_COMPUTED)]
    )
    act = (await session.execute(select(Activity))).scalar_one()
    assert float(act.avg_power_w) == 250.0  # raw_stream wins (CONF-R2)
    return athlete, file_src, api_src, act.activity_id


async def test_deactivate_reresolves_to_next_best_and_reactivate_restores(
    session: AsyncSession,
) -> None:
    """EVOL-R2/DM-SUB-R5: deactivating the winning source re-resolves the affected field
    to the next-best retained provider with NO re-fetch, degrades coverage honestly, and
    reactivating re-resolves UPWARD automatically — zero candidate rows deleted."""
    _athlete, file_src, _api_src, activity_id = await _ingest_two_sources(session)
    before = (await session.execute(select(SourceCandidate))).scalars().all()
    await deactivate_source(session, file_src)
    act = await session.get(Activity, activity_id)
    assert act is not None
    assert float(act.avg_power_w) == 240.0  # next-best provider now wins
    cov = act.coverage["avg_power_w"]
    assert cov["present"] is True
    assert cov["fidelity"] == Fidelity.PLATFORM_COMPUTED.value  # reduced fidelity surfaced
    after = (await session.execute(select(SourceCandidate))).scalars().all()
    assert len(after) == len(before)  # ING-SUB-R2: nothing deleted, fully reversible
    await reactivate_source(session, file_src)
    act = await session.get(Activity, activity_id)
    assert act is not None
    assert float(act.avg_power_w) == 250.0  # upward re-resolution (ING-SUB-R7)
    assert act.coverage["avg_power_w"]["fidelity"] == Fidelity.RAW_STREAM.value


async def test_policy_version_recorded_and_changes_with_trust_profile(
    session: AsyncSession,
) -> None:
    """CONF-R6: the canonical record carries the policy version that produced it, and a
    trust-profile change re-resolves from retained candidates (no re-fetch) under a NEW
    recorded version — flipping the winner per the new policy."""
    athlete, file_src, api_src, activity_id = await _ingest_two_sources(session)
    act = await session.get(Activity, activity_id)
    assert act is not None
    v1 = act.policy_version
    assert v1 is not None and v1.startswith("v1:")
    # Configuration change: demote the file source's avg_power_w below the api source.
    file_desc = await session.get(SourceDescriptor, file_src)
    assert file_desc is not None
    file_desc.trust_profile = {"avg_power_w": Fidelity.SUMMARY_ONLY.value}
    await session.commit()
    await re_resolve_activity(session, athlete, activity_id)
    act = await session.get(Activity, activity_id)
    assert act is not None
    assert float(act.avg_power_w) == 240.0  # the api source wins under the new policy
    assert act.policy_version is not None
    assert act.policy_version != v1  # the NEW policy version is recorded
    _ = api_src


async def test_field_resolution_records_winner_pointer_and_rule(
    session: AsyncSession,
) -> None:
    """LIN-R3: every resolved field persists a resolution record with the winning
    candidate pointer into source_candidate, the considered candidate set, and the
    CONF-R2 rule that decided — without copying any source envelope onto the record."""
    athlete, file_src, _api, activity_id = await _ingest_two_sources(session)
    act = await session.get(Activity, activity_id)
    assert act is not None
    record = act.field_resolution["avg_power_w"]
    assert record["rule"] == "trust_tier"  # raw_stream beat platform_computed at step 1
    winner_row = await session.get(SourceCandidate, uuid.UUID(record["winner_candidate_id"]))
    assert winner_row is not None
    assert winner_row.source_descriptor_id == file_src  # traceable to its origin
    assert len(record["considered_candidate_ids"]) == 2
    _ = athlete


async def test_tombstone_removes_contribution_but_never_cascades(
    session: AsyncSession,
) -> None:
    """UPS-R5: a source deletion is a typed tombstone candidate — the multi-source
    canonical record persists and re-resolves to the remaining source; only when the
    LAST contributor is tombstoned is the canonical record removed."""
    athlete, file_src, api_src, activity_id = await _ingest_two_sources(session)
    assert await ingest_tombstone(session, athlete, file_src, "f-1", GboType.ACTIVITY)
    act = await session.get(Activity, activity_id)
    assert act is not None  # multi-source record persists (no cascade delete)
    assert float(act.avg_power_w) == 240.0  # the surviving source's value
    tomb = (
        await session.execute(
            select(SourceCandidate).where(SourceCandidate.is_tombstone.is_(True))
        )
    ).scalars().all()
    assert len(tomb) == 1  # the deletion is itself a typed, versioned candidate
    assert await ingest_tombstone(session, athlete, api_src, "a-1", GboType.ACTIVITY)
    assert await session.get(Activity, activity_id) is None  # last contributor gone
    retained = (await session.execute(select(SourceCandidate))).scalars().all()
    assert len(retained) == 4  # 2 superseded originals + 2 tombstones, all retained


async def test_tombstone_for_unknown_object_is_a_noop(session: AsyncSession) -> None:
    """UPS-R5: a tombstone for a never-ingested native id is an idempotent no-op —
    it neither fabricates a candidate nor disturbs the canonical store."""
    athlete, file_src, _api, activity_id = await _ingest_two_sources(session)
    assert not await ingest_tombstone(session, athlete, file_src, "ghost", GboType.ACTIVITY)
    assert await session.get(Activity, activity_id) is not None


async def test_withdrawal_surfaces_substituted_coverage_and_clears_on_return(
    session: AsyncSession,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DM-SUB-R4/R5 + doc 30 ING-SUB-R4/R7: with a DECLARED equivalence class for the
    channel, withdrawing the top-tier source re-resolves to the lower member and the
    canonical coverage badges ``substituted`` with the displaced top tier recorded;
    re-connecting the source re-resolves UPWARD and clears the marker."""
    classes = tmp_path_factory.mktemp("eq") / "classes.toml"
    classes.write_text(
        "[[canonical.equivalence_class]]\n"
        'channel = "avg_power_w"\n'
        "[[canonical.equivalence_class.members]]\n"
        'metric = "direct_power"\ntier = "raw_stream"\n'
        'note = "direct meter watts"\npenalty = "none"\n'
        "[[canonical.equivalence_class.members]]\n"
        'metric = "platform_power"\ntier = "platform_computed"\n'
        'note = "vendor-estimated watts"\npenalty = "moderate"\n'
    )
    monkeypatch.setenv("WATTWISE_EQUIVALENCE_CLASSES_FILE", str(classes))
    eq._load.cache_clear()
    try:
        _athlete, file_src, _api_src, activity_id = await _ingest_two_sources(session)
        await deactivate_source(session, file_src)
        act = await session.get(Activity, activity_id)
        assert act is not None
        cov = act.coverage["avg_power_w"]
        assert cov["fidelity"] == "substituted"  # the downgrade is a coverage signal
        assert cov["substitution"] == {"class": "avg_power_w", "from_fidelity": "raw_stream"}
        assert float(act.avg_power_w) == 240.0  # next-best member, never a blank/zero
        await reactivate_source(session, file_src)
        act = await session.get(Activity, activity_id)
        assert act is not None
        cov = act.coverage["avg_power_w"]
        assert cov["fidelity"] == "raw_stream"  # upward re-resolution...
        assert cov["substitution"] is None  # ...clears the substitution marker
    finally:
        eq._load.cache_clear()
