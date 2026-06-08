"""CON-R4 cross-source conflict-resolution proof suite (doc 80 CON-R4).

Proves, through the REAL :class:`~wattwise_core.ingestion.ingest.IngestService` write
path against the canonical store, that when overlapping sources supply the SAME
canonical fact:

* the documented total order (trust-tier → confidence → recency → completeness →
  stable tiebreak, doc 20 §6 CONF-R2) selects the winner;
* identity resolution collapses the same real session from N sources to ONE canonical
  activity (doc 20 §4.3 MAP-R9..R12); and
* lineage records the conflict + the winning/losing candidates (doc 30 PRV-R3..R6,
  doc 20 §5 LIN-R3) — reconstructed from the candidate store, since the canonical
  Activity/coverage records fidelity + ``disputed`` but NEVER a source identity
  (doc 20 §6 CONF-R2 / GBO-AC-3).

CROSS-SOURCE PROVENANCE NOTE. The canonical ``Activity``/``coverage`` deliberately carry
no source identity (see ``coverage_for`` — only ``present``/``fidelity``/``disputed``).
So "which source contributed the winning value" is NOT readable from the canonical
record; it is RECONSTRUCTED from the tier-2 candidate store by re-running the SAME pure
resolver the ingest path uses (``field_candidates`` → ``resolve_field``) over the live
(non-superseded) ``SourceCandidate`` rows whose ``resolved_activity_id`` points at the
canonical activity, then reading ``ResolvedField.winning_source_descriptor_id``. Each
scenario asserts the canonical value EQUALS that reconstructed winner's payload value,
proving the canonical write and the lineage agree.
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
from wattwise_core.domain.enums import Fidelity, GboType
from wattwise_core.ingestion._canonical import field_candidates
from wattwise_core.ingestion.dedup import ResolvedField, resolve_field
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.ingestion.trust import load_trust_policy
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
_FETCHED = _dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- fixtures


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A session over a fresh in-memory canonical schema (SQLite-only, no network)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _fresh_session() -> tuple[AsyncSession, object]:
    """A second independent in-memory store (for the order-independence scenario)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return factory(), engine


async def _seed(session: AsyncSession) -> tuple[str, str, str]:
    """Seed the athlete, the cycling sport, and TWO source descriptors (file + api)."""
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


async def _set_profile(
    session: AsyncSession, descriptor_id: str, profile: dict[str, object]
) -> None:
    """Declare a descriptor's per-channel ``trust_profile`` (LIN-R1 configuration data).

    This is the documented per-channel trust mechanism (CONF-R3 / PRV-R7): it lets
    DIFFERENT fields of the same activity win from DIFFERENT sources without any code
    change. The ingest path's ``TrustPolicy`` reads exactly this.
    """
    desc = await session.get(SourceDescriptor, uuid.UUID(descriptor_id))
    assert desc is not None
    desc.trust_profile = profile
    await session.commit()


# --------------------------------------------------------- candidate construction helpers


def _ride(
    *,
    native_id: str,
    scalars: dict[str, float],
    seconds: int = 3600,
    tier: Fidelity,
    start: _dt.datetime = _START,
    confidence: float = 1.0,
    observed_at: _dt.datetime | None = None,
    hash_salt: str = "",
) -> GboCandidate:
    """An activity candidate carrying an arbitrary set of canonical scalar fields.

    The ``content_hash`` folds every input so two distinct candidates never collide on
    the candidate-key uniqueness constraint.
    """
    payload: dict[str, object] = {
        "start_time": start,
        "sport": "cycling",
        "elapsed_time_s": seconds,
        "moving_time_s": seconds,
        **scalars,
    }
    digest = content_hash(
        f"{native_id}:{sorted(scalars.items())}:{seconds}:{start.isoformat()}:{hash_salt}".encode()
    )
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=digest,
        payload=payload,
        trust_tier=tier,
        confidence=confidence,
        observed_at=observed_at,
        fetched_at=_FETCHED,
    )


# ---------------------------------------------------------- lineage-reconstruction helpers


async def _the_activity(session: AsyncSession) -> Activity:
    """The single canonical activity (asserts the single-count invariant en route)."""
    return (await session.execute(select(Activity))).scalars().one()


async def _live_candidates(
    session: AsyncSession, activity_id: uuid.UUID
) -> list[SourceCandidate]:
    """The live (non-superseded) activity candidates resolved to ``activity_id``.

    This IS the cross-source lineage the resolver consumes: every retained per-source
    observation that contributed to the canonical activity (doc 20 §5 LIN-R3).
    """
    stmt = select(SourceCandidate).where(
        SourceCandidate.gbo_type == GboType.ACTIVITY,
        SourceCandidate.resolved_activity_id == activity_id,
        SourceCandidate.is_superseded.is_(False),
    )
    return list((await session.execute(stmt)).scalars().all())


async def _provenance(
    session: AsyncSession, athlete_id: str, activity_id: uuid.UUID, field_name: str
) -> ResolvedField:
    """Reconstruct a field's cross-source winner from the candidate store.

    Re-runs the SAME pure total order the ingest write path applies (``field_candidates``
    → ``resolve_field``) over the live candidates resolved to ``activity_id``, under the
    EXACT effective per-channel trust the ingest path used (the real
    :class:`~wattwise_core.ingestion.trust.TrustPolicy` — per-athlete override → descriptor
    profile → adapter tier, PRV-R7). The returned :class:`ResolvedField` names the winning
    source (``winning_source_descriptor_id``) and ALL considered sources — the provenance
    the canonical Activity/coverage deliberately omits (GBO-AC-3).
    """
    candidates = await _live_candidates(session, activity_id)
    policy = await load_trust_policy(session, uuid.UUID(athlete_id), candidates)
    contributors = field_candidates(
        candidates, field_name, lambda c: policy.tier(c, field_name)
    )
    resolved = resolve_field(contributors, dispute_tolerance=0.05)
    assert resolved is not None
    return resolved


async def _candidate_by_source(
    session: AsyncSession, activity_id: uuid.UUID, descriptor_id: str
) -> SourceCandidate:
    """The live candidate contributed by ``descriptor_id`` (its retained observation)."""
    for c in await _live_candidates(session, activity_id):
        if str(c.source_descriptor_id) == descriptor_id:
            return c
    raise AssertionError(f"no live candidate for source {descriptor_id}")


def _coverage_carries_no_source_identity(coverage: dict[str, object]) -> None:
    """Assert the canonical coverage holds fidelity/disputed but NO source identity."""
    for entry in coverage.values():
        assert isinstance(entry, dict)
        keys = set(entry)
        assert "fidelity" in keys
        assert {"source_descriptor_id", "source_native_id", "source", "source_key"} & keys == set()


# =========================================================================== scenario 1
# Single canonical identity across sources (MAP-R9..R12).


async def test_two_sources_same_session_collapse_to_one_activity(session: AsyncSession) -> None:
    """Overlapping start (±120s) + duration tol + compatible sport → ONE activity.

    Asserts (a) exactly one canonical ``activity`` row and (b) BOTH source candidates
    carry ``resolved_activity_id`` pointing at it (the lineage back-pointer, LIN-R3).
    """
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    file_cand = _ride(
        native_id="file-1", scalars={"avg_power_w": 250.0}, seconds=3600,
        tier=Fidelity.RAW_STREAM,
    )
    # Same real session from the API source: start drifts +90s, duration +5s — within tol.
    api_cand = _ride(
        native_id="api-1", scalars={"avg_power_w": 251.0}, seconds=3605,
        tier=Fidelity.PLATFORM_COMPUTED, start=_START + _dt.timedelta(seconds=90),
    )
    await ingest.ingest(athlete_id, file_src, [file_cand])
    await ingest.ingest(athlete_id, api_src, [api_cand])
    await session.commit()

    act = await _the_activity(session)  # exactly ONE (single-count, DEDUP-R1)
    candidates = await _live_candidates(session, act.activity_id)
    # Both sources' candidates point at the single canonical activity (LIN-R3 lineage).
    assert {str(c.source_descriptor_id) for c in candidates} == {file_src, api_src}
    assert all(c.resolved_activity_id == act.activity_id for c in candidates)


# =========================================================================== scenario 2
# Field-level cross-source provenance (CONF-R3 / LIN-R3).


async def test_different_fields_win_from_different_sources(session: AsyncSession) -> None:
    """Different scalar fields win from DIFFERENT sources for ONE canonical activity.

    The documented per-channel trust mechanism (CONF-R3 / PRV-R7): source A is the
    declared authority for ``avg_power_w``, source B for ``avg_hr_bpm`` +
    ``avg_cadence_rpm`` (each via its descriptor ``trust_profile``). Both report all
    three fields for the SAME session, but per-channel trust splits the winners — power
    resolves from A, HR + cadence from B.

    PROVENANCE CAVEAT (PSI-3): the ``winning_source_descriptor_id`` here is RE-DERIVED by
    re-running the SAME resolver (``field_candidates`` → ``resolve_field``) the ingest path
    used, so it is a DETERMINISM / CONSISTENCY cross-check (the reconstruction agrees with
    itself), NOT an independent provenance proof. What makes the source attribution real is
    the ADJACENT literal value-check: each competing field carries PAIRWISE-DISTINCT values
    across the two sources (asserted below), so the canonical value provably came from the
    winning source's payload and could not have come from the loser's. NO source identity is
    asserted to leak onto the Activity/coverage (CON-R2 / GBO-AC-3).
    """
    athlete_id, src_a, src_b = await _seed(session)
    # A is the power authority; B is the HR + cadence authority (declared per channel).
    await _set_profile(
        session, src_a,
        {"avg_power_w": Fidelity.RAW_STREAM.value,
         "avg_hr_bpm": Fidelity.SUMMARY_ONLY.value,
         "avg_cadence_rpm": Fidelity.SUMMARY_ONLY.value},
    )
    await _set_profile(
        session, src_b,
        {"avg_power_w": Fidelity.SUMMARY_ONLY.value,
         "avg_hr_bpm": Fidelity.RAW_STREAM.value,
         "avg_cadence_rpm": Fidelity.RAW_STREAM.value},
    )
    ingest = IngestService(session)
    cand_a = _ride(
        native_id="a-1",
        scalars={"avg_power_w": 250.0, "avg_hr_bpm": 130.0, "avg_cadence_rpm": 80.0},
        tier=Fidelity.PLATFORM_COMPUTED,
    )
    # B: same session (identical start/duration), different values for all three fields.
    cand_b = _ride(
        native_id="b-1",
        scalars={"avg_power_w": 999.0, "avg_hr_bpm": 152.0, "avg_cadence_rpm": 92.0},
        tier=Fidelity.PLATFORM_COMPUTED,
        hash_salt="b",
    )
    # PSI-3: the competing per-field VALUES are pairwise-DISTINCT across the two sources,
    # so the adjacent literal value-check below provably identifies the winning source (a
    # canonical value equal to A's but not B's could ONLY have come from A, and vice versa).
    for fname in ("avg_power_w", "avg_hr_bpm", "avg_cadence_rpm"):
        assert cand_a.payload[fname] != cand_b.payload[fname]

    await ingest.ingest(athlete_id, src_a, [cand_a])
    await ingest.ingest(athlete_id, src_b, [cand_b])
    await session.commit()

    act = await _the_activity(session)

    # avg_power_w: source A wins per channel; canonical equals A's payload value.
    power = await _provenance(session, athlete_id, act.activity_id, "avg_power_w")
    assert power.winning_source_descriptor_id == src_a
    assert float(act.avg_power_w) == pytest.approx(250.0)

    # avg_hr_bpm + avg_cadence_rpm: source B wins per channel — winner read from the
    # candidate store, canonical value matches B's payload (cross-source field split).
    hr = await _provenance(session, athlete_id, act.activity_id, "avg_hr_bpm")
    assert hr.winning_source_descriptor_id == src_b
    assert int(act.avg_hr_bpm) == 152

    cad = await _provenance(session, athlete_id, act.activity_id, "avg_cadence_rpm")
    assert cad.winning_source_descriptor_id == src_b
    assert int(act.avg_cadence_rpm) == 92

    # Each winner's value is reconstructable from its OWN candidate payload (LIN-R3).
    a_cand = await _candidate_by_source(session, act.activity_id, src_a)
    b_cand = await _candidate_by_source(session, act.activity_id, src_b)
    assert a_cand.payload["avg_power_w"] == 250.0
    assert b_cand.payload["avg_hr_bpm"] == 152.0 and b_cand.payload["avg_cadence_rpm"] == 92.0
    # CON-R2 / GBO-AC-3: the canonical coverage carries fidelity, never a source identity.
    _coverage_carries_no_source_identity(act.coverage)


# =========================================================================== scenario 3
# Recency tiebreak end-to-end (CONF-R2 step 3) + confidence-decides-first companion.


async def test_recency_breaks_tie_through_ingest(session: AsyncSession) -> None:
    """Same tier + same confidence + different observed_at → MORE RECENT wins end-to-end.

    Both sources are PLATFORM_COMPUTED with confidence 1.0, so the resolver reaches step
    3 (recency). The later ``observed_at`` value wins through the REAL ingest path, and
    the candidate store confirms the winning source is the more-recent observation.
    """
    athlete_id, src_a, src_b = await _seed(session)
    ingest = IngestService(session)
    older = _ride(
        native_id="old-1", scalars={"avg_power_w": 200.0}, tier=Fidelity.PLATFORM_COMPUTED,
        confidence=1.0, observed_at=_dt.datetime(2026, 6, 1, 7, 0, tzinfo=UTC),
    )
    newer = _ride(
        native_id="new-1", scalars={"avg_power_w": 240.0}, tier=Fidelity.PLATFORM_COMPUTED,
        confidence=1.0, observed_at=_dt.datetime(2026, 6, 1, 10, 0, tzinfo=UTC), hash_salt="n",
    )
    await ingest.ingest(athlete_id, src_a, [older])
    await ingest.ingest(athlete_id, src_b, [newer])
    await session.commit()

    act = await _the_activity(session)
    # Recency decides: the later observation (240W from src_b) wins through ingest.
    assert float(act.avg_power_w) == pytest.approx(240.0)
    prov = await _provenance(session, athlete_id, act.activity_id, "avg_power_w")
    assert prov.winning_source_descriptor_id == src_b


async def test_confidence_decides_before_recency(session: AsyncSession) -> None:
    """COMPANION: confidence (step 2) decides BEFORE recency (step 3).

    Same tier; the OLDER observation carries HIGHER confidence, so it must win over a
    newer-but-less-confident value — proving recency is consulted only AFTER confidence.
    """
    athlete_id, src_a, src_b = await _seed(session)
    ingest = IngestService(session)
    confident_old = _ride(
        native_id="old-1", scalars={"avg_power_w": 200.0}, tier=Fidelity.PLATFORM_COMPUTED,
        confidence=0.95, observed_at=_dt.datetime(2026, 6, 1, 7, 0, tzinfo=UTC),
    )
    unsure_new = _ride(
        native_id="new-1", scalars={"avg_power_w": 240.0}, tier=Fidelity.PLATFORM_COMPUTED,
        confidence=0.40, observed_at=_dt.datetime(2026, 6, 1, 10, 0, tzinfo=UTC), hash_salt="n",
    )
    await ingest.ingest(athlete_id, src_a, [confident_old])
    await ingest.ingest(athlete_id, src_b, [unsure_new])
    await session.commit()

    act = await _the_activity(session)
    # Confidence wins despite the LOSING side being more recent (200W, src_a).
    assert float(act.avg_power_w) == pytest.approx(200.0)
    prov = await _provenance(session, athlete_id, act.activity_id, "avg_power_w")
    assert prov.winning_source_descriptor_id == src_a


# =========================================================================== scenario 4
# Ingest-order independence / determinism (CONF-R4 / GBO-AC-1).


async def _ingest_pair_in_order(
    first_native: str, second_native: str
) -> tuple[float, bool, int]:
    """Ingest two conflicting sources in a given order into a FRESH store; report outcome.

    Returns ``(winning_power, disputed_flag, activity_count)`` so two orders can be
    compared for byte-stable identity of outcome.
    """
    s, engine = await _fresh_session()
    try:
        athlete_id, file_src, api_src = await _seed(s)
        by_src = {
            "file": (file_src, _ride(
                native_id="file-1", scalars={"avg_power_w": 200.0}, tier=Fidelity.RAW_STREAM,
            )),
            "api": (api_src, _ride(
                native_id="api-1", scalars={"avg_power_w": 320.0},
                tier=Fidelity.PLATFORM_COMPUTED, hash_salt="api",
            )),
        }
        ingest = IngestService(s)
        for key in (first_native, second_native):
            descriptor, cand = by_src[key]
            await ingest.ingest(athlete_id, descriptor, [cand])
        await s.commit()
        act = (await s.execute(select(Activity))).scalars().one()
        count = len((await s.execute(select(Activity))).scalars().all())
        coverage = act.coverage["avg_power_w"]
        assert isinstance(coverage, dict)
        return float(act.avg_power_w), bool(coverage["disputed"]), count
    finally:
        await s.close()
        await engine.dispose()  # type: ignore[attr-defined]


async def test_ingest_order_independence_is_byte_stable() -> None:
    """A→B and B→A (two fresh stores) yield IDENTICAL canonical outcome (CONF-R4).

    Same winner, same disputed flag, same single-count — order of arrival must not
    change the resolved fact (the determinism guarantee, GBO-AC-1).
    """
    ab = await _ingest_pair_in_order("file", "api")
    ba = await _ingest_pair_in_order("api", "file")
    assert ab == ba
    winner, disputed, count = ab
    assert winner == pytest.approx(200.0)  # RAW_STREAM file beats PLATFORM api either way
    assert disputed is True  # 200 vs 320 disagree beyond tolerance regardless of order
    assert count == 1


# =========================================================================== scenario 5
# Identity: fingerprint vs fuzzy (MAP-R10).


async def test_shared_native_id_outside_window_stays_separate(session: AsyncSession) -> None:
    """NEGATIVE (D1): a shared ``source_native_id`` OUTSIDE ±2h does NOT cross-source merge.

    ``source_native_id`` is the PER-SOURCE dedup key, NOT a cross-source identity: two
    DIFFERENT sources that happen to collide on the same native id (e.g. two stripped FITs
    yielding a degenerate ``garmin|||`` file_id, or two unrelated sessions reusing an id)
    must NOT be merged on that token. Here two different sources share the SAME native id
    but start 6h apart — far OUTSIDE the ±2h identity window — so the conservative
    windowed-only resolver (DEDUP-R7) keeps them as TWO separate canonical activities. A
    genuine cross-window strong-fingerprint match (MAP-R10) is DEFERRED and would require a
    TYPED ``strong_fingerprint`` distinct from ``source_native_id`` — never the dedup key.
    """
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    first = _ride(
        native_id="device-uuid-xyz", scalars={"avg_power_w": 200.0}, tier=Fidelity.RAW_STREAM,
    )
    far = _ride(
        native_id="device-uuid-xyz", scalars={"avg_power_w": 210.0},
        tier=Fidelity.PLATFORM_COMPUTED, start=_START + _dt.timedelta(hours=6), hash_salt="far",
    )
    await ingest.ingest(athlete_id, file_src, [first])
    await ingest.ingest(athlete_id, api_src, [far])
    await session.commit()

    # TWO separate canonical activities — no false cross-source merge on the native id.
    activities = (await session.execute(select(Activity))).scalars().all()
    assert len(activities) == 2
    # Each activity carries exactly ONE source's candidate (the real workouts are intact).
    counts: list[int] = []
    for a in activities:
        counts.append(len(await _live_candidates(session, a.activity_id)))
    assert sorted(counts) == [1, 1]


async def test_fuzzy_match_inside_window_collapses_outside_stays_separate(
    session: AsyncSession,
) -> None:
    """Distinct fingerprints: fuzzy match WITHIN the window collapses; OUTSIDE stays two.

    All three rides have DIFFERENT ``source_native_id``s (no fingerprint shortcut). Two
    start 90s apart (within ±120s) and collapse to one activity; the third starts 10 min
    apart (outside the window) and stays a SEPARATE canonical activity (MAP-R10
    conservatism, DEDUP-R7).
    """
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    base = _ride(native_id="f-1", scalars={"avg_power_w": 200.0}, tier=Fidelity.RAW_STREAM)
    near = _ride(
        native_id="a-1", scalars={"avg_power_w": 205.0}, tier=Fidelity.PLATFORM_COMPUTED,
        start=_START + _dt.timedelta(seconds=90), hash_salt="near",
    )
    far = _ride(
        native_id="a-2", scalars={"avg_power_w": 210.0}, tier=Fidelity.PLATFORM_COMPUTED,
        start=_START + _dt.timedelta(minutes=10), hash_salt="far",
    )
    await ingest.ingest(athlete_id, file_src, [base])
    await ingest.ingest(athlete_id, api_src, [near])  # within window → collapses with base
    await ingest.ingest(athlete_id, api_src, [far])  # 10 min apart → separate activity
    await session.commit()

    activities = (await session.execute(select(Activity))).scalars().all()
    # Exactly TWO canonical activities: {base+near} collapsed, {far} separate.
    assert len(activities) == 2
    # The collapsed activity is the one carrying BOTH sources' candidates.
    counts: list[int] = []
    for a in activities:
        counts.append(len(await _live_candidates(session, a.activity_id)))
    assert sorted(counts) == [1, 2]


# =========================================================================== scenario 6
# Disputed lineage (CONF-R5).


async def test_disputed_field_keeps_winner_and_retains_both_candidates(
    session: AsyncSession,
) -> None:
    """Material disagreement → disputed=True, a usable winner, BOTH values retained.

    Two sources disagree on power far beyond tolerance (200 vs 320). The canonical field
    is flagged ``disputed: true``, a usable winner is still chosen (never blanked or
    averaged), and BOTH candidate values remain in the candidate store — the losing
    candidate is explicitly recoverable for explanation (CONF-R5 / LIN-R3).
    """
    athlete_id, file_src, api_src = await _seed(session)
    ingest = IngestService(session)
    await ingest.ingest(
        athlete_id, file_src,
        [_ride(native_id="file-1", scalars={"avg_power_w": 200.0}, tier=Fidelity.RAW_STREAM)],
    )
    await ingest.ingest(
        athlete_id, api_src,
        [_ride(
            native_id="api-1", scalars={"avg_power_w": 320.0},
            tier=Fidelity.PLATFORM_COMPUTED, hash_salt="api",
        )],
    )
    await session.commit()

    act = await _the_activity(session)
    coverage = act.coverage["avg_power_w"]
    assert isinstance(coverage, dict)
    # Disagreement surfaced, not hidden.
    assert coverage["disputed"] is True
    # A usable winner is still chosen — the highest-trust value, never blanked/averaged.
    assert act.avg_power_w is not None
    assert float(act.avg_power_w) == pytest.approx(200.0)

    # The lineage retains BOTH the winning AND the losing candidate values (explainable).
    prov = await _provenance(session, athlete_id, act.activity_id, "avg_power_w")
    assert prov.disputed is True
    assert prov.winning_source_descriptor_id == file_src
    assert set(prov.considered_source_ids) == {file_src, api_src}
    file_cand = await _candidate_by_source(session, act.activity_id, file_src)
    api_cand = await _candidate_by_source(session, act.activity_id, api_src)
    assert file_cand.payload["avg_power_w"] == 200.0  # winner retained
    assert api_cand.payload["avg_power_w"] == 320.0  # loser retained (not discarded)
