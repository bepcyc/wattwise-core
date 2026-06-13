"""Integration tests for the PRODUCTION agent runtime (``wattwise_core.agent.engine``).

These exercise the deployable :class:`GraphAgentEngine` and its concrete services against
a real canonical store — the assembly the API drives — rather than the in-flight graph
seams the unit tests cover. They pin three things the built-stack agent depends on, none
of which the model-key-less smoke could reach:

- the compiled LangGraph is invoked through the deliverables' ``CoachGraph.run`` seam (the
  :class:`_CompiledCoachGraph` adapter supplies the durable-thread config ``ainvoke``
  requires) and produces a terminal answer (GRAPH-R1 / CKPT-R3);
- a NUMBER claim grounds VERBATIM against the canonical analytic via the resolved-ahead
  snapshot path (GROUND-R7) — the deployed grounder can confirm numbers, not only scrub
  them;
- a model-fabricated number can never ship: it scrubs out and the run degrades (GROUND-R6).
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.capabilities import CanonicalEvidence
from wattwise_core.agent.contracts import ClaimKind, GroundDecision, RunStatus
from wattwise_core.agent.deliverables import (
    HRV_UNAVAILABLE_CLAUSE,
    Readiness,
    _ReadinessNarration,
    first_sentence,
    leads_with_state,
    readiness_assessment,
)
from wattwise_core.agent.engine import (
    ClaimGrounder,
    GraphAgentEngine,
    _ClaimSchema,
    _ExtractedClaim,
    _PlanSchema,
    build_agent_engine,
)
from wattwise_core.agent.engine_services import CoachBundle
from wattwise_core.agent.locale import LocalePolicy
from wattwise_core.agent.model import FakeModel
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.config import load_settings
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, HrvMethod, ReadinessVerdict, SignatureOrigin
from wattwise_core.identity import OWNER_ATHLETE_ID, OWNER_SUBJECT
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.persistence.models import (
    Athlete,
    Base,
    DailyWellness,
    FitnessSignature,
    SourceDescriptor,
    Sport,
)
from wattwise_core.persistence.types import utcnow
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.integration

UTC = _dt.UTC
_RIDE_DAYS = (_dt.date(2026, 6, 1), _dt.date(2026, 6, 2), _dt.date(2026, 6, 3))


class _DatabaseStub:
    """A minimal :class:`~wattwise_core.persistence.Database` substitute over one engine.

    Exposes only the ``session()`` async-context the engine reads through. It intentionally
    does NOT replicate the real engine's commit-on-success / rollback-on-error semantics —
    valid here ONLY because :class:`GraphAgentEngine` is READ-ONLY (the agent run writes no
    canonical rows; agent state lives in the in-memory checkpointer). It must not be reused
    for a write-path engine, where the missing commit/rollback would mask a real defect.
    """

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    def session(self) -> _SessionCtx:
        return _SessionCtx(self._factory)


class _SessionCtx:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        self._session = self._factory()
        return self._session

    async def __aexit__(self, *exc: object) -> None:
        assert self._session is not None
        await self._session.close()


@pytest_asyncio.fixture
async def seeded() -> AsyncIterator[tuple[AnalyticsService, _DatabaseStub, AsyncSession]]:
    """A canonical store seeded with the owner + an FTP signature + three 100-TSS rides."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        session.add(Athlete(athlete_id=OWNER_ATHLETE_ID, sex="male", reference_timezone="UTC"))
        descriptor = SourceDescriptor(
            source_key="file_import", display_name="Activity files", kind="file_upload"
        )
        session.add(descriptor)
        session.add(
            FitnessSignature(
                athlete_id=OWNER_ATHLETE_ID,
                signature_type="cycling",
                effective_date=_dt.date(2024, 1, 1),
                ftp_w=250.0,
                origin=SignatureOrigin.MEASURED,
            )
        )
        await session.flush()
        ingest = IngestService(session)
        for i, day in enumerate(_RIDE_DAYS):
            await ingest.ingest(
                str(OWNER_ATHLETE_ID), str(descriptor.source_descriptor_id), [_ride(f"r{i}", day)]
            )
        await session.commit()
        yield AnalyticsService(session), _DatabaseStub(factory), session
    await engine.dispose()


def _ride(native_id: str, day: _dt.date) -> GboCandidate:
    """A constant-250 W, 1 h cycling ride (TSS == 100 at FTP 250)."""
    seconds, watts = 3600, 250.0
    payload = {
        "start_time": _dt.datetime(day.year, day.month, day.day, 8, 0, tzinfo=UTC),
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
        content_hash=content_hash(native_id.encode()),
        payload=payload,
        trust_tier=Fidelity.RAW_STREAM,
        fetched_at=_dt.datetime(2026, 6, 4, 9, 0, tzinfo=UTC),
    )


async def test_engine_answer_runs_graph_to_completed(
    seeded: tuple[AnalyticsService, _DatabaseStub, AsyncSession],
) -> None:
    """The engine drives the compiled graph to a terminal grounded answer (GRAPH-R1).

    The scripted model states the CANONICAL current fitness as a dateless NUMBER claim, so the
    real grounder grounds it with a citation and the run completes. (A number-free draft over a
    gathered metric capability no longer completes — STATUS-R1 degrades it honestly; the
    completed-with-zero-citations semantics this test once pinned was the issue-44/-45 defect.)
    """
    svc, database, _ = seeded
    today = _dt.datetime.now(UTC).date()
    series = await svc.pmc(str(OWNER_ATHLETE_ID), today - _dt.timedelta(days=42), today)
    ctl = next(day.value.ctl for day in reversed(series) if day.available)
    model = FakeModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NUMBER,
                        text=f"your fitness is {ctl:.2f}",
                        metric="ctl",
                        value=ctl,
                    )
                ]
            ),
        },
        prose=f"Your form is in a good place this week — fitness is around {ctl:.2f}.",
    )
    engine = GraphAgentEngine(database, model)  # type: ignore[arg-type]
    answer = await engine.answer(
        athlete_id=OWNER_SUBJECT,
        question="How am I?",
        thread_id=None,
        response_length="standard",
        follow_up=None,
        locale="en",
    )
    assert answer.status is RunStatus.COMPLETED
    assert "good place" in answer.answer_text
    assert answer.citations, "a completed data-grounded answer carries >=1 citation (STATUS-R1)"


async def test_claimgrounder_grounds_number_against_canonical(
    seeded: tuple[AnalyticsService, _DatabaseStub, AsyncSession],
) -> None:
    """A NUMBER claim stating the canonical CTL grounds VERBATIM with a citation (GROUND-R7)."""
    svc, _, _ = seeded
    series = await svc.pmc(str(OWNER_ATHLETE_ID), _dt.date(2026, 6, 1), _dt.date(2026, 6, 3))
    ctl = series[-1].value.ctl  # the canonical value the claim must match within tolerance
    model = FakeModel(
        scripted={
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NUMBER,
                        text=f"your fitness is {ctl:.2f}",
                        metric="ctl",
                        value=ctl,
                        as_of="2026-06-03",
                    )
                ]
            )
        }
    )
    grounder = ClaimGrounder(model, svc)
    result = await grounder.ground(
        athlete_id=str(OWNER_ATHLETE_ID),
        draft=f"Your fitness is {ctl:.2f} and steady.",
        retrieved={"weekly_load": series},
    )
    assert result.decision is GroundDecision.PROCEED
    grounded = [c for c in result.survivors if c.citation is not None]
    assert grounded, "the canonical NUMBER claim must survive grounded with a citation"
    assert grounded[0].citation["metric"] == "ctl"
    assert grounded[0].citation["value"] == ctl  # verbatim canonical value, never re-derived


async def test_claimgrounder_unparseable_past_date_scrubs_not_grounds_latest(
    seeded: tuple[AnalyticsService, _DatabaseStub, AsyncSession],
) -> None:
    """H2 fabrication: a claim with a FAILED-PARSE date is SCRUBBED, never grounded against latest.

    The model says "On June 1 your fitness was <latest-ctl>", extracted with ``as_of="June 1"`` — a
    date token that does NOT parse as ISO. The OLD behaviour fell back to the LATEST PMC day, so the
    claim (whose value equals the LATEST ctl) GROUNDED — fabricating a past-dated fact from today's
    value. The fix fails closed: an unparseable date resolves to ``None``, so the number is SCRUBBED
    and the run does NOT proceed. The latest ctl must NOT reach the body under a past-date claim.
    """
    svc, _, _ = seeded
    series = await svc.pmc(str(OWNER_ATHLETE_ID), _dt.date(2026, 6, 1), _dt.date(2026, 6, 3))
    latest_ctl = series[-1].value.ctl
    model = FakeModel(
        scripted={
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NUMBER,
                        text=f"On June 1 your fitness was {latest_ctl:.2f}",
                        metric="ctl",
                        value=latest_ctl,
                        as_of="June 1",  # a date token that FAILS to parse as ISO
                    )
                ]
            )
        }
    )
    grounder = ClaimGrounder(model, svc)
    result = await grounder.ground(
        athlete_id=str(OWNER_ATHLETE_ID),
        draft=f"On June 1 your fitness was {latest_ctl:.2f}.",
        retrieved={"weekly_load": series},
    )
    assert not result.survivors, "an unparseable-date claim must NOT ground against the latest day"
    assert f"{latest_ctl:.2f}" not in result.scrubbed_text, "the past-dated number must be scrubbed"
    assert result.decision is not GroundDecision.PROCEED


async def test_engine_grounded_number_reaches_answer_citations(
    seeded: tuple[AnalyticsService, _DatabaseStub, AsyncSession],
) -> None:
    """A grounded NUMBER survives projection with a citation on the answer (GROUND-R5).

    The metric citation carries a resolvable ``record_id`` (``{metric}@{as_of}``), so the
    deliverables layer keeps it instead of dropping it — a grounded number must never reach
    the athlete uncited.
    """
    svc, database, _ = seeded
    series = await svc.pmc(str(OWNER_ATHLETE_ID), _dt.date(2026, 6, 1), _dt.date(2026, 6, 3))
    ctl = series[-1].value.ctl
    model = FakeModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NUMBER,
                        text=f"your fitness is at {ctl:.2f}",
                        metric="ctl",
                        value=ctl,
                        as_of="2026-06-03",
                    )
                ]
            ),
        },
        prose=f"Your fitness is at {ctl:.2f} and holding steady.",
    )
    engine = GraphAgentEngine(database, model)  # type: ignore[arg-type]
    answer = await engine.answer(
        athlete_id=str(OWNER_ATHLETE_ID),
        question="What's my fitness number?",
        thread_id=None,
        response_length="standard",
        follow_up=None,
        locale="en",
    )
    assert answer.status is RunStatus.COMPLETED
    assert answer.citations, "a grounded number must reach the answer with a citation (GROUND-R5)"
    assert answer.citations[0].metric == "ctl"
    assert answer.citations[0].record_id == "ctl@2026-06-03"


async def test_engine_answer_carries_stable_observation_drillable_on_followup(
    seeded: tuple[AnalyticsService, _DatabaseStub, AsyncSession],
) -> None:
    """COACH-R8: a grounded answer carries a stable-id observation a drill follow-up targets.

    The first turn grounds a CTL number, so the answer MUST carry a distinct observation with a
    STABLE ``observation_id`` (the drill/reveal handle, COACH-R8) and the grounded citation behind
    it. A ``drill`` follow-up on the SAME durable thread, targeting that exact id, MUST surface the
    grounded ``{metric, value, as_of}`` number VERBATIM (VOICE-R9) — proving drill-by-id is no
    longer vacuous in production (the ground node now writes the observations channel). The id is
    server-generated and stable, so the client can pass it straight back.
    """
    svc, database, _ = seeded
    series = await svc.pmc(str(OWNER_ATHLETE_ID), _dt.date(2026, 6, 1), _dt.date(2026, 6, 3))
    ctl = series[-1].value.ctl
    model = FakeModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NUMBER,
                        text=f"your fitness is at {ctl:.2f}",
                        metric="ctl",
                        value=ctl,
                        as_of="2026-06-03",
                    )
                ]
            ),
        },
        prose=f"Your fitness is at {ctl:.2f} and holding steady.",
    )
    engine = GraphAgentEngine(database, model)  # type: ignore[arg-type]
    first = await engine.answer(
        athlete_id=str(OWNER_ATHLETE_ID),
        question="What's my fitness number?",
        thread_id=None,
        response_length="standard",
        follow_up=None,
        locale="en",
    )
    assert first.status is RunStatus.COMPLETED
    # The answer carries a distinct, drillable observation with a STABLE id (COACH-R8).
    assert first.observations, "a grounded answer must carry a stable-id observation (COACH-R8)"
    obs = first.observations[0]
    assert obs.observation_id, "the observation MUST carry a stable id to target"
    assert any(c.metric == "ctl" for c in obs.citations), "grounded number behind the observation"

    # A drill follow-up targeting that exact id surfaces the grounded number on the SAME thread.
    drilled = await engine.answer(
        athlete_id=str(OWNER_ATHLETE_ID),
        question="What's my fitness number?",
        thread_id=first.thread_id,
        response_length="standard",
        follow_up={"kind": "drill", "target_ref": obs.observation_id},
        locale="en",
    )
    assert drilled.thread_id == first.thread_id, "a drill follow-up must reuse the SAME thread"
    assert any(c.metric == "ctl" and c.record_id == "ctl@2026-06-03" for c in drilled.citations), (
        "the drill-by-id MUST surface the grounded number verbatim (COACH-R8/VOICE-R9)"
    )


async def test_claimgrounder_scrubs_fabricated_number(
    seeded: tuple[AnalyticsService, _DatabaseStub, AsyncSession],
) -> None:
    """A fabricated number is contradicted by canonical and never published (GROUND-R7/R9).

    The seeded athlete HAS a canonical CTL, so a fabricated ``999`` (with no as-of date) is now
    checked against the latest-available canonical value (the §16 dateless fallback) and comes back
    CONTRADICTED — the canonical value EXISTS and differs — which drives a bounded REGENERATE
    (re-draft with the offending span corrected, GROUND-R9), not ABSTAIN. The guarantee under test
    is unchanged: the fabricated ``999`` is scrubbed/replaced by the canonical value and NEVER
    published as stated, and a contradicted claim carries no citation.
    """
    svc, _, _ = seeded
    model = FakeModel(
        scripted={
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NUMBER, text="Your CTL is 999", metric="ctl", value=999.0
                    )
                ]
            )
        }
    )
    grounder = ClaimGrounder(model, svc)
    result = await grounder.ground(
        athlete_id=str(OWNER_ATHLETE_ID), draft="Your CTL is 999 and rising.", retrieved={}
    )
    # The fabricated number is contradicted by the live canonical value and replaced in place;
    # 999 never reaches the athlete, and the contradicted claim carries no citation (GROUND-R9).
    assert result.decision is GroundDecision.REGENERATE
    assert all(c.citation is None for c in result.survivors)
    assert "999" not in result.scrubbed_text


async def test_production_grounder_replans_on_a_missing_metric_not_immediate_abstain(
    seeded: tuple[AnalyticsService, _DatabaseStub, AsyncSession],
) -> None:
    """GROUND-R6: the PRODUCTION grounder emits REPLAN when a claimed metric is unavailable.

    The seeded athlete has CTL/power data but NO HRV (no wellness rows), so a draft claiming an HRV
    number scrubs to NOTHING publishable — the deliverable can no longer answer. The metric is a
    re-gatherable gap (GROUND-R9 ``replan`` = "missing evidence"), so the LIVE aggregator (real
    ``_decide``, not a scripted stub) MUST emit ``REPLAN`` — proving the previously-dead
    ``ground -> reflect -> plan_retrieval`` recovery edge fires from production grounding. The
    unverified HRV number is still scrubbed (fail-closed, GROUND-R7).
    """
    svc, _, _ = seeded
    model = FakeModel(
        scripted={
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NUMBER, text="Your HRV is 65", metric="hrv", value=65.0
                    )
                ]
            )
        }
    )
    grounder = ClaimGrounder(model, svc)
    result = await grounder.ground(
        athlete_id=str(OWNER_ATHLETE_ID), draft="Your HRV is 65 and healthy.", retrieved={}
    )
    assert not result.survivors, "the unavailable HRV number must NOT ground"
    assert "65" not in result.scrubbed_text, "the unverified number is scrubbed (GROUND-R7)"
    # The load-bearing assertion: the live aggregator recovers via REPLAN, not an immediate ABSTAIN.
    assert result.decision is GroundDecision.REPLAN
    assert result.decision is not GroundDecision.ABSTAIN


async def test_metric_snapshot_reads_canonical_value(
    seeded: tuple[AnalyticsService, _DatabaseStub, AsyncSession],
) -> None:
    """The canonical evidence resolves the same CTL the grounder snapshots (GROUND-R7)."""
    svc, _, _ = seeded
    evidence = CanonicalEvidence(svc, str(OWNER_ATHLETE_ID))
    value = await evidence.metric_value("ctl", "2026-06-03")
    assert value is not None and value > 0


def test_build_agent_engine_is_none_without_model_key() -> None:
    """OSS boots without an LLM key: the engine builder returns None, not a broken engine."""
    settings = load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="k" * 32,
    )

    class _Db:  # the builder only reads settings.llm_api_key; the db is never touched here
        pass

    assert build_agent_engine(_Db(), settings) is None  # type: ignore[arg-type]


# --- readiness/form deliverable (QA-EVAL-R2.4 / API-R41) --------------------------
#
# These drive the REAL :meth:`GraphAgentEngine.readiness` against a self-dating canonical
# store (rides are seeded relative to ``utcnow()`` so the latest computed TSB lands on
# "today" regardless of the wall clock), so the delivered verdict is deterministic.


def _watt_ride(native_id: str, day: _dt.date, *, watts: float) -> GboCandidate:
    """A constant-``watts``, 1 h cycling ride. TSS == (watts/250)^2 * 100 at FTP 250."""
    seconds = 3600
    payload = {
        "start_time": _dt.datetime(day.year, day.month, day.day, 8, 0, tzinfo=UTC),
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
        content_hash=content_hash(native_id.encode()),
        payload=payload,
        trust_tier=Fidelity.RAW_STREAM,
        fetched_at=_dt.datetime(2026, 6, 4, 9, 0, tzinfo=UTC),
    )


async def _seed_store(
    rides: list[tuple[_dt.date, float]],
    *,
    hrv_day_rmssd: tuple[_dt.date, float] | None = None,
    hrv_baseline_band: tuple[float, float] | None = None,
) -> tuple[_DatabaseStub, AnalyticsService, AsyncSession]:
    """Seed a fresh canonical store with the owner + FTP signature + given rides (+HRV).

    Each ride is ``(day, watts)``; an optional ``hrv_day_rmssd`` adds a summary-only HRV
    wellness row (RMSSD ms) for that day so the readiness gather can read a canonical HRV.
    ``hrv_baseline_band`` sets the ``(low_ms, high_ms)`` HRV-baseline band on that same
    wellness row so the readiness gather can read a live baseline (its midpoint) and the
    oracle's HRV-suppression nudge can fire end-to-end (COACH-R1 #2).
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    session = factory()
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    session.add(Athlete(athlete_id=OWNER_ATHLETE_ID, sex="male", reference_timezone="UTC"))
    descriptor = SourceDescriptor(
        source_key="file_import", display_name="Activity files", kind="file_upload"
    )
    session.add(descriptor)
    session.add(
        FitnessSignature(
            athlete_id=OWNER_ATHLETE_ID,
            signature_type="cycling",
            effective_date=_dt.date(2024, 1, 1),
            ftp_w=250.0,
            origin=SignatureOrigin.MEASURED,
        )
    )
    if hrv_day_rmssd is not None:
        day, rmssd = hrv_day_rmssd
        low, high = hrv_baseline_band if hrv_baseline_band is not None else (None, None)
        session.add(
            DailyWellness(
                athlete_id=OWNER_ATHLETE_ID,
                local_date=day,
                hrv_rmssd_ms=rmssd,
                hrv_baseline_low_ms=low,
                hrv_baseline_high_ms=high,
                hrv_method=HrvMethod.RMSSD,
            )
        )
    await session.flush()
    ingest = IngestService(session)
    descriptor_id = str(descriptor.source_descriptor_id)
    for i, (day, watts) in enumerate(rides):
        await ingest.ingest(
            str(OWNER_ATHLETE_ID), descriptor_id, [_watt_ride(f"k{i}", day, watts=watts)]
        )
    await session.commit()
    return _DatabaseStub(factory), AnalyticsService(session), session


def _rest_narrator() -> FakeModel:
    """A FakeModel scripting a coherent state-first 'rest' narration + claim extraction."""
    return FakeModel(
        scripted={
            "_ReadinessNarration": _ReadinessNarration(
                summary_text="You're carrying deep fatigue, so today is for rest.",
                verdict=ReadinessVerdict.REST,
            ),
            "_ClaimSchema": _ClaimSchema(claims=[]),
        }
    )


def _maintain_narrator() -> FakeModel:
    """A FakeModel scripting a coherent state-first 'maintain' narration + claim extraction."""
    return FakeModel(
        scripted={
            "_ReadinessNarration": _ReadinessNarration(
                summary_text="You're in a steady place — keep things as planned.",
                verdict=ReadinessVerdict.MAINTAIN,
            ),
            "_ClaimSchema": _ClaimSchema(claims=[]),
        }
    )


def _no_numeric_readiness_field(readiness: Readiness) -> bool:
    """True iff the deliverable exposes no numeric ``readiness`` KPI/score attribute (API-R41)."""
    return not any(
        name in {"readiness", "readiness_score", "score"} for name in Readiness.__slots__
    )


async def test_readiness_deep_negative_form_is_rest_state_first() -> None:
    """Deep-negative form -> a metric-consistent 'rest' verdict, state-first (QA-EVAL-R2.4).

    Five hard rides ending today drive TSB well below the fatigue floor, so the delivered
    verdict is ``rest``; the summary leads with a number-LESS state sentence and there is no
    numeric readiness field on the deliverable.
    """
    today = utcnow().date()
    rides = [(today - _dt.timedelta(days=4 - i), 320.0) for i in range(5)]
    database, _, _ = await _seed_store(rides)
    engine = GraphAgentEngine(database, _rest_narrator())  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is ReadinessVerdict.REST
    assert not any(ch.isdigit() for ch in first_sentence(readiness.summary_text))
    assert _no_numeric_readiness_field(readiness)


async def test_readiness_form_number_surfaces_only_as_grounded_citation() -> None:
    """A form NUMBER in the narration grounds VERBATIM against canonical form (GROUND-R5/R7).

    The summary still LEADS with a number-less state sentence (COACH-R7); the form number
    appears only in a later sentence and survives as a grounded ``form`` citation — the
    on-demand backing — never as a hero readiness number. The verdict stays the canonical
    ``rest``, and there is no numeric readiness field on the deliverable (API-R41).
    """
    today = utcnow().date()
    rides = [(today - _dt.timedelta(days=4 - i), 320.0) for i in range(5)]
    database, svc, _ = await _seed_store(rides)
    series = await svc.pmc(str(OWNER_ATHLETE_ID), today - _dt.timedelta(days=14), today)
    form = series[-1].value.tsb  # the canonical form the claim must match within tolerance
    iso = today.isoformat()
    model = FakeModel(
        scripted={
            "_ReadinessNarration": _ReadinessNarration(
                summary_text=(
                    f"You're deep in fatigue, so today is for rest. Your form sits at {form:.1f}."
                ),
                verdict=ReadinessVerdict.REST,
            ),
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NUMBER,
                        text=f"Your form sits at {form:.1f}",
                        metric="form",
                        value=form,
                        as_of=iso,
                    )
                ]
            ),
        }
    )
    engine = GraphAgentEngine(database, model)  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is ReadinessVerdict.REST
    assert not any(ch.isdigit() for ch in first_sentence(readiness.summary_text))  # state-first
    assert readiness.citations, "the form number must reach the deliverable as a grounded citation"
    assert readiness.citations[0].metric == "form"
    assert readiness.citations[0].record_id == f"form@{iso}"
    assert _no_numeric_readiness_field(readiness)


async def test_readiness_fresh_form_is_go() -> None:
    """A tapered, fresh canonical form (TSB > fresh threshold) -> a 'go' verdict (QA-EVAL-R2.4)."""
    today = utcnow().date()
    # A 10-day block of solid rides ~4 weeks ago, then a full taper to today -> CTL > ATL.
    rides = [(today - _dt.timedelta(days=37 - i), 274.0) for i in range(10)]
    database, _, _ = await _seed_store(rides)
    model = FakeModel(
        scripted={
            "_ReadinessNarration": _ReadinessNarration(
                summary_text="You're fresh and ready for a hard day.",
                verdict=ReadinessVerdict.GO,
            ),
            "_ClaimSchema": _ClaimSchema(claims=[]),
        }
    )
    engine = GraphAgentEngine(database, model)  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is ReadinessVerdict.GO
    assert not any(ch.isdigit() for ch in first_sentence(readiness.summary_text))


class _SystemRecordingModel(FakeModel):
    """A FakeModel that ALSO records the system prompt each structured call received."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.systems: list[str] = []

    async def structured(self, *, system: str, data: str, schema: type):  # type: ignore[override,no-untyped-def]
        self.systems.append(system)
        return await super().structured(system=system, data=data, schema=schema)


async def test_readiness_narrator_system_carries_the_requested_language_directive() -> None:
    """The readiness narrator's system prompt is composed via the any-language DIRECTIVE (#17).

    A non-en/de/ru locale ("es") reaches the narrator as a config-templated directive naming that
    exact language tag — proving the readiness path is directive-driven (the SAME ``compose_system``
    seam the free-form answer uses), NOT served by an enumerated readiness language pack.
    """
    today = utcnow().date()
    rides = [(today - _dt.timedelta(days=4 - i), 320.0) for i in range(5)]
    database, _, _ = await _seed_store(rides)
    model = _SystemRecordingModel(
        scripted={
            "_ReadinessNarration": _ReadinessNarration(
                summary_text="Hoy toca descansar.", verdict=ReadinessVerdict.REST
            ),
            "_ClaimSchema": _ClaimSchema(claims=[]),
        }
    )
    # A config-templated pass-through coach bundle (NO Spanish pack — only the directive template).
    coach = CoachBundle(
        readiness_system="readiness persona",
        locales=LocalePolicy.from_config(
            {"en": {"compose_directive": "Reply in English.", "limitation": "Not enough data."}},
            "en",
            passthrough_enabled=True,
            passthrough_directive="Answer in the language with IETF tag '{language_tag}'.",
        ),
    )
    engine = GraphAgentEngine(database, model, coach=coach)  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT, locale="es")
    assert readiness.verdict is ReadinessVerdict.REST
    assert model.systems, "the narrator must have been called with a composed system prompt"
    narrator_system = model.systems[0]
    assert narrator_system.startswith("readiness persona")
    assert "'es'" in narrator_system  # the directive names the requested (unenumerated) language
    assert "Reply in English." not in narrator_system  # not the default pack (no enumeration)


async def test_readiness_consistency_gate_overrides_model_go_to_rest() -> None:
    """A model proposing 'go' under deep-negative form is overridden to 'rest' (COACH-R3 / EVAL-R5).

    The deterministic gate (``readiness_consistent``) rejects the model verdict; the delivered
    verdict is the canonical ``rest`` and a coverage caveat records the override (fail-closed).
    """
    today = utcnow().date()
    rides = [(today - _dt.timedelta(days=4 - i), 320.0) for i in range(5)]
    database, _, _ = await _seed_store(rides)
    model = FakeModel(
        scripted={
            "_ReadinessNarration": _ReadinessNarration(
                summary_text="You're fresh and ready to crush a hard one.",
                verdict=ReadinessVerdict.GO,  # contradicts deep-negative form
            ),
            "_ClaimSchema": _ClaimSchema(claims=[]),
        }
    )
    engine = GraphAgentEngine(database, model)  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is ReadinessVerdict.REST  # canonical wins, not the model's 'go'
    assert readiness.coverage is not None
    assert readiness.coverage["verdict_override"] == "model_inconsistent_with_metrics"
    # the delivered lead describes the canonical (rest) state, not the model's 'go' prose
    assert "fresh" not in readiness.summary_text.lower()


async def test_readiness_abstains_truthfully_without_form() -> None:
    """Unavailable form -> verdict None + a truthful insufficient-data summary (GROUND-R6).

    Exercises the real :func:`readiness_assessment` abstain branch with an unavailable form
    (the fail-closed input the engine's gather yields when no PMC day is computable): no
    verdict, no number, an honest state sentence, no model call needed, nothing to cite.
    """
    readiness = await readiness_assessment(
        OWNER_SUBJECT,
        form=None,  # canonical form unavailable -> the oracle abstains
        as_of=None,
        hrv_rmssd=None,
        hrv_baseline=None,
        narrate=None,
        grounder=None,
    )
    assert readiness.verdict is None
    assert "enough" in readiness.summary_text.lower()
    assert not readiness.citations  # nothing grounded to cite
    assert _no_numeric_readiness_field(readiness)


async def test_readiness_hrv_unavailable_states_so_and_uses_form() -> None:
    """Form present, HRV absent -> verdict from form, summary states HRV unavailable (GROUND-R7)."""
    today = utcnow().date()
    rides = [(today - _dt.timedelta(days=4 - i), 320.0) for i in range(5)]
    database, _, _ = await _seed_store(rides)  # no HRV wellness row seeded
    engine = GraphAgentEngine(database, _rest_narrator())  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is ReadinessVerdict.REST  # from form alone
    assert "hrv" in readiness.summary_text.lower()
    assert readiness.coverage is not None
    assert "hrv" in readiness.coverage["inputs_unavailable"]


# A constant-watts ride at ~50 TSS (NP/FTP = sqrt(0.5)); a steady block of these ending a few
# rest days before today lands the canonical form (TSB) in the MAINTAIN band with a real CTL.
_MAINTAIN_WATTS = 250.0 * math.sqrt(0.5)


def _maintain_band_rides(today: _dt.date) -> list[tuple[_dt.date, float]]:
    """A 40-ride steady block ending 3 rest days before today -> MAINTAIN-band form, CTL>0.

    The taper-into-rest brings ATL back toward CTL so the latest computed TSB sits in the
    neutral (MAINTAIN) band while a real chronic base (CTL well above the cold-start epsilon)
    has accumulated — the setup the HRV-suppression nudge needs to be visible.
    """
    return [(today - _dt.timedelta(days=42 - i), _MAINTAIN_WATTS) for i in range(40)]


async def test_readiness_cold_start_abstains_not_confident_maintain() -> None:
    """Zero rides -> (0,0)-seed cold-start abstains, never a confident MAINTAIN (GROUND-R6).

    A brand-new athlete with NO activities still gets the honest (0,0) PMC origin seed, so the
    latest computed day reads ctl≈atl≈tsb≈0. The cold-start epsilon (READINESS_MIN_FITNESS_CTL)
    treats that ~0 CTL as "no real fitness signal", so the REAL engine flow yields verdict None
    + a DEGRADED status + the truthful insufficient-data summary — not a "keep training" verdict
    on an empty base. This exercises the abstain path through the engine's gather, not just a
    direct ``readiness_assessment(form=None)`` call.
    """
    database, _, _ = await _seed_store([])  # no rides at all
    engine = GraphAgentEngine(database, FakeModel())  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is None  # abstains rather than emitting MAINTAIN on no data
    assert readiness.status is RunStatus.DEGRADED
    assert "enough" in readiness.summary_text.lower()  # the truthful _ABSTAIN_SENTENCE
    assert not readiness.citations
    assert _no_numeric_readiness_field(readiness)


async def test_readiness_live_hrv_baseline_nudges_maintain_to_ease() -> None:
    """A live HRV baseline + suppressed RMSSD nudges MAINTAIN one step to EASE (COACH-R1 #2).

    Form sits in the MAINTAIN band; the seeded wellness row carries an HRV baseline band whose
    midpoint (50 ms) is read live by the gather, and an RMSSD (30 ms) clearly below baseline*0.9.
    The oracle's HRV-suppression nudge therefore fires end-to-end and the delivered verdict is
    EASE, proving the previously-dead live HRV-baseline path is wired (the gather no longer
    hardcodes baseline=None).
    """
    today = utcnow().date()
    rides = _maintain_band_rides(today)
    database, _, _ = await _seed_store(
        rides,
        hrv_day_rmssd=(today, 30.0),  # suppressed: 30 < midpoint(50) * 0.9 = 45
        hrv_baseline_band=(40.0, 60.0),  # midpoint 50 ms read live by the gather
    )
    model = FakeModel(
        scripted={
            "_ReadinessNarration": _ReadinessNarration(
                summary_text="You're carrying some fatigue, so ease off a little today.",
                verdict=ReadinessVerdict.EASE,
            ),
            "_ClaimSchema": _ClaimSchema(claims=[]),
        }
    )
    engine = GraphAgentEngine(database, model)  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is ReadinessVerdict.EASE  # MAINTAIN nudged toward REST by live HRV
    assert readiness.coverage is not None
    assert "hrv" in readiness.coverage["inputs_used"]  # the live baseline made HRV usable


async def test_readiness_baseline_absent_stays_maintain_from_form_alone() -> None:
    """Same suppressed RMSSD but NO baseline -> HRV unusable, verdict stays MAINTAIN (control).

    The control for the live-baseline nudge: identical MAINTAIN-band form and a low RMSSD, but
    no baseline band on the wellness row, so the gather reads baseline=None, HRV is recorded
    unavailable, and the verdict is read from form alone (MAINTAIN). This isolates the baseline
    as the cause of the EASE nudge above.
    """
    today = utcnow().date()
    rides = _maintain_band_rides(today)
    database, _, _ = await _seed_store(
        rides,
        hrv_day_rmssd=(today, 30.0),  # low RMSSD, but no baseline to compare against
    )
    engine = GraphAgentEngine(database, _maintain_narrator())  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is ReadinessVerdict.MAINTAIN  # no baseline -> no nudge
    assert readiness.coverage is not None
    assert "hrv" in readiness.coverage["inputs_unavailable"]


async def test_readiness_state_first_lead_rejects_digit_anywhere_in_first_sentence() -> None:
    """A number in the FIRST sentence of the model lead is rejected; the lead falls back (COACH-R7).

    The model narrates a state-led first sentence that nonetheless carries a digit
    ("Your form of 12 looks strong today."). COACH-R7 wants the first sentence number-LIGHT, so
    the tightened gate falls back to the deterministic per-verdict state sentence. Driven through
    the REAL deliverable/engine, the DELIVERED summary's first sentence carries no digit.
    """
    today = utcnow().date()
    rides = [(today - _dt.timedelta(days=4 - i), 320.0) for i in range(5)]  # deep-negative -> REST
    database, _, _ = await _seed_store(rides)
    model = FakeModel(
        scripted={
            "_ReadinessNarration": _ReadinessNarration(
                summary_text="Your form of 12 looks strong today. Take it easy.",
                verdict=ReadinessVerdict.REST,  # consistent with deep-negative form (no override)
            ),
            "_ClaimSchema": _ClaimSchema(claims=[]),
        }
    )
    engine = GraphAgentEngine(database, model)  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is ReadinessVerdict.REST
    lead = first_sentence(readiness.summary_text)
    assert not any(ch.isdigit() for ch in lead)  # digit-laden lead fell back to the state sentence
    assert leads_with_state(readiness.summary_text)
    assert "12" not in lead


async def test_readiness_hrv_unavailable_clause_is_matchable_and_voice_clean() -> None:
    """The HRV-unavailable case emits the PUBLIC clause verbatim, state-led + digit-free (FIX 7).

    Drives the REAL deliverable for an HRV-unavailable case (form present, no wellness row at
    all) and asserts the delivered summary CONTAINS the exact public ``HRV_UNAVAILABLE_CLAUSE``
    (so the sibling eval voice grader can import + match it), while the FIRST sentence stays
    state-led and digit-free.
    """
    today = utcnow().date()
    rides = [(today - _dt.timedelta(days=4 - i), 320.0) for i in range(5)]
    database, _, _ = await _seed_store(rides)  # no HRV wellness row
    engine = GraphAgentEngine(database, _rest_narrator())  # type: ignore[arg-type]
    readiness = await engine.readiness(athlete_id=OWNER_SUBJECT)
    assert readiness.verdict is ReadinessVerdict.REST
    assert HRV_UNAVAILABLE_CLAUSE in readiness.summary_text  # exact public prod phrasing
    lead = first_sentence(readiness.summary_text)
    assert leads_with_state(readiness.summary_text)
    assert not any(ch.isdigit() for ch in lead)
