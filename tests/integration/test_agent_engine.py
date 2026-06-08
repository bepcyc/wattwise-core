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
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.capabilities import CanonicalEvidence
from wattwise_core.agent.contracts import ClaimKind, GroundDecision, RunStatus
from wattwise_core.agent.engine import (
    ClaimGrounder,
    GraphAgentEngine,
    _ClaimSchema,
    _ExtractedClaim,
    _PlanSchema,
    build_agent_engine,
)
from wattwise_core.agent.model import FakeModel
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.config import load_settings
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, SignatureOrigin
from wattwise_core.identity import OWNER_ATHLETE_ID, OWNER_SUBJECT
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.persistence.models import (
    Athlete,
    Base,
    FitnessSignature,
    SourceDescriptor,
    Sport,
)
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
    """The engine drives the compiled graph to a terminal grounded answer (GRAPH-R1)."""
    _, database, _ = seeded
    model = FakeModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[_ExtractedClaim(kind=ClaimKind.STATEMENT, text="trending up")]
            ),
        },
        prose="Your form is in a good place this week.",
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


async def test_claimgrounder_scrubs_fabricated_number(
    seeded: tuple[AnalyticsService, _DatabaseStub, AsyncSession],
) -> None:
    """A fabricated number with no resolvable canonical value scrubs and abstains (GROUND-R6)."""
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
    # Nothing publishable survives the fabricated number -> abstain (the engine's finalize
    # then replaces the body with an explicit limitation, GROUND-R6); the verbatim span the
    # model pointed at is scrubbed out of the grounded text here.
    assert result.decision is GroundDecision.ABSTAIN
    assert all(c.citation is None for c in result.survivors)
    assert "999" not in result.scrubbed_text


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
