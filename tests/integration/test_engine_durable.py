"""Engine-level durable HITL + follow-up tests on a REAL pool (D-P2 / COACH-R2/-R8, CKPT-R5/-R9).

These drive the deployable :class:`~wattwise_core.agent.engine.GraphAgentEngine` end to end over a
DURABLE :class:`SqlAlchemyCheckpointSaver` (its dedicated agent-state pool replacing the in-memory
checkpointer), pinning the Core-C slice of the durable-resume work:

* **F-FOLLOWUP** — a follow-up turn (the caller passes the prior ``thread_id`` back) resumes the
  SAME durable thread; ``expand`` climbs the verbosity ladder and ``drill``/``reveal_numbers``
  surface the verbatim grounded numbers, never starting a divergent thread (COACH-R8).
* **F-APPROVE / F-EDIT / F-REJECT** — a multi-day PLAN deliverable pauses at ``awaiting_approval``
  with a live interrupt; the decision endpoint resumes it through the DURABLE saver
  (``Command(resume)``) WITHOUT recomputing the head node; ``approve`` finalizes, ``reject``
  resumes un-approved, ``edit`` RE-GROUNDS the edited body so an unverified workout name is scrubbed
  while a canonical one survives (GROUND-R3 / CKPT-R2/-R5).
* **F-409 / F-404 / F-XID** — a double-decision, an unknown interrupt, and a cross-athlete decision
  each fail closed (``DecisionRefused``); the run is never resumed twice (CKPT-R9).

CRITICAL (skill §7): the agent-state saver runs on a **file-backed SQLite engine with a real
connection pool** (WAL + busy_timeout), NEVER ``:memory:``/``StaticPool`` — a single-connection
setup false-greens durable-resume behaviour (the durable saver's benign-race ``_ensure_thread``
rollback would poison the shared connection). PostgreSQL / MariaDB legs run when ``WATTWISE_PG_DSN``
/ ``WATTWISE_MARIADB_DSN`` are set, proving the behaviour is backend-portable.
"""

from __future__ import annotations

import datetime as _dt
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.contracts import ClaimKind, RunStatus
from wattwise_core.agent.engine import (
    DecisionRefused,
    GraphAgentEngine,
    _ClaimSchema,
    _ExtractedClaim,
    _PlanSchema,
)
from wattwise_core.agent.memory import (  # registers agent_memory_item on AgentStateBase
    MemoryItemKind,
    OssMemoryStore,
)
from wattwise_core.agent.model import FakeModel
from wattwise_core.agent.state_db import AgentStateDatabase, build_agent_state_database
from wattwise_core.agent.state_store import AgentStateBase
from wattwise_core.domain.enums import SignatureOrigin
from wattwise_core.persistence.models import (
    Athlete,
    Base,
    FitnessSignature,
    SourceDescriptor,
    Sport,
)

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

pytestmark = pytest.mark.integration

ATHLETE_A = "00000000-0000-7000-8000-0000000000c1"
ATHLETE_B = "00000000-0000-7000-8000-0000000000c2"


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + long busy_timeout per SQLite connection so the real pool serialises writers."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


def _state_db_backends() -> list[ParameterSet]:
    """File-SQLite always; PG/MariaDB only when their throwaway DSN env var is set."""
    cases: list[ParameterSet] = [pytest.param(None, id="sqlite")]
    pg = os.environ.get("WATTWISE_PG_DSN")
    cases.append(
        pytest.param(pg, id="postgresql", marks=pytest.mark.skipif(not pg, reason="no PG DSN"))
    )
    maria = os.environ.get("WATTWISE_MARIADB_DSN")
    cases.append(
        pytest.param(
            maria, id="mariadb", marks=pytest.mark.skipif(not maria, reason="no MariaDB DSN")
        )
    )
    return cases


@pytest_asyncio.fixture(params=_state_db_backends())
async def state_db(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[AgentStateDatabase]:
    """A DEDICATED agent-state database over a REAL multi-connection pool (skill §7).

    SQLite is file-backed (real pool) with WAL enabled per connection — deliberately NOT
    ``:memory:``/StaticPool, which hides the saver's get-or-create race. PG/MariaDB use their
    throwaway DSN and are reset to an empty agent-state schema before the test runs.
    """
    backend_dsn = request.param
    if backend_dsn is None:
        dsn = f"sqlite+aiosqlite:///{tmp_path}/agent.sqlite"
        db = build_agent_state_database(dsn=dsn)
        event.listen(db.engine.sync_engine, "connect", _enable_sqlite_wal)
    else:
        db = build_agent_state_database(dsn=backend_dsn)
        async with db.engine.begin() as conn:
            await conn.run_sync(AgentStateBase.metadata.drop_all)
    await db.create_all()
    try:
        yield db
    finally:
        async with db.engine.begin() as conn:
            await conn.run_sync(AgentStateBase.metadata.drop_all)
        await db.dispose()


# --- canonical store stub (the engine is READ-ONLY against it) --------------------------


class _DatabaseStub:
    """A minimal canonical ``Database`` substitute over one engine (the engine reads only)."""

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
async def canonical() -> AsyncIterator[_DatabaseStub]:
    """An in-memory canonical store seeded with the owner athlete + an FTP signature."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        for aid in (ATHLETE_A, ATHLETE_B):
            session.add(Athlete(athlete_id=uuid.UUID(aid), sex="male", reference_timezone="UTC"))
            session.add(
                FitnessSignature(
                    athlete_id=uuid.UUID(aid),
                    signature_type="cycling",
                    effective_date=_dt.date(2024, 1, 1),
                    ftp_w=250.0,
                    origin=SignatureOrigin.MEASURED,
                )
            )
        session.add(
            SourceDescriptor(
                source_key="file_import", display_name="Activity files", kind="file_upload"
            )
        )
        await session.commit()
    try:
        yield _DatabaseStub(factory)
    finally:
        await engine.dispose()


def _answer_model() -> FakeModel:
    """A FakeModel scripting a grounded free-form answer (one STATEMENT claim, no numbers)."""
    return FakeModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[_ExtractedClaim(kind=ClaimKind.STATEMENT, text="trending up")]
            ),
        },
        prose="Your form is in a good place this week.",
    )


def _plan_model() -> FakeModel:
    """A FakeModel scripting a PLAN whose prescribed workout NAMES are canonical (ground)."""
    return FakeModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NAME, text="endurance ride", as_of="endurance ride"
                    ),
                    _ExtractedClaim(kind=ClaimKind.STATEMENT, text="build base"),
                ]
            ),
        },
        prose="Day 1 endurance ride. Day 2 rest day. Build your base.",
    )


def _engine(
    canonical: _DatabaseStub, state_db: AgentStateDatabase, model: FakeModel
) -> GraphAgentEngine:
    return GraphAgentEngine(canonical, model, state_db=state_db)  # type: ignore[arg-type]


async def test_followup_expand_resumes_same_durable_thread(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """F-FOLLOWUP: an expand follow-up COMPLETES on the SAME durable thread (COACH-R8/CKPT-R5).

    The first turn opens a durable thread; the follow-up passes that ``thread_id`` back, so it must
    land on the SAME ``(athlete_id, conversation_id)`` scope (the reversible thread id) and complete
    — not compute a divergent thread and start a duplicate run (the bug the reversible mapping
    fixes). Both turns COMPLETE on the durable saver, proving the run-scoped reset across the turn
    boundary works through the real pool.
    """
    engine = _engine(canonical, state_db, _answer_model())
    first = await engine.answer(
        athlete_id=ATHLETE_A,
        question="How am I doing?",
        thread_id=None,
        response_length="standard",
        follow_up=None,
        locale="en",
    )
    assert first.status is RunStatus.COMPLETED
    assert first.thread_id.startswith(f"{ATHLETE_A}:")

    follow = await engine.answer(
        athlete_id=ATHLETE_A,
        question="How am I doing?",
        thread_id=first.thread_id,
        response_length="standard",
        follow_up={"kind": "expand", "target_ref": None},
        locale="en",
    )
    assert follow.status is RunStatus.COMPLETED
    assert follow.thread_id == first.thread_id, "an expand follow-up must reuse the SAME thread"


async def test_followup_reveal_numbers_runs_on_same_thread(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """F-FOLLOWUP: a reveal-numbers follow-up runs the drill branch on the SAME thread (COACH-R8).

    A ``reveal_numbers`` follow-up exercises the verbatim-citation reveal branch (VOICE-R9): it
    reuses the prior durable thread and completes, surfacing only numbers the graph already
    grounded (the reveal merges grounded citations, never invents one). Here the answer carries no
    grounded number, so the reveal simply completes on the same thread without fabricating one.
    """
    engine = _engine(canonical, state_db, _answer_model())
    first = await engine.answer(
        athlete_id=ATHLETE_A,
        question="What's my fitness?",
        thread_id=None,
        response_length="standard",
        follow_up=None,
        locale="en",
    )
    follow = await engine.answer(
        athlete_id=ATHLETE_A,
        question="What's my fitness?",
        thread_id=first.thread_id,
        response_length="standard",
        follow_up={"kind": "reveal_numbers", "target_ref": None},
        locale="en",
    )
    assert follow.thread_id == first.thread_id, "a reveal follow-up must reuse the SAME thread"
    assert follow.status is RunStatus.COMPLETED
    # The reveal never fabricates a number: no citation appears that the graph did not ground.
    assert all(c.record_id for c in follow.citations)


# --- F-MEMORY (MEM-R4 durable-memory recall + episode write on the run path) -------------


class _ContextCapturingModel(FakeModel):
    """A FakeModel that records every ``compose`` context so a test can assert recall reached it."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.compose_contexts: list[str] = []

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_contexts.append(context)
        return await super().compose(system=system, context=context, max_tokens=max_tokens)


async def _memory_rows(state_db: AgentStateDatabase, athlete_id: str) -> list[Any]:
    """Read the athlete's durable memory rows through the OSS recall seam (newest first)."""
    async with state_db.session() as session:
        store = OssMemoryStore(session)
        return list(await store.fetch_relevant(athlete_id=athlete_id, query="", limit=50))


async def test_run_recalls_durable_memory_into_compose_and_writes_an_episode(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """MEM-R4: a coaching run RECALLS durable memory into compose AND records an episode.

    The previously-orphaned MemoryStore seam is wired into the run path: BEFORE composing, the
    engine recalls durable memory via ``fetch_relevant`` and projects it into the compose context
    (personalization, MEM-R1/-R2) — so a pre-seeded preference reaches the model's context.
    AFTER a COMPLETED run the engine records the turn as an episode through ``write_episode`` behind
    the SAME seam (MEM-R4). Both halves are asserted: the seeded preference text appears in the
    captured compose context (recall wired), and a NEW memory episode exists after the run (write
    wired). Memory is personalization only — it never supplies a canonical number (MEM-R1).
    """

    async with state_db.session() as session:
        store = OssMemoryStore(session)
        await store.write_episode(
            athlete_id=ATHLETE_A,
            kind=MemoryItemKind.CONSTRAINT,
            content="only train Tuesday and Thursday evenings",
            trusted=True,
        )
    before = await _memory_rows(state_db, ATHLETE_A)
    assert len(before) == 1

    model = _ContextCapturingModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[_ExtractedClaim(kind=ClaimKind.STATEMENT, text="trending up")]
            ),
        },
        prose="Your form is in a good place this week.",
    )
    engine = _engine(canonical, state_db, model)
    answer = await engine.answer(
        athlete_id=ATHLETE_A,
        question="How should I train this week?",
        thread_id=None,
        response_length="standard",
        follow_up=None,
        locale="en",
    )
    assert answer.status is RunStatus.COMPLETED

    # (1) RECALL wired: the seeded durable-memory item reached the compose context.
    assert model.compose_contexts, "compose must have run at least once"
    assert any(
        "only train Tuesday and Thursday evenings" in ctx for ctx in model.compose_contexts
    ), "recalled durable memory MUST reach the compose context (MEM-R4 recall-before-compose)"

    # (2) WRITE wired: a NEW episode was recorded after the completed run (MEM-R4 write-episode).
    after = await _memory_rows(state_db, ATHLETE_A)
    assert len(after) == len(before) + 1, "a completed run MUST record an episode (MEM-R4)"


# --- F-APPROVE / F-EDIT / F-REJECT (durable resume) -------------------------------------


async def test_plan_pauses_then_approve_resumes_durably(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """F-APPROVE: a PLAN pauses ``awaiting_approval``, then ``approve`` resumes to COMPLETED.

    The PLAN deliverable pauses at the durable interrupt-gate carrying a live ``interrupt_id``; the
    decision drives ``Command(resume)`` through the DURABLE saver and finalizes COMPLETED — no
    recompute of the pre-interrupt head node (CKPT-R2/-R5).
    """
    engine = _engine(canonical, state_db, _plan_model())
    plan = await engine.plan_deliverable(athlete_id=ATHLETE_A, request="give me a week plan")
    assert plan.status is RunStatus.AWAITING_APPROVAL
    assert plan.interrupt_id is not None
    assert "endurance ride" in plan.plan_text  # the canonical workout name grounded, not scrubbed

    resumed = await engine.decision(
        athlete_id=ATHLETE_A,
        thread_id=plan.thread_id,
        interrupt_id=plan.interrupt_id,
        decision="approve",
    )
    assert resumed.status is RunStatus.COMPLETED


async def test_plan_reject_resumes_durably(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """F-REJECT: a ``reject`` decision resumes the paused plan through the durable saver.

    Reject is still a resume (the run finalizes), but un-approved — it must not raise/recompute and
    must consume the live interrupt so it cannot be acted on twice (CKPT-R9).
    """
    engine = _engine(canonical, state_db, _plan_model())
    plan = await engine.plan_deliverable(athlete_id=ATHLETE_A, request="week plan")
    assert plan.status is RunStatus.AWAITING_APPROVAL and plan.interrupt_id is not None

    resumed = await engine.decision(
        athlete_id=ATHLETE_A,
        thread_id=plan.thread_id,
        interrupt_id=plan.interrupt_id,
        decision="reject",
    )
    assert resumed.status in (RunStatus.COMPLETED, RunStatus.DEGRADED)
    # The interrupt is consumed: a second decision (even approve) is refused.
    with pytest.raises(DecisionRefused):
        await engine.decision(
            athlete_id=ATHLETE_A,
            thread_id=plan.thread_id,
            interrupt_id=plan.interrupt_id,
            decision="approve",
        )


async def test_plan_edit_with_invented_name_is_rejected_degraded(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """F-EDIT (H3): an edit that does NOT fully ground is REJECTED -> DEGRADED, never shipped.

    The edited plan names an INVENTED workout ("magic super workout") plus a CANONICAL one
    ("threshold intervals"). Re-grounding scrubs the invented name -> the edit does NOT decide
    ``PROCEED`` (a surviving + a scrubbed claim is ``regenerate``), so the engine REJECTS the edit:
    the run resolves to ``DEGRADED`` and the delivered body is the already-grounded PRE-EDIT plan,
    NOT the partial/untrusted edit. The invented name never reaches the athlete (GROUND-R3 / H3).
    """
    engine = _engine(canonical, state_db, _plan_model())
    plan = await engine.plan_deliverable(athlete_id=ATHLETE_A, request="week plan")
    assert plan.status is RunStatus.AWAITING_APPROVAL and plan.interrupt_id is not None

    # The re-grounding extracts a NAME claim per workout the edited body prescribes.
    engine._model.set_response(  # type: ignore[attr-defined]
        _ClaimSchema(
            claims=[
                _ExtractedClaim(
                    kind=ClaimKind.NAME, text="magic super workout", as_of="magic super workout"
                ),
                _ExtractedClaim(
                    kind=ClaimKind.NAME, text="threshold intervals", as_of="threshold intervals"
                ),
            ]
        )
    )
    resumed = await engine.decision(
        athlete_id=ATHLETE_A,
        thread_id=plan.thread_id,
        interrupt_id=plan.interrupt_id,
        decision="edit",
        edited_plan="Do magic super workout then threshold intervals.",
    )
    assert resumed.status is RunStatus.DEGRADED, "a non-fully-grounded edit must be rejected"
    assert "magic super workout" not in resumed.plan_text, "invented workout must never ship"
    # The untrusted edit body is NOT what is delivered; the pre-edit grounded plan is.
    assert "Do magic super workout then threshold intervals." not in resumed.plan_text


async def test_plan_edit_fully_canonical_grounds_and_completes(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """F-EDIT (H3): an edit whose every claim grounds is ACCEPTED -> COMPLETED with the edit body.

    The edited plan names ONLY canonical workouts, so re-grounding decides ``PROCEED`` with
    non-empty grounded text — the edit is accepted and its body becomes the delivered plan (R3).
    """
    engine = _engine(canonical, state_db, _plan_model())
    plan = await engine.plan_deliverable(athlete_id=ATHLETE_A, request="week plan")
    assert plan.status is RunStatus.AWAITING_APPROVAL and plan.interrupt_id is not None

    engine._model.set_response(  # type: ignore[attr-defined]
        _ClaimSchema(
            claims=[
                _ExtractedClaim(kind=ClaimKind.NAME, text="recovery ride", as_of="recovery ride"),
                _ExtractedClaim(
                    kind=ClaimKind.NAME, text="threshold intervals", as_of="threshold intervals"
                ),
            ]
        )
    )
    resumed = await engine.decision(
        athlete_id=ATHLETE_A,
        thread_id=plan.thread_id,
        interrupt_id=plan.interrupt_id,
        decision="edit",
        edited_plan="Do a recovery ride then threshold intervals.",
    )
    assert resumed.status is RunStatus.COMPLETED
    assert "recovery ride" in resumed.plan_text and "threshold intervals" in resumed.plan_text


# --- F-409 / F-404 / F-XID (fail-closed decision guard) ---------------------------------


async def test_double_decision_is_refused(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """F-409: a second decision on a consumed interrupt is refused (no double-resume, CKPT-R9)."""
    engine = _engine(canonical, state_db, _plan_model())
    plan = await engine.plan_deliverable(athlete_id=ATHLETE_A, request="week plan")
    assert plan.interrupt_id is not None
    await engine.decision(
        athlete_id=ATHLETE_A,
        thread_id=plan.thread_id,
        interrupt_id=plan.interrupt_id,
        decision="approve",
    )
    with pytest.raises(DecisionRefused):
        await engine.decision(
            athlete_id=ATHLETE_A,
            thread_id=plan.thread_id,
            interrupt_id=plan.interrupt_id,
            decision="approve",
        )


async def test_unknown_interrupt_is_refused(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """F-404: a decision against an interrupt id that was never recorded is refused (CKPT-R9)."""
    engine = _engine(canonical, state_db, _plan_model())
    plan = await engine.plan_deliverable(athlete_id=ATHLETE_A, request="week plan")
    with pytest.raises(DecisionRefused):
        await engine.decision(
            athlete_id=ATHLETE_A,
            thread_id=plan.thread_id,
            interrupt_id="int-never-recorded",
            decision="approve",
        )


async def test_cross_athlete_decision_is_refused(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """F-XID: athlete B cannot consume athlete A's interrupt; A's plan stays consumable (CKPT-R9).

    The consume guard is athlete-scoped, so B's decision matches no row and is refused — never
    resuming A's run. A can still legitimately approve its own plan afterwards.
    """
    engine = _engine(canonical, state_db, _plan_model())
    plan = await engine.plan_deliverable(athlete_id=ATHLETE_A, request="week plan")
    assert plan.interrupt_id is not None

    with pytest.raises(DecisionRefused):
        await engine.decision(
            athlete_id=ATHLETE_B,
            thread_id=plan.thread_id,
            interrupt_id=plan.interrupt_id,
            decision="approve",
        )
    # A's interrupt was not collaterally consumed — A can still approve it.
    resumed = await engine.decision(
        athlete_id=ATHLETE_A,
        thread_id=plan.thread_id,
        interrupt_id=plan.interrupt_id,
        decision="approve",
    )
    assert resumed.status is RunStatus.COMPLETED
