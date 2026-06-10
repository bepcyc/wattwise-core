"""Integration tests: the agent CONSUMES active canonical goals (GBO-R38 / API-R32 / API-R35).

doc 20 GBO-R38 hands goal-aware PLANNING to the agent/analytics specs ("the store enforces typing
only; goal-aware PLANNING ... is owned by the agent ... which read this entity"); doc 60 API-R32 /
API-R35 confirm "goal-aware plan generation reads those goals through the agent path". These tests
prove the deployable :class:`GraphAgentEngine` reads the athlete's ACTIVE canonical ``Goal`` rows
and FLOWS them into the agent inputs so they reach the plan / load-review (digest) compose context
— a non-active (terminal) goal does NOT flow, and a foreign athlete's goal never leaks (AGT-SEC-R1).

Run over a REAL multi-connection pool (file-SQLite + WAL + busy_timeout, skill §7 — NEVER
``:memory:``/StaticPool): the engine opens a per-deliverable canonical session through the pool, so
the active-goal read exercises the same arrangement production uses. A context-capturing FakeModel
records the prose ``context`` the compose node hands the model, which is exactly where consumed
goals must appear.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.contracts import ClaimKind
from wattwise_core.agent.engine import (
    GraphAgentEngine,
    _ClaimSchema,
    _ExtractedClaim,
    _PlanSchema,
)
from wattwise_core.agent.model import FakeModel
from wattwise_core.agent.state_db import AgentStateDatabase, build_agent_state_database
from wattwise_core.domain.enums import GoalStatus, GoalType, SignatureOrigin
from wattwise_core.persistence.models import (
    Athlete,
    Base,
    FitnessSignature,
    Goal,
    SourceDescriptor,
    Sport,
)

pytestmark = pytest.mark.integration

UTC = _dt.UTC

ATHLETE_A = "00000000-0000-7000-8000-0000000000a1"
ATHLETE_B = "00000000-0000-7000-8000-0000000000a2"


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + long busy_timeout per SQLite connection so the real pool serialises writers."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


class _CapturingModel(FakeModel):
    """A FakeModel that records every compose ``context`` so a test can assert what flowed in."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.contexts: list[str] = []

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        """Record the context the compose node assembled, then return the canned prose."""
        self.contexts.append(context)
        return await super().compose(system=system, context=context, max_tokens=max_tokens)


def _plan_model() -> _CapturingModel:
    """A FakeModel scripting a PLAN whose prescribed workout NAME is canonical (grounds)."""
    return _CapturingModel(
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


class _DatabaseStub:
    """A minimal canonical ``Database`` substitute over one REAL pooled engine factory."""

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


async def _seed(session: AsyncSession) -> None:
    """Seed two owners, each with an FTP signature; A gets an active + a terminal goal."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    session.add(
        SourceDescriptor(
            source_key="file_import", display_name="Activity files", kind="file_upload"
        )
    )
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
    # A's ACTIVE goal (must flow), A's ABANDONED goal (must NOT flow), B's active goal (foreign).
    session.add(
        Goal(
            goal_id=uuid.uuid4(),
            athlete_id=uuid.UUID(ATHLETE_A),
            sport="cycling",
            goal_type=GoalType.EVENT,
            title="Win the Dolomites gran fondo",
            target_event="Maratona dles Dolomites",
            target_date=_dt.date(2026, 7, 5),
            status=GoalStatus.ACTIVE,
        )
    )
    session.add(
        Goal(
            goal_id=uuid.uuid4(),
            athlete_id=uuid.UUID(ATHLETE_A),
            sport="cycling",
            goal_type=GoalType.PROCESS,
            title="Abandoned winter base block",
            status=GoalStatus.ABANDONED,
        )
    )
    session.add(
        Goal(
            goal_id=uuid.uuid4(),
            athlete_id=uuid.UUID(ATHLETE_B),
            sport="cycling",
            goal_type=GoalType.EVENT,
            title="Foreign secret objective",
            status=GoalStatus.ACTIVE,
        )
    )
    await session.commit()


@pytest_asyncio.fixture
async def canonical(tmp_path: Path) -> AsyncIterator[_DatabaseStub]:
    """A canonical store over a REAL file-SQLite pool (WAL), seeded with owners + goals."""
    dsn = f"sqlite+aiosqlite:///{tmp_path}/canonical.sqlite"
    engine = create_async_engine(dsn)
    event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        await _seed(session)
    try:
        yield _DatabaseStub(factory)
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def state_db(tmp_path: Path) -> AsyncIterator[AgentStateDatabase]:
    """A DEDICATED agent-state database over a REAL file-SQLite pool (WAL), separate engine."""
    dsn = f"sqlite+aiosqlite:///{tmp_path}/agent_state.sqlite"
    db = build_agent_state_database(dsn=dsn)
    event.listen(db.engine.sync_engine, "connect", _enable_sqlite_wal)
    await db.create_all()
    try:
        yield db
    finally:
        await db.dispose()


@pytest.mark.asyncio
async def test_plan_consumes_active_goal_into_compose_context(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """A PLAN run flows the athlete's ACTIVE goal into the compose context (GBO-R38 / API-R32).

    The deployed engine reads the athlete's active canonical ``Goal`` and surfaces it to the compose
    node so the plan is goal-aware — the goal's athlete-facing label and target event appear in the
    prose context the model drafts from. Mutation-proof: if the engine does not read active goals
    into the inputs, the goal text never reaches the context and this assertion fails.
    """
    model = _plan_model()
    engine = GraphAgentEngine(canonical, model, state_db=state_db)  # type: ignore[arg-type]
    await engine.plan_deliverable(athlete_id=ATHLETE_A, request="give me a week plan")
    joined = "\n".join(model.contexts)
    assert "Win the Dolomites gran fondo" in joined  # the active goal's label reached compose
    assert "Maratona dles Dolomites" in joined  # the target event reached compose too


@pytest.mark.asyncio
async def test_plan_does_not_consume_terminal_goal(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """Only ACTIVE goals flow; a terminal (abandoned) goal is NOT surfaced (GBO-R39 active-only)."""
    model = _plan_model()
    engine = GraphAgentEngine(canonical, model, state_db=state_db)  # type: ignore[arg-type]
    await engine.plan_deliverable(athlete_id=ATHLETE_A, request="give me a week plan")
    joined = "\n".join(model.contexts)
    assert "Abandoned winter base block" not in joined  # terminal goals are not planned toward


@pytest.mark.asyncio
async def test_plan_does_not_consume_foreign_goal(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """A foreign athlete's active goal never leaks into this athlete's plan (AGT-SEC-R1)."""
    model = _plan_model()
    engine = GraphAgentEngine(canonical, model, state_db=state_db)  # type: ignore[arg-type]
    await engine.plan_deliverable(athlete_id=ATHLETE_A, request="give me a week plan")
    joined = "\n".join(model.contexts)
    assert "Foreign secret objective" not in joined  # athlete-scoped read only (AUTH-R3)
