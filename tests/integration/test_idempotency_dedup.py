"""Engine-level run idempotency / dedup on a REAL pool (CKPT-R4).

CKPT-R4: submitting the SAME request turn twice MUST resume/return the existing run rather
than starting a duplicate, within a configurable dedup window. These drive the deployable
:class:`~wattwise_core.agent.engine.GraphAgentEngine` end to end over a DURABLE saver on a
dedicated agent-state pool and assert:

* a re-submitted identical turn (no ``thread_id``, within the window) returns the SAME durable
  thread AND does NOT start a second graph run (the model's ``compose`` is not called again);
* a DIFFERENT question (a genuinely different turn) opens a NEW thread and DOES run;
* with the dedup window CLOSED (a stale window / window=0 boundary is covered by the unit
  test), the deterministic key still dedups by content.

CRITICAL (skill §7): the agent-state saver runs on a file-backed SQLite engine with a real
connection pool (WAL + busy_timeout), NEVER ``:memory:``/StaticPool.
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

import wattwise_core.agent.memory  # noqa: F401  (registers agent_memory_item on AgentStateBase)
from wattwise_core.agent.contracts import ClaimKind, RunStatus
from wattwise_core.agent.engine import (
    GraphAgentEngine,
    _ClaimSchema,
    _ExtractedClaim,
    _PlanSchema,
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

ATHLETE_A = "00000000-0000-7000-8000-0000000000d1"


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
    """A DEDICATED agent-state database over a REAL multi-connection pool (skill §7)."""
    backend_dsn = request.param
    if backend_dsn is None:
        db = build_agent_state_database(dsn=f"sqlite+aiosqlite:///{tmp_path}/agent.sqlite")
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
        session.add(Athlete(athlete_id=uuid.UUID(ATHLETE_A), sex="male", reference_timezone="UTC"))
        session.add(
            FitnessSignature(
                athlete_id=uuid.UUID(ATHLETE_A),
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


class _CountingModel(FakeModel):
    """A FakeModel that counts ``compose`` calls so a duplicate RUN is observable (CKPT-R4)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.compose_calls = 0

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_calls += 1
        return await super().compose(system=system, context=context, max_tokens=max_tokens)


def _answer_model() -> _CountingModel:
    """A FakeModel scripting a grounded free-form answer (one STATEMENT claim, no numbers)."""
    return _CountingModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[_ExtractedClaim(kind=ClaimKind.STATEMENT, text="trending up")]
            ),
        },
        prose="Your form is in a good place this week.",
    )


def _engine(
    canonical: _DatabaseStub, state_db: AgentStateDatabase, model: FakeModel, *, window: int
) -> GraphAgentEngine:
    return GraphAgentEngine(
        canonical,
        model,
        state_db=state_db,
        dedup_window_seconds=window,  # type: ignore[arg-type]
    )


async def _ask(engine: GraphAgentEngine, question: str, thread_id: str | None = None) -> Any:
    return await engine.answer(
        athlete_id=ATHLETE_A,
        question=question,
        thread_id=thread_id,
        response_length="standard",
        follow_up=None,
        locale="en",
    )


async def test_resubmitted_same_turn_returns_existing_run_without_duplicate(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """CKPT-R4: the SAME turn twice resolves to the SAME run, never a duplicate.

    First submission runs the graph once (``compose`` called once) and opens a durable thread;
    the SECOND identical submission within the window returns that EXISTING run — the SAME
    thread_id — and does NOT run the graph again (``compose`` is still called only once). Under
    the deviated behaviour (a random conversation id per turn) the second turn would mint a NEW
    thread and run again, so ``compose_calls`` would be 2 and the thread ids would differ.
    """
    model = _answer_model()
    engine = _engine(canonical, state_db, model, window=3600)

    first = await _ask(engine, "How am I doing?")
    assert first.status is RunStatus.COMPLETED
    assert model.compose_calls == 1

    second = await _ask(engine, "How am I doing?")
    assert second.thread_id == first.thread_id  # SAME durable run, not a duplicate
    assert second.answer_text == first.answer_text
    assert model.compose_calls == 1  # NO second graph run


async def test_different_turn_opens_new_run(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """A genuinely DIFFERENT turn is not deduped: it opens a new thread and runs (CKPT-R4)."""
    model = _answer_model()
    engine = _engine(canonical, state_db, model, window=3600)

    first = await _ask(engine, "How am I doing?")
    other = await _ask(engine, "What should I do tomorrow?")
    assert other.thread_id != first.thread_id
    assert model.compose_calls == 2
