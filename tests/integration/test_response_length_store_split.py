"""End-to-end store-split proof: PUT /v1/user-settings/response-length reaches the RUN path.

The HIGH store-split fix (doc 50 VOICE-R8 §382 / MEM-R1): the persisted answer-length default is an
agent-interaction preference in the dedicated AGENT-STATE store, NOT a canonical §3 master-data
column. The bug was a split source — the run path READ the agent-state preference while
``PUT /v1/user-settings/response-length`` WROTE the canonical ``Athlete.default_response_length``
column, so a saved preference NEVER reached the run.

This test proves the fix END TO END on a REAL agent-state pool (file-SQLite + WAL, NEVER
``:memory:``/``StaticPool`` — skill §7): one shared :class:`GraphAgentEngine` backs BOTH the
user-settings ``response_length_store`` seam AND ``engine.answer``. We PUT ``detailed`` through the
real HTTP router, then call ``engine.answer(..., response_length=None)`` and assert the run resolved
the PERSISTED ``detailed`` — not the ``standard`` fallback. ``answer_question`` is stubbed to
CAPTURE the resolved ``response_length`` the engine threads in (the real graph/LLM is out of scope;
assertion is purely that the persisted preference reaches the run-path default, the store-split).

MUTATION-PROOF: with NO PUT, the same call resolves ``standard``; only the PUT flips it to
``detailed``. So if the write did not reach the read (the original bug, or a regression), the
``detailed`` assertion fails — the test cannot pass without the write actually landing in the store
the run reads.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import wattwise_core.agent.engine as engine_module
from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import AgentAnswer
from wattwise_core.agent.engine import GraphAgentEngine
from wattwise_core.agent.model import FakeModel
from wattwise_core.agent.state_db import AgentStateDatabase, build_agent_state_database
from wattwise_core.api.auth import Principal, Scope, authenticate
from wattwise_core.api.errors import install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import user_settings as settings_router
from wattwise_core.persistence.models import Athlete, Base

pytestmark = pytest.mark.integration

ATHLETE = "00000000-0000-7000-8000-0000000000e1"


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + long busy_timeout per connection so the real agent-state pool serialises writers."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


class _DatabaseStub:
    """A minimal canonical ``Database`` substitute (the response-length seam never reads it)."""

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


@dataclass
class Env:
    """The wired user-settings app + the shared engine + its real-pool agent-state store."""

    client: AsyncClient
    engine: GraphAgentEngine
    state_db: AgentStateDatabase


@pytest_asyncio.fixture
async def env(tmp_path: Path) -> AsyncIterator[Env]:
    """One shared engine backing BOTH the user-settings seam and ``engine.answer`` (real pool)."""
    canonical_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with canonical_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    canonical_factory = async_sessionmaker(
        canonical_engine, expire_on_commit=False, class_=AsyncSession
    )
    async with canonical_factory() as session:
        session.add(Athlete(athlete_id=uuid.UUID(ATHLETE), sex="male", reference_timezone="UTC"))
        await session.commit()

    state_db = build_agent_state_database(dsn=f"sqlite+aiosqlite:///{tmp_path}/agent.sqlite")
    event.listen(state_db.engine.sync_engine, "connect", _enable_sqlite_wal)
    await state_db.create_all()
    engine = GraphAgentEngine(
        _DatabaseStub(canonical_factory),  # type: ignore[arg-type]
        FakeModel(),
        state_db=state_db,
    )

    app = FastAPI()
    app.state.rate_limiter = RateLimiter()
    install_error_handlers(app)
    app.include_router(settings_router.router)
    app.dependency_overrides.update(
        {
            authenticate: lambda: Principal(subject=ATHLETE, scopes=frozenset(Scope)),
            settings_router.require_read_scope: lambda: None,
            settings_router.require_write_scope: lambda: None,
            settings_router.current_athlete_id: lambda: ATHLETE,
            settings_router.response_length_store: lambda: engine,
        }
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        yield Env(client, engine, state_db)
    await state_db.dispose()
    await canonical_engine.dispose()


def _capture_answer_length(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Stub ``answer_question`` to CAPTURE the resolved ``response_length`` the engine threads in.

    The real graph/LLM run is out of scope for the store-split assertion: we only need the value the
    engine resolved from the persisted preference and passed into the deliverable. Returns the list
    the stub appends each captured length to.
    """
    captured: list[str | None] = []

    async def _stub(*_args: Any, response_length: Any = None, **_kwargs: Any) -> AgentAnswer:
        captured.append(response_length)
        return AgentAnswer(
            status=RunStatus.COMPLETED,
            thread_id="t",
            answer_html="<p>ok</p>",
            answer_text="ok",
        )

    monkeypatch.setattr(engine_module, "answer_question", _stub)
    return captured


async def test_put_response_length_reaches_the_run_default(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PUT ``detailed`` → ``engine.answer(response_length=None)`` resolves ``detailed``.

    The persisted preference written through the HTTP PUT MUST become the run-path default the
    engine applies when a request carries no per-request length (VOICE-R8 §382 / MEM-R1). This is
    the exact path the HIGH fix repairs: write and read now share ONE agent-state source.
    """
    put = await env.client.put(
        "/v1/user-settings/response-length", json={"response_length": "detailed"}
    )
    assert put.status_code == 200

    captured = _capture_answer_length(monkeypatch)
    await env.engine.answer(
        athlete_id=ATHLETE,
        question="how am I doing?",
        thread_id=None,
        response_length=None,
        follow_up=None,
        locale="en",
    )
    assert captured == ["detailed"], "the run applied the PERSISTED preference, not the fallback"


async def test_run_default_is_standard_without_a_put(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION-PROOF: with NO PUT, the same call resolves ``standard`` — only the PUT flips it.

    This pins that the ``detailed`` result above is CAUSED by the write reaching the read: absent
    the write the run-path default is the spec fallback ``standard``. A write that did not reach the
    read (the original store-split bug) would leave BOTH cases ``standard`` and fail the test above.
    """
    captured = _capture_answer_length(monkeypatch)
    await env.engine.answer(
        athlete_id=ATHLETE,
        question="how am I doing?",
        thread_id=None,
        response_length=None,
        follow_up=None,
        locale="en",
    )
    assert captured == ["standard"], "no persisted preference → the VOICE-R8 default standard"
