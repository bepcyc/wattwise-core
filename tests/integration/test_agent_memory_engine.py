"""Engine-level durable MEMORY read/erase seam tests on a REAL pool (MEM-R3/-R4, API).

These drive the athlete-scoped memory list / get / delete / erase seam the deployable
:class:`~wattwise_core.agent.engine.GraphAgentEngine` exposes (``engine.list_memory`` /
``get_memory`` / ``delete_memory`` / ``erase_memory``) against the dedicated agent-state store, the
read+erase surface the GET/DELETE ``/v1/agent/memory`` endpoints need. They pin the guarantees that
make the seam trustworthy:

* **scope** — the seam lists / fetches / deletes only the SERVER-DERIVED owner's rows; another
  athlete's id is never listed, returned, or deleted, and a foreign / unknown id reads as absent
  (``None`` / a no-op delete) — never a cross-identity leak (MEM-R3 / AGT-SEC-R1, fail-closed);
* **erasure** — a per-id delete and a whole-athlete erase actually remove rows (MEM-R3 privacy
  MUST), and the count is reported honestly;
* **ordering** — the list is deterministic (newest first), so a paged read is reproducible.

CRITICAL (skill §7): the agent-state store runs on a **file-backed SQLite engine with a real
connection pool** (WAL + busy_timeout), NEVER ``:memory:``/``StaticPool`` — a single-connection
setup false-greens scope/erase behaviour. PostgreSQL / MariaDB legs run when ``WATTWISE_PG_DSN`` /
``WATTWISE_MARIADB_DSN`` are set, proving the behaviour is backend-portable.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import wattwise_core.agent.memory  # noqa: F401  (registers agent_memory_item on AgentStateBase)
from wattwise_core.agent.engine import GraphAgentEngine
from wattwise_core.agent.memory import MemoryItemKind, OssMemoryStore
from wattwise_core.agent.model import FakeModel
from wattwise_core.agent.state_db import AgentStateDatabase, build_agent_state_database
from wattwise_core.agent.state_store import AgentStateBase

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

pytestmark = pytest.mark.integration

ATHLETE_A = "00000000-0000-7000-8000-0000000000d1"
ATHLETE_B = "00000000-0000-7000-8000-0000000000d2"


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

    SQLite is file-backed (real pool) with WAL — deliberately NOT ``:memory:``/StaticPool.
    PG/MariaDB use their throwaway DSN and are reset to an empty agent-state schema first.
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


class _DatabaseStub:
    """A minimal canonical ``Database`` substitute (the memory seam never reads it)."""

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
    """A throwaway in-memory canonical store (unused by the memory seam, required by the engine)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield _DatabaseStub(factory)
    finally:
        await engine.dispose()


def _engine(canonical: _DatabaseStub, state_db: AgentStateDatabase) -> GraphAgentEngine:
    """The deployable engine over the throwaway canonical store + REAL-pool agent-state store."""
    return GraphAgentEngine(canonical, FakeModel(), state_db=state_db)  # type: ignore[arg-type]


async def _write(
    state_db: AgentStateDatabase, *, athlete_id: str, content: str
) -> str:
    """Write one trusted PREFERENCE episode for an athlete; returns its memory item id."""
    async with state_db.session() as session:
        store = OssMemoryStore(session)
        item = await store.write_episode(
            athlete_id=athlete_id,
            kind=MemoryItemKind.PREFERENCE,
            content=content,
            trusted=True,
        )
    return item.memory_item_id


async def test_list_memory_is_owner_scoped_and_newest_first(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """List returns only the owner's rows, newest first; another athlete's are excluded (MEM-R3)."""
    await _write(state_db, athlete_id=ATHLETE_A, content="prefers morning rides")
    await _write(state_db, athlete_id=ATHLETE_A, content="hates the trainer")
    await _write(state_db, athlete_id=ATHLETE_B, content="b-only secret")
    engine = _engine(canonical, state_db)
    rows = await engine.list_memory(athlete_id=ATHLETE_A)
    contents = [r.content for r in rows]
    assert contents == ["hates the trainer", "prefers morning rides"]  # newest first
    assert "b-only secret" not in contents  # cross-athlete row never listed (scope)


async def test_get_memory_returns_owner_row_and_hides_foreign(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """Get returns the owner's row by id; a foreign / unknown id reads as absent (fail-closed).

    A's own id resolves; B's id queried under A returns ``None`` (never disclosed — the router maps
    that to a 404), and a non-UUID token is also absent rather than an error (AGT-SEC-R1).
    """
    a_id = await _write(state_db, athlete_id=ATHLETE_A, content="prefers tempo work")
    b_id = await _write(state_db, athlete_id=ATHLETE_B, content="b-only secret")
    engine = _engine(canonical, state_db)
    got = await engine.get_memory(athlete_id=ATHLETE_A, memory_item_id=a_id)
    assert got is not None and got.content == "prefers tempo work"
    assert await engine.get_memory(athlete_id=ATHLETE_A, memory_item_id=b_id) is None  # foreign row
    assert await engine.get_memory(athlete_id=ATHLETE_A, memory_item_id="not-a-uuid") is None


async def test_delete_memory_erases_owner_row_only(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """Per-id delete removes the owner's row (True) and refuses a foreign id (False) (MEM-R3).

    Deleting A's own id returns ``True`` and the row is gone; deleting B's id UNDER A returns
    ``False`` and B's row survives — a cross-athlete delete erases nothing (privacy MUST, PRIV-R8).
    """
    a_id = await _write(state_db, athlete_id=ATHLETE_A, content="prefers tempo work")
    b_id = await _write(state_db, athlete_id=ATHLETE_B, content="b-only secret")
    engine = _engine(canonical, state_db)
    assert await engine.delete_memory(athlete_id=ATHLETE_A, memory_item_id=a_id) is True
    assert await engine.get_memory(athlete_id=ATHLETE_A, memory_item_id=a_id) is None  # gone
    # B's row queried under A deletes nothing and B still owns it.
    assert await engine.delete_memory(athlete_id=ATHLETE_A, memory_item_id=b_id) is False
    assert await engine.get_memory(athlete_id=ATHLETE_B, memory_item_id=b_id) is not None


async def test_delete_unknown_id_is_false_not_error(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """Deleting an unknown / non-UUID id returns ``False`` (fail-closed), never raising (MEM-R3)."""
    engine = _engine(canonical, state_db)
    assert await engine.delete_memory(athlete_id=ATHLETE_A, memory_item_id="not-a-uuid") is False
    unknown = "00000000-0000-7000-8000-0000000000ff"
    assert await engine.delete_memory(athlete_id=ATHLETE_A, memory_item_id=unknown) is False


async def test_erase_memory_clears_owner_only(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """Whole-athlete erase removes ALL of the owner's rows + reports the count; B is untouched."""
    await _write(state_db, athlete_id=ATHLETE_A, content="one")
    await _write(state_db, athlete_id=ATHLETE_A, content="two")
    await _write(state_db, athlete_id=ATHLETE_B, content="b-only secret")
    engine = _engine(canonical, state_db)
    erased = await engine.erase_memory(athlete_id=ATHLETE_A)
    assert erased == 2  # both of A's rows removed, honestly counted
    assert await engine.list_memory(athlete_id=ATHLETE_A) == []
    # B's row is untouched by A's erasure (scope).
    b_rows = await engine.list_memory(athlete_id=ATHLETE_B)
    assert [r.content for r in b_rows] == ["b-only secret"]


async def test_list_memory_paginates_with_limit_and_offset(
    canonical: _DatabaseStub, state_db: AgentStateDatabase
) -> None:
    """Limit + offset page the deterministic newest-first list without overlap (MEM-R4)."""
    for i in range(5):
        await _write(state_db, athlete_id=ATHLETE_A, content=f"item-{i}")
    engine = _engine(canonical, state_db)
    page1 = await engine.list_memory(athlete_id=ATHLETE_A, limit=2, offset=0)
    page2 = await engine.list_memory(athlete_id=ATHLETE_A, limit=2, offset=2)
    assert [r.content for r in page1] == ["item-4", "item-3"]  # newest first
    assert [r.content for r in page2] == ["item-2", "item-1"]
    assert {r.memory_item_id for r in page1}.isdisjoint({r.memory_item_id for r in page2})
