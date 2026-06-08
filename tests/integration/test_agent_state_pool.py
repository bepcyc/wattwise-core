"""SPIKE-3: a dedicated agent-state pool is deadlock-free where a SHARED pool deadlocks.

What this proves (and what it does NOT):

The durable agent runtime can hold one connection (the run's own transaction on the
canonical store) WHILE the checkpointer concurrently needs a second connection to persist
a checkpoint after every node transition (CKPT-R1). If BOTH connections are drawn from the
SAME pool at ``pool_size=1, max_overflow=0``, the second checkout can never be satisfied —
the first connection is still held — so it blocks forever: a pool-exhaustion DEADLOCK.

We reproduce that minimal "hold-one-while-needing-a-second" shape directly:

* **SHARED arrangement** (one engine, pool_size=1): a task checks out the single connection
  (runs a query under an open session A), then, still holding it, tries to open session B
  from the SAME engine and run a query. B can never get a connection -> the whole coroutine
  never returns. We assert it TIMES OUT (``asyncio.TimeoutError``) — the observable signature
  of the deadlock.
* **DEDICATED arrangement**: the outer connection comes from engine-1 (canonical-like) and
  the inner (saver) connection comes from a SEPARATE :class:`AgentStateDatabase` engine. Both
  pools have their own connection available, so the nested operation completes. We assert it
  finishes well under a generous timeout.

HONESTY / what this does NOT prove:
* The timeout is a *proxy* for "blocked forever" — in principle a pathologically slow but
  non-deadlocked operation could also time out. We make that distinction reliable by (a)
  using a SHORT 2s timeout for the deadlock (a single trivial query that would otherwise
  return in <50ms) and a SEPARATE, generous 5s timeout for the dedicated case, and (b)
  asserting the dedicated case returns its real result, not merely "did not time out". The
  deadlock is structural (a connection that is provably never released within the coroutine),
  not load-dependent, so the 2s timeout is not racing a slow query.
* This is a pool-mechanics proof, not an end-to-end durable-graph proof (that is SPIKE-1 /
  ``test_durable_resume.py``). It isolates the ONE property Step 2 adds: the separate pool.
* sqlite ``:memory:``/StaticPool is always a single shared connection, so it cannot show the
  contrast — we use file-backed sqlite (real QueuePool) and the container PG/MariaDB legs.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from wattwise_core.agent.state_db import AgentStateDatabase

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

pytestmark = pytest.mark.integration

# A trivial query that returns immediately on every backend (proves "would be instant if a
# connection were available" — so a timeout means blocked-on-connection, not slow-query).
_PING = text("SELECT 1")

_DEADLOCK_TIMEOUT_S = 2.0  # the shared pool must block PAST this (deadlock signature)
_COMPLETE_TIMEOUT_S = 5.0  # the dedicated pool must finish WELL under this


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + busy_timeout so file-sqlite concurrent access serializes, not lock-errors."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


def _make_engine(backend_dsn: str | None, tmp_path: Path, *, tag: str) -> AsyncEngine:
    """A pool_size=1, max_overflow=0 engine on the chosen backend (the contention setup).

    ``pool_size=1, max_overflow=0`` is the minimal pool that can be exhausted: there is
    exactly ONE connection, so holding it and asking for a second deadlocks (shared case).
    File-sqlite uses ``tag`` to get its own DB file so the two dedicated-case engines are
    truly separate databases as well as separate pools.
    """
    if backend_dsn is None:
        dsn = f"sqlite+aiosqlite:///{tmp_path}/{tag}.sqlite"
        engine = create_async_engine(
            dsn,
            connect_args={"timeout": 30},
            pool_size=1,
            max_overflow=0,
        )
        event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
        return engine
    return create_async_engine(backend_dsn, pool_size=1, max_overflow=0)


def _engine_backends() -> list[ParameterSet]:
    """File-SQLite always; PG/MariaDB only when their throwaway DSN env var is set.

    Mirrors ``test_durable_resume.py`` so the PG/MariaDB legs run under the db-portability
    job. ``None`` selects the per-test file-SQLite engine; a string is the DSN verbatim.
    """
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


@pytest_asyncio.fixture(params=_engine_backends())
async def backend_dsn(request: pytest.FixtureRequest) -> AsyncIterator[str | None]:
    """The chosen backend DSN (``None`` => per-test file-sqlite)."""
    yield request.param


async def _checkout_then_inner(
    outer_factory: async_sessionmaker[AsyncSession],
    inner_factory: async_sessionmaker[AsyncSession],
) -> int:
    """Hold one connection (session A) while needing a second (session B), then return.

    Opens session A and runs a query so its connection is genuinely CHECKED OUT (asyncpg /
    aiomysql / aiosqlite all check out lazily on first execute), and KEEPS it open. While
    still holding A, opens session B from ``inner_factory`` and runs a query. If
    ``inner_factory`` shares A's single-connection pool, B can never get a connection and
    this coroutine never returns (deadlock). If it has its own pool, B proceeds and we get a
    result. Returns the inner query's scalar so the caller asserts real completion.
    """
    async with outer_factory() as session_a:
        # Force the single connection to be checked out and held for the whole block.
        await session_a.execute(_PING)
        async with inner_factory() as session_b:
            result = await session_b.execute(_PING)
            return int(result.scalar_one())


async def test_shared_pool_deadlocks(
    backend_dsn: str | None, tmp_path: Path
) -> None:
    """SHARED pool (one engine, size=1): holding the connection while needing a second hangs.

    The inner session draws from the SAME single-connection pool as the held outer session,
    so it can never check out a connection -> the coroutine blocks forever -> we observe the
    deadlock as a TIMEOUT. (A non-deadlocked nested query would return in milliseconds.)
    """
    engine = _make_engine(backend_dsn, tmp_path, tag="shared")
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        with pytest.raises(asyncio.TimeoutError):
            # SAME factory for outer and inner -> one shared connection -> deadlock.
            await asyncio.wait_for(
                _checkout_then_inner(factory, factory),
                timeout=_DEADLOCK_TIMEOUT_S,
            )
    finally:
        await engine.dispose()


async def test_dedicated_pool_is_deadlock_free(
    backend_dsn: str | None, tmp_path: Path
) -> None:
    """DEDICATED pools: the saver's own AgentStateDatabase pool removes the contention.

    Outer (canonical-like) connection from engine-1; inner (saver) connection from a SEPARATE
    :class:`AgentStateDatabase` engine. CRITICALLY, the agent-state engine is built at the
    SAME ``pool_size=1, max_overflow=0`` as the shared arm, so SEPARATION is the ONLY
    difference between the two arms (not spare default-pool capacity): if the inner factory
    drew from the canonical single-connection pool it would deadlock exactly like
    ``test_shared_pool_deadlocks``; because each arm is its own size-1 pool, each has its own
    connection available and the nested operation COMPLETES — proving Step 2's separate pool
    is what makes a durable run + checkpointer deadlock-free even at pool_size=1
    (SPIKE-3, ARCH-R13/DEPLOY-R4).
    """
    canonical_engine = _make_engine(backend_dsn, tmp_path, tag="canonical")
    canonical_factory = async_sessionmaker(
        canonical_engine, expire_on_commit=False, class_=AsyncSession
    )
    # The dedicated agent-state DB owns its OWN engine/pool (separate DB file for sqlite),
    # ALSO at pool_size=1/max_overflow=0 so the ONLY variable vs the shared arm is separation.
    agent_dsn = (
        f"sqlite+aiosqlite:///{tmp_path}/agent_state.sqlite"
        if backend_dsn is None
        else backend_dsn
    )
    agent_db = AgentStateDatabase(dsn=agent_dsn, pool_size=1, max_overflow=0)
    # WAL + busy_timeout for the file-sqlite agent engine too (mirrors _make_engine), so its
    # size-1 pool serializes cleanly rather than raising "database is locked".
    if backend_dsn is None:
        event.listen(agent_db.engine.sync_engine, "connect", _enable_sqlite_wal)
    try:
        assert agent_db.engine is not canonical_engine
        assert agent_db.engine.pool is not canonical_engine.pool
        # Both arms are size-1 pools: separation, not spare capacity, is the proven variable.
        assert agent_db.engine.pool.size() == 1
        assert canonical_engine.pool.size() == 1
        result = await asyncio.wait_for(
            _checkout_then_inner(canonical_factory, agent_db.session_factory),
            timeout=_COMPLETE_TIMEOUT_S,
        )
        assert result == 1, "the inner saver query completed on its dedicated pool"
    finally:
        await agent_db.dispose()
        await canonical_engine.dispose()
