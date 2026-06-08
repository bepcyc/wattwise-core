"""Unit tests for the dedicated agent-state database (ARCH-R13, DEPLOY-R4, SPIKE-3).

Proves the two properties that make the agent-state store deadlock-safe and correct:

* **StaticPool sharing** — over sqlite ``:memory:`` the :class:`AgentStateDatabase`
  uses a single shared in-memory DB, so a row written via one session from its factory
  is visible via another session from the SAME factory. A non-StaticPool ``:memory:``
  engine would give each connection its own empty DB and the saver (many sessions) would
  break — this test pins that StaticPool is actually used.
* **engine separation** — the agent-state engine is a DISTINCT object from the canonical
  ``Database`` engine; they NEVER share a pool (ARCH-R13). The separate pool is what
  removes the SPIKE-3 deadlock coupling.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.pool import StaticPool

import wattwise_core.agent.memory  # noqa: F401  (registers agent_memory_item on AgentStateBase)
from wattwise_core.agent.memory import MemoryItem, MemoryItemKind
from wattwise_core.agent.state_db import (
    AgentStateDatabase,
    build_agent_state_database,
    create_agent_state_engine,
)
from wattwise_core.config import load_settings
from wattwise_core.persistence.engine import Database

ATHLETE = "00000000-0000-7000-8000-00000000000a"
MEMORY_DSN = "sqlite+aiosqlite:///:memory:"


def _memory_settings(dsn: str = MEMORY_DSN):  # type: ignore[no-untyped-def]
    """Dev settings carrying a sqlite DSN (in-memory by default; no external secrets)."""
    return load_settings(app__environment="development", database_dsn=dsn)


async def test_memory_database_uses_static_pool_shared_db() -> None:
    """``:memory:`` agent-state DB uses StaticPool -> one shared DB across sessions.

    Writing through one session and reading through ANOTHER session from the same factory
    must observe the row: only a StaticPool (single reused connection) makes the in-memory
    DB shared. A default-pooled ``:memory:`` engine would open a fresh empty DB per
    connection and the read would miss.
    """
    agent_db = AgentStateDatabase(_memory_settings())
    assert isinstance(agent_db.engine.pool, StaticPool)
    await agent_db.create_all()

    # Write via session #1.
    async with agent_db.session() as writer:
        writer.add(
            MemoryItem(
                athlete_id=uuid.UUID(ATHLETE),
                kind=MemoryItemKind.PREFERENCE,
                content="prefers morning rides",
            )
        )

    # Read via a SEPARATE session #2 from the SAME factory: must see the committed row,
    # which is only possible because StaticPool shares one in-memory DB.
    async with agent_db.session() as reader:
        rows = (await reader.execute(select(MemoryItem))).scalars().all()
    assert len(rows) == 1
    assert rows[0].content == "prefers morning rides"

    await agent_db.dispose()


async def test_agent_state_engine_is_distinct_from_canonical() -> None:
    """The agent-state engine/pool is a DISTINCT object from the canonical one (ARCH-R13).

    SPIKE-3 deadlock-freedom rests on this separation: the durable run's canonical
    connection and the saver's agent-state connection never come from the same pool.
    """
    settings = _memory_settings()
    canonical = Database(settings)
    agent_db = build_agent_state_database(settings)
    try:
        assert agent_db.engine is not canonical.engine
        assert agent_db.engine.pool is not canonical.engine.pool
        # The saver consumes exactly this session factory (CKPT-R1).
        assert agent_db.session_factory is not None
    finally:
        await canonical.dispose()
        await agent_db.dispose()


async def test_explicit_dsn_overrides_canonical(tmp_path: Path) -> None:
    """An explicit agent-state DSN is honoured (DEPLOY-R4 distinct write credential).

    DISCRIMINATING: settings carry a DIFFERENT DSN (a file-sqlite path, which yields a
    NON-StaticPool engine) while the explicit ``dsn=`` is the ``:memory:`` DSN (which
    yields a StaticPool). We assert the engine's pool is :class:`StaticPool`, so if the
    override were IGNORED and the builder fell back to the file-path settings, the pool
    would be a QueuePool and this assertion would FAIL — i.e. the test can actually tell
    the override apart from the fallback (the prior version passed the same string for
    both and could not).
    """
    file_dsn = f"sqlite+aiosqlite:///{tmp_path}/canonical.sqlite"
    settings = _memory_settings(dsn=file_dsn)
    agent_db = AgentStateDatabase(settings, dsn=MEMORY_DSN)
    try:
        # Override (:memory: -> StaticPool) won; the file-path settings fallback (QueuePool)
        # was NOT used.
        assert isinstance(agent_db.engine.pool, StaticPool)
        await agent_db.create_all()
    finally:
        await agent_db.dispose()


async def test_session_rolls_back_on_error() -> None:
    """``session()`` rolls back (fail-closed) when the body raises — no partial write.

    Writes a row inside ``async with db.session()`` then raises; the exception MUST
    propagate AND the row MUST NOT persist (the rollback arm, not the commit arm, ran).
    A SEPARATE later session reads zero rows, proving nothing was committed.
    """
    agent_db = AgentStateDatabase(_memory_settings())
    await agent_db.create_all()
    try:
        sentinel = RuntimeError("boom inside session body")
        with pytest.raises(RuntimeError) as excinfo:
            async with agent_db.session() as writer:
                writer.add(
                    MemoryItem(
                        athlete_id=uuid.UUID(ATHLETE),
                        kind=MemoryItemKind.PREFERENCE,
                        content="should be rolled back",
                    )
                )
                await writer.flush()  # send the INSERT so a missing rollback WOULD persist it
                raise sentinel
        assert excinfo.value is sentinel  # the original error propagated unchanged

        # A fresh session sees zero rows: the failed transaction was rolled back, not committed.
        async with agent_db.session() as reader:
            rows = (await reader.execute(select(MemoryItem))).scalars().all()
        assert len(rows) == 0
    finally:
        await agent_db.dispose()


def test_missing_dsn_fails_closed() -> None:
    """No resolvable DSN -> fail closed (ARCH-R13/BOOT-R4), never an undefined engine.

    The config layer normally guarantees ``database_dsn`` (it raises ConfigError if it is
    absent), so we exercise the engine builder's own defensive guard with a stub Settings
    whose ``database_dsn`` is ``None``: it MUST refuse rather than build an engine on a
    missing DSN.
    """

    class _NoDsnSettings:
        database_dsn = None

    with pytest.raises(RuntimeError):
        create_agent_state_engine(_NoDsnSettings())  # type: ignore[arg-type]
