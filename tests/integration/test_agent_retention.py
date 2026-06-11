"""Agent-state retention-window sweeper for durable checkpoints (CKPT-R8, PRIV-R7).

CKPT-R8: checkpoints MUST expire per the configured retention window (the PRIV-R7 agent-state
retention category covering BOTH (a) durable run checkpoints/threads/writes/interrupts AND
(b) durable athlete memory, which MAY hold health-adjacent content per MEM-R3). These run on a
REAL file-backed SQLite pool (WAL) — never ``:memory:``/StaticPool — and assert:

* rows OLDER than the configured window are deleted, NEWER rows survive — for BOTH the
  checkpoint sub-category AND ``MemoryItem`` (PRIV-R7 sub-category (b));
* a non-positive window (``0`` = retain indefinitely) sweeps NOTHING (memory included);
* the sweep deletes children-first so the foreign keys hold (writes/interrupts/checkpoints
  before their thread).

Per-athlete PURGE (the other half of CKPT-R8) is fulfilled by ``erase_athlete`` and proven by
``tests/integration/test_erasure_executor.py`` / ``test_gdpr_erasure.py``; this file adds only
the TIME-WINDOW expiry that GAP_SPEC found missing.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.memory import MemoryItem, MemoryItemKind
from wattwise_core.agent.retention import sweep_expired_checkpoints
from wattwise_core.agent.state_store import (
    AgentCheckpoint,
    AgentInterrupt,
    AgentStateBase,
    AgentThread,
    AgentWrite,
)

pytestmark = pytest.mark.integration

ATHLETE = uuid.UUID("00000000-0000-7000-8000-00000000000a")
NOW = _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC)


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + busy_timeout per connection so the file pool serialises (real pool)."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Session factory over a REAL file-SQLite pool (WAL + FK enforcement)."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/agent.sqlite", connect_args={"timeout": 30}
    )
    event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
    async with engine.begin() as conn:
        await conn.run_sync(AgentStateBase.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    finally:
        await engine.dispose()


async def _seed_thread(session: AsyncSession, thread_id: str, *, created_at: _dt.datetime) -> None:
    """Seed a thread + checkpoint + write + interrupt + memory item, all at ``created_at``."""
    session.add(
        AgentThread(
            thread_id=thread_id,
            athlete_id=ATHLETE,
            conversation_id=thread_id,
            created_at=created_at,
        )
    )
    await session.flush()  # the parent thread must exist before its FK children (FK ON)
    session.add(
        AgentCheckpoint(
            thread_id=thread_id,
            checkpoint_ns="",
            checkpoint_id=f"{thread_id}-cp",
            athlete_id=ATHLETE,
            schema_version=2,
            checkpoint_type="msgpack",
            checkpoint_blob=b"x",
            metadata_blob={},
            created_at=created_at,
        )
    )
    session.add(
        AgentWrite(
            thread_id=thread_id,
            checkpoint_ns="",
            checkpoint_id=f"{thread_id}-cp",
            task_id="t1",
            idx=0,
            channel="draft",
            value_type="msgpack",
            value_blob=b"y",
            created_at=created_at,
        )
    )
    session.add(
        AgentInterrupt(
            thread_id=thread_id,
            athlete_id=ATHLETE,
            interrupt_id=f"{thread_id}-int",
            status="live",
            created_at=created_at,
        )
    )
    # PRIV-R7 sub-category (b): durable athlete memory (MAY hold health-adjacent content, MEM-R3).
    session.add(
        MemoryItem(
            athlete_id=ATHLETE,
            kind=MemoryItemKind.CONSTRAINT,
            content=f"{thread_id} left-knee injury — avoid high-intensity intervals",
            inferred=False,
            created_at=created_at,
        )
    )


async def _count(session: AsyncSession, model: type[Any]) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def test_sweep_expires_old_checkpoints_keeps_recent(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """Rows older than the configured window are expired; newer rows survive (CKPT-R8)."""
    old_at = NOW - _dt.timedelta(days=120)
    fresh_at = NOW - _dt.timedelta(days=10)
    async with factory() as session:
        await _seed_thread(session, "old-thread", created_at=old_at)
        await _seed_thread(session, "fresh-thread", created_at=fresh_at)
        await session.commit()

    async with factory() as session:
        report = await sweep_expired_checkpoints(session, retention_days=90, now=NOW)
        await session.commit()

    assert report.deleted_threads == 1
    assert report.deleted_checkpoints == 1
    assert report.deleted_writes == 1
    assert report.deleted_interrupts == 1
    assert report.deleted_memory == 1  # PRIV-R7 (b): the old durable-memory row is expired
    async with factory() as session:
        assert await _count(session, AgentThread) == 1  # only the fresh thread remains
        assert await _count(session, AgentCheckpoint) == 1
        assert await _count(session, AgentWrite) == 1
        assert await _count(session, AgentInterrupt) == 1
        assert await _count(session, MemoryItem) == 1  # the fresh memory row survives the window
        survivor = (await session.execute(select(AgentThread.thread_id))).scalar_one()
        assert survivor == "fresh-thread"
        memory_survivor = (await session.execute(select(MemoryItem.content))).scalar_one()
        assert memory_survivor.startswith("fresh-thread")  # the old, not the fresh, was swept


async def test_zero_window_retains_indefinitely(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """A non-positive window sweeps NOTHING (0 = retain indefinitely), mirroring raw_file_days."""
    async with factory() as session:
        await _seed_thread(session, "ancient", created_at=NOW - _dt.timedelta(days=10_000))
        await session.commit()

    async with factory() as session:
        report = await sweep_expired_checkpoints(session, retention_days=0, now=NOW)
        await session.commit()

    assert report.total == 0
    assert report.deleted_memory == 0  # a 0 window never sweeps durable memory either
    async with factory() as session:
        assert await _count(session, AgentThread) == 1
        assert await _count(session, AgentCheckpoint) == 1
        assert await _count(session, MemoryItem) == 1  # health-adjacent memory retained too


async def test_sweep_is_children_first_no_fk_violation(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """The sweep deletes an expired thread's children first, so a RESTRICT FK holds (CKPT-R8).

    With ``PRAGMA foreign_keys=ON`` an out-of-order delete (thread before its checkpoint/write/
    interrupt) would raise an IntegrityError; the sweep completing cleanly proves the order.
    """
    async with factory() as session:
        await _seed_thread(session, "expired", created_at=NOW - _dt.timedelta(days=200))
        await session.commit()

    async with factory() as session:
        report = await sweep_expired_checkpoints(session, retention_days=90, now=NOW)
        await session.commit()  # would raise on an FK violation if order were wrong

    assert report.deleted_threads == 1
    async with factory() as session:
        assert await _count(session, AgentThread) == 0
        assert await _count(session, AgentCheckpoint) == 0
        assert await _count(session, AgentWrite) == 0
        assert await _count(session, AgentInterrupt) == 0
        assert await _count(session, MemoryItem) == 0
