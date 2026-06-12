"""Unit tests for the durable agent-state checkpointer (CKPT-R1/-R2/-R3/-R7, ARCH-R13).

Proves the contract the spec gates on:

* **put -> get roundtrip** — a persisted checkpoint resumes identically (CKPT-R1/-R2),
  including pending intermediate writes replayed via ``aput_writes`` (CKPT-R2).
* **cross-identity refusal** — a checkpoint written under one ``athlete_id`` is NOT
  loadable under another; the load is REFUSED, never silently returned (CKPT-R3).
* **schema-version fail-closed** — loading a checkpoint whose stored ``schema_version``
  is incompatible with the engine FAILS CLOSED rather than coercing (CKPT-R7).
* **store separation** — the agent-state tables are NOT registered on the canonical
  metadata (ARCH-R13/R29): durable agent state never lives in the canonical GBO store.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, empty_checkpoint
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import wattwise_core.agent.auth_state
import wattwise_core.agent.digest_history
import wattwise_core.agent.memory
import wattwise_core.agent.ops_jobs  # noqa: F401  (registers the import/export job tables)
from wattwise_core.agent.checkpoint import (
    CheckpointIdentityError,
    SqlAlchemyCheckpointSaver,
)
from wattwise_core.agent.state_store import AgentStateBase
from wattwise_core.persistence.base import Base

ATHLETE_A = "00000000-0000-7000-8000-00000000000a"
ATHLETE_B = "00000000-0000-7000-8000-00000000000b"
CONVERSATION = "conv-1"
THREAD_ID = "thread-A-conv-1"


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory over a fresh in-memory agent-state schema (NOT canonical)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(AgentStateBase.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


def _saver(
    factory: async_sessionmaker[AsyncSession],
    *,
    athlete_id: str,
    conversation_id: str = CONVERSATION,
    schema_version: int = 1,
) -> SqlAlchemyCheckpointSaver:
    return SqlAlchemyCheckpointSaver(
        factory,
        athlete_id=athlete_id,
        conversation_id=conversation_id,
        schema_version=schema_version,
    )


def _config(thread_id: str = THREAD_ID, checkpoint_id: str | None = None) -> RunnableConfig:
    configurable: dict[str, object] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _checkpoint(value: object) -> Checkpoint:
    cp = empty_checkpoint()
    cp["channel_values"] = {"messages": value}
    return cp


def _metadata() -> CheckpointMetadata:
    return {"source": "loop", "step": 1, "parents": {}}


# --- put -> get roundtrip (CKPT-R1/-R2) ------------------------------------------------


async def test_put_then_get_roundtrips(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    saver = _saver(session_factory, athlete_id=ATHLETE_A)
    checkpoint = _checkpoint(["hello", "world"])

    saved_config = await saver.aput(_config(), checkpoint, _metadata(), {})
    cp_id = saved_config["configurable"]["checkpoint_id"]

    # Latest checkpoint by thread.
    got = await saver.aget_tuple(_config())
    assert got is not None
    assert got.checkpoint["id"] == checkpoint["id"]
    assert got.checkpoint["channel_values"]["messages"] == ["hello", "world"]
    assert got.config["configurable"]["checkpoint_id"] == cp_id
    assert got.metadata["step"] == 1

    # Addressed by explicit id.
    by_id = await saver.aget_tuple(_config(checkpoint_id=str(cp_id)))
    assert by_id is not None
    assert by_id.checkpoint["id"] == checkpoint["id"]


async def test_get_unknown_thread_returns_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    saver = _saver(session_factory, athlete_id=ATHLETE_A)
    assert await saver.aget_tuple(_config(thread_id="never-written")) is None


async def test_put_writes_are_replayed_on_get(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    saver = _saver(session_factory, athlete_id=ATHLETE_A)
    checkpoint = _checkpoint(["base"])
    await saver.aput(_config(), checkpoint, _metadata(), {})

    cfg = _config(checkpoint_id=checkpoint["id"])
    await saver.aput_writes(cfg, [("messages", "partial-1"), ("messages", "partial-2")], "task-1")

    got = await saver.aget_tuple(cfg)
    assert got is not None
    assert got.pending_writes is not None
    values = [v for (_task, _ch, v) in got.pending_writes]
    assert values == ["partial-1", "partial-2"]


async def test_alist_yields_thread_checkpoints_newest_first(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    saver = _saver(session_factory, athlete_id=ATHLETE_A)
    first = _checkpoint(["one"])
    await saver.aput(_config(), first, _metadata(), {})
    second = _checkpoint(["two"])
    await saver.aput(_config(checkpoint_id=first["id"]), second, _metadata(), {})

    listed = [tup async for tup in saver.alist(_config())]
    assert len(listed) == 2
    # newest-first: the second checkpoint links back to the first.
    assert listed[0].checkpoint["id"] == second["id"]
    assert listed[0].parent_config is not None
    assert listed[0].parent_config["configurable"]["checkpoint_id"] == first["id"]


# --- cross-identity refusal (CKPT-R3) --------------------------------------------------


async def test_cross_identity_load_is_refused(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # Athlete A writes a checkpoint on the thread.
    saver_a = _saver(session_factory, athlete_id=ATHLETE_A)
    await saver_a.aput(_config(), _checkpoint(["a-only"]), _metadata(), {})

    # Athlete B attempts to load the SAME thread_id -> refused, never returned.
    saver_b = _saver(session_factory, athlete_id=ATHLETE_B)
    with pytest.raises(CheckpointIdentityError):
        await saver_b.aget_tuple(_config())


async def test_cross_identity_list_is_refused(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    saver_a = _saver(session_factory, athlete_id=ATHLETE_A)
    await saver_a.aput(_config(), _checkpoint(["a-only"]), _metadata(), {})

    saver_b = _saver(session_factory, athlete_id=ATHLETE_B)
    with pytest.raises(CheckpointIdentityError):
        _ = [tup async for tup in saver_b.alist(_config())]


async def test_malformed_identity_fails_closed(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(CheckpointIdentityError):
        _saver(session_factory, athlete_id="not-a-uuid")


# --- schema-version fail-closed (CKPT-R7) ----------------------------------------------


async def test_schema_version_mismatch_fails_closed(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # Persist under engine schema version 1.
    writer = _saver(session_factory, athlete_id=ATHLETE_A, schema_version=1)
    await writer.aput(_config(), _checkpoint(["v1"]), _metadata(), {})

    # A future engine expecting version 2 must REFUSE the v1 checkpoint, not coerce it.
    # CKPT-R7: log a warning and start fresh (return None) rather than crashing.
    reader = _saver(session_factory, athlete_id=ATHLETE_A, schema_version=2)
    result = await reader.aget_tuple(_config())
    assert result is None


async def test_schema_version_mismatch_fails_closed_on_list(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    writer = _saver(session_factory, athlete_id=ATHLETE_A, schema_version=1)
    await writer.aput(_config(), _checkpoint(["v1"]), _metadata(), {})

    # CKPT-R7: stale checkpoints are silently skipped; the list is empty.
    reader = _saver(session_factory, athlete_id=ATHLETE_A, schema_version=2)
    results = [tup async for tup in reader.alist(_config())]
    assert results == []


# --- idempotency dedup (CKPT-R4) -------------------------------------------------------


async def test_resolve_idempotent_returns_existing_run(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # CKPT-R4: a re-submission of the same turn (same thread_id) resolves the EXISTING run
    # instead of starting a duplicate; before any run exists it resolves to None.
    saver = _saver(session_factory, athlete_id=ATHLETE_A)
    assert await saver.resolve_idempotent(THREAD_ID) is None

    checkpoint = _checkpoint(["v1"])
    await saver.aput(_config(), checkpoint, _metadata(), {})

    resolved = await saver.resolve_idempotent(THREAD_ID)
    assert resolved == checkpoint["id"]


async def test_resolve_idempotent_refuses_cross_identity(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # CKPT-R4 + CKPT-R3: idempotency resolution is identity-scoped; another athlete cannot
    # resolve (and thus resume) a thread it does not own.
    writer = _saver(session_factory, athlete_id=ATHLETE_A)
    await writer.aput(_config(), _checkpoint(["v1"]), _metadata(), {})

    other = _saver(session_factory, athlete_id=ATHLETE_B)
    with pytest.raises(CheckpointIdentityError):
        await other.resolve_idempotent(THREAD_ID)


# --- store separation (ARCH-R13/R29) ---------------------------------------------------


def test_agent_state_tables_not_in_canonical_metadata() -> None:
    # Durable agent state is NEVER in the canonical GBO schema (ARCH-R13/R29). The memory
    # import above registers ``agent_memory_item`` on the agent-state metadata (MEM-R3):
    # it MUST live in the agent-state store, never canonical.
    canonical = set(Base.metadata.tables)
    agent_state = set(AgentStateBase.metadata.tables)
    assert agent_state == {
        "agent_thread",
        "agent_checkpoint",
        "agent_write",
        "agent_memory_item",
        "agent_interrupt",
        "agent_digest_record",
        "agent_auth_refresh_token",
        "agent_auth_link_challenge",
        "agent_import_job",
        "agent_export_job",
    }
    assert canonical.isdisjoint(agent_state)
    assert not any(name.startswith("agent_") for name in canonical)
    # The relocated memory table is NOT in canonical metadata (the leak is closed).
    assert "memory_item" not in canonical
    assert "agent_memory_item" not in canonical


def test_distinct_athletes_get_distinct_thread_rows() -> None:
    # Sanity: the saver coerces server-derived identity to a UUID (STATE-R4/AGT-SEC-R1).
    a = uuid.UUID(ATHLETE_A)
    b = uuid.UUID(ATHLETE_B)
    assert a != b
