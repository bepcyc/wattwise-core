"""Shared-factory saver admission uses real transactions (CKPT-R2b/-R9)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from langgraph.checkpoint.base import empty_checkpoint

from wattwise_core.agent import checkpoint_interrupts as ledger
from wattwise_core.agent.checkpoint import SqlAlchemyCheckpointSaver
from wattwise_core.agent.state_db import AgentStateDatabase

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[AgentStateDatabase]:
    database = AgentStateDatabase(dsn=f"sqlite+aiosqlite:///{tmp_path / 'state.sqlite'}")
    await database.create_all()
    try:
        yield database
    finally:
        await database.dispose()


def _saver(store: AgentStateDatabase) -> SqlAlchemyCheckpointSaver:
    return SqlAlchemyCheckpointSaver(
        store.session_factory,
        athlete_id="00000000-0000-7000-8000-00000000000a",
        conversation_id="conversation",
    )


async def _checkpoint(saver: SqlAlchemyCheckpointSaver) -> Any:
    return await saver.aput(
        {"configurable": {"thread_id": "thread", "checkpoint_ns": ""}},
        empty_checkpoint(),
        {"source": "loop", "step": 1, "parents": {}},
        {},
    )


@pytest.mark.parametrize("operation", ["checkpoint", "pending", "record", "consume"])
async def test_new_savers_wait_before_database_work(
    store: AgentStateDatabase, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A cancelled waiter never reaches the DB; all four write paths subsequently progress."""
    first, second = _saver(store), _saver(store)
    config = await _checkpoint(first)
    await first.record_interrupt("thread", "interrupt")
    entered, release, started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    attempts: list[str] = []
    ensure_first, ensure_second = first._ensure_thread, second._ensure_thread
    consume = ledger.consume_interrupt

    async def hold(*args: Any) -> Any:
        result = await ensure_first(*args)
        entered.set()
        await release.wait()
        return result

    async def observe_ensure(*args: Any) -> Any:
        attempts.append(operation)
        return await ensure_second(*args)

    async def observe_consume(*args: Any) -> Any:
        attempts.append(operation)
        return await consume(*args)

    async def write() -> None:
        started.set()
        if operation == "checkpoint":
            await _checkpoint(second)
        elif operation == "pending":
            await second.aput_writes(config, [("draft", "second")], "second")
        elif operation == "record":
            await second.record_interrupt("thread", "next-interrupt")
        else:
            assert await second.consume_interrupt("thread", "interrupt")

    monkeypatch.setattr(first, "_ensure_thread", hold)
    monkeypatch.setattr(second, "_ensure_thread", observe_ensure)
    monkeypatch.setattr(ledger, "consume_interrupt", observe_consume)
    holder = asyncio.create_task(first.aput_writes(config, [("draft", "first")], "first"))
    waiter: asyncio.Task[None] | None = None
    try:
        await entered.wait()
        waiter = asyncio.create_task(write())
        # Event.set does not yield: write reaches either admission or the instrumented DB
        # phase before this task resumes, making this assertion independent of timer delays.
        await started.wait()
        assert attempts == [], "a new saver must wait before starting database work"
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await holder
        await write()
        assert attempts == [operation]
    finally:
        release.set()
        if waiter is not None and not waiter.done():
            waiter.cancel()
        await asyncio.gather(
            holder, *([waiter] if waiter is not None else []), return_exceptions=True
        )


@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_writer_releases_other_savers(
    store: AgentStateDatabase, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    """A failure or cancellation inside admission cannot poison later requests."""
    first, second = _saver(store), _saver(store)
    config = await _checkpoint(first)
    entered = asyncio.Event()

    async def fail(*args: Any) -> None:
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise ValueError("injected write failure")

    monkeypatch.setattr(first, "_ensure_thread", fail)
    holder = asyncio.create_task(first.aput_writes(config, [("draft", "lost")], "failed"))
    await entered.wait()
    if cancel:
        holder.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else ValueError):
        await holder
    await second.aput_writes(config, [("draft", "kept")], "successful")
    restored = await second.aget_tuple(config)
    assert restored is not None
    assert restored.pending_writes == [("successful", "draft", "kept")]
