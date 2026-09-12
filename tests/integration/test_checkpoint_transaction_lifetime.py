"""Real-pool pending-write preparation and atomicity (CKPT-R2a/-R8, AGT-SEC-R4)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from wattwise_core.agent.checkpoint import SqlAlchemyCheckpointSaver
from wattwise_core.agent.state_db import AgentStateDatabase

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("serialization_fails", [False, True])
async def test_pending_payload_preparation_releases_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serialization_fails: bool
) -> None:
    """A separate writer progresses during serialization; failed batches persist nothing."""
    path = tmp_path / "agent-state.sqlite"
    database = AgentStateDatabase(dsn=f"sqlite+aiosqlite:///{path}")
    await database.create_all()
    saver = SqlAlchemyCheckpointSaver(
        database.session_factory,
        athlete_id="00000000-0000-7000-8000-00000000000a",
        conversation_id="conversation",
    )
    try:
        config = await saver.aput(
            {"configurable": {"thread_id": "thread", "checkpoint_ns": ""}},
            empty_checkpoint(),
            {"source": "loop", "step": 1, "parents": {}},
            {},
        )
        serialize = saver.serde.dumps_typed
        calls = 0
        peer_writes = 0

        def serialize_with_peer(value: Any) -> tuple[str, bytes]:
            nonlocal calls, peer_writes
            calls += 1
            if calls == 2:
                # This independent connection makes progress WHILE payload preparation runs.
                # The short timeout bounds a deterministic failure if an earlier channel INSERT
                # still owns the writer lock; no sleeps or scheduler races are required.
                with sqlite3.connect(path, timeout=0.05) as peer:
                    result = peer.execute(
                        "UPDATE agent_thread SET conversation_id=conversation_id "
                        "WHERE thread_id='thread'"
                    )
                    peer_writes += result.rowcount
                if serialization_fails:
                    raise ValueError("injected serialization failure")
            return serialize(value)

        monkeypatch.setattr(saver.serde, "dumps_typed", serialize_with_peer)
        writes = [
            ("athlete_id", "identity@example.com"),
            ("draft", "Contact athlete@example.com"),
            ("__resume__", "first"),
            ("__resume__", "latest"),
        ]
        if serialization_fails:
            with pytest.raises(ValueError, match="injected serialization failure"):
                await saver.aput_writes(config, writes, "task")
        else:
            await saver.aput_writes(config, writes, "task")
        assert peer_writes == 1
        restored = await saver.aget_tuple(config)
        assert restored is not None
        assert restored.pending_writes is not None
        if serialization_fails:
            assert restored.pending_writes == []
        else:
            pending = [(channel, value) for _, channel, value in restored.pending_writes]
            assert pending[0] == ("__resume__", "latest")  # reserved negative index, last wins
            assert pending[1] == ("athlete_id", "identity@example.com")  # identity stays intact
            assert pending[2][0] == "draft"
            assert "athlete@example.com" not in pending[2][1]
            assert "Contact" in pending[2][1]
            assert len(pending) == 3
    finally:
        await database.dispose()
