"""PII redaction of durable checkpoints + provider sends (AGT-SEC-R4, CKPT-R8 §10).

AGT-SEC-R4 mandates that PII in messages and checkpoints be redacted per policy BEFORE
persistence AND before being sent to any third-party model provider where policy requires;
"redaction MUST be covered by tests." These tests plant high-confidence PII/secret spans and
assert that NEITHER the persisted checkpoint blob NOR the outbound provider payload carries
them.

The persistence leg runs on a REAL file-backed SQLite pool (WAL) — never ``:memory:`` /
StaticPool — so the blob is round-tripped through a genuine connection exactly as production
writes it. The provider-send leg drives :class:`OpenAICompatibleModel` through an injected
recording stub client (no network) and inspects the exact ``messages`` payload sent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, empty_checkpoint
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    SqlAlchemyCheckpointSaver,
)
from wattwise_core.agent.model import OpenAICompatibleModel
from wattwise_core.agent.redaction import redact_state_payload
from wattwise_core.agent.state_store import AgentCheckpoint, AgentStateBase, AgentWrite
from wattwise_core.api.redaction import redact_text
from wattwise_core.config import load_settings

pytestmark = pytest.mark.unit

ATHLETE_A = "00000000-0000-7000-8000-00000000000a"
CONVERSATION = "conv-1"
THREAD_ID = f"{ATHLETE_A}:{CONVERSATION}"

# Planted secrets the athlete's words / a draft might carry — exactly the high-confidence
# classes the central redactor masks (email, provider key, phone run).
SECRET_EMAIL = "athlete.private@example.com"
SECRET_KEY = "sk-ABCDEFGHIJKLMNOPQRSTUV"
SECRET_PHONE = "+1 415 555 0199"


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + busy_timeout on each connection so the file pool serializes (real pool)."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


@pytest_asyncio.fixture
async def factory(
    tmp_path: Path,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Session factory over a REAL file-SQLite pool (WAL), NOT ``:memory:``/StaticPool."""
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


def _saver(factory: async_sessionmaker[AsyncSession]) -> SqlAlchemyCheckpointSaver:
    return SqlAlchemyCheckpointSaver(
        factory,
        athlete_id=ATHLETE_A,
        conversation_id=CONVERSATION,
        schema_version=CHECKPOINT_SCHEMA_VERSION,
    )


def _config() -> RunnableConfig:
    return {"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}}


def _checkpoint_with_pii() -> Checkpoint:
    cp = empty_checkpoint()
    cp["channel_values"] = {
        "messages": [
            {"role": "user", "content": f"My injury notes are at {SECRET_EMAIL}"},
        ],
        "request_text": f"call me on {SECRET_PHONE}",
        "draft": f"key {SECRET_KEY} leaked",
    }
    return cp


def _metadata() -> CheckpointMetadata:
    return {"source": "loop", "step": 1, "parents": {}}


# --- persistence-path redaction (AGT-SEC-R4 / CKPT-R8) --------------------------------


async def test_persisted_checkpoint_blob_carries_no_planted_pii(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """The persisted checkpoint blob MUST NOT contain any planted PII (AGT-SEC-R4)."""
    saver = _saver(factory)
    await saver.aput(_config(), _checkpoint_with_pii(), _metadata(), {})

    async with factory() as session:
        row = (
            await session.execute(select(AgentCheckpoint).limit(1))
        ).scalar_one()
    raw_bytes: bytes = row.checkpoint_blob
    # The blob is the serialized state; no planted secret may survive in it.
    assert SECRET_EMAIL.encode() not in raw_bytes
    assert SECRET_KEY.encode() not in raw_bytes
    assert b"4155550199" not in raw_bytes  # the phone digits, separator-stripped
    # The mask token IS present — proving the strings were redacted, not merely absent.
    assert b"[redacted]" in raw_bytes


async def test_persisted_write_value_carries_no_planted_pii(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """A node's pending write value MUST be redacted before persistence (AGT-SEC-R4/CKPT-R2)."""
    saver = _saver(factory)
    cfg: RunnableConfig = {
        "configurable": {"thread_id": THREAD_ID, "checkpoint_ns": "", "checkpoint_id": "cp-1"}
    }
    await saver.aput_writes(cfg, [("draft", f"contact {SECRET_EMAIL}")], task_id="t1")

    async with factory() as session:
        row = (await session.execute(select(AgentWrite).limit(1))).scalar_one()
    assert SECRET_EMAIL.encode() not in row.value_blob


async def test_redacted_checkpoint_still_round_trips(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """Redaction preserves structure: a redacted checkpoint deserializes + resumes (CKPT-R2)."""
    saver = _saver(factory)
    await saver.aput(_config(), _checkpoint_with_pii(), _metadata(), {})
    got = await saver.aget_tuple(_config())
    assert got is not None
    values = got.checkpoint["channel_values"]
    # Structure intact (the message list + keys survive), only the PII substrings masked.
    assert isinstance(values["messages"], list)
    assert "[redacted]" in values["messages"][0]["content"]
    assert SECRET_PHONE not in values["request_text"]


# --- provider-send-path redaction (AGT-SEC-R4 "before ... third-party provider") ------


class _Recorder:
    """Records the messages passed to the provider parse/create calls (no network)."""

    def __init__(self) -> None:
        self.last_messages: list[dict[str, Any]] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last_messages = kwargs["messages"]
        message = type("M", (), {"content": "warm prose", "refusal": None, "parsed": None})()
        return type("C", (), {"choices": [type("Ch", (), {"message": message})()]})()


class _Client:
    def __init__(self, recorder: _Recorder) -> None:
        self.chat = type("Chat", (), {"completions": recorder})()


def _model(*, redact: bool, recorder: _Recorder) -> OpenAICompatibleModel:
    settings = load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        agent__model="test-model",
        agent__temperature=0.5,
        agent__max_output_tokens=256,
        agent__redact_provider_payloads=redact,
    )
    return OpenAICompatibleModel(settings=settings, client=_Client(recorder))  # type: ignore[arg-type]


async def test_compose_redacts_outbound_payload_when_policy_requires() -> None:
    """When policy requires it, the outbound provider payload carries no PII (AGT-SEC-R4)."""
    recorder = _Recorder()
    model = _model(redact=True, recorder=recorder)
    await model.compose(system="coach voice", context=f"athlete email {SECRET_EMAIL}")
    sent = recorder.last_messages
    assert sent is not None
    blob = "".join(m["content"] for m in sent)
    assert SECRET_EMAIL not in blob
    assert "[redacted]" in blob


async def test_compose_sends_raw_when_policy_off() -> None:
    """With the policy OFF the payload is sent verbatim (mutation guard for the gate)."""
    recorder = _Recorder()
    model = _model(redact=False, recorder=recorder)
    await model.compose(system="coach voice", context=f"athlete email {SECRET_EMAIL}")
    sent = recorder.last_messages
    assert sent is not None
    blob = "".join(m["content"] for m in sent)
    assert SECRET_EMAIL in blob


def test_central_redactor_is_reused_not_reinvented() -> None:
    """The agent redactor delegates to the central redactor (one redaction policy)."""
    assert redact_state_payload(SECRET_EMAIL) == redact_text(SECRET_EMAIL)


def test_redact_state_payload_preserves_container_types() -> None:
    """Type-preserving recursion (tuple/set survive) so a serialized blob deserializes."""
    out = redact_state_payload(({SECRET_EMAIL}, ("x", SECRET_KEY), {"k": 1}))
    assert isinstance(out, tuple)
    assert isinstance(out[0], set)
    assert out[1] == ("x", "[redacted]")  # tuple preserved, PII leaf masked
    assert out[2] == {"k": 1}  # non-text scalars untouched
