"""Durable HITL resume safety on a REAL multi-connection pool (SPIKE-1, CKPT-R1/-R2/-R3).

This is the permanent regression suite for the durable-resume bug: driving a langgraph
``interrupt()`` then resuming with ``Command(resume=...)`` through the real
``SqlAlchemyCheckpointSaver`` used to raise ``IntegrityError`` from
``aput_writes -> _ensure_thread``, and an earlier ``ON CONFLICT DO NOTHING`` patch silently
DROPPED the human decision (the special ``__resume__`` channel collided at positional idx 0
with a branch write). Both failure modes are pinned here.

CRITICAL: every concurrency / round-trip test runs on a **file-backed SQLite engine with a
real connection pool** (WAL + busy_timeout so concurrent writers serialize), NEVER
``sqlite :memory:`` / ``StaticPool`` — the in-memory single-connection setup is exactly what
made the reverted fix false-green (one connection cannot race, so the ``_ensure_thread``
conflict path was never exercised). PostgreSQL / MariaDB legs run when ``WATTWISE_PG_DSN`` /
``WATTWISE_MARIADB_DSN`` are set, proving the ``ON CONFLICT DO UPDATE`` upsert is portable.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import WRITES_IDX_MAP, empty_checkpoint
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import wattwise_core.agent.memory  # noqa: F401  (registers agent_memory_item on AgentStateBase)
from wattwise_core.agent.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointIdentityError,
    CheckpointSchemaVersionError,
    SqlAlchemyCheckpointSaver,
)
from wattwise_core.agent.state_store import (
    AgentInterrupt,
    AgentStateBase,
    AgentThread,
    AgentWrite,
)

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

pytestmark = pytest.mark.integration

ATHLETE_A = "00000000-0000-7000-8000-00000000000a"
ATHLETE_B = "00000000-0000-7000-8000-00000000000b"
CONVERSATION = "conv-1"
THREAD_ID = "thread-A-conv-1"


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """Put each new SQLite connection in WAL with a long busy timeout (concurrent writers).

    File-backed SQLite uses a real (multi-connection) pool, but its default rollback-journal
    mode makes concurrent writers fail with ``database is locked`` instead of serializing.
    WAL + a 30s ``busy_timeout`` lets the racing ``aput``/``aput_writes`` connections block
    briefly and serialize — so the test exercises the genuine multi-connection thread-create
    race the ``:memory:`` StaticPool fixture cannot, without spurious lock errors. Other
    backends (PG/MariaDB) ignore this listener (it only fires for sqlite engines).
    """
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


def _engine_backends() -> list[ParameterSet]:
    """File-SQLite always; PG/MariaDB only when their throwaway DSN env var is set.

    Mirrors ``tests/integration/test_migrations.py`` so the PG/MariaDB legs really run under
    the CI db-portability job, proving the DO-UPDATE upsert is portable. ``None`` selects the
    per-test file-SQLite engine; a string is used as the DSN verbatim.
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


async def _reset_agent_state(factory: async_sessionmaker[AsyncSession]) -> None:
    """Drop+recreate the agent-state schema so each container-backed leg starts empty."""
    engine = factory.kw["bind"]
    async with engine.begin() as conn:
        await conn.run_sync(AgentStateBase.metadata.drop_all)
        await conn.run_sync(AgentStateBase.metadata.create_all)


@pytest_asyncio.fixture(params=_engine_backends())
async def factory(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Session factory over a REAL multi-connection pool on the chosen backend.

    SQLite is file-backed (real pool) with WAL enabled per connection — deliberately NOT
    ``:memory:``/StaticPool, which hides the thread-create race. PG/MariaDB use their
    throwaway DSN and are reset to an empty agent-state schema before the test runs.
    """
    backend_dsn = request.param
    if backend_dsn is None:
        dsn = f"sqlite+aiosqlite:///{tmp_path}/agent.sqlite"
        engine = create_async_engine(dsn, connect_args={"timeout": 30})
        event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
    else:
        engine = create_async_engine(backend_dsn)
    async with engine.begin() as conn:
        await conn.run_sync(AgentStateBase.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield factory
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(AgentStateBase.metadata.drop_all)
        await engine.dispose()


def _saver(
    factory: async_sessionmaker[AsyncSession],
    *,
    athlete_id: str = ATHLETE_A,
    conversation_id: str = CONVERSATION,
    schema_version: int = CHECKPOINT_SCHEMA_VERSION,
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


# --- 1. SPIKE-1 resume round-trip (permanent regression test) --------------------------


class _GraphState(TypedDict, total=False):
    """Minimal state for the head -> gate -> sink HITL graph."""

    value: int
    decision: dict[str, Any]


async def test_interrupt_then_resume_round_trips(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """interrupt() then Command(resume=...) completes the run without recomputing ``head``.

    The reported bug: resuming through the real saver raised ``IntegrityError`` from
    ``_ensure_thread``; this proves the run pauses at the interrupt, resumes cleanly, reaches
    the sink, and does NOT re-run the pre-interrupt ``head`` node (durable replay, CKPT-R2).
    """
    head_runs: list[int] = []

    def head(state: _GraphState) -> dict[str, Any]:
        head_runs.append(1)  # side effect: must fire exactly ONCE across pause+resume
        return {"value": state.get("value", 0) + 1}

    def gate(_state: _GraphState) -> dict[str, Any]:
        decision = interrupt({"awaiting": "approval"})  # pauses the run durably
        return {"decision": decision}

    def sink(state: _GraphState) -> dict[str, Any]:
        return {"value": state["value"] + 100}

    builder: StateGraph[_GraphState, Any, _GraphState, _GraphState] = StateGraph(_GraphState)
    builder.add_node("head", head)
    builder.add_node("gate", gate)
    builder.add_node("sink", sink)
    builder.add_edge(START, "head")
    builder.add_edge("head", "gate")
    builder.add_edge("gate", "sink")
    builder.add_edge("sink", END)
    graph = builder.compile(checkpointer=_saver(factory))

    cfg = _config()
    paused = await graph.ainvoke({"value": 0}, cfg)
    # (a) it PAUSED at the interrupt: langgraph surfaces an ``__interrupt__`` and the run did
    # not reach the sink (no +100 applied; value is still the head result).
    assert "__interrupt__" in paused
    assert paused["value"] == 1
    assert head_runs == [1]

    resumed = await graph.ainvoke(Command(resume={"approved": True}), cfg)
    # (b) does not raise; (c) reaches the sink; (b') head NOT recomputed (counter stays 1).
    assert "__interrupt__" not in resumed
    assert resumed["decision"] == {"approved": True}
    assert resumed["value"] == 101, "sink ran on the head result; head was replayed, not re-run"
    assert head_runs == [1], "head must NOT be recomputed on resume (durable replay)"


# --- 2. concurrent _ensure_thread idempotency under a real pool ------------------------


async def test_concurrent_ensure_thread_is_idempotent(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """N concurrent first-writes for one thread create exactly ONE row, no IntegrityError.

    This is the test that MUST race: file-SQLite/PG/MariaDB use a real >1-connection pool,
    so concurrent ``aput``/``aput_writes`` first-touches collide on the
    ``(athlete_id, conversation_id)`` unique constraint and exercise the ``_ensure_thread``
    fresh-session re-resolve. On ``:memory:``/StaticPool this would false-green (one
    connection cannot race).
    """
    saver = _saver(factory)
    n = 16

    async def one(i: int) -> None:
        if i % 2 == 0:
            await saver.aput_writes(_config(checkpoint_id="cp-x"), [("c", f"v{i}")], f"task-{i}")
        else:
            cp = _checkpoint(f"cp-{i}")
            await saver.aput(_config(), cp[1], _metadata(), {})

    # Verify the pool really hands out more than one connection (else the race is fake).
    assert factory.kw["bind"].pool.size() >= 1
    await asyncio.gather(*(one(i) for i in range(n)))

    async with factory() as session:
        rows = (
            (await session.execute(select(AgentThread).where(AgentThread.thread_id == THREAD_ID)))
            .scalars()
            .all()
        )
    assert len(rows) == 1, "exactly one agent_thread row despite the concurrent get-or-create"
    assert rows[0].athlete_id == uuid.UUID(ATHLETE_A)


def _checkpoint(cp_id: str) -> tuple[str, dict[str, Any]]:
    cp = empty_checkpoint()
    cp["id"] = cp_id
    cp["channel_values"] = {"messages": [cp_id]}
    return cp_id, cp


def _metadata() -> dict[str, Any]:
    return {"source": "loop", "step": 1, "parents": {}}


# --- 3. special-channel __resume__ preservation (deterministic) ------------------------


async def _load_writes(
    factory: async_sessionmaker[AsyncSession], task_id: str
) -> dict[str, tuple[int, Any]]:
    """Return ``{channel: (idx, decoded_value)}`` for one task's persisted writes."""
    saver = _saver(factory)
    async with factory() as session:
        rows = (
            (await session.execute(select(AgentWrite).where(AgentWrite.task_id == task_id)))
            .scalars()
            .all()
        )
    out: dict[str, tuple[int, Any]] = {}
    for row in rows:
        value = saver.serde.loads_typed((row.value_type, row.value_blob))
        out[row.channel] = (row.idx, value)
    return out


async def test_resume_channel_not_collided_and_last_write_wins(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """``__resume__`` (HITL decision) survives a positional-0 branch write, and DO UPDATE wins.

    Branch write is at positional idx 0; ``__resume__`` would also map to 0 under the broken
    positional keying and be clobbered. With ``WRITES_IDX_MAP`` keying it lands at idx -4, so
    BOTH writes persist. A second delivery of ``__resume__`` with a different value overwrites
    (last-write-wins), proving ``ON CONFLICT DO UPDATE`` rather than ``DO NOTHING``.
    """
    saver = _saver(factory)
    cfg = _config(checkpoint_id="cp-1")
    await saver.aput(_config(), _checkpoint("cp-1")[1], _metadata(), {})

    # branch write at positional 0, __resume__ at positional 1 (maps to reserved idx -4).
    await saver.aput_writes(
        cfg, [("branch_chan", "branch-val"), ("__resume__", {"approved": True})], "task-r"
    )
    loaded = await _load_writes(factory, "task-r")
    assert loaded["branch_chan"][0] == 0
    assert loaded["__resume__"][0] == WRITES_IDX_MAP["__resume__"] == -4
    assert loaded["__resume__"][1] == {"approved": True}, "the human decision must NOT be dropped"

    # last-write-wins: re-deliver __resume__ with a different value -> overwrites (DO UPDATE).
    await saver.aput_writes(cfg, [("__resume__", {"approved": False})], "task-r")
    reloaded = await _load_writes(factory, "task-r")
    assert reloaded["__resume__"][1] == {"approved": False}, "DO UPDATE: last write wins"
    assert reloaded["branch_chan"][1] == "branch-val", "the untouched branch write is intact"


# --- 4. CKPT-R3 preserved under the new ensure/upsert path -----------------------------


async def test_cross_identity_write_is_refused(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """Athlete B cannot write/ensure against athlete A's existing thread (CKPT-R3).

    The new fresh-session re-resolve in ``_ensure_thread`` still routes through
    ``_resolve_thread``, so a cross-identity thread is REFUSED — never silently adopted.
    """
    saver_a = _saver(factory, athlete_id=ATHLETE_A)
    await saver_a.aput(_config(), _checkpoint("cp-a")[1], _metadata(), {})

    saver_b = _saver(factory, athlete_id=ATHLETE_B)
    with pytest.raises(CheckpointIdentityError):
        await saver_b.aput(_config(), _checkpoint("cp-b")[1], _metadata(), {})
    with pytest.raises(CheckpointIdentityError):
        await saver_b.aput_writes(_config(checkpoint_id="cp-a"), [("c", "v")], "task-b")


async def test_cross_identity_refused_on_probe_path(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """CKPT-R3 holds on the post-ensure resolve AFTER a collided create (kills MUT-1).

    ``_ensure_thread`` is ensure-first: the atomic insert-or-ignore (UPS-R2 seam) runs
    BEFORE the resolve, so a cross-identity caller's create statement genuinely collides
    with the other athlete's committed row (PK ``thread_id``) and is absorbed by the
    database — and the ONLY thing standing between that absorbed collision and silently
    adopting the foreign row is the post-ensure ``_resolve_thread``. This test drives
    athlete **B** through that exact path against athlete A's committed thread and pins
    that the post-ensure resolve runs and REFUSES (``CheckpointIdentityError``), never
    returns A's row under identity B.

    MUT-1 (post-ensure resolve bypassing ``_resolve_thread``, e.g.
    ``session.get(AgentThread, thread_id)``) drops the athlete-scope check: B would adopt
    A's row, ``aput`` would proceed, no error would surface — this test MUST fail.
    """
    cross_thread = "thread-probe-cross"

    # Athlete A owns the thread row at ``cross_thread`` (committed, visible to fresh sessions).
    saver_a = _saver(factory, athlete_id=ATHLETE_A, conversation_id="conv-A")
    await saver_a.aput(_config(thread_id=cross_thread), _checkpoint("cp-a")[1], _metadata(), {})

    # Athlete B's ensure statement collides with A's row; the post-ensure resolve must refuse.
    saver_b = _saver(factory, athlete_id=ATHLETE_B, conversation_id="conv-B")
    real_resolve = type(saver_b)._resolve_thread  # unbound; called with (saver, session, tid)
    calls = {"n": 0}

    async def _counting_resolve(session: AsyncSession, thread_id: str) -> AgentThread | None:
        # Delegate to the REAL resolver, counting invocations so the test proves the
        # post-ensure athlete-scope check actually executed (not short-circuited away).
        calls["n"] += 1
        return await real_resolve(saver_b, session, thread_id)

    saver_b._resolve_thread = _counting_resolve  # type: ignore[method-assign]

    with pytest.raises(CheckpointIdentityError):
        await saver_b.aput(_config(thread_id=cross_thread), _checkpoint("cp-b")[1], _metadata(), {})
    assert calls["n"] >= 1, "the post-ensure _resolve_thread athlete-scope check did not run"


async def test_genuine_integrity_error_fails_closed(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """A non-race constraint violation fails CLOSED out of ``_ensure_thread`` (kills MUT-2).

    The ensure seam may only absorb the BENIGN concurrent-create collision on the
    ``thread_id`` key it is declared on (after which the row resolves). A genuine
    violation of a DIFFERENT constraint, whose requested row therefore does NOT exist
    afterward, MUST surface as an error — never a silent success.

    Setup: one saver bound to (A, ``conv-fc``) commits a thread at ``T1``. We then call
    ``_ensure_thread`` DIRECTLY (on its own session) for a DIFFERENT thread_id ``T2``;
    the ensure statement inserts ``(T2, A, conv-fc)``, which collides with the ``T1`` row
    on the ``UNIQUE(athlete_id, conversation_id)`` constraint — NOT the declared
    ``thread_id`` conflict key. On PostgreSQL/SQLite the ``ON CONFLICT (thread_id)``
    statement re-raises that violation as ``IntegrityError``; on MariaDB ``INSERT
    IGNORE`` absorbs it, the ``T2`` row still does not exist, and the post-ensure
    resolve fails closed with ``CheckpointError``. Either way an error propagates
    straight out of ``_ensure_thread``.

    Calling ``_ensure_thread`` directly (rather than via ``aput``) is what makes this
    assertion NON-VACUOUS on ALL three backends: if MUT-2 swallows the error and ``aput``
    then proceeds to INSERT the ``agent_checkpoint`` row, PG/MariaDB would raise a
    DOWNSTREAM FK violation (T2's thread row was never created) and mask the mutation —
    a false green. Pinning the error at the ``_ensure_thread`` boundary isolates the
    seam's fail-closed contract itself.

    MUT-2 (the post-ensure miss made fail-open, e.g. ``return existing`` / ``return
    None`` instead of raising when the row did not resolve) swallows this genuine error,
    so this test MUST fail.
    """
    saver = _saver(factory, athlete_id=ATHLETE_A, conversation_id="conv-fc")
    # T1 occupies (A, conv-fc) — the unique pair the T2 insert will collide with.
    await saver.aput(_config(thread_id="T1"), _checkpoint("cp-t1")[1], _metadata(), {})

    # T2 is a new thread_id but the SAME (athlete, conversation) -> a UNIQUE violation that
    # does NOT correspond to a concurrent create of T2, so T2 never exists and the ensure
    # must fail closed. Assert at the _ensure_thread boundary so no downstream INSERT can
    # mask the mutation on any backend.
    with pytest.raises((IntegrityError, CheckpointError)):
        async with factory() as session:
            await saver._ensure_thread(session, "T2")


# --- 5. F-SCHEMA-BUMP: an old-version checkpoint fails closed at v2 (CKPT-R7) -----------


async def test_old_schema_version_checkpoint_fails_closed(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """A v1 checkpoint cannot be loaded under the bumped v2 engine — it FAILS CLOSED (CKPT-R7).

    The D-P2 turn-boundary redesign reshaped ``AgentState`` (turn_id/run_epoch, the
    decrease-to-floor monotonic counters, turn-keyed accumulators), so the engine schema
    version was bumped 1 -> 2. A checkpoint written under the OLD shape lacks those channels
    and MUST NOT be silently coerced into the new reducers; loading it raises
    ``CheckpointSchemaVersionError`` ("start fresh + log"), never returns a coerced tuple. This
    also pins that the constant really advanced to >= 2 (a revert to 1 makes both savers agree
    and the test fails).
    """
    assert CHECKPOINT_SCHEMA_VERSION >= 2, "the schema version must be bumped past the v1 shape"
    # Write a checkpoint stamped with the OLD v1 version (a pre-D-P2 row on disk).
    writer = _saver(factory, schema_version=1)
    await writer.aput(_config(), _checkpoint("cp-v1")[1], _metadata(), {})

    # The live engine (default v2) must refuse to load that row, on BOTH the point-get and the
    # list path (the two read seams that stamp-check), rather than coerce the old blob.
    reader = _saver(factory)  # schema_version defaults to CHECKPOINT_SCHEMA_VERSION (2)
    with pytest.raises(CheckpointSchemaVersionError):
        await reader.aget_tuple(_config())
    with pytest.raises(CheckpointSchemaVersionError):
        async for _ in reader.alist(_config()):
            pass


# --- 6. AgentInterrupt ledger: record + atomic guarded consume (CKPT-R9) ----------------

INTERRUPT_ID = "int-approval-1"


async def _interrupt_rows(
    factory: async_sessionmaker[AsyncSession], thread_id: str
) -> list[AgentInterrupt]:
    """Return the ledger rows for a thread (newest grouping not needed; few rows)."""
    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(AgentInterrupt).where(AgentInterrupt.thread_id == thread_id)
                )
            )
            .scalars()
            .all()
        )
    return list(rows)


async def test_record_interrupt_is_idempotent(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """``record_interrupt`` writes exactly ONE live row and re-recording is a no-op (CKPT-R9).

    The gate may raise the SAME interrupt more than once (a replayed superstep); the unique
    ``(thread_id, interrupt_id)`` plus insert-or-ignore means the ledger keeps a single live
    row and a re-record never errors nor resurrects/duplicates it.
    """
    saver = _saver(factory)
    await saver.record_interrupt(THREAD_ID, INTERRUPT_ID)
    await saver.record_interrupt(THREAD_ID, INTERRUPT_ID)  # idempotent re-record

    rows = await _interrupt_rows(factory, THREAD_ID)
    assert len(rows) == 1, "exactly one ledger row despite the double record"
    assert rows[0].status == "live"
    assert rows[0].interrupt_id == INTERRUPT_ID
    assert rows[0].athlete_id == uuid.UUID(ATHLETE_A)


async def test_double_decision_second_is_refused(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """F-409: a second consume of the same interrupt returns False (already-consumed → 409).

    The first ``consume_interrupt`` flips the live row to ``consumed`` and returns True (the
    decision endpoint resumes); a second consume finds no live row and returns False, so the
    endpoint MUST answer 409 rather than resume the graph twice (fail-closed, CKPT-R9).
    """
    saver = _saver(factory)
    await saver.record_interrupt(THREAD_ID, INTERRUPT_ID)

    assert await saver.consume_interrupt(THREAD_ID, INTERRUPT_ID) is True
    assert await saver.consume_interrupt(THREAD_ID, INTERRUPT_ID) is False

    rows = await _interrupt_rows(factory, THREAD_ID)
    assert [r.status for r in rows] == ["consumed"], "the row is consumed, not re-livened"


async def test_unknown_interrupt_is_refused(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """F-404: consuming an interrupt that was never recorded returns False (→ 404).

    No live row exists, so the guarded UPDATE matches zero rows (``rowcount==0``) and the
    endpoint answers 404 — it must never resume on a fabricated interrupt id.
    """
    saver = _saver(factory)
    # Record a DIFFERENT interrupt so the thread exists but the queried id is absent.
    await saver.record_interrupt(THREAD_ID, INTERRUPT_ID)

    assert await saver.consume_interrupt(THREAD_ID, "int-never-recorded") is False
    # The genuinely-recorded interrupt is untouched (still consumable).
    assert await saver.consume_interrupt(THREAD_ID, INTERRUPT_ID) is True


async def test_cross_athlete_decision_is_refused(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """F-XID: athlete B cannot consume athlete A's interrupt; A's row stays live (CKPT-R9).

    The consume guard is athlete-scoped (``AND athlete_id = <bound>``), so B's UPDATE matches
    zero rows and returns False — never resuming A's run. A's interrupt is NOT collaterally
    consumed and remains consumable by A (the guard is precise, not a blanket flip).
    """
    saver_a = _saver(factory, athlete_id=ATHLETE_A, conversation_id=CONVERSATION)
    await saver_a.record_interrupt(THREAD_ID, INTERRUPT_ID)

    # Athlete B targets the SAME thread_id + interrupt_id but is a different principal.
    saver_b = _saver(factory, athlete_id=ATHLETE_B, conversation_id=CONVERSATION)
    assert await saver_b.consume_interrupt(THREAD_ID, INTERRUPT_ID) is False

    rows = await _interrupt_rows(factory, THREAD_ID)
    assert [r.status for r in rows] == ["live"], "B's refused attempt left A's row live"
    # A can still legitimately consume its own interrupt afterwards.
    assert await saver_a.consume_interrupt(THREAD_ID, INTERRUPT_ID) is True


async def test_concurrent_consume_exactly_one_wins(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """F-CONCURRENT: N concurrent consumes of ONE interrupt — exactly one True (atomic, CKPT-R9).

    This MUST run on a real >1-connection pool (file-SQLite WAL / PG / MariaDB), never
    ``:memory:``/StaticPool: only a genuine multi-connection race exercises the atomic guarded
    UPDATE. The conditional ``WHERE status='live'`` makes the flip a single atomic compare-and-
    set, so exactly ONE racer observes ``rowcount==1`` (resume) and every other sees
    ``rowcount==0`` (409) — there is never a double-resume, and the row ends ``consumed`` once.
    """
    saver = _saver(factory)
    await saver.record_interrupt(THREAD_ID, INTERRUPT_ID)

    # Verify the pool can hand out more than one connection (else the race is fake).
    assert factory.kw["bind"].pool.size() >= 1
    n = 16
    results = await asyncio.gather(
        *(saver.consume_interrupt(THREAD_ID, INTERRUPT_ID) for _ in range(n))
    )
    assert sum(results) == 1, "exactly ONE concurrent decision may win the atomic consume"

    rows = await _interrupt_rows(factory, THREAD_ID)
    assert [r.status for r in rows] == ["consumed"], "the single live row ended consumed once"


async def test_ensure_row_isolation_does_not_stick_to_the_pooled_connection() -> None:
    """``ensure_row``'s READ COMMITTED leg never leaks isolation into later checkouts (UPS-R2).

    The MySQL-family leg of :func:`ensure_row` runs its statement at ``READ COMMITTED`` via a
    per-connection execution option. SQLAlchemy resets a checkout-time isolation override on
    pool check-in (the dialect's ``reset_isolation_level``); this pins that invariant on a REAL
    MariaDB pool sized to ONE connection, so the very same DBAPI connection is re-checked-out
    and must report the server default (REPEATABLE-READ) again — a sticky override would
    silently run every later transaction at the wrong isolation.
    """
    maria = os.environ.get("WATTWISE_MARIADB_DSN")
    if not maria:
        pytest.skip("no MariaDB DSN (container leg only)")
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from wattwise_core.agent.state_store import AgentThread
    from wattwise_core.persistence.upsert import ensure_row

    engine = create_async_engine(maria, pool_size=1, max_overflow=0)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(AgentStateBase.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        await ensure_row(
            factory,
            AgentThread.__table__,
            {
                "thread_id": f"iso-{uuid.uuid4()}",
                "athlete_id": uuid.uuid4(),
                "conversation_id": "iso",
                "created_at": _dt.datetime.now(_dt.UTC),
            },
            conflict_keys=["thread_id"],
        )
        async with factory() as probe:
            level = (
                await probe.execute(text("SELECT @@transaction_isolation"))
            ).scalar_one()
        assert str(level).upper().replace("_", "-") == "REPEATABLE-READ", (
            f"isolation leaked across pool check-in: {level!r}"
        )
    finally:
        await engine.dispose()
