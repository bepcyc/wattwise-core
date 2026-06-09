"""Multi-turn run-scoped reset over the REAL durable saver+pool (CKPT-R5; F-8TURN, F-LEAK).

These are the integration-level guards for the new-turn boundary protocol: a single durable
thread is reused across MANY ``/ask`` turns, and the head node (``ingest_request``) resets the
run-scoped channels at each boundary so the reverted cross-turn bugs cannot reappear:

* **F-8TURN** — drive >=8 sequential turns on ONE durable thread through the real
  ``SqlAlchemyCheckpointSaver``. The reverted bug: a durable thread carrying ``count=N`` from a
  prior turn made the NEXT turn's first ``count=1`` write a monotonic-DECREASE that the strict
  reducer raised on (force-degrade). With ``_turn_monotonic`` + the head-node reset, every turn
  must finish ``COMPLETED`` and the run-scoped counters must START AT 0 each turn (the head
  resets to the sentinel floor; subsequent nodes tick up from there).
* **F-LEAK** — turn-2 must not be able to cite turn-1's evidence. Each turn's gateway returns a
  turn-UNIQUE record key; the grounder records the ``retrieved`` keys it is handed. The reset +
  turn-keyed reducer mean turn-2's grounder sees ONLY turn-2's record — never turn-1's.

CRITICAL (skill §7): the saver runs on a **file-backed SQLite engine with a real connection
pool** (WAL + busy_timeout), NEVER ``:memory:``/``StaticPool`` — a single-connection setup
false-greens durable-resume/turn-boundary behaviour. PostgreSQL / MariaDB legs run when
``WATTWISE_PG_DSN`` / ``WATTWISE_MARIADB_DSN`` are set, proving the behaviour is backend-portable.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import wattwise_core.agent.memory  # noqa: F401  (registers agent_memory_item on AgentStateBase)
from wattwise_core.agent.checkpoint import SqlAlchemyCheckpointSaver
from wattwise_core.agent.contracts import (
    AgentState,
    Claim,
    ClaimKind,
    GroundDecision,
    GroundedClaim,
    GroundingResult,
    GroundVerdict,
    ReflectDecision,
    ReflectVerdict,
    RetrievalRequest,
    RunStatus,
)
from wattwise_core.agent.graph import AgentServices, build_graph
from wattwise_core.agent.graph_state import read_retrieved
from wattwise_core.agent.state_store import AgentStateBase

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

pytestmark = pytest.mark.integration

ATHLETE = "00000000-0000-7000-8000-0000000000aa"
CONVERSATION = "conv-multi"
THREAD_ID = "thread-multi-conv"


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + long busy_timeout per SQLite connection so the real pool serialises writers."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


def _engine_backends() -> list[ParameterSet]:
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


@pytest_asyncio.fixture(params=_engine_backends())
async def factory(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Session factory over a REAL multi-connection pool on the chosen backend (skill §7)."""
    backend_dsn = request.param
    if backend_dsn is None:
        dsn = f"sqlite+aiosqlite:///{tmp_path}/agent.sqlite"
        engine = create_async_engine(dsn, connect_args={"timeout": 30})
        event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
    else:
        engine = create_async_engine(backend_dsn)
    async with engine.begin() as conn:
        await conn.run_sync(AgentStateBase.metadata.create_all)
    made = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield made
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(AgentStateBase.metadata.drop_all)
        await engine.dispose()


# --- fakes (satisfy the public seams only; mirror tests/unit/test_graph.py) -------------


class _Model:
    """Deterministic ``ChatModel`` stub: scripts the reflect verdict, drafts from context."""

    def __init__(self, *, reflect_verdict: ReflectVerdict = ReflectVerdict.REPLAN) -> None:
        self.compose_calls = 0
        self._reflect_verdict = reflect_verdict

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        if schema is ReflectDecision:
            return ReflectDecision(verdict=self._reflect_verdict)  # type: ignore[return-value]
        raise NotImplementedError(schema.__name__)

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_calls += 1
        return f"draft#{self.compose_calls}"


class _Planner:
    """One capability request per call (a distinct ``n`` so requests never dedupe away)."""

    def __init__(self) -> None:
        self.calls = 0

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        self.calls += 1
        return [RetrievalRequest(capability="pmc", params={"n": self.calls})]


class _TurnGateway:
    """Returns a turn-UNIQUE record key each gather so cross-turn leakage is observable.

    The key embeds ``self.turn`` (set by the test before each turn), so turn-1's record is
    ``rec:turn-1`` and turn-2's is ``rec:turn-2`` — a turn-2 reader that still sees ``rec:turn-1``
    would prove a leak across the durable thread.
    """

    def __init__(self) -> None:
        self.turn = ""

    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        return {f"rec:{self.turn}": {"value": 42.0, "relevance": 1.0}}


class _Coverage:
    """No open gaps -> the happy path runs straight through to a COMPLETED finalize."""

    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        return set()


class _RecordingGrounder:
    """Proceeds, and RECORDS the ``retrieved`` keys it was handed on each grounding call.

    The grounder reads ``retrieved`` through the same turn-keyed view every node uses, so the
    keys captured here are exactly what the run could cite — the F-LEAK observation point.
    """

    def __init__(self) -> None:
        self.seen_keys: list[list[str]] = []

    async def ground(
        self, *, athlete_id: str, draft: str, retrieved: Mapping[str, Any]
    ) -> GroundingResult:
        self.seen_keys.append(sorted(retrieved.keys()))
        claim = Claim(kind=ClaimKind.NUMBER, text="42", value=42.0)
        survivor = GroundedClaim(
            claim=claim, verdict=GroundVerdict.GROUNDED, citation={"metric": "pmc"}
        )
        return GroundingResult(
            decision=GroundDecision.PROCEED, claims=(survivor,), scrubbed_text=draft
        )


def _services() -> tuple[_Model, AgentServices, _TurnGateway, _RecordingGrounder]:
    gateway = _TurnGateway()
    grounder = _RecordingGrounder()
    svc = AgentServices(
        planner=_Planner(), gateway=gateway, coverage=_Coverage(), grounder=grounder
    )
    return _Model(), svc, gateway, grounder


def _turn_input(turn_label: str) -> AgentState:
    """A fresh ``/ask`` turn input: a NEW ``turn_id`` on the SAME athlete + thread.

    A normal ``ainvoke`` mints a fresh ``turn_id`` per turn; the durable thread (and its
    ``run_epoch``) persists, so ``ingest_request`` sees ``turn_id != run_epoch`` and resets.
    ``athlete_id`` and ``idempotency_key`` are the THREAD's write-once immutable identity
    (STATE-R4) and so are STABLE across the turns of one durable thread — only ``turn_id`` and
    the request body vary per turn (the per-turn discriminator that drives the run-scoped reset).
    """
    return AgentState(
        athlete_id=ATHLETE,
        trigger="user_turn",
        request_text=f"question for {turn_label}",
        locale="en",
        idempotency_key="idem-thread-multi",
        thread_id=THREAD_ID,
        turn_id=turn_label,
    )


def _config() -> RunnableConfig:
    return {"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}, "recursion_limit": 50}


def _saver(factory: async_sessionmaker[AsyncSession]) -> SqlAlchemyCheckpointSaver:
    return SqlAlchemyCheckpointSaver(
        factory, athlete_id=ATHLETE, conversation_id=CONVERSATION
    )


# --- F-8TURN -----------------------------------------------------------------------------


async def test_eight_turns_on_one_durable_thread_all_complete(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """F-8TURN: >=8 sequential turns on ONE durable thread all COMPLETE, counters reset each turn.

    The reverted bug: a durable thread carrying ``count=N`` made the next turn's first
    ``count=1`` a monotonic-decrease the strict reducer raised on. Here every turn must finish
    ``COMPLETED`` and the run-scoped counters must be back at 0 (no gaps/redrafts/replans this
    turn), proving the head-node reset to the sentinel floor and the turn-monotonic reducer.
    """
    model, svc, gateway, _grounder = _services()
    graph = build_graph(model, svc, _saver(factory))
    cfg = _config()

    n_turns = 9
    for i in range(1, n_turns + 1):
        gateway.turn = f"turn-{i}"
        out = await graph.ainvoke(_turn_input(f"turn-{i}"), config=cfg)
        assert out["status"] is RunStatus.COMPLETED, f"turn {i} must complete on the durable thread"
        # Run-scoped counters reset each turn (no recovery cycles this turn -> all 0).
        assert out["reflection_count"] == 0, f"turn {i}: reflection_count not reset"
        assert out["redraft_count"] == 0, f"turn {i}: redraft_count not reset"
        # node_visits is a run-scoped counter too: it must NOT keep climbing across turns.
        # A single clean turn visits a small fixed number of nodes; if it accumulated across
        # 9 turns it would be ~9x larger. Bound it well under the cross-turn accumulation.
        visits = out["node_visits"]
        assert visits <= 12, f"turn {i}: node_visits did not reset ({visits})"
        # run_epoch advanced to this turn (the head node stamped it on reset).
        assert out["run_epoch"] == f"turn-{i}"


# --- F-LEAK ------------------------------------------------------------------------------


async def test_turn_two_cannot_cite_turn_one_evidence(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """F-LEAK: turn-2's grounder sees ONLY turn-2's record — never turn-1's (leak backstop).

    Each turn's gateway returns a turn-unique key; the grounder records the ``retrieved`` keys it
    was handed. After two turns on the SAME durable thread the captured sets must be exactly
    ``[{rec:turn-1}, {rec:turn-2}]`` — turn-1's record must be ABSENT from turn-2 (the head-node
    reset + the turn-keyed reducer drop the stale evidence at the boundary, CKPT-R5).
    """
    model, svc, gateway, grounder = _services()
    graph = build_graph(model, svc, _saver(factory))
    cfg = _config()

    gateway.turn = "turn-1"
    out1 = await graph.ainvoke(_turn_input("turn-1"), config=cfg)
    assert out1["status"] is RunStatus.COMPLETED

    gateway.turn = "turn-2"
    out2 = await graph.ainvoke(_turn_input("turn-2"), config=cfg)
    assert out2["status"] is RunStatus.COMPLETED

    assert grounder.seen_keys == [["rec:turn-1"], ["rec:turn-2"]], (
        "turn-2 grounding must see only turn-2 evidence; turn-1's record must not leak across "
        "the durable thread"
    )
    # The persisted channel itself carries only turn-2's record (the reducer self-reset).
    assert read_retrieved(out2) == {"rec:turn-2": {"value": 42.0, "relevance": 1.0}}
    assert "rec:turn-1" not in read_retrieved(out2)
