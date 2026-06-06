"""Unit tests for the compiled agent graph (doc 50 GRAPH-R*, REFLECT-R*, OUTCOME-R1).

Offline and self-contained: every collaborator is an in-test fake that satisfies the
public seams in :mod:`wattwise_core.agent.contracts` (a ``ChatModel`` and the four
injected services bundled in :class:`AgentServices`). No sibling agent module is
imported (ARCH-R21) — the graph is exercised purely through ``build_graph`` and the
typed state. The checkpointer is langgraph's in-memory saver.

Asserted behaviours:

* GRAPH-R2/OUTCOME-R1 happy path: a clean run reaches the single ``finalize`` sink and
  emits exactly one ``RunStatus.COMPLETED``.
* REFLECT-R4 coverage exhaustion: when gaps never close, the reflection budget is spent
  exactly ``MAX_REFLECTIONS`` times and the run **degrades** (not loops, not errors).
* GRAPH-R3 redraft cycle is bounded: a grounder that always asks to REGENERATE spends
  the redraft budget exactly ``MAX_REDRAFTS`` times then settles, never looping forever.
* The approval gate (between ground and finalize) yields ``AWAITING_APPROVAL`` on an
  abstaining grounding decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from wattwise_core.agent.contracts import (
    AgentState,
    Claim,
    ClaimKind,
    GroundDecision,
    GroundedClaim,
    GroundingResult,
    GroundVerdict,
    RetrievalRequest,
    RunStatus,
)
from wattwise_core.agent.graph import (
    MAX_REDRAFTS,
    MAX_REFLECTIONS,
    AgentServices,
    build_graph,
)

pytestmark = pytest.mark.unit


# --- fakes (satisfy the public contracts only) -----------------------------------------


class FakeModel:
    """Deterministic ``ChatModel`` stub: counts compose calls, returns a fixed draft."""

    def __init__(self) -> None:
        self.compose_calls = 0

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        raise NotImplementedError("graph nodes under test do not call structured()")

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_calls += 1
        return f"draft#{self.compose_calls}: {context.splitlines()[0]}"


class FakePlanner:
    """Returns one capability request per call; records the athlete-free inputs seen."""

    def __init__(self) -> None:
        self.calls = 0

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        self.calls += 1
        return [RetrievalRequest(capability="pmc", params={"n": self.calls})]


class FakeGateway:
    """Resolves each request to a canonical record; asserts a server-derived id flows."""

    def __init__(self) -> None:
        self.seen_athlete_ids: list[str] = []

    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        self.seen_athlete_ids.append(athlete_id)
        return {f"rec:{r.capability}": {"value": 42.0} for r in requests}


class FakeCoverage:
    """Coverage assessor whose open-gap set is scripted to drive routing."""

    def __init__(self, gaps: set[str]) -> None:
        self._gaps = gaps

    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        return set(self._gaps)


class FakeGrounder:
    """Grounder returning a scripted aggregate decision; counts invocations."""

    def __init__(self, decision: GroundDecision) -> None:
        self._decision = decision
        self.calls = 0

    async def ground(
        self, *, athlete_id: str, draft: str, retrieved: Mapping[str, Any]
    ) -> GroundingResult:
        self.calls += 1
        claim = Claim(kind=ClaimKind.NUMBER, text="42", value=42.0)
        survivor = GroundedClaim(
            claim=claim, verdict=GroundVerdict.GROUNDED, citation={"metric": "pmc"}
        )
        return GroundingResult(
            decision=self._decision, claims=(survivor,), scrubbed_text=draft
        )


def _services(*, gaps: set[str], decision: GroundDecision) -> tuple[FakeModel, AgentServices]:
    model = FakeModel()
    svc = AgentServices(
        planner=FakePlanner(),
        gateway=FakeGateway(),
        coverage=FakeCoverage(gaps),
        grounder=FakeGrounder(decision),
    )
    return model, svc


def _input(athlete_id: str = "athlete-1") -> AgentState:
    return AgentState(
        athlete_id=athlete_id,
        trigger="user_turn",
        request_text="how is my fitness trending?",
        locale="en",
        idempotency_key="idem-1",
    )


def _config(thread: str) -> RunnableConfig:
    # recursion_limit bounds the supersteps — if any cycle were unbounded the run
    # would raise GraphRecursionError instead of completing within this budget.
    return {"configurable": {"thread_id": thread}, "recursion_limit": 50}


# --- tests -----------------------------------------------------------------------------


async def test_happy_path_completes() -> None:
    """No gaps, grounder proceeds -> single COMPLETED outcome (OUTCOME-R1)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("happy"))

    assert out["status"] is RunStatus.COMPLETED
    assert out["reflection_count"] == 0
    assert out["redraft_count"] == 0
    assert out["coverage_caveat"] is None
    # draft was produced and grounded exactly once (no recovery cycles).
    assert model.compose_calls == 1
    assert svc.grounder.calls == 1  # type: ignore[attr-defined]
    # citations survived grounding.
    assert out["citations"] == [{"metric": "pmc"}]


async def test_coverage_exhaustion_degrades() -> None:
    """Persistent gaps spend the reflection budget then DEGRADE (REFLECT-R4)."""
    model, svc = _services(gaps={"missing_ftp"}, decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("degrade"))

    assert out["status"] is RunStatus.DEGRADED
    # reflection budget spent exactly to the bound, not beyond.
    assert out["reflection_count"] == MAX_REFLECTIONS
    # the unresolved gap is surfaced as a caveat for the athlete-facing layer.
    assert out["coverage_caveat"] == {"open_gaps": ["missing_ftp"]}


async def test_redraft_cycle_is_bounded() -> None:
    """A grounder stuck on REGENERATE spends the redraft budget then settles (GRAPH-R3)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.REGENERATE)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("redraft"))

    # bounded: redraft happened exactly MAX_REDRAFTS times, no infinite loop.
    assert out["redraft_count"] == MAX_REDRAFTS
    # initial compose + one recompose per redraft.
    assert model.compose_calls == MAX_REDRAFTS + 1
    # budget exhausted with a non-proceed decision -> degraded, never an error.
    assert out["status"] is RunStatus.DEGRADED


async def test_replan_cycle_is_bounded() -> None:
    """A grounder stuck on REPLAN spends the reflection budget then settles (GRAPH-R3)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.REPLAN)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("replan"))

    assert out["reflection_count"] == MAX_REFLECTIONS
    assert out["status"] is RunStatus.DEGRADED


async def test_abstain_awaits_approval() -> None:
    """An abstaining grounding decision routes through the gate to AWAITING_APPROVAL."""
    model, svc = _services(gaps=set(), decision=GroundDecision.ABSTAIN)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("abstain"))

    assert out["status"] is RunStatus.AWAITING_APPROVAL


async def test_server_derived_identity_flows_not_client() -> None:
    """The gateway only ever sees the server-set athlete id (AGT-SEC-R1)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    await graph.ainvoke(_input(athlete_id="srv-id-99"), config=_config("identity"))

    assert svc.gateway.seen_athlete_ids == ["srv-id-99"]  # type: ignore[attr-defined]


async def test_missing_identity_fails_closed() -> None:
    """A run without a server-derived athlete id fails closed, not silently (AGT-SEC-R1)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    bad = _input()
    del bad["athlete_id"]
    with pytest.raises(ValueError, match="athlete_id"):
        await graph.ainvoke(bad, config=_config("noident"))


async def test_interrupt_before_finalize_pauses_run() -> None:
    """With approval interrupts on, the graph pauses before the single sink (GRAPH-R2)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver(), interrupt_on_approval=True)
    cfg = _config("pause")

    paused = await graph.ainvoke(_input(), config=cfg)
    # no terminal status yet: the run is held at the gate, before finalize.
    assert "status" not in paused or paused.get("status") is None

    # resuming (None input) runs finalize to a single COMPLETED outcome.
    resumed = await graph.ainvoke(None, config=cfg)
    assert resumed["status"] is RunStatus.COMPLETED
