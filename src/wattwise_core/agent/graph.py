"""The trustworthy coaching agent's compiled LangGraph state machine (doc 50).

Requirements implemented here:

* GRAPH-R1 — one ``StateGraph`` over :class:`~wattwise_core.agent.contracts.AgentState`,
  compiled with a durable checkpointer so every run is resumable.
* GRAPH-R2 — the fixed node spine
  ``ingest_request -> plan_retrieval -> gather -> assess_coverage -> reflect ->
  compose -> ground -> interrupt_gate -> finalize`` with the approval gate sitting
  between grounding and finalisation.
* GRAPH-R3 — the only cycles permitted are
  ``assess_coverage -> reflect -> plan_retrieval`` (coverage recovery),
  ``ground -> compose`` (redraft recovery) and ``ground -> reflect ->
  plan_retrieval`` (re-plan recovery). No other back-edge exists.
* GRAPH-R4 — every node is a pure function of ``(state, injected services)``
  returning a partial :class:`AgentState` update; nodes never mutate the input
  mapping in place and never reach for ambient globals.
* GRAPH-R5 — services (the model, the canonical capability/gather layer and the
  deterministic grounder) are injected at :func:`build_graph` time behind typed
  seams; this module imports no sibling agent in-flight file (ARCH-R21) — it
  depends only on :mod:`wattwise_core.agent.contracts`.
* REFLECT-R4 — reflection and redraft are governed by strictly bounded monotonic
  counters; on exhaustion the run degrades gracefully (a :data:`RunStatus.DEGRADED`
  outcome) rather than looping forever or raising.
* OUTCOME-R1 — ``finalize`` is the single sink and emits exactly one
  :class:`RunStatus`.

Identity is server-derived only (AGT-SEC-R1): ``athlete_id`` is read from the
immutable input state and never taken from a model- or tool-produced value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from wattwise_core.agent.contracts import (
    AgentState,
    ChatModel,
    GroundDecision,
    GroundingResult,
    RetrievalRequest,
    RunStatus,
)

# Bounded recovery budgets (REFLECT-R4). Kept as module constants so the cycle
# bounds are auditable in one place; both are small and strictly enforced.
MAX_REFLECTIONS = 2
MAX_REDRAFTS = 2


class GraphNode(Protocol):
    """A graph node: a pure call from typed state to a partial update (GRAPH-R4).

    Structurally matches langgraph's node protocol (``__call__(state) -> Any``) so the
    strict-typed builder accepts both sync and async node implementations without
    reaching into langgraph internals.
    """

    def __call__(self, state: AgentState) -> Any: ...


# --- injected-service seams (GRAPH-R5) -------------------------------------------------
#
# The graph never imports the concrete planner / capability / grounding modules
# (sibling in-flight files, ARCH-R21). It depends only on these narrow Protocols,
# satisfied by whatever bundle :func:`build_graph` is handed. Everything is keyed
# off the server-derived ``athlete_id`` carried in the immutable input state.


@runtime_checkable
class Planner(Protocol):
    """Selects the next batch of canonical capability requests (PLAN-R*).

    ``plan`` is pure w.r.t. the graph: it reads the (immutable) request plus the
    accumulated coverage gaps and returns the capability requests to gather. It
    returns an empty sequence when nothing further is worth retrieving.
    """

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]: ...


@runtime_checkable
class CapabilityGateway(Protocol):
    """Resolves capability requests to canonical evidence records (TOOL-R1).

    Maps 1:1 onto the analytics/canonical service; returns a record per resolved
    request keyed by a canonical record id. Numbers are verbatim canonical values
    — this layer fabricates nothing (fail-closed).
    """

    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class CoverageAssessor(Protocol):
    """Deterministically reports which planned needs remain uncovered (PLAN-R*)."""

    def assess(
        self, *, request_text: str | None, retrieved: Mapping[str, Any]
    ) -> set[str]: ...


@runtime_checkable
class Grounder(Protocol):
    """Deterministic fail-closed grounder over a draft's claims (GROUND-R*).

    Verifies each claimed number/name/URL against canonical evidence, scrubs the
    unmatched, and returns an aggregate :class:`GroundingResult` carrying the
    bounded recovery :class:`GroundDecision`.
    """

    async def ground(
        self, *, athlete_id: str, draft: str, retrieved: Mapping[str, Any]
    ) -> GroundingResult: ...


@dataclass(frozen=True, slots=True)
class AgentServices:
    """The injected service bundle the graph nodes call (GRAPH-R5).

    A frozen record of the four seams above. Bundling them keeps node signatures
    and :func:`build_graph` to one ``svc`` argument while preserving per-seam typing.
    """

    planner: Planner
    gateway: CapabilityGateway
    coverage: CoverageAssessor
    grounder: Grounder


# --- node implementations (GRAPH-R4: pure (state, svc) -> partial update) ---------------


def _athlete_id(state: AgentState) -> str:
    """Read the server-derived athlete id from immutable input (AGT-SEC-R1).

    Fail-closed: a run with no server-set identity cannot proceed; we never invent
    or accept a model/tool-supplied id.
    """
    athlete_id = state.get("athlete_id")
    if not athlete_id:
        raise ValueError("athlete_id is server-derived and required (AGT-SEC-R1)")
    return athlete_id


def _ingest_request(state: AgentState) -> dict[str, Any]:
    """Open the run: record the inbound turn as the first message (GRAPH-R2 head).

    Pure: validates identity, normalises the request into working memory, seeds no
    counters (those default via the state reducers).
    """
    _athlete_id(state)
    text = state.get("request_text")
    messages: list[dict[str, Any]] = []
    if text:
        messages.append({"role": "user", "content": text})
    return {"messages": messages}


def _make_plan_retrieval(svc: AgentServices) -> GraphNode:
    async def plan_retrieval(state: AgentState) -> dict[str, Any]:
        """Choose the next canonical capability requests (PLAN-R*).

        Reads accumulated coverage gaps + what is already retrieved; stashes the
        chosen requests in working memory for ``gather``. Pure update.
        """
        athlete_id = _athlete_id(state)
        gaps = sorted(state.get("coverage_gaps", set()))
        already = sorted(state.get("retrieved", {}).keys())
        requests = await svc.planner.plan(
            request_text=state.get("request_text"), gaps=gaps, already=already
        )
        plan_msg = {
            "role": "system",
            "kind": "plan",
            "athlete_id": athlete_id,
            "requests": [{"capability": r.capability, "params": r.params} for r in requests],
        }
        return {"messages": [plan_msg]}

    return plan_retrieval


def _last_plan_requests(state: AgentState) -> list[RetrievalRequest]:
    """Recover the most recent plan's requests from working memory (pure read)."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "plan":
            raw = msg.get("requests", [])
            return [RetrievalRequest(capability=r["capability"], params=r["params"]) for r in raw]
    return []


def _make_gather(svc: AgentServices) -> GraphNode:
    async def gather(state: AgentState) -> dict[str, Any]:
        """Resolve the planned requests to canonical evidence (TOOL-R1).

        Merges resolved records into ``retrieved`` via the state's keyed-merge
        reducer; fabricates nothing.
        """
        athlete_id = _athlete_id(state)
        requests = _last_plan_requests(state)
        if not requests:
            return {}
        records = await svc.gateway.gather(athlete_id=athlete_id, requests=requests)
        return {"retrieved": dict(records)}

    return gather


def _make_assess_coverage(
    svc: AgentServices,
) -> GraphNode:
    async def assess_coverage(state: AgentState) -> dict[str, Any]:
        """Compute remaining coverage gaps deterministically (PLAN-R*).

        The set-union reducer accumulates gaps across cycles; an empty result here
        does NOT clear prior gaps, so routing relies on the freshly computed set
        recorded on the message log rather than the cumulative state field.
        """
        _athlete_id(state)
        gaps = svc.coverage.assess(
            request_text=state.get("request_text"), retrieved=state.get("retrieved", {})
        )
        marker = {"role": "system", "kind": "coverage", "open_gaps": sorted(gaps)}
        return {"coverage_gaps": set(gaps), "messages": [marker]}

    return assess_coverage


def _open_gaps(state: AgentState) -> list[str]:
    """Read the freshest coverage assessment (pure)."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "coverage":
            gaps = msg.get("open_gaps", [])
            return list(gaps)
    return []


def _reflect(state: AgentState) -> dict[str, Any]:
    """Spend one unit of the bounded reflection budget (REFLECT-R4).

    Pure: increments the monotonic ``reflection_count`` and records a reflection
    note. Whether another planning pass is allowed is decided by the routing
    function, not here, so this node never loops.
    """
    _athlete_id(state)
    count = state.get("reflection_count", 0) + 1
    note = {"role": "system", "kind": "reflect", "reflection_count": count}
    return {"reflection_count": count, "messages": [note]}


def _make_compose(
    svc: AgentServices, model: ChatModel, coach_system: str
) -> GraphNode:
    async def compose(state: AgentState) -> dict[str, Any]:
        """Draft prose from canonical evidence using the injected model (DELIV-R*).

        On a redraft cycle the redraft counter is already spent by the router; this
        node only produces the next draft. Pure update of ``draft``.
        """
        athlete_id = _athlete_id(state)
        retrieved = state.get("retrieved", {})
        context = _render_context(athlete_id, state.get("request_text"), retrieved)
        draft = await model.compose(system=coach_system, context=context)
        return {"draft": draft, "messages": [{"role": "assistant", "kind": "draft"}]}

    return compose


def _render_context(athlete_id: str, request_text: str | None, retrieved: Mapping[str, Any]) -> str:
    """Serialise canonical evidence for the composer (pure, deterministic order)."""
    lines = [f"request: {request_text or ''}"]
    for key in sorted(retrieved):
        lines.append(f"{key}: {retrieved[key]}")
    return "\n".join(lines)


def _make_ground(svc: AgentServices) -> GraphNode:
    async def ground(state: AgentState) -> dict[str, Any]:
        """Deterministically verify the draft and decide recovery (GROUND-R*).

        Records the grounder's aggregate decision, the scrubbed text and the
        surviving citations. The routing function reads the decision to choose
        proceed / redraft / replan / abstain within the bounded budgets.
        """
        athlete_id = _athlete_id(state)
        draft = state.get("draft") or ""
        result = await svc.grounder.ground(
            athlete_id=athlete_id, draft=draft, retrieved=state.get("retrieved", {})
        )
        citations = [
            gc.citation for gc in result.survivors if gc.citation is not None
        ]
        verdict_msg = {
            "role": "system",
            "kind": "ground",
            "decision": result.decision.value,
        }
        return {
            "grounded_text": result.scrubbed_text,
            "citations": citations,
            "messages": [verdict_msg],
        }

    return ground


def _last_ground_decision(state: AgentState) -> GroundDecision | None:
    """Read the freshest grounding decision (pure)."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "ground":
            return GroundDecision(msg["decision"])
    return None


def _interrupt_gate(state: AgentState) -> dict[str, Any]:
    """Human-approval checkpoint between grounding and finalisation (GRAPH-R2).

    Records whether the grounded deliverable requires explicit athlete approval
    before it is emitted (a prescriptive/abstaining outcome). The compiled graph
    interrupts before ``finalize`` when configured; this node only stamps the
    decision so ``finalize`` can emit the correct single status. Pure update.
    """
    _athlete_id(state)
    decision = _last_ground_decision(state)
    approval_required = decision is GroundDecision.ABSTAIN
    gate = {
        "role": "system",
        "kind": "gate",
        "approval_required": approval_required,
    }
    return {"messages": [gate]}


def _approval_required(state: AgentState) -> bool:
    """Read the gate's approval flag (pure)."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("kind") == "gate":
            return bool(msg.get("approval_required"))
    return False


def _finalize(state: AgentState) -> dict[str, Any]:
    """The single sink: emit exactly one :class:`RunStatus` (OUTCOME-R1).

    Status is derived deterministically from the bounded-budget and grounding
    state already recorded — never from a model output:

    * exhausted reflection budget with gaps still open -> ``DEGRADED`` (REFLECT-R4);
    * grounder abstained / approval required -> ``AWAITING_APPROVAL``;
    * otherwise -> ``COMPLETED``.
    """
    _athlete_id(state)
    decision = _last_ground_decision(state)
    reflections = state.get("reflection_count", 0)
    redrafts = state.get("redraft_count", 0)
    gaps_open = bool(_open_gaps(state))
    budget_spent = reflections >= MAX_REFLECTIONS or redrafts >= MAX_REDRAFTS

    if decision is GroundDecision.ABSTAIN or _approval_required(state):
        status = RunStatus.AWAITING_APPROVAL
    elif budget_spent and (gaps_open or decision is not GroundDecision.PROCEED):
        status = RunStatus.DEGRADED
    else:
        status = RunStatus.COMPLETED

    caveat = {"open_gaps": _open_gaps(state)} if gaps_open else None
    return {"status": status, "coverage_caveat": caveat}


# --- routing (GRAPH-R3: the only permitted cycles) -------------------------------------


def _route_after_assess(state: AgentState) -> str:
    """assess_coverage -> reflect (recover) | compose (proceed) (GRAPH-R3).

    Loops back through reflection only while gaps remain AND the reflection budget
    is unspent; otherwise proceeds to compose. Bounded by ``MAX_REFLECTIONS`` so the
    cycle cannot run forever (REFLECT-R4).
    """
    if _open_gaps(state) and state.get("reflection_count", 0) < MAX_REFLECTIONS:
        return "reflect"
    return "compose"


def _route_after_reflect(state: AgentState) -> str:
    """reflect -> plan_retrieval (GRAPH-R3).

    Reflection always feeds another planning pass; the budget that admits a
    reflection is enforced upstream (``_route_after_assess`` / ``_route_after_ground``),
    so this edge is unconditional and the cycle stays bounded.
    """
    return "plan_retrieval"


def _route_after_ground(state: AgentState) -> str:
    """ground -> compose (redraft) | reflect (replan) | interrupt_gate (GRAPH-R3).

    * REGENERATE: redraft within ``MAX_REDRAFTS`` (ground -> compose);
    * REPLAN: gather more evidence within ``MAX_REFLECTIONS`` (ground -> reflect ->
      plan_retrieval);
    * PROCEED / ABSTAIN, or any budget exhausted: fall through to the approval gate.

    All recovery edges are budget-gated, so neither loop is unbounded (REFLECT-R4):
    on exhaustion control proceeds to the gate and the run degrades, never errors.
    """
    decision = _last_ground_decision(state)
    if decision is GroundDecision.REGENERATE and state.get("redraft_count", 0) < MAX_REDRAFTS:
        return "compose"
    if decision is GroundDecision.REPLAN and state.get("reflection_count", 0) < MAX_REFLECTIONS:
        return "reflect"
    return "interrupt_gate"


def _make_redraft_tick() -> GraphNode:
    """Spend one redraft-budget unit when ground routes back to compose.

    Implemented as a pass-through stamped onto ``compose`` via the router below so
    the monotonic ``redraft_count`` advances exactly once per redraft cycle.
    """

    def tick(state: AgentState) -> dict[str, Any]:
        return {"redraft_count": state.get("redraft_count", 0) + 1}

    return tick


def build_graph(
    model: ChatModel,
    svc: AgentServices,
    checkpointer: BaseCheckpointSaver[Any],
    *,
    coach_system: str = "",
    interrupt_on_approval: bool = False,
) -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    """Assemble and compile the agent graph (GRAPH-R1/R2/R3/R5).

    All services are injected; the returned graph is compiled with the supplied
    durable ``checkpointer`` so runs are resumable. When ``interrupt_on_approval``
    is set the compiled graph pauses before ``finalize`` (the approval gate), letting
    a server resume after human approval; tests drive it without that pause.
    """
    builder: StateGraph[AgentState, Any, AgentState, AgentState] = StateGraph(AgentState)

    # ``input_schema=AgentState`` binds each node's input type so the strict-typed
    # builder accepts the ``(AgentState) -> partial`` node signatures (GRAPH-R4).
    builder.add_node("ingest_request", _ingest_request, input_schema=AgentState)
    builder.add_node("plan_retrieval", _make_plan_retrieval(svc), input_schema=AgentState)
    builder.add_node("gather", _make_gather(svc), input_schema=AgentState)
    builder.add_node("assess_coverage", _make_assess_coverage(svc), input_schema=AgentState)
    builder.add_node("reflect", _reflect, input_schema=AgentState)
    builder.add_node("redraft_tick", _make_redraft_tick(), input_schema=AgentState)
    builder.add_node("compose", _make_compose(svc, model, coach_system), input_schema=AgentState)
    builder.add_node("ground", _make_ground(svc), input_schema=AgentState)
    builder.add_node("interrupt_gate", _interrupt_gate, input_schema=AgentState)
    builder.add_node("finalize", _finalize, input_schema=AgentState)

    # Fixed spine (GRAPH-R2).
    builder.add_edge(START, "ingest_request")
    builder.add_edge("ingest_request", "plan_retrieval")
    builder.add_edge("plan_retrieval", "gather")
    builder.add_edge("gather", "assess_coverage")

    # Cycle 1: assess_coverage -> reflect -> plan_retrieval (GRAPH-R3).
    builder.add_conditional_edges(
        "assess_coverage", _route_after_assess, {"reflect": "reflect", "compose": "compose"}
    )
    builder.add_conditional_edges(
        "reflect", _route_after_reflect, {"plan_retrieval": "plan_retrieval"}
    )

    builder.add_edge("compose", "ground")

    # Cycle 2 (ground -> compose, redraft) + cycle 3 (ground -> reflect -> plan).
    # The redraft path runs through ``redraft_tick`` so the bounded counter advances.
    builder.add_conditional_edges(
        "ground",
        _route_after_ground,
        {"compose": "redraft_tick", "reflect": "reflect", "interrupt_gate": "interrupt_gate"},
    )
    builder.add_edge("redraft_tick", "compose")

    # Gate -> single sink (GRAPH-R2 / OUTCOME-R1).
    builder.add_edge("interrupt_gate", "finalize")
    builder.add_edge("finalize", END)

    interrupt_before = ["finalize"] if interrupt_on_approval else None
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


__all__ = [
    "MAX_REDRAFTS",
    "MAX_REFLECTIONS",
    "AgentServices",
    "CapabilityGateway",
    "CoverageAssessor",
    "Grounder",
    "Planner",
    "build_graph",
]
