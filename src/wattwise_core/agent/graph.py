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
  depends only on :mod:`wattwise_core.agent.contracts` and the stable structured-output
  helper :mod:`wattwise_core.agent.structured` (itself contracts-only), used to obtain the
  reflect node's provider-enforced §6 verdict (REFLECT-R2 / STRUCT-R1).
* REFLECT-R4 — reflection and redraft are governed by strictly bounded monotonic
  counters; on exhaustion the run degrades gracefully (a :data:`RunStatus.DEGRADED`
  outcome) rather than looping forever or raising.
* OUTCOME-R1 — ``finalize`` is the single sink and emits exactly one
  :class:`RunStatus`.

Identity is server-derived only (AGT-SEC-R1): ``athlete_id`` is read from the
immutable input state and never taken from a model- or tool-produced value.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from wattwise_core.agent import graph_routing as routing
from wattwise_core.agent import graph_state as gs
from wattwise_core.agent.contracts import (
    AgentState,
    ChatModel,
    GroundDecision,
    RunStatus,
    stamp_coverage_gaps,
    stamp_retrieved,
)
from wattwise_core.agent.seams import (
    AgentServices,
    GraphNode,
    entitlement_max_tool_iterations,
    entitlement_node_visit_ceiling,
)

# Bounded recovery budgets (REFLECT-R4), re-exported from :mod:`graph_state` so the cycle
# bounds are auditable in one place; both are small and strictly enforced.
MAX_REFLECTIONS = gs.MAX_REFLECTIONS
MAX_REDRAFTS = gs.MAX_REDRAFTS

# Fallback node-visit ceiling (GRAPH-R5) for a caller that injects NO entitlement-supplied
# bound. The AUTHORITATIVE ceiling is read FROM the resolved entitlement carried on the
# cost gate (AGT-ENT-R1: the engine reads its gated limits from the entitlement, never a
# hardcode) — see :func:`_resolve_node_visit_ceiling`. This constant is only the explicit
# ``build_graph`` argument default for an isolated caller whose gate carries no bound
# (e.g. a unit test constructing the graph with a bare ``AgentServices``); a deployment's
# real ceiling lives in config (``entitlement.node_visit_ceiling``) and flows through the
# entitlement. An overall step bound, independent of the per-cycle counters, so a
# pathological run terminates GRACEFULLY at ``finalize`` with a ``degraded`` status instead
# of raising langgraph's ``GraphRecursionError`` (OUTCOME-R3).
DEFAULT_NODE_VISIT_CEILING = 60

# Fallback tool-iteration bound (AGT-ENT-R4) for a caller that injects NO entitlement-supplied
# bound — the tool-loop analogue of :data:`DEFAULT_NODE_VISIT_CEILING`. The AUTHORITATIVE bound is
# read FROM the resolved entitlement carried on the cost gate (AGT-ENT-R1) via
# :func:`~wattwise_core.agent.seams.entitlement_max_tool_iterations`; this constant is only the
# explicit ``build_graph`` argument default for an isolated caller whose gate carries no bound. It
# bounds the number of REAL gather/tool resolutions a single run may perform, independently of the
# per-cycle reflect/redraft counters and the overall node-visit ceiling, so a re-plan loop that
# keeps gathering terminates GRACEFULLY at ``compose`` (degraded), never by raising. Generous so a
# legit run (which gathers only a handful of times) never trips it.
DEFAULT_MAX_TOOL_ITERATIONS = 16


# --- node implementations (GRAPH-R4: pure (state, svc) -> partial update) ---------------
#
# The pure state readers + the deterministic terminal-status logic live in
# :mod:`wattwise_core.agent.graph_state` (``gs``); the node factories below own only the
# service calls + the partial-update assembly, keeping each file under the size ceilings.


def _make_ingest_request(svc: AgentServices) -> GraphNode:
    async def ingest_request(state: AgentState) -> dict[str, Any]:
        """Open the run: run-scoped reset + cost-admission gate + record the inbound turn.

        GRAPH-R2 head / CKPT-R5: the SINGLE head node, hence the single writer of the new-turn
        reset. On a NEW turn (``turn_id != run_epoch``) it resets every run-scoped channel via
        ``gs.reset_run_scoped`` (counters -> floor 0; ``retrieved`` / ``coverage_gaps`` -> empty
        stamped with the new ``turn_id``; ``run_epoch`` -> ``turn_id``) so a durable thread reused
        across turns never leaks turn-1 evidence/counters into turn-2; ``Command(resume)`` does
        NOT run the head, preserving the channels across the pause. Then the cost-admission gate
        (COST-R2; OSS no-op), request normalisation, a per-call cost event (STATE-R3); a refused
        admission stops the run -> budget_exceeded (COST-R4)."""
        athlete_id = gs.athlete_id(state)
        admitted = await svc.cost_gate.admit(athlete_id=athlete_id, state=state)
        text = state.get("request_text")
        messages: list[dict[str, Any]] = []
        if not admitted:
            messages.append({"role": "system", "kind": "budget", "admitted": False})
        elif text:
            messages.append({"role": "user", "content": text})
        update: dict[str, Any] = {
            "messages": messages,
            "cost_events": [{"node": "ingest_request", "kind": "admission", "admitted": admitted}],
        }
        if gs.is_new_turn(state):
            # Reset FIRST: ``reset_run_scoped`` sets ``node_visits`` to the floor 0 itself, so do
            # NOT ``tick_visit`` here (a tick would write N+1 and the reducer would reject the
            # mid-turn rewind). Subsequent nodes this turn tick up from 0.
            update.update(gs.reset_run_scoped(state))
            return update
        return gs.tick_visit(state, update)

    return ingest_request


def _make_plan_retrieval(svc: AgentServices) -> GraphNode:
    async def plan_retrieval(state: AgentState) -> dict[str, Any]:
        """Choose the next canonical capability requests (PLAN-R*). Pure update."""
        athlete_id = gs.athlete_id(state)
        gaps = sorted(gs.read_coverage_gaps(state))
        already = sorted(gs.read_retrieved(state).keys())
        requests = await svc.planner.plan(
            request_text=state.get("request_text"), gaps=gaps, already=already
        )
        plan_msg = {
            "role": "system",
            "kind": "plan",
            "athlete_id": athlete_id,
            "requests": [{"capability": r.capability, "params": r.params} for r in requests],
        }
        return gs.tick_visit(state, {"messages": [plan_msg]})

    return plan_retrieval


def _make_gather(svc: AgentServices) -> GraphNode:
    async def gather(state: AgentState) -> dict[str, Any]:
        """Resolve the planned requests to canonical evidence (TOOL-R1, STATE-R6).

        Merges resolved records into ``retrieved`` via the state's bounded keyed-merge
        reducer; fabricates nothing. A STATE-R6 reducer truncation is surfaced into
        ``coverage_gaps`` so the omission is visible downstream.
        """
        athlete_id = gs.athlete_id(state)
        requests = gs.last_plan_requests(state)
        if not requests:
            return gs.tick_visit(state, {})
        records = await svc.gateway.gather(athlete_id=athlete_id, requests=requests)
        # Every write to the turn-keyed accumulators MUST be stamped with the current turn_id
        # so the reducer self-resets across a turn boundary (CKPT-R5 leak backstop).
        tid = gs.turn_id(state)
        # Advance the monotonic tool-iteration counter on each REAL capability resolution
        # (AGT-ENT-R4): this is the gather/tool-loop step the entitlement's max_tool_iterations
        # bounds. A no-op gather (no planned requests, handled above) does NOT advance it, so the
        # bound counts only real tool work.
        update: dict[str, Any] = {
            "retrieved": stamp_retrieved(tid, dict(records)),
            "tool_iterations": state.get("tool_iterations", 0) + 1,
        }
        truncated = gs.retrieved_truncation_gaps(state, dict(records))
        if truncated:
            update["coverage_gaps"] = stamp_coverage_gaps(tid, truncated)
        return gs.tick_visit(state, update)

    return gather


def _make_assess_coverage(svc: AgentServices) -> GraphNode:
    async def assess_coverage(state: AgentState) -> dict[str, Any]:
        """Compute remaining coverage gaps deterministically (PLAN-R*).

        Routing reads the freshly computed set recorded on the message log; the set-union
        reducer accumulates the typed field across cycles.
        """
        gs.athlete_id(state)
        gaps = svc.coverage.assess(
            request_text=state.get("request_text"), retrieved=gs.read_retrieved(state)
        )
        marker = {"role": "system", "kind": "coverage", "open_gaps": sorted(gaps)}
        update: dict[str, Any] = {
            "coverage_gaps": stamp_coverage_gaps(gs.turn_id(state), set(gaps)),
            "messages": [marker],
        }
        return gs.tick_visit(state, update)

    return assess_coverage


def _make_reflect(model: ChatModel) -> GraphNode:
    async def reflect(state: AgentState) -> dict[str, Any]:
        """Emit a structured §6 reflect verdict over the closed enum (REFLECT-R2).

        Spends one unit of the bounded reflection budget (REFLECT-R4) and obtains a
        provider-enforced ``ReflectDecision`` via ``run_structured`` (STRUCT-R1); the
        routing function reads the verdict.
        """
        gs.athlete_id(state)
        count = state.get("reflection_count", 0) + 1
        decision = await gs.reflect_decision(model, state)
        note = {
            "role": "system",
            "kind": "reflect",
            "reflection_count": count,
            "verdict": decision.verdict.value,
            "add_requests": list(decision.add_requests),
        }
        return gs.tick_visit(state, {"reflection_count": count, "messages": [note]})

    return reflect


def _make_compose(svc: AgentServices, model: ChatModel, coach_system: str) -> GraphNode:
    async def compose(state: AgentState) -> dict[str, Any]:
        """Draft prose from canonical evidence within a token budget (DELIV-R*, MODEL-R3).

        Untrusted content is wrapped in delimited data envelopes (INJECT-R1); on a context
        overflow the lowest-relevance records are trimmed and the trim is recorded in
        coverage_gaps. The redraft counter is already spent by the router upstream.
        """
        gs.athlete_id(state)
        retrieved = gs.read_retrieved(state)
        context, trimmed = gs.render_context(state.get("request_text"), retrieved)
        draft = await model.compose(system=coach_system, context=context)
        update: dict[str, Any] = {
            "draft": draft,
            "messages": [{"role": "assistant", "kind": "draft"}],
        }
        if trimmed:
            update["coverage_gaps"] = stamp_coverage_gaps(gs.turn_id(state), {"context_trimmed"})
        return gs.tick_visit(state, update)

    return compose


def _make_ground(svc: AgentServices) -> GraphNode:
    async def ground(state: AgentState) -> dict[str, Any]:
        """Deterministically verify the draft and decide recovery (GROUND-R*).

        Records the grounder's aggregate decision, the scrubbed text, a server-side
        sanitized HTML body (AGT-SEC-R2), and the surviving citations.
        """
        athlete_id = gs.athlete_id(state)
        draft = state.get("draft") or ""
        result = await svc.grounder.ground(
            athlete_id=athlete_id, draft=draft, retrieved=gs.read_retrieved(state)
        )
        citations = [gc.citation for gc in result.survivors if gc.citation is not None]
        verdict_msg = {"role": "system", "kind": "ground", "decision": result.decision.value}
        return gs.tick_visit(
            state,
            {
                "grounded_text": result.scrubbed_text,
                "grounded_html": gs.safe_html(result.scrubbed_text),
                "citations": citations,
                "messages": [verdict_msg],
            },
        )

    return ground


def _make_interrupt_gate(recorder: gs.InterruptRecorder | None) -> GraphNode:
    async def interrupt_gate(state: AgentState) -> dict[str, Any]:
        """Approval checkpoint between grounding and finalisation (GRAPH-R2, CKPT-R5/-R9).

        ONLY when the grounded deliverable is an approval-gated PLAN: it first persists a
        ``live`` ``AgentInterrupt`` ledger row (via the injected ``recorder`` = the durable
        checkpointer; CKPT-R9) BEFORE suspending, so a decision arriving against this thread
        always finds a live row to atomically CONSUME (guarded UPDATE) and can never resume
        twice. Then it PAUSES at a durable langgraph ``interrupt`` carrying
        ``{grounded_plan, thread_id, interrupt_id}``, emitting ``awaiting_approval`` HERE (it
        does not reach ``finalize``). A grounder abstain is NOT approval (it degrades at
        finalize); with no approval-gated plan the gate is a pass-through. ``recorder is None``
        (an in-memory checkpointer, the OSS/test default) raises the interrupt but records no
        row — nothing durable can be consumed against it.
        """
        gs.athlete_id(state)
        if not gs.plan_requires_approval(state):
            return gs.tick_visit(state, {})
        interrupt_id = str(uuid.uuid4())
        thread_id = state.get("thread_id")
        # KNOWN-ISSUE (HITL-hardening, out of scope here): langgraph re-runs this node body on every
        # RESUME before ``interrupt()`` returns the resume value, so this mints a FRESH uuid and
        # re-records a NEW ``live`` row on each resume — the interrupt-identity scheme needs a
        # redesign (stable per-pause id). Do NOT fix here; tracked separately.
        if recorder is not None and thread_id:
            await recorder.record_interrupt(thread_id, interrupt_id)
        payload = {
            "status": RunStatus.AWAITING_APPROVAL.value,
            "interrupt_id": interrupt_id,
            "thread_id": thread_id,
            "grounded_plan": state.get("grounded_text"),
        }
        # Durable pause: yields the awaiting_approval payload and suspends until a matching
        # approve/reject/edit decision resumes the thread (CKPT-R5/-R9).
        decision = interrupt(payload)
        approved = (
            bool(decision.get("approved")) if isinstance(decision, Mapping) else bool(decision)
        )
        return gs.tick_visit(
            state,
            {
                "interrupt_id": interrupt_id,
                "messages": [
                    {
                        "role": "system",
                        "kind": "approval",
                        "interrupt_id": interrupt_id,
                        "approved": approved,
                    }
                ],
            },
        )

    return interrupt_gate


def _make_finalize(svc: AgentServices, ceiling: int) -> GraphNode:
    async def finalize(state: AgentState) -> dict[str, Any]:
        """The single sink: emit completed|degraded|budget_exceeded only (OUTCOME-R1).

        ``awaiting_approval`` is emitted ONLY by ``interrupt_gate`` (a paused durable
        interrupt) and never here. Status is derived deterministically (OUTCOME-R5) by
        ``gs.terminal_status``; the cost-SETTLE gate is called here (COST-R3; OSS no-op).
        """
        athlete_id = gs.athlete_id(state)
        await svc.cost_gate.settle(athlete_id=athlete_id, state=state)
        decision = gs.last_ground_decision(state)
        status = gs.terminal_status(state, decision, ceiling)
        caveat = gs.build_caveat(state, status, decision)
        update: dict[str, Any] = {
            "status": status,
            "coverage_caveat": caveat,
            "cost_rollup": gs.cost_rollup(state, status),
            "thread_id": state.get("thread_id"),
            "cost_events": [{"node": "finalize", "kind": "settle", "status": status.value}],
        }
        # Fail closed on abstain (GROUND-R6): when grounding could not verify enough to
        # answer, the deliverable MUST be an explicit "insufficient grounded data"
        # limitation — never the last scrubbed draft (a partial non-answer). Replace the
        # body so the projected deliverable states the limitation, not a stale draft.
        if decision is GroundDecision.ABSTAIN:
            limitation = gs.limitation_text(state)
            update["grounded_text"] = limitation
            update["grounded_html"] = gs.safe_html(limitation)
            update["citations"] = []
        return gs.tick_visit(state, update)

    return finalize


def build_graph(
    model: ChatModel,
    svc: AgentServices,
    checkpointer: BaseCheckpointSaver[Any],
    *,
    coach_system: str = "",
    node_visit_ceiling: int = DEFAULT_NODE_VISIT_CEILING,
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
) -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    """Assemble and compile the agent graph (GRAPH-R1/R2/R3/R5).

    All services are injected; the returned graph is compiled with the supplied durable
    ``checkpointer`` so runs are resumable. ``awaiting_approval`` is emitted by a DURABLE
    langgraph ``interrupt`` raised inside ``interrupt_gate`` ONLY for an approval-gated
    PLAN deliverable (CKPT-R5) — Phase-1 ships none, so the gate is a pass-through.

    Two non-monetary local guards bound the run, BOTH read FROM the resolved entitlement carried
    on ``svc.cost_gate`` (AGT-ENT-R1, config-loaded guards) when present, else the explicit
    arguments (a caller override wins; an isolated caller with a bare gate falls back to the
    module defaults): (1) the ``node_visit_ceiling`` bounds TOTAL node visits — a breach routes to
    ``finalize`` with ``degraded`` (GRAPH-R5/OUTCOME-R3), never a GraphRecursionError; (2) the
    ``max_tool_iterations`` bounds the gather/tool loop independently — a breach STOPS re-planning
    and routes to ``compose`` (AGT-ENT-R4) so the run still composes a grounded answer from what it
    has. Both degrade GRACEFULLY, never raise.
    """
    builder: StateGraph[AgentState, Any, AgentState, AgentState] = StateGraph(AgentState)
    # Read the non-monetary local guards FROM the resolved entitlement carried on the cost gate
    # (AGT-ENT-R1) — the OSS plan's config-loaded guards govern the run; an explicit caller
    # override wins, and an isolated caller with a bare gate falls back to the module default.
    ceiling = entitlement_node_visit_ceiling(svc, DEFAULT_NODE_VISIT_CEILING, node_visit_ceiling)
    tool_bound = entitlement_max_tool_iterations(
        svc, DEFAULT_MAX_TOOL_ITERATIONS, max_tool_iterations
    )
    # The interrupt-gate persists its ``live`` AgentInterrupt ledger row through the
    # checkpointer when (and only when) it satisfies the ``record_interrupt`` seam (CKPT-R9);
    # an in-memory checkpointer does not, so the gate records nothing (and nothing durable can
    # be consumed against it). Detected structurally (ARCH-R21: no concrete-saver import).
    recorder = checkpointer if isinstance(checkpointer, gs.InterruptRecorder) else None

    # ``input_schema=AgentState`` binds each node's input type so the strict-typed
    # builder accepts the ``(AgentState) -> partial`` node signatures (GRAPH-R4).
    builder.add_node("ingest_request", _make_ingest_request(svc), input_schema=AgentState)
    builder.add_node("plan_retrieval", _make_plan_retrieval(svc), input_schema=AgentState)
    builder.add_node("gather", _make_gather(svc), input_schema=AgentState)
    builder.add_node("assess_coverage", _make_assess_coverage(svc), input_schema=AgentState)
    builder.add_node("reflect", _make_reflect(model), input_schema=AgentState)
    builder.add_node("redraft_tick", routing.make_redraft_tick(), input_schema=AgentState)
    builder.add_node("compose", _make_compose(svc, model, coach_system), input_schema=AgentState)
    builder.add_node("ground", _make_ground(svc), input_schema=AgentState)
    builder.add_node("interrupt_gate", _make_interrupt_gate(recorder), input_schema=AgentState)
    builder.add_node("finalize", _make_finalize(svc, ceiling), input_schema=AgentState)

    # Fixed spine (GRAPH-R2). Admission-refused short-circuits ingest -> finalize (COST-R4).
    builder.add_edge(START, "ingest_request")
    builder.add_conditional_edges(
        "ingest_request",
        routing.route_after_ingest,
        {"plan_retrieval": "plan_retrieval", "finalize": "finalize"},
    )
    builder.add_edge("plan_retrieval", "gather")
    builder.add_edge("gather", "assess_coverage")

    # Cycle 1: assess_coverage -> reflect -> plan_retrieval (GRAPH-R3); ceiling -> finalize.
    # The tool-iteration bound stops the re-plan loop at compose (AGT-ENT-R4).
    builder.add_conditional_edges(
        "assess_coverage",
        routing.make_route_after_assess(ceiling, tool_bound),
        {"reflect": "reflect", "compose": "compose", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "reflect",
        routing.make_route_after_reflect(ceiling, tool_bound),
        {"plan_retrieval": "plan_retrieval", "compose": "compose", "finalize": "finalize"},
    )

    builder.add_edge("compose", "ground")

    # Cycle 2 (ground -> compose, redraft) + cycle 3 (ground -> reflect -> plan).
    # The redraft path runs through ``redraft_tick`` so the bounded counter advances.
    builder.add_conditional_edges(
        "ground",
        routing.make_route_after_ground(ceiling),
        {
            "compose": "redraft_tick",
            "reflect": "reflect",
            "interrupt_gate": "interrupt_gate",
            "finalize": "finalize",
        },
    )
    builder.add_edge("redraft_tick", "compose")

    # Gate -> single sink (GRAPH-R2 / OUTCOME-R1).
    builder.add_edge("interrupt_gate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "DEFAULT_MAX_TOOL_ITERATIONS",
    "DEFAULT_NODE_VISIT_CEILING",
    "MAX_REDRAFTS",
    "MAX_REFLECTIONS",
    "AgentServices",
    "build_graph",
]
