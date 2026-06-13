"""The agent graph's conditional-edge routing functions (GRAPH-R3: the only permitted cycles).

Factored out of :mod:`wattwise_core.agent.graph` (QUAL-R9 module-size split) so the graph file
holds only the node factories + the ``build_graph`` assembly. Everything here is a pure function
of the typed :class:`~wattwise_core.agent.contracts.AgentState` (GRAPH-R4): each
``_make_route_*`` factory closes over the resolved non-monetary bounds (the node-visit ceiling and
the entitlement's tool-iteration bound, AGT-ENT-R1) and returns the langgraph conditional-edge
callable that picks the next node. It depends only on :mod:`wattwise_core.agent.graph_state` (the
pure state readers + the bounded recovery budgets) and the closed enums in
:mod:`wattwise_core.agent.contracts`, never on a sibling in-flight agent file (ARCH-R21), so it
sits strictly BELOW :mod:`wattwise_core.agent.graph` in the import graph (no cycle).

The two cycle bounds both degrade GRACEFULLY (GRAPH-R5/OUTCOME-R3), never raise: a node-visit
breach routes to ``finalize`` (degraded); a tool-iteration breach STOPS re-planning and routes to
``compose`` so the run still composes a grounded answer from what it has (AGT-ENT-R4).
"""

from __future__ import annotations

from typing import Any

from wattwise_core.agent import graph_state as gs
from wattwise_core.agent.contracts import (
    AgentState,
    GroundDecision,
    ReflectVerdict,
)
from wattwise_core.agent.seams import GraphNode

# Bounded recovery budgets (REFLECT-R4) the routers compare against, sourced from graph_state.
MAX_REFLECTIONS = gs.MAX_REFLECTIONS
MAX_REDRAFTS = gs.MAX_REDRAFTS


def make_route_after_assess(ceiling: int, max_tool_iterations: int) -> Any:
    """Build the ``assess_coverage`` conditional-edge router (GRAPH-R3/R5)."""

    def _route_after_assess(state: AgentState) -> str:
        """assess_coverage -> reflect | compose | finalize (GRAPH-R3/R5).

        Loops back through reflection only while gaps remain AND the reflection budget is
        unspent. A node-visit-ceiling breach routes straight to ``finalize`` (degraded). A
        tool-iteration-bound breach (the entitlement's gather/tool guard, AGT-ENT-R4) STOPS
        re-planning and routes to ``compose`` — so the tool loop is bounded independently of
        node_visits and the run still composes a grounded answer from what it has (graceful).
        """
        if gs.over_ceiling(state, ceiling):
            return "finalize"
        if gs.over_tool_ceiling(state, max_tool_iterations):
            return "compose"
        if gs.plan_requires_approval(state):
            return "compose"
        if gs.open_gaps(state) and state.get("reflection_count", 0) < MAX_REFLECTIONS:
            return "reflect"
        return "compose"

    return _route_after_assess


def make_route_after_reflect(ceiling: int, max_tool_iterations: int) -> Any:
    """Build the ``reflect`` conditional-edge router (REFLECT-R2a/GRAPH-R5)."""

    def _route_after_reflect(state: AgentState) -> str:
        """reflect -> plan_retrieval | compose | finalize on the §6 verdict (REFLECT-R2a).

        ``replan`` -> plan_retrieval; ``answer_with_caveat`` -> compose;
        ``give_up_gracefully`` -> compose (REFLECT-R3: a caveated graceful-decline draft,
        never an empty body); a ceiling breach -> finalize (GRAPH-R5). A
        tool-iteration-bound breach STOPS re-planning and routes to ``compose`` (AGT-ENT-R4),
        so a ``replan`` verdict cannot drive the gather/tool loop past the entitlement's bound.
        """
        if gs.over_ceiling(state, ceiling):
            return "finalize"
        verdict = gs.last_reflect_verdict(state)
        if verdict is ReflectVerdict.ANSWER_WITH_CAVEAT:
            return "compose"
        if verdict is ReflectVerdict.GIVE_UP_GRACEFULLY:
            return "compose"
        if gs.over_tool_ceiling(state, max_tool_iterations):
            return "compose"
        return "plan_retrieval"

    return _route_after_reflect


def make_route_after_ground(ceiling: int) -> Any:
    """Build the ``ground`` conditional-edge router (GRAPH-R3/GROUND-R9)."""

    def _route_after_ground(state: AgentState) -> str:
        """ground -> compose (redraft) | reflect (replan) | interrupt_gate | finalize.

        REGENERATE redrafts within ``MAX_REDRAFTS``; REPLAN re-plans within ``MAX_REFLECTIONS``.
        A REGENERATE that has EXHAUSTED ``redraft_count`` does NOT abstain while reflection budget
        remains: it FALLS THROUGH to ``replan`` (reflect) if ``reflection_count < MAX_REFLECTIONS``
        (REFLECT-R4, spec §225/§451 "fall through to replan ... or abstain") so the run tries to
        re-gather before degrading — still strictly bounded by the two distinct monotonic counters,
        never an unbounded loop. PROCEED/ABSTAIN or full budget-exhaustion falls through to the
        gate; a node-visit-ceiling breach routes straight to ``finalize`` (GRAPH-R5).
        """
        if gs.over_ceiling(state, ceiling):
            return "finalize"
        decision = gs.last_ground_decision(state)
        reflections_left = state.get("reflection_count", 0) < MAX_REFLECTIONS
        if decision is GroundDecision.REGENERATE:
            if state.get("redraft_count", 0) < MAX_REDRAFTS:
                return "compose"
            # Redraft budget spent without a fully grounded result -> fall through to a bounded
            # re-plan if the reflection budget remains, else abstain at the gate (REFLECT-R4).
            if reflections_left:
                return "reflect"
        if decision is GroundDecision.REPLAN and reflections_left:
            return "reflect"
        return "interrupt_gate"

    return _route_after_ground


def make_redraft_tick() -> GraphNode:
    """Spend one redraft-budget unit when ground routes back to compose (REFLECT-R4)."""

    def tick(state: AgentState) -> dict[str, Any]:
        return gs.tick_visit(state, {"redraft_count": state.get("redraft_count", 0) + 1})

    return tick


def route_after_ingest(state: AgentState) -> str:
    """ingest_request -> finalize (admission refused) | plan_retrieval (COST-R4)."""
    return "finalize" if gs.budget_exceeded(state) else "plan_retrieval"


__all__ = [
    "MAX_REDRAFTS",
    "MAX_REFLECTIONS",
    "make_redraft_tick",
    "make_route_after_assess",
    "make_route_after_ground",
    "make_route_after_reflect",
    "route_after_ingest",
]
