"""Injected-service seams for the agent graph (GRAPH-R5, ARCH-R21).

The graph never imports the concrete planner / capability / grounding modules (sibling
in-flight files, ARCH-R21). It depends only on these narrow Protocols, satisfied by
whatever bundle :func:`~wattwise_core.agent.graph.build_graph` is handed. Everything is
keyed off the server-derived ``athlete_id`` carried in the immutable input state
(AGT-SEC-R1). Factoring the seams here keeps :mod:`wattwise_core.agent.graph` under the
module-size ceiling (QUAL-R9).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from wattwise_core.agent.contracts import (
    AgentState,
    GroundingResult,
    RetrievalRequest,
)


class GraphNode(Protocol):
    """A graph node: a pure call from typed state to a partial update (GRAPH-R4).

    Structurally matches langgraph's node protocol (``__call__(state) -> Any``) so the
    strict-typed builder accepts both sync and async node implementations without
    reaching into langgraph internals.
    """

    def __call__(self, state: AgentState) -> Any: ...


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


@runtime_checkable
class CostGate(Protocol):
    """Entitlement/cost gate seam called at ingest (admission) + finalize (settle).

    COST-R1/R2/R3, AGT-SEC-R3: the admission and settle GATE POINTS exist in OSS even as
    a non-monetary no-op enforcing only local guards; a commercial metered resolver plugs
    in here WITHOUT an agent-graph change. ``admit`` returns ``True`` to proceed or
    ``False`` to stop the run at the node boundary with ``budget_exceeded`` (COST-R4).
    """

    async def admit(self, *, athlete_id: str, state: AgentState) -> bool: ...

    async def settle(self, *, athlete_id: str, state: AgentState) -> None: ...


class NoopCostGate:
    """OSS default :class:`CostGate`: a non-monetary no-op that always admits (COST-R1).

    The local guard (the node-visit ceiling, max reflect/redraft) is enforced by the graph
    itself; this seam exists so a metered resolver replaces it without a graph change.
    """

    async def admit(self, *, athlete_id: str, state: AgentState) -> bool:
        return True

    async def settle(self, *, athlete_id: str, state: AgentState) -> None:
        return None


@dataclass(frozen=True, slots=True)
class AgentServices:
    """The injected service bundle the graph nodes call (GRAPH-R5).

    A frozen record of the seams above. Bundling them keeps node signatures and
    ``build_graph`` to one ``svc`` argument while preserving per-seam typing. The
    ``cost_gate`` defaults to the OSS no-op (COST-R1).
    """

    planner: Planner
    gateway: CapabilityGateway
    coverage: CoverageAssessor
    grounder: Grounder
    cost_gate: CostGate = field(default_factory=NoopCostGate)


__all__ = [
    "AgentServices",
    "CapabilityGateway",
    "CostGate",
    "CoverageAssessor",
    "GraphNode",
    "Grounder",
    "NoopCostGate",
    "Planner",
]
