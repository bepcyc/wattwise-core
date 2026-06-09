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
from wattwise_core.config import get_settings
from wattwise_core.entitlement import (
    Entitlements,
    OssEntitlementResolver,
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

    COST-R1/R2/R3, AGT-SEC-R3, AGT-ENT-R3: the admission and settle GATE POINTS exist in OSS;
    they CHECK the carried entitlement's feature flags and FAIL CLOSED when a flag is ungranted
    (``admit`` -> ``False`` -> the graph stops at the node boundary with ``budget_exceeded``,
    COST-R4). The OSS default plan permits everything, so OSS admits; a commercial plan that
    ungrants ``can_use_agent`` IS enforced WITHOUT an agent-graph change.

    The concrete :class:`EntitlementCostGate` ADDITIONALLY exposes an ``entitlement`` property
    (the resolved, server-derived plan the graph reads its non-monetary local guards from —
    the node-visit ceiling and the tool-iteration bound, AGT-ENT-R1). That property is
    deliberately NOT part of this minimal admit/settle Protocol so a deployment MAY plug a
    different gate shape through the same seam; the graph reads ``entitlement`` defensively
    (``getattr``) and falls back when a gate does not carry it.
    """

    async def admit(self, *, athlete_id: str, state: AgentState) -> bool: ...

    async def settle(self, *, athlete_id: str, state: AgentState) -> None: ...


class EntitlementCostGate:
    """The cost gate that CHECKS the carried entitlement and FAILS CLOSED (AGT-ENT-R3/-R4).

    Carries the resolved, server-derived :class:`Entitlements` (the resolve -> attach -> check
    seam). ``admit`` CHECKS the agent feature flag (``can_use_agent``): an ungranted flag
    refuses the run (returns ``False``) so the graph stops at the next node boundary and
    finalizes ``budget_exceeded`` (COST-R4) — never a fail-OPEN admit. Under the OSS
    all-permissive plan the flag is granted, so OSS admits every run. There is NO monetary
    budget (COMM-R20): the non-monetary local guards carried on the entitlement are enforced
    OUTSIDE this gate, never as a monetary reserve-then-settle here — the GRAPH reads the
    node-visit ceiling + the tool-iteration bound from :attr:`entitlement` (the routers stop
    the loop on a breach), the ENGINE reads the token bound (model output budget) + the
    wall-clock deadline from it, and the API RATE-LIMITER reads the request-rate bound. ``settle``
    is a non-monetary no-op (no rolling total to settle against under the OSS default).
    """

    __slots__ = ("_entitlement",)

    def __init__(self, entitlement: Entitlements | None = None) -> None:
        # An OSS all-permissive grant by default (every flag True). Production attaches the
        # config-resolved plan (with the loaded non-monetary bounds) via the default factory
        # below; a commercial resolver attaches a metered plan. A bare ``Entitlements()`` has
        # zero bounds (the "no entitlement-supplied ceiling" sentinel the graph falls back from).
        self._entitlement = entitlement if entitlement is not None else Entitlements()

    @property
    def entitlement(self) -> Entitlements:
        """The resolved entitlement the gate checks + the graph reads its bounds from."""
        return self._entitlement

    async def admit(self, *, athlete_id: str, state: AgentState) -> bool:
        """Admit the run only if the carried entitlement grants the agent feature (AGT-ENT-R3).

        Fails CLOSED: an entitlement that does not grant ``can_use_agent`` refuses admission
        (returns ``False``), so the graph routes ingest -> finalize and emits ``budget_exceeded``
        (COST-R4) rather than running ungated. Under the OSS all-permissive plan the flag is
        granted and every run is admitted. ``athlete_id`` is server-derived (AGT-SEC-R1); this
        gate never reads identity from a model/tool/payload.
        """
        return self._entitlement.can_use_agent

    async def settle(self, *, athlete_id: str, state: AgentState) -> None:
        """Non-monetary settle no-op (COMM-R20): no monetary reservation/rolling total in OSS."""
        return None


def _default_cost_gate() -> EntitlementCostGate:
    """Build the OSS default cost gate carrying the config-resolved all-permissive plan.

    The :class:`AgentServices` DEFAULT ``cost_gate`` (the factory below) for a caller that builds
    the bundle without naming one: the OSS :class:`OssEntitlementResolver` resolves the single
    all-permissive plan with the NON-monetary local guards LOADED FROM CONFIG (CFG-R1a — never
    hardcoded, AGT-ENT-R1), and the gate carries it so the graph reads its node-visit ceiling +
    tool-iteration bound FROM the resolved entitlement. If config is not loadable (e.g. an isolated
    unit constructing ``AgentServices`` with no settings on disk), it degrades to a bare
    all-permissive grant (zero bounds), and the graph falls back to its explicit ``build_graph``
    arguments — OSS behavior is unchanged either way. The flag is always granted here, so the OSS
    default always admits. (NOTE: the deployable engine ``GraphAgentEngine`` does NOT rely on this
    default — it builds an :class:`EntitlementCostGate` from the run's EFFECTIVE entitlement, the
    per-request one when threaded, else its config-resolved default — MED-2; this factory backs
    direct ``AgentServices`` constructors.)
    """
    try:
        plan = OssEntitlementResolver.from_settings(get_settings()).resolve("")
    except Exception:  # config-absent isolated callers degrade to a bare all-permissive grant
        plan = Entitlements()
    return EntitlementCostGate(plan)


@dataclass(frozen=True, slots=True)
class AgentServices:
    """The injected service bundle the graph nodes call (GRAPH-R5).

    A frozen record of the seams above. Bundling them keeps node signatures and
    ``build_graph`` to one ``svc`` argument while preserving per-seam typing. The
    ``cost_gate`` defaults to the OSS :class:`EntitlementCostGate` carrying the
    config-resolved all-permissive plan (AGT-ENT-R4) — a REAL resolve -> attach -> check
    seam, not a fail-open no-op.
    """

    planner: Planner
    gateway: CapabilityGateway
    coverage: CoverageAssessor
    grounder: Grounder
    cost_gate: CostGate = field(default_factory=_default_cost_gate)


def entitlement_node_visit_ceiling(svc: AgentServices, default_ceiling: int, explicit: int) -> int:
    """The node-visit ceiling, READ FROM the carried entitlement, not a hardcode (AGT-ENT-R1).

    Precedence (highest first):

    1. An EXPLICIT ``build_graph`` argument that DIFFERS from ``default_ceiling`` (the module
       default sentinel) — a caller forcing a specific ceiling (e.g. a test driving a tiny
       bound). It wins so a caller-supplied override is honored.
    2. The resolved, server-derived entitlement on ``svc.cost_gate`` — the OSS plan's
       non-monetary node-visit/step guard (AGT-ENT-R4), LOADED FROM CONFIG (CFG-R1a). The graph
       reads the ceiling FROM that entitlement rather than the hardcoded constant, so a config
       override of the bound is HONORED (ENT-R1-AC) and a commercial plan carrying a tighter
       ceiling is enforced WITHOUT an agent-graph change.
    3. The module-default fallback for an isolated caller whose gate carries no bound (the
       "no entitlement-supplied ceiling" sentinel) and who passed no explicit override.

    The production engine passes the module default explicitly, so step 1 is a no-op for it and
    the entitlement's config-loaded ceiling (step 2) governs the real agent run.
    """
    if explicit != default_ceiling:
        return explicit
    carried = getattr(svc.cost_gate, "entitlement", None)
    bound = getattr(carried, "node_visit_ceiling", 0)
    if isinstance(bound, int) and not isinstance(bound, bool) and bound > 0:
        return bound
    return explicit


def entitlement_max_tool_iterations(svc: AgentServices, default_bound: int, explicit: int) -> int:
    """The tool-iteration bound, READ FROM the carried entitlement, not a hardcode (AGT-ENT-R1).

    The tool-loop analogue of :func:`entitlement_node_visit_ceiling`, with the identical
    precedence ladder so the two non-monetary guards resolve the same way:

    1. An EXPLICIT ``build_graph`` argument that DIFFERS from ``default_bound`` (the module
       default sentinel) — a caller forcing a specific bound (e.g. a test driving a tiny
       tool-iteration budget). It wins so a caller-supplied override is honored.
    2. The resolved, server-derived entitlement on ``svc.cost_gate`` — the OSS plan's
       non-monetary tool-iteration guard (AGT-ENT-R4), LOADED FROM CONFIG (CFG-R1a). The graph
       reads the bound FROM that entitlement rather than a hardcoded constant, so a config
       override is HONORED (ENT-R1-AC) and a commercial plan carrying a tighter tool-iteration
       bound is enforced WITHOUT an agent-graph change.
    3. The module-default fallback for an isolated caller whose gate carries no bound (the
       "no entitlement-supplied bound" sentinel) and who passed no explicit override.

    The production engine passes the module default explicitly, so step 1 is a no-op for it and
    the entitlement's config-loaded bound (step 2) governs the real agent run.
    """
    if explicit != default_bound:
        return explicit
    carried = getattr(svc.cost_gate, "entitlement", None)
    bound = getattr(carried, "max_tool_iterations", 0)
    if isinstance(bound, int) and not isinstance(bound, bool) and bound > 0:
        return bound
    return explicit


__all__ = [
    "AgentServices",
    "CapabilityGateway",
    "CostGate",
    "CoverageAssessor",
    "EntitlementCostGate",
    "GraphNode",
    "Grounder",
    "Planner",
    "entitlement_max_tool_iterations",
    "entitlement_node_visit_ceiling",
]
