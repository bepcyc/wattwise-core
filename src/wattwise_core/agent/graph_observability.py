"""Production observability recorders the agent graph nodes call (AGT-OBS-R1/-R4/-R7, OBS-R4).

The focused sibling of :mod:`wattwise_core.agent.graph` (QUAL-R9 size split) that owns the
metric/trace recording the ``ground`` and ``finalize`` nodes invoke, plus the node-span wrapper
every spine node is registered through, so the graph module stays under the size ceiling. These
are pure side-effect recorders onto the run trace + the process metrics surface (no graph state
mutated); outside a bound run the trace reads are no-ops.

Cited requirements: AGT-OBS-R1 (one run = one trace, per-node spans), AGT-OBS-R4 (per-scrub
observability), AGT-OBS-R7 (alertable health/quality signals), OBS-R4 (agent-quality signals
recorded in production).
"""

from __future__ import annotations

import inspect
from typing import Any

from wattwise_core.agent import graph_state as gs
from wattwise_core.agent.contracts import (
    AgentState,
    GroundDecision,
    GroundingResult,
    GroundVerdict,
    RunStatus,
)
from wattwise_core.agent.seams import GraphNode
from wattwise_core.observability import metrics as obs_metrics
from wattwise_core.observability import runtrace


def traced(name: str, node: GraphNode) -> GraphNode:
    """Wrap a node so each execution emits a span under the run trace (AGT-OBS-R1).

    Opens a ``span(name)`` around the node body so every node execution is recorded with
    start/end, status (``error`` if the body raises), and parent linkage under the single
    run trace; outside a run (no active trace) the span is a no-op and the node runs unchanged.
    Model/tool spans the node opens via the model seam nest INSIDE this node span. Supports
    both async and sync node implementations (the redraft-tick node is sync, GRAPH-R4): an
    awaitable result is awaited inside the span so its latency is captured.
    """

    async def traced_node(state: AgentState) -> Any:
        with runtrace.span(name):
            result = node(state)
            if inspect.isawaitable(result):
                return await result
            return result

    return traced_node


def record_grounding(result: GroundingResult) -> None:
    """Record the grounding run + per-scrub observability for production (AGT-OBS-R4, OBS-R4).

    Counts the grounding run and the claims this draft scrubbed (UNGROUNDED/CONTRADICTED — the
    "what was removed" of AGT-OBS-R4) onto the run trace + the production metrics surface, so the
    rolling grounding-scrub rate (AGT-OBS-R7) is monitorable in production, not only in CI.
    """
    obs_metrics.get_registry().increment(obs_metrics.GROUNDING_RUNS)
    scrubbed = sum(
        1
        for c in result.claims
        if c.verdict in (GroundVerdict.UNGROUNDED, GroundVerdict.CONTRADICTED)
    )
    runtrace.record_scrubs(scrubbed)


def record_terminal(status: RunStatus, decision: GroundDecision | None) -> None:
    """Record the alertable per-run terminal signals on the metrics surface (AGT-OBS-R7, OBS-R4).

    Counts the terminal status (so ``degraded``/``budget_exceeded`` rates are monitorable),
    a refusal on a grounder abstain, a reflection-exhaustion when the run spent the full
    reflection budget, and observes the per-run latency + computed cost (p50/p95 + cost per run).
    Read from the active run trace; outside a run it is a no-op.
    """
    registry = obs_metrics.get_registry()
    registry.increment(obs_metrics.RUN_TERMINAL, labels={"status": status.value})
    if decision is GroundDecision.ABSTAIN:
        registry.increment(obs_metrics.REFUSALS)
    trace = runtrace.active_trace()
    if trace is None:
        return
    if trace.reflection_count() >= gs.MAX_REFLECTIONS:
        registry.increment(obs_metrics.REFLECTION_EXHAUSTIONS)
    registry.observe(obs_metrics.RUN_LATENCY_SECONDS, trace.elapsed_seconds())
    registry.observe(obs_metrics.RUN_COST_USD, trace.rollup(status.value)["total_cost_usd"])


__all__ = ["record_grounding", "record_terminal", "traced"]
