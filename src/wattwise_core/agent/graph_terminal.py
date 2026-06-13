"""Deterministic terminal-status + coverage-caveat logic (OUTCOME-R1/-R4/-R5; STATUS-R1).

Factored out of :mod:`wattwise_core.agent.graph_state` (QUAL-R9 module-size ceiling) as a
focused leaf: the single, side-effect-free selection of a run's ONE terminal
:class:`~wattwise_core.agent.contracts.RunStatus`, its typed coverage caveat, and the per-run
cost rollup. It reads only the small state readers in :mod:`graph_state` plus the run-trace
seam — no model or service call (GRAPH-R4). ``graph_state`` re-exports these names from its
own surface (a bottom-of-file import) so existing ``gs.terminal_status`` call sites are
unchanged; the cycle is broken because the readers this module imports are fully defined
before ``graph_state`` re-imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from wattwise_core.agent.contracts import (
    AgentState,
    CoverageCaveat,
    GroundDecision,
    RunStatus,
)
from wattwise_core.agent.graph_state import (
    budget_exceeded,
    open_gaps,
    over_ceiling,
    read_retrieved,
)
from wattwise_core.observability import runtrace


def gathered_metric_capability(state: AgentState) -> bool:
    """True iff this run gathered at least one canonical metric capability (STATUS-R1).

    A run whose deliverable is legitimately number-free (a pure motivational/scheduling
    follow-up that selected no metric capability) gathers NO canonical record, so it is exempt
    from the STATUS-R1 grounded-substance gate by construction. Any gathered capability record
    is a canonical metric capability (the registry is metric-only), so a non-empty gathered set
    means the request WAS data-grounded — a PROCEED with zero grounded survivors over it is the
    inverted-honesty defect STATUS-R1 forbids.
    """
    return bool(read_retrieved(state))


def grounded_survivor_count(state: AgentState) -> int:
    """The number of grounded survivors this run published, read off the projected citations.

    The ground node projects each grounded, citable survivor 1:1 into ``citations`` (every
    GROUNDED verdict carries its ``{metric, value, as_of}`` citation), so the citation count is
    a faithful proxy for the grounded-survivor count STATUS-R1 keys on — without re-running the
    grounder. Zero citations on a data-grounded PROCEED means nothing canonical survived.
    """
    cites = state.get("citations")
    return len(cites) if isinstance(cites, Sequence) else 0


def _empty_survivor_degrade(state: AgentState, decision: GroundDecision | None) -> bool:
    """True iff this is the STATUS-R1 data-grounded-PROCEED-with-zero-survivors honest refusal."""
    return (
        decision is GroundDecision.PROCEED
        and gathered_metric_capability(state)
        and grounded_survivor_count(state) == 0
    )


def is_honest_refusal(
    state: AgentState, status: RunStatus, decision: GroundDecision | None
) -> bool:
    """True iff the run is a fail-closed honest refusal that must ship the limitation copy.

    Either the grounder ABSTAINED (GROUND-R6), or terminal_status just degraded a data-grounded
    PROCEED that published ZERO grounded survivors (STATUS-R1). Both replace the body with the
    localized "insufficient grounded data" limitation and carry ``degraded`` fidelity.
    """
    if decision is GroundDecision.ABSTAIN:
        return True
    return status is RunStatus.DEGRADED and _empty_survivor_degrade(state, decision)


def terminal_status(state: AgentState, decision: GroundDecision | None, ceiling: int) -> RunStatus:
    """Deterministically pick the single terminal status (OUTCOME-R1/-R5; no self-grading).

    ``awaiting_approval`` is NEVER produced here (it is yielded by the durable interrupt at
    interrupt_gate). ``budget_exceeded`` on a refused admission; ``degraded`` on a ceiling
    breach, a grounder abstain, a bound-exhausted non-PROCEED recovery, open gaps, OR a
    data-grounded PROCEED that published ZERO grounded survivors (STATUS-R1); otherwise
    ``completed``.
    """
    if budget_exceeded(state):
        return RunStatus.BUDGET_EXCEEDED
    # DEGRADE on: a node-visit ceiling breach (GRAPH-R5); a non-PROCEED ground decision — a
    # grounder abstain OR a bound-exhausted regenerate/replan recovery that could not be
    # re-grounded (GROUND-R9/REFLECT-R4), never a complete-looking answer; open coverage gaps;
    # or the STATUS-R1 honest refusal — a PROCEED whose request WAS data-grounded (gathered
    # canonical capabilities) but published ZERO grounded survivors, a number-free non-answer
    # masquerading as a completion (the inverted-honesty defect). A run that gathered no metric
    # capability is number-free by design and exempt by construction.
    degrade = (
        over_ceiling(state, ceiling)
        or (decision is not None and decision is not GroundDecision.PROCEED)
        or open_gaps(state)
        or _empty_survivor_degrade(state, decision)
    )
    return RunStatus.DEGRADED if degrade else RunStatus.COMPLETED


def build_caveat(
    state: AgentState, status: RunStatus, decision: GroundDecision | None
) -> dict[str, Any] | None:
    """Build the typed OUTCOME-R4 coverage caveat for a non-completed outcome.

    A grounder abstain AND a STATUS-R1 empty-survivor degrade are both honest fail-closed
    non-answers (no concrete value could be verified), so both carry ``degraded`` fidelity —
    not the softer ``partial`` of a coverage-gap degrade that still answered.
    """
    gaps = open_gaps(state)
    if status is RunStatus.COMPLETED and not gaps:
        return None
    fidelity = "degraded" if is_honest_refusal(state, status, decision) else "partial"
    caveat = CoverageCaveat(missing=tuple(sorted(gaps)), fidelity=fidelity)
    return caveat.model_dump()


def cost_rollup(state: AgentState, status: RunStatus) -> dict[str, Any]:
    """Per-run cost/latency rollup carried on every outcome (OUTCOME-R2, AGT-OBS-R2).

    Carries the node-visit + cost-event counts (OUTCOME-R2) AND, when a run trace is active, the
    AGT-OBS-R2 per-run rollup read from the recorded model/tool spans: total prompt/completion
    tokens, total cost, total latency, the model-tier mix, the reflection count, and the scrub
    count — read from the real provider usage, never fabricated. Outside a run (no active trace)
    only the deterministic counts are present.
    """
    events = state.get("cost_events", [])
    rollup: dict[str, Any] = {
        "node_visits": state.get("node_visits", 0),
        "cost_event_count": len(events),
        "status": status.value,
    }
    trace = runtrace.active_trace()
    if trace is not None:
        rollup.update(trace.rollup(status.value))
    return rollup


__all__ = [
    "build_caveat",
    "cost_rollup",
    "gathered_metric_capability",
    "grounded_survivor_count",
    "is_honest_refusal",
    "terminal_status",
]
