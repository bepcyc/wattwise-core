"""The multi-day training PLAN deliverable: a grounded, approval-gated prescription (COACH-R2).

This is the focused sibling of :mod:`wattwise_core.agent.deliverables` that owns the multi-day
PLAN deliverable and NOTHING else (COACH-R2 / COACH-R1 #3 / CKPT-R5/-R9). It reuses the SHARED
graph-driving + projection primitives from the LEAF :mod:`wattwise_core.agent.projection` module
(the :class:`~wattwise_core.agent.projection.CoachGraph` seam, the run-input builder, and the
terminal-state projectors), so it depends DOWNWARD on that leaf — there is no
``deliverables`` <-> ``plan_deliverable`` cycle, and ``deliverables`` re-exports ``Plan``/``plan``
so every public import path stays stable.

Unlike the free-form answer, a PLAN is a PRESCRIPTIVE, APPROVAL-GATED deliverable: the run carries
a ``plan_deliverable`` marker so the graph's ``interrupt_gate`` (graph_state.plan_requires_approval)
pauses at a durable ``interrupt`` and finalizes at ``awaiting_approval`` carrying the
``interrupt_id`` the decision endpoint consumes (CKPT-R9). The prescribed workout NAMES ground
against the canonical workout library (GROUND-R2); the deliverable projects only what survives
grounding (OUTCOME-R2) and never surfaces un-grounded text.

Cited requirements: COACH-R1 #3, COACH-R2, CKPT-R5/-R9, GRAPH-R2.1, GROUND-R2/-R5, OUTCOME-R2,
AGT-SEC-R1, AGT-SEC-R2.
"""

from __future__ import annotations

import html as _html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from wattwise_core.agent.contracts import AgentState, RunStatus
from wattwise_core.agent.projection import (
    CoachGraph,
    ResponseLength,
)
from wattwise_core.agent.projection import (
    as_seq as _as_seq,
)
from wattwise_core.agent.projection import (
    build_inputs as _build_inputs,
)
from wattwise_core.agent.projection import (
    coverage_caveat as _coverage_caveat,
)
from wattwise_core.agent.projection import (
    generate_followups as _generate_followups,
)
from wattwise_core.agent.projection import (
    outputs as _outputs,
)
from wattwise_core.agent.projection import (
    project_observations as _project_observations,
)
from wattwise_core.agent.voice import Citation, Observation, _opt_str, _project_citations


@dataclass(frozen=True, slots=True)
class Plan:
    """A multi-day grounded training PLAN deliverable (COACH-R2, COACH-R1 #3).

    A prescriptive deliverable: grounded multi-day prescriptions (each workout NAME resolved
    against the canonical workout library, GROUND-R2) projected from a ``user_turn`` plan run.
    Unlike the free-form answer it is an APPROVAL-GATED deliverable (CKPT-R5/-R9): the graph
    raises a durable ``interrupt`` and the run finalizes at ``awaiting_approval`` carrying
    ``interrupt_id`` + the grounded plan body, which the athlete approves/edits/rejects via the
    decision endpoint before it becomes final. Same grounded-body/citation guarantees as
    :class:`~wattwise_core.agent.deliverables.AgentAnswer`; ``interrupt_id`` is set ONLY when the
    run paused for approval.
    """

    status: RunStatus
    thread_id: str
    plan_html: str
    plan_text: str
    interrupt_id: str | None = None
    observations: tuple[Observation, ...] = ()
    citations: tuple[Citation, ...] = ()
    suggested_followups: tuple[str, ...] = ()
    coverage_caveat: Mapping[str, Any] | None = None


async def plan(
    graph: CoachGraph,
    athlete_id: str,
    request: str,
    *,
    locale: str,
    response_length: ResponseLength = "detailed",
    thread_id: str | None = None,
    conversation_id: str | None = None,
    requires_approval: bool = True,
) -> Plan:
    """Drive the graph for a multi-day grounded training PLAN (COACH-R2 / CKPT-R5).

    Builds a ``user_turn`` run carrying the plan request AND a ``plan_deliverable`` marker
    (``requires_approval``) so the graph's ``interrupt_gate`` knows this is an approval-gated PLAN
    deliverable, not a free-form answer (graph_state.plan_requires_approval reads the marker). The
    prescribed workout NAMES ground against the canonical workout library, and the grounded
    multi-day body is projected into :class:`Plan`. When ``requires_approval`` and a DURABLE
    checkpointer is wired, the run pauses at the gate and finalizes ``awaiting_approval`` with the
    ``interrupt_id`` the decision endpoint consumes (CKPT-R9); otherwise it projects the grounded
    plan straight through. Identity is server-derived (AGT-SEC-R1); no un-grounded text is
    surfaced (OUTCOME-R2).
    """
    inputs = _build_inputs(
        athlete_id=athlete_id,
        trigger="user_turn",
        locale=locale,
        request_text=request,
        response_length=response_length,
        thread_id=thread_id,
        conversation_id=conversation_id,
    )
    marker = {"role": "system", "kind": "plan_deliverable", "requires_approval": requires_approval}
    inputs["messages"] = [*list(inputs.get("messages") or []), marker]
    final = await graph.run(inputs)
    return _project_plan(final)


def _interrupt_payload(final: AgentState) -> Mapping[str, Any] | None:
    """The approval-gate interrupt payload when the run PAUSED, else ``None`` (CKPT-R5).

    A durable ``interrupt`` makes langgraph return the accumulated state with an ``__interrupt__``
    entry — a sequence of ``Interrupt`` objects whose ``.value`` is the payload ``interrupt_gate``
    raised (``{status, interrupt_id, thread_id, grounded_plan}``). The channel updates the gate
    would have written are NOT applied (the node suspended), so the ``awaiting_approval`` status /
    interrupt_id live HERE, not in the state channels.
    """
    # ``__interrupt__`` is a langgraph RUNTIME key, not an AgentState channel, so read it off the
    # terminal mapping generically rather than through the TypedDict key set.
    raw = cast(Mapping[str, Any], final).get("__interrupt__")
    if not raw:
        return None
    first = raw[0] if isinstance(raw, Sequence) else raw
    value = getattr(first, "value", None)
    return value if isinstance(value, Mapping) else None


def _project_plan(final: AgentState) -> Plan:
    """Project a plan run's terminal state into :class:`Plan` (OUTCOME-R2 / CKPT-R5).

    When the run PAUSED for approval (``__interrupt__`` present) the deliverable is
    ``awaiting_approval`` carrying the gate's ``interrupt_id`` + the grounded plan body from the
    interrupt payload; otherwise it projects the finalize state exactly like the answer (the
    grounded body, status, citations).
    """
    fallback_thread = (
        _opt_str(final.get("thread_id")) or _opt_str(final.get("idempotency_key")) or ""
    )
    payload = _interrupt_payload(final)
    if payload is not None:
        body = _opt_str(payload.get("grounded_plan")) or _opt_str(final.get("grounded_text")) or ""
        return Plan(
            status=RunStatus.AWAITING_APPROVAL,
            thread_id=_opt_str(payload.get("thread_id")) or fallback_thread,
            plan_html=safe_plan_html(body),
            plan_text=body,
            interrupt_id=_opt_str(payload.get("interrupt_id")),
        )
    html, text, status, projected_thread = _outputs(final)
    observations = _project_observations(_as_seq(final.get("observations")))
    return Plan(
        status=status,
        thread_id=projected_thread or fallback_thread,
        plan_html=html,
        plan_text=text,
        interrupt_id=None,
        observations=observations,
        citations=_project_citations(_as_seq(final.get("citations"))),
        suggested_followups=_generate_followups(status, observations),
        coverage_caveat=_coverage_caveat(final),
    )


def safe_plan_html(text: str) -> str:
    """Server-side-escaped HTML for an awaiting-approval plan body (AGT-SEC-R2).

    Mirrors the graph's ``safe_html``: the grounded plan TEXT is escaped (so any ``<``/``>``
    cannot become live markup) and wrapped in one paragraph; no raw model HTML is aliased. The
    final body the API renders is sanitized again at the boundary (API-R13).
    """
    return f"<p>{_html.escape(text)}</p>" if text else ""


__all__ = ["Plan", "plan", "safe_plan_html"]
