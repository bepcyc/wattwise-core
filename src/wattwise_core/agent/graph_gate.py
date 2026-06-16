"""The approval interrupt-gate node (GRAPH-R2, CKPT-R5/-R9).

Factored out of :mod:`wattwise_core.agent.graph` (QUAL-R9 module-size ceiling) as a focused
leaf: the single human-in-the-loop approval checkpoint between `ground` and `finalize`.
``graph`` imports :func:`make_interrupt_gate` and wires it as the ``interrupt_gate`` node.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from langgraph.types import interrupt

from wattwise_core.agent import graph_state as gs
from wattwise_core.agent.contracts import AgentState, GroundDecision, RunStatus
from wattwise_core.agent.seams import GraphNode


def make_interrupt_gate(recorder: gs.InterruptRecorder | None) -> GraphNode:
    """Build the approval checkpoint node (GRAPH-R2, CKPT-R5/-R9)."""

    async def interrupt_gate(state: AgentState) -> dict[str, Any]:
        """Approval checkpoint between grounding and finalisation (GRAPH-R2, CKPT-R5/-R9).

        Pauses ONLY for an approval-gated PLAN the grounder STANDS BEHIND (a ``PROCEED`` decision):
        it persists a ``live`` ``AgentInterrupt`` ledger row (via the injected ``recorder`` = the
        durable checkpointer; CKPT-R9) BEFORE suspending, so a decision arriving against this thread
        always finds a live row to atomically CONSUME and can never resume twice, then PAUSES at a
        durable langgraph ``interrupt`` carrying ``{grounded_plan, thread_id, interrupt_id}`` and
        emits ``awaiting_approval`` HERE (it does not reach ``finalize``). ``recorder is None`` (an
        in-memory checkpointer, the OSS/test default) raises the interrupt but records no row.

        DECISION-AWARE (issue #25): a NON-``PROCEED`` plan run is NEVER put to a human decision —
        ``ground`` writes ``grounded_text`` on every pass, so pausing on an ABSTAIN/REGENERATE body
        would ask the athlete to approve a plan the grounder ruled unpublishable (or whose
        prescriptions were scrubbed). Such a run falls through to ``finalize`` and degrades like
        every other deliverable; with no approval-gated plan the gate is a pass-through.
        """
        gs.athlete_id(state)
        if not gs.plan_requires_approval(state):
            return gs.tick_visit(state, {})
        if gs.last_ground_decision(state) is not GroundDecision.PROCEED:
            # The grounder does not stand behind this body — degrade at finalize, never solicit
            # a human approval on a non-PROCEED plan (issue #25).
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


__all__ = ["make_interrupt_gate"]
