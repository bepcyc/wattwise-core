"""The compiled-graph adapter + run bounds for the engine, factored off (QUAL-R9 size split).

The focused sibling of :mod:`wattwise_core.agent.engine` that owns the seam between the deployable
:class:`~wattwise_core.agent.engine.GraphAgentEngine` and a compiled LangGraph: the conversation-id
derivation (CKPT-R3), the langgraph superstep/recursion bound that sits above the node-visit
ceiling, the :class:`_CompiledCoachGraph` adapter that drives the graph through the deliverables'
``CoachGraph`` seam AND enforces the resolved entitlement's WALL-CLOCK deadline (AGT-ENT-R4), and
the graceful degraded terminal state a wall-clock breach projects (OUTCOME-R3). Splitting these out
keeps the engine module under the size ceiling while this cohesive graph-driving surface lives in
one place (mirroring :mod:`engine_extras` / :mod:`engine_services`).

Cited requirements: CKPT-R2/-R3, GRAPH-R1, OUTCOME-R2/-R3/-R4, GROUND-R3, AGT-ENT-R1/-R4.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from wattwise_core.agent.checkpoint import SqlAlchemyCheckpointSaver
from wattwise_core.agent.contracts import AgentState, CoverageCaveat, RunStatus
from wattwise_core.agent.deliverables import AgentAnswer, answer_from_state
from wattwise_core.agent.graph import DEFAULT_NODE_VISIT_CEILING
from wattwise_core.agent.graph_state import limitation_text, plan_requires_approval
from wattwise_core.agent.projection import (
    conversation_id_of,
    idempotent_conversation_id,
    new_conversation_id,
    thread_id_for,
)

# The compiled-graph type the deliverables drive through the ``CoachGraph`` seam.
_CompiledGraph = CompiledStateGraph[AgentState, Any, AgentState, AgentState]

# The node-visit ceiling the production graph is compiled with, and the langgraph superstep bound —
# kept TOGETHER so the invariant ``recursion_limit > ceiling`` holds for whatever ceiling is
# configured. The bound sits ABOVE the ceiling so a pathological run finalizes gracefully (degraded,
# OUTCOME-R3) via the graph's own ceiling rather than raising a GraphRecursionError first; the
# bounded reflect/redraft counters guarantee termination well before either bound on every legal
# path.
NODE_VISIT_CEILING = DEFAULT_NODE_VISIT_CEILING
RECURSION_LIMIT = NODE_VISIT_CEILING + 20


def conversation_id_for(athlete_id: str, thread_id: str | None) -> str:
    """The saver-bound conversation id for a run, REVERSIBLE with the thread_id (CKPT-R3).

    A follow-up/decision passes the prior ``thread_id`` back: the conversation id is recovered
    from it (``conversation_id_of``) so the saver binds to the SAME durable thread. A fresh turn
    (no ``thread_id``) mints a new conversation id; the SAME value is passed to the deliverable so
    the thread_id it builds (``{athlete_id}:{conversation_id}``) matches the saver's bound scope —
    otherwise the graph config's thread_id and the saver's thread row would diverge.
    """
    if thread_id is not None:
        return conversation_id_of(thread_id)
    return new_conversation_id()


def conversation_id_for_turn(
    athlete_id: str, thread_id: str | None, request_text: str | None, dedup_window_seconds: int
) -> str:
    """The conversation id for a ``/ask`` turn — DETERMINISTIC for a fresh turn (CKPT-R4).

    A follow-up passes its ``thread_id`` back, so the conversation id is recovered from it (the
    run lands on the SAME durable thread). A FRESH turn (no ``thread_id``) derives the id
    deterministically from the turn (athlete + question + dedup-window bucket) so a re-submitted
    SAME turn within the window collapses onto ONE thread where ``resolve_existing_answer`` finds
    the existing run — never a random id that would spawn a duplicate run (the deviated bug).
    """
    if thread_id is not None:
        return conversation_id_of(thread_id)
    return idempotent_conversation_id(
        athlete_id=athlete_id,
        trigger="user_turn",
        request_text=request_text or "",
        dedup_window_seconds=dedup_window_seconds,
    )


async def resolve_existing_answer(
    saver: SqlAlchemyCheckpointSaver,
    *,
    athlete_id: str,
    conversation_id: str,
    follow_up_thread_id: str | None,
) -> AgentAnswer | None:
    """Return the EXISTING run's projected answer for a deduped FRESH turn, else ``None`` (CKPT-R4).

    A FOLLOW-UP (``follow_up_thread_id`` set) is a NEW turn on an existing thread and is NEVER
    deduped — it returns ``None`` so the caller runs the graph. A FRESH turn is keyed
    deterministically to a durable thread (``idempotent_conversation_id``), so a re-submitted SAME
    turn within the window lands on the SAME ``(athlete_id, conversation_id)``; this resolves that
    thread and, if a checkpoint exists, projects its terminal state (``answer_from_state``) so the
    re-submission RETURNS the existing run rather than a duplicate. Cross-identity ownership is
    refused in ``resolve_idempotent`` (CKPT-R3).
    """
    if follow_up_thread_id is not None:
        return None
    thread_id = thread_id_for(athlete_id, conversation_id)
    if await saver.resolve_idempotent(thread_id) is None:
        return None
    tuple_ = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
    if tuple_ is None:
        return None
    return answer_from_state(cast(AgentState, tuple_.checkpoint.get("channel_values", {})))


def wall_clock_degraded(state: AgentState) -> AgentState:
    """Graceful DEGRADED terminal state for a run past its wall-clock budget (AGT-ENT-R4).

    The fail-closed result of the per-run wall-clock deadline (the entitlement's non-monetary
    wall-clock guard): a terminal :class:`AgentState` the deliverable projection (``_outputs`` /
    ``coverage_caveat``) renders as a graceful degraded answer — NEVER a crash, a partial/ungrounded
    body, or a bubbled exception. ``status`` is :data:`RunStatus.DEGRADED`; the grounded body is
    EMPTY (no partial draft escapes ungrounded, GROUND-R3); the typed coverage caveat is
    source-agnostic ``degraded`` fidelity (OUTCOME-R4) — a jargon-free signal that the run could not
    finish in time, leaking no internal token. The durable ``thread_id`` is preserved so a follow-up
    can resume the SAME conversation. The localized limitation copy is carried as the body for any
    surface that renders the text verbatim (it is the jargon-free "couldn't finish" floor).
    """
    limitation = limitation_text(state)
    caveat = CoverageCaveat(fidelity="degraded").model_dump()
    return AgentState(
        athlete_id=state.get("athlete_id", ""),
        trigger=state.get("trigger", "user_turn"),
        thread_id=state.get("thread_id") or state.get("idempotency_key"),
        status=RunStatus.DEGRADED,
        grounded_text=limitation,
        grounded_html="",
        citations=[],
        coverage_caveat=caveat,
    )


class CompiledCoachGraph:
    """Adapt a compiled LangGraph to the deliverables' :class:`CoachGraph` seam (GRAPH-R1).

    ``deliverables.answer_question`` drives the graph through the typed async ``run(state)``
    seam; a compiled langgraph instead exposes ``ainvoke`` and REQUIRES a per-run config
    carrying the durable ``thread_id`` (the checkpointer key, CKPT-R3) plus a recursion
    bound. This wrapper supplies both from the immutable input state so the production engine
    invokes the graph exactly as the grounded-Q&A deliverable expects — without it the
    deliverable's ``graph.run`` call would not resolve against the bare compiled graph.

    It ALSO enforces the resolved entitlement's WALL-CLOCK deadline (AGT-ENT-R4) at this single
    graph-invoke point: ``run`` wraps ``ainvoke`` in ``asyncio.wait_for`` and, on a timeout, FAILS
    CLOSED GRACEFULLY to a degraded terminal state (:func:`wall_clock_degraded`) — never a bubbled
    :class:`TimeoutError` or a partial answer. The deadline value is the carried entitlement's bound
    (AGT-ENT-R1, config-loaded), threaded in by the engine; ``resume`` is NOT bounded by it (a HITL
    resume is a fresh, short continuation, not the long initial run).

    The wall-clock deadline is DELIBERATELY NOT applied to a PAUSABLE approval-gated PLAN run
    (the only run that durably PAUSES at ``interrupt_gate``). On that path the gate commits a
    ``live`` ``AgentInterrupt`` ledger row (CKPT-R9) just BEFORE langgraph suspends, and the pause
    surfaces back through this same ``run`` as the ``__interrupt__`` terminal. If a wall-clock
    deadline fired in the narrow window after that row commits but before ``ainvoke`` returned the
    pause, ``run`` would return :func:`wall_clock_degraded` (no ``interrupt_id``) while leaving the
    ``live`` row ORPHANED forever — no decision could ever consume it. Wall-clock is the WRONG bound
    for that run anyway: a pause is human think-time, not compute, and the plan BUILD is already
    bounded by the node-visit ceiling + the tool-iteration bound (both degrade gracefully) and the
    superstep ``recursion_limit`` backstop. So the deadline guards the AUTONOMOUS answer/digest
    paths (which never pause) and is skipped for the pausable plan path (mitigation (a) for the
    orphaned-ledger-row race). The plan path is identified from the input state's
    ``plan_deliverable`` marker — the SAME marker ``interrupt_gate`` keys its pause on
    (``plan_requires_approval``).
    """

    def __init__(self, compiled: _CompiledGraph, *, wall_clock_seconds: float | None) -> None:
        self._compiled = compiled
        # A non-positive / absent bound (an isolated caller with a bare all-permissive grant and no
        # config) means NO wall-clock deadline — ``asyncio.wait_for(timeout=None)`` preserves the
        # exact pre-deadline behavior. Production carries a positive config-loaded bound (CFG-R1a),
        # so the deadline is real there; no value is baked into code for the fallback.
        self._wall_clock_seconds = (
            wall_clock_seconds if wall_clock_seconds and wall_clock_seconds > 0 else None
        )

    async def run(self, state: AgentState) -> AgentState:
        """Invoke the compiled graph with the durable-thread config (CKPT-R3, OUTCOME-R2).

        The thread id MUST come from the run's own ``(athlete_id, conversation_id)`` scope
        (CKPT-R3); it fails closed if absent rather than aliasing onto a shared key. The whole-run
        wall-clock deadline (AGT-ENT-R4) bounds the invoke; a breach degrades GRACEFULLY rather than
        raising (the run could not finish in its time budget — OUTCOME-R3, never a crash).

        EXCEPTION (mitigation (a), orphaned-ledger-row race): a PAUSABLE approval-gated PLAN run is
        NOT wall-clock-bounded here. It durably PAUSES at ``interrupt_gate`` (which commits a
        ``live`` ledger row just before suspending), and a deadline firing in that pause window
        would return a degraded answer while orphaning the ``live`` row forever. The plan build is
        bounded by the node-visit + tool-iteration ceilings and the superstep backstop, and a pause
        is human think-time (not compute), so wall-clock is the wrong bound there — see the class
        docstring. The autonomous answer/digest paths (which never pause) keep the deadline.
        """
        thread_id = state.get("thread_id") or state.get("idempotency_key")
        if not thread_id:
            raise ValueError("agent run state carries no durable thread id (CKPT-R3)")
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        # No wall-clock deadline on the pausable approval-gated plan path: the pause window could
        # otherwise orphan the ``live`` interrupt row ``interrupt_gate`` commits before suspending.
        timeout = None if plan_requires_approval(state) else self._wall_clock_seconds
        try:
            result = await asyncio.wait_for(
                self._compiled.ainvoke(state, config=config),
                timeout=timeout,
            )
        except TimeoutError:  # asyncio.TimeoutError is an alias of builtin TimeoutError (3.11+)
            # Fail closed GRACEFULLY (AGT-ENT-R4 wall-clock guard): the run exceeded its time
            # budget, so project a degraded terminal answer instead of bubbling the timeout. The
            # durable checkpoint is untouched (the cancelled ainvoke wrote no terminal state); a
            # follow-up may resume the same thread.
            return wall_clock_degraded(state)
        return cast(AgentState, result)

    async def resume(self, command: Command[Any], config: RunnableConfig) -> AgentState:
        """Resume a paused run with ``Command(resume=...)`` on the SAME durable thread (CKPT-R2).

        The head node does NOT re-run (no recompute, no fresh turn_id); the pre-interrupt nodes
        replay from the checkpoint rather than re-executing. Returns the terminal state.
        """
        result = await self._compiled.ainvoke(command, config=config)
        return cast(AgentState, result)


__all__ = [
    "NODE_VISIT_CEILING",
    "RECURSION_LIMIT",
    "CompiledCoachGraph",
    "conversation_id_for",
    "conversation_id_for_turn",
    "resolve_existing_answer",
    "wall_clock_degraded",
]
