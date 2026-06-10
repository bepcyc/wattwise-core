"""Leaf graph-driving + terminal-state projection primitives shared by the deliverables.

This is a LEAF of the deliverables family (ARCH-R21 / QUAL-R9): it owns the primitives that
EVERY graph-driven deliverable (the free-form answer + weekly digest in
:mod:`wattwise_core.agent.deliverables`, the multi-day plan in
:mod:`wattwise_core.agent.plan_deliverable`) shares — building the write-once immutable graph
INPUTS for a run (the reversible durable ``thread_id`` + the fresh per-turn ``turn_id``,
CKPT-R3/-R5) and PROJECTING a terminal :class:`~wattwise_core.agent.contracts.AgentState` into
the typed body/status/observations/citations the deliverables surface (OUTCOME-R2).

It imports only the contracts and the LEAF :mod:`wattwise_core.agent.voice` primitives, so it
sits strictly BELOW the deliverable modules in the import graph (no cycle). Hoisting these
shared primitives here — rather than into one deliverable that the other imports back — is what
keeps the deliverable family acyclic: each deliverable depends DOWNWARD on this leaf, and the
deliverable modules re-export the public names so historical import paths stay stable.

Cited requirements: STATE-R2/-R4, CKPT-R3/-R4/-R5, GRAPH-R2.1, OUTCOME-R2/-R3/-R4/-R5,
GROUND-R5, AGT-SEC-R1.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from wattwise_core.agent.contracts import AgentState, RunStatus, Trigger
from wattwise_core.agent.voice import (
    Citation,
    Observation,
    ResponseLength,
    _opt_str,
    _project_citations,
)


@runtime_checkable
class CoachGraph(Protocol):
    """The agent-graph seam the deliverables drive (GRAPH-R1, doc 50 §3).

    The concrete stateful graph is an in-flight sibling; the deliverables reach it ONLY through
    this typed seam so they import no sibling graph file (ARCH-R21). The graph runs the full
    ``ingest_request -> ... -> finalize`` topology and returns the terminal :class:`AgentState`
    with its grounded outputs filled. Identity/scope are the graph's structural concern
    (AGT-SEC-R1); this seam never widens them. Defined in this LEAF module so both
    :mod:`wattwise_core.agent.deliverables` and :mod:`wattwise_core.agent.plan_deliverable`
    depend DOWNWARD on it (no cycle).
    """

    async def run(self, state: AgentState) -> AgentState: ...

# Separator between the athlete scope and the conversation id INSIDE a durable thread_id.
# A thread_id is ``{athlete_id}:{conversation_id}`` (CKPT-R3) and MUST be REVERSIBLE so a
# follow-up turn (or a HITL decision) resumes the SAME durable thread: the engine derives the
# saver's bound ``conversation_id`` back from the path ``thread_id`` by splitting on the FIRST
# separator only (a conversation_id may itself contain the separator). athlete ids are
# server-derived UUID strings that never contain it, so the split is unambiguous.
_THREAD_SEP = ":"


def new_conversation_id() -> str:
    """Mint a fresh, stable conversation id for a NEW durable thread (CKPT-R3).

    A new ``/ask`` turn with no existing ``thread_id`` opens a new conversation; its id is a
    random uuid (NOT derived from ``trigger:request_text``) so a later follow-up — which carries
    a DIFFERENT request body — resumes the SAME thread by passing the thread_id back, rather than
    computing a divergent thread and starting a duplicate run (the bug at the old
    ``convo = trigger:request_text`` derivation).
    """
    return uuid.uuid4().hex


def thread_id_for(athlete_id: str, conversation_id: str) -> str:
    """The reversible durable thread id for ``(athlete_id, conversation_id)`` (CKPT-R3)."""
    return f"{athlete_id}{_THREAD_SEP}{conversation_id}"


def conversation_id_of(thread_id: str) -> str:
    """Recover the conversation id from a durable thread id (inverse of :func:`thread_id_for`).

    Splits on the FIRST separator only so a conversation_id that itself contains the separator
    round-trips. A thread_id with no separator (a legacy/opaque id) yields itself, so the saver
    still binds to a stable conversation scope rather than failing closed.
    """
    _, sep, convo = thread_id.partition(_THREAD_SEP)
    return convo if sep else thread_id


def build_inputs(
    *,
    athlete_id: str,
    trigger: Trigger,
    locale: str,
    request_text: str | None,
    response_length: ResponseLength = "standard",
    conversation_id: str | None = None,
    thread_id: str | None = None,
    follow_up: Mapping[str, Any] | None = None,
    active_goals: Sequence[Mapping[str, Any]] | None = None,
    recalled_memory: Sequence[Mapping[str, Any]] | None = None,
) -> AgentState:
    """Assemble the write-once immutable graph inputs for a run (STATE-R2/-R4, CKPT-R5).

    ``athlete_id`` flows straight from the authenticated caller (AGT-SEC-R1) and is NEVER taken
    from any model/tool output. ``request_text`` is present iff the trigger is ``user_turn`` (the
    STATE-R2 discriminated union); a scheduled digest carries none, and its intent is fixed by
    the trigger (GRAPH-R2.1) with no intent model call.

    The durable ``thread_id`` is the stable ``(athlete_id, conversation_id)`` identifier the
    checkpointer keys on (CKPT-R3) and is REVERSIBLE (see :func:`conversation_id_of`): when the
    caller passes an existing ``thread_id`` (a follow-up resuming a conversation) it is used
    VERBATIM so the run lands on the SAME durable thread; otherwise a fresh thread is opened from
    ``conversation_id`` (a digest's deterministic id, or a new random one for a free-form turn).
    ``idempotency_key`` is the per-turn dedup key (CKPT-R4), carried separately.

    ``turn_id`` is minted FRESH for every ``/ask`` ``ainvoke`` (CKPT-R5): on a durable thread
    reused across turns it is the per-turn discriminator that makes ``ingest_request`` reset the
    run-scoped channels (counters/retrieved/coverage_gaps) so turn-N never leaks into turn-N+1.
    ``follow_up`` is carried so the projection can shape an expand/drill/reveal-numbers turn
    (COACH-R8). ``response_length`` governs verbosity (VOICE-R8), never truth.

    ``active_goals`` are the athlete's ACTIVE canonical goals, read SERVER-side by the caller (the
    engine) from the GBO store and projected into the immutable inputs so the agent plans TOWARD
    them (GBO-R38 / API-R32 / API-R35): goal-aware planning/load-review is owned by the agent, which
    reads the canonical Goal entity. They are user-authored INTENT, never an analytic number
    (MEM-R1) — they steer the compose prompt context, not grounding. Carried only when present.

    ``recalled_memory`` are durable athlete-memory items recalled SERVER-side by the caller (the
    engine) through the ONE MemoryStore/recall seam (MEM-R4) and projected into the immutable inputs
    so the agent personalizes its answer (stated goals/constraints/preferences in the athlete's own
    words, MEM-R1/-R2). Like ``active_goals`` they are personalization context, never an analytic
    number (MEM-R1) — they steer the compose prompt context, not grounding. Carried when present.
    """
    if thread_id is None:
        convo = conversation_id or new_conversation_id()
        thread_id = thread_id_for(athlete_id, convo)
    state: AgentState = {
        "athlete_id": athlete_id,
        "trigger": trigger,
        "request_text": request_text,
        "locale": locale,
        "thread_id": thread_id,
        "idempotency_key": thread_id,
        "response_length": response_length,
        "turn_id": uuid.uuid4().hex,
    }
    if follow_up is not None:
        state["messages"] = [{"role": "user", "kind": "follow_up", **dict(follow_up)}]
    if active_goals:
        state["active_goals"] = [dict(goal) for goal in active_goals]
    if recalled_memory:
        state["recalled_memory"] = [dict(item) for item in recalled_memory]
    return state


def as_seq(raw: Any) -> Sequence[Mapping[str, Any]]:
    """Narrow an optional graph output list to a sequence of mappings, else empty."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def project_observations(raw: Sequence[Mapping[str, Any]]) -> tuple[Observation, ...]:
    """Project graph observations into stable-id :class:`Observation`s (COACH-R8).

    Drops any observation lacking a stable id or text (a follow-up could not target it); its
    citations are projected and id-filtered like top-level citations.
    """
    observations: list[Observation] = []
    for item in raw:
        obs_id = _opt_str(item.get("observation_id"))
        text = _opt_str(item.get("text"))
        if not obs_id or not text:
            continue
        raw_cites = item.get("citations", ())
        cites = _project_citations(raw_cites) if isinstance(raw_cites, Sequence) else ()
        observations.append(Observation(observation_id=obs_id, text=text, citations=cites))
    return tuple(observations)


def generate_followups(
    status: RunStatus, observations: Sequence[Observation]
) -> tuple[str, ...]:
    """Generate small jargon-free follow-up prompts the engine owns (COACH-R8, VOICE-R9).

    The engine GENERATES this copy (OSS); a thin client only renders it. Prompts are
    presentation over the existing grounded thread (introduce no new claim/number) and must stay
    athlete-native + jargon-free (VOICE-R2): a ``reveal_numbers`` handle is offered only when
    there are grounded observations to reveal numbers behind; the other is the canonical
    ``expand`` prompt. A degraded run offers neither a numbers-reveal it cannot honor nor an
    internals leak — just the expand prompt.
    """
    reveal = "Show me the numbers behind that"
    expand = "Tell me more"
    if status is RunStatus.DEGRADED or not observations:
        return (expand,)
    return (reveal, expand)


def outputs(final: AgentState) -> tuple[str, str, RunStatus, str]:
    """Read the grounded body/status/thread off a terminal state (OUTCOME-R2).

    Returns ``(html, text, status, thread_id)``. Falls back text->html so a deliverable always
    has both bodies for the API to sanitize; status defaults to ``degraded`` if a graph somehow
    omitted it, since a missing terminal status is a reduced-confidence outcome, never a
    fabricated ``completed`` (OUTCOME-R3/-R5, fail-closed).
    """
    html = _opt_str(final.get("grounded_html")) or ""
    text = _opt_str(final.get("grounded_text")) or ""
    html = html or text
    text = text or html
    status = final.get("status")
    status = status if isinstance(status, RunStatus) else RunStatus.DEGRADED
    # The durable thread_id is the (athlete_id, conversation_id)-scoped checkpointer id
    # (CKPT-R3/OUTCOME-R2), NOT the per-turn idempotency key (CKPT-R4).
    thread_id = _opt_str(final.get("thread_id")) or _opt_str(final.get("idempotency_key")) or ""
    return html, text, status, thread_id


def coverage_caveat(final: AgentState) -> Mapping[str, Any] | None:
    """Return the typed coverage caveat for a degraded outcome (OUTCOME-R4), else None."""
    caveat = final.get("coverage_caveat")
    return caveat if isinstance(caveat, Mapping) else None


__all__ = [
    "Citation",
    "CoachGraph",
    "Observation",
    "ResponseLength",
    "as_seq",
    "build_inputs",
    "conversation_id_of",
    "coverage_caveat",
    "generate_followups",
    "new_conversation_id",
    "outputs",
    "project_observations",
    "thread_id_for",
]
