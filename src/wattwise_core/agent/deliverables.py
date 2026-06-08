"""Phase-1 coach deliverables: grounded Q&A + the weekly digest (doc 50).

This module is the thin, typed PROJECTION layer between the agent graph (the
in-flight a5 sibling, reached only through the :class:`CoachGraph` seam below) and
the athlete-facing deliverable contracts the API renders. It owns the two Phase-1
deliverables and NOTHING else: a free-form grounded answer
(:func:`answer_question`) and the weekly digest, which IS the weekly load review
(COACH-R1 #1) — one deliverable, one name (:func:`weekly_digest`). Readiness, the
multi-day plan, insight, and briefing are specced (COACH-R1 #2-#5) but are a LATER
phase and are deliberately absent here.

Each function DRIVES the graph with the right immutable trigger (GRAPH-R2.1):
``answer_question`` uses ``user_turn`` carrying the question text;
``weekly_digest`` uses ``scheduled_digest`` with no request text. It then projects
ONLY the graph's grounded outputs (OUTCOME-R2: never un-grounded model text) into a
typed dataclass carrying the status-discriminated outcome, the sanitized-later
HTML/text body, the stable-id observations (COACH-R8), the surviving grounded
citations (GROUND-R5), and small jargon-free follow-up prompts.

The voice contract is enforced as a PRESENTATION layer over the graph's
fail-closed grounding, never a relaxation of it (VOICE-R7): the deliverable LEADS
with a plain-language state observation (COACH-R7) and foregrounds at most the
configured number-density cap of explicit numbers (VOICE-R7/-R8), with every
surfaced number already grounded and cited by the graph. This module rewrites no
number and certifies no groundedness — it projects what the graph grounded and
runs the DETERMINISTIC leads-with-state / number-count checks that are the gate
for those two presentation properties (EVAL-R5b.1).

Cited requirements: COACH-R1, COACH-R5, COACH-R8, OUTCOME-R1/-R2/-R3/-R4,
GRAPH-R2.1, STATE-R2, GROUND-R5/-R7, VOICE-R1/-R2/-R7/-R8, LANG-R2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from wattwise_core.agent.contracts import (
    AgentState,
    RunStatus,
    Trigger,
)

# The shared voice/projection primitives live in the LEAF :mod:`voice` module so BOTH this
# module and :mod:`readiness_deliverable` depend DOWNWARD on them (no cycle). They are
# imported here and RE-EXPORTED (see ``__all__`` + the name list below) so every historical
# ``from wattwise_core.agent.deliverables import Citation/Observation/leads_with_state/...``
# path keeps resolving unchanged (ARCH-R21 / QUAL-R9).
from wattwise_core.agent.voice import (
    Citation,
    Observation,
    ResponseLength,
    _enforce_number_cap,
    _opt_str,
    _project_citations,
    count_foregrounded_numbers,
    first_sentence,
    leads_with_state,
    number_cap,
)


@runtime_checkable
class CoachGraph(Protocol):
    """The agent-graph seam these deliverables drive (GRAPH-R1, doc 50 §3).

    The concrete stateful graph is an in-flight sibling; this module reaches it ONLY
    through this typed seam so it imports no sibling graph file (ARCH-R21). The graph
    runs the full ``ingest_request -> ... -> finalize`` topology and returns the
    terminal :class:`AgentState` with its grounded outputs filled. Identity/scope are
    the graph's structural concern (AGT-SEC-R1); this seam never widens them.
    """

    async def run(self, state: AgentState) -> AgentState: ...


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """A grounded free-form answer projected from a ``user_turn`` run (COACH-R1).

    Carries the status-discriminated outcome (OUTCOME-R1), the grounded body as both
    HTML and text (sanitized later by the API, AGT-SEC-R2), the stable-id observations
    (COACH-R8), the surviving citations (GROUND-R5), and small jargon-free follow-up
    prompts (VOICE-R2/-R9). ``coverage_caveat`` is the typed missing/stale-input note
    for a ``degraded`` outcome (OUTCOME-R4); the API renders it in coach voice.
    """

    status: RunStatus
    thread_id: str
    answer_html: str
    answer_text: str
    observations: tuple[Observation, ...] = ()
    citations: tuple[Citation, ...] = ()
    suggested_followups: tuple[str, ...] = ()
    coverage_caveat: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Digest:
    """The weekly digest == weekly load review (COACH-R1 #1), one deliverable.

    A grounded trailing-week review: completed sessions (with canonical activity
    citations), weekly load vs the prior trend (CTL/ATL/ramp from canonical PMC), and
    flags — projected from a ``scheduled_digest`` run. Same fields/guarantees as
    :class:`AgentAnswer`; it simply LEADS with a state phrase (COACH-R7) and abstains
    visibly (``degraded`` + caveat) when the week's canonical inputs are missing
    rather than guessing (OUTCOME-R3/-R4, GROUND-R7).
    """

    status: RunStatus
    thread_id: str
    week_end: str
    digest_html: str
    digest_text: str
    observations: tuple[Observation, ...] = ()
    citations: tuple[Citation, ...] = ()
    suggested_followups: tuple[str, ...] = ()
    coverage_caveat: Mapping[str, Any] | None = None


# --- graph driving + projection ---


def _build_inputs(
    *,
    athlete_id: str,
    trigger: Trigger,
    locale: str,
    request_text: str | None,
    response_length: ResponseLength = "standard",
    conversation_id: str | None = None,
) -> AgentState:
    """Assemble the write-once immutable graph inputs for a run (STATE-R2/-R4).

    ``athlete_id`` flows straight from the authenticated caller (AGT-SEC-R1) and is
    NEVER taken from any model/tool output. ``request_text`` is present iff the trigger
    is ``user_turn`` (the STATE-R2 discriminated union); a scheduled digest carries
    none, and its intent is fixed by the trigger (GRAPH-R2.1) with no intent model call.
    The durable ``thread_id`` is the stable ``(athlete_id, conversation_id)`` identifier
    the checkpointer keys on (CKPT-R3); ``idempotency_key`` is the per-turn dedup key
    (CKPT-R4) — a distinct concept carried separately so the outcome can address both.
    ``response_length`` governs verbosity/number-foregrounding (VOICE-R8), never truth.
    """
    convo = conversation_id or f"{trigger}:{request_text or ''}"
    thread_id = f"{athlete_id}:{convo}"
    state: AgentState = {
        "athlete_id": athlete_id,
        "trigger": trigger,
        "request_text": request_text,
        "locale": locale,
        "thread_id": thread_id,
        "idempotency_key": thread_id,
        "response_length": response_length,
    }
    return state


def _project_observations(
    raw: Sequence[Mapping[str, Any]],
) -> tuple[Observation, ...]:
    """Project graph observations into stable-id :class:`Observation`s (COACH-R8).

    Drops any observation lacking a stable id or text (a follow-up could not target
    it); its citations are projected and id-filtered like top-level citations.
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


def _generate_followups(
    status: RunStatus, observations: Sequence[Observation]
) -> tuple[str, ...]:
    """Generate small jargon-free follow-up prompts the engine owns (COACH-R8, VOICE-R9).

    The engine GENERATES this copy (OSS); a thin client only renders it. Prompts are
    presentation over the existing grounded thread (introduce no new claim/number) and
    must stay athlete-native + jargon-free (VOICE-R2): a ``reveal_numbers`` handle is
    offered only when there are grounded observations to reveal numbers behind; the
    other is the canonical ``expand`` prompt. A degraded run offers neither a
    numbers-reveal it cannot honor nor an internals leak — just the expand prompt.
    """
    reveal = "Show me the numbers behind that"
    expand = "Tell me more"
    if status is RunStatus.DEGRADED or not observations:
        return (expand,)
    return (reveal, expand)


def _outputs(final: AgentState) -> tuple[str, str, RunStatus, str]:
    """Read the grounded body/status/thread off a terminal state (OUTCOME-R2).

    Returns ``(html, text, status, thread_id)``. Falls back text->html so a deliverable
    always has both bodies for the API to sanitize; status defaults to ``degraded`` if a
    graph somehow omitted it, since a missing terminal status is a reduced-confidence
    outcome, never a fabricated ``completed`` (OUTCOME-R3/-R5, fail-closed).
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


def _coverage_caveat(final: AgentState) -> Mapping[str, Any] | None:
    """Return the typed coverage caveat for a degraded outcome (OUTCOME-R4), else None."""
    caveat = final.get("coverage_caveat")
    return caveat if isinstance(caveat, Mapping) else None


async def answer_question(
    graph: CoachGraph,
    athlete_id: str,
    question: str,
    *,
    locale: str,
    response_length: ResponseLength = "standard",
) -> AgentAnswer:
    """Drive the graph for a grounded free-form answer to ``question`` (COACH-R1).

    Builds a ``user_turn`` run carrying the question (GRAPH-R2.1), runs the graph, and
    projects ONLY its grounded outputs into :class:`AgentAnswer` (OUTCOME-R2). The
    athlete identity is server-derived (AGT-SEC-R1) and never trusted from the model;
    ``response_length`` governs verbosity/number-foregrounding in the graph's compose,
    never truth (VOICE-R8). No un-grounded text is ever surfaced.
    """
    inputs = _build_inputs(
        athlete_id=athlete_id,
        trigger="user_turn",
        locale=locale,
        request_text=question,
        response_length=response_length,
    )
    final = await graph.run(inputs)
    html, text, status, thread_id = _outputs(final)
    cap = number_cap(response_length)
    html, text = _enforce_number_cap(html, text, cap)
    observations = _project_observations(_as_seq(final.get("observations")))
    return AgentAnswer(
        status=status,
        thread_id=thread_id,
        answer_html=html,
        answer_text=text,
        observations=observations,
        citations=_project_citations(_as_seq(final.get("citations"))),
        suggested_followups=_generate_followups(status, observations),
        coverage_caveat=_coverage_caveat(final),
    )


async def weekly_digest(graph: CoachGraph, athlete_id: str, week_end: str) -> Digest:
    """Drive the graph for the weekly digest (== weekly load review, COACH-R1 #1).

    Builds a ``scheduled_digest`` run — intent fixed by the trigger, no request text and
    no intent model call (GRAPH-R2.1) — runs the graph, and projects its grounded
    trailing-week review into :class:`Digest`. The digest LEADS with a state phrase
    (COACH-R7) and, when the week's canonical inputs are missing, ships ``degraded`` with
    a truthful caveat rather than guessing (OUTCOME-R3/-R4, GROUND-R7). Locale resolves
    to the configured default for an unattended run (``en``, LANG-R4); the graph applies
    the athlete's persisted preference where present (LANG-R2).
    """
    inputs = _build_inputs(
        athlete_id=athlete_id,
        trigger="scheduled_digest",
        locale="en",
        request_text=None,
        conversation_id=f"digest:{week_end}",
    )
    final = await graph.run(inputs)
    html, text, status, thread_id = _outputs(final)
    html, text = _enforce_number_cap(html, text, number_cap("standard"))
    observations = _project_observations(_as_seq(final.get("observations")))
    return Digest(
        status=status,
        thread_id=thread_id,
        week_end=week_end,
        digest_html=html,
        digest_text=text,
        observations=observations,
        citations=_project_citations(_as_seq(final.get("citations"))),
        suggested_followups=_generate_followups(status, observations),
        coverage_caveat=_coverage_caveat(final),
    )


def _as_seq(raw: Any) -> Sequence[Mapping[str, Any]]:
    """Narrow an optional graph output list to a sequence of mappings, else empty."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


# Re-export the readiness/form deliverable, which lives in the focused sibling module
# :mod:`readiness_deliverable` (QUAL-R9 size split). This import is now strictly ONE-WAY:
# ``readiness_deliverable`` imports its shared voice/projection primitives from the LEAF
# :mod:`voice` module (NOT from here), so there is no longer a ``deliverables`` <->
# ``readiness_deliverable`` cycle — the former load-order-dependent bottom binding is gone.
# The import is kept at the BOTTOM only as a tidy convention (the names re-exported below all
# belong to the readiness sibling); ``readiness_deliverable`` is independently importable as a
# standalone first import. Every public path stays stable — ``Readiness`` /
# ``readiness_assessment`` etc. remain importable from ``wattwise_core.agent.deliverables``.
# ``_ReadinessNarration`` is re-exported by NAME (intentionally NOT in ``__all__``) so the
# historical ``from wattwise_core.agent.deliverables import _ReadinessNarration`` path the
# eval/integration tests use still resolves after the readiness split — hence its F401.
from wattwise_core.agent.readiness_deliverable import (  # noqa: E402
    HRV_UNAVAILABLE_CLAUSE,  # noqa: F401  re-exported by name; not in __all__
    Readiness,
    ReadinessGrounder,
    StructuredNarrationError,
    StructuredNarrator,
    _ReadinessNarration,  # noqa: F401  re-exported by name; not in __all__
    readiness_assessment,
)

__all__ = [
    "AgentAnswer",
    "Citation",
    "CoachGraph",
    "Digest",
    "Observation",
    "Readiness",
    "ReadinessGrounder",
    "ResponseLength",
    "StructuredNarrationError",
    "StructuredNarrator",
    "answer_question",
    "count_foregrounded_numbers",
    "first_sentence",
    "leads_with_state",
    "number_cap",
    "readiness_assessment",
    "weekly_digest",
]
