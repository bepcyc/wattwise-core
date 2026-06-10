"""Coach deliverables: grounded Q&A (+ follow-ups) + the weekly digest (doc 50).

This module is the thin, typed PROJECTION layer between the agent graph (reached only
through the :class:`~wattwise_core.agent.projection.CoachGraph` seam) and the
athlete-facing deliverable contracts the API renders. It owns the free-form grounded
answer (:func:`answer_question`) and the weekly digest, which IS the weekly load review
(COACH-R1 #1) — one deliverable, one name (:func:`weekly_digest`). The readiness/form
deliverable and the multi-day PLAN deliverable live in focused siblings
(:mod:`readiness_deliverable`, :mod:`plan_deliverable`) and are RE-EXPORTED here so every
historical ``from ...deliverables import ...`` path stays stable (QUAL-R9 size split). The
shared graph-driving + projection primitives live in the LEAF :mod:`projection` module.

Each function DRIVES the graph with the right immutable trigger (GRAPH-R2.1):
``answer_question`` uses ``user_turn`` carrying the question text (and, on a COACH-R8
follow-up, the SAME durable thread so an expand/drill/reveal turn continues the
conversation); ``weekly_digest`` uses ``scheduled_digest`` with no request text. It then
projects ONLY the graph's grounded outputs (OUTCOME-R2: never un-grounded model text) into
a typed dataclass carrying the status-discriminated outcome, the sanitized-later HTML/text
body, the stable-id observations (COACH-R8), the surviving grounded citations (GROUND-R5),
and small jargon-free follow-up prompts.

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
from typing import Any

from wattwise_core.agent.contracts import RunStatus

# The shared graph-driving + terminal-state projection primitives live in the LEAF
# :mod:`projection` module so BOTH this module and :mod:`plan_deliverable` depend DOWNWARD on
# them (no cycle): the agent-graph seam (:class:`CoachGraph`), building the run inputs (reversible
# thread_id + per-turn turn_id) and projecting a terminal state into the typed
# body/observations/citations (OUTCOME-R2).
from wattwise_core.agent.projection import (
    CoachGraph,
    conversation_id_of,
    new_conversation_id,
    thread_id_for,
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

# The shared voice/projection primitives live in the LEAF :mod:`voice` module so BOTH this
# module and :mod:`readiness_deliverable` depend DOWNWARD on them (no cycle). They are
# imported here and RE-EXPORTED (see ``__all__`` + the name list below) so every historical
# ``from wattwise_core.agent.deliverables import Citation/Observation/leads_with_state/...``
# path keeps resolving unchanged (ARCH-R21 / QUAL-R9).
from wattwise_core.agent.voice import (
    Citation,
    Observation,
    ResponseLength,
    VoicePresentation,
    _project_citations,
    count_foregrounded_numbers,
    enforce_presentation,
    first_sentence,
    leads_with_state,
    number_cap,
)

# The default presentation policy when a caller injects none (the FakeGraph test seam and the
# weekly digest's unattended run): an empty config-loaded map — a surviving internal token is
# still SCRUBBED to a neutral phrase, never shown as a code (fail-closed VOICE-R2). The engine
# wires the loaded ``[agent.metric_aliases]``-derived policy in for every real answer path.
_DEFAULT_PRESENTATION = VoicePresentation()


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


# --- COACH-R8 follow-up kinds (expand / drill / reveal_numbers) over the same thread ---

# The verbosity ladder a COACH-R8 ``expand`` follow-up climbs one rung up (VOICE-R8).
_LENGTH_LADDER: tuple[ResponseLength, ...] = ("short", "standard", "detailed")


def _expanded_length(current: ResponseLength) -> ResponseLength:
    """The next length up for an ``expand`` follow-up; saturates at ``detailed`` (COACH-R8)."""
    idx = _LENGTH_LADDER.index(current) if current in _LENGTH_LADDER else 1
    return _LENGTH_LADDER[min(idx + 1, len(_LENGTH_LADDER) - 1)]


def _follow_up_kind(follow_up: Mapping[str, Any] | None) -> str | None:
    """The follow-up kind (``expand``/``drill``/``reveal_numbers``) or ``None`` (COACH-R8)."""
    if follow_up is None:
        return None
    kind = follow_up.get("kind")
    return str(kind) if kind is not None else None


def _follow_up_target(follow_up: Mapping[str, Any] | None) -> str | None:
    """The stable observation id a ``drill``/``reveal_numbers`` follow-up targets (COACH-R8)."""
    if follow_up is None:
        return None
    ref = follow_up.get("target_ref")
    return str(ref) if ref is not None else None


def _reveal_observation(
    observations: Sequence[Observation], target_ref: str | None
) -> tuple[Observation, ...]:
    """The observation(s) a ``drill``/``reveal_numbers`` follow-up reveals VERBATIM (COACH-R8).

    A ``drill``/``reveal_numbers`` follow-up surfaces the ALREADY-grounded canonical numbers
    behind a prior observation — the verbatim ``{metric, value, as_of}`` citations the graph
    grounded on the thread, never a new claim (VOICE-R9 / GROUND-R7). When a ``target_ref`` is
    given the matching observation is returned (its grounded citations are the reveal); with no
    target every observation carrying grounded numbers is revealed. The numbers are read
    VERBATIM off the prior grounded state — this layer recomputes nothing.
    """
    if target_ref is not None:
        return tuple(o for o in observations if o.observation_id == target_ref)
    return tuple(o for o in observations if o.citations)


async def answer_question(
    graph: CoachGraph,
    athlete_id: str,
    question: str,
    *,
    locale: str,
    response_length: ResponseLength = "standard",
    thread_id: str | None = None,
    conversation_id: str | None = None,
    follow_up: Mapping[str, Any] | None = None,
    presentation: VoicePresentation | None = None,
    recalled_memory: Sequence[Mapping[str, Any]] | None = None,
) -> AgentAnswer:
    """Drive the graph for a grounded free-form answer to ``question`` (COACH-R1, COACH-R8).

    Builds a ``user_turn`` run carrying the question (GRAPH-R2.1), runs the graph, and
    projects ONLY its grounded outputs into :class:`AgentAnswer` (OUTCOME-R2). The
    athlete identity is server-derived (AGT-SEC-R1) and never trusted from the model.

    ``recalled_memory`` are durable athlete-memory items the engine recalled SERVER-side through the
    MemoryStore seam (MEM-R4); they flow into the run inputs so the agent personalizes its answer
    (MEM-R1) — personalization context only, never a canonical number (the §7 grounder still reads
    every number live).

    A ``follow_up`` (COACH-R8) reuses the SAME durable thread (the caller passes its
    ``thread_id`` back, CKPT-R3) and shapes the turn by kind: ``expand`` climbs one rung up the
    verbosity ladder (VOICE-R8) so the next answer says more; ``drill``/``reveal_numbers``
    reveal the ALREADY-grounded canonical ``{metric, value, as_of}`` numbers behind the
    targeted observation VERBATIM (VOICE-R9 / GROUND-R7) — a higher number cap lets those
    foregrounded numbers through, and the targeted observation's grounded citations are surfaced
    on the answer. ``response_length`` governs verbosity, never truth (VOICE-R8). No un-grounded
    text is ever surfaced.

    AFTER grounding, the athlete-facing prose passes the deterministic PRESENTATION gate
    (``presentation``, the config-loaded :class:`VoicePresentation`): raw internal metric tokens
    are translated to athlete-native language (VOICE-R2), a metrics-report lead is repaired to a
    state read (COACH-R7), and the foregrounded-number count is held to the per-length cap
    (VOICE-R7). This is presentation ONLY — it rewrites no grounded number and changes no
    citation; the grounded ``{metric, value, as_of}`` numbers stay available as the on-demand
    reveal-numbers backing (GROUND-R5/-R7).
    """
    policy = presentation if presentation is not None else _DEFAULT_PRESENTATION
    kind = _follow_up_kind(follow_up)
    if kind == "expand":
        response_length = _expanded_length(response_length)
    inputs = _build_inputs(
        athlete_id=athlete_id,
        trigger="user_turn",
        locale=locale,
        request_text=question,
        response_length=response_length,
        thread_id=thread_id,
        conversation_id=conversation_id,
        follow_up=follow_up,
        recalled_memory=recalled_memory,
    )
    final = await graph.run(inputs)
    html, text, status, out_thread_id = _outputs(final)
    observations = _project_observations(_as_seq(final.get("observations")))
    citations = _project_citations(_as_seq(final.get("citations")))
    # A reveal/drill follow-up foregrounds the verbatim grounded numbers it was asked to reveal
    # (so the cap is lifted to the detailed ceiling); a normal turn holds the per-length cap.
    # EITHER WAY the prose is still scrubbed of raw internal tokens and led with a state read
    # (VOICE-R2/COACH-R7) — a reveal surfaces grounded NUMBERS, never the internal metric CODES.
    # Every number shown is one the graph already grounded; the pass rewrites none of them.
    if kind in ("drill", "reveal_numbers"):
        revealed = _reveal_observation(observations, _follow_up_target(follow_up))
        citations = _merge_revealed_citations(citations, revealed)
        html, text = enforce_presentation(
            html, text, response_length="detailed", presentation=policy
        )
    else:
        html, text = enforce_presentation(
            html, text, response_length=response_length, presentation=policy
        )
    return AgentAnswer(
        status=status,
        thread_id=out_thread_id,
        answer_html=html,
        answer_text=text,
        observations=observations,
        citations=citations,
        suggested_followups=_generate_followups(status, observations),
        coverage_caveat=_coverage_caveat(final),
    )


def _merge_revealed_citations(
    citations: tuple[Citation, ...], revealed: Sequence[Observation]
) -> tuple[Citation, ...]:
    """Append a revealed observation's grounded citations, de-duplicated by record id (COACH-R8).

    The reveal surfaces the verbatim canonical numbers behind the targeted observation; they
    join the answer's own grounded citations without duplicating a record already cited (the
    numbers are read off the prior grounding, never recomputed, GROUND-R7).
    """
    seen = {c.record_id for c in citations}
    extra = [
        c
        for obs in revealed
        for c in obs.citations
        if c.record_id and c.record_id not in seen
    ]
    return (*citations, *extra)


async def weekly_digest(
    graph: CoachGraph,
    athlete_id: str,
    week_end: str,
    *,
    presentation: VoicePresentation | None = None,
    active_goals: Sequence[Mapping[str, Any]] | None = None,
) -> Digest:
    """Drive the graph for the weekly digest (== weekly load review, COACH-R1 #1).

    Builds a ``scheduled_digest`` run — intent fixed by the trigger, no request text and
    no intent model call (GRAPH-R2.1) — runs the graph, and projects its grounded
    trailing-week review into :class:`Digest`. The digest LEADS with a state phrase
    (COACH-R7) and, when the week's canonical inputs are missing, ships ``degraded`` with
    a truthful caveat rather than guessing (OUTCOME-R3/-R4, GROUND-R7). ``active_goals`` are the
    athlete's ACTIVE canonical goals the engine read server-side; they flow into the run inputs so
    the weekly load review (== this digest, COACH-R1 #1) is goal-aware (GBO-R38 / API-R32). Locale
    resolves to the configured default for an unattended run (``en``, LANG-R4); the graph applies
    the athlete's persisted preference where present (LANG-R2).
    """
    inputs = _build_inputs(
        athlete_id=athlete_id,
        trigger="scheduled_digest",
        locale="en",
        request_text=None,
        conversation_id=f"digest:{week_end}",
        active_goals=active_goals,
    )
    final = await graph.run(inputs)
    html, text, status, thread_id = _outputs(final)
    # Same deterministic presentation gate as the free-form answer: scrub raw internal tokens
    # to athlete-native language (VOICE-R2), repair a metrics-report lead to a state read
    # (COACH-R7), and hold the standard-length number cap (VOICE-R7). Presentation only — the
    # grounded citations below are untouched (GROUND-R5/-R7).
    policy = presentation if presentation is not None else _DEFAULT_PRESENTATION
    html, text = enforce_presentation(
        html, text, response_length="standard", presentation=policy
    )
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
# Re-export the multi-day PLAN deliverable from its focused sibling :mod:`plan_deliverable`
# (QUAL-R9 size split, COACH-R2). Like ``readiness_deliverable`` the import is ONE-WAY: the plan
# sibling imports its shared graph-driving/projection primitives from the LEAF :mod:`projection`
# module (NOT from here), so there is no ``deliverables`` <-> ``plan_deliverable`` cycle. Every
# public path stays stable — ``Plan`` / ``plan`` remain importable from ``deliverables``.
# Re-export the data-quality / coverage DIAGNOSIS deliverable from its focused sibling
# :mod:`diagnose_deliverable` (QUAL-R9 size split, API-R15). The import is ONE-WAY (the diagnose
# sibling depends only on the contracts + analytics, never back on ``deliverables``), so there is
# no cycle; re-exporting keeps ``AgentDiagnosis``/``InputCoverage``/``InputStatus`` importable from
# ``wattwise_core.agent.deliverables`` like every other deliverable type.
from wattwise_core.agent.diagnose_deliverable import (  # noqa: E402
    AgentDiagnosis,
    InputCoverage,
    InputStatus,
    diagnose_coverage,
)
from wattwise_core.agent.plan_deliverable import (  # noqa: E402
    Plan,
    plan,
    safe_plan_html,
)
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
    "AgentDiagnosis",
    "Citation",
    "CoachGraph",
    "Digest",
    "InputCoverage",
    "InputStatus",
    "Observation",
    "Plan",
    "Readiness",
    "ReadinessGrounder",
    "ResponseLength",
    "StructuredNarrationError",
    "StructuredNarrator",
    "answer_question",
    "conversation_id_of",
    "count_foregrounded_numbers",
    "diagnose_coverage",
    "first_sentence",
    "leads_with_state",
    "new_conversation_id",
    "number_cap",
    "plan",
    "readiness_assessment",
    "safe_plan_html",
    "thread_id_for",
    "weekly_digest",
]
