"""The Insight + Briefing coach deliverables (COACH-R1 #4/#5; QUAL-R9 size split).

The focused sibling of :mod:`wattwise_core.agent.deliverables` that owns the last two named
COACH-R1 deliverables, completing the canonical five-deliverable set (weekly digest, readiness/
form assessment, multi-day plan, INSIGHT, BRIEFING — names canonical across the product):

* **Insight** (COACH-R1 #4 — *"surface something I'd otherwise miss"*): a SHORT, single-topic,
  grounded observation about the athlete's recent data, each claim cited to a canonical record
  or analytic — a focused, standalone deliverable, never a catch-all. It is driven as a
  ``user_turn`` run over the topic and projected at the SHORT response length (the insight is
  short by contract).
* **Briefing** (COACH-R1 #5 — *"a proactive heads-up before I train, without me asking"*): a
  short, proactively-generated grounded summary driven by the ``scheduled_briefing`` trigger
  for the ONE screen named by the immutable ``briefing_screen`` input (GRAPH-R2.1: intent is
  fixed deterministically by ``(trigger, briefing_screen)`` — no intent model call, no request
  text), assembled from the same grounded inputs as the readiness assessment and digest.

Both share the ONE grounding pipeline (§7) and the one coach voice (§13a) with every other
deliverable — they differ only in scope and trigger, never in grounding guarantees (COACH-R1).
Both carry the originating durable ``thread_id`` plus stable-id observations and
``suggested_followups[]`` so a later turn can follow up on a stored insight/briefing without
re-stating context (COACH-R8). Like its siblings this module is a thin typed PROJECTION over
the :class:`~wattwise_core.agent.projection.CoachGraph` seam (OUTCOME-R2: only the graph's
grounded outputs are surfaced; this layer rewrites no number and certifies no groundedness).

Cited requirements: COACH-R1, COACH-R5, COACH-R8, GRAPH-R2.1, STATE-R2, OUTCOME-R1/-R2/-R3/-R4,
GROUND-R5/-R7, VOICE-R2/-R7, LANG-R2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.projection import (
    CoachGraph,
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
from wattwise_core.agent.voice import (
    Citation,
    Observation,
    VoicePresentation,
    _project_citations,
    enforce_presentation,
)

# The default presentation policy when a caller injects none (the FakeGraph test seam): an
# empty config-loaded map — a surviving internal token is still SCRUBBED to a neutral phrase,
# never shown as a code (fail-closed VOICE-R2). The engine wires the loaded policy in.
_DEFAULT_PRESENTATION = VoicePresentation()


@dataclass(frozen=True, slots=True)
class Insight:
    """A short, single-topic grounded observation deliverable (COACH-R1 #4).

    Same projection guarantees as the other deliverables: status-discriminated outcome
    (OUTCOME-R1), grounded body (OUTCOME-R2), stable-id observations + follow-up affordances
    over the originating durable ``thread_id`` (COACH-R8), surviving citations (GROUND-R5),
    and the typed coverage caveat for a degraded outcome (OUTCOME-R4).
    """

    status: RunStatus
    thread_id: str
    insight_html: str
    insight_text: str
    observations: tuple[Observation, ...] = ()
    citations: tuple[Citation, ...] = ()
    suggested_followups: tuple[str, ...] = ()
    coverage_caveat: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Briefing:
    """The proactive pre-session briefing deliverable for ONE screen (COACH-R1 #5).

    Produced by a ``scheduled_briefing`` run (GRAPH-R2.1): no request text, intent fixed by
    ``(trigger, briefing_screen)``. Derivable only from the canonical inputs it cites — a
    briefing introduces no ungrounded content (COACH-R1 #5); carries the originating
    ``thread_id`` so the athlete can follow up on a stored briefing (COACH-R8).
    """

    status: RunStatus
    thread_id: str
    briefing_screen: str
    briefing_html: str
    briefing_text: str
    observations: tuple[Observation, ...] = ()
    citations: tuple[Citation, ...] = ()
    suggested_followups: tuple[str, ...] = ()
    coverage_caveat: Mapping[str, Any] | None = None


async def insight(
    graph: CoachGraph,
    athlete_id: str,
    topic: str,
    *,
    locale: str = "en",
    presentation: VoicePresentation | None = None,
    thread_id: str | None = None,
    conversation_id: str | None = None,
) -> Insight:
    """Drive the graph for a short, single-topic grounded insight (COACH-R1 #4).

    Builds a ``user_turn`` run over the single ``topic`` (the focused subject the insight
    surfaces — STATE-R2's discriminated union requires request text on this trigger), runs the
    graph, and projects ONLY its grounded outputs (OUTCOME-R2) at the SHORT response length —
    an insight is a short, standalone observation, not a catch-all. The presentation gate
    scrubs internal tokens, repairs the lead to a state read, and holds the short-length
    number cap (VOICE-R2/-R7); grounded citations are untouched (GROUND-R5/-R7). Identity is
    server-derived (AGT-SEC-R1).
    """
    policy = presentation if presentation is not None else _DEFAULT_PRESENTATION
    inputs = _build_inputs(
        athlete_id=athlete_id,
        trigger="user_turn",
        locale=locale,
        request_text=topic,
        response_length="short",
        thread_id=thread_id,
        conversation_id=conversation_id,
    )
    final = await graph.run(inputs)
    html, text, status, out_thread_id = _outputs(final)
    html, text = enforce_presentation(html, text, response_length="short", presentation=policy)
    observations = _project_observations(_as_seq(final.get("observations")))
    return Insight(
        status=status,
        thread_id=out_thread_id,
        insight_html=html,
        insight_text=text,
        observations=observations,
        citations=_project_citations(_as_seq(final.get("citations"))),
        suggested_followups=_generate_followups(status, observations),
        coverage_caveat=_coverage_caveat(final),
    )


async def briefing(
    graph: CoachGraph,
    athlete_id: str,
    briefing_screen: str,
    *,
    locale: str = "en",
    presentation: VoicePresentation | None = None,
    conversation_id: str | None = None,
    active_goals: Sequence[Mapping[str, Any]] | None = None,
) -> Briefing:
    """Drive the graph for the proactive one-screen briefing (COACH-R1 #5, GRAPH-R2.1).

    Builds a ``scheduled_briefing`` run — NO request text and no intent model call; the
    deliverable is fully determined by ``(trigger, briefing_screen)`` (GRAPH-R2.1) — runs the
    graph, and projects its grounded summary at the SHORT length (a briefing is a short
    heads-up). The run scopes to the authenticated athlete exactly as a user turn does and
    passes the same cost-admission gate (GRAPH-R2.1); a missing-input morning ships a visible
    ``degraded`` + caveat rather than a guess (OUTCOME-R3/-R4, GROUND-R7). ``active_goals``
    flow in so the heads-up is goal-aware (GBO-R38).
    """
    policy = presentation if presentation is not None else _DEFAULT_PRESENTATION
    inputs = _build_inputs(
        athlete_id=athlete_id,
        trigger="scheduled_briefing",
        locale=locale,
        request_text=None,
        response_length="short",
        conversation_id=conversation_id or f"briefing:{briefing_screen}",
        briefing_screen=briefing_screen,
        active_goals=active_goals,
    )
    final = await graph.run(inputs)
    html, text, status, thread_id = _outputs(final)
    html, text = enforce_presentation(html, text, response_length="short", presentation=policy)
    observations = _project_observations(_as_seq(final.get("observations")))
    return Briefing(
        status=status,
        thread_id=thread_id,
        briefing_screen=briefing_screen,
        briefing_html=html,
        briefing_text=text,
        observations=observations,
        citations=_project_citations(_as_seq(final.get("citations"))),
        suggested_followups=_generate_followups(status, observations),
        coverage_caveat=_coverage_caveat(final),
    )


__all__ = ["Briefing", "Insight", "briefing", "insight"]
