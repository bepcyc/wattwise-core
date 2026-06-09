"""Wire shapes for the agent router — the typed request/response models (API-R11).

The focused sibling of :mod:`wattwise_core.api.routers.agent_routes` that owns ONLY the
Pydantic request/response models the agent surface serializes (QUAL-R9 size split): the
``POST /v1/agent/ask`` request + its status-discriminated response, the
``POST /v1/agent/threads/{thread_id}/decision`` HITL request/response (API-R12a), the
``GET /v1/agent/readiness`` response, and the small citation/observation/follow-up
members they nest. ``agent_routes`` imports these back and re-exports the public ones, so
every public path (``AgentAskRequest`` / ``AgentAskResponse`` / ``ReadinessResponse`` /
``AgentDecisionRequest`` / ``AgentDecisionResponse``) stays importable from
``agent_routes`` exactly as before. NO route, NO dependency seam, and NO projection logic
lives here — only the wire vocabulary.

Boundary invariants encoded in the shapes:

- **SCHEMA-R4** request/follow-up/decision/readiness models set ``additionalProperties:false``
  so a forged/misnamed field (e.g. an injected ``athlete_id`` or a numeric readiness score) is
  a ``422`` rather than silently accepted.
- **API-R11c** the answer response carries NO billing/budget/model machinery (no
  ``usage``/``cost_*``/token/``model_tier``/``reasoning``/model name).
- **API-R11a / API-R12a** the answer response surfaces ``awaiting_approval`` (an approval-gated
  multi-day PLAN paused at the durable interrupt, CKPT-R9): it carries the ``interrupt_id`` the
  decision endpoint consumes plus the grounded plan body. The ``edited_plan`` of a decision is
  required IFF the decision is ``edit`` (a cross-field rule, not a type).
- **API-R41 / COACH-R7** readiness is a typed VERDICT, never a number: the readiness
  response has no numeric readiness KPI/score field by design.

Requirement IDs: API-R11, API-R11a, API-R11c, API-R11d, API-R11e, API-R11f, API-R12a,
API-R41, SCHEMA-R4, COACH-R7, CKPT-R9.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import AgentAnswer, Citation, Plan, Readiness
from wattwise_core.api.sanitize import sanitize_html

#: The athlete's HITL verdict on a paused approval-gated PLAN (API-R12a / CKPT-R9).
DecisionKind = Literal["approve", "reject", "edit"]

#: The athlete-facing answer-length enum (API-R11f); default ``standard``.
ResponseLength = Literal["short", "standard", "detailed"]
FollowUpKind = Literal["expand", "drill", "reveal_numbers"]


class FollowUp(BaseModel):
    """Typed conversational follow-up over the durable thread (API-R11e).

    ``additionalProperties:false`` (SCHEMA-R4) rejects any unknown nested property so a
    forged/misnamed field can never be silently accepted.
    """

    model_config = ConfigDict(extra="forbid")

    kind: FollowUpKind
    target_ref: str | None = None


class AgentAskRequest(BaseModel):
    """``POST /v1/agent/ask`` request body (API-R11).

    ``question`` is required UNLESS a ``follow_up`` is present (API-R11e); it is bounded
    to 2000 chars (LIMIT-R5). Identity is NOT a field here — it is server-derived
    (AUTH-R3); a client cannot name the athlete it acts as. ``additionalProperties:false``
    (SCHEMA-R4) rejects any unknown body property (e.g. a forged ``athlete_id``) with a
    ``422`` rather than silently dropping it. ``response_length`` is optional: when
    omitted the engine applies the athlete's persisted preference, else ``standard``
    (API-R11f).
    """

    model_config = ConfigDict(extra="forbid")

    question: str | None = Field(default=None, min_length=1, max_length=2000)
    thread_id: str | None = None
    response_length: ResponseLength | None = None
    follow_up: FollowUp | None = None
    language: Literal["en", "de", "ru"] | None = None
    stream: bool = False


class CitationOut(BaseModel):
    """One on-demand grounded citation (API-R11d): ``{metric, value, as_of}`` only.

    References a canonical metric value/date; NEVER an external provider name
    (API-R13 / AUTH-R15). ``source_kind`` is the fixed ``canonical`` marker.
    """

    citation_id: str
    metric: str | None = None
    value: float | None = None
    as_of: str | None = None
    source_kind: Literal["canonical"] = "canonical"


class ObservationOut(BaseModel):
    """One athlete-facing observation with its stable expand/drill handle (API-R11e)."""

    observation_id: str
    text: str


class GroundingOut(BaseModel):
    """The grounding block: the grounded flag + on-demand citations (API-R11d)."""

    grounded: bool
    citations: list[CitationOut]


class SuggestedFollowupOut(BaseModel):
    """An optional athlete-native drill-down chip (API-R11e); jargon-free copy."""

    kind: FollowUpKind
    label: str
    target_ref: str | None = None


class DegradedOut(BaseModel):
    """The ``degraded`` member payload: human caveat + typed coverage caveat (API-R11a)."""

    reason_text: str
    coverage_caveat: dict[str, Any] | None = None


class AgentAskResponse(BaseModel):
    """The status-discriminated ``AgentAskResponse`` union (API-R11a / API-R12a).

    OSS surfaces ``completed``, ``degraded``, AND ``awaiting_approval`` — the last is an
    approval-gated multi-day PLAN that paused at the durable interrupt-gate (CKPT-R9): it
    carries the ``interrupt_id`` the ``POST …/decision`` endpoint consumes plus the grounded
    plan body (``plan_html`` already sanitized, ``plan_text``). (``budget_exceeded`` remains
    commercial-only and is never produced by the OSS engine.) Carries NO billing/budget/model
    machinery (API-R11c): there is deliberately no ``usage``/``cost_*``/token/``model_tier``/
    ``reasoning`` field on this schema. ``answer_html`` is already sanitized (API-R13). The
    plan fields are populated ONLY for the ``awaiting_approval`` member; ``answer_html``/
    ``answer_text`` carry the plan body there too so a client that ignores ``status`` still
    renders the grounded prose.
    """

    status: Literal["completed", "degraded", "awaiting_approval"]
    thread_id: str
    trace_id: str
    answer_html: str
    answer_text: str
    observations: list[ObservationOut]
    grounding: GroundingOut
    suggested_followups: list[SuggestedFollowupOut] = Field(default_factory=list)
    degraded: DegradedOut | None = None
    interrupt_id: str | None = None
    plan_html: str | None = None
    plan_text: str | None = None


class AgentDecisionRequest(BaseModel):
    """``POST /v1/agent/threads/{thread_id}/decision`` request body (API-R12a / CKPT-R9).

    The athlete's HITL verdict on a paused approval-gated PLAN. ``additionalProperties:false``
    (SCHEMA-R4) rejects any unknown body property (e.g. a forged ``athlete_id`` or a renamed
    ``thread_id`` — the thread is the PATH param, never a body field). The cross-field rule is
    enforced here so the engine never sees an inconsistent decision: ``edited_plan`` is REQUIRED
    when (and only when) ``decision == "edit"`` (an edit with no body, or a non-edit carrying an
    ``edited_plan``, is a ``422`` validation error, not a model call). The ``edited_plan`` is
    re-grounded by the engine before resume (GROUND-R3) — it can never smuggle an unverified
    number/name past grounding.
    """

    model_config = ConfigDict(extra="forbid")

    interrupt_id: str = Field(min_length=1)
    decision: DecisionKind
    edited_plan: str | None = Field(default=None, max_length=20000)

    @model_validator(mode="after")
    def _edited_plan_iff_edit(self) -> AgentDecisionRequest:
        """``edited_plan`` is present IFF the decision is ``edit`` (API-R12a cross-field rule)."""
        if self.decision == "edit" and not (self.edited_plan and self.edited_plan.strip()):
            raise ValueError("edited_plan is required when decision is 'edit'")
        if self.decision != "edit" and self.edited_plan is not None:
            raise ValueError("edited_plan is only allowed when decision is 'edit'")
        return self


class AgentDecisionResponse(BaseModel):
    """The ``POST …/decision`` response: the finalized PLAN after resume (API-R12a / CKPT-R9).

    A winning decision atomically consumed the live interrupt and resumed the durable thread; the
    body is the now-terminal grounded plan (``approve`` finalizes it; ``edit`` returns the
    re-grounded edit; ``reject`` returns the resumed-without-approval outcome). ``status`` is the
    terminal run status the resume reached (typically ``completed``); ``plan_html`` is already
    sanitized (API-R13). Carries NO billing/budget/model machinery (API-R11c).
    """

    status: Literal["completed", "degraded"]
    thread_id: str
    trace_id: str
    decision: DecisionKind
    plan_html: str
    plan_text: str
    observations: list[ObservationOut] = Field(default_factory=list)
    grounding: GroundingOut
    suggested_followups: list[SuggestedFollowupOut] = Field(default_factory=list)


#: The readiness/form verdict enum on the wire (API-R41 / COACH-R7). A typed STATE, never
#: a number — there is deliberately no numeric readiness field on the response.
ReadinessVerdictOut = Literal["go", "maintain", "ease", "rest"]


class ReadinessResponse(BaseModel):
    """The ``GET /v1/agent/readiness`` response (API-R41).

    Readiness is a typed VERDICT (``go|maintain|ease|rest``), NOT a number: this schema
    carries NO numeric ``readiness`` KPI/score field by design (API-R41 / COACH-R7). The
    ``verdict`` is ``null`` when readiness cannot be assessed (insufficient grounded data,
    GROUND-R6). ``summary_text`` LEADS with a warm, number-light state sentence (COACH-R7);
    the form number is demoted to on-demand grounded ``citations`` ({metric,value,as_of})
    only (GROUND-R5/R7). ``summary_html`` is server-side sanitized (API-R13 / SCHEMA-R7).
    ``coverage`` carries the typed inputs-used/unavailable map + any consistency-override
    caveat; it never carries billing/model machinery (API-R11c). ``additionalProperties``
    is closed so no forged/numeric readiness field can be smuggled in.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: ReadinessVerdictOut | None
    as_of: str | None = None
    trace_id: str
    summary_html: str
    summary_text: str
    observations: list[ObservationOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    coverage: dict[str, Any] | None = None
    suggested_followups: list[SuggestedFollowupOut] = Field(default_factory=list)


# --- projection (deliverable -> sanitized wire shape) ----------------------------
# These free functions own the SINGLE place a deliverable becomes its wire model + is
# server-side sanitized (API-R13 / SCHEMA-R7). They live with the wire vocabulary (QUAL-R9
# size split) and are imported back by the router, which keeps only HTTP plumbing.

#: The per-language warm reason_text for a degraded outcome (API-R11a / API-R37). The
#: structured ``coverage_caveat`` carries the machine basis; this is its human gloss in
#: the athlete's selected language (en/de/ru), externalized as catalog copy (QUAL-R13).
DEGRADED_REASON_BY_LOCALE: Final[dict[str, str]] = {
    "en": "I built this with what we have — a source is offline.",
    "de": "Ich habe das mit den vorhandenen Daten erstellt — eine Quelle ist offline.",
    "ru": "Я собрал это из того, что есть — один источник недоступен.",
}


def grounded_flag(answer: AgentAnswer) -> bool:
    """True iff the engine produced a grounded terminal outcome (API-R12).

    A ``completed`` or ``degraded`` outcome is grounded (degraded is partial-coverage
    grounded, never fabricated). Any other/absent status (e.g. ``awaiting_approval`` on the
    free-form ``/ask`` path, which carries no interrupt_id) is treated as ungrounded so the
    endpoint fails closed rather than emitting an ungrounded answer.
    """
    return answer.status in (RunStatus.COMPLETED, RunStatus.DEGRADED)


def citations_out(citations: Sequence[Citation]) -> list[CitationOut]:
    """Project surviving grounded citations into the wire shape (API-R11d).

    Shared by the answer, plan, and readiness renders — all project the same canonical
    ``{metric, value, as_of}`` + record-id citation shape, so none carries an external
    provider name (API-R13 / AUTH-R15).
    """
    return [
        CitationOut(citation_id=cit.record_id, metric=cit.metric, value=cit.value, as_of=cit.as_of)
        for cit in citations
    ]


def _observations_out(
    items: Sequence[Any],
) -> list[ObservationOut]:
    """Project stable-id observations into the wire shape (API-R11e); shared by every render."""
    return [ObservationOut(observation_id=o.observation_id, text=o.text) for o in items]


def _expand_chips(labels: Sequence[str]) -> list[SuggestedFollowupOut]:
    """Project jargon-free follow-up LABELS into ``expand`` chips (API-R11e); shared plan/answer."""
    return [SuggestedFollowupOut(kind="expand", label=label) for label in labels]


def _degraded_out(answer: AgentAnswer, locale: str) -> DegradedOut | None:
    """Build the ``degraded`` member payload, else ``None`` (API-R11a / API-R37).

    Present only for a ``degraded`` outcome: the human ``reason_text`` in the athlete's
    selected language (en/de/ru, API-R37) plus the typed ``coverage_caveat`` (source-agnostic
    missing/substituted/stale state), passed through without inventing a number.
    """
    if answer.status is not RunStatus.DEGRADED:
        return None
    caveat = dict(answer.coverage_caveat) if answer.coverage_caveat is not None else None
    reason = DEGRADED_REASON_BY_LOCALE.get(locale, DEGRADED_REASON_BY_LOCALE["en"])
    return DegradedOut(reason_text=reason, coverage_caveat=caveat)


def render_response(answer: AgentAnswer, trace_id: str, locale: str) -> AgentAskResponse:
    """Render a grounded :class:`AgentAnswer` into the sanitized response union (API-R11a).

    ``answer_html`` is sanitized HERE (API-R13 / SCHEMA-R7) before it leaves the API. Maps the OSS
    terminal status to the union's closed member; only ``completed``/``degraded`` reach this render
    (a paused plan is rendered by :func:`render_plan_awaiting`). The degraded human caveat is
    localized to ``locale`` (API-R37).
    """
    member: Literal["completed", "degraded"] = (
        "degraded" if answer.status is RunStatus.DEGRADED else "completed"
    )
    return AgentAskResponse(
        status=member,
        thread_id=answer.thread_id,
        trace_id=trace_id,
        answer_html=sanitize_html(answer.answer_html),
        answer_text=answer.answer_text,
        observations=_observations_out(answer.observations),
        grounding=GroundingOut(grounded=True, citations=citations_out(answer.citations)),
        suggested_followups=_expand_chips(answer.suggested_followups),
        degraded=_degraded_out(answer, locale),
    )


def render_plan_awaiting(plan: Plan, trace_id: str) -> AgentAskResponse:
    """Render a PAUSED approval-gated PLAN as the ``awaiting_approval`` union member (API-R12a).

    Surfaces the ``interrupt_id`` the decision endpoint consumes (CKPT-R9) + the grounded plan body
    in BOTH the dedicated ``plan_*`` fields and the shared ``answer_*`` fields (so a client that
    ignores ``status`` still renders the prose). ``plan_html`` is sanitized HERE (API-R13);
    grounding is ALWAYS true for a surfaced plan (only grounded prescriptions survive, OUTCOME-R2).
    """
    safe_html = sanitize_html(plan.plan_html)
    return AgentAskResponse(
        status="awaiting_approval",
        thread_id=plan.thread_id,
        trace_id=trace_id,
        answer_html=safe_html,
        answer_text=plan.plan_text,
        observations=_observations_out(plan.observations),
        grounding=GroundingOut(grounded=True, citations=citations_out(plan.citations)),
        suggested_followups=_expand_chips(plan.suggested_followups),
        interrupt_id=plan.interrupt_id,
        plan_html=safe_html,
        plan_text=plan.plan_text,
    )


def render_decision(
    plan: Plan, decision: DecisionKind, trace_id: str
) -> AgentDecisionResponse:
    """Render the resumed, now-terminal PLAN into the decision response (API-R12a / CKPT-R9).

    ``plan_html`` is sanitized HERE (API-R13). A resumed plan is grounded (degraded only when a
    source went offline mid-run); the terminal status maps to the closed ``completed``/``degraded``
    member. The body is the finalized plan — for an ``edit`` the engine already replaced it with the
    RE-GROUNDED edit (GROUND-R3), so the API never sees un-grounded edited prose.
    """
    member: Literal["completed", "degraded"] = (
        "degraded" if plan.status is RunStatus.DEGRADED else "completed"
    )
    return AgentDecisionResponse(
        status=member,
        thread_id=plan.thread_id,
        trace_id=trace_id,
        decision=decision,
        plan_html=sanitize_html(plan.plan_html),
        plan_text=plan.plan_text,
        observations=_observations_out(plan.observations),
        grounding=GroundingOut(grounded=True, citations=citations_out(plan.citations)),
        suggested_followups=_expand_chips(plan.suggested_followups),
    )


def _readiness_followups_out(readiness: Readiness) -> list[SuggestedFollowupOut]:
    """Project the jargon-free reveal-the-numbers chips (API-R11e / VOICE-R9)."""
    return [
        SuggestedFollowupOut(kind="reveal_numbers", label=label)
        for label in readiness.suggested_followups
    ]


def render_readiness(readiness: Readiness, trace_id: str) -> ReadinessResponse:
    """Render the readiness deliverable into the sanitized typed response (API-R41).

    ``summary_html`` is sanitized HERE (API-R13 / SCHEMA-R7). The verdict is the StrEnum value (or
    ``None`` when the deliverable abstained); there is no numeric readiness field. ``coverage``
    passes through the engine's typed map unchanged (no invented number).
    """
    coverage = dict(readiness.coverage) if readiness.coverage is not None else None
    return ReadinessResponse(
        verdict=readiness.verdict.value if readiness.verdict is not None else None,
        as_of=readiness.as_of,
        trace_id=trace_id,
        summary_html=sanitize_html(readiness.summary_html),
        summary_text=readiness.summary_text,
        observations=_observations_out(readiness.observations),
        citations=citations_out(readiness.citations),
        coverage=coverage,
        suggested_followups=_readiness_followups_out(readiness),
    )


# The diagnose / digest / memory wire shapes live in the focused :mod:`agent_breadth_schemas`
# sibling (QUAL-R9 size split); they are re-exported here so every public path stays importable
# from ``agent_schemas`` exactly as before. The import sits AFTER the shared members above
# (``ObservationOut`` / ``GroundingOut`` / ``_expand_chips`` / ``DEGRADED_REASON_BY_LOCALE`` …) so
# the breadth-schema module can import them without an import cycle.
from wattwise_core.api.routers.agent_breadth_schemas import (  # noqa: E402
    AgentDiagnosisResponse,
    DeliveryChannelOut,
    DigestBody,
    DigestCadenceOut,
    DigestStatusOut,
    DigestSubscribeRequest,
    DigestSubscriptionList,
    DigestSubscriptionOut,
    InputCoverageOut,
    MemoryEraseAck,
    MemoryItemList,
    MemoryItemOut,
    WeekdayOut,
    memory_item_out,
    render_diagnosis,
    render_digest,
)

__all__ = [
    "DEGRADED_REASON_BY_LOCALE",
    "AgentAskRequest",
    "AgentAskResponse",
    "AgentDecisionRequest",
    "AgentDecisionResponse",
    "AgentDiagnosisResponse",
    "CitationOut",
    "DecisionKind",
    "DegradedOut",
    "DeliveryChannelOut",
    "DigestBody",
    "DigestCadenceOut",
    "DigestStatusOut",
    "DigestSubscribeRequest",
    "DigestSubscriptionList",
    "DigestSubscriptionOut",
    "FollowUp",
    "FollowUpKind",
    "GroundingOut",
    "InputCoverageOut",
    "MemoryEraseAck",
    "MemoryItemList",
    "MemoryItemOut",
    "ObservationOut",
    "ReadinessResponse",
    "ReadinessVerdictOut",
    "ResponseLength",
    "SuggestedFollowupOut",
    "WeekdayOut",
    "citations_out",
    "grounded_flag",
    "memory_item_out",
    "render_decision",
    "render_diagnosis",
    "render_digest",
    "render_plan_awaiting",
    "render_readiness",
    "render_response",
]
