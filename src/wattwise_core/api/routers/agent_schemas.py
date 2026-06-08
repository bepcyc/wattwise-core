"""Wire shapes for the agent router — the typed request/response models (API-R11).

The focused sibling of :mod:`wattwise_core.api.routers.agent_routes` that owns ONLY the
Pydantic request/response models the agent surface serializes (QUAL-R9 size split): the
``POST /v1/agent/ask`` request + its status-discriminated response, the
``GET /v1/agent/readiness`` response, and the small citation/observation/follow-up
members they nest. ``agent_routes`` imports these back and re-exports the public ones, so
every public path (``AgentAskRequest`` / ``AgentAskResponse`` / ``ReadinessResponse``)
stays importable from ``agent_routes`` exactly as before. NO route, NO dependency seam,
and NO projection logic lives here — only the wire vocabulary.

Boundary invariants encoded in the shapes:

- **SCHEMA-R4** request/follow-up/readiness models set ``additionalProperties:false`` so a
  forged/misnamed field (e.g. an injected ``athlete_id`` or a numeric readiness score) is a
  ``422`` rather than silently accepted.
- **API-R11c** the answer response carries NO billing/budget/model machinery (no
  ``usage``/``cost_*``/token/``model_tier``/``reasoning``/model name).
- **API-R41 / COACH-R7** readiness is a typed VERDICT, never a number: the readiness
  response has no numeric readiness KPI/score field by design.

Requirement IDs: API-R11, API-R11a, API-R11c, API-R11d, API-R11e, API-R11f, API-R41,
SCHEMA-R4, COACH-R7.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    """The status-discriminated ``AgentAskResponse`` union (API-R11a).

    OSS surfaces ``completed`` and ``degraded`` (the engine never produces
    ``awaiting_approval``/``budget_exceeded`` in OSS). Carries NO billing/budget/model
    machinery (API-R11c): there is deliberately no ``usage``/``cost_*``/token/
    ``model_tier``/``reasoning`` field on this schema. ``answer_html`` is already
    sanitized (API-R13).
    """

    status: Literal["completed", "degraded"]
    thread_id: str
    trace_id: str
    answer_html: str
    answer_text: str
    observations: list[ObservationOut]
    grounding: GroundingOut
    suggested_followups: list[SuggestedFollowupOut] = Field(default_factory=list)
    degraded: DegradedOut | None = None


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


__all__ = [
    "AgentAskRequest",
    "AgentAskResponse",
    "CitationOut",
    "DegradedOut",
    "FollowUp",
    "FollowUpKind",
    "GroundingOut",
    "ObservationOut",
    "ReadinessResponse",
    "ReadinessVerdictOut",
    "ResponseLength",
    "SuggestedFollowupOut",
]
