"""Wire shapes for the agent BREADTH router — diagnose / digest / memory models (API-R14/R15).

The focused sibling of :mod:`wattwise_core.api.routers.agent_schemas` (QUAL-R9 size split) that owns
ONLY the Pydantic request/response models the breadth surfaces serialize plus the free-function
projections that turn a deliverable / canonical row into its sanitized-later wire shape:

- the ``POST /v1/agent/diagnose`` response (:class:`AgentDiagnosisResponse`, API-R15);
- the weekly-digest body (:class:`DigestBody`, ``GET …/digest/last``, API-R14) and the standing
  digest-subscription request/response models (:class:`DigestSubscribeRequest` /
  :class:`DigestSubscriptionOut` / :class:`DigestSubscriptionList`, API-R14 / GBO-R46);
- the per-item memory models (:class:`MemoryItemOut` / :class:`MemoryItemList` /
  :class:`MemoryEraseAck`, API-R15a / MEM-R3).

``agent_schemas`` re-exports the public names so every path stays importable from there. NO route,
NO dependency seam, and NO model call lives here — only the wire vocabulary + the deterministic
projections. The shared citation/observation/follow-up members + the localized degraded copy are
imported from :mod:`agent_schemas` so there is ONE definition of each (no drift).

Boundary invariants encoded in the shapes:

- **SCHEMA-R4** the request models set ``additionalProperties:false`` so a forged/misnamed field
  (e.g. an injected ``athlete_id``, a spoofed ``verified``) is a ``422`` rather than silently kept.
- **API-R11c** no response carries billing/budget/model machinery.
- **API-R15 / VOICE-R7** a diagnosis / digest reports coverage and grounded prose, never a made-up
  number; the memory shapes carry personalization context only, never a canonical value (MEM-R1).

Requirement IDs: API-R14, API-R15, API-R15a, API-R11c, API-R13, GBO-R46, GBO-R47, MEM-R1, MEM-R2,
MEM-R3, OUTCOME-R3, SCHEMA-R4, VOICE-R7.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import Digest
from wattwise_core.agent.diagnose_deliverable import AgentDiagnosis, InputCoverage
from wattwise_core.agent.memory import RecalledItem
from wattwise_core.api.routers.agent_request import catalog_locale
from wattwise_core.api.routers.agent_schemas import (
    DEGRADED_REASON_BY_LOCALE,
    DegradedOut,
    GroundingOut,
    ObservationOut,
    SuggestedFollowupOut,
    _expand_chips,
    _observations_out,
    citations_out,
)
from wattwise_core.api.sanitize import sanitize_html

# --- /v1/agent/diagnose — data-quality / coverage diagnosis (API-R15) -------------


class DiagnosisCheckOut(BaseModel):
    """One deterministic data-health CHECK on the wire (API-R15).

    ``code`` is the stable machine id the client branches on (the canonical input key from
    the data-health/coverage probe); ``ok`` is its boolean pass state (``present``); the
    ``detail`` is the jargon-free athlete-native narration of the check (VOICE-R2). There
    is deliberately NO numeric field — a diagnosis reports coverage, never a canonical
    value (VOICE-R7 / GROUND-R7).
    """

    code: str
    ok: bool
    detail: str


class AgentDiagnosisResponse(BaseModel):
    """The ``POST /v1/agent/diagnose`` response: ``AgentDiagnosis`` (API-R15).

    The spec-named shape — a list of deterministic ``checks`` ``{code, ok, detail}`` over
    the athlete's canonical coverage plus the rolled-up ``overall_ok``. The checks are
    SOURCED from the deterministic data-health/coverage probe (doc 30 HLT-R*) — never
    invented (GROUND-R7); the narration in ``detail`` stays jargon-free (API-R21).
    ``as_of`` is the ISO date the probe windowed against; ``trace_id`` correlates the
    request. Carries NO athlete-facing numbers (VOICE-R7) and NO billing/model machinery
    (API-R11c). ``additionalProperties`` is closed so no numeric field can be smuggled in.
    """

    model_config = ConfigDict(extra="forbid")

    checks: list[DiagnosisCheckOut]
    overall_ok: bool
    as_of: str
    trace_id: str


def render_diagnosis(diagnosis: AgentDiagnosis, trace_id: str) -> AgentDiagnosisResponse:
    """Render the deterministic diagnosis deliverable into ``AgentDiagnosis`` (API-R15).

    Maps each typed coverage line to a ``{code, ok, detail}`` check verbatim — ``ok`` is
    the ``present`` state, ``detail`` narrates the label (+ the typed reason for a failed
    check) without inventing a value (GROUND-R7 / VOICE-R7). ``overall_ok`` is true only
    when EVERY check passes (fail-closed roll-up).
    """
    checks = [_check_out(i) for i in diagnosis.inputs]
    return AgentDiagnosisResponse(
        checks=checks,
        overall_ok=all(c.ok for c in checks),
        as_of=diagnosis.as_of,
        trace_id=trace_id,
    )


def _check_out(coverage: InputCoverage) -> DiagnosisCheckOut:
    """Project one coverage line onto the ``{code, ok, detail}`` check (API-R15)."""
    ok = coverage.status.value == "present"
    detail = coverage.label if ok else f"{coverage.label} — {coverage.status.value}"
    return DiagnosisCheckOut(code=coverage.key, ok=ok, detail=detail)


# --- /v1/agent/digest — the weekly digest + its standing subscription (API-R14) ---


class DigestBody(BaseModel):
    """The grounded weekly-digest body returned by ``GET /v1/agent/digest/last`` (API-R14).

    A status-discriminated weekly load review (== the COACH-R1 #1 deliverable): ``completed`` or
    ``degraded`` (a week whose canonical inputs are missing abstains VISIBLY rather than guessing,
    OUTCOME-R3). ``digest_html`` is already server-side sanitized (API-R13 / SCHEMA-R7). Carries the
    grounded ``observations`` + on-demand ``citations`` and NO billing/model machinery (API-R11c).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "degraded"]
    thread_id: str
    week_end: str
    trace_id: str
    digest_html: str
    digest_text: str
    observations: list[ObservationOut] = Field(default_factory=list)
    grounding: GroundingOut
    suggested_followups: list[SuggestedFollowupOut] = Field(default_factory=list)
    degraded: DegradedOut | None = None


def render_digest(digest: Digest, trace_id: str, locale: str) -> DigestBody:
    """Render the weekly digest deliverable into the sanitized typed body (API-R14 / API-R13).

    ``digest_html`` is sanitized HERE before it leaves the API (API-R13 / SCHEMA-R7). A ``degraded``
    week surfaces the localized human caveat (API-R37) over the typed ``coverage_caveat`` — the
    week's inputs were missing, never fabricated (OUTCOME-R3/-R4). Grounding is always true for a
    surfaced digest (only grounded prose survives).
    """
    member: Literal["completed", "degraded"] = (
        "degraded" if digest.status is RunStatus.DEGRADED else "completed"
    )
    degraded = None
    if digest.status is RunStatus.DEGRADED:
        caveat = dict(digest.coverage_caveat) if digest.coverage_caveat is not None else None
        reason = DEGRADED_REASON_BY_LOCALE[catalog_locale(locale)]
        degraded = DegradedOut(reason_text=reason, coverage_caveat=caveat)
    return DigestBody(
        status=member,
        thread_id=digest.thread_id,
        week_end=digest.week_end,
        trace_id=trace_id,
        digest_html=sanitize_html(digest.digest_html),
        digest_text=digest.digest_text,
        observations=_observations_out(digest.observations),
        grounding=GroundingOut(grounded=True, citations=citations_out(digest.citations)),
        suggested_followups=_expand_chips(digest.suggested_followups),
        degraded=degraded,
    )


#: The wire vocab for a digest schedule (GBO-R46): a Mon-first weekday token, identical on the wire.
DigestCadenceOut = Literal["daily", "weekly", "monthly"]
WeekdayOut = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DeliveryChannelOut = Literal["web", "email", "telegram"]
DigestStatusOut = Literal["active", "paused", "cancelled"]


class DigestSubscribeRequest(BaseModel):
    """``POST /v1/agent/digest/subscribe`` request body (API-R14 / GBO-R46).

    The ONE standing digest schedule for the server-derived owner. Identity is NOT a field here
    (AUTH-R3); ``additionalProperties:false`` (SCHEMA-R4) rejects any forged/misnamed property
    (e.g. an injected ``athlete_id``). ``hour_local`` is the athlete-LOCAL firing hour (0-23),
    NEVER a UTC hour (GBO-R47). ``weekday`` is the Mon-first token (GBO-R46b), required for a weekly
    cadence. ``channels`` is the ordered set of delivery channels (GBO-R46c); the ``email`` channel
    is GATED — a digest e-mail is delivered only once the owner's email is verified (GBO-R49), so a
    subscription that names ``email`` before the address is verified is refused (router-side 422).
    """

    model_config = ConfigDict(extra="forbid")

    cadence: DigestCadenceOut
    hour_local: int = Field(ge=0, le=23)
    weekday: WeekdayOut | None = None
    channels: list[DeliveryChannelOut] = Field(min_length=1)

    @model_validator(mode="after")
    def _weekday_iff_weekly(self) -> DigestSubscribeRequest:
        """A weekly cadence REQUIRES a weekday; daily/monthly must NOT carry one (GBO-R46b)."""
        if self.cadence == "weekly" and self.weekday is None:
            raise ValueError("weekday is required when cadence is 'weekly'")
        if self.cadence != "weekly" and self.weekday is not None:
            raise ValueError("weekday is only allowed when cadence is 'weekly'")
        return self


class DigestSubscriptionOut(BaseModel):
    """One standing digest subscription on the wire (API-R14 / GBO-R46).

    Mirrors the canonical :class:`~wattwise_core.persistence.models.notify.DigestSubscription`:
    the surrogate ``subscription_id``, the schedule (cadence / weekday / athlete-local hour), the
    ordered channels, and the lifecycle ``status``. Carries no athlete identity (server-derived,
    AUTH-R3) and no billing/model machinery (API-R11c).
    """

    subscription_id: str
    cadence: DigestCadenceOut
    weekday: WeekdayOut | None
    hour_local: int
    channels: list[str]
    status: DigestStatusOut


class DigestSubscriptionList(BaseModel):
    """``GET /v1/agent/digest/subscriptions``: the owner's standing subscriptions (API-R14)."""

    data: list[DigestSubscriptionOut]


class DigestPage(BaseModel):
    """The PAGE-R4 envelope page block for the weekly-review history (API-R14 / §5)."""

    limit: int
    next_cursor: str | None = None
    has_more: bool


class DigestList(BaseModel):
    """``GET /v1/agent/digest/list``: the PAGINATED weekly-review history (API-R14 / §5).

    Each item is the SAME sanitized :class:`DigestBody` the ``digest/last`` surface
    returns (replayed verbatim from the stored grounded review — never recomputed,
    GROUND-R7), wrapped in the PAGE-R4 envelope behind a signed opaque cursor (PAGE-R5).
    """

    data: list[DigestBody]
    page: DigestPage


# --- /v1/agent/memory — the per-item read + erase seam (API-R15a / MEM-R3) --------


class MemoryItemOut(BaseModel):
    """One durable memory item on the wire — the spec ``MemoryItem`` (API-R15a / MEM-R1).

    Personalization context ONLY — never a canonical analytic number (MEM-R1): there is
    deliberately no numeric field. ``memory_id`` is the stable handle the per-item
    GET/DELETE address (MEM-R3 erasure); ``summary_text`` is the athlete-native content
    (API-R21); ``created_at``/``updated_at`` are the ISO instants the item was captured /
    last revised.
    """

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    kind: str
    summary_text: str
    created_at: str
    updated_at: str


class MemoryItemList(BaseModel):
    """The ``GET /v1/agent/memory`` response: the owner's durable memory rows (API-R15a)."""

    data: list[MemoryItemOut]


def memory_item_out(item: RecalledItem) -> MemoryItemOut:
    """Project one durable memory row onto the spec ``MemoryItem`` shape (API-R15a / MEM-R1).

    Carries personalization context only — never a canonical number (the store structurally
    has no numeric field, MEM-R1). ``updated_at`` falls back to the capture instant when the
    row has never been revised.
    """
    updated = item.updated_at or item.recorded_at
    return MemoryItemOut(
        memory_id=item.memory_item_id,
        kind=item.kind.value,
        summary_text=item.content,
        created_at=item.recorded_at.isoformat(),
        updated_at=updated.isoformat(),
    )


__all__ = [
    "AgentDiagnosisResponse",
    "DeliveryChannelOut",
    "DiagnosisCheckOut",
    "DigestBody",
    "DigestCadenceOut",
    "DigestList",
    "DigestPage",
    "DigestStatusOut",
    "DigestSubscribeRequest",
    "DigestSubscriptionList",
    "DigestSubscriptionOut",
    "MemoryItemList",
    "MemoryItemOut",
    "WeekdayOut",
    "memory_item_out",
    "render_diagnosis",
    "render_digest",
]
