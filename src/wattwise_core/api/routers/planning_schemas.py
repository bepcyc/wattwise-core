"""Wire shapes for the planning router — request/response models + projections (API-R32).

The focused sibling of :mod:`wattwise_core.api.routers.planning` that owns ONLY the Pydantic
request/response models the planning surface serializes plus the free-function projections that
turn a deliverable / canonical ORM row into its sanitized-later wire shape (QUAL-R9 size split).
``planning`` imports these back, so every public path stays importable. NO route, NO dependency
seam, and NO model call lives here — only the wire vocabulary + the deterministic projections.

Boundary invariants encoded in the shapes:

- **SCHEMA-R4** :class:`PlanRequest` sets ``additionalProperties:false`` so a forged/misnamed field
  (e.g. an injected ``athlete_id``) is a ``422`` rather than silently accepted.
- **API-R11c** the generated-plan response reuses the agent answer union and carries NO
  billing/budget/model machinery.
- **API-R32** the read views are source-agnostic (no provider name, AUTH-R15); the schedule view is
  READ-ONLY — there is no per-day mutation shape here (``schedule_adjustment`` is post-v1).

Requirement IDs: API-R32, API-R11a, API-R11c, API-R13, AUTH-R15, GBO-R29, GBO-R30a, GBO-R30b,
PAGE-R1, RUN-R4.1, SCHEMA-R4.
"""

from __future__ import annotations

import datetime as _dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import AgentAnswer as _AgentAnswer
from wattwise_core.agent.deliverables import Plan as PlanDeliverable
from wattwise_core.api.routers.agent_schemas import (
    AgentAskResponse,
    DegradedOut,
    GroundingOut,
    ResponseLength,
    render_plan_awaiting,
    render_response,
)
from wattwise_core.domain.enums import PlanDayIntent, PlanStatus
from wattwise_core.persistence.models import Plan as PlanRow
from wattwise_core.persistence.models import PlanDay, Workout

#: The languages this surface localizes athlete-facing copy into (API-R37).
ResponseLanguage = Literal["en", "de", "ru"]

#: The per-language phase-gated "plan generation not switched on" copy (RUN-R4.1 / API-R37).
PHASE_GATED_BY_LOCALE: dict[str, str] = {
    "en": "Plan generation isn't switched on for this account yet.",
    "de": "Die Planerstellung ist fuer dieses Konto noch nicht aktiviert.",
    "ru": "Sostavlenie plana poka ne podklyuchyeno dlya etoy uchyotnoy zapisi.",
}


# --- POST /v1/planning/workouts — request + render --------------------------------


class PlanRequest(BaseModel):
    """``POST /v1/planning/workouts`` request body (API-R32).

    Identity is NOT a field here — it is server-derived (AUTH-R3); a client cannot name the athlete
    it acts as. ``additionalProperties:false`` (SCHEMA-R4) rejects any unknown/forged body property
    (e.g. an injected ``athlete_id``) with a ``422``. ``request`` is the athlete's free-text plan
    ask (bounded to 2000 chars, LIMIT-R5); ``thread_id`` continues a prior durable thread when set;
    ``response_length`` defaults to ``detailed`` for a multi-day plan when omitted (API-R11f).
    """

    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None
    response_length: ResponseLength | None = None
    language: ResponseLanguage | None = None


def resolve_length(body: PlanRequest) -> ResponseLength:
    """A multi-day plan defaults to ``detailed`` verbosity when omitted (API-R11f)."""
    return body.response_length or "detailed"


def resolve_locale(
    body: PlanRequest, accept_language: str | None, persisted: str | None = None
) -> str:
    """Resolve the response language per the API-R37 precedence chain.

    Body ``language`` -> ``Accept-Language`` -> the PERSISTED setting (the language
    subtag of ``athlete.primary_locale``, loaded server-side) -> the engine ``en``
    baseline — identical to the agent path so the plan surface honors the stored
    default (API-R37) without a per-call override mutating it.
    """
    if body.language is not None:
        return body.language
    if accept_language:
        for part in accept_language.split(","):
            tag = part.split(";", 1)[0].strip().lower()[:2]
            if tag in PHASE_GATED_BY_LOCALE:
                return tag
    if persisted in PHASE_GATED_BY_LOCALE:
        return persisted
    return "en"


def phase_gated_response(locale: str, thread_id: str | None, trace_id: str) -> AgentAskResponse:
    """The typed ``degraded`` "plan generation not switched on" answer (RUN-R4.1 / phase-gating).

    Plan generation is phase-gated (doc 60 §phase-gating): when the wired engine has no LLM it does
    not implement ``plan_deliverable``, so the endpoint fails closed to a typed, jargon-free
    ``degraded`` answer (no internals leaked, VOICE-R2/-R3) — NEVER a fabricated plan, never a raw
    error. Same :class:`AgentAskResponse` union the live path returns, with the typed caveat.
    """
    text = PHASE_GATED_BY_LOCALE.get(locale, PHASE_GATED_BY_LOCALE["en"])
    return AgentAskResponse(
        status="degraded",
        thread_id=thread_id or "unconfigured",
        trace_id=trace_id,
        answer_html=f"<p>{text}</p>",
        answer_text=text,
        observations=[],
        grounding=GroundingOut(grounded=True, citations=[]),
        degraded=DegradedOut(reason_text=text, coverage_caveat={"reason": "agent_unconfigured"}),
    )


def render_plan(plan: PlanDeliverable, trace_id: str, locale: str) -> AgentAskResponse:
    """Render a generated :class:`Plan` into the sanitized response union (API-R11a / API-R12a).

    A paused approval-gated plan (``awaiting_approval`` + ``interrupt_id``) is rendered by
    :func:`render_plan_awaiting` so it surfaces the ``interrupt_id`` the EXISTING decision endpoint
    consumes (CKPT-R9). A non-paused terminal plan (``completed``/``degraded``) is adapted into the
    shared answer projection — both go through the same server-side sanitizer (API-R13).
    """
    if plan.status is RunStatus.AWAITING_APPROVAL and plan.interrupt_id is not None:
        return render_plan_awaiting(plan, trace_id)
    answer = _AgentAnswer(
        status=plan.status,
        thread_id=plan.thread_id,
        answer_html=plan.plan_html,
        answer_text=plan.plan_text,
        observations=plan.observations,
        citations=plan.citations,
        suggested_followups=plan.suggested_followups,
        coverage_caveat=plan.coverage_caveat,
    )
    return render_response(answer, trace_id, locale)


# --- GET /v1/planning/workouts — paginated read view ------------------------------


class WorkoutStepOut(BaseModel):
    """One prescribed workout step: its target zones/durations (GBO-R29).

    Projected source-agnostically from the canonical typed step (the ``Workout.steps`` JSON array);
    every field is optional because a step may prescribe by duration OR by target zone/power, and a
    rest step carries neither. Unknown extra keys on a step are dropped so no source-shaped field
    leaks (AUTH-R15).
    """

    intent: str | None = None
    duration_s: int | None = None
    target_low: float | None = None
    target_high: float | None = None
    target_unit: str | None = None


class PrescribedWorkout(BaseModel):
    """A canonical prescribed workout template projected for the read view (GBO-R29 / API-R32).

    Source-agnostic (no provider name, AUTH-R15). ``athlete_id`` is omitted (identity is the
    server-derived caller's, AUTH-R3); ``shared`` marks a NULL-athlete library template (TEN-R1).
    ``steps`` is the ordered typed step array (target zones/durations).
    """

    workout_id: str
    name: str
    sport: str | None = None
    shared: bool
    steps: list[WorkoutStepOut]


class PageOut(BaseModel):
    """The cursor-pagination page envelope (PAGE-R1): clamp + opaque next cursor."""

    limit: int
    next_cursor: str | None = None
    has_more: bool


class PrescribedWorkoutList(BaseModel):
    """The paginated prescribed-workout read response (PAGE-R1)."""

    data: list[PrescribedWorkout]
    page: PageOut


def _opt_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _opt_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _step_out(step: object) -> WorkoutStepOut:
    """Project one canonical step JSON object into the typed wire step (GBO-R29)."""
    s = step if isinstance(step, dict) else {}
    return WorkoutStepOut(
        intent=_opt_str(s.get("intent")),
        duration_s=_opt_int(s.get("duration_s")),
        target_low=_opt_float(s.get("target_low")),
        target_high=_opt_float(s.get("target_high")),
        target_unit=_opt_str(s.get("target_unit")),
    )


def prescribed(row: Workout) -> PrescribedWorkout:
    """Project a canonical :class:`Workout` row into the read-view wire shape (GBO-R29)."""
    steps = row.steps if isinstance(row.steps, list) else []
    return PrescribedWorkout(
        workout_id=str(row.workout_id),
        name=row.name,
        sport=row.sport,
        shared=row.athlete_id is None,
        steps=[_step_out(s) for s in steps],
    )


# --- GET /v1/planning/schedule — read-only plan/plan-day view ---------------------


class ScheduleDay(BaseModel):
    """One immutable plan day in the schedule view (GBO-R30b).

    Read-only: a ``plan_day`` is IMMUTABLE once generated (GBO-R30b/R31) and there is NO per-day
    mutation surface in v1 (``schedule_adjustment`` is post-v1, API-R32). ``workout_id`` is ``null``
    for a rest marker; ``rationale`` is the grounded coaching note when present.
    """

    plan_date: _dt.date
    intent: PlanDayIntent
    workout_id: str | None = None
    rationale: str | None = None


class Schedule(BaseModel):
    """The read-only schedule over the active plan for a ``from``/``to`` range (API-R32).

    A typed projection of the persisted :class:`Plan` + its immutable ``PlanDay`` rows for the
    server-derived athlete. There is NO active plan -> ``plan_id: null`` + empty ``days`` (a typed
    empty view, never a ``404``). Read-only in v1: no per-day mutation is surfaced here.
    """

    plan_id: str | None = None
    status: PlanStatus | None = None
    start_date: _dt.date | None = None
    end_date: _dt.date | None = None
    days: list[ScheduleDay]


def schedule_day(row: PlanDay) -> ScheduleDay:
    """Project one immutable :class:`PlanDay` row into the read-view wire shape (GBO-R30b)."""
    return ScheduleDay(
        plan_date=row.plan_date,
        intent=row.intent,
        workout_id=str(row.workout_id) if row.workout_id is not None else None,
        rationale=row.rationale,
    )


def schedule_of(plan: PlanRow | None, days: list[PlanDay]) -> Schedule:
    """Project the active plan (+ its in-range days) into the read-only :class:`Schedule` (API-R32).

    No active plan -> a typed empty schedule (``plan_id: null``), never a ``404`` (GBO-R30a).
    """
    if plan is None:
        return Schedule(days=[])
    return Schedule(
        plan_id=str(plan.plan_id),
        status=plan.status,
        start_date=plan.start_date,
        end_date=plan.end_date,
        days=[schedule_day(d) for d in days],
    )


__all__ = [
    "PHASE_GATED_BY_LOCALE",
    "PageOut",
    "PlanRequest",
    "PrescribedWorkout",
    "PrescribedWorkoutList",
    "ResponseLanguage",
    "Schedule",
    "ScheduleDay",
    "WorkoutStepOut",
    "phase_gated_response",
    "prescribed",
    "render_plan",
    "resolve_length",
    "resolve_locale",
    "schedule_day",
    "schedule_of",
]
