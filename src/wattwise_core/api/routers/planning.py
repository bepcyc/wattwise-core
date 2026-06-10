"""Planning router — the agent-backed plan surface + the read-only plan views (doc 60 §planning).

Serves the three ``/v1/planning/*`` endpoints (doc 60 §planning, API-R32); the wire shapes +
projections live in the focused sibling :mod:`planning_schemas` (QUAL-R9 size split):

- **``POST /v1/planning/workouts``** (scope ``agent``, AUTH-R13) — the agent-backed multi-day PLAN
  generation. It reaches the injected :class:`PlanningEngine` (the same ``GraphAgentEngine`` the
  ``/v1/agent`` surface drives, ARCH-R21) through the typed ``plan_deliverable`` seam and renders
  the status-discriminated :class:`~wattwise_core.api.routers.agent_schemas.AgentAskResponse`
  union — the approval-gated PLAN finalizes ``awaiting_approval`` carrying ``interrupt_id`` + plan
  body, which the EXISTING ``POST /v1/agent/threads/{thread_id}/decision`` endpoint
  approves/edits/rejects to resume the SAME durable thread (API-R12a / CKPT-R9). Plan generation is
  PHASE-GATED (doc 60 §phase-gating): an OSS deployment with no LLM cannot generate a plan (the
  wired engine does not implement ``plan_deliverable``), so this fails closed to a ``degraded`` "not
  yet available" answer (RUN-R4.1). The endpoint requires the ``agent`` scope and debits the
  per-athlete ``agent`` rate bucket (LIMIT-R2).
- **``GET /v1/planning/workouts``** (scope ``read``, AUTH-R11) — the cursor-paginated read view over
  the persisted canonical :class:`~wattwise_core.persistence.models.planning.Workout` library
  (athlete-owned + NULL-athlete shared templates, TEN-R1), projected source-agnostically into
  :class:`~wattwise_core.api.routers.planning_schemas.PrescribedWorkout` (target zones/durations,
  GBO-R29).
- **``GET /v1/planning/schedule``** (scope ``read``, AUTH-R11) — the read-only
  :class:`~wattwise_core.api.routers.planning_schemas.Schedule` over the persisted
  :class:`~wattwise_core.persistence.models.planning.Plan` / ``PlanDay`` for a typed ``from``/``to``
  range. Read-only in v1: there is NO per-day mutation surface here (a ``schedule_adjustment`` is
  post-v1, API-R32), so the view never mutates an immutable plan day (GBO-R30b/R42).

Acting athlete identity is server-derived (AUTH-R3 / AUTH-R18) from the bearer token via
:func:`current_athlete_id`; the client never supplies it, and every read/generate acts ONLY on that
one server-derived id. The identity/scope/engine/session/cursor-key dependencies are override seams
the app factory wires (FastAPI ``dependency_overrides``), mirroring the agent/performance/activities
routers. No field is source-shaped or carries a provider name (AUTH-R15); the agent response carries
NO billing/budget/model machinery (API-R11c) and is server-side sanitized at the boundary (API-R13).

Requirement IDs: API-R32, API-R12a, API-R11a, API-R11c, API-R13, AUTH-R3, AUTH-R11, AUTH-R13,
AUTH-R15, AUTH-R18, GBO-R29, GBO-R30a, GBO-R30b, CKPT-R9, LIMIT-R2, PAGE-R1, PAGE-R3, PAGE-R7,
RUN-R4.1, SCHEMA-R4.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Annotated, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, Header, Query, Request
from sqlalchemy import asc, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.deliverables import Plan as PlanDeliverable
from wattwise_core.api.errors import ProblemError, resolve_trace_id
from wattwise_core.api.pagination import clamp_limit, decode_cursor, encode_cursor
from wattwise_core.api.problems import not_found, range_reversed
from wattwise_core.api.ratelimit import LimitClass, RateLimiter
from wattwise_core.api.routers.agent_routes import attached_entitlement, persisted_locale
from wattwise_core.api.routers.agent_schemas import AgentAskResponse
from wattwise_core.api.routers.planning_schemas import (
    PageOut,
    PlanRequest,
    PrescribedWorkoutList,
    Schedule,
    phase_gated_response,
    prescribed,
    render_plan,
    resolve_length,
    resolve_locale,
    schedule_of,
)
from wattwise_core.domain.enums import PlanStatus
from wattwise_core.entitlement import Entitlements
from wattwise_core.persistence.models import Plan as PlanRow
from wattwise_core.persistence.models import PlanDay, Workout

router = APIRouter(prefix="/v1/planning", tags=["planning"])


# --- engine seam (injected; reached only through this Protocol, ARCH-R21) --------


@runtime_checkable
class PlanningEngine(Protocol):
    """The plan-generation seam this router drives (the multi-day PLAN deliverable projection).

    The wired concrete engine is the SAME ``GraphAgentEngine`` the ``/v1/agent`` surface uses
    (ARCH-R21): this router reaches it ONLY through this typed ``plan_deliverable`` seam, never the
    in-flight graph. ``athlete_id`` is passed server-derived (AUTH-R3) and never trusted from the
    model. The returned :class:`~wattwise_core.agent.deliverables.Plan` already carries the engine's
    grounded multi-day body; with the approval gate it finalizes ``awaiting_approval`` carrying the
    ``interrupt_id`` the EXISTING ``POST /v1/agent/threads/{id}/decision`` endpoint consumes
    (CKPT-R9). The OSS no-LLM engine does NOT implement this method (plan generation is
    phase-gated); the router detects its absence and fails closed to a degraded answer (RUN-R4.1).
    """

    async def plan_deliverable(
        self,
        *,
        athlete_id: str,
        request: str,
        thread_id: str | None,
        locale: str,
        response_length: str,
        requires_approval: bool,
        entitlement: Entitlements | None = None,
    ) -> PlanDeliverable: ...


# --- dependency seams (overridden by the app factory) ----------------------------


def require_agent_scope() -> None:
    """Gate plan GENERATION on the ``agent`` scope (AUTH-R13); app factory overrides it."""
    raise ProblemError("insufficient-scope")  # pragma: no cover - replaced by the app factory


def require_read_scope() -> None:
    """Gate the plan READ views on the ``read`` scope (AUTH-R11); app factory overrides it."""
    raise ProblemError("insufficient-scope")  # pragma: no cover - replaced by the app factory


def current_athlete_id() -> str:
    """Server-derived acting athlete id (AUTH-R3); app factory overrides it (fail-closed)."""
    raise ProblemError("unauthenticated")  # pragma: no cover - replaced by the app factory


def planning_engine() -> PlanningEngine:
    """Provide the request-scoped :class:`PlanningEngine`; the app factory overrides it."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def current_session() -> AsyncSession:
    """Request-scoped DB session seam; the app factory overrides it (fail-closed)."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def rate_limiter() -> RateLimiter:
    """Provide the process-wide :class:`RateLimiter`; the app factory overrides it."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def cursor_signing_key() -> str:
    """Provide the cursor HMAC signing key; the app factory overrides it (PAGE-R5)."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


_Agent = Depends(require_agent_scope)
_Read = Depends(require_read_scope)
AthleteId = Annotated[str, Depends(current_athlete_id)]
Engine = Annotated[PlanningEngine, Depends(planning_engine)]
Session = Annotated[AsyncSession, Depends(current_session)]
Limiter = Annotated[RateLimiter, Depends(rate_limiter)]
PersistedLocale = Annotated[str | None, Depends(persisted_locale)]
CursorKey = Annotated[str, Depends(cursor_signing_key)]


def _uid(value: str) -> uuid.UUID:
    """Coerce the server-derived athlete id; an unparsable id reads as a not-found scope."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise not_found() from exc


# --- POST /v1/planning/workouts — agent-backed plan generation (API-R32 / API-R12a) ---


@router.post(
    "/workouts",
    response_model=AgentAskResponse,
    dependencies=[_Agent],
    operation_id="generatePlannedWorkouts",
)
async def generate_workouts(
    request: Request,
    body: PlanRequest,
    engine: Engine,
    athlete_id: AthleteId,
    limiter: Limiter,
    stored_locale: PersistedLocale,
    accept_language: Annotated[str | None, Header()] = None,
) -> AgentAskResponse:
    """Generate a multi-day grounded training PLAN, approval-gated (API-R32 / API-R12a / COACH-R2).

    Requires the ``agent`` scope (AUTH-R13) and debits the per-athlete ``agent`` rate bucket
    (``20/min``, LIMIT-R2) keyed by the server-derived id (AUTH-R3). Drives the injected
    :class:`PlanningEngine` with the server-derived ``athlete_id`` (never a client value) and
    renders the status-discriminated :class:`AgentAskResponse`: the approval-gated plan finalizes
    ``awaiting_approval`` carrying ``interrupt_id`` + the grounded plan body so the EXISTING
    ``POST /v1/agent/threads/{thread_id}/decision`` endpoint approves/edits/rejects it to resume the
    SAME durable thread (CKPT-R9). Plan generation is PHASE-GATED (doc 60 §phase-gating): an OSS
    deployment with no LLM cannot generate, so it fails closed to a typed ``degraded`` "not yet
    available" answer (RUN-R4.1). The body carries NO billing/model machinery (API-R11c) and is
    server-side sanitized (API-R13).
    """
    limiter.check(athlete_id, LimitClass.AGENT)
    trace_id = resolve_trace_id(request)
    locale = resolve_locale(body, accept_language, stored_locale)
    generate = getattr(engine, "plan_deliverable", None)
    if generate is None:
        # Phase-gated: the wired (no-LLM) engine cannot generate a plan (RUN-R4.1).
        return phase_gated_response(locale, body.thread_id, trace_id)
    plan = await generate(
        athlete_id=athlete_id,
        request=body.request,
        thread_id=body.thread_id,
        locale=locale,
        response_length=resolve_length(body),
        requires_approval=True,
        entitlement=attached_entitlement(request),
    )
    return render_plan(plan, trace_id, locale)


# --- GET /v1/planning/workouts — paginated read view over the canonical library (API-R32) ---


def _workout_cursor_params() -> dict[str, str]:
    """The fixed filter fingerprint the workout cursor is bound to (PAGE-R6); identity-only."""
    return {"view": "planning_workouts"}


async def _query_workouts(
    session: AsyncSession, athlete_id: str, *, cursor: str | None, key: str, limit: int
) -> list[Workout]:
    """Keyset-paginated workout query over owned + shared templates, tie-broken on id (PAGE-R7).

    Returns the athlete's own templates AND the NULL-athlete shared library (TEN-R1), ordered by
    ``(created_at, workout_id)`` so the opaque keyset cursor pages deterministically (PAGE-R7).
    """
    owned = or_(Workout.athlete_id == _uid(athlete_id), Workout.athlete_id.is_(None))
    clauses = [owned]
    if cursor is not None:
        c_time, c_id = decode_cursor(cursor, params=_workout_cursor_params(), key=key)
        clauses.append(tuple_(Workout.created_at, Workout.workout_id) > (c_time, uuid.UUID(c_id)))
    stmt = (
        select(Workout)
        .where(*clauses)
        .order_by(asc(Workout.created_at), asc(Workout.workout_id))
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


@router.get(
    "/workouts",
    response_model=PrescribedWorkoutList,
    dependencies=[_Read],
    operation_id="listPlannedWorkouts",
)
async def list_workouts(
    session: Session,
    athlete_id: AthleteId,
    key: CursorKey,
    limiter: Limiter,
    *,
    limit: Annotated[int, Query(ge=1, json_schema_extra={"maximum": 200})] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> PrescribedWorkoutList:
    """List the canonical prescribed-workout library, cursor-paginated (API-R32 / PAGE-R1).

    Requires the ``read`` scope (AUTH-R11) and debits the per-athlete ``read`` rate bucket
    (``120/min``, LIMIT-R1) keyed by the server-derived id (AUTH-R3 / LIMIT-R6) — the read views
    are rate-limited like every other per-subject surface, not just the generation path. Reads
    ONLY the server-derived athlete's own templates plus the shared NULL-athlete library (TEN-R1);
    each is projected source-agnostically into :class:`PrescribedWorkout` (target zones/durations,
    GBO-R29). ``limit`` is clamped to ``[1, 200]`` (PAGE-R3); the opaque signed ``cursor`` pages the
    ``(created_at, workout_id)`` keyset (PAGE-R7).
    """
    limiter.check(athlete_id, LimitClass.READ)
    bounded = clamp_limit(int(limit))  # PAGE-R3 clamp; never unbounded / offset
    rows = await _query_workouts(session, athlete_id, cursor=cursor, key=key, limit=bounded + 1)
    has_more = len(rows) > bounded
    page_rows = rows[:bounded]
    last = page_rows[-1] if (has_more and page_rows) else None
    nxt = (
        encode_cursor(
            last.created_at, str(last.workout_id), params=_workout_cursor_params(), key=key
        )
        if last is not None
        else None
    )
    return PrescribedWorkoutList(
        data=[prescribed(r) for r in page_rows],
        page=PageOut(limit=bounded, next_cursor=nxt, has_more=has_more),
    )


# --- GET /v1/planning/schedule — read-only plan/plan-day view (API-R32) ----------


async def _active_plan(session: AsyncSession, athlete_id: str) -> PlanRow | None:
    """The athlete's most-recent ACTIVE plan, or ``None`` (the schedule view's anchor, GBO-R30a)."""
    stmt = (
        select(PlanRow)
        .where(PlanRow.athlete_id == _uid(athlete_id), PlanRow.status == PlanStatus.ACTIVE)
        .order_by(PlanRow.start_date.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def _plan_days(
    session: AsyncSession, plan_id: uuid.UUID, frm: _dt.date, to: _dt.date
) -> list[PlanDay]:
    """The plan's immutable days within ``[from, to]``, date-ordered (read-only, GBO-R30b)."""
    stmt = (
        select(PlanDay)
        .where(PlanDay.plan_id == plan_id, PlanDay.plan_date >= frm, PlanDay.plan_date <= to)
        .order_by(asc(PlanDay.plan_date))
    )
    return list((await session.execute(stmt)).scalars().all())


@router.get(
    "/schedule",
    response_model=Schedule,
    dependencies=[_Read],
    operation_id="getPlanningSchedule",
)
async def get_schedule(
    session: Session,
    athlete_id: AthleteId,
    limiter: Limiter,
    *,
    frm: Annotated[_dt.date, Query(alias="from", description="Inclusive local start date.")],
    to: Annotated[_dt.date, Query(description="Inclusive local end date.")],
) -> Schedule:
    """Read the active plan's immutable schedule for a date range (API-R32; read-only in v1).

    Requires the ``read`` scope (AUTH-R11) and debits the per-athlete ``read`` rate bucket
    (``120/min``, LIMIT-R1) keyed by the server-derived id (AUTH-R3 / LIMIT-R6). Projects the
    server-derived athlete's most-recent ACTIVE :class:`Plan` and its immutable ``PlanDay`` rows in
    ``[from, to]`` (GBO-R30b). No active plan -> a typed empty schedule (``plan_id: null``), never a
    ``404``. ``from > to`` -> ``422`` (ERR-R6). There is NO per-day mutation here — a
    ``schedule_adjustment`` is post-v1 (API-R32).
    """
    limiter.check(athlete_id, LimitClass.READ)
    if frm > to:
        raise range_reversed("from")
    plan = await _active_plan(session, athlete_id)
    days = await _plan_days(session, plan.plan_id, frm, to) if plan is not None else []
    return schedule_of(plan, days)


__all__ = [
    "PlanRequest",
    "PlanningEngine",
    "PrescribedWorkoutList",
    "Schedule",
    "current_athlete_id",
    "current_session",
    "cursor_signing_key",
    "planning_engine",
    "rate_limiter",
    "require_agent_scope",
    "require_read_scope",
    "router",
]

#: OpenAPI security metadata (DOC-R3): the scopes this seam gate requires.
require_agent_scope.required_scopes = ('agent',)  # type: ignore[attr-defined]

#: OpenAPI security metadata (DOC-R3): the scopes this seam gate requires.
require_read_scope.required_scopes = ('read',)  # type: ignore[attr-defined]
