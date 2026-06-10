"""Goals router — the single-owner ``/v1/goals`` training-goal CRUD surface (doc 60 §8.13).

Serves the five ``/v1/goals`` endpoints (API-R35) over the canonical
:class:`~wattwise_core.persistence.models.planning.Goal` entity (API-R32 — every mutation backs a
real canonical entity, no orphan write): list (cursor-paginated + typed-filtered by
status/sport/from/to + typed-sorted on the ``target_date``/``created_at`` allow-list), create
(``201`` + ``Location``), get-one, patch, and delete (``204``). The agent plans TOWARD these goals
through the agent path (doc 50 / GBO-R38), reading the active rows this surface authors.

Boundary contract enforced here:

- **AUTH-R3 / AUTH-R18** the acting athlete identity is server-derived from the verified bearer
  token (never read from body/query/path); every read/write acts ONLY on that one server-derived
  id, and no request body carries a writable caller-identity field (SCHEMA-R4).
- **AUTH-R11** reads require the ``read`` scope; every mutation (POST/PATCH/DELETE) requires the
  ``write`` scope — a token without it is ``403 insufficient-scope`` (AUTH-R7).
- **API-R51** an unknown OR a foreign ``goal_id`` → ``404 not-found`` on every ``{goal_id}`` verb
  (existence-not-ownership; in OSS there is only the one owner, so the scoped read is the check).
- **GBO-R38** ``goal.sport`` is validated against the runtime sport registry; an unregistered code
  → ``422`` with ``errors[].code = "unknown_sport"`` BEFORE any write (no partial mutation).
- **GBO-R39** ``DELETE`` is a SOFT close: it sets a TERMINAL ``status`` (``abandoned``) and returns
  ``204`` — it NEVER hard-deletes the row, so goal history stays auditable. This reconciles
  API-R35's ``DELETE -> 204`` with GBO-R39's "Closing a goal MUST set status to a terminal value,
  never delete it".

The identity/scope/session/cursor-key/limiter dependencies are override seams the app factory wires
(FastAPI ``dependency_overrides``), mirroring the planning/athlete routers. No field is
source-shaped or carries a provider name (AUTH-R15).

Requirement IDs: API-R32, API-R35, API-R51, AUTH-R3, AUTH-R7, AUTH-R11, AUTH-R15, AUTH-R18,
GBO-R36, GBO-R37, GBO-R38, GBO-R39, PAGE-R1, PAGE-R2, PAGE-R3, PAGE-R5, PAGE-R7, SCHEMA-R4, ERR-R6.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy import asc, desc, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.activity_schemas import Page
from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.api.pagination import clamp_limit, decode_cursor, encode_cursor
from wattwise_core.api.problems import not_found, range_reversed
from wattwise_core.api.ratelimit import LimitClass, RateLimiter
from wattwise_core.api.routers.goals_schemas import (
    GoalCreateRequest,
    GoalList,
    GoalOut,
    GoalSortKey,
    GoalSortOrder,
    GoalUpdateRequest,
    goal_out,
)
from wattwise_core.domain.enums import GoalStatus
from wattwise_core.persistence.models import Goal, Sport

router = APIRouter(prefix="/v1/goals", tags=["goals"])

#: The sentinel a NULL ``target_date`` is COALESCED to for the keyset axis (PAGE-R7). The
#: ``target_date`` is nullable, so a raw keyset comparison over it returns NULL for a NULL-dated row
#: and DROPS it across pages; coalescing NULL -> this floor in BOTH the ORDER BY and the keyset
#: predicate makes a NULL-dated goal a concrete, comparable value that pages losslessly. It mirrors
#: the cursor side's epoch anchor in :func:`_keyset_datetime` so the two agree byte-for-byte.
_DATE_FLOOR = _dt.date.min


def _sort_axis(sort: str) -> Any:
    """Resolve the allow-listed sort key to its keyset axis expression (PAGE-R2/R7).

    ``created_at`` (the default) is a NOT-NULL datetime used directly. ``target_date`` is nullable,
    so it is COALESCED to :data:`_DATE_FLOOR` — NULL-dated goals then sort at the floor and page
    losslessly through the keyset (see :data:`_DATE_FLOOR`).
    """
    if sort == "target_date":
        return func.coalesce(Goal.target_date, _DATE_FLOOR)
    return Goal.created_at


# --- dependency seams (overridden by the app factory) ---------------------------


def require_read_scope() -> None:
    """Gate on the ``read`` scope (AUTH-R11); the app factory overrides it (fail-closed)."""
    raise ProblemError("insufficient-scope")  # pragma: no cover - replaced by the app factory


def require_write_scope() -> None:
    """Gate on the ``write`` scope (AUTH-R11); the app factory overrides it (fail-closed)."""
    raise ProblemError("insufficient-scope")  # pragma: no cover - replaced by the app factory


def current_athlete_id() -> str:
    """Server-derived acting athlete id (AUTH-R3); app factory overrides it (fail-closed)."""
    raise ProblemError("unauthenticated")  # pragma: no cover - replaced by the app factory


def current_session() -> AsyncSession:
    """Request-scoped DB session seam; the app factory overrides it (fail-closed)."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def rate_limiter() -> RateLimiter:
    """Provide the process-wide :class:`RateLimiter`; the app factory overrides it."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def cursor_signing_key() -> str:
    """Provide the cursor HMAC signing key; the app factory overrides it (PAGE-R5)."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


_Read = Depends(require_read_scope)
_Write = Depends(require_write_scope)
AthleteId = Annotated[str, Depends(current_athlete_id)]
Session = Annotated[AsyncSession, Depends(current_session)]
Limiter = Annotated[RateLimiter, Depends(rate_limiter)]
CursorKey = Annotated[str, Depends(cursor_signing_key)]


# --- helpers --------------------------------------------------------------------


def _uid(value: str) -> uuid.UUID:
    """Coerce the server-derived athlete id; an unparsable id reads as a not-found scope."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise not_found() from exc


def _goal_uid(value: str) -> uuid.UUID:
    """Coerce a path ``goal_id``; a malformed id resolves to ``404`` (API-R51, fail-closed)."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise not_found() from exc


def _unknown_sport() -> ProblemError:
    """A ``422 validation-error`` for an unregistered sport code (GBO-R38; no new type)."""
    return ProblemError(
        "validation-error",
        errors=[FieldError(code="unknown_sport", message="", pointer="/sport")],
    )


async def _sport_exists(session: AsyncSession, sport_code: str) -> bool:
    """Whether ``sport_code`` is a registered sport (GBO-R38 runtime registry)."""
    return await session.get(Sport, sport_code) is not None


async def _load_goal(session: AsyncSession, athlete_id: str, goal_id: str) -> Goal:
    """Load the owner's goal by id, or fail closed ``404`` (API-R51 existence-not-ownership).

    Scopes the lookup to the server-derived athlete (AUTH-R3): an unknown id OR a foreign
    athlete's id both miss and raise ``404 not-found`` — a foreign goal is never disclosed.
    """
    stmt = select(Goal).where(
        Goal.goal_id == _goal_uid(goal_id), Goal.athlete_id == _uid(athlete_id)
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise not_found()
    return row


def _cursor_params(
    *,
    status_f: GoalStatus | None,
    sport: str | None,
    frm: _dt.date | None,
    to: _dt.date | None,
    sort: str,
    order: str,
) -> dict[str, str]:
    """The filter/sort fingerprint a cursor is bound to (PAGE-R6); identity-only fields."""
    return {
        "status": status_f.value if status_f else "",
        "sport": sport or "",
        "from": frm.isoformat() if frm else "",
        "to": to.isoformat() if to else "",
        "sort": sort,
        "order": order,
    }


async def _query_goals(
    session: AsyncSession,
    athlete_id: str,
    *,
    status: GoalStatus | None,
    sport: str | None,
    frm: _dt.date | None,
    to: _dt.date | None,
    sort: str,
    order: str,
    cursor: str | None,
    key: str,
    limit: int,
) -> list[Goal]:
    """Keyset-paginated, athlete-scoped, typed-filtered + typed-sorted goal query (PAGE-R7).

    Scoped to the server-derived athlete (AUTH-R3). Optional typed filters narrow by status, sport,
    and a ``target_date`` window. Ordered by the allow-listed sort axis, ALWAYS tie-broken on the
    ``goal_id`` keyset so the opaque signed cursor pages deterministically (PAGE-R7).
    """
    axis = _sort_axis(sort)
    direction = desc if order == "desc" else asc
    clauses: list[Any] = [Goal.athlete_id == _uid(athlete_id)]
    if status is not None:
        clauses.append(Goal.status == status)
    if sport is not None:
        clauses.append(Goal.sport == sport)
    if frm is not None:
        clauses.append(Goal.target_date >= frm)
    if to is not None:
        clauses.append(Goal.target_date <= to)
    if cursor is not None:
        params = _cursor_params(
            status_f=status, sport=sport, frm=frm, to=to, sort=sort, order=order
        )
        c_time, c_id = decode_cursor(cursor, params=params, key=key)
        keyset = tuple_(axis, Goal.goal_id)
        anchor = (_cursor_axis_value(sort, c_time), uuid.UUID(c_id))
        clauses.append(keyset < anchor if order == "desc" else keyset > anchor)
    stmt = (
        select(Goal)
        .where(*clauses)
        .order_by(direction(axis), direction(Goal.goal_id))
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


def _cursor_axis_value(sort: str, c_time: _dt.datetime) -> Any:
    """Lift the cursor's stored datetime back onto the sort axis' native type (PAGE-R7)."""
    return c_time.date() if sort == "target_date" else c_time


def _keyset_datetime(row: Goal, sort: str) -> _dt.datetime:
    """The row's keyset datetime for the active sort axis (PAGE-R7).

    A ``target_date`` sort lifts the (nullable) date onto the UTC datetime cursor axis; a row with
    no ``target_date`` anchors at the epoch so it sorts stably at the low end. ``created_at`` is
    already a UTC datetime.
    """
    if sort == "target_date":
        d = row.target_date or _dt.date.min
        return _dt.datetime.combine(d, _dt.time.min, _dt.UTC)
    return row.created_at


# --- §8.13 list -----------------------------------------------------------------


@router.get("", response_model=GoalList, operation_id="listGoals", dependencies=[_Read])
async def list_goals(
    session: Session,
    athlete_id: AthleteId,
    key: CursorKey,
    limiter: Limiter,
    *,
    status: Annotated[GoalStatus | None, Query()] = None,
    sport: Annotated[str | None, Query()] = None,
    frm: Annotated[_dt.date | None, Query(alias="from")] = None,
    to: Annotated[_dt.date | None, Query()] = None,
    sort: Annotated[GoalSortKey, Query()] = "created_at",
    order: Annotated[GoalSortOrder, Query()] = "desc",
    limit: Annotated[int, Query(ge=1, json_schema_extra={"maximum": 200})] = 50,
    cursor: Annotated[str | None, Query()] = None,
) -> GoalList:
    """List the owner's training goals, cursor-paginated + typed-filtered + typed-sorted (API-R35).

    Requires the ``read`` scope (AUTH-R11) and debits the per-athlete ``read`` bucket (LIMIT-R1)
    keyed on the server-derived id (AUTH-R3). Filters by ``status``/``sport``/``from``/``to`` (the
    ``target_date`` window); ``from > to`` → ``422`` (ERR-R6). Sort is the ``{target_date,
    created_at}`` allow-list (PAGE-R2; an off-list value is rejected by the typed enum). The
    ``limit`` is clamped to ``[1, 200]`` (PAGE-R3); the opaque signed cursor pages keyset (PAGE-R7).
    """
    limiter.check(athlete_id, LimitClass.READ)
    if frm is not None and to is not None and frm > to:
        raise range_reversed("from")
    bounded = clamp_limit(int(limit))
    rows = await _query_goals(
        session, athlete_id, status=status, sport=sport, frm=frm, to=to,
        sort=sort, order=order, cursor=cursor, key=key, limit=bounded + 1,
    )
    has_more = len(rows) > bounded
    page_rows = rows[:bounded]
    last = page_rows[-1] if (has_more and page_rows) else None
    nxt = (
        encode_cursor(
            _keyset_datetime(last, sort),
            str(last.goal_id),
            params=_cursor_params(
                status_f=status, sport=sport, frm=frm, to=to, sort=sort, order=order
            ),
            key=key,
        )
        if last is not None
        else None
    )
    return GoalList(
        data=[goal_out(r) for r in page_rows],
        page=Page(limit=bounded, next_cursor=nxt, has_more=has_more),
    )


# --- §8.13 create ---------------------------------------------------------------


@router.post(
    "",
    response_model=GoalOut,
    status_code=status.HTTP_201_CREATED,
    operation_id="createGoal",
    dependencies=[_Write],
)
async def create_goal(
    body: GoalCreateRequest,
    response: Response,
    session: Session,
    athlete_id: AthleteId,
    limiter: Limiter,
) -> GoalOut:
    """Create a training goal backed by the canonical entity → ``201`` + ``Location`` (API-R35).

    Requires the ``write`` scope (AUTH-R11) and debits the per-athlete ``mutating`` bucket
    (LIMIT-R2). The ``sport`` is validated against the runtime registry (GBO-R38); an unregistered
    code is rejected ``422 unknown_sport`` BEFORE any write (no partial mutation). The row is keyed
    by the server-derived ``athlete_id`` (AUTH-R3, never a client value) and a fresh ``goal_id``;
    the ``Location`` header points at the created resource (API-R35).
    """
    limiter.check(athlete_id, LimitClass.MUTATING)
    if not await _sport_exists(session, body.sport):
        raise _unknown_sport()
    row = Goal(
        athlete_id=_uid(athlete_id),
        sport=body.sport,
        goal_type=body.goal_type,
        title=body.title,
        status=body.status,
        target_event=body.target_event,
        target_date=body.target_date,
        target_metric=body.target_metric,
        target_value=body.target_value,
        priority=body.priority,
        notes=body.notes,
    )
    session.add(row)
    await session.flush()
    response.headers["Location"] = f"/v1/goals/{row.goal_id}"
    return goal_out(row)


# --- §8.13 get one --------------------------------------------------------------


@router.get("/{goal_id}", response_model=GoalOut, operation_id="getGoal", dependencies=[_Read])
async def get_goal(
    goal_id: str, session: Session, athlete_id: AthleteId, limiter: Limiter
) -> GoalOut:
    """Read one of the owner's goals; an unknown/foreign id → ``404 not-found`` (API-R51)."""
    limiter.check(athlete_id, LimitClass.READ)
    row = await _load_goal(session, athlete_id, goal_id)
    return goal_out(row)


# --- §8.13 patch ----------------------------------------------------------------


@router.patch(
    "/{goal_id}", response_model=GoalOut, operation_id="updateGoal", dependencies=[_Write]
)
async def update_goal(
    goal_id: str,
    body: GoalUpdateRequest,
    session: Session,
    athlete_id: AthleteId,
    limiter: Limiter,
) -> GoalOut:
    """Update only the supplied fields of one goal (API-R35); an unknown id → ``404`` (API-R51).

    Requires the ``write`` scope and debits the ``mutating`` bucket. A ``sport`` change is checked
    against the registry (GBO-R38) before any mutation; an unregistered code is a 422 unknown_sport.
    Setting ``status`` to a terminal value CLOSES the goal (GBO-R39) without deleting it. Acts on
    the server-derived owner id only (AUTH-R3).
    """
    limiter.check(athlete_id, LimitClass.MUTATING)
    row = await _load_goal(session, athlete_id, goal_id)
    if body.sport is not None and not await _sport_exists(session, body.sport):
        raise _unknown_sport()
    fields = body.model_dump(exclude_unset=True)
    for name, value in fields.items():
        setattr(row, name, value)
    await session.flush()
    return goal_out(row)


# --- §8.13 delete (soft close, GBO-R39) -----------------------------------------


@router.delete(
    "/{goal_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    operation_id="deleteGoal",
    dependencies=[_Write],
)
async def delete_goal(
    goal_id: str, session: Session, athlete_id: AthleteId, limiter: Limiter
) -> Response:
    """Soft-close a goal to a terminal status → ``204`` (API-R35 / GBO-R39); unknown id → ``404``.

    Requires the ``write`` scope and debits the ``mutating`` bucket. GBO-R39 mandates that closing a
    goal sets a TERMINAL ``status`` and NEVER deletes the row (goal history is auditable), while
    API-R35 specifies ``DELETE -> 204``: this endpoint reconciles both — it sets ``status =
    abandoned`` (a terminal value) and returns ``204`` with no body. The row survives, so a later
    GET still resolves it (200), not a 404. Acts ONLY on the server-derived owner id (AUTH-R3); an
    unknown/foreign id → ``404 not-found`` (API-R51).
    """
    limiter.check(athlete_id, LimitClass.MUTATING)
    row = await _load_goal(session, athlete_id, goal_id)
    row.status = GoalStatus.ABANDONED
    await session.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = [
    "GoalCreateRequest",
    "GoalList",
    "GoalOut",
    "GoalUpdateRequest",
    "current_athlete_id",
    "current_session",
    "cursor_signing_key",
    "rate_limiter",
    "require_read_scope",
    "require_write_scope",
    "router",
]

#: OpenAPI security metadata (DOC-R3): the scopes this seam gate requires.
require_read_scope.required_scopes = ('read',)  # type: ignore[attr-defined]

#: OpenAPI security metadata (DOC-R3): the scopes this seam gate requires.
require_write_scope.required_scopes = ('write',)  # type: ignore[attr-defined]
