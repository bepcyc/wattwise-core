"""Wire shapes for the Goals router — request/response models + projection (API-R35).

The focused sibling of :mod:`wattwise_core.api.routers.goals` that owns ONLY the Pydantic
request/response models the Goals surface serializes plus the deterministic projection that turns a
canonical :class:`~wattwise_core.persistence.models.planning.Goal` ORM row into its wire shape
(QUAL-R9 size split). ``goals`` imports these back; NO route, NO dependency seam, and NO model call
lives here — only the wire vocabulary + the projection.

Boundary invariants encoded in the shapes:

- **AUTH-R3 / SCHEMA-R4** :class:`GoalCreateRequest` / :class:`GoalUpdateRequest` set
  ``additionalProperties:false`` so a forged/misnamed field (e.g. an injected ``athlete_id``) is a
  ``422`` rather than silently accepted — identity is NEVER a request field (server-derived).
- **API-R35** :class:`GoalOut` is exactly the spec's ``Goal`` shape
  ``{goal_id, title, goal_type, target_date, target_metric, target_value, sport, status,
  created_at}`` — it carries NO ``athlete_id`` (identity is the caller's).
- **GBO-R36/R38** the canonical vocabulary types (:class:`GoalType`, :class:`GoalTargetMetric`,
  :class:`GoalStatus`) are domain enums; ``sport`` is a registry code validated in the router.

Requirement IDs: API-R35, API-R32, AUTH-R3, AUTH-R18, GBO-R36, GBO-R37, GBO-R38, GBO-R39,
SCHEMA-R4, ERR-R6.
"""

from __future__ import annotations

import datetime as _dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wattwise_core.api.activity_schemas import Page
from wattwise_core.domain.enums import GoalStatus, GoalTargetMetric, GoalType
from wattwise_core.persistence.models import Goal

#: The list sort allow-list (PAGE-R2 / spec §8.13); default ``created_at``.
GoalSortKey = Literal["target_date", "created_at"]
GoalSortOrder = Literal["asc", "desc"]


class GoalOut(BaseModel):
    """The canonical ``Goal`` projected for the API (API-R35, doc 60:1342).

    Exactly the spec's ``Goal`` response shape. ``athlete_id`` is omitted (identity is the
    server-derived caller's, AUTH-R3); ``target_event``/``priority``/``notes``/``updated_at`` are
    NOT part of the documented wire shape and are not surfaced here, but ``target_event`` IS carried
    so an ``event`` goal renders its named target — both as the §8.13 list/get shape and a superset
    the create/patch echo. All fields beyond the §8.13 core are optional.
    """

    goal_id: str = Field(json_schema_extra={"format": "uuid"})
    title: str
    goal_type: GoalType
    target_date: _dt.date | None = None
    target_metric: GoalTargetMetric | None = None
    target_value: float | None = None
    sport: str
    status: GoalStatus
    created_at: _dt.datetime
    target_event: str | None = None


class GoalCreateRequest(BaseModel):
    """``POST /v1/goals`` body — the canonical ``Goal`` fields (API-R35 / GBO-R36).

    Identity is NOT a field — it is server-derived (AUTH-R3); a client cannot name the athlete it
    acts as. ``additionalProperties:false`` (SCHEMA-R4) rejects any unknown/forged property (e.g. an
    injected ``athlete_id``) with a ``422``. ``sport`` is validated against the runtime registry in
    the router (GBO-R38). ``status`` defaults to ``active`` — a new goal the agent plans toward; a
    caller MAY open one already terminal. ``target_value`` is interpreted in ``target_metric``'s
    canonical unit (GBO-R38); an ``event`` goal MAY omit metric/value and carry only the event+date.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=256)
    goal_type: GoalType
    sport: str = Field(min_length=1, max_length=64)
    status: GoalStatus = GoalStatus.ACTIVE
    target_event: str | None = Field(default=None, max_length=256)
    target_date: _dt.date | None = None
    target_metric: GoalTargetMetric | None = None
    target_value: float | None = None
    priority: int | None = Field(default=None, ge=0, le=32767)
    notes: str | None = Field(default=None, max_length=4096)


class GoalUpdateRequest(BaseModel):
    """``PATCH /v1/goals/{goal_id}`` body — the settable ``Goal`` fields (API-R35).

    Identity is NOT a field (AUTH-R3); ``additionalProperties:false`` (SCHEMA-R4) rejects a forged
    property. Every field is optional: the PATCH updates only the fields present (an omitted field
    is left untouched). Setting ``status`` to a terminal value is how a goal is CLOSED (GBO-R39) —
    the row is never deleted by a PATCH either. ``goal_type``/``sport``/``target_*`` may be fixed.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=256)
    goal_type: GoalType | None = None
    sport: str | None = Field(default=None, min_length=1, max_length=64)
    status: GoalStatus | None = None
    target_event: str | None = Field(default=None, max_length=256)
    target_date: _dt.date | None = None
    target_metric: GoalTargetMetric | None = None
    target_value: float | None = None
    priority: int | None = Field(default=None, ge=0, le=32767)
    notes: str | None = Field(default=None, max_length=4096)


class GoalList(BaseModel):
    """The paginated goals list response (PAGE-R1)."""

    data: list[GoalOut]
    page: Page


def goal_out(row: Goal) -> GoalOut:
    """Project a canonical :class:`Goal` row into the §8.13 wire shape (API-R35).

    Source-agnostic; ``athlete_id`` is never surfaced (identity is the caller's, AUTH-R3).
    ``target_value`` is coerced to ``float`` from the canonical numeric column.
    """
    return GoalOut(
        goal_id=str(row.goal_id),
        title=row.title,
        goal_type=row.goal_type,
        target_date=row.target_date,
        target_metric=row.target_metric,
        target_value=None if row.target_value is None else float(row.target_value),
        sport=row.sport,
        status=row.status,
        created_at=row.created_at,
        target_event=row.target_event,
    )


__all__ = [
    "GoalCreateRequest",
    "GoalList",
    "GoalOut",
    "GoalSortKey",
    "GoalSortOrder",
    "GoalUpdateRequest",
    "goal_out",
]
