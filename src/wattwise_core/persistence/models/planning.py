"""Planning cluster: workouts, plans, plan days, goals, schedule adjustments.

Owning requirements:

* ``workout`` — canonical prescription template (GBO-R29/R29a); NULL ``athlete_id``
  = shared library template (TEN-R1 dual-ownership). Ordered ``steps`` JSON.
* ``plan`` — ordered set of plan days (GBO-R30/R30a); immutable lineage JSON.
* ``plan_day`` — IMMUTABLE once generated (GBO-R30b/R31); key
  ``(plan_id, plan_date)``.
* ``goal`` — user-authored objective (GBO-R36/R37/R38/R39); no lineage/coverage.
* ``schedule_adjustment`` — ONE override on one plan day (GBO-R40/R41/R42); layers on
  top of an immutable plan day, never mutates it.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import Date, ForeignKey, Index, String, Text, UniqueConstraint, event
from sqlalchemy.orm import Mapped, mapped_column, validates
from sqlalchemy.orm.attributes import get_history

from wattwise_core.domain.enums import (
    AdjustmentOrigin,
    AdjustmentStatus,
    AdjustmentType,
    GoalStatus,
    GoalTargetMetric,
    GoalType,
    PlanDayIntent,
    PlanStatus,
)
from wattwise_core.domain.workout_steps import validate_workout_steps
from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.types import (
    enum_column,
    fk_uuid_column,
    json_column,
    numeric_column,
    pk_column,
    smallint_column,
)


class Workout(Base, TimestampMixin):
    """Canonical prescription template (GBO-R29a).

    NULL ``athlete_id`` = shared system/library template (TEN-R1 dual-ownership);
    non-NULL = athlete-owned. ``steps`` is an ordered, typed step array (GBO-R29).
    """

    __tablename__ = "workout"

    workout_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID | None] = fk_uuid_column("athlete.athlete_id", nullable=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    # NULL = sport-agnostic. Soft reference into the sport registry.
    sport: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("sport.sport_code"), nullable=True, index=True
    )
    steps: Mapped[list[dict[str, object]]] = json_column(nullable=False, default=list)

    @validates("steps")
    def _validate_steps(self, _key: str, value: object) -> list[dict[str, object]]:
        """Enforce the GBO-R29 step schema on EVERY ORM write of ``steps``.

        The step array MUST validate against the typed step schema before persistence
        (GBO-R29/R29a); an invalid step refuses the write loudly rather than landing an
        untyped prescription.
        """
        return validate_workout_steps(value)


class Goal(Base, TimestampMixin):
    """User-authored training objective (GBO-R36).

    Surrogate ``goal_id`` PK is the canonical upsert key; no source lineage/coverage.
    Closing sets a terminal ``status``, never deletes (GBO-R39).
    """

    __tablename__ = "goal"
    __table_args__ = (
        Index("ix_goal_athlete_status_target_date", "athlete_id", "status", "target_date"),
    )

    goal_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    # registry code (GBO-R38); NOT NULL. Soft reference into the sport registry.
    sport: Mapped[str] = mapped_column(
        String(64), ForeignKey("sport.sport_code"), nullable=False, index=True
    )
    goal_type: Mapped[GoalType] = enum_column(GoalType, nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    target_event: Mapped[str | None] = mapped_column(String(256), nullable=True)
    target_date: Mapped[_dt.date | None] = mapped_column(Date, nullable=True)
    target_metric: Mapped[GoalTargetMetric | None] = enum_column(GoalTargetMetric, nullable=True)
    target_value: Mapped[float | None] = numeric_column(nullable=True)
    status: Mapped[GoalStatus] = enum_column(GoalStatus, nullable=False)
    priority: Mapped[int | None] = smallint_column(nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class Plan(Base, TimestampMixin):
    """Ordered set of plan days over a date range (GBO-R30a).

    IMMUTABLE once generated (GBO-R31); changes flow via ``schedule_adjustment``.
    ``lineage`` records the agent/engine version + canonical input-snapshot ids.
    """

    __tablename__ = "plan"

    plan_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    goal_id: Mapped[uuid.UUID | None] = fk_uuid_column("goal.goal_id", nullable=True)
    start_date: Mapped[_dt.date] = mapped_column(Date, nullable=False)
    end_date: Mapped[_dt.date] = mapped_column(Date, nullable=False)
    status: Mapped[PlanStatus] = enum_column(PlanStatus, nullable=False)
    lineage: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)


class PlanDay(Base, TimestampMixin):
    """One day of a plan (GBO-R30b); IMMUTABLE once generated.

    Key ``(plan_id, plan_date)`` UNIQUE. ``workout_id`` NULL = rest marker.
    """

    __tablename__ = "plan_day"
    __table_args__ = (
        UniqueConstraint("plan_id", "plan_date", name="uq_plan_day_plan_date"),
        Index("ix_plan_day_plan_date", "plan_id", "plan_date"),
    )

    plan_day_id: Mapped[uuid.UUID] = pk_column()
    plan_id: Mapped[uuid.UUID] = fk_uuid_column("plan.plan_id", nullable=False)
    plan_date: Mapped[_dt.date] = mapped_column(Date, nullable=False)
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    workout_id: Mapped[uuid.UUID | None] = fk_uuid_column("workout.workout_id", nullable=True)
    intent: Mapped[PlanDayIntent] = enum_column(PlanDayIntent, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)


class ScheduleAdjustment(Base, TimestampMixin):
    """ONE override targeting exactly one plan day of one plan (GBO-R40).

    Surrogate ``schedule_adjustment_id`` PK is the canonical upsert key. Layers over
    an immutable plan day (GBO-R42); supersede sets a terminal status, never deletes.
    MUST reference an existing ``(plan_id, plan_date)``.
    """

    __tablename__ = "schedule_adjustment"
    __table_args__ = (
        Index(
            "ix_schedule_adjustment_plan_date_status",
            "plan_id",
            "plan_date",
            "status",
        ),
    )

    schedule_adjustment_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    plan_id: Mapped[uuid.UUID] = fk_uuid_column("plan.plan_id", nullable=False)
    plan_date: Mapped[_dt.date] = mapped_column(Date, nullable=False)
    adjustment_type: Mapped[AdjustmentType] = enum_column(AdjustmentType, nullable=False)
    target_plan_date: Mapped[_dt.date | None] = mapped_column(Date, nullable=True)
    replacement_workout_id: Mapped[uuid.UUID | None] = fk_uuid_column(
        "workout.workout_id", nullable=True
    )
    origin: Mapped[AdjustmentOrigin] = enum_column(AdjustmentOrigin, nullable=False)
    status: Mapped[AdjustmentStatus] = enum_column(AdjustmentStatus, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)


# --- GBO-R31 immutability guards -------------------------------------------------
# A generated plan is IMMUTABLE: athlete-/agent-requested changes are modeled as
# ``schedule_adjustment`` overrides, never in-place mutation, so the plan stays
# reproducible from its lineage. The guards are ORM-level events, so EVERY session
# flush enforces them — not just one polite write path.

# Plan columns that MAY change after generation: the lifecycle status transition
# (active -> completed | superseded) and the bookkeeping timestamp.
_PLAN_MUTABLE = frozenset({"status", "updated_at"})


@event.listens_for(PlanDay, "before_update")
def _forbid_plan_day_update(_mapper: object, _connection: object, target: PlanDay) -> None:
    """Refuse ANY update of a generated ``plan_day`` row (GBO-R31 immutability)."""
    raise ValueError(
        f"plan_day ({target.plan_id}, {target.plan_date}) is immutable once generated "
        "(GBO-R31): model the change as a schedule_adjustment override"
    )


@event.listens_for(Plan, "before_update")
def _forbid_plan_mutation(_mapper: object, _connection: object, target: Plan) -> None:
    """Refuse updates of a generated ``plan``'s content columns (GBO-R31).

    Only the lifecycle ``status`` (and ``updated_at``) may change; dates, lineage,
    goal, and ownership are frozen so the generated plan stays reproducible.
    """
    for column in Plan.__table__.columns.keys():  # noqa: SIM118 - Table.columns is not a plain dict
        if column in _PLAN_MUTABLE:
            continue
        history = get_history(target, column)
        if history.has_changes():
            raise ValueError(
                f"plan {target.plan_id} is immutable once generated (GBO-R31): "
                f"column {column!r} cannot be mutated in place"
            )


__all__ = ["Goal", "Plan", "PlanDay", "ScheduleAdjustment", "Workout"]
