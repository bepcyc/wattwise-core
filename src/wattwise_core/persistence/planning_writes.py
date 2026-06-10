"""The plan write seam (GBO-R31): generated plans land WITH auditable lineage.

The only sanctioned way to persist a generated plan. It enforces, at write time:

* ``lineage`` MUST name the generating agent/engine version AND the canonical
  input-snapshot identifiers (fitness state, signature) used to produce the plan, so
  the plan is reproducible and auditable (GBO-R31) — a lineage missing either is
  refused loudly (fail-closed), never silently defaulted;
* every ``plan_day`` lands inside the plan's date span with the plan's athlete
  (GBO-R30a/R30b) and the ``(plan_id, plan_date)`` uniqueness holds by construction.

Once written, the rows are guarded immutable by the ORM-level GBO-R31 events in
:mod:`wattwise_core.persistence.models.planning`.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.enums import PlanDayIntent, PlanStatus
from wattwise_core.persistence.models import Plan, PlanDay

# The lineage keys GBO-R31 makes mandatory for reproducibility.
_REQUIRED_LINEAGE_KEYS = ("engine_version", "input_snapshot_ids")


class PlanLineageError(ValueError):
    """A plan write whose lineage cannot reproduce the plan (refused, GBO-R31)."""


@dataclass(frozen=True, slots=True)
class PlanDaySpec:
    """One prescribed day of a plan to be generated (GBO-R30b shape)."""

    plan_date: _dt.date
    intent: PlanDayIntent
    workout_id: uuid.UUID | None = None  # None = rest day
    rationale: str | None = None


async def create_plan(
    session: AsyncSession,
    *,
    athlete_id: uuid.UUID,
    start_date: _dt.date,
    end_date: _dt.date,
    days: list[PlanDaySpec],
    lineage: dict[str, object],
    goal_id: uuid.UUID | None = None,
    status: PlanStatus = PlanStatus.ACTIVE,
) -> Plan:
    """Persist one generated plan + its days with mandatory lineage (GBO-R31).

    ``lineage`` MUST carry ``engine_version`` (the agent/engine that generated it) and
    ``input_snapshot_ids`` (the canonical input snapshots — fitness state, signature —
    it was generated from); anything less is refused so an unreproducible plan can
    never land. Day dates must be unique and inside ``[start_date, end_date]``.
    """
    _validate_lineage(lineage)
    if end_date < start_date:
        raise ValueError(f"plan end_date {end_date} precedes start_date {start_date}")
    seen: set[_dt.date] = set()
    for day in days:
        if not start_date <= day.plan_date <= end_date:
            raise ValueError(f"plan_day {day.plan_date} falls outside the plan span")
        if day.plan_date in seen:
            raise ValueError(f"duplicate plan_day {day.plan_date} (GBO-R31 uniqueness)")
        seen.add(day.plan_date)
    plan = Plan(
        athlete_id=athlete_id,
        goal_id=goal_id,
        start_date=start_date,
        end_date=end_date,
        status=status,
        lineage=dict(lineage),
    )
    session.add(plan)
    await session.flush()
    for day in days:
        session.add(
            PlanDay(
                plan_id=plan.plan_id,
                plan_date=day.plan_date,
                athlete_id=athlete_id,
                workout_id=day.workout_id,
                intent=day.intent,
                rationale=day.rationale,
            )
        )
    await session.flush()
    return plan


def _validate_lineage(lineage: dict[str, object]) -> None:
    """Refuse a lineage that cannot reproduce the plan (GBO-R31, fail-closed)."""
    missing = [key for key in _REQUIRED_LINEAGE_KEYS if not lineage.get(key)]
    if missing:
        raise PlanLineageError(
            f"plan lineage is missing {missing} (GBO-R31: a generated plan records the "
            "agent/engine version + canonical input-snapshot ids, or it is not written)"
        )


__all__ = ["PlanDaySpec", "PlanLineageError", "create_plan"]
