"""Read the athlete's ACTIVE canonical goals into the agent inputs (GBO-R38 / API-R32 / API-R35).

doc 20 GBO-R38 assigns goal-aware PLANNING to the agent ("the store enforces typing only; goal-aware
PLANNING ... is owned by the agent/analytics specs, which read this entity"); doc 60 API-R32 /
API-R35 confirm "goal-aware plan generation reads those goals through the agent path". This leaf is
that read: it loads the server-derived athlete's ACTIVE :class:`~wattwise_core.persistence.models.
planning.Goal` rows and projects each to a plain, serializable dict the run inputs carry into the
plan / load-review compose context (the engine threads it through ``build_inputs`` into the state).

It is read-only over the canonical store and athlete-scoped (AGT-SEC-R1): identity is the
server-derived ``athlete_id`` only, never a model/tool value. The projection drops nothing
analytic — a goal is user-authored INTENT (MEM-R1), never a canonical number — so the inputs steer
the draft, not grounding. The query follows the canonical ``goal (athlete_id, status, target_date)``
index (doc 20 §3.9): ACTIVE goals for the athlete, nearest ``target_date`` first.

Cited requirements: GBO-R36, GBO-R38, GBO-R39, API-R32, API-R35, AGT-SEC-R1, MEM-R1.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import asc, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.enums import GoalStatus
from wattwise_core.persistence.models import Goal


async def active_goals_for(session: AsyncSession, athlete_id: str) -> list[dict[str, Any]]:
    """Load the athlete's ACTIVE canonical goals, projected for the agent inputs (GBO-R38).

    Athlete-scoped (AGT-SEC-R1) and ACTIVE-only (terminal goals are not planned toward, GBO-R39),
    ordered by nearest ``target_date`` first (the canonical §3.9 index axis; NULL target dates sort
    last). A malformed athlete id yields no goals rather than raising — the deliverable still runs,
    just without goal context (fail-soft for an input layer, never fail-open identity).
    """
    try:
        aid = uuid.UUID(athlete_id)
    except (ValueError, AttributeError):  # pragma: no cover - athlete_id is server-derived/valid
        return []
    stmt = (
        select(Goal)
        .where(Goal.athlete_id == aid, Goal.status == GoalStatus.ACTIVE)
        .order_by(asc(Goal.target_date.is_(None)), asc(Goal.target_date), asc(Goal.goal_id))
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [_project(row) for row in rows]


def _project(row: Goal) -> dict[str, Any]:
    """Project one active :class:`Goal` to a plain serializable dict for the run inputs (GBO-R36).

    Only the typed canonical fields a planner reasons over — no ``athlete_id`` (identity is the
    caller's), no source lineage (a goal carries none). Enum/numeric values are coerced to plain
    str/float so the dict serialises into the checkpointed state cleanly.
    """
    return {
        "title": row.title,
        "goal_type": row.goal_type.value,
        "sport": row.sport,
        "target_event": row.target_event,
        "target_date": row.target_date.isoformat() if row.target_date is not None else None,
        "target_metric": row.target_metric.value if row.target_metric is not None else None,
        "target_value": None if row.target_value is None else float(row.target_value),
    }


__all__ = ["active_goals_for"]
