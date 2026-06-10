"""Row→wire mapping + local-date resolution for the activities router (doc 60 §13).

Factored out of :mod:`wattwise_core.api.routers.activities` so the router stays within the
QUAL-R9 module-size ceiling. These are the source-agnostic helpers the activity endpoints
compose: the canonical ``Activity`` → ``ActivitySummary`` projection, the athlete-LOCAL
day resolution for display (GBO-R33/R35, §3.8), the owner load (whose reference tz is
authoritative for ``local_date``, fail-closed per CFG-R6), and the nullable-numeric coercion.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.activity_schemas import ActivitySummary
from wattwise_core.api.problems import not_found
from wattwise_core.persistence.localdate import project_local_date
from wattwise_core.persistence.models import Activity, Athlete


def f(value: object) -> float | None:
    """Coerce a nullable numeric column to ``float | None`` for the wire shape."""
    return None if value is None else float(value)  # type: ignore[arg-type]


def local_date_of(act: Activity, athlete: Athlete) -> _dt.date:
    """The activity's athlete-LOCAL day (§3.8, GBO-R35) for display — NOT the UTC date.

    Prefers the persisted ``activity.local_date`` (the ingest projection); recomputes from
    ``start_time`` + the athlete's effective-dated reference tz when absent (GBO-R34), so a
    pre-projection row still surfaces the correct local day rather than the UTC ``.date()``.
    """
    return project_local_date(act.start_time, athlete, prior_local_date=act.local_date)


def summary(act: Activity, local_date: _dt.date) -> ActivitySummary:
    """Project a canonical ``Activity`` row onto the ``ActivitySummary`` wire shape (§13)."""
    return ActivitySummary(
        activity_id=str(act.activity_id),
        local_date=local_date,
        sport=act.sport,
        start_time=act.start_time,
        elapsed_time_s=act.elapsed_time_s,
        moving_time_s=act.moving_time_s,
        distance_m=f(act.distance_m),
        avg_power_w=f(act.avg_power_w),
        has_power=act.has_power,
        has_hr=act.has_hr,
        has_gps=act.has_gps,
        has_cadence=act.has_cadence,
    )


async def owner_or_not_found(session: AsyncSession, athlete_id: str) -> Athlete:
    """Load the owner whose reference tz is authoritative for ``local_date`` display (§3.8).

    A missing athlete row means the reference tz is unknowable, so the local-date surface
    fails closed (CFG-R6) rather than guessing a UTC default — a ``404`` for the owner whose
    activities are being read.
    """
    owner = await session.get(Athlete, _uid(athlete_id))
    if owner is None:
        raise not_found()
    return owner


def _uid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise not_found() from exc


__all__ = ["f", "local_date_of", "owner_or_not_found", "summary"]
