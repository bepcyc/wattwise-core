"""Dashboard router — the composed home view (``/v1/dashboard``, API-R10 §8.2).

``GET /v1/dashboard/metrics`` composes the current training state (CTL/ATL/TSB, the
trailing-week load, the last activity, and week-over-week trend deltas) from the SAME
canonical analytics service every chart endpoint reads (API-R30) — no client-side
recomputation and no second math path. A value that cannot be computed is a typed
``null`` (ANL-R3/R4), never a fabricated ``0``. ``GET /v1/dashboard/alerts`` derives
typed, athlete-native alerts from the deterministic data-coverage diagnosis (the same
checks surface API-R15 narrates) — ``severity`` is the closed ``info|warning|critical``
enum (SCHEMA-R3) and ``message_text`` is jargon-free copy (API-R21).

No field is source-shaped or names a provider (AUTH-R15); identity is server-derived
(AUTH-R3); both endpoints require ``read`` (AUTH-R11) and ride the per-athlete rate
limit (LIMIT-R1). The analytics/identity/scope seams are the performance router's own
dependency objects, so the app factory's existing overrides wire this router too.

Requirement IDs: API-R10 (§8.2), API-R21, API-R29, AUTH-R3, AUTH-R11, AUTH-R15,
SCHEMA-R3 (severity), SCHEMA-R6 (computed_at), PAGE-R3/R4, LIMIT-R1.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.diagnose_deliverable import InputStatus, diagnose_coverage
from wattwise_core.analytics.result import is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.api.deps import DbSession, RateLimit
from wattwise_core.api.pagination import clamp_limit
from wattwise_core.api.routers.performance import (
    analytics_service,
    current_athlete_id,
    require_read_scope,
)
from wattwise_core.persistence.models import Activity

router = APIRouter(prefix="/v1/dashboard", tags=["dashboard"], dependencies=[RateLimit])

#: The PMC trailing window the composed metrics integrate over (matches the chronic
#: 42-day EWMA constant the analytics layer reasons over; the slice read is 8 days).
_TREND_DAYS = 8


class LastActivity(BaseModel):
    """The most recent canonical activity, source-blind (AUTH-R15)."""

    activity_id: str
    start_time: _dt.datetime
    sport: str


class DashboardMetrics(BaseModel):
    """``GET /v1/dashboard/metrics``: the composed home-view numbers (§8.2).

    Every analytic scalar is nullable — "not computable" is a typed ``null`` with the
    week-over-week deltas null alongside (ANL-R3/R4), never a zero.
    """

    model_config = ConfigDict(extra="forbid")

    fitness: float | None
    fatigue: float | None
    form: float | None
    fitness_delta_7d: float | None
    form_delta_7d: float | None
    weekly_load: float | None
    last_activity: LastActivity | None
    computed_at: _dt.datetime


class Alert(BaseModel):
    """One typed dashboard alert (§8.2): closed ``severity`` + athlete-native copy."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    severity: Literal["info", "warning", "critical"]
    message_text: str


class AlertPage(BaseModel):
    """The PAGE-R4 page block of the alert list."""

    limit: int
    next_cursor: str | None = None
    has_more: bool


class AlertList(BaseModel):
    """``GET /v1/dashboard/alerts``: the bounded alert page (PAGE-R3/R4)."""

    data: list[Alert]
    page: AlertPage


@router.get(
    "/metrics",
    response_model=DashboardMetrics,
    operation_id="getDashboardMetrics",
    dependencies=[Depends(require_read_scope)],
)
async def dashboard_metrics(
    svc: Annotated[AnalyticsService, Depends(analytics_service)],
    athlete_id: Annotated[str, Depends(current_athlete_id)],
    session: DbSession,
) -> DashboardMetrics:
    """The composed home-view metrics (§8.2): current PMC state + week trend + last ride.

    Reads the canonical PMC series (PMC-R1/R3 seeding) for today and 7 days ago, sums
    the trailing-week resolved daily load (LOAD-R1), and surfaces the newest canonical
    activity. Anything not computable is a typed ``null`` (ANL-R3/R4).
    """
    today = _dt.datetime.now(_dt.UTC).date()
    frm = today - _dt.timedelta(days=_TREND_DAYS - 1)
    series = await svc.pmc(athlete_id, frm, today)
    cur = series[-1].value if series and is_computed(series[-1]) else None
    prior = series[0].value if series and is_computed(series[0]) else None
    loads = await svc.daily_load_series(athlete_id, today - _dt.timedelta(days=6), today)
    known = [v for v in loads.values() if v is not None]
    return DashboardMetrics(
        fitness=cur.ctl if cur else None,
        fatigue=cur.atl if cur else None,
        form=cur.tsb if cur else None,
        fitness_delta_7d=(cur.ctl - prior.ctl) if cur and prior else None,
        form_delta_7d=(cur.tsb - prior.tsb) if cur and prior else None,
        weekly_load=sum(known) if known else None,
        last_activity=await _last_activity(session, athlete_id),
        computed_at=_dt.datetime.now(_dt.UTC),
    )


@router.get(
    "/alerts",
    response_model=AlertList,
    operation_id="listDashboardAlerts",
    dependencies=[Depends(require_read_scope)],
)
async def dashboard_alerts(
    svc: Annotated[AnalyticsService, Depends(analytics_service)],
    athlete_id: Annotated[str, Depends(current_athlete_id)],
    limit: Annotated[int, Query(json_schema_extra={"maximum": 200})] = 50,
) -> AlertList:
    """Typed readiness/data-health alerts for the home view (§8.2).

    Derives alerts DETERMINISTICALLY from the canonical coverage diagnosis (the same
    fail-closed checks API-R15 narrates): a MISSING analytic input raises a ``warning``,
    a STALE one an ``info`` — never a fabricated number, and no provider name appears
    (AUTH-R15). The list is naturally bounded by the closed check set; ``limit`` is
    still clamped/rejected per PAGE-R3.
    """
    bounded = clamp_limit(int(limit))
    diagnosis = await diagnose_coverage(svc, athlete_id=athlete_id)
    alerts: list[Alert] = []
    for inp in diagnosis.inputs:
        if inp.status is InputStatus.PRESENT:
            continue
        severity: Literal["info", "warning", "critical"] = (
            "warning" if inp.status is InputStatus.MISSING else "info"
        )
        verb = "isn't available yet" if inp.status is InputStatus.MISSING else "is out of date"
        alerts.append(
            Alert(
                alert_id=f"coverage:{inp.key}",
                severity=severity,
                message_text=f"Your {inp.label.lower()} {verb}.",
            )
        )
    page_rows = alerts[:bounded]
    return AlertList(
        data=page_rows,
        page=AlertPage(limit=bounded, next_cursor=None, has_more=len(alerts) > bounded),
    )


async def _last_activity(session: AsyncSession, athlete_id: str) -> LastActivity | None:
    """The newest canonical activity, or ``None`` for an empty history (never an error)."""
    try:
        owner = uuid.UUID(athlete_id)
    except (ValueError, AttributeError):
        return None
    row = (
        await session.execute(
            select(Activity)
            .where(Activity.athlete_id == owner)
            .order_by(Activity.start_time.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    return LastActivity(
        activity_id=str(row.activity_id), start_time=row.start_time, sport=row.sport
    )


__all__ = ["Alert", "DashboardMetrics", "router"]
