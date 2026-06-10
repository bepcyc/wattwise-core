"""Endurance-score gather path for the analytics service (ES-R1/ES-R2, QUAL-R9 split).

The service-side gather that assembles the three declared endurance-score components
(CTL, the MMP durability ratio, the latest aerobic-decoupling drift) from the canonical
store and hands them to the pure composition in :mod:`wattwise_core.analytics.endurance_score`.
Split out of ``_service_loaders`` to honor the module size ceiling (QUAL-R9).
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING

from wattwise_core.analytics import endurance_score as _es
from wattwise_core.analytics._service_loaders import _curve_point
from wattwise_core.analytics.constants import (
    ES_LONG_DURATION_S,
    ES_SHORT_DURATION_S,
    ES_WINDOW_DAYS,
)
from wattwise_core.analytics.result import (
    Computed,
    MetricResult,
    Unavailable,
    UnavailableReason,
    is_computed,
)

if TYPE_CHECKING:
    from wattwise_core.analytics.service import AnalyticsService

async def _gather_endurance_score(
    svc: AnalyticsService, athlete_id: str, as_of: _dt.date
) -> MetricResult[float]:
    """Gather the ES-R1 upstream components through the service and compose (ES-R2).

    Reads ONLY upstream `MetricResult`s produced by the canonical service capabilities
    (CTL from :meth:`~wattwise_core.analytics.service.AnalyticsService.pmc`, the
    durability ratio from the sport-partitioned power curve, the most recent computed
    aerobic decoupling in the window) — never a raw stream (ES-R2); the numeric
    composition lives in the pure :mod:`~wattwise_core.analytics.endurance_score`.
    The power components are gathered for the athlete's canonical ``current_sport``
    (no hardcoded sport): an unset sport fails those components closed and the
    configured ES-R2 missing-component policy decides the outcome.
    """
    ctl: MetricResult[float]
    if await svc._earliest_activity_date(athlete_id) is None:
        # PMC's honest (0,0) cold-start origin means an athlete with NO training history
        # "computes" CTL == 0 — a wrong-but-plausible chronic-load signal for scoring
        # (ANL-R4); with no history the score abstains instead (mirrors the readiness
        # READINESS_MIN_FITNESS_CTL guard's reasoning).
        ctl = Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "no training history: CTL carries no chronic-load signal to score",
        )
    elif (pmc_days := await svc.pmc(athlete_id, as_of, as_of)) and is_computed(pmc_days[-1]):
        ctl = Computed(value=float(pmc_days[-1].value.ctl))
    else:
        ctl = Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, "no computed PMC day (CTL)")
    sport = await svc.current_sport(athlete_id)
    from_date = as_of - _dt.timedelta(days=ES_WINDOW_DAYS)
    no_sport = Unavailable(
        UnavailableReason.MISSING_REQUIRED_INPUT,
        "athlete has no canonical current_sport to partition power components by",
    )
    ratio: MetricResult[float] = no_sport
    decoupling: MetricResult[float] = no_sport
    if sport is not None:
        curve = await svc.power_curve(athlete_id, from_date, as_of, sport=sport)
        ratio = _es.durability_ratio(
            _curve_point(curve, ES_LONG_DURATION_S), _curve_point(curve, ES_SHORT_DURATION_S)
        )
        decoupling = await _latest_decoupling(svc, athlete_id, from_date, as_of)
    return _es.endurance_score(ctl, ratio, decoupling, sport=sport)


async def _latest_decoupling(
    svc: AnalyticsService, athlete_id: str, from_date: _dt.date, to_date: _dt.date
) -> MetricResult[float]:
    """The most recent activity's ``Computed`` aerobic decoupling in the window.

    Scans the resolved canonical activities newest-first and returns the first
    computed decoupling; none computable ⇒ typed ``Unavailable`` (never ``0``-drift).
    """
    activities = await svc._activities_in_range(athlete_id, from_date, to_date)
    for act in sorted(activities, key=lambda a: a.start_time, reverse=True):
        res = await svc.aerobic_decoupling(str(act.activity_id))
        if is_computed(res):
            return res
    return Unavailable(
        UnavailableReason.MISSING_REQUIRED_INPUT,
        "no activity in the window yields a computed aerobic decoupling",
    )

