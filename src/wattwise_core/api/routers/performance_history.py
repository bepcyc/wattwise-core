"""Performance history router — the PAGINATED ``/v1/performance`` collections.

The two ``/v1/performance/*`` surfaces that are PAGINATED collections rather than
continuous chart series (SCHEMA-R8 / appendix §5, §12): ``best-efforts`` and
``threshold-history``. They are split out of the continuous-series performance router
(``performance.py``) so each module stays within the QUAL-R9 module-size ceiling, while
SHARING that router's dependency seams (``require_read_scope`` / ``current_athlete_id`` /
``analytics_service``) — the app factory overrides those SAME function objects once, so
both routers are wired by one set of overrides (no second wiring step).

- ``GET /v1/performance/best-efforts`` → paginated ``BestEffort`` (API-R30 §5): each item
  is exactly ``MMP(duration_s)`` derived from the power curve (BEST-R1, single source of
  truth); a duration with no valid window is typed-unavailable, never ``0`` or a
  shorter-duration substitute (BEST-R3).
- ``GET /v1/performance/threshold-history`` → paginated ``ThresholdPoint`` (API-R30
  EXCEPTION): a canonical READ of the versioned ``fitness_signature`` history (doc 20
  §3.6 / GBO-R26), NOT a doc-40 computed metric — the documented exception to the
  one-backing-doc-40-model rule.

Both are returned as a single page (``next_cursor=None``): in OSS they are small bounded
collections (a handful of MMP durations / effective-dated signatures). No item carries a
source/provider name (AUTH-R15); fidelity is the SCHEMA-R9 ``coverage`` only.

Requirement IDs: API-R30, API-R31, SCHEMA-R8, SCHEMA-R9, AUTH-R3, AUTH-R11, AUTH-R15,
BEST-R1, BEST-R2, BEST-R3, MMP-R1, GBO-R26, PAGE-R1.
"""

from __future__ import annotations

import datetime as _dt
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from wattwise_core.analytics.result import MetricResult, is_computed
from wattwise_core.api.activity_schemas import Page
from wattwise_core.api.chart_schemas import CoverageDescriptor
from wattwise_core.api.deps import RateLimit
from wattwise_core.api.perf_helpers import coverage_for as _coverage_for
from wattwise_core.api.perf_helpers import day_label as _day_label
from wattwise_core.api.perf_helpers import duration_label as _duration_label
from wattwise_core.api.perf_helpers import opt_float as _opt_float
from wattwise_core.api.routers.performance import (
    AthleteId,
    Service,
    date_range,
    require_read_scope,
)

router = APIRouter(prefix="/v1/performance", tags=["performance"], dependencies=[RateLimit])

_Read = Depends(require_read_scope)
#: The same typed ``(from, to)`` range dependency the continuous-series router uses
#: (``from > to`` → ``422``, PAGE-R8/ERR-R6); reused so both routers share one contract.
_Range = Annotated[tuple[_dt.date, _dt.date], Depends(date_range)]


# --- paginated-collection items (best-efforts §5, threshold-history §12) ---------


class BestEffort(BaseModel):
    """One best-effort item (appendix §5; BEST-R1/R2).

    ``power_watts`` is exactly ``MMP(duration_s)`` (the single-source-of-truth derivation,
    BEST-R1; nullable when no valid window — never ``0`` or a shorter-duration substitute,
    BEST-R3). ``local_date`` and ``activity_id`` are the BEST-R2 lineage of the originating
    activity that produced this duration's peak (MMP-R4), so the agent can cite "your best
    5-minute power came from <activity on date>"; both are ``null`` when the effort is
    typed-unavailable (no winning window). Source-agnostic; fidelity is the SCHEMA-R9
    ``coverage`` only (AUTH-R15).
    """

    duration_s: int
    label: str
    power_watts: float | None
    local_date: _dt.date | None
    activity_id: str | None
    coverage: CoverageDescriptor


class BestEffortPage(BaseModel):
    """Paginated ``GET /v1/performance/best-efforts`` envelope (PAGE-R4, SCHEMA-R8).

    The project-wide cursor-pagination envelope (``data`` + a ``page`` sub-object). The OSS
    best-efforts collection is a small bounded MMP grid, so it is a single page
    (``has_more=False``, ``next_cursor=None``).
    """

    data: list[BestEffort]
    page: Page


class ThresholdPoint(BaseModel):
    """One threshold-history item (appendix §12; canonical read of ``fitness_signature``).

    A canonical doc-20 read (GBO-R26), NOT a doc-40 computed metric (the API-R30
    exception). Each item carries the signature's effective ``local_date`` and the
    nullable threshold fields; ``origin`` is the canonical provenance class (doc 20),
    never a source name (AUTH-R15).
    """

    local_date: _dt.date
    label: str
    ftp_w: float | None
    cp_w: float | None
    threshold_hr_bpm: int | None
    w_prime_j: float | None
    vo2max: float | None
    max_hr_bpm: int | None
    resting_hr_bpm: int | None
    origin: str
    coverage: CoverageDescriptor


class ThresholdHistoryPage(BaseModel):
    """Paginated ``GET /v1/performance/threshold-history`` envelope (PAGE-R4, SCHEMA-R8).

    The project-wide cursor-pagination envelope (``data`` + a ``page`` sub-object). The OSS
    threshold history is a small bounded set of effective-dated signatures, so it is a
    single page (``has_more=False``, ``next_cursor=None``).
    """

    data: list[ThresholdPoint]
    page: Page


# --- routes ----------------------------------------------------------------------


@router.get(
    "/best-efforts",
    response_model=BestEffortPage,
    operation_id="getBestEfforts",
    dependencies=[_Read],
)
async def best_efforts(svc: Service, athlete_id: AthleteId, rng: _Range) -> BestEffortPage:
    """Best efforts per duration, DERIVED from the MMP curve (BEST-R1) → paginated ``BestEffort``.

    A best effort for duration ``d`` is exactly ``MMP(d)`` — the single source of truth, so
    there is no second maximization path (BEST-R1). The shape is a PAGINATED collection, not a
    continuous chart series (SCHEMA-R8 / appendix §5): each item carries its own coverage and
    its ``duration_s``. A duration with no valid window is typed-unavailable, never ``0`` or a
    shorter-duration substitute (BEST-R3).
    """
    frm, to = rng
    curve = await svc.power_curve(athlete_id, frm, to)
    items = [_best_effort_item(d, curve[d]) for d in sorted(curve)]
    page = Page(limit=len(items), next_cursor=None, has_more=False)
    return BestEffortPage(data=items, page=page)


def _best_effort_item(d: int, res: MetricResult[Any]) -> BestEffort:
    """One best-effort item: ``MMP(d)`` power + the BEST-R2 lineage of its source (BEST-R1/R2/R3).

    ``power_watts`` is the winning window's mean power (or ``None`` when typed-unavailable,
    BEST-R3). ``local_date``/``activity_id`` are read from the MMP result's provenance
    (MMP-R4: which activity produced this duration's peak); both are ``None`` when the effort
    is unavailable or the provenance does not name a single originating activity.
    """
    if not is_computed(res):
        return BestEffort(
            duration_s=d,
            label=_duration_label(d),
            power_watts=None,
            local_date=None,
            activity_id=None,
            coverage=_coverage_for(res),
        )
    activity_id = res.provenance.activity_ids[0] if res.provenance.activity_ids else None
    raw_local_date = res.provenance.reference_params.get("local_date")
    local_date = raw_local_date if isinstance(raw_local_date, _dt.date) else None
    return BestEffort(
        duration_s=d,
        label=_duration_label(d),
        power_watts=res.value.mean_power_w,
        local_date=local_date,
        activity_id=activity_id,
        coverage=_coverage_for(res),
    )


@router.get(
    "/threshold-history",
    response_model=ThresholdHistoryPage,
    operation_id="getThresholdHistory",
    dependencies=[_Read],
)
async def threshold_history(
    svc: Service, athlete_id: AthleteId, rng: _Range
) -> ThresholdHistoryPage:
    """Effective-dated FTP/threshold history → paginated ``ThresholdPoint`` (API-R30 exception).

    This is a canonical READ of the versioned ``fitness_signature`` rows (doc 20 §3.6 /
    GBO-R26), NOT a doc-40 computed metric — the documented API-R30 exception. The shape is a
    PAGINATED collection, not a continuous chart series (SCHEMA-R8 / appendix §12): each item
    carries its own coverage and the signature's effective ``local_date``. ``origin`` is the
    canonical provenance class (doc 20), never a source name (AUTH-R15). An athlete with no
    signature in range → an empty page (never a fabricated zero).
    """
    frm, to = rng
    signatures = await svc.threshold_history(athlete_id, frm, to)
    items = [_threshold_point(sig) for sig in signatures]
    page = Page(limit=len(items), next_cursor=None, has_more=False)
    return ThresholdHistoryPage(data=items, page=page)


def _threshold_point(sig: Any) -> ThresholdPoint:
    """One threshold-history item from a ``fitness_signature`` row (appendix §12).

    The provenance of this canonical read is the recording's ``origin`` class (doc 20),
    surfaced both as the ``origin`` field and as the coverage ``fidelity`` — the
    API-R30-exception's source-agnostic descriptor (a present, real signature row).
    """
    origin = str(sig.origin.value if hasattr(sig.origin, "value") else sig.origin)
    coverage = CoverageDescriptor(present=True, fidelity=origin)
    return ThresholdPoint(
        local_date=sig.effective_date,
        label=_day_label(sig.effective_date),
        ftp_w=_opt_float(sig.ftp_w),
        cp_w=_opt_float(sig.cp_w),
        threshold_hr_bpm=sig.threshold_hr_bpm,
        w_prime_j=_opt_float(sig.w_prime_j),
        vo2max=_opt_float(sig.vo2max),
        max_hr_bpm=sig.max_hr_bpm,
        resting_hr_bpm=sig.resting_hr_bpm,
        origin=origin,
        coverage=coverage,
    )


__all__ = [
    "BestEffort",
    "BestEffortPage",
    "ThresholdHistoryPage",
    "ThresholdPoint",
    "router",
]
