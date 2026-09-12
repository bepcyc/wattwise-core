"""Performance analytics router — the chart-ready ``/v1/performance/*`` surface.

Serves the canonical analytics views (doc 60 appendix §1-§10) as chart-ready,
source-agnostic payloads with no client-side recomputation (API-R31). Every endpoint
maps to exactly one doc-40 model (API-R30) and reads computed analytics ONLY through
:class:`~wattwise_core.analytics.service.AnalyticsService`. A metric that cannot be
computed fails closed (ANL-R3/R4): a per-point value becomes a typed ``null`` surfaced
through ``coverage`` (never a fabricated ``0``, API-R29); an endpoint precondition
failure becomes ``422 analytics-precondition-unmet`` with a machine ``errors[].code``
(e.g. ``cp_insufficient_points``/``hrv_dsp_unavailable``, ERR-R9).

Acting athlete identity is server-derived from the bearer token (AUTH-R3) via
:func:`current_athlete_id`; the client never supplies it. Every endpoint requires the
``read`` scope (AUTH-R11). The identity/scope/service dependencies are override seams
the app factory wires (FastAPI ``dependency_overrides``). No field is source-shaped or
carries a provider name (AUTH-R15/ANL-R1); fidelity is the SCHEMA-R9 ``coverage`` only.

Requirement IDs: API-R29, API-R30, API-R31, ERR-R9, AUTH-R3, AUTH-R11, AUTH-R15,
ANL-R1, ANL-R3, ANL-R4, SCHEMA-R8, SCHEMA-R9, PMC-R1, CP-R1, CP-R4, MMP-R1, WBAL-R1,
HRV-R0, HRV-R5, DEC-R1, TRIMP-R1, TRIMP-R3.
"""

from __future__ import annotations

import datetime as _dt
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from wattwise_core.analytics.result import Unavailable, is_computed
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.api.chart_schemas import ChartSeries, CoverageDescriptor, SeriesPoint
from wattwise_core.api.deps import RateLimit
from wattwise_core.api.errors import ProblemError
from wattwise_core.api.perf_helpers import (
    activities_in_local_range as _activities_in_range,
)
from wattwise_core.api.perf_helpers import (
    empty_coverage as _empty_coverage,
)
from wattwise_core.api.perf_helpers import (
    now as _now,
)
from wattwise_core.api.perf_helpers import (
    present_coverage as _present_coverage,
)
from wattwise_core.api.problems import precondition_unmet, range_reversed

# The pure per-point payload builders (one per doc-40 chart model, API-R30) live in the
# focused :mod:`performance_points` sibling (QUAL-R9 size split); behavior is unchanged.
from wattwise_core.api.routers.performance_points import (
    _coggan_point,
    _cp_point,
    _dec_point,
    _guard_trimp_domain,
    _hrv_point,
    _load_point,
    _mmp_point,
    _pmc_point,
    _pmc_summary,
    _trimp_point,
    _wbal_point,
)

router = APIRouter(prefix="/v1/performance", tags=["performance"], dependencies=[RateLimit])

# --- dependency seams (overridden by the app factory) ---------------------------


def require_read_scope() -> None:
    """Gate on the ``read`` scope (AUTH-R11); the app factory overrides it (fail-closed)."""
    raise ProblemError("insufficient-scope")  # pragma: no cover - replaced by the app factory


def current_athlete_id() -> str:
    """Server-derived acting athlete id (AUTH-R3); app factory overrides it (fail-closed)."""
    raise ProblemError("unauthenticated")  # pragma: no cover - replaced by the app factory


def analytics_service() -> AnalyticsService:
    """Provide the request-scoped :class:`AnalyticsService`; app factory overrides it."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


_Read = Depends(require_read_scope)
AthleteId = Annotated[str, Depends(current_athlete_id)]
Service = Annotated[AnalyticsService, Depends(analytics_service)]


# --- wire shapes + coverage/label helpers — see chart_schemas / perf_helpers ------


def _precondition_unmet(code: str, detail: str) -> ProblemError:
    """A ``422 analytics-precondition-unmet`` carrying the machine code (ERR-R9).

    Raised as a catalog :class:`ProblemError` (NOT a framework ``HTTPException`` whose
    structured detail the status-only handler discards) so the slug AND the machine
    ``errors[].code`` reach the client to branch on the fail-closed analytics contract.
    """
    return precondition_unmet(code, detail)


def date_range(
    frm: Annotated[_dt.date, Query(alias="from", description="Inclusive local start date.")],
    to: Annotated[_dt.date, Query(description="Inclusive local end date.")],
) -> tuple[_dt.date, _dt.date]:
    """Typed ``(from, to)`` range dependency; ``from > to`` → ``422`` (PAGE-R8/ERR-R6)."""
    if frm > to:
        raise range_reversed("from")
    return frm, to


Range = Annotated[tuple[_dt.date, _dt.date], Depends(date_range)]


# --- §1 PMC: load vs. capacity --------------------------------------------------


@router.get(
    "/load-fitness",
    response_model=ChartSeries,
    operation_id="getLoadFitness",
    dependencies=[_Read],
)
async def load_fitness(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """PMC fitness/fatigue/form over time (PMC-R1) → chart-ready ``PMCSeries``."""
    frm, to = rng
    series = await svc.pmc(athlete_id, frm, to)
    if len(series) == 1 and isinstance(series[0], Unavailable):
        raise _precondition_unmet("pmc_seed_unavailable", series[0].detail)
    # The per-day canonical load that feeds the EWMA (LOAD-R1) — the SAME series
    # ``load_metrics`` reads — so the PMC chart's ``load`` agrees with fitness/fatigue instead
    # of always rendering ``null`` (#120). ``daily_load_series`` fills the same dense [frm, to]
    # calendar PMC does, so the per-day lookup aligns (a real 0.0 rest day stays 0.0, never null).
    loads = await svc.daily_load_series(athlete_id, frm, to)
    days = [frm + _dt.timedelta(days=i) for i in range(len(series))]
    items = [_pmc_point(day, res, loads.get(day)) for day, res in zip(days, series, strict=True)]
    last = series[-1] if series else None
    summary = _pmc_summary(last)
    return ChartSeries(
        items=items,
        x_axis="local_date",
        method="pmc_ewma",
        summary=summary,
        coverage=_empty_coverage(),
        computed_at=_now(),
    )


# --- §2 daily load vs. stress ---------------------------------------------------


@router.get(
    "/load-metrics",
    response_model=ChartSeries,
    operation_id="getLoadMetrics",
    dependencies=[_Read],
)
async def load_metrics(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Canonical daily training load vs. stress signals (LOAD-R1) → ``LoadMetrics``."""
    frm, to = rng
    loads = await svc.daily_load_series(athlete_id, frm, to)
    items = [_load_point(day, loads[day]) for day in sorted(loads)]
    total = sum(v for v in loads.values() if v is not None)
    summary = {"canonical_load_total": total, "trimp_points_total": None, "load_model": "power_tss"}
    return ChartSeries(
        items=items,
        x_axis="local_date",
        method="daily_load_sum",
        summary=summary,
        coverage=_empty_coverage(),
        computed_at=_now(),
    )


# --- §3 critical power ----------------------------------------------------------


@router.get(
    "/critical-power",
    response_model=ChartSeries,
    operation_id="getCriticalPower",
    dependencies=[_Read],
)
async def critical_power(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Fitted CP/W' over the observed MMP curve (CP-R1) → ``CriticalPowerFit``."""
    frm, to = rng
    curve = await svc.power_curve(athlete_id, frm, to)
    fit = await svc.critical_power(athlete_id, frm, to)
    if isinstance(fit, Unavailable):
        raise _precondition_unmet("cp_insufficient_points", fit.detail)
    items = [_cp_point(d, curve.get(d), fit) for d in sorted(curve)]
    extra = fit.quality.extra
    summary = {
        "cp_w": fit.value.cp_w,
        "w_prime_j": fit.value.w_prime_j,
        "r_squared": fit.value.r2,
        "points_used": extra.get("n_points"),
        "model": "linear_work_time",
    }
    return ChartSeries(
        items=items,
        x_axis="duration_s",
        method="cp_linear_work_time",
        summary=summary,
        coverage=_present_coverage(fit.quality),
        computed_at=_now(),
    )


# --- §4 power-duration curve ----------------------------------------------------


@router.get(
    "/power-curve",
    response_model=ChartSeries,
    operation_id="getPowerCurve",
    dependencies=[_Read],
)
async def power_curve(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Mean-maximal power per duration (MMP-R1) → ``PowerCurve`` (non-increasing)."""
    frm, to = rng
    curve = await svc.power_curve(athlete_id, frm, to)
    items = [_mmp_point(d, curve[d]) for d in sorted(curve)]
    return ChartSeries(
        items=items,
        x_axis="duration_s",
        method="mean_maximal_power",
        summary={},
        coverage=_empty_coverage(),
        computed_at=_now(),
    )


# --- §6 Coggan NP/IF/TSS --------------------------------------------------------


@router.get(
    "/coggan",
    response_model=ChartSeries,
    operation_id="getCogganMetrics",
    dependencies=[_Read],
)
async def coggan(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Per-activity NP/IF/TSS over time (NP-R1/IF-R1/TSS-R1) → ``CogganMetrics``."""
    frm, to = rng
    activities = await _activities_in_range(svc, athlete_id, frm, to)
    items = [_coggan_point(aid, day, await svc.coggan(aid)) for aid, day in activities]
    return ChartSeries(
        items=items,
        x_axis="local_date",
        method="coggan_np_if_tss",
        summary={"ftp_w": None},
        coverage=_empty_coverage(),
        computed_at=_now(),
    )


# --- §7 W'balance ---------------------------------------------------------------


@router.get(
    "/w-balance",
    response_model=ChartSeries,
    operation_id="getWBalance",
    dependencies=[_Read],
)
async def w_balance(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Per-activity anaerobic W' balance over time (WBAL-R1) → ``WBalanceSeries``."""
    frm, to = rng
    activities = await _activities_in_range(svc, athlete_id, frm, to)
    items = [_wbal_point(aid, day, await svc.w_balance(aid)) for aid, day in activities]
    return ChartSeries(
        items=items,
        x_axis="local_date",
        method="skiba_2012_differential",
        summary={"clamping_policy": "raw", "model": "skiba_2012_differential"},
        coverage=_empty_coverage(),
        computed_at=_now(),
    )


# --- §8 HRV ---------------------------------------------------------------------


@router.get(
    "/hrv",
    response_model=ChartSeries,
    operation_id="getHrv",
    dependencies=[_Read],
)
async def hrv(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Time-domain HRV trend over time (HRV-R0/R3) → ``HRVMetrics`` (never zeros)."""
    frm, to = rng
    days = [frm + _dt.timedelta(days=i) for i in range((to - frm).days + 1)]
    items = [_hrv_point(day, await svc.hrv(athlete_id, day)) for day in days]
    if not any(p.coverage.present for p in items):
        raise _precondition_unmet("hrv_dsp_unavailable", "no HRV recording in range")
    summary = {"avg_rmssd_ms": None, "hrv_spectral_method": None}
    return ChartSeries(
        items=items,
        x_axis="local_date",
        method="hrv_time_domain",
        summary=summary,
        coverage=_empty_coverage(),
        computed_at=_now(),
    )


# --- §9 aerobic decoupling ------------------------------------------------------


@router.get(
    "/aerobic-decoupling",
    response_model=ChartSeries,
    operation_id="getAerobicDecoupling",
    dependencies=[_Read],
)
async def aerobic_decoupling(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Per-activity aerobic decoupling over time (DEC-R1) → ``AerobicDecoupling``."""
    frm, to = rng
    activities = await _activities_in_range(svc, athlete_id, frm, to)
    items = [_dec_point(aid, day, await svc.aerobic_decoupling(aid)) for aid, day in activities]
    return ChartSeries(
        items=items,
        x_axis="local_date",
        method="aerobic_decoupling",
        summary={},
        coverage=_empty_coverage(),
        computed_at=_now(),
    )


# --- §10 TRIMP ------------------------------------------------------------------


@router.get(
    "/trimp",
    response_model=ChartSeries,
    operation_id="getTrimp",
    dependencies=[_Read],
)
async def trimp(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Per-activity TRIMP over time (TRIMP-R1) → ``TrimpSeries`` (Banister-HRR)."""
    frm, to = rng
    activities = await _activities_in_range(svc, athlete_id, frm, to)
    results = [(aid, day, await svc.trimp(aid)) for aid, day in activities]
    _guard_trimp_domain(results)
    items = [_trimp_point(aid, day, res) for aid, day, res in results]
    total = sum(float(r.value) for _, _, r in results if is_computed(r))
    summary = {"trimp_points_total": total, "load_model": "hr_load"}
    return ChartSeries(
        items=items,
        x_axis="local_date",
        method="banister_hr_load",
        summary=summary,
        coverage=_empty_coverage(),
        computed_at=_now(),
    )


# --- shared activity enumeration ------------------------------------------------


__all__ = [
    "ChartSeries",
    "CoverageDescriptor",
    "SeriesPoint",
    "analytics_service",
    "current_athlete_id",
    "require_read_scope",
    "router",
]

#: OpenAPI security metadata (DOC-R3): the scopes this seam gate requires.
require_read_scope.required_scopes = ("read",)  # type: ignore[attr-defined]
