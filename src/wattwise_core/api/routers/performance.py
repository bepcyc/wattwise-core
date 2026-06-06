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
import math
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from wattwise_core.analytics.result import (
    Computed,
    MetricResult,
    Unavailable,
    UnavailableReason,
    is_computed,
)
from wattwise_core.analytics.service import AnalyticsService

router = APIRouter(prefix="/v1/performance", tags=["performance"])

# HRV series keys (HRV-R3/R5): time-domain + frequency-domain band powers (doc 60 §8).
_HRV_KEYS: tuple[str, ...] = (
    "rmssd_ms", "sdnn_ms", "pnn50_pct", "mean_nn_ms", "lf_power", "hf_power", "lf_hf_ratio",
)


# --- dependency seams (overridden by the app factory) ---------------------------


def require_read_scope() -> None:
    """Gate on the ``read`` scope (AUTH-R11); the app factory overrides it (fail-closed)."""
    raise HTTPException(  # pragma: no cover - replaced by the app factory
        status_code=status.HTTP_403_FORBIDDEN, detail="insufficient-scope"
    )


def current_athlete_id() -> str:
    """Server-derived acting athlete id (AUTH-R3); app factory overrides it (fail-closed)."""
    raise HTTPException(  # pragma: no cover - replaced by the app factory
        status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthenticated"
    )


def analytics_service() -> AnalyticsService:
    """Provide the request-scoped :class:`AnalyticsService`; app factory overrides it."""
    raise HTTPException(  # pragma: no cover - replaced by the app factory
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="internal-error"
    )


_Read = Depends(require_read_scope)
AthleteId = Annotated[str, Depends(current_athlete_id)]
Service = Annotated[AnalyticsService, Depends(analytics_service)]


# --- wire shapes (SCHEMA-R8 / SCHEMA-R9) ----------------------------------------


class CoverageDescriptor(BaseModel):
    """Source-agnostic per-point/scalar coverage descriptor (SCHEMA-R9; no source name)."""

    present: bool
    fidelity: str
    gap_fraction: float = 0.0
    disputed: bool = False
    provisional: bool = False
    substitution: dict[str, Any] | None = None


class SeriesPoint(BaseModel):
    """One chart point (SCHEMA-R8): an X-axis key, ``label``, named values, coverage.

    The per-activity variant also carries ``activity_id`` so two activities on the
    same calendar day are uniquely addressable (Coggan/W'balance/decoupling/TRIMP).
    """

    local_date: _dt.date | None = None
    duration_s: int | None = None
    activity_id: str | None = None
    label: str
    values: dict[str, float | None]
    coverage: CoverageDescriptor


class ChartSeries(BaseModel):
    """Chart-ready time-series envelope (API-R31): items + precomputed ``summary``."""

    items: list[SeriesPoint]
    x_axis: str
    method: str
    summary: dict[str, Any]
    coverage: CoverageDescriptor
    computed_at: _dt.datetime


def _present_coverage(quality: Any) -> CoverageDescriptor:
    """Map a computed metric's ``QualityReport`` to a present coverage (PMC-R6 provisional)."""
    extra = getattr(quality, "extra", {}) or {}
    fidelity = str(extra.get("fidelity", "raw_stream"))
    return CoverageDescriptor(
        present=True,
        fidelity=fidelity,
        gap_fraction=1.0 - float(getattr(quality, "coverage_fraction", 1.0)),
        provisional=bool(extra.get("provisional", False)),
    )


_FAILED_REASONS = frozenset({
    UnavailableReason.MISSING_DEPENDENCY,
    UnavailableReason.POOR_FIT,
    UnavailableReason.OUT_OF_DOMAIN,
})


def _absent_coverage(result: Unavailable) -> CoverageDescriptor:
    """Map a typed :class:`Unavailable` to typed-absence coverage (ANL-R4; no reason leak)."""
    fidelity = "absent_failed" if result.reason in _FAILED_REASONS else "absent_true"
    return CoverageDescriptor(present=False, fidelity=fidelity, gap_fraction=1.0)


def _coverage_for(result: MetricResult[Any]) -> CoverageDescriptor:
    """Coverage for either branch of a :class:`MetricResult`."""
    return _present_coverage(result.quality) if is_computed(result) else _absent_coverage(result)


def _value_of(result: MetricResult[float]) -> float | None:
    """The scalar value of a numeric result, or typed ``null`` (never ``0``)."""
    return float(result.value) if is_computed(result) else None


def _opt_float(value: Any) -> float | None:
    """Coerce a finite numeric (e.g. a quality ``extra`` stat) to ``float | None``."""
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _precondition_unmet(code: str, detail: str) -> HTTPException:
    """A ``422 analytics-precondition-unmet`` with a machine code (ERR-R9)."""
    return HTTPException(
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        detail={
            "type": "analytics-precondition-unmet",
            "errors": [{"code": code, "detail": detail}],
        },
    )


def date_range(
    frm: Annotated[_dt.date, Query(alias="from", description="Inclusive local start date.")],
    to: Annotated[_dt.date, Query(description="Inclusive local end date.")],
) -> tuple[_dt.date, _dt.date]:
    """Typed ``(from, to)`` range dependency; ``from > to`` → ``422`` (PAGE-R8)."""
    if frm > to:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail={"type": "validation-error", "errors": [{"parameter": "from"}]},
        )
    return frm, to


def _now() -> _dt.datetime:
    """Server timestamp for the precomputed ``computed_at`` (wall-clock at edge)."""
    return _dt.datetime.now(tz=_dt.UTC)


def _day_label(day: _dt.date) -> str:
    """Jargon-free X-tick label for a calendar day (API-R21)."""
    return day.strftime("%b ") + str(day.day)


def _duration_label(seconds: int) -> str:
    """Jargon-free X-tick label for a power-duration grid point (API-R21)."""
    return f"{seconds // 60} min" if seconds % 60 == 0 else f"{seconds} sec"


def _empty_coverage() -> CoverageDescriptor:
    """The summary-level coverage descriptor for a present series (SCHEMA-R9)."""
    return CoverageDescriptor(present=True, fidelity="raw_stream")


Range = Annotated[tuple[_dt.date, _dt.date], Depends(date_range)]


# --- §1 PMC: load vs. capacity --------------------------------------------------


@router.get("/load-fitness", response_model=ChartSeries, dependencies=[_Read])
async def load_fitness(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """PMC fitness/fatigue/form over time (PMC-R1) → chart-ready ``PMCSeries``."""
    frm, to = rng
    series = await svc.pmc(athlete_id, frm, to)
    if len(series) == 1 and isinstance(series[0], Unavailable):
        raise _precondition_unmet("pmc_seed_unavailable", series[0].detail)
    days = [frm + _dt.timedelta(days=i) for i in range(len(series))]
    items = [_pmc_point(day, res) for day, res in zip(days, series, strict=True)]
    last = series[-1] if series else None
    summary = _pmc_summary(last)
    return ChartSeries(
        items=items, x_axis="local_date", method="pmc_ewma",
        summary=summary, coverage=_empty_coverage(), computed_at=_now(),
    )


def _pmc_point(day: _dt.date, res: MetricResult[Any]) -> SeriesPoint:
    """One PMC calendar point: fitness/fatigue/form values + coverage (PMC-R1/R6)."""
    if is_computed(res):
        vals: dict[str, float | None] = {
            "fitness": res.value.ctl, "fatigue": res.value.atl, "form": res.value.tsb,
            "load": None,
        }
    else:
        vals = {"fitness": None, "fatigue": None, "form": None, "load": None}
    return SeriesPoint(
        local_date=day, label=_day_label(day), values=vals, coverage=_coverage_for(res)
    )


def _pmc_summary(last: MetricResult[Any] | None) -> dict[str, Any]:
    """Precomputed current CTL/ATL/TSB + EWMA constants + seed (PMC-R3)."""
    cur = last.value if last is not None and is_computed(last) else None
    return {
        "fitness": cur.ctl if cur else None,
        "fatigue": cur.atl if cur else None,
        "form": cur.tsb if cur else None,
        "ewma_constants": {"tau_fitness": 42, "tau_fatigue": 7},
        "seed": None,
    }


# --- §2 daily load vs. stress ---------------------------------------------------


@router.get("/load-metrics", response_model=ChartSeries, dependencies=[_Read])
async def load_metrics(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Canonical daily training load vs. stress signals (LOAD-R1) → ``LoadMetrics``."""
    frm, to = rng
    loads = await svc.daily_load_series(athlete_id, frm, to)
    items = [_load_point(day, loads[day]) for day in sorted(loads)]
    total = sum(v for v in loads.values() if v is not None)
    summary = {"canonical_load_total": total, "trimp_points_total": None, "load_model": "power_tss"}
    return ChartSeries(
        items=items, x_axis="local_date", method="daily_load_sum",
        summary=summary, coverage=_empty_coverage(), computed_at=_now(),
    )


def _load_point(day: _dt.date, value: float | None) -> SeriesPoint:
    """One daily-load point; a ``None`` is a surfaced typed-unavailable, never ``0``."""
    present = value is not None
    cov = CoverageDescriptor(present=present, fidelity="raw_stream" if present else "absent_true",
                             gap_fraction=0.0 if present else 1.0)
    vals: dict[str, float | None] = {
        "canonical_load": value, "power_load": value, "hr_load": None, "trimp_points": None,
    }
    return SeriesPoint(local_date=day, label=_day_label(day), values=vals, coverage=cov)


# --- §3 critical power ----------------------------------------------------------


@router.get("/critical-power", response_model=ChartSeries, dependencies=[_Read])
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
        "cp_w": fit.value.cp_w, "w_prime_j": fit.value.w_prime_j,
        "r_squared": fit.value.r2, "points_used": extra.get("n_points"),
        "model": "linear_work_time",
    }
    return ChartSeries(
        items=items, x_axis="duration_s", method="cp_linear_work_time",
        summary=summary, coverage=_present_coverage(fit.quality), computed_at=_now(),
    )


def _cp_point(d: int, observed: MetricResult[Any] | None, fit: Computed[Any]) -> SeriesPoint:
    """One duration-grid point: observed MMP + CP-model prediction ``W'/t + CP``."""
    obs = observed.value.mean_power_w if observed is not None and is_computed(observed) else None
    predicted = fit.value.w_prime_j / d + fit.value.cp_w if d > 0 else None
    cov = _coverage_for(observed) if observed is not None else _absent_coverage(
        Unavailable(UnavailableReason.INSUFFICIENT_DATA)
    )
    return SeriesPoint(
        duration_s=d, label=_duration_label(d),
        values={"power_watts": obs, "predicted_power_watts": predicted}, coverage=cov,
    )


# --- §4 power-duration curve ----------------------------------------------------


@router.get("/power-curve", response_model=ChartSeries, dependencies=[_Read])
async def power_curve(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Mean-maximal power per duration (MMP-R1) → ``PowerCurve`` (non-increasing)."""
    frm, to = rng
    curve = await svc.power_curve(athlete_id, frm, to)
    items = [_mmp_point(d, curve[d]) for d in sorted(curve)]
    return ChartSeries(
        items=items, x_axis="duration_s", method="mean_maximal_power",
        summary={}, coverage=_empty_coverage(), computed_at=_now(),
    )


def _mmp_point(d: int, res: MetricResult[Any]) -> SeriesPoint:
    """One MMP duration point: best mean power for the duration (nullable per MMP-R5)."""
    val = res.value.mean_power_w if is_computed(res) else None
    return SeriesPoint(
        duration_s=d, label=_duration_label(d),
        values={"power_watts": val}, coverage=_coverage_for(res),
    )


# --- §6 Coggan NP/IF/TSS --------------------------------------------------------


@router.get("/coggan", response_model=ChartSeries, dependencies=[_Read])
async def coggan(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Per-activity NP/IF/TSS over time (NP-R1/IF-R1/TSS-R1) → ``CogganMetrics``."""
    frm, to = rng
    activities = await _activities_in_range(svc, athlete_id, frm, to)
    items = [_coggan_point(aid, day, await svc.coggan(aid)) for aid, day in activities]
    return ChartSeries(
        items=items, x_axis="local_date", method="coggan_np_if_tss",
        summary={"ftp_w": None}, coverage=_empty_coverage(), computed_at=_now(),
    )


def _coggan_point(activity_id: str, day: _dt.date, res: MetricResult[Any]) -> SeriesPoint:
    """One per-activity Coggan point: TSS / IF / VI, each propagating unavailable."""
    if is_computed(res):
        b = res.value
        vals: dict[str, float | None] = {
            "tss": _value_of(b.tss), "intensity_factor": _value_of(b.if_),
            "variability_index": _value_of(b.variability_index),
        }
        cov = _coverage_for(b.tss)
    else:
        vals = {"tss": None, "intensity_factor": None, "variability_index": None}
        cov = _coverage_for(res)
    return SeriesPoint(
        local_date=day, activity_id=activity_id, label=_day_label(day),
        values=vals, coverage=cov,
    )


# --- §7 W'balance ---------------------------------------------------------------


@router.get("/w-balance", response_model=ChartSeries, dependencies=[_Read])
async def w_balance(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Per-activity anaerobic W' balance over time (WBAL-R1) → ``WBalanceSeries``."""
    frm, to = rng
    activities = await _activities_in_range(svc, athlete_id, frm, to)
    items = [_wbal_point(aid, day, await svc.w_balance(aid)) for aid, day in activities]
    return ChartSeries(
        items=items, x_axis="local_date", method="skiba_2012_differential",
        summary={"clamping_policy": "raw", "model": "skiba_2012_differential"},
        coverage=_empty_coverage(), computed_at=_now(),
    )


def _wbal_point(activity_id: str, day: _dt.date, res: MetricResult[Any]) -> SeriesPoint:
    """One per-activity W'-balance point: start/end/min W' in joules (may be negative)."""
    if is_computed(res):
        series = res.value.series
        start = float(series[0]) if series.size else None
        end = float(series[-1]) if series.size else None
        vals: dict[str, float | None] = {
            "w_prime_start_joules": start, "w_prime_end_joules": end,
            "w_prime_min_joules": float(res.value.w_prime_balance_min),
        }
    else:
        vals = dict.fromkeys(
            ("w_prime_start_joules", "w_prime_end_joules", "w_prime_min_joules")
        )
    return SeriesPoint(
        local_date=day, activity_id=activity_id, label=_day_label(day),
        values=vals, coverage=_coverage_for(res),
    )


# --- §8 HRV ---------------------------------------------------------------------


@router.get("/hrv", response_model=ChartSeries, dependencies=[_Read])
async def hrv(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Time-domain HRV trend over time (HRV-R0/R3) → ``HRVMetrics`` (never zeros)."""
    frm, to = rng
    days = [frm + _dt.timedelta(days=i) for i in range((to - frm).days + 1)]
    items = [_hrv_point(day, await svc.hrv(athlete_id, day)) for day in days]
    if not any(p.coverage.present for p in items):
        raise _precondition_unmet("hrv_dsp_unavailable", "no HRV recording in range")
    summary = {"avg_rmssd_ms": None, "hrv_spectral_method": None}
    return ChartSeries(
        items=items, x_axis="local_date", method="hrv_time_domain",
        summary=summary, coverage=_empty_coverage(), computed_at=_now(),
    )


def _hrv_point(day: _dt.date, res: MetricResult[Any]) -> SeriesPoint:
    """One HRV day: time-domain metrics; freq-domain typed-unavailable when absent."""
    vals: dict[str, float | None] = dict.fromkeys(_HRV_KEYS)
    if is_computed(res):
        td = res.value
        vals.update(
            rmssd_ms=_opt_float(td.rmssd_ms), sdnn_ms=_opt_float(td.sdnn_ms),
            pnn50_pct=_opt_float(td.pnn50_pct), mean_nn_ms=_opt_float(td.mean_nn_ms),
        )
        cov = _present_coverage(res.quality)
    else:
        cov = _absent_coverage(res)
    return SeriesPoint(local_date=day, label=_day_label(day), values=vals, coverage=cov)


# --- §9 aerobic decoupling ------------------------------------------------------


@router.get("/aerobic-decoupling", response_model=ChartSeries, dependencies=[_Read])
async def aerobic_decoupling(svc: Service, athlete_id: AthleteId, rng: Range) -> ChartSeries:
    """Per-activity aerobic decoupling over time (DEC-R1) → ``AerobicDecoupling``."""
    frm, to = rng
    activities = await _activities_in_range(svc, athlete_id, frm, to)
    items = [_dec_point(aid, day, await svc.aerobic_decoupling(aid)) for aid, day in activities]
    return ChartSeries(
        items=items, x_axis="local_date", method="aerobic_decoupling",
        summary={}, coverage=_empty_coverage(), computed_at=_now(),
    )


def _dec_point(activity_id: str, day: _dt.date, res: MetricResult[float]) -> SeriesPoint:
    """One per-activity decoupling point: %, plus first/second-half efficiency ratios."""
    if is_computed(res):
        extra = res.quality.extra
        vals: dict[str, float | None] = {
            "decoupling_pct": float(res.value),
            "first_half_ratio": _opt_float(extra.get("eff_first_half")),
            "second_half_ratio": _opt_float(extra.get("eff_second_half")),
        }
    else:
        vals = {"decoupling_pct": None, "first_half_ratio": None, "second_half_ratio": None}
    return SeriesPoint(
        local_date=day, activity_id=activity_id, label=_day_label(day),
        values=vals, coverage=_coverage_for(res),
    )


# --- §10 TRIMP ------------------------------------------------------------------


@router.get("/trimp", response_model=ChartSeries, dependencies=[_Read])
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
        items=items, x_axis="local_date", method="banister_hr_load",
        summary=summary, coverage=_empty_coverage(), computed_at=_now(),
    )


def _guard_trimp_domain(results: list[tuple[str, _dt.date, MetricResult[float]]]) -> None:
    """Map a wholly-uncomputable series to ``422`` with the TRIMP-R3 machine code.

    Every point missing thresholds → ``trimp_missing_thresholds``; an out-of-domain HR
    reserve → ``trimp_hr_domain_invalid``. A mix with ≥1 computed point degrades visibly
    per-point instead (API-R29).
    """
    if any(is_computed(r) for _, _, r in results):
        return
    reasons = {r.reason for _, _, r in results if isinstance(r, Unavailable)}
    if UnavailableReason.OUT_OF_DOMAIN in reasons:
        raise _precondition_unmet("trimp_hr_domain_invalid", "HR_max must exceed HR_rest")
    if reasons and reasons <= {UnavailableReason.MISSING_REQUIRED_INPUT}:
        raise _precondition_unmet("trimp_missing_thresholds", "HR thresholds absent")


def _trimp_point(activity_id: str, day: _dt.date, res: MetricResult[float]) -> SeriesPoint:
    """One per-activity TRIMP point: Banister-HRR points (zonal variant null here)."""
    vals: dict[str, float | None] = {"trimp_points": _value_of(res), "trimp_zonal": None}
    return SeriesPoint(
        local_date=day, activity_id=activity_id, label=_day_label(day),
        values=vals, coverage=_coverage_for(res),
    )


# --- shared activity enumeration ------------------------------------------------


async def _activities_in_range(
    svc: AnalyticsService, athlete_id: str, frm: _dt.date, to: _dt.date
) -> list[tuple[str, _dt.date]]:
    """The athlete's resolved activities in range as ``(activity_id, local_date)``, time-ordered."""
    activities = await svc._activities_in_range(athlete_id, frm, to)
    activities.sort(key=lambda a: a.start_time)
    return [(str(a.activity_id), a.start_time.date()) for a in activities]


__all__ = [
    "ChartSeries", "CoverageDescriptor", "SeriesPoint",
    "analytics_service", "current_athlete_id", "require_read_scope", "router",
]
