"""Per-point series builders for the performance router (doc 60 appendix, QUAL-R9 split).

The focused sibling of :mod:`wattwise_core.api.routers.performance` that owns the pure
per-point payload builders each chart endpoint maps its computed analytics through — one
builder per doc-40 model (API-R30): PMC, daily load, CP fit, MMP, Coggan, W'balance, HRV,
aerobic decoupling and TRIMP (with its TRIMP-R3 whole-series 422 guard). Every builder
propagates a typed-unavailable as a ``null`` value surfaced through ``coverage`` — never a
fabricated ``0`` (ANL-R3/R4, API-R29). Behavior is unchanged from the pre-split module.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from wattwise_core.analytics.result import (
    Computed,
    MetricResult,
    Unavailable,
    UnavailableReason,
    is_computed,
)
from wattwise_core.api.chart_schemas import CoverageDescriptor, SeriesPoint
from wattwise_core.api.perf_helpers import (
    absent_coverage as _absent_coverage,
)
from wattwise_core.api.perf_helpers import (
    coverage_for as _coverage_for,
)
from wattwise_core.api.perf_helpers import (
    day_label as _day_label,
)
from wattwise_core.api.perf_helpers import (
    duration_label as _duration_label,
)
from wattwise_core.api.perf_helpers import (
    opt_float as _opt_float,
)
from wattwise_core.api.perf_helpers import (
    present_coverage as _present_coverage,
)
from wattwise_core.api.perf_helpers import (
    value_of as _value_of,
)
from wattwise_core.api.problems import precondition_unmet

# HRV series keys (HRV-R3/R5): time-domain + frequency-domain band powers (doc 60 §8).
_HRV_KEYS: tuple[str, ...] = (
    "rmssd_ms",
    "sdnn_ms",
    "pnn50_pct",
    "mean_nn_ms",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
)


def _pmc_point(day: _dt.date, res: MetricResult[Any], load: float | None = None) -> SeriesPoint:
    """One PMC calendar point: fitness/fatigue/form values + coverage (PMC-R1/R6).

    ``load`` is the day's canonical training load that FEEDS the EWMA (LOAD-R1, doc 60 §1) —
    the same per-day value ``/load-metrics`` surfaces — threaded through verbatim so the PMC
    chart's load magnitude agrees with fitness/fatigue (which integrate it) instead of always
    reading ``null`` (#120). It is typed-null ONLY when genuinely unavailable; a real ``0.0``
    rest day is a genuine value and is NEVER coalesced to null (LOAD-R1 / PMC-R2).
    """
    if is_computed(res):
        vals: dict[str, float | None] = {
            "fitness": res.value.ctl,
            "fatigue": res.value.atl,
            "form": res.value.tsb,
            "load": load,
        }
    else:
        vals = {"fitness": None, "fatigue": None, "form": None, "load": load}
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


def _load_point(day: _dt.date, value: float | None) -> SeriesPoint:
    """One daily-load point; a ``None`` is a surfaced typed-unavailable, never ``0``."""
    present = value is not None
    cov = CoverageDescriptor(
        present=present,
        fidelity="raw_stream" if present else "absent_true",
        gap_fraction=0.0 if present else 1.0,
    )
    vals: dict[str, float | None] = {
        "canonical_load": value,
        "power_load": value,
        "hr_load": None,
        "trimp_points": None,
    }
    return SeriesPoint(local_date=day, label=_day_label(day), values=vals, coverage=cov)


def _cp_point(d: int, observed: MetricResult[Any] | None, fit: Computed[Any]) -> SeriesPoint:
    """One duration-grid point: observed MMP + CP-model prediction ``W'/t + CP``."""
    obs = observed.value.mean_power_w if observed is not None and is_computed(observed) else None
    predicted = fit.value.w_prime_j / d + fit.value.cp_w if d > 0 else None
    cov = (
        _coverage_for(observed)
        if observed is not None
        else _absent_coverage(Unavailable(UnavailableReason.INSUFFICIENT_DATA))
    )
    return SeriesPoint(
        duration_s=d,
        label=_duration_label(d),
        values={"power_watts": obs, "predicted_power_watts": predicted},
        coverage=cov,
    )


def _mmp_point(d: int, res: MetricResult[Any]) -> SeriesPoint:
    """One MMP duration point: best mean power for the duration (nullable per MMP-R5)."""
    val = res.value.mean_power_w if is_computed(res) else None
    return SeriesPoint(
        duration_s=d,
        label=_duration_label(d),
        values={"power_watts": val},
        coverage=_coverage_for(res),
    )


def _coggan_point(activity_id: str, day: _dt.date, res: MetricResult[Any]) -> SeriesPoint:
    """One per-activity Coggan point: TSS / IF / VI, each propagating unavailable."""
    if is_computed(res):
        b = res.value
        vals: dict[str, float | None] = {
            "tss": _value_of(b.tss),
            "intensity_factor": _value_of(b.if_),
            "variability_index": _value_of(b.variability_index),
        }
        cov = _coverage_for(b.tss)
    else:
        vals = {"tss": None, "intensity_factor": None, "variability_index": None}
        cov = _coverage_for(res)
    return SeriesPoint(
        local_date=day,
        activity_id=activity_id,
        label=_day_label(day),
        values=vals,
        coverage=cov,
    )


def _wbal_point(activity_id: str, day: _dt.date, res: MetricResult[Any]) -> SeriesPoint:
    """One per-activity W'-balance point: start/end/min W' in joules (may be negative)."""
    if is_computed(res):
        series = res.value.series
        start = float(series[0]) if series.size else None
        end = float(series[-1]) if series.size else None
        vals: dict[str, float | None] = {
            "w_prime_start_joules": start,
            "w_prime_end_joules": end,
            "w_prime_min_joules": float(res.value.w_prime_balance_min),
        }
    else:
        vals = dict.fromkeys(("w_prime_start_joules", "w_prime_end_joules", "w_prime_min_joules"))
    return SeriesPoint(
        local_date=day,
        activity_id=activity_id,
        label=_day_label(day),
        values=vals,
        coverage=_coverage_for(res),
    )


def _hrv_point(day: _dt.date, res: MetricResult[Any]) -> SeriesPoint:
    """One HRV day: time-domain metrics; freq-domain typed-unavailable when absent."""
    vals: dict[str, float | None] = dict.fromkeys(_HRV_KEYS)
    if is_computed(res):
        td = res.value
        vals.update(
            rmssd_ms=_opt_float(td.rmssd_ms),
            sdnn_ms=_opt_float(td.sdnn_ms),
            pnn50_pct=_opt_float(td.pnn50_pct),
            mean_nn_ms=_opt_float(td.mean_nn_ms),
        )
        cov = _present_coverage(res.quality)
    else:
        cov = _absent_coverage(res)
    return SeriesPoint(local_date=day, label=_day_label(day), values=vals, coverage=cov)


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
        local_date=day,
        activity_id=activity_id,
        label=_day_label(day),
        values=vals,
        coverage=_coverage_for(res),
    )


def _guard_trimp_domain(results: list[tuple[str, _dt.date, MetricResult[float]]]) -> None:
    """Map a wholly-uncomputable series to ``422`` with the TRIMP-R3 machine code.

    Every point missing thresholds → ``trimp_missing_thresholds``; an out-of-domain HR
    reserve → ``trimp_hr_domain_invalid``. A mix with ≥1 computed point degrades visibly
    per-point instead (API-R29). Raised as the catalog ``analytics-precondition-unmet``
    problem carrying the machine ``errors[].code`` (ERR-R9).
    """
    if any(is_computed(r) for _, _, r in results):
        return
    reasons = {r.reason for _, _, r in results if isinstance(r, Unavailable)}
    if UnavailableReason.OUT_OF_DOMAIN in reasons:
        raise precondition_unmet("trimp_hr_domain_invalid", "HR_max must exceed HR_rest")
    if reasons and reasons <= {UnavailableReason.MISSING_REQUIRED_INPUT}:
        raise precondition_unmet("trimp_missing_thresholds", "HR thresholds absent")


def _trimp_point(activity_id: str, day: _dt.date, res: MetricResult[float]) -> SeriesPoint:
    """One per-activity TRIMP point: Banister-HRR points (zonal variant null here)."""
    vals: dict[str, float | None] = {"trimp_points": _value_of(res), "trimp_zonal": None}
    return SeriesPoint(
        local_date=day,
        activity_id=activity_id,
        label=_day_label(day),
        values=vals,
        coverage=_coverage_for(res),
    )
