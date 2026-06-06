"""Performance Management Chart -- CTL / ATL / TSB (doc 40 section 3, PMC-R1..R6).

The PMC is exponentially weighted moving averages of a per-day training-load
series ``L(d)`` on a contiguous daily calendar grid:

.. code-block:: text

    alpha   = 1 - exp(-1/tau)                                       (PMC-R2)
    CTL(d)  = CTL(d-1) + alpha_CTL * (L(d) - CTL(d-1))   tau_CTL=42d (PMC-R1)
    ATL(d)  = ATL(d-1) + alpha_ATL * (L(d) - ATL(d-1))   tau_ATL=7d  (PMC-R1)
    TSB(d)  = CTL(d-1) - ATL(d-1)                  (previous-day)    (PMC-R1)

The decay-and-impulse ``exp(-1/tau)`` form is mandatory (NOT ``alpha=2/(N+1)``).

This module is a PURE, deterministic function (ANL-R2/R30): no I/O, no
wall-clock, no global state, no RNG. It returns one typed
:data:`~wattwise_core.analytics.result.MetricResult` per EVERY materialized
calendar date (PMC-R6 -- skipping a calendar day is a defect), never a bare
number (ANL-R3). Uncomputable paths fail closed (ANL-R4): a windowed query with
no derivable pre-window seed returns :class:`Unavailable` ``NOT_SEEDED`` (PMC-R5),
never a silent zero-seed.

Day kinds (PMC-R6):

* **true rest** -- a calendar date with a *known* ``L = 0`` (no activity, no open
  coverage gap). The EWMA decay advances normally; the day's
  :class:`~wattwise_core.analytics.result.QualityReport` is clean.
* **provisional** -- a calendar date that overlaps an OPEN coverage gap (load not
  yet known, may be back-filled). The EWMA still advances treating ``L = 0`` (the
  grid is never skipped, PMC-R6) but the day -- and every window result touching
  it -- is flagged ``provisional`` in its ``QualityReport``; the value may revise.

Caller encoding: a provisional day is signalled by ``None`` in the input series;
a known load (including a definitive ``L = 0`` true-rest day) is a finite float.

Windowed equivalence (PMC-R4, the DEFINING property): a query for ``[d0, d1]``
seeded from the carried-forward ``(CTL(d0-1), ATL(d0-1))`` MUST equal the
full-from-origin computation restricted to ``[d0, d1]`` to ``abs <= 1e-9*max(1,|v|)``.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from wattwise_core.analytics.constants import (
    ATL_TIME_CONSTANT_DAYS,
    CTL_TIME_CONSTANT_DAYS,
    WINDOWED_EQUIV_ABS_TOL,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)

__all__ = [
    "PmcDay",
    "PmcSeed",
    "ewma_alpha",
    "pmc",
]

# A provisional day decays as if L=0 but is flagged (PMC-R6); a known finite load
# (including a definitive true-rest L=0) is taken at face value. Accepted as a
# covariant Sequence (so a plain ``list[float]`` is fine) OR a date-keyed Mapping.
_DailyLoad = Sequence[float | None]
_DailyLoadInput = _DailyLoad | Mapping[_dt.date, float | None]


@dataclass(frozen=True, slots=True)
class PmcDay:
    """One materialized calendar day of the chart (the value of each Computed)."""

    ctl: float
    atl: float
    tsb: float


@dataclass(frozen=True, slots=True)
class PmcSeed:
    """Carried-forward pre-window state ``(CTL(d0-1), ATL(d0-1))`` (PMC-R3)."""

    ctl_prev: float
    atl_prev: float


def ewma_alpha(tau_days: float) -> float:
    """``alpha = 1 - exp(-1/tau)`` -- the impulse-response EWMA factor (PMC-R2).

    NOT the ``alpha = 2/(N+1)`` form. ``tau`` must be a positive, finite number
    of days.
    """
    if not math.isfinite(tau_days) or tau_days <= 0.0:
        raise ValueError("tau must be a positive finite number of days (PMC-R2)")
    return 1.0 - math.exp(-1.0 / tau_days)


def _normalize_input(
    daily_load: _DailyLoadInput,
) -> tuple[list[float | None], list[_dt.date] | None]:
    """Return ``(loads, dates)`` -- a dense per-calendar-day series.

    A ``dict`` keyed by date is densified across EVERY calendar day from the min
    to the max key (PMC-R6: no calendar day may be skipped); a missing key in the
    span becomes a provisional (``None``) day. A list is already a dense run of
    consecutive calendar days; ``dates`` is ``None`` (positional).
    """
    if isinstance(daily_load, Mapping):
        if not daily_load:
            return [], []
        for k in daily_load:
            if not isinstance(k, _dt.date):
                raise TypeError("dict keys must be datetime.date (local_date, PMC-R6)")
        d0 = min(daily_load)
        d1 = max(daily_load)
        dates = [d0 + _dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
        loads = [daily_load.get(d) for d in dates]
        return loads, dates
    return list(daily_load), None


def _validate_loads(loads: list[float | None]) -> None:
    """A known load must be a finite real (ANL-R4/R32). ``None`` = provisional."""
    for x in loads:
        if x is None:
            continue
        if not math.isfinite(float(x)):
            raise ValueError("daily load must be finite or None (provisional) (ANL-R32)")


def _resolve_window(window: tuple[int, int] | None, n: int) -> tuple[int, int]:
    """Resolve the inclusive integer window bounds within the dense grid."""
    if window is None:
        return 0, n - 1
    d0, d1 = window
    out_of_range = (
        d0 < 0
        or d1 < d0
        or (n > 0 and d1 >= n)
        or (n == 0 and (d0 != 0 or d1 != -1))
    )
    if out_of_range:
        raise ValueError("window indices out of range of the daily grid")
    return d0, d1


def _resolve_seed(
    seed: PmcSeed | tuple[float, float] | None, d0: int
) -> tuple[float, float] | Unavailable:
    """Resolve ``(CTL(d0-1), ATL(d0-1))`` per PMC-R3/R5, or fail closed (NOT_SEEDED)."""
    if seed is None:
        if d0 == 0:
            # First-ever day: the only honest origin seed (PMC-R3/R5).
            return 0.0, 0.0
        # Mid-history window with no carried-forward state: fail closed (PMC-R5).
        return Unavailable(
            UnavailableReason.NOT_SEEDED,
            detail=(
                f"no pre-window seed for d0={d0} (not the first day); "
                "refusing to zero-seed a mid-history window (PMC-R5)"
            ),
        )
    if isinstance(seed, PmcSeed):
        seed_ctl, seed_atl = seed.ctl_prev, seed.atl_prev
    else:
        seed_ctl, seed_atl = float(seed[0]), float(seed[1])
    if not (math.isfinite(seed_ctl) and math.isfinite(seed_atl)):
        return Unavailable(
            UnavailableReason.NOT_SEEDED,
            detail="seed (CTL(d0-1), ATL(d0-1)) is non-finite (PMC-R5/R32)",
        )
    return seed_ctl, seed_atl


def pmc(
    daily_load: _DailyLoadInput,
    *,
    tau_ctl: float = CTL_TIME_CONSTANT_DAYS,
    tau_atl: float = ATL_TIME_CONSTANT_DAYS,
    seed: PmcSeed | tuple[float, float] | None = None,
    window: tuple[int, int] | None = None,
    sport: str | None = None,
) -> list[MetricResult[PmcDay]]:
    """Compute the per-day PMC series (one :class:`MetricResult` per calendar day).

    Parameters
    ----------
    daily_load:
        Either a ``list[float | None]`` indexed by *consecutive* calendar days
        (index 0 = origin), or a ``dict[date, float | None]`` (densified across
        every calendar day in span -- PMC-R6). A finite float is a known daily
        load (``0.0`` = definitive true-rest day); ``None`` marks a *provisional*
        day overlapping an open coverage gap (decays as ``L = 0`` but flagged).
    tau_ctl, tau_atl:
        EWMA time constants in days (default 42 / 7, PMC-R1). Configurable.
    seed:
        Carried-forward pre-window state ``(CTL(d0-1), ATL(d0-1))`` (PMC-R3). When
        ``window`` starts mid-history this seed is REQUIRED; ``None`` then means no
        valid seed -> :class:`Unavailable` ``NOT_SEEDED`` (PMC-R5). When ``window``
        is absent (or starts at the first-ever day, index 0) the seed defaults to
        the only honest origin seed ``(0, 0)``.
    window:
        Optional inclusive integer index window ``(d0, d1)`` into the (densified)
        series. The returned list covers exactly ``[d0, d1]``. By construction the
        result equals the full-from-origin computation restricted to that window to
        ``abs <= 1e-9*max(1,|v|)`` (PMC-R4) WHEN ``seed`` is the carried-forward
        state for ``d0``.
    sport:
        Optional resolved sport recorded in lineage (PMC is sport-partitioned via
        its load series upstream; the recurrence itself is sport-agnostic).

    Returns
    -------
    A list of :data:`MetricResult` -- one per materialized calendar day in scope,
    in calendar order, NEVER skipping a day (PMC-R6). On the no-seed mid-history
    path a single-element ``[Unavailable(NOT_SEEDED)]`` is returned.
    """
    alpha_ctl = ewma_alpha(tau_ctl)
    alpha_atl = ewma_alpha(tau_atl)

    loads, dates = _normalize_input(daily_load)
    _validate_loads(loads)
    n = len(loads)

    d0, d1 = _resolve_window(window, n)

    resolved_seed = _resolve_seed(seed, d0)
    if isinstance(resolved_seed, Unavailable):
        return [resolved_seed]
    seed_ctl, seed_atl = resolved_seed

    if n == 0:
        return []

    return _run(
        loads=loads,
        dates=dates,
        d0=d0,
        d1=d1,
        seed_ctl=seed_ctl,
        seed_atl=seed_atl,
        alpha_ctl=alpha_ctl,
        alpha_atl=alpha_atl,
        tau_ctl=tau_ctl,
        tau_atl=tau_atl,
        sport=sport,
    )


def _pmc_day_result(
    *,
    ctl: float,
    atl: float,
    tsb: float,
    provisional: bool,
    day_index: int,
    local_date: _dt.date | None,
    tau_ctl: float,
    tau_atl: float,
    sport: str | None,
) -> MetricResult[PmcDay]:
    """Build one day's :class:`PmcDay` result from the recurrence outputs (PMC-R1).

    A provisional day (the load was a gap) carries reduced confidence and a gap-flag in
    its :class:`QualityReport`. A non-finite CTL/ATL/TSB fails closed to ``OUT_OF_DOMAIN``
    so no NaN/Inf escapes into a ``Computed`` (ANL-R32).
    """
    if not (math.isfinite(ctl) and math.isfinite(atl) and math.isfinite(tsb)):
        # No NaN/Inf may escape into a Computed (ANL-R32).
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            detail="non-finite PMC value (ANL-R32)",
        )

    quality = QualityReport(
        coverage_fraction=1.0,
        sample_rate_hz=None,
        gap_count=1 if provisional else 0,
        confidence=0.5 if provisional else 1.0,
        extra={
            "provisional": provisional,
            "day_kind": "provisional" if provisional else "true_rest_or_load",
            "tau_ctl_days": tau_ctl,
            "tau_atl_days": tau_atl,
        },
    )
    lineage = InputLineage(
        sport=sport,
        channels=("daily_load",),
        reference_params={
            "tau_ctl_days": tau_ctl,
            "tau_atl_days": tau_atl,
            "local_date": local_date.isoformat() if local_date is not None else None,
            "day_index": day_index,
        },
    )
    return Computed(
        value=PmcDay(ctl=ctl, atl=atl, tsb=tsb),
        quality=quality,
        provenance=lineage,
    )


def _run(
    *,
    loads: list[float | None],
    dates: list[_dt.date] | None,
    d0: int,
    d1: int,
    seed_ctl: float,
    seed_atl: float,
    alpha_ctl: float,
    alpha_atl: float,
    tau_ctl: float,
    tau_atl: float,
    sport: str | None,
) -> list[MetricResult[PmcDay]]:
    """Drive the per-day recurrence over ``[d0, d1]`` from a resolved seed."""
    results: list[MetricResult[PmcDay]] = []
    ctl_prev = seed_ctl
    atl_prev = seed_atl
    for i in range(d0, d1 + 1):
        raw = loads[i]
        provisional = raw is None
        load = 0.0 if raw is None else float(raw)

        # TSB is the PREVIOUS-day balance, evaluated BEFORE today's impulse (PMC-R1).
        tsb = ctl_prev - atl_prev
        ctl = ctl_prev + alpha_ctl * (load - ctl_prev)
        atl = atl_prev + alpha_atl * (load - atl_prev)

        results.append(
            _pmc_day_result(
                ctl=ctl,
                atl=atl,
                tsb=tsb,
                provisional=provisional,
                day_index=i,
                local_date=dates[i] if dates is not None else None,
                tau_ctl=tau_ctl,
                tau_atl=tau_atl,
                sport=sport,
            )
        )
        ctl_prev, atl_prev = ctl, atl

    return results


def windowed_equiv_tol(value: float) -> float:
    """The PMC-R4 windowed-equivalence tolerance ``1e-9*max(1,|v|)`` (test helper)."""
    return WINDOWED_EQUIV_ABS_TOL * max(1.0, abs(value))
