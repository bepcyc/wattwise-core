"""Mean-Maximal Power curve, Critical-Power / W-prime fit, and best efforts (doc 40).

This module realizes the cycling-power **power-curve** family:

* :func:`mmp` — the Mean-Maximal-Power curve (MMP-R1..R5): for each duration ``d``
  in the grid, ``MMP(d)`` is the maximum, over all start offsets, of the arithmetic
  mean power across any *contiguous valid* ``d``-second window of resampled 1 Hz
  power. A window is valid only if it contains ``d`` contiguous non-gap seconds
  (a gap longer than ``max_interp_gap_s`` invalidates it). It is non-increasing in
  ``d`` by construction (MMP-R3) — never clamped to enforce that. A duration with no
  valid window fails closed to ``Unavailable(INSUFFICIENT_DATA)`` while shorter
  durations stay ``Computed`` (MMP-R5); a missing power channel fails closed to
  ``Unavailable(MISSING_REQUIRED_INPUT)``.
* :func:`best_effort` — the best effort for a duration is *derived from* the MMP curve
  (BEST-R1): it equals ``MMP(d)`` exactly, with the same provenance window. There is
  no second maximization path.
* :func:`cp_wprime` — the 2-parameter critical-power fit (CP-R1..R6). It *consumes*
  MMP points (never recomputes maxima) and fits the linear work-time model
  ``W(t) = W-prime + CP·t`` (with ``W = P·t``) by ordinary least squares. Gates
  (CP-R3/R4): ≥ ``CP_MIN_POINTS`` distinct durations, ``max/min ≥ CP_DURATION_RATIO_MIN``,
  ``R² ≥ CP_R2_MIN``, and ``CP > 0`` and ``W-prime > 0``. Gate failure fails closed to
  ``Unavailable(INSUFFICIENT_DATA)`` (too few / too clustered) or
  ``Unavailable(POOR_FIT)`` (R²/sign), never clamped or fabricated. A contributing
  duration *strictly* above ``CP_LONG_DURATION_BIAS_S`` raises a non-blocking
  long-duration-bias quality flag (CP-R6); the 1200 s endpoint does not trip it.

All functions are pure and deterministic (ANL-R2/R30): no I/O, no wall-clock, no RNG,
no global mutable state. Every result is a typed :data:`MetricResult` envelope
(ANL-R3) and fails closed (ANL-R4) — never returns 0, a clamped/default value, or a
NaN/Inf inside a ``Computed`` (ANL-R32). The fit uses :func:`numpy.polyfit`, a
deterministic closed-form least-squares solver (ANL-R30).

These metrics are cycling-power-specific (doc 40 §5): they require a true mechanical
power channel and are reported per the cycling-power applicability map.

Requirements implemented: MMP-R1, MMP-R2, MMP-R3, MMP-R4, MMP-R5, BEST-R1, BEST-R2,
BEST-R3, BEST-R4, CP-R1, CP-R2, CP-R3, CP-R4, CP-R5, CP-R6, ANL-R2, ANL-R3, ANL-R4,
ANL-R5, ANL-R30, ANL-R31, ANL-R32, ANL-R33.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, replace

import numpy as np

from wattwise_core.analytics.constants import MMP_DURATION_GRID_S

# CP-W' fitting lives in the sibling :mod:`cp` module (QUAL-R9 module-size split);
# its public names are re-exported here so callers/tests keep importing them from
# ``wattwise_core.analytics.mmp_cp`` unchanged.
from wattwise_core.analytics.cp import (
    CPFit,
    _ols_standard_errors,
    cp_wprime,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)
from wattwise_core.analytics.series import (
    DEFAULT_MAX_INTERP_GAP_S,
    FloatArray,
)

# Cycling-power family: a true mechanical power channel is required (doc 40 §5).
APPLICABLE_SPORTS: tuple[str, ...] = ("cycling",)


@dataclass(frozen=True, slots=True)
class MMPWindow:
    """Provenance of a single MMP / best-effort peak: the winning window + value.

    ``duration_s`` is the *requested* curve duration (the x-axis grid point). The
    achieving effort is the contiguous window ``[start_index_s, end_index_s]`` whose
    real length is ``window_len_s`` seconds (``>= duration_s``, since the curve is the
    best sustainable average for AT LEAST ``duration_s`` seconds; MMP-R1/R3).
    ``mean_power_w`` is that window's arithmetic mean power (W).
    """

    duration_s: int  # requested curve duration (x-axis point)
    mean_power_w: float
    start_index_s: int  # 0-based offset into the 1 Hz power array
    end_index_s: int  # inclusive end offset (== start_index_s + window_len_s - 1)
    window_len_s: int  # real length of the achieving window (>= duration_s)


def _exact_window_peak(
    csum_valid: FloatArray, csum_vals: FloatArray, n: int, d: int
) -> tuple[float, int] | None:
    """Peak mean over contiguous *valid* windows of EXACTLY length ``d`` (MMP-R1).

    A window ``[i, i+d)`` is valid only if all ``d`` of its seconds are non-gap
    (a gap longer than ``max_interp_gap_s`` is already ``NaN`` from the resampler).
    Returns ``(best_mean, best_start)`` or ``None`` when no valid length-``d`` window
    exists. The first (smallest-offset) maximal window wins ties, for deterministic
    provenance (ANL-R30). Uses prefix sums for O(1) per-window queries.
    """
    if d <= 0 or n < d:
        return None
    best_mean = -np.inf
    best_start = -1
    for i in range(0, n - d + 1):
        j = i + d
        if csum_valid[j] - csum_valid[i] != d:
            continue  # window straddles at least one gap second -> invalid
        mean = (csum_vals[j] - csum_vals[i]) / d
        if mean > best_mean:
            best_mean = mean
            best_start = i
    if best_start < 0:
        return None
    return float(best_mean), best_start


def mmp(
    power_1hz: FloatArray,
    grid: tuple[int, ...] = MMP_DURATION_GRID_S,
    *,
    max_interp_gap_s: float = DEFAULT_MAX_INTERP_GAP_S,
    sport: str = "cycling",
) -> dict[int, MetricResult[MMPWindow]]:
    """Mean-Maximal-Power curve over a duration grid (MMP-R1/R2/R3/R5).

    ``power_1hz`` is a uniform 1 Hz power array (``NaN`` = gap), already resampled
    with :func:`~wattwise_core.analytics.series.resample_to_1hz` so that gaps longer
    than ``max_interp_gap_s`` are ``NaN`` (MMP-R1). For each duration ``d`` in
    ``grid`` the result maps ``d`` to a :class:`Computed` carrying the
    :class:`MMPWindow` peak (mean power + winning window) when at least one valid
    contiguous ``d``-second window exists, otherwise to
    ``Unavailable(INSUFFICIENT_DATA)`` (MMP-R5) — a partially-available curve is
    normal, never collapsed.

    A wholly-empty power channel (no array / all gaps and the grid still wants a
    window) yields per-duration ``Unavailable`` results. A power array with *zero*
    samples maps every duration to ``MISSING_REQUIRED_INPUT`` (MMP-R5): there is no
    channel to maximize over.

    The curve is non-increasing in ``d`` by construction (MMP-R3): ``MMP(d)`` is the
    best mean over any valid window of length ``>= d``, so the achieving effort for a
    longer duration can never exceed a shorter one — no clamping or fabrication. The
    ``MMPWindow`` carries the real achieving window (``window_len_s >= duration_s``).
    The ``max_interp_gap_s`` parameter is recorded for lineage only; gap encoding is
    already done by the resampler.
    """
    results: dict[int, MetricResult[MMPWindow]] = {}
    n = power_1hz.size

    # Mean-maximal POWER is a cycling-power-specific metric (ANL-R11/R12): requested for a
    # sport without mechanical power it is NOT applicable — return a typed unavailable,
    # never a plausible number from a power channel that does not mean the same thing.
    if sport not in APPLICABLE_SPORTS:
        return {
            int(d): Unavailable(
                reason=UnavailableReason.NOT_APPLICABLE_FOR_SPORT,
                detail=f"mean-maximal power is not defined for sport {sport!r}",
            )
            for d in grid
        }

    # No power channel at all -> the required input is absent (MMP-R5).
    if n == 0:
        for d in grid:
            results[int(d)] = Unavailable(
                reason=UnavailableReason.MISSING_REQUIRED_INPUT,
                detail="no power samples in stream",
            )
        return results

    valid_mask = ~np.isnan(power_1hz)
    coverage = int(valid_mask.sum()) / n if n else 0.0
    gap_count = _count_gaps(valid_mask)

    exact_peak = _exact_peaks_by_length(power_1hz, valid_mask, grid, n)

    for d in grid:
        di = int(d)
        results[di] = _mmp_result_for_duration(
            exact_peak,
            di,
            n,
            coverage=coverage,
            gap_count=gap_count,
            max_interp_gap_s=max_interp_gap_s,
            sport=sport,
        )
    return results


def _exact_peaks_by_length(
    power_1hz: FloatArray,
    valid_mask: FloatArray,
    grid: tuple[int, ...],
    n: int,
) -> dict[int, tuple[float, int]]:
    """Exactly-length-``L`` peak ``(mean, start)`` for every ``L`` in ``[min_grid, n]``.

    Builds the prefix sums once (ANL-R30 determinism) and maximises each integer
    length so the at-least-``d`` envelope (MMP-R3) draws from real, gap-free windows
    (MMP-R1). Lengths shorter than the smallest grid duration are never needed.
    """
    # Prefix sums for O(1) window queries, computed once (ANL-R30 determinism).
    valid_int = valid_mask.astype(np.int64)
    filled = np.where(valid_mask, power_1hz, 0.0)
    csum_valid = np.concatenate(([0], np.cumsum(valid_int))).astype(np.float64)
    csum_vals = np.concatenate(([0.0], np.cumsum(filled)))

    grid_ints = [int(d) for d in grid]
    min_d = min(grid_ints) if grid_ints else 1
    exact_peak: dict[int, tuple[float, int]] = {}
    for length in range(min_d, n + 1):
        peak = _exact_window_peak(csum_valid, csum_vals, n, length)
        if peak is not None:
            exact_peak[length] = peak
    return exact_peak


def _mmp_result_for_duration(
    exact_peak: dict[int, tuple[float, int]],
    di: int,
    n: int,
    *,
    coverage: float,
    gap_count: int,
    max_interp_gap_s: float,
    sport: str,
) -> MetricResult[MMPWindow]:
    """The MMP result for one grid duration: the envelope window, or fail closed.

    No valid window of length ``>= di`` ⇒ ``INSUFFICIENT_DATA`` (MMP-R5; a partially
    available curve is normal). A non-finite mean ⇒ ``OUT_OF_DOMAIN`` (ANL-R32).
    Otherwise a :class:`Computed` carrying the achieving :class:`MMPWindow`.
    """
    win = _at_least_envelope(exact_peak, di, n)
    if win is None:
        return Unavailable(
            reason=UnavailableReason.INSUFFICIENT_DATA,
            detail=(f"no valid contiguous window of length >= {di}s (longest valid run too short)"),
        )
    if not np.isfinite(win.mean_power_w):  # pragma: no cover - guard (ANL-R32)
        return Unavailable(
            reason=UnavailableReason.OUT_OF_DOMAIN,
            detail=f"non-finite MMP({di}s)",
        )
    return Computed(
        value=win,
        quality=QualityReport(
            coverage_fraction=coverage,
            sample_rate_hz=1.0,
            gap_count=gap_count,
            confidence=1.0,
            extra={
                "duration_s": di,
                "window_len_s": win.window_len_s,
                "window_valid": True,
            },
        ),
        provenance=InputLineage(
            sport=sport,
            channels=("power",),
            reference_params={"max_interp_gap_s": max_interp_gap_s},
        ),
    )


def _at_least_envelope(
    exact_peak: dict[int, tuple[float, int]], d: int, n: int
) -> MMPWindow | None:
    """At-least-``d`` envelope from precomputed exactly-length peaks (MMP-R1/R3).

    ``MMP(d) = max over L >= d of exactly-length-L peak``. Returns the achieving
    window (shortest, earliest on ties). ``None`` when no valid window of length
    ``>= d`` exists (MMP-R5).
    """
    best_mean = -np.inf
    best_len = -1
    best_start = -1
    for length in range(d, n + 1):
        peak = exact_peak.get(length)
        if peak is None:
            continue
        mean, start = peak
        if mean > best_mean:  # strict -> shortest/earliest achieving window on ties
            best_mean = mean
            best_len = length
            best_start = start
    if best_start < 0:
        return None
    return MMPWindow(
        duration_s=d,
        mean_power_w=float(best_mean),
        start_index_s=best_start,
        end_index_s=best_start + best_len - 1,
        window_len_s=best_len,
    )


def best_effort(
    power_1hz: FloatArray,
    d: int,
    *,
    max_interp_gap_s: float = DEFAULT_MAX_INTERP_GAP_S,
    sport: str = "cycling",
) -> MetricResult[MMPWindow]:
    """Best effort for duration ``d``, DERIVED from MMP (BEST-R1).

    A best effort is exactly ``MMP(d)`` with its provenance window — there is no
    second maximization path (single source of truth). No valid window in scope
    fails closed to ``Unavailable(INSUFFICIENT_DATA)`` (BEST-R3); a zero-length power
    channel fails closed to ``MISSING_REQUIRED_INPUT`` (BEST is power-derived).
    """
    return mmp(
        power_1hz,
        (int(d),),
        max_interp_gap_s=max_interp_gap_s,
        sport=sport,
    )[int(d)]


def _count_gaps(valid_mask: FloatArray) -> int:
    """Count contiguous gap runs (a transition into a ``NaN`` run)."""
    gaps = 0
    prev_valid = True
    for v in valid_mask:
        if not v and prev_valid:
            gaps += 1
        prev_valid = bool(v)
    return gaps


def stamp_curve_origin(
    res: Computed[MMPWindow], *, activity_id: str, local_date: _dt.date
) -> Computed[MMPWindow]:
    """Carry the originating activity into an aggregate-curve duration's lineage (MMP-R4).

    The single-activity :func:`mmp` is provenance-blind to which activity it ran over, so the
    multi-activity aggregator stamps the WINNING activity's identity onto the per-duration
    result: which activity produced this duration's peak (``activity_id`` in ``InputLineage``)
    and its ``local_date`` (in ``reference_params``), so a best-effort consumer can cite "your
    best 5-minute power came from <activity on date>" (BEST-R2). Takes opaque primitives only —
    never a source NAME — so formula-layer lineage stays provenance-blind (ANL-R33).
    """
    lineage = replace(
        res.provenance,
        activity_ids=(activity_id,),
        reference_params={**res.provenance.reference_params, "local_date": local_date},
    )
    return replace(res, provenance=lineage)


__all__ = [
    "APPLICABLE_SPORTS",
    "CPFit",
    "MMPWindow",
    # Re-exported from the sibling :mod:`cp` module so callers/tests keep importing
    # the CP-W' API (incl. the OLS-SE helper used by the property suite) from here.
    "_ols_standard_errors",
    "best_effort",
    "cp_wprime",
    "mmp",
    "stamp_curve_origin",
]
