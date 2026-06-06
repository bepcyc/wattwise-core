"""Critical-Power / W-prime 2-parameter fit (doc 40 §, CP-R1..R6).

The CP fit *consumes* MMP points (never recomputes maxima — CP-R1) and fits the
linear work-time model ``W(t) = W-prime + CP·t`` (with ``W = P·t``) by ordinary least
squares (:func:`numpy.polyfit`, deterministic — ANL-R30). Gates (CP-R3/R4): at least
``CP_MIN_POINTS`` distinct in-domain durations, a wide-enough duration spread
(``max/min >= CP_DURATION_RATIO_MIN``), ``R² >= CP_R2_MIN``, and physically meaningful
signs (``CP > 0`` and ``W-prime > 0``). Gate failure fails closed to
``Unavailable(INSUFFICIENT_DATA)`` (too few / too clustered) or
``Unavailable(POOR_FIT)`` (R²/sign), never clamped or fabricated; a NaN/Inf fit
parameter is ``OUT_OF_DOMAIN`` (ANL-R32). A contributing duration *strictly* above
``CP_LONG_DURATION_BIAS_S`` raises a non-blocking long-duration-bias quality flag
(CP-R6); the threshold endpoint itself does not trip it.

Split out of :mod:`wattwise_core.analytics.mmp_cp` for the module size ceiling
(QUAL-R9); :class:`CPFit` and :func:`cp_wprime` are re-exported from ``mmp_cp`` so
callers and tests can keep importing them from there unchanged.

Requirements implemented: CP-R1..R6, ANL-R2, ANL-R3, ANL-R4, ANL-R5, ANL-R30,
ANL-R32, ANL-R33.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wattwise_core.analytics.constants import (
    CP_DOMAIN_MAX_S,
    CP_DOMAIN_MIN_S,
    CP_DURATION_RATIO_MIN,
    CP_LONG_DURATION_BIAS_S,
    CP_MIN_POINTS,
    CP_R2_MIN,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)
from wattwise_core.analytics.series import FloatArray


@dataclass(frozen=True, slots=True)
class CPFit:
    """A critical-power fit result (CP-R1/R3): parameters + goodness-of-fit."""

    cp_w: float
    w_prime_j: float
    r2: float
    se_cp: float
    se_wprime: float
    residuals: tuple[float, ...]


def _ols_standard_errors(
    t: FloatArray, residuals: FloatArray, n_points: int
) -> tuple[float, float]:
    """Standard errors of slope (CP) and intercept (W-prime) for OLS (CP-R3).

    Closed-form OLS SEs: with residual variance ``s2 = SSR / (n - 2)`` and
    ``S_tt = sum((t - t_bar)**2)``,
    ``SE(slope) = sqrt(s2 / S_tt)`` and
    ``SE(intercept) = sqrt(s2 * (1/n + t_bar**2 / S_tt))``.

    With exactly 2 points (the minimum the gates allow only if configured below
    ``CP_MIN_POINTS``) the line is interpolating, residual variance is 0, and SEs are
    0. ``n - 2 = 0`` is guarded to avoid division-by-zero (returns 0/0 reported as
    0.0, never NaN inside a Computed -- ANL-R32).
    """
    dof = n_points - 2
    s_tt = float(np.sum((t - np.mean(t)) ** 2))
    ssr = float(np.sum(residuals**2))
    if dof <= 0 or s_tt <= 0.0:
        # Interpolating / degenerate spread: zero estimable error.
        return 0.0, 0.0
    s2 = ssr / dof
    t_bar = float(np.mean(t))
    se_slope = float(np.sqrt(s2 / s_tt))
    se_intercept = float(np.sqrt(s2 * (1.0 / n_points + t_bar**2 / s_tt)))
    return se_slope, se_intercept


def _select_cp_domain_points(
    mmp_points: dict[int, float],
    *,
    domain_min_s: int,
    domain_max_s: int,
    min_points: int,
    duration_ratio_min: float,
) -> tuple[list[int], dict[int, float]] | Unavailable:
    """Select in-domain distinct durations and apply the pre-fit gates (CP-R2/R3/R4).

    Keeps only finite-power points whose duration is in ``[domain_min_s, domain_max_s]``
    (CP-R2; a NaN/Inf power never enters the fit — ANL-R32). Fails closed with
    ``INSUFFICIENT_DATA`` if fewer than ``min_points`` distinct durations remain, or if
    the durations are too clustered (``max/min < duration_ratio_min``). Returns
    ``(distinct_durations, in_domain)`` on success.
    """
    in_domain: dict[int, float] = {}
    for dur, pwr in mmp_points.items():
        di = int(dur)
        if di < domain_min_s or di > domain_max_s:
            continue  # outside valid domain -> excluded from the fit (CP-R2)
        if not np.isfinite(pwr):
            continue  # never let a NaN/Inf power enter the fit (ANL-R32)
        in_domain[di] = float(pwr)

    distinct_durations = sorted(in_domain)
    n_points = len(distinct_durations)

    # Gate: enough distinct durations (CP-R3 / CP-R4 -> INSUFFICIENT_DATA).
    if n_points < min_points:
        return Unavailable(
            reason=UnavailableReason.INSUFFICIENT_DATA,
            detail=(
                f"need >= {min_points} distinct in-domain durations, "
                f"got {n_points} in [{domain_min_s}, {domain_max_s}]s"
            ),
        )

    # Gate: durations span a wide-enough range (not clustered) (CP-R3/R4).
    min_dur = distinct_durations[0]
    max_dur = distinct_durations[-1]
    if min_dur <= 0 or (max_dur / min_dur) < duration_ratio_min:
        return Unavailable(
            reason=UnavailableReason.INSUFFICIENT_DATA,
            detail=(
                f"durations too clustered: max/min = "
                f"{max_dur / min_dur:.3f} < {duration_ratio_min}"
            ),
        )
    return distinct_durations, in_domain


def _fit_cp_model(
    distinct_durations: list[int],
    in_domain: dict[int, float],
    *,
    r2_min: float,
) -> CPFit | Unavailable:
    """OLS fit of ``W(t) = W' + CP·t`` plus the fit-quality gates (CP-R1/R3/R4).

    Solves the linear work-time regression with :func:`numpy.polyfit` (deterministic;
    ANL-R30) and returns a :class:`CPFit` on success. Fails closed: zero work variance
    ⇒ ``POOR_FIT``; non-finite params ⇒ ``OUT_OF_DOMAIN``; ``R² < r2_min`` or a
    non-positive CP/W' (wrong sign) ⇒ ``POOR_FIT``.
    """
    n_points = len(distinct_durations)
    # Build the linear work-time system W = W' + CP*t, where W = P*t (CP-R1).
    t = np.array(distinct_durations, dtype=np.float64)
    p = np.array([in_domain[d] for d in distinct_durations], dtype=np.float64)
    w = p * t  # work in joules

    # OLS via numpy.polyfit (deterministic closed-form least squares; ANL-R30).
    # polyfit(t, w, 1) -> [slope, intercept] = [CP, W'].
    coeffs = np.polyfit(t, w, 1)
    cp_w = float(coeffs[0])
    w_prime_j = float(coeffs[1])

    w_pred = cp_w * t + w_prime_j
    residuals = w - w_pred

    # R² of the linear work-time regression (CP-R3).
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((w - np.mean(w)) ** 2))
    if ss_tot <= 0.0:
        # All work values identical -> degenerate, cannot fit a meaningful line.
        return Unavailable(
            reason=UnavailableReason.POOR_FIT,
            detail="zero variance in work-time data; no determinable slope",
        )
    r2 = 1.0 - ss_res / ss_tot

    if not np.isfinite(cp_w) or not np.isfinite(w_prime_j) or not np.isfinite(r2):
        return Unavailable(
            reason=UnavailableReason.OUT_OF_DOMAIN,
            detail="non-finite fit parameters",
        )

    # Standard errors of slope (CP) and intercept (W') from OLS (CP-R3).
    se_cp, se_wprime = _ols_standard_errors(t, residuals, n_points)

    # Gate: goodness-of-fit (CP-R3/R4 -> POOR_FIT).
    if r2 < r2_min:
        return Unavailable(
            reason=UnavailableReason.POOR_FIT,
            detail=f"R^2 = {r2:.6f} < r2_min = {r2_min}",
        )

    # Gate: physically meaningful sign (CP-R3/R4 -> POOR_FIT).
    if cp_w <= 0.0 or w_prime_j <= 0.0:
        return Unavailable(
            reason=UnavailableReason.POOR_FIT,
            detail=f"non-positive fit: CP = {cp_w:.4f} W, W' = {w_prime_j:.1f} J",
        )

    return CPFit(
        cp_w=cp_w,
        w_prime_j=w_prime_j,
        r2=r2,
        se_cp=se_cp,
        se_wprime=se_wprime,
        residuals=tuple(float(r) for r in residuals),
    )


def _build_cp_result(
    fit: CPFit,
    distinct_durations: list[int],
    *,
    domain_min_s: int,
    domain_max_s: int,
    long_duration_bias_s: int,
    sport: str,
) -> Computed[CPFit]:
    """Assemble the ``Computed`` envelope with the long-duration-bias flag (CP-R3/R6).

    Any *contributing* duration *strictly* greater than ``long_duration_bias_s`` trips
    the non-blocking ``long_duration_bias`` flag and downgrades confidence (CP-R6); the
    threshold endpoint itself never trips it.
    """
    # Long-duration-bias flag: STRICTLY above threshold trips it (CP-R6).
    offending = [d for d in distinct_durations if d > long_duration_bias_s]
    long_bias = len(offending) > 0
    confidence = 0.75 if long_bias else 1.0

    extra: dict[str, object] = {
        "r2": fit.r2,
        "se_cp": fit.se_cp,
        "se_wprime": fit.se_wprime,
        "n_points": len(distinct_durations),
        "durations_s": distinct_durations,
        "long_duration_bias": long_bias,
    }
    if long_bias:
        extra["long_duration_bias_detail"] = {
            "offending_durations_s": offending,
            "threshold_s": long_duration_bias_s,
        }

    return Computed(
        value=fit,
        quality=QualityReport(
            coverage_fraction=1.0,
            sample_rate_hz=None,
            gap_count=0,
            confidence=confidence,
            extra=extra,
        ),
        provenance=InputLineage(
            sport=sport,
            channels=("power",),
            reference_params={
                "domain_min_s": domain_min_s,
                "domain_max_s": domain_max_s,
                "model": "linear_work_time",
            },
        ),
    )


def cp_wprime(
    mmp_points: dict[int, float],
    *,
    domain_min_s: int = CP_DOMAIN_MIN_S,
    domain_max_s: int = CP_DOMAIN_MAX_S,
    min_points: int = CP_MIN_POINTS,
    duration_ratio_min: float = CP_DURATION_RATIO_MIN,
    r2_min: float = CP_R2_MIN,
    long_duration_bias_s: int = CP_LONG_DURATION_BIAS_S,
    sport: str = "cycling",
) -> MetricResult[CPFit]:
    """Fit the 2-parameter critical-power model to MMP points (CP-R1..R6).

    ``mmp_points`` maps duration (seconds) → maximal mean power (W) from real efforts
    (CONSUMES the MMP curve, never recomputes maxima — CP-R1). Only points whose
    duration lies in the valid domain ``[domain_min_s, domain_max_s]`` participate in
    the fit (CP-R2); durations outside the domain are excluded. The fit is the linear
    work-time regression ``W(t) = W-prime + CP·t`` with ``W = P·t`` (CP-R1) solved by
    ordinary least squares (:func:`numpy.polyfit`, deterministic — ANL-R30).

    Gates (CP-R3/R4):

    * ``< min_points`` distinct in-domain durations → ``INSUFFICIENT_DATA``.
    * ``max_dur / min_dur < duration_ratio_min`` (too clustered) → ``INSUFFICIENT_DATA``.
    * ``R² < r2_min`` → ``POOR_FIT``.
    * ``CP ≤ 0`` or ``W-prime ≤ 0`` (wrong sign) → ``POOR_FIT``.

    A successful fit returns a :class:`Computed` carrying :class:`CPFit`
    (``cp_w``, ``w_prime_j``, ``r2``, ``se_cp``, ``se_wprime``, ``residuals``) with
    the goodness-of-fit in the :class:`QualityReport` (CP-R3). If any *contributing*
    duration is *strictly* greater than ``long_duration_bias_s`` the quality report
    carries a non-blocking ``long_duration_bias`` flag (CP-R6) with the offending
    durations and a downgraded confidence; the endpoint itself never trips it.
    """
    selection = _select_cp_domain_points(
        mmp_points,
        domain_min_s=domain_min_s,
        domain_max_s=domain_max_s,
        min_points=min_points,
        duration_ratio_min=duration_ratio_min,
    )
    if isinstance(selection, Unavailable):
        return selection
    distinct_durations, in_domain = selection

    fit = _fit_cp_model(distinct_durations, in_domain, r2_min=r2_min)
    if isinstance(fit, Unavailable):
        return fit

    return _build_cp_result(
        fit,
        distinct_durations,
        domain_min_s=domain_min_s,
        domain_max_s=domain_max_s,
        long_duration_bias_s=long_duration_bias_s,
        sport=sport,
    )


__all__ = [
    "CPFit",
    "_ols_standard_errors",
    "cp_wprime",
]
