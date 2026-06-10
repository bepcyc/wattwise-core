"""Banister-HRR TRIMP / HR-load — the canonical HR training load (doc 40 §7D).

Implements the ONE canonical HR-load variant (TRIMP-R1): Banister's exponential
TRIMP weighted by the heart-rate reserve (HRR), plus the distinctly-labelled
zone-weighted variant ``hr_load_zonal`` (TRIMP-R2). This load family is
sport-agnostic (doc 40 §5): any activity carrying an HR channel qualifies, so it
NEVER returns ``NOT_APPLICABLE_FOR_SPORT``.

Formulas (doc 40 §3, TRIMP-R1)::

    HRR(t) = (HR(t) - HR_rest) / (HR_max - HR_rest)  in [0, 1]
    TRIMP  = Σ over valid seconds  Δt_min · HRR(t) · a · e^(b · HRR(t))

``Δt_min`` is the per-sample duration in minutes (``1/60`` at 1 Hz). The sex
constants (``a`` multiplicative, ``b`` exponential — NEVER conflated) are the
published Banister fits (doc 40 §3 / constants.py)::

    male:    a = 0.64, b = 1.92
    female:  a = 0.86, b = 1.67

Fail-closed contract (doc 40 §6, ANL-R4):

* Absent HR channel / HR_max / HR_rest        -> ``MISSING_REQUIRED_INPUT`` (TRIMP-R3).
* Both present but ``HR_max <= HR_rest``       -> ``OUT_OF_DOMAIN`` (TRIMP-R3) — the
  HR reserve is non-positive; we never divide by it.
* No valid HR seconds at all                   -> ``INSUFFICIENT_DATA``.
* A non-finite TRIMP would result              -> ``OUT_OF_DOMAIN`` (ANL-R32).
* Absent sex                                   -> per ``require_sex``: either a
  documented sex-neutral default pair (recorded, reduced confidence, TRIMP-R1) or
  ``MISSING_REQUIRED_INPUT``.

``HRR`` samples outside ``[0, 1]`` are clamped to ``[0, 1]`` for *weighting only*
(default policy, TRIMP-R3); the clamped fraction is recorded in the
``QualityReport``. The raw HR is never altered.

Every metric here is a PURE function (no I/O, no wall-clock, no RNG, no global
state; ANL-R2/R30) returning a typed :data:`MetricResult` envelope (ANL-R3), never
a bare number. The produced :class:`Computed` is always labelled with its
``load_model`` (``hr_load`` or ``hr_load_zonal``, TRIMP-R2/R4) and never relabelled
as ``power_tss``.

Requirement IDs: TRIMP-R1, TRIMP-R2, TRIMP-R3, TRIMP-R4, TRIMP-R5;
ANL-R2/R3/R4/R5/R30/R31/R32/R33.
"""

from __future__ import annotations

import math

import numpy as np

from wattwise_core.analytics.constants import (
    TRIMP_A_FEMALE,
    TRIMP_A_MALE,
    TRIMP_B_FEMALE,
    TRIMP_B_MALE,
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
    Stream,
    resample_to_1hz,
)

# ``load_model`` labels owned by the HR-load family (subset of the closed LOAD-R2
# set). These strings are the mandatory, never-relabelled provenance tags
# (TRIMP-R2/R4): zonal is ``hr_load_zonal``, Banister is ``hr_load``; neither is
# ever surfaced as ``power_tss``.
LOAD_MODEL_HR_LOAD = "hr_load"
LOAD_MODEL_HR_LOAD_ZONAL = "hr_load_zonal"

# Documented sex-neutral default Banister pair (TRIMP-R1): used only when sex is
# absent AND the config does not require it. It is the unweighted average of the
# published male/female fits (a multiplicative, b exponential kept SEPARATE — never
# conflated). Recorded in lineage + flagged with reduced confidence so a consumer
# can see the value is sex-modelled, not sex-resolved. Defined here (not in
# constants.py) because the shared constants module owns only the two published
# per-sex pairs; this is a derived, metric-local degradation default.
# ANL-R11: sport-agnostic — meaningful for EVERY sport that supplies the required HR
# channel; ``None`` (not an enumerated set) is the declared sport-agnostic marker, so
# there is NO sport gate (absence of HR is MISSING_REQUIRED_INPUT, never
# NOT_APPLICABLE_FOR_SPORT).
APPLICABLE_SPORTS: None = None

TRIMP_A_SEX_NEUTRAL = (TRIMP_A_MALE + TRIMP_A_FEMALE) / 2.0  # = 0.75
TRIMP_B_SEX_NEUTRAL = (TRIMP_B_MALE + TRIMP_B_FEMALE) / 2.0  # = 1.795

# Confidence assigned to a sex-neutral (sex-absent) Banister result.
_SEX_NEUTRAL_CONFIDENCE = 0.7

# Recognised canonical sex tokens. Anything else (or ``None``) is treated as
# "sex absent" — there is no source-name branching, only a value lookup.
_SEX_CONSTANTS: dict[str, tuple[float, float]] = {
    "male": (TRIMP_A_MALE, TRIMP_B_MALE),
    "m": (TRIMP_A_MALE, TRIMP_B_MALE),
    "female": (TRIMP_A_FEMALE, TRIMP_B_FEMALE),
    "f": (TRIMP_A_FEMALE, TRIMP_B_FEMALE),
}


def _resolve_sex_constants(
    sex: str | None, *, require_sex: bool
) -> tuple[float, float, bool] | Unavailable:
    """Resolve the (a, b) Banister pair for ``sex``.

    Returns ``(a, b, sex_neutral)`` where ``sex_neutral`` is True iff the documented
    sex-neutral default was used. When ``sex`` is absent/unrecognised and
    ``require_sex`` is set, returns ``Unavailable(MISSING_REQUIRED_INPUT)``.
    """
    key = sex.strip().lower() if isinstance(sex, str) else None
    pair = _SEX_CONSTANTS.get(key) if key is not None else None
    if pair is not None:
        a, b = pair
        return a, b, False
    if require_sex:
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "athlete sex is required for the Banister-HRR sex constants but is absent",
        )
    return TRIMP_A_SEX_NEUTRAL, TRIMP_B_SEX_NEUTRAL, True


def _validate_banister_inputs(
    hr_stream: Stream | None,
    hr_max: float | None,
    hr_rest: float | None,
) -> tuple[Stream, float, float] | Unavailable:
    """Presence + HR-reserve domain gates for Banister-HRR (TRIMP-R1/R3).

    Absent ``hr_stream``/``hr_max``/``hr_rest`` ⇒ ``MISSING_REQUIRED_INPUT``; a
    non-finite or non-positive HR reserve (``hr_max <= hr_rest``) ⇒ ``OUT_OF_DOMAIN``
    so the reserve is never divided by (TRIMP-R3). Returns the narrowed
    ``(hr_stream, hr_max, hr_rest)`` when valid.
    """
    if hr_stream is None or hr_max is None or hr_rest is None:
        missing = [
            name
            for name, val in (
                ("hr_stream", hr_stream),
                ("hr_max", hr_max),
                ("hr_rest", hr_rest),
            )
            if val is None
        ]
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            f"required HR input(s) absent: {', '.join(missing)}",
        )

    # Present-but-invalid: a non-positive HR reserve. Never divide by it (TRIMP-R3).
    if not math.isfinite(hr_max) or not math.isfinite(hr_rest) or hr_max <= hr_rest:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            f"HR_max ({hr_max}) must be finite and strictly greater than "
            f"HR_rest ({hr_rest}); HR reserve is non-positive",
        )
    return hr_stream, float(hr_max), float(hr_rest)


def _build_banister_result(
    trimp: float,
    *,
    n_grid: int,
    n_valid: int,
    n_clamped: int,
    a: float,
    b: float,
    sex_neutral: bool,
    hr_max: float,
    hr_rest: float,
    sex: str | None,
    sport: str | None,
) -> Computed[float]:
    """Assemble the Banister-HRR ``Computed`` envelope (TRIMP-R1/R3).

    Confidence is downgraded when the sex-neutral default constants were used; the
    clamped-HRR fraction and the (a, b) constants are recorded in the
    :class:`QualityReport`.
    """
    coverage = n_valid / n_grid if n_grid > 0 else 0.0
    confidence = _SEX_NEUTRAL_CONFIDENCE if sex_neutral else 1.0
    quality = QualityReport(
        coverage_fraction=coverage,
        sample_rate_hz=1.0,
        gap_count=n_grid - n_valid,
        confidence=confidence,
        extra={
            "load_model": LOAD_MODEL_HR_LOAD,
            "clamped_hrr_fraction": n_clamped / n_valid,
            "n_valid_seconds": n_valid,
            "sex_neutral_constants": sex_neutral,
            "trimp_a": a,
            "trimp_b": b,
        },
    )
    provenance = InputLineage(
        sport=sport,
        channels=("heart_rate",),
        reference_params={
            "hr_max": float(hr_max),
            "hr_rest": float(hr_rest),
            "sex": sex if not sex_neutral else None,
            "load_model": LOAD_MODEL_HR_LOAD,
        },
    )
    return Computed(value=trimp, quality=quality, provenance=provenance)


def banister_hr_load(
    hr_stream: Stream | None,
    hr_max: float | None,
    hr_rest: float | None,
    sex: str | None,
    *,
    require_sex: bool = False,
    max_interp_gap_s: float = DEFAULT_MAX_INTERP_GAP_S,
    sport: str | None = None,
) -> MetricResult[float]:
    """Canonical Banister-HRR TRIMP / HR-load (TRIMP-R1).

    Resamples ``hr_stream`` to 1 Hz (so ``Δt_min = 1/60``), computes the
    heart-rate-reserve weighting and sums the exponential Banister kernel over every
    valid second. Returns a :class:`Computed` labelled ``load_model='hr_load'``.

    Fail-closed (doc 40 §6):

    * ``hr_stream``/``hr_max``/``hr_rest`` absent -> ``MISSING_REQUIRED_INPUT``.
    * ``hr_max <= hr_rest`` (non-positive reserve) -> ``OUT_OF_DOMAIN``.
    * no valid HR seconds -> ``INSUFFICIENT_DATA``.
    * non-finite result -> ``OUT_OF_DOMAIN`` (ANL-R32).
    * ``sex`` absent -> sex-neutral default (reduced confidence) or, if
      ``require_sex``, ``MISSING_REQUIRED_INPUT`` (TRIMP-R1).

    HRR outside ``[0, 1]`` is clamped to ``[0, 1]`` for weighting only; the clamped
    fraction is recorded in :class:`QualityReport` (TRIMP-R3).
    """
    validated = _validate_banister_inputs(hr_stream, hr_max, hr_rest)
    if isinstance(validated, Unavailable):
        return validated
    hr_stream, hr_max, hr_rest = validated

    sex_res = _resolve_sex_constants(sex, require_sex=require_sex)
    if isinstance(sex_res, Unavailable):
        return sex_res
    a, b, sex_neutral = sex_res

    hr_1hz = resample_to_1hz(hr_stream, max_interp_gap_s=max_interp_gap_s)
    n_grid = int(hr_1hz.size)
    valid_mask = ~np.isnan(hr_1hz)
    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid == 0:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            "no valid HR samples after resampling to 1 Hz",
        )

    reserve = float(hr_max) - float(hr_rest)
    hr_valid = hr_1hz[valid_mask]
    hrr_raw = (hr_valid - float(hr_rest)) / reserve

    # Clamp HRR to [0, 1] for weighting only (TRIMP-R3); record the clamped fraction.
    out_of_band = (hrr_raw < 0.0) | (hrr_raw > 1.0)
    n_clamped = int(np.count_nonzero(out_of_band))
    hrr = np.clip(hrr_raw, 0.0, 1.0)

    dt_min = 1.0 / 60.0  # 1 Hz sample duration in minutes (TRIMP-R1).
    per_second = dt_min * hrr * a * np.exp(b * hrr)
    trimp = float(np.sum(per_second))

    if not math.isfinite(trimp):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "Banister-HRR TRIMP evaluated to a non-finite value (ANL-R32)",
        )

    return _build_banister_result(
        trimp,
        n_grid=n_grid,
        n_valid=n_valid,
        n_clamped=n_clamped,
        a=a,
        b=b,
        sex_neutral=sex_neutral,
        hr_max=hr_max,
        hr_rest=hr_rest,
        sex=sex,
        sport=sport,
    )


def _validate_zone_spec(
    bounds: np.ndarray, wts: np.ndarray
) -> Unavailable | None:
    """Validate zone boundaries + weights; return a typed Unavailable or None.

    Domain preconditions (all -> ``OUT_OF_DOMAIN``): non-empty, finite, strictly
    ascending boundaries; finite weights with ``len(weights) == len(bounds) + 1``.
    """
    if bounds.size == 0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN, "zone_boundaries must be non-empty"
        )
    if not np.all(np.isfinite(bounds)):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN, "zone_boundaries must all be finite"
        )
    if np.any(np.diff(bounds) <= 0.0):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "zone_boundaries must be strictly ascending",
        )
    if wts.size != bounds.size + 1:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            f"weights length ({wts.size}) must equal len(zone_boundaries) + 1 "
            f"({bounds.size + 1})",
        )
    if not np.all(np.isfinite(wts)):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN, "weights must all be finite"
        )
    return None


def hr_load_zonal(
    hr_stream: Stream | None,
    zone_boundaries: list[float] | tuple[float, ...] | None,
    weights: list[float] | tuple[float, ...] | None,
    *,
    max_interp_gap_s: float = DEFAULT_MAX_INTERP_GAP_S,
    sport: str | None = None,
) -> MetricResult[float]:
    """Zone-weighted HR-load variant (TRIMP-R2), labelled ``load_model='hr_load_zonal'``.

    Time-in-zone (minutes) times per-zone weight, summed over the declared zone
    boundaries. With ``k`` ascending boundaries there are ``k + 1`` zones; ``weights``
    must therefore have length ``len(zone_boundaries) + 1``. A 1 Hz second whose HR is
    ``< zone_boundaries[0]`` falls in zone 0; ``>= zone_boundaries[-1]`` in the last
    zone.

    This variant is offered ONLY under its distinct label and is NEVER relabelled as
    the canonical Banister ``hr_load`` nor as ``power_tss`` (TRIMP-R2/R4).

    Fail-closed (doc 40 §6):

    * ``hr_stream``/``zone_boundaries``/``weights`` absent -> ``MISSING_REQUIRED_INPUT``.
    * empty boundaries, non-ascending/non-finite boundaries, a
      ``len(weights) != len(boundaries) + 1`` mismatch, or non-finite weights
      -> ``OUT_OF_DOMAIN``.
    * no valid HR seconds -> ``INSUFFICIENT_DATA``.
    * non-finite result -> ``OUT_OF_DOMAIN`` (ANL-R32).
    """
    if hr_stream is None or zone_boundaries is None or weights is None:
        missing = [
            name
            for name, val in (
                ("hr_stream", hr_stream),
                ("zone_boundaries", zone_boundaries),
                ("weights", weights),
            )
            if val is None
        ]
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            f"required zonal input(s) absent: {', '.join(missing)}",
        )

    bounds = np.asarray(zone_boundaries, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)

    spec_error = _validate_zone_spec(bounds, wts)
    if spec_error is not None:
        return spec_error

    hr_1hz = resample_to_1hz(hr_stream, max_interp_gap_s=max_interp_gap_s)
    n_grid = int(hr_1hz.size)
    valid_mask = ~np.isnan(hr_1hz)
    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid == 0:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            "no valid HR samples after resampling to 1 Hz",
        )

    hr_valid = hr_1hz[valid_mask]
    # Zone index per valid second: count boundaries strictly <= HR (np.digitize with
    # right=False puts HR == boundary into the upper zone). Deterministic, pure.
    zone_idx = np.digitize(hr_valid, bounds, right=False)
    seconds_in_zone = np.bincount(zone_idx, minlength=wts.size).astype(np.float64)
    minutes_in_zone = seconds_in_zone / 60.0
    load = float(np.dot(minutes_in_zone, wts))

    if not math.isfinite(load):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "zonal HR-load evaluated to a non-finite value (ANL-R32)",
        )

    return _build_zonal_result(
        load,
        n_grid=n_grid,
        n_valid=n_valid,
        seconds_in_zone=seconds_in_zone,
        bounds=bounds,
        wts=wts,
        sport=sport,
    )


def _build_zonal_result(
    load: float,
    *,
    n_grid: int,
    n_valid: int,
    seconds_in_zone: np.ndarray,
    bounds: np.ndarray,
    wts: np.ndarray,
    sport: str | None,
) -> Computed[float]:
    """Assemble the zonal HR-load ``Computed`` envelope (TRIMP-R2).

    Records the per-zone dwell seconds, boundaries and weights under the distinct
    ``hr_load_zonal`` label (never relabelled as canonical ``hr_load`` — TRIMP-R2/R4).
    """
    coverage = n_valid / n_grid if n_grid > 0 else 0.0
    quality = QualityReport(
        coverage_fraction=coverage,
        sample_rate_hz=1.0,
        gap_count=n_grid - n_valid,
        confidence=1.0,
        extra={
            "load_model": LOAD_MODEL_HR_LOAD_ZONAL,
            "n_valid_seconds": n_valid,
            "seconds_in_zone": tuple(float(x) for x in seconds_in_zone),
            "zone_boundaries": tuple(float(x) for x in bounds),
            "zone_weights": tuple(float(x) for x in wts),
        },
    )
    provenance = InputLineage(
        sport=sport,
        channels=("heart_rate",),
        reference_params={
            "zone_boundaries": tuple(float(x) for x in bounds),
            "weights": tuple(float(x) for x in wts),
            "load_model": LOAD_MODEL_HR_LOAD_ZONAL,
        },
    )
    return Computed(value=load, quality=quality, provenance=provenance)


__all__ = [
    "APPLICABLE_SPORTS",
    "LOAD_MODEL_HR_LOAD",
    "LOAD_MODEL_HR_LOAD_ZONAL",
    "TRIMP_A_SEX_NEUTRAL",
    "TRIMP_B_SEX_NEUTRAL",
    "banister_hr_load",
    "hr_load_zonal",
]
