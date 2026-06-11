"""Durability / fatigue resistance — the work-conditioned power decrement (doc 40 §10).

Every other power metric in the engine (MMP, CP/W', NP/IF/TSS) is a **fresh-state**
number: it describes what an athlete can do rested, or averaged over a whole effort.
But fresh fitness systematically *over-predicts* performance in prolonged events, and
the size of that over-prediction — **durability**, a.k.a. fatigue resistance — varies
enormously between athletes of identical fresh fitness. Durability is now treated as an
independent performance parameter: the rider who retains the highest fraction of fresh
power late in a race typically wins it.

This module computes durability the way the field literature defines it: the **decrement
in maximal sustainable power for a target duration, fresh vs. fatigued**, where
"fatigued" is reached by accumulating work — and, crucially, *intensity-weighted* work,
because the intensity of prior work (not the raw kilojoule count) drives the downward
shift of the power-duration relationship (Spragg et al. 2024, *Eur J Sport Sci*; review
in *Eur J Appl Physiol* 2025). The fatigue axis here is therefore **accumulated work
above Critical Power** (the W'-expenditure signal), not total kJ.

Two public functions, both pure (ANL-R2/R30) and fail-closed (ANL-R4):

* :func:`accumulated_work_above_cp_j` — the per-second cumulative intensity-weighted
  work axis ``Σ max(0, P(t) - CP)`` in joules (Δt = 1 s). This is the missing primitive
  the rest of the module is built on; its final value is the activity's total
  work-above-CP (the long-dormant ``DerivedActivityMetric.work_above_cp_j`` finally has
  a definition).
* :func:`durability_decrement` — splits the ride at the second where accumulated
  work-above-CP first reaches a (per-athlete) ``fatigue_threshold_j``, takes the best
  ``target_duration_s``-second mean power in the *fresh* segment before the split and in
  the *fatigued* segment after it, and returns the decrement
  ``100 · (1 - fatigued/fresh)`` (positive ⇒ power dropped under fatigue, the expected
  sign; negative ⇒ the athlete went *harder* late, reported raw, never clamped — the
  decoupling-sign convention).

**Sufficiency is the default path, not the edge case (DUR-R5).** A durability number
needs a genuinely fatigued state *and* hard-enough efforts in both segments to mean
anything — most rides have neither. When the threshold is never reached, or either
segment lacks a fully-seeded ``target_duration_s`` window, the result fails closed to
``Unavailable(INSUFFICIENT_DATA)`` rather than fabricating a confident "100 % retained".
A non-blocking ``fresh_effort_below_cp`` quality flag (DUR-R6) is raised when the fresh
"best" effort did not even reach CP, i.e. it was probably not maximal and the ratio
should be read with care — surfaced honestly rather than silently trusted.

The fatigue **threshold** is deliberately *not* a global constant (a fixed kJ figure
means different things for different athletes and breaks the cross-athlete comparison
durability relies on). :func:`fatigue_threshold_from_wprime` anchors it to the athlete's
own anaerobic capacity — a configurable multiple of ``W'`` of work-above-CP — so the
pure metric receives an already-resolved, per-athlete joule threshold. A richer
per-athlete policy (a percentile of the athlete's own accumulated-work distribution)
belongs to the service layer and is out of this pure module's scope.

This is a cycling-power-specific metric (doc 40 §5): it requires a true mechanical power
channel and Critical Power. The sport-agnostic internal:external variant (HR:pace
decline vs. accumulated work) for running and swimming reuses the decoupling /
efficiency machinery and is a separate, declared step.

Engine-wide contract honoured: pure functions, no I/O / wall-clock / RNG / global state
(ANL-R2/R30); typed :data:`MetricResult` envelope, never a bare number (ANL-R3); fails
closed with the exact reason (ANL-R4, doc 40 §6); no NaN/Inf inside a ``Computed``
(ANL-R32 ⇒ ``OUT_OF_DOMAIN``); every ``Computed`` carries a :class:`QualityReport` +
:class:`InputLineage` (ANL-R5/R33); cycling-power applicability gates on the canonical
``sport`` value, never a source name (ANL-R11/R12).

Requirements implemented (proposed family, issue #26): DUR-R1 (intensity-weighted
fatigue axis), DUR-R2 (fresh/fatigued split at the threshold crossing), DUR-R3 (best
target-duration mean power per segment), DUR-R4 (decrement sign convention, reported
raw), DUR-R5 (sufficiency / fail-closed default), DUR-R6 (non-maximal-fresh quality
flag), DUR-R7 (per-athlete threshold from W'), DUR-R8 (cycling-power applicability).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wattwise_core.analytics.constants import (
    DURABILITY_TARGET_DURATION_S,
    DURABILITY_WPRIME_MULTIPLE,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)
from wattwise_core.analytics.series import FloatArray, trailing_rolling_mean

# Cycling-power family: a true mechanical power channel + CP are required (doc 40 §5).
APPLICABLE_SPORTS: tuple[str, ...] = ("cycling",)

POWER_CHANNEL = "power"


@dataclass(frozen=True, slots=True)
class DurabilityDecrement:
    """The value carried by a :class:`Computed` durability-decrement result.

    ``fresh_best_power_w`` / ``fatigued_best_power_w`` are the best
    ``target_duration_s``-second mean powers (W) in the fresh segment (before the
    fatigue-threshold crossing) and the fatigued segment (at/after it).
    ``retained_fraction = fatigued / fresh`` and ``decrement_pct = 100·(1 - retained)``
    (positive ⇒ power dropped under fatigue; reported raw, DUR-R4). ``split_elapsed_s``
    is the second at which accumulated work-above-CP first reached
    ``fatigue_threshold_j``; ``work_above_cp_total_j`` is the whole-ride total.
    """

    target_duration_s: int
    fresh_best_power_w: float
    fatigued_best_power_w: float
    retained_fraction: float
    decrement_pct: float
    fatigue_threshold_j: float
    split_elapsed_s: int
    work_above_cp_total_j: float


def accumulated_work_above_cp_j(power_1hz: FloatArray, cp_w: float) -> FloatArray:
    """Per-second cumulative intensity-weighted work above CP, in joules (DUR-R1).

    The fatigue axis: ``cumsum_t Σ max(0, P(t) - CP)`` over a uniform 1 Hz power series
    (``Δt = 1 s`` ⇒ watts·second = joules). A gap (``NaN``) contributes **zero** work
    for that second and carries the running total forward unchanged (never a fabricated
    expenditure, never a reset). Pure and deterministic; the array is the same length as
    the input and non-decreasing by construction.

    This is the W'-expenditure signal Spragg (2024) identifies as the driver of the
    power-duration downward shift — intensity-weighted, unlike a raw-kJ
    (``cumsum(P)``) axis which counts an easy hour the same as a hard one.
    """
    p = np.asarray(power_1hz, dtype=np.float64)
    above = np.where(np.isnan(p), 0.0, np.maximum(0.0, p - cp_w))
    return np.cumsum(above)


def fatigue_threshold_from_wprime(
    w_prime_j: float | None, *, multiple: float = DURABILITY_WPRIME_MULTIPLE
) -> MetricResult[float]:
    """Resolve a per-athlete fatigue threshold as a multiple of ``W'`` (DUR-R7).

    Anchoring the "fatigued" threshold to the athlete's own anaerobic capacity ``W'``
    (rather than a global kJ literal) keeps the metric comparable across athletes of
    different size and fitness. Returns ``multiple · W'`` joules of work-above-CP, or a
    typed :class:`Unavailable`: absent ``W'`` ⇒ ``MISSING_REQUIRED_INPUT``; a
    non-finite/non-positive ``W'`` or ``multiple`` ⇒ ``OUT_OF_DOMAIN``.
    """
    if w_prime_j is None:
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "W' is required to anchor a per-athlete fatigue threshold",
        )
    if not math.isfinite(w_prime_j) or w_prime_j <= 0.0:
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, "W' must be positive and finite")
    if not math.isfinite(multiple) or multiple <= 0.0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN, "W' multiple must be positive and finite"
        )
    return Computed(value=float(multiple) * float(w_prime_j))


def _best_mean_power(power_1hz: FloatArray, duration_s: int) -> float | None:
    """Best fully-seeded ``duration_s``-second mean power over a 1 Hz segment (DUR-R3).

    Reuses the seeded trailing rolling mean (NP-R2 semantics): a window counts only
    when all ``duration_s`` of its seconds are valid (non-gap), so a window straddling a
    gap never produces a short-window mean. Returns the best such mean, or ``None`` when
    no fully-seeded window exists in the segment (segment too short / too gappy).
    """
    if power_1hz.size < duration_s:
        return None
    rolling = trailing_rolling_mean(power_1hz, duration_s)
    if not np.any(~np.isnan(rolling)):
        return None
    return float(np.nanmax(rolling))


def _validate_inputs(
    power_1hz: FloatArray | None,
    cp_w: float,
    fatigue_threshold_j: float,
    sport: str,
) -> FloatArray | Unavailable:
    """Presence + domain + applicability gates, or a typed :class:`Unavailable`.

    Sport outside the cycling-power family ⇒ ``NOT_APPLICABLE_FOR_SPORT`` (ANL-R12);
    an absent/empty/all-gap power stream ⇒ ``MISSING_REQUIRED_INPUT``; a non-finite or
    non-positive CP or fatigue threshold ⇒ ``OUT_OF_DOMAIN``.
    """
    if sport not in APPLICABLE_SPORTS:
        return Unavailable(
            UnavailableReason.NOT_APPLICABLE_FOR_SPORT,
            f"durability is a cycling-power-family metric, not defined for sport {sport!r}",
        )
    if power_1hz is None or power_1hz.size == 0 or not np.any(~np.isnan(power_1hz)):
        return Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, "no valid power channel")
    if not math.isfinite(cp_w) or cp_w <= 0.0:
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, "CP must be positive and finite")
    if not math.isfinite(fatigue_threshold_j) or fatigue_threshold_j <= 0.0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN, "fatigue threshold must be positive and finite"
        )
    return np.asarray(power_1hz, dtype=np.float64)


def durability_decrement(
    power_1hz: FloatArray | None,
    cp_w: float,
    *,
    fatigue_threshold_j: float,
    target_duration_s: int = DURABILITY_TARGET_DURATION_S,
    sport: str = "cycling",
) -> MetricResult[DurabilityDecrement]:
    """Work-conditioned durability decrement for a target duration (DUR-R1..R8).

    Parameters
    ----------
    power_1hz:
        Uniform 1 Hz power series in watts (``Δt = 1 s``); ``NaN`` marks a gap. Must be
        already resampled to 1 Hz by the caller (ANL-R8), as for :func:`wbal`.
    cp_w:
        Critical Power in watts (canonical, time-effective; ANL-R9). Defines the
        intensity-weighted fatigue axis (work above CP) and the maximal-effort quality
        flag.
    fatigue_threshold_j:
        Joules of accumulated work-above-CP that mark the fresh→fatigued boundary
        (DUR-R2). A **per-athlete** value — derive it with
        :func:`fatigue_threshold_from_wprime`, never a global kJ literal.
    target_duration_s:
        The probe duration (default 5 min): durability is the decrement of the best
        ``target_duration_s``-second power. Must be positive.
    sport:
        Canonical sport (ANL-R11). Durability is cycling-power-specific; a non-cycling
        sport fails closed with ``NOT_APPLICABLE_FOR_SPORT`` before any computation.

    Returns
    -------
    MetricResult[DurabilityDecrement]
        ``Computed`` carrying the decrement + its provenance, or a typed
        :class:`Unavailable`. The sufficiency path is the default (DUR-R5): the
        threshold not being reached, or either segment lacking a fully-seeded
        ``target_duration_s`` effort, is ``INSUFFICIENT_DATA`` — never a fabricated
        full-retention number.
    """
    if target_duration_s <= 0:
        raise ValueError("target_duration_s must be positive")

    validated = _validate_inputs(power_1hz, cp_w, fatigue_threshold_j, sport)
    if isinstance(validated, Unavailable):
        return validated
    power = validated

    # DUR-R1: intensity-weighted fatigue axis (cumulative work above CP, joules).
    cum_work = accumulated_work_above_cp_j(power, cp_w)
    total_work = float(cum_work[-1]) if cum_work.size else 0.0

    # DUR-R2/R5: the fresh→fatigued split is the first second at which accumulated
    # work-above-CP reaches the threshold. cum_work is non-decreasing, so searchsorted
    # gives that crossing in O(log n). Never reached ⇒ the athlete was never in a
    # fatigued state ⇒ fail closed (the default, honest path).
    split = int(np.searchsorted(cum_work, fatigue_threshold_j, side="left"))
    if split >= power.size:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"accumulated work above CP ({total_work:.0f} J) never reached the fatigue "
            f"threshold ({fatigue_threshold_j:.0f} J); no fatigued state to measure",
        )

    # DUR-R3: best target-duration effort in each segment.
    fresh_best = _best_mean_power(power[:split], target_duration_s)
    fatigued_best = _best_mean_power(power[split:], target_duration_s)
    if fresh_best is None or fatigued_best is None:
        which = "fresh" if fresh_best is None else "fatigued"
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"{which} segment has no fully-seeded {target_duration_s}s effort "
            "(segment too short or too gappy for a durability comparison)",
        )
    if fresh_best <= 0.0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "fresh best power is non-positive; durability ratio undefined",
        )

    retained = fatigued_best / fresh_best
    decrement_pct = 100.0 * (1.0 - retained)
    if not (math.isfinite(retained) and math.isfinite(decrement_pct)):  # ANL-R32
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, "non-finite durability decrement")

    # DUR-R6: a fresh "best" effort below CP was almost certainly not maximal, so the
    # ratio is suspect — flag it (non-blocking), never silently trust it.
    fresh_below_cp = fresh_best < cp_w

    value = DurabilityDecrement(
        target_duration_s=target_duration_s,
        fresh_best_power_w=fresh_best,
        fatigued_best_power_w=fatigued_best,
        retained_fraction=retained,
        decrement_pct=decrement_pct,
        fatigue_threshold_j=float(fatigue_threshold_j),
        split_elapsed_s=split,
        work_above_cp_total_j=total_work,
    )
    quality = QualityReport(
        coverage_fraction=float(np.count_nonzero(~np.isnan(power))) / power.size,
        sample_rate_hz=1.0,
        gap_count=int(np.count_nonzero(np.isnan(power))),
        confidence=1.0,
        extra={
            "target_duration_s": target_duration_s,
            "fresh_window_s": int(split),
            "fatigued_window_s": int(power.size - split),
            "fresh_effort_below_cp": fresh_below_cp,
            "work_above_cp_total_j": total_work,
        },
    )
    provenance = InputLineage(
        sport=sport,
        channels=(POWER_CHANNEL,),
        reference_params={
            "cp_w": float(cp_w),
            "target_duration_s": target_duration_s,
            "fatigue_threshold_j": float(fatigue_threshold_j),
        },
    )
    return Computed(value=value, quality=quality, provenance=provenance)


__all__ = [
    "APPLICABLE_SPORTS",
    "DurabilityDecrement",
    "accumulated_work_above_cp_j",
    "durability_decrement",
    "fatigue_threshold_from_wprime",
]
