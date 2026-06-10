"""Endurance score — a declared, bounded composition of upstream metrics (doc 40 §7C).

``endurance-score`` is a single bounded scalar in ``[0, 100]`` summarizing current
aerobic endurance capacity (ES-R1). It is COMPOSED, not invented (ES-R2): a pure
function of three upstream :data:`MetricResult`\\ s — chronic training load ``CTL``
(PMC-R1), the long-duration power durability ratio ``MMP(long)/MMP(short)`` (Section
7A), and aerobic decoupling (Section 9; lower drift → higher score). It introduces NO
new physiological model, reads NO raw streams, and embeds NO hidden constants — every
weight and normalization knob lives in external configuration (``defaults.toml``
``[analytics] endurance_score_*``, CFG-R1a), documented there and re-cited here:

* ``f_ctl        = clamp(ctl / ES_CTL_FULL_SCALE, 0, 1)`` — monotone non-decreasing,
* ``f_durability = clamp((ratio - floor) / (ceiling - floor), 0, 1)`` — non-decreasing,
* ``f_decoupling = clamp(1 - drift_pct / ES_DECOUPLING_FULL_PENALTY_PCT, 0, 1)`` —
  monotone non-INcreasing (higher drift ⇒ not-higher score, ES-R3),
* ``score = 100 · Σ wᵢ·fᵢ / Σ wᵢ`` over the PRESENT components.

Missing-component policy (ES-R2): ``CTL`` is the non-substitutable component — when it
is ``Unavailable`` the score is ``Unavailable(MISSING_REQUIRED_INPUT)``. When a power
component (durability / decoupling) is ``Unavailable`` — including the
``NOT_APPLICABLE_FOR_SPORT`` case for a non-power sport (§7C sport-applicability) —
the configured policy decides: ``ES_ALLOW_PARTIAL`` declares any subset containing CTL
valid, so the score composes on the available components with the weights renormalized,
reduced ``confidence`` and the present/missing components recorded in ``QualityReport``
(ES-R2b); otherwise it fails closed. A missing component is NEVER silently scored as
``0`` (ANL-R4) — it is excluded from the weight renormalization instead.

The clamp to ``[0, 100]`` is part of the named normalization (ES-R3) — a declared
bound, not a fabricated value. Pure module (ANL-R2/R30): no DB, no I/O, no wall-clock.
"""

from __future__ import annotations

import math

from wattwise_core.analytics.constants import (
    ES_ALLOW_PARTIAL,
    ES_CTL_FULL_SCALE,
    ES_DECOUPLING_FULL_PENALTY_PCT,
    ES_DURABILITY_CEILING,
    ES_DURABILITY_FLOOR,
    ES_PARTIAL_CONFIDENCE_PENALTY,
    ES_WEIGHT_CTL,
    ES_WEIGHT_DECOUPLING,
    ES_WEIGHT_DURABILITY,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
    is_computed,
)

# ES is the cycling-power-REALIZED composition (§7C sport-applicability): the score
# itself is declared for every sport (ANL-R11 — the composition is sport-agnostic),
# while its power-family COMPONENTS gate on sport upstream; an inapplicable component
# flows through the ES-R2 missing-component policy, never a cross-sport surrogate.
APPLICABLE_SPORTS: None = None  # None == sport-agnostic composition (ANL-R11)


def _clamp01(x: float) -> float:
    """Clamp to ``[0, 1]`` — part of the named component normalization (ES-R3)."""
    return min(1.0, max(0.0, x))


def _component_scores(
    ctl: float, durability_ratio: float | None, decoupling_pct: float | None
) -> dict[str, float]:
    """The documented per-component ``[0,1]`` normalizations (ES-R1, config-driven)."""
    scores = {"ctl": _clamp01(ctl / ES_CTL_FULL_SCALE)}
    if durability_ratio is not None:
        band = ES_DURABILITY_CEILING - ES_DURABILITY_FLOOR
        scores["durability"] = _clamp01((durability_ratio - ES_DURABILITY_FLOOR) / band)
    if decoupling_pct is not None:
        scores["decoupling"] = _clamp01(1.0 - decoupling_pct / ES_DECOUPLING_FULL_PENALTY_PCT)
    return scores


_WEIGHTS: dict[str, float] = {
    "ctl": ES_WEIGHT_CTL,
    "durability": ES_WEIGHT_DURABILITY,
    "decoupling": ES_WEIGHT_DECOUPLING,
}


def _finite_or_none(result: MetricResult[float]) -> float | None:
    """A ``Computed`` finite value, else ``None`` (non-finite is treated as absent)."""
    if not is_computed(result):
        return None
    v = float(result.value)
    return v if math.isfinite(v) else None


def endurance_score(
    ctl: MetricResult[float],
    durability_ratio: MetricResult[float],
    decoupling_pct: MetricResult[float],
    *,
    sport: str | None = None,
) -> MetricResult[float]:
    """Compose the bounded ``[0,100]`` endurance score from upstream results (ES-R1/R2/R3).

    Pure function of the three upstream :data:`MetricResult`\\ s — never reads raw
    streams (ES-T1). ``CTL`` missing ⇒ ``Unavailable(MISSING_REQUIRED_INPUT)``; a
    missing power component composes on the declared-valid CTL-containing subset with
    reduced confidence iff ``ES_ALLOW_PARTIAL`` (ES-R2b), else fails closed. Present
    and missing components are always recorded in ``QualityReport.extra``; a missing
    component is never scored as ``0`` (ANL-R4).
    """
    ctl_value = _finite_or_none(ctl)
    if ctl_value is None:
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "endurance-score requires CTL (non-substitutable component, ES-R2)",
        )
    scores = _component_scores(
        ctl_value,
        _finite_or_none(durability_ratio),
        _finite_or_none(decoupling_pct),
    )
    missing = tuple(sorted(set(_WEIGHTS) - set(scores)))
    if missing and not ES_ALLOW_PARTIAL:
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            f"endurance-score components unavailable: {', '.join(missing)} "
            "(partial composition not declared valid, ES-R2)",
        )
    weight_sum = sum(_WEIGHTS[name] for name in scores)
    if weight_sum <= 0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "endurance-score configured weights over present components sum to <= 0",
        )
    raw = 100.0 * sum(_WEIGHTS[name] * scores[name] for name in scores) / weight_sum
    value = min(100.0, max(0.0, raw))  # ES-R3: the bound is part of the normalization
    confidence = ES_PARTIAL_CONFIDENCE_PENALTY if missing else 1.0
    quality = QualityReport(
        confidence=confidence,
        extra={
            "components_present": tuple(sorted(scores)),
            "components_missing": missing,
        },
    )
    lineage = InputLineage(sport=sport, channels=tuple(sorted(scores)))
    return Computed(value=value, quality=quality, provenance=lineage)


def durability_ratio(
    long_mmp_w: MetricResult[float], short_mmp_w: MetricResult[float]
) -> MetricResult[float]:
    """``MMP(long)/MMP(short)`` from two upstream curve points (ES-R1 durability input).

    Composed of upstream results only; either point ``Unavailable`` propagates as
    ``Unavailable(MISSING_REQUIRED_INPUT)`` and a non-positive short-duration power is
    ``OUT_OF_DOMAIN`` (a ratio over it is undefined) — never a fabricated ratio.
    """
    if not is_computed(long_mmp_w) or not is_computed(short_mmp_w):
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "durability ratio requires both MMP(long) and MMP(short) curve points",
        )
    long_w, short_w = float(long_mmp_w.value), float(short_mmp_w.value)
    if not (math.isfinite(long_w) and math.isfinite(short_w)) or short_w <= 0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "durability ratio needs finite MMP values and MMP(short) > 0",
        )
    return Computed(value=long_w / short_w)


__all__ = [
    "APPLICABLE_SPORTS",
    "durability_ratio",
    "endurance_score",
]
