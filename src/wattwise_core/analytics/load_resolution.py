"""HR-path load-model resolution honouring the athlete default (doc 40 LOAD-R4).

The per-athlete ``default_training_load_model`` preference overrides the engine's
automatic HR-path choice (LOAD-R4): on the HR path BOTH ``hr_load`` (Banister-HRR,
TRIMP-R1) and ``hr_load_zonal`` (zone-weighted, TRIMP-R2) are applicable only when their
inputs exist. The canonical default is ``hr_load``. ``hr_load_zonal`` is produced ONLY
when the athlete prefers it AND declared HR-zone boundaries + per-zone weights are
resolvable; otherwise the engine falls back to the automatic ``hr_load`` and RECORDS the
substitution in ``QualityReport`` (LOAD-R4) — it never fabricates the preferred model's
inputs (ANL-R4) and never forces ``power_tss`` onto a power-less activity.

These are PURE functions (ANL-R2/R30): no I/O, no wall-clock, no session — the canonical
inputs (HR_max/HR_rest, sex, the stored preference) are resolved by the caller (the
service, which owns the store access) and passed in. Keeping LOAD-R4 here keeps the
single canonical service surface thin and within the module size ceiling.
"""

from __future__ import annotations

from wattwise_core.analytics.result import Computed, MetricResult, QualityReport, is_computed
from wattwise_core.analytics.series import Stream
from wattwise_core.analytics.trimp import (
    LOAD_MODEL_HR_LOAD,
    LOAD_MODEL_HR_LOAD_ZONAL,
    banister_hr_load,
)

__all__ = ["resolve_hr_load"]


def _with_substitution(result: Computed[float], *, requested: str) -> Computed[float]:
    """Record a LOAD-R4 model substitution in ``QualityReport`` (auditable fallback).

    When the athlete's preferred load model is not applicable for an activity, the engine
    falls back to the automatic LOAD-R3 selection and MUST record the substitution so the
    chosen and effective models are auditable (LOAD-R4). The model that ACTUALLY produced
    the load is already in ``result.quality.extra["load_model"]``; this only adds the
    requested-but-inapplicable model + a flag, never altering the numeric value.
    """
    quality = QualityReport(
        coverage_fraction=result.quality.coverage_fraction,
        sample_rate_hz=result.quality.sample_rate_hz,
        gap_count=result.quality.gap_count,
        confidence=result.quality.confidence,
        extra={
            **result.quality.extra,
            "requested_load_model": requested,
            "load_model_substituted": True,
        },
    )
    return Computed(value=result.value, quality=quality, provenance=result.provenance)


def resolve_hr_load(
    hr: Stream | None,
    hr_max: float | None,
    hr_rest: float | None,
    sex: str | None,
    *,
    preferred_load_model: str | None,
) -> MetricResult[float] | None:
    """Resolve the HR-path load for one activity honouring the athlete default (LOAD-R4).

    Returns the labeled HR-load :class:`MetricResult` (``hr_load`` or ``hr_load_zonal``),
    or ``None`` when there is no HR channel at all (so the caller surfaces an absent HR
    load, LM-R2). When the preferred model is ``hr_load_zonal`` but that variant is not
    applicable (no declared HR-zone boundaries + per-zone weights — the canonical store
    carries zone boundaries but no per-zone load WEIGHTS, TRIMP-R2), the engine falls back
    to the automatic Banister ``hr_load`` and records the substitution (LOAD-R4).
    """
    if hr is None:
        return None
    zonal = _zonal_hr_load() if preferred_load_model == LOAD_MODEL_HR_LOAD_ZONAL else None
    if zonal is not None and is_computed(zonal):
        return zonal
    banister = banister_hr_load(hr, hr_max, hr_rest, sex)
    if preferred_load_model == LOAD_MODEL_HR_LOAD_ZONAL and is_computed(banister):
        # The preferred zonal model was not applicable: fall back + record (LOAD-R4).
        return _with_substitution(banister, requested=LOAD_MODEL_HR_LOAD_ZONAL)
    return banister


def _zonal_hr_load() -> MetricResult[float] | None:
    """The zone-weighted HR load when declared HR-zone boundaries + weights exist.

    ``hr_load_zonal`` requires declared HR-zone boundaries AND per-zone weights
    (TRIMP-R2). The canonical ``training_zone_set`` carries zone boundaries but no
    per-zone load WEIGHTS, so without a declared weight source the zonal model is not
    applicable for the activity and this returns ``None`` (the caller then falls back to
    ``hr_load`` per LOAD-R4). It never fabricates weights (ANL-R4).
    """
    return None


# Re-export for callers that label the bundle without re-deriving the constant.
HR_LOAD = LOAD_MODEL_HR_LOAD
