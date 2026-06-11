"""Analytics-engine settings mixin (doc 40 constants).

Split out of settings.py to keep both the Settings class and the settings module
under the QUAL-R9 size ceilings. Values live in defaults.toml (CFG-R1a); this module
declares only typed schema + range constraints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class _AnalyticsSettings(BaseModel):
    """Analytics-engine constants (doc 40), mixed into :class:`Settings`.

    Declared as a separate mixin to keep the Settings class body under the QUAL-R9
    size ceiling (the same pattern as :class:`_GroundingSettings`). Values live in
    defaults.toml (CFG-R1a); this declares only typed schema + range constraints.
    """

    analytics__ctl_time_constant_days: float = Field(gt=0)
    analytics__atl_time_constant_days: float = Field(gt=0)
    # DEGR-R2 substitution confidence multiplier (in (0,1]); the VALUE lives in defaults.toml
    # (CFG-R1a), this declares only the typed schema + range constraint.
    analytics__training_load_confidence_penalty: float = Field(gt=0, le=1)
    # GBO-R28 fail-closed floor for a stored MODELED signature's fit quality (R^2);
    # the VALUE lives in defaults.toml (CFG-R1a) — this is schema + range only.
    analytics__signature_min_fit_r2: float = Field(ge=0, le=1)
    # Endurance-score (ES-R1) declared weighting + normalization; the VALUES live in
    # defaults.toml (CFG-R1a) — this declares only the typed schema + range constraints.
    # Weights are relative, non-negative; the composition renormalizes over the present
    # components, so only the SUM must be positive (validated at composition time).
    analytics__endurance_score_weight_ctl: float = Field(ge=0)
    analytics__endurance_score_weight_curve_shape: float = Field(ge=0)
    analytics__endurance_score_weight_decoupling: float = Field(ge=0)
    analytics__endurance_score_ctl_full_scale: float = Field(gt=0)
    analytics__endurance_score_curve_shape_floor: float = Field(ge=0)
    analytics__endurance_score_curve_shape_ceiling: float = Field(gt=0)
    analytics__endurance_score_decoupling_full_penalty_pct: float = Field(gt=0)
    analytics__endurance_score_allow_partial: bool
    analytics__endurance_score_partial_confidence_penalty: float = Field(gt=0, le=1)
    analytics__endurance_score_window_days: int = Field(ge=1)
    analytics__endurance_score_long_duration_s: int = Field(ge=1)
    analytics__endurance_score_short_duration_s: int = Field(ge=1)
    # Durability / fatigue resistance (DUR-R1..R8, issue #26); the VALUES live in
    # defaults.toml (CFG-R1a) — this declares only the typed schema + range constraints.
    analytics__durability_target_duration_s: int = Field(ge=1)
    analytics__durability_wprime_multiple: float = Field(gt=0)
    # CP-R3/R4 pre-fit power-degeneracy epsilon (relative MMP power spread below which
    # the fit refuses with INSUFFICIENT_DATA before any regression); the VALUE lives in
    # defaults.toml (CFG-R1a) — this declares only the typed schema + range constraint.
    analytics__cp_power_spread_epsilon: float = Field(gt=0, lt=1)
    # Session-RPE load mapping knobs (SRPE-R1): the reported-exertion full scale (CR-10)
    # and the load one hour at full scale reads as (the TSS-commensurate anchor); the
    # VALUES live in defaults.toml (CFG-R1a) — this declares only schema + range.
    analytics__srpe_rpe_full_scale: float = Field(gt=0)
    analytics__srpe_load_per_hour_at_full_scale: float = Field(gt=0)
