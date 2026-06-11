"""Analytics-knob settings fields, split off the Settings class (QUAL-R9 decomposition).

A plain field-carrier mixin (NOT a settings source itself), mirroring
:class:`~wattwise_core.config.settings._GroundingSettings`: :class:`Settings` inherits these
so the env mapping, dotted ``section__key`` names, and attribute access are identical — the
fields simply live in this focused module to keep both the Settings class body (200-line
ceiling) and the settings module (400-line ceiling) under the QUAL-R9 size limits.

Schema-only (CFG-R1a): each field declares its type + validation constraints only; the
concrete VALUES live in ``defaults.toml`` (overridable by the operator file / env). A field
absent from every layer fails the boot closed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class _AnalyticsSettings(BaseModel):
    """Analytics computation knobs carried on a mixin to keep Settings size-capped."""

    # Durability / fatigue resistance (DUR-R1..R8, issue #26): the target long-duration
    # window and the W' multiple defining the fatigued state the resistance is measured at.
    analytics__durability_target_duration_s: int = Field(ge=1)
    analytics__durability_wprime_multiple: float = Field(gt=0)
    # Session-RPE load mapping (SRPE-R1): the reported-exertion full scale (CR-10) and the
    # load one hour at full scale reads as (the TSS-commensurate anchor).
    analytics__srpe_rpe_full_scale: float = Field(gt=0)
    analytics__srpe_load_per_hour_at_full_scale: float = Field(gt=0)
