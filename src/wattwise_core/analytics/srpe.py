"""Session-RPE training load (Foster) — the last-resort ``training_load`` member (SRPE-R1).

Implements the ONE canonical subjective load variant: Foster's session-RPE method,
scaled into the ``training_load`` equivalence class's TSS-commensurate currency via the
industry RPE-as-intensity mapping (the convention intervals.icu uses)::

    srpe_load = (RPE / rpe_full_scale)^2 * (duration_s / 3600) * load_per_hour_at_full_scale

With the packaged defaults (CR-10 full scale 10, per-hour anchor 100) the mapping has
the same shape as ``TSS = IF^2 * hours * 100`` with ``RPE/10`` standing in for IF, so
an hour at the athlete's reported maximum reads as 100. Foster's original arbitrary
units (``RPE x duration_min``) are recorded in the :class:`QualityReport` so the raw
published quantity stays auditable. Both scale knobs are externalized config
(``defaults.toml`` ``[analytics]``, CFG-R1a), never code literals.

This load family is sport-agnostic (ANL-R11): any session carrying an athlete-reported
exertion and a duration qualifies — including the power-less, HR-less sessions
(strength work, most swims) no other class member can price — so it NEVER returns
``NOT_APPLICABLE_FOR_SPORT``.

Fail-closed contract (doc 40 §6, ANL-R4):

* Absent ``perceived_exertion`` / ``duration_s``  -> ``MISSING_REQUIRED_INPUT``.
* Non-finite or out-of-scale RPE (outside ``[0, full_scale]``), or a non-finite or
  non-positive duration                            -> ``OUT_OF_DOMAIN``. The report is
  never clamped into validity (a clamped exertion is a fabricated one).
* A non-finite load would result                   -> ``OUT_OF_DOMAIN`` (ANL-R32).

A reported RPE of 0 over a valid duration is an honest ``Computed(0.0)`` — the athlete
said "rest-easy", which is a real report, not an absence.

Every function here is PURE (no I/O, no wall-clock, no RNG; ANL-R2/R30) returning a
typed :data:`MetricResult` envelope (ANL-R3), never a bare number. The produced
:class:`Computed` is always labelled ``load_model='srpe_load'`` (LOAD-R2 closed set)
and never relabelled as ``power_tss`` or ``hr_load``.

Requirement IDs: SRPE-R1; LOAD-R2/R3; DM-SUB-R1; ANL-R2/R3/R4/R5/R11/R30/R32.
"""

from __future__ import annotations

import math

from wattwise_core.analytics.constants import (
    SRPE_LOAD_PER_HOUR_AT_FULL_SCALE,
    SRPE_RPE_FULL_SCALE,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)

# The mandatory, never-relabelled provenance tag (LOAD-R2 closed set): this member is
# surfaced ONLY under its own label, exactly like ``hr_load`` vs ``power_tss``.
LOAD_MODEL_SRPE = "srpe_load"

# ANL-R11: sport-agnostic — ``None`` (not an enumerated set) is the declared
# sport-agnostic marker, so there is NO sport gate (absence of the report is
# MISSING_REQUIRED_INPUT, never NOT_APPLICABLE_FOR_SPORT).
APPLICABLE_SPORTS: None = None

_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_MINUTE = 60.0


def _validate_srpe_inputs(
    perceived_exertion: float | None, duration_s: float | None
) -> tuple[float, float] | Unavailable:
    """Presence + domain gates for the session-RPE load (SRPE-R1).

    Absent inputs are ``MISSING_REQUIRED_INPUT``; a non-finite or out-of-scale RPE, or a
    non-finite or non-positive duration, is ``OUT_OF_DOMAIN`` — present-but-invalid is
    never silently repaired (ANL-R4). Returns the narrowed ``(rpe, duration_s)`` floats.
    """
    if perceived_exertion is None or duration_s is None:
        missing = [
            name
            for name, val in (
                ("perceived_exertion", perceived_exertion),
                ("duration_s", duration_s),
            )
            if val is None
        ]
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            f"required session-RPE input(s) absent: {', '.join(missing)}",
        )
    rpe = float(perceived_exertion)
    duration = float(duration_s)
    if not math.isfinite(rpe) or rpe < 0.0 or rpe > SRPE_RPE_FULL_SCALE:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            f"perceived_exertion ({rpe}) must be finite and within "
            f"[0, {SRPE_RPE_FULL_SCALE}] (CR-10 scale); reports are never clamped",
        )
    if not math.isfinite(duration) or duration <= 0.0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            f"duration_s ({duration}) must be finite and strictly positive",
        )
    return rpe, duration


def srpe_load(
    perceived_exertion: float | None,
    duration_s: float | None,
    *,
    sport: str | None = None,
) -> MetricResult[float]:
    """Canonical session-RPE load (SRPE-R1), labelled ``load_model='srpe_load'``.

    Maps the athlete-reported CR-10 exertion and the session duration into the
    ``training_load`` class's TSS-commensurate currency::

        (RPE / rpe_full_scale)^2 * (duration_s / 3600) * load_per_hour_at_full_scale

    Foster's raw arbitrary units (``RPE x duration_min``) ride along in the
    :class:`QualityReport` (``foster_au``). Confidence is 1.0 at the metric level —
    the class-level substitution machinery (DEGR-R2) owns the fidelity downgrade when
    this member stands in for power-TSS, exactly as it does for the HR member.
    """
    validated = _validate_srpe_inputs(perceived_exertion, duration_s)
    if isinstance(validated, Unavailable):
        return validated
    rpe, duration = validated

    intensity = rpe / SRPE_RPE_FULL_SCALE
    load = intensity * intensity * (duration / _SECONDS_PER_HOUR) * SRPE_LOAD_PER_HOUR_AT_FULL_SCALE
    if not math.isfinite(load):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "session-RPE load evaluated to a non-finite value (ANL-R32)",
        )

    quality = QualityReport(
        coverage_fraction=1.0,
        sample_rate_hz=None,
        gap_count=0,
        confidence=1.0,
        extra={
            "load_model": LOAD_MODEL_SRPE,
            "foster_au": rpe * (duration / _SECONDS_PER_MINUTE),
            "rpe": rpe,
            "duration_s": duration,
        },
    )
    provenance = InputLineage(
        sport=sport,
        channels=("perceived_exertion",),
        reference_params={
            "rpe_full_scale": SRPE_RPE_FULL_SCALE,
            "load_per_hour_at_full_scale": SRPE_LOAD_PER_HOUR_AT_FULL_SCALE,
            "load_model": LOAD_MODEL_SRPE,
        },
    )
    return Computed(value=load, quality=quality, provenance=provenance)


__all__ = [
    "APPLICABLE_SPORTS",
    "LOAD_MODEL_SRPE",
    "srpe_load",
]
