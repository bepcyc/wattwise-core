"""The typed workout step schema + signature-relative target resolution (GBO-R29).

A ``workout.steps`` array is canonical prescription data; every step MUST validate
against this schema BEFORE it is stored (GBO-R29/R29a "MUST validate against the step
schema") — the ORM model invokes :func:`validate_workout_steps` on assignment so no
write path can land an untyped step. Targets are expressible RELATIVE to the
athlete's fitness signature (``power_pct_cp`` / ``hr_pct_threshold``):
:func:`resolve_step_target` re-resolves a relative step against the CURRENT signature
values, so the same workout adapts when thresholds change — and returns ``None``
(a typed gap, never a fabricated number) when the signature lacks the needed base.
"""

from __future__ import annotations

from typing import Any

from wattwise_core.domain.enums import WorkoutStepIntent, WorkoutTargetType


class WorkoutStepError(ValueError):
    """A step violating the GBO-R29 schema (the write is refused, fail-closed)."""


_NUMERIC_FIELDS = ("target_low", "target_high", "duration_s", "distance_m")
_ALLOWED_KEYS = frozenset(
    {"target_type", "intent", "target_low", "target_high", "duration_s", "distance_m"}
)


def validate_workout_steps(steps: object) -> list[dict[str, Any]]:
    """Validate an ordered step array against the GBO-R29 step schema.

    Each step MUST carry a ``target_type`` and ``intent`` enum member, numeric (or
    absent) ``target_low``/``target_high`` with ``low <= high``, and EXACTLY ONE of
    ``duration_s`` / ``distance_m`` (strictly positive). Unknown keys are rejected —
    the step schema is closed, like the canonical payload contract (MAP-R2 spirit).
    Returns the validated list; raises :class:`WorkoutStepError` otherwise.
    """
    if not isinstance(steps, list):
        raise WorkoutStepError(f"steps must be a list, got {type(steps).__name__}")
    for index, step in enumerate(steps):
        _validate_step(step, index)
    return steps


def _validate_step(step: object, index: int) -> None:
    if not isinstance(step, dict):
        raise WorkoutStepError(f"step {index}: must be an object")
    unknown = set(step) - _ALLOWED_KEYS
    if unknown:
        raise WorkoutStepError(f"step {index}: unknown keys {sorted(unknown)}")
    try:
        WorkoutTargetType(str(step.get("target_type")))
        WorkoutStepIntent(str(step.get("intent")))
    except ValueError as exc:
        raise WorkoutStepError(f"step {index}: {exc}") from exc
    for fname in _NUMERIC_FIELDS:
        value = step.get(fname)
        if value is not None and not isinstance(value, int | float):
            raise WorkoutStepError(f"step {index}: {fname} must be numeric or absent")
    low, high = step.get("target_low"), step.get("target_high")
    if low is not None and high is not None and low > high:
        raise WorkoutStepError(f"step {index}: target_low > target_high")
    duration, distance = step.get("duration_s"), step.get("distance_m")
    if (duration is None) == (distance is None):
        raise WorkoutStepError(
            f"step {index}: exactly one of duration_s / distance_m is required (GBO-R29)"
        )
    extent = duration if duration is not None else distance
    if extent is not None and extent <= 0:
        raise WorkoutStepError(f"step {index}: the step extent must be positive")


def resolve_step_target(
    step: dict[str, Any], *, cp_w: float | None = None, threshold_hr_bpm: float | None = None
) -> tuple[float | None, float | None] | None:
    """Resolve one step's (low, high) target against the CURRENT signature (GBO-R29).

    Absolute target types pass through unchanged. ``power_pct_cp`` resolves against
    ``cp_w`` and ``hr_pct_threshold`` against ``threshold_hr_bpm`` (percent values,
    e.g. ``95`` -> 0.95 x base). When the signature lacks the required base the step
    CANNOT be resolved: returns ``None`` — a typed gap the caller surfaces, never a
    fabricated absolute target.
    """
    target_type = WorkoutTargetType(str(step.get("target_type")))
    low, high = step.get("target_low"), step.get("target_high")
    if target_type == WorkoutTargetType.POWER_PCT_CP:
        return _scale(low, high, cp_w)
    if target_type == WorkoutTargetType.HR_PCT_THRESHOLD:
        return _scale(low, high, threshold_hr_bpm)
    return (low, high)


def _scale(
    low: float | None, high: float | None, base: float | None
) -> tuple[float | None, float | None] | None:
    if base is None:
        return None  # signature lacks the base: typed gap, never a fabricated target
    pct = 1.0 / 100.0
    return (
        None if low is None else low * pct * base,
        None if high is None else high * pct * base,
    )


__all__ = ["WorkoutStepError", "resolve_step_target", "validate_workout_steps"]
