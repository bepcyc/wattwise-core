"""DM-SUB-R1..R5 metric-equivalence class registry + substitution surfacing tests.

Proves the registry is externalized configuration (the packaged worked example loads;
an operator file overrides it), that class members validate against the SINGLE ranked
GAP-R2 fidelity vocabulary (a resolution-outcome token is rejected), and that the
DM-SUB-R4 hook yields the ``substituted`` marker carrying the displaced top tier
ONLY when a declared class's winner ranks below its top member — never fabricated
for an undeclared channel or a top-tier winner. The signature-relative workout-step
resolution (GBO-R29) is covered here too: a missing signature base yields a typed
``None``, never a fabricated absolute target.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wattwise_core.domain import equivalence as eq
from wattwise_core.domain.enums import Fidelity
from wattwise_core.domain.workout_steps import (
    WorkoutStepError,
    resolve_step_target,
    validate_workout_steps,
)

pytestmark = pytest.mark.unit


def test_packaged_training_load_class_loads() -> None:
    """DM-SUB-R1: the shipped declaration carries the training_load worked example —
    ordered members, each with a ranked tier, a semantic note, and a penalty; the
    class's top tier is raw_stream (power-based TSS)."""
    cls = eq.class_for("training_load")
    assert cls is not None
    assert cls.top_tier == Fidelity.RAW_STREAM
    assert [m.tier for m in cls.members] == [
        Fidelity.RAW_STREAM, Fidelity.PLATFORM_COMPUTED, Fidelity.MODELED,
    ]
    assert all(m.note and m.penalty for m in cls.members)


def test_substitution_marker_only_when_below_class_top(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DM-SUB-R4: a winner BELOW its declared class top tier yields the substituted
    marker recording the displaced top tier; a top-tier winner, or a channel with no
    declared class, yields none — substitution is surfaced only when real."""
    classes = tmp_path / "classes.toml"
    classes.write_text(
        """
[[canonical.equivalence_class]]
channel = "avg_power_w"
[[canonical.equivalence_class.members]]
metric = "direct_power"
tier = "raw_stream"
note = "direct meter watts"
penalty = "none"
[[canonical.equivalence_class.members]]
metric = "platform_power"
tier = "platform_computed"
note = "vendor-estimated watts"
penalty = "moderate"
"""
    )
    monkeypatch.setenv("WATTWISE_EQUIVALENCE_CLASSES_FILE", str(classes))
    eq._load.cache_clear()
    try:
        marker = eq.substitution_for("avg_power_w", Fidelity.PLATFORM_COMPUTED)
        assert marker is not None
        assert marker.equivalence_class == "avg_power_w"
        assert marker.from_fidelity == Fidelity.RAW_STREAM  # the displaced top tier
        assert eq.substitution_for("avg_power_w", Fidelity.RAW_STREAM) is None
        assert eq.substitution_for("avg_hr_bpm", Fidelity.SUMMARY_ONLY) is None
    finally:
        eq._load.cache_clear()


def test_outcome_token_rejected_as_class_member_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DM-SUB-R1: a class member may carry ONLY a ranked fidelity tier — declaring a
    resolution-outcome token (substituted) as a member tier fails the load closed."""
    classes = tmp_path / "classes.toml"
    classes.write_text(
        """
[[canonical.equivalence_class]]
channel = "avg_power_w"
[[canonical.equivalence_class.members]]
metric = "bogus"
tier = "substituted"
note = "not a tier"
penalty = "none"
"""
    )
    monkeypatch.setenv("WATTWISE_EQUIVALENCE_CLASSES_FILE", str(classes))
    eq._load.cache_clear()
    try:
        with pytest.raises(ValueError, match="resolution outcome"):
            eq.equivalence_classes()
    finally:
        eq._load.cache_clear()


def test_step_targets_resolve_relative_to_signature() -> None:
    """GBO-R29: a power_pct_cp step re-resolves against the CURRENT CP — and returns a
    typed None when the signature lacks the base, never a fabricated absolute target."""
    step = {
        "target_type": "power_pct_cp", "intent": "work",
        "target_low": 90.0, "target_high": 100.0, "duration_s": 1200,
    }
    validate_workout_steps([step])
    resolved = resolve_step_target(step, cp_w=300.0)
    assert resolved == (270.0, 300.0)
    assert resolve_step_target(step, cp_w=None) is None  # typed gap, not a guess


def test_step_schema_requires_exactly_one_extent() -> None:
    """GBO-R29: a step must carry exactly ONE of duration_s / distance_m — both or
    neither refuse validation."""
    base = {"target_type": "open", "intent": "steady"}
    with pytest.raises(WorkoutStepError):
        validate_workout_steps([{**base}])
    with pytest.raises(WorkoutStepError):
        validate_workout_steps([{**base, "duration_s": 600, "distance_m": 1000.0}])
    validate_workout_steps([{**base, "distance_m": 1000.0}])
