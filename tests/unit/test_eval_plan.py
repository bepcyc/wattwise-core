"""Multi-day-PLAN coach-quality eval-suite tests (QA-EVAL-R2.5 / COACH-R2 / COACH-R3).

These tests gate the ``plan`` suite the way ``test_eval.py`` gates ``readiness``: the
positive fixtures MUST clear all three deterministic 100% certificates — grounding (every
prescription grounds fail-closed through the SHIPPED grounder, COACH-R2/GROUND-R8),
progression (the weekly CTL ramp computed via the canonical PMC EWMA stays within the
stated bound, QA-EVAL-R2.5), and consistency (the plan's peak day-load is consistent with
the readiness verdict, COACH-R3) — and the dataset's NEGATIVE fixtures drive the teeth:
each is asserted to FAIL its named certificate, so the gate is provably non-vacuous (a
real defect — an ungrounded prescription, an over-steep ramp, or a low-readiness/high-load
co-occurrence — cannot pass). Every test is deterministic and network-free (TIER-R1).
"""

from __future__ import annotations

import pytest

from wattwise_core.eval.plan_suite import (
    PlanGrade,
    _consistency_failure,
    _grounding_failure,
    _load,
    _progression_failure,
    _weekly_ctl_ramp,
    grade_plan,
)

pytestmark = pytest.mark.unit


def _case(case_id: str, *, group: str = "cases") -> dict:
    """Fetch one named fixture from the ``plan`` dataset's positive/negative group."""
    return next(c for c in _load()[group] if c["id"] == case_id)


# --------------------------------------------------------------------------- #
# Dataset shape (QA-EVAL-R1 / EVAL-R8)                                         #
# --------------------------------------------------------------------------- #


def test_plan_dataset_is_versioned_and_unique() -> None:
    """The plan dataset carries a version and uniquely-id'd positive + negative cases."""
    data = _load()
    assert data["dataset_version"], "dataset MUST carry a version (QA-EVAL-R1)"
    assert data["suite"] == "plan"
    assert data["cases"], "MUST carry at least one positive case"
    assert data["negative_cases"], "MUST carry negative (teeth) cases"
    ids = [c["id"] for c in (*data["cases"], *data["negative_cases"])]
    assert len(ids) == len(set(ids)), "case ids MUST be unique"


# --------------------------------------------------------------------------- #
# Positive cases clear all three certificates (QA-EVAL-R2.5 100% gate)         #
# --------------------------------------------------------------------------- #


def test_grade_plan_passes() -> None:
    """Every positive plan fixture clears grounding, progression, and consistency at 100%."""
    grade = grade_plan()
    assert grade.passed
    assert grade.failures == ()
    assert grade.grounding_rate == 1.0
    assert grade.progression_rate == 1.0
    assert grade.consistency_rate == 1.0


def test_grade_plan_grounds_every_prescription() -> None:
    """No positive plan leaves a prescription ungrounded — each names a canonical item."""
    for case in _load()["cases"]:
        assert _grounding_failure(case) is None, case["id"]


def test_grade_plan_scrubs_planted_ungrounded_prescription() -> None:
    """The scrub fixture's invented number AND unknown workout name are scrubbed (COACH-R2)."""
    case = _case("ungrounded-prescription-scrubbed")
    # Sanity: the dataset really does plant both an out-of-band number and an unknown name.
    expected = set(case["expected"]["scrubbed_prescriptions"])
    assert expected == {"cp_w@999.0", "Galactic Power Blaster"}
    # Through the SHIPPED grounder the planted prescriptions are scrubbed -> no failure.
    assert _grounding_failure(case) is None


# --------------------------------------------------------------------------- #
# Progression certificate teeth (QA-EVAL-R2.5) — canonical PMC EWMA decides    #
# --------------------------------------------------------------------------- #


def test_progression_certificate_flags_over_steep_ramp() -> None:
    """Teeth: a plan whose weekly CTL ramp exceeds the stated bound MUST be flagged."""
    case = _case("ramp-too-steep", group="negative_cases")
    ramp = _weekly_ctl_ramp(case)
    bound = case["progression_bound"]["max_ctl_ramp_per_week"]
    assert ramp > bound, f"the over-steep fixture must ramp past its bound (got {ramp:.2f})"
    reason = _progression_failure(case)
    assert reason is not None
    assert "exceeds stated bound" in reason


def test_progression_certificate_allows_a_taper() -> None:
    """A taper (negative ramp) is always within an upward progression bound (not flagged)."""
    case = _case("recovery-week-grounded-ease")
    assert _weekly_ctl_ramp(case) < 0.0, "the recovery week should de-load (negative ramp)"
    assert _progression_failure(case) is None


def test_progression_uses_canonical_pmc_not_a_reimplementation() -> None:
    """The ramp is the canonical PMC EWMA result, not a naive TSS sum (QA-EVAL-R2.5)."""
    case = _case("build-week-grounded-go")
    ramp = _weekly_ctl_ramp(case)
    naive_total_tss = sum(d["tss"] for d in case["days"])
    # A naive cumulative-TSS ramp would be ~hundreds; the EWMA ramp is bounded and small.
    assert abs(ramp) < naive_total_tss
    assert ramp == pytest.approx(-0.57, abs=0.1)


# --------------------------------------------------------------------------- #
# Consistency certificate teeth (COACH-R3) — verdict<->load                    #
# --------------------------------------------------------------------------- #


def test_consistency_certificate_flags_low_readiness_high_load() -> None:
    """Teeth: an 'ease' verdict with a VO2/peak day is the COACH-R3 forbidden co-occurrence."""
    case = _case("low-readiness-high-load-inconsistent", group="negative_cases")
    reason = _consistency_failure(case)
    assert reason is not None
    assert "COACH-R3 inconsistency" in reason


def test_consistency_certificate_allows_go_with_a_peak_day() -> None:
    """A 'go' verdict may schedule a high-load peak day (consistent, not flagged)."""
    case = _case("build-week-grounded-go")
    assert _consistency_failure(case) is None


# --------------------------------------------------------------------------- #
# Grade gate teeth (mirrors the readiness FIX-1 non-empty-failures guard)      #
# --------------------------------------------------------------------------- #


def test_plan_grade_fails_on_nonempty_failures_despite_perfect_rates() -> None:
    """A recorded failure MUST fail the gate even when all three rates read 1.0."""
    grade = PlanGrade(
        total=1,
        grounded=1,
        within_bound=1,
        consistent=1,
        failures=("synthetic: a prescription survived ungrounded",),
    )
    assert grade.grounding_rate == 1.0
    assert grade.progression_rate == 1.0
    assert grade.consistency_rate == 1.0
    assert grade.passed is False


def test_plan_grade_passes_only_when_failures_empty() -> None:
    """Control: identical perfect rates with NO recorded failures still pass the gate."""
    grade = PlanGrade(total=1, grounded=1, within_bound=1, consistent=1, failures=())
    assert grade.passed is True
