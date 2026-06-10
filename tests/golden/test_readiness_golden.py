"""Golden-reference tests for the readiness/form assessment oracle (QA-EVAL-R2.4).

Requirement coverage: QA-EVAL-R2.4 (verdict direction consistent with metrics —
deep-negative form is never a hard "go" day), COACH-R1 #2 / SCHEMA-R3 (verdict is a
typed state, not a number; missing inputs recorded not guessed), COACH-R3 / EVAL-R5
(code decides the verdict; :func:`readiness_consistent` is the certificate),
GROUND-R6/R7 (form unavailable => abstain; HRV unavailable => recorded), TEST-R1.

Fixture origin / derivation note
--------------------------------
Each expected verdict is HAND-DERIVED directly from the spec band cutoffs, which are
INDEPENDENT of the module under test:

    form > 5.0                  => GO        ("fresh")
    -10.0 <= form <= 5.0        => MAINTAIN  ("neutral")
    -20.0 <= form < -10.0       => EASE      ("productive_fatigue")
    form < -20.0                => REST      ("deep_fatigue")

The HRV nudge moves ONE step toward REST along GO->MAINTAIN->EASE->REST (clamped at
REST) iff ``hrv_rmssd < hrv_baseline * (1 - 0.10)``; it can only ever increase
caution. The boundary values (5.0, -10.0, -20.0) are pinned explicitly to lock the
inclusive/exclusive band edges.
"""

from __future__ import annotations

import math

import pytest

from wattwise_core.analytics.readiness import (
    ReadinessAssessment,
    assess_readiness,
    readiness_consistent,
)
from wattwise_core.domain.enums import ReadinessVerdict

pytestmark = pytest.mark.golden


# --- one explicit case per band -------------------------------------------------


def test_fresh_form_is_go() -> None:
    """form > 5 => GO ("fresh"); HRV absent so it is recorded unavailable."""
    a = assess_readiness(form=12.0)
    assert a.verdict is ReadinessVerdict.GO
    assert a.rationale == "fresh"
    assert a.form == 12.0
    assert a.hrv_rmssd is None
    assert a.inputs_used == ("form",)
    assert a.inputs_unavailable == ("hrv",)


def test_neutral_form_is_maintain() -> None:
    """-10 <= form <= 5 => MAINTAIN ("neutral")."""
    a = assess_readiness(form=0.0)
    assert a.verdict is ReadinessVerdict.MAINTAIN
    assert a.rationale == "neutral"
    assert a.inputs_used == ("form",)
    assert a.inputs_unavailable == ("hrv",)


def test_productive_fatigue_is_ease() -> None:
    """-20 <= form < -10 => EASE ("productive_fatigue")."""
    a = assess_readiness(form=-15.0)
    assert a.verdict is ReadinessVerdict.EASE
    assert a.rationale == "productive_fatigue"


def test_deep_fatigue_is_rest() -> None:
    """form < -20 => REST ("deep_fatigue") — the QA-EVAL-R2.4 core direction."""
    a = assess_readiness(form=-30.0)
    assert a.verdict is ReadinessVerdict.REST
    assert a.rationale == "deep_fatigue"


# --- boundary values (inclusive/exclusive edges) --------------------------------


def test_boundary_form_eq_fresh_floor_is_maintain() -> None:
    """form == 5.0 is INCLUSIVE in MAINTAIN (GO is strictly above 5.0)."""
    a = assess_readiness(form=5.0)
    assert a.verdict is ReadinessVerdict.MAINTAIN
    assert a.rationale == "neutral"


def test_boundary_form_eq_neutral_floor_is_maintain() -> None:
    """form == -10.0 is INCLUSIVE in MAINTAIN (EASE is strictly below -10.0)."""
    a = assess_readiness(form=-10.0)
    assert a.verdict is ReadinessVerdict.MAINTAIN
    assert a.rationale == "neutral"


def test_boundary_form_eq_fatigue_floor_is_ease() -> None:
    """form == -20.0 is INCLUSIVE in EASE (REST is strictly below -20.0)."""
    a = assess_readiness(form=-20.0)
    assert a.verdict is ReadinessVerdict.EASE
    assert a.rationale == "productive_fatigue"


# --- HRV nudge ------------------------------------------------------------------


def test_hrv_suppression_nudges_neutral_to_ease() -> None:
    """Neutral form + suppressed HRV => one step more cautious (MAINTAIN -> EASE)."""
    # baseline 50, suppressed threshold = 50 * 0.9 = 45; 40 < 45 => suppressed.
    a = assess_readiness(form=0.0, hrv_rmssd=40.0, hrv_baseline=50.0)
    assert a.verdict is ReadinessVerdict.EASE
    assert a.rationale == "hrv_suppressed"
    assert a.inputs_used == ("form", "hrv")
    assert a.inputs_unavailable == ()
    assert a.hrv_rmssd == 40.0


def test_hrv_suppression_clamps_at_rest() -> None:
    """Deep-fatigue form already REST; a suppressed HRV nudge clamps at REST."""
    a = assess_readiness(form=-30.0, hrv_rmssd=10.0, hrv_baseline=50.0)
    assert a.verdict is ReadinessVerdict.REST
    assert a.rationale == "hrv_suppressed"


def test_hrv_present_but_normal_no_nudge() -> None:
    """HRV at/above the suppression threshold does NOT nudge; rationale stays band tag."""
    # threshold = 50 * 0.9 = 45; 48 >= 45 => no nudge.
    a = assess_readiness(form=0.0, hrv_rmssd=48.0, hrv_baseline=50.0)
    assert a.verdict is ReadinessVerdict.MAINTAIN
    assert a.rationale == "neutral"
    assert a.inputs_used == ("form", "hrv")
    assert a.inputs_unavailable == ()


def test_hrv_never_nudges_toward_go() -> None:
    """HRV is fail-safe: a normal HRV on a fresh day never escalates beyond the band."""
    a = assess_readiness(form=12.0, hrv_rmssd=60.0, hrv_baseline=50.0)
    assert a.verdict is ReadinessVerdict.GO  # unchanged; no upward nudge exists
    assert a.rationale == "fresh"


def test_hrv_absent_records_unavailable() -> None:
    """No HRV inputs => "hrv" in inputs_unavailable; verdict from form alone."""
    a = assess_readiness(form=-15.0)
    assert a.inputs_used == ("form",)
    assert a.inputs_unavailable == ("hrv",)
    assert a.verdict is ReadinessVerdict.EASE


def test_hrv_partial_inputs_are_unavailable() -> None:
    """Only one of (rmssd, baseline) present => HRV is unusable, recorded unavailable."""
    a = assess_readiness(form=0.0, hrv_rmssd=40.0, hrv_baseline=None)
    assert a.inputs_used == ("form",)
    assert a.inputs_unavailable == ("hrv",)
    assert a.verdict is ReadinessVerdict.MAINTAIN
    assert a.rationale == "neutral"


def test_hrv_nonpositive_baseline_is_unavailable() -> None:
    """A zero/negative baseline is invalid => HRV unusable, no nudge."""
    a = assess_readiness(form=0.0, hrv_rmssd=40.0, hrv_baseline=0.0)
    assert a.inputs_used == ("form",)
    assert a.inputs_unavailable == ("hrv",)
    assert a.verdict is ReadinessVerdict.MAINTAIN


# --- form unavailable / non-finite ----------------------------------------------


def test_form_none_is_verdict_none() -> None:
    """form=None => cannot assess; verdict None, rationale form_unavailable (GROUND-R6)."""
    a = assess_readiness(form=None)
    assert a == ReadinessAssessment(
        verdict=None,
        form=None,
        hrv_rmssd=None,
        inputs_used=(),
        inputs_unavailable=("form", "hrv"),
        rationale="form_unavailable",
    )


def test_form_none_with_hrv_records_only_form_unavailable_for_used() -> None:
    """form=None but HRV present: still cannot assess; form drives the unavailability."""
    a = assess_readiness(form=None, hrv_rmssd=40.0, hrv_baseline=50.0)
    assert a.verdict is None
    assert a.rationale == "form_unavailable"
    assert a.inputs_used == ()
    # HRV was usable, so it is NOT listed unavailable; only form is.
    assert a.inputs_unavailable == ("form",)
    assert a.hrv_rmssd == 40.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_form_treated_unavailable(bad: float) -> None:
    """Non-finite form (nan/inf) fails closed like None (ANL-R32 / GROUND-R7)."""
    a = assess_readiness(form=bad)
    assert a.verdict is None
    assert a.form is None
    assert a.rationale == "form_unavailable"
    assert a.inputs_used == ()
    assert a.inputs_unavailable == ("form", "hrv")


# --- readiness_consistent (EVAL-R5 / COACH-R3) ----------------------------------


def test_consistent_deep_negative_form_with_go_is_false() -> None:
    """The QA-EVAL-R2.4 core: a deep-negative form can NEVER certify a GO verdict."""
    assert readiness_consistent(ReadinessVerdict.GO, form=-30.0) is False


def test_consistent_matching_verdict_is_true() -> None:
    """The deterministic verdict certifies itself."""
    assert readiness_consistent(ReadinessVerdict.REST, form=-30.0) is True
    assert readiness_consistent(ReadinessVerdict.GO, form=12.0) is True


def test_consistent_form_none_is_false() -> None:
    """No verdict can be certified against a missing form (GROUND-R6)."""
    assert readiness_consistent(ReadinessVerdict.GO, form=None) is False
    assert readiness_consistent(ReadinessVerdict.REST, form=None) is False


def test_consistent_respects_hrv_nudge() -> None:
    """Consistency uses the SAME inputs incl. HRV: nudged verdict is the certified one."""
    # neutral form nudged to EASE by suppressed HRV.
    assert (
        readiness_consistent(ReadinessVerdict.EASE, form=0.0, hrv_rmssd=40.0, hrv_baseline=50.0)
        is True
    )
    # the un-nudged MAINTAIN is NOT consistent once HRV suppression is present.
    assert (
        readiness_consistent(ReadinessVerdict.MAINTAIN, form=0.0, hrv_rmssd=40.0, hrv_baseline=50.0)
        is False
    )
