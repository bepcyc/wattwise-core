"""Property-based tests for the readiness/form assessment oracle (QA-EVAL-R2.4).

Per-metric property IDs:

* RDY-T1 -- determinism (ANL-R30): identical inputs => identical assessment.
* RDY-T2 -- monotonicity: for finite forms a < b (HRV absent), the verdict for the
  fresher form ``b`` is NEVER less aggressive than for ``a`` (aggressiveness GO >
  MAINTAIN > EASE > REST).
* RDY-T3 -- deep-negative invariant (the QA-EVAL-R2.4 core): any ``form <
  READINESS_FATIGUE_FLOOR`` (HRV absent) yields REST, and NEVER GO.
* RDY-T4 -- HRV is fail-safe: an HRV nudge never INCREASES aggressiveness vs the
  no-HRV verdict (it is monotone-cautious).
* RDY-T5 -- consistency round-trip (EVAL-R5 / COACH-R3): whenever the deterministic
  verdict is not None, ``readiness_consistent(assess(...).verdict, ...)`` is True.

Generators (TEST-R2): finite forms over a wide TSB span (well past every band edge),
HRV rmssd/baseline over realistic ranges, with shrinking.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.constants import READINESS_FATIGUE_FLOOR
from wattwise_core.analytics.readiness import assess_readiness, readiness_consistent
from wattwise_core.domain.enums import ReadinessVerdict

pytestmark = pytest.mark.property

_SETTINGS = settings(max_examples=200, deadline=None)

# Finite form (TSB) spanning well past every band edge in both directions.
_form = st.floats(min_value=-80.0, max_value=80.0, allow_nan=False, allow_infinity=False)
# Realistic HRV RMSSD (ms) and a strictly-positive baseline.
_hrv = st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False)
_baseline = st.floats(min_value=5.0, max_value=150.0, allow_nan=False, allow_infinity=False)

# Aggressiveness rank GO=0 (most) .. REST=3 (least), for monotonicity comparisons.
_RANK: dict[ReadinessVerdict, int] = {
    ReadinessVerdict.GO: 0,
    ReadinessVerdict.MAINTAIN: 1,
    ReadinessVerdict.EASE: 2,
    ReadinessVerdict.REST: 3,
}


def _aggressiveness(verdict: ReadinessVerdict | None) -> int:
    """Lower == more aggressive (GO). ``verdict`` is never None on these paths."""
    assert verdict is not None
    return _RANK[verdict]


# --- RDY-T1: determinism --------------------------------------------------------


@_SETTINGS
@given(form=_form, hrv=st.none() | _hrv, baseline=st.none() | _baseline)
def test_t1_determinism(form: float, hrv: float | None, baseline: float | None) -> None:
    a = assess_readiness(form=form, hrv_rmssd=hrv, hrv_baseline=baseline)
    b = assess_readiness(form=form, hrv_rmssd=hrv, hrv_baseline=baseline)
    assert a == b


# --- RDY-T2: monotonicity (HRV absent) ------------------------------------------


@_SETTINGS
@given(a=_form, b=_form)
def test_t2_monotone_fresher_never_less_aggressive(a: float, b: float) -> None:
    """For finite forms a < b, aggressiveness(assess(b)) >= aggressiveness(assess(a))."""
    lo, hi = sorted((a, b))
    v_lo = assess_readiness(form=lo).verdict
    v_hi = assess_readiness(form=hi).verdict
    # Higher form (hi, fresher) is never LESS aggressive (never a higher rank number).
    assert _aggressiveness(v_hi) <= _aggressiveness(v_lo)


# --- RDY-T3: deep-negative invariant (QA-EVAL-R2.4 core) ------------------------


@_SETTINGS
@given(
    form=st.floats(
        max_value=READINESS_FATIGUE_FLOOR - 1e-9,
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e6,
    )
)
def test_t3_deep_negative_form_is_rest_never_go(form: float) -> None:
    """Any form strictly below the fatigue floor (HRV absent) => REST, never GO."""
    v = assess_readiness(form=form).verdict
    assert v is ReadinessVerdict.REST
    assert v is not ReadinessVerdict.GO


# --- RDY-T4: HRV nudge is fail-safe (monotone-cautious) -------------------------


@_SETTINGS
@given(form=_form, hrv=_hrv, baseline=_baseline)
def test_t4_hrv_never_increases_aggressiveness(
    form: float, hrv: float, baseline: float
) -> None:
    """The HRV nudge never makes the verdict MORE aggressive than form alone."""
    no_hrv = assess_readiness(form=form).verdict
    with_hrv = assess_readiness(form=form, hrv_rmssd=hrv, hrv_baseline=baseline).verdict
    # with_hrv is at least as cautious (rank >=) as the form-only verdict.
    assert _aggressiveness(with_hrv) >= _aggressiveness(no_hrv)


# --- RDY-T5: consistency round-trip (EVAL-R5 / COACH-R3) ------------------------


@_SETTINGS
@given(form=_form, hrv=st.none() | _hrv, baseline=st.none() | _baseline)
def test_t5_consistent_roundtrip(
    form: float, hrv: float | None, baseline: float | None
) -> None:
    """A non-None deterministic verdict always certifies itself (round-trip)."""
    v = assess_readiness(form=form, hrv_rmssd=hrv, hrv_baseline=baseline).verdict
    assert v is not None  # finite form here => always assessable
    assert (
        readiness_consistent(v, form=form, hrv_rmssd=hrv, hrv_baseline=baseline) is True
    )
