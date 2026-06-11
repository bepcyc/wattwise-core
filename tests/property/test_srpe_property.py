"""Property-based tests for the session-RPE load (SRPE-R1, doc 40 §6 fail-closed).

Covers, mirroring the per-metric property pattern (doc 40 §11.1):

* SRPE-T1 — closed-form oracle agreement: the load equals the declared RPE-as-intensity
  mapping ``(RPE/full_scale)^2 * hours * per_hour`` exactly, and the Foster arbitrary
  units (``RPE x duration_min``) ride in the QualityReport.
* SRPE-T2 — invalid values (out-of-scale RPE, non-positive/non-finite duration)
  -> OUT_OF_DOMAIN; reports are never clamped into validity.
* SRPE-T3 — missing input (absent RPE / duration) -> MISSING_REQUIRED_INPUT.
* SRPE-T4 — monotonic: a not-lower RPE at the same duration, or a not-shorter duration
  at the same RPE, never yields a lower load.
* SRPE-T5 — labelling: always ``load_model='srpe_load'``, never relabelled; a reported
  RPE of 0 over a valid duration is an honest Computed(0.0), not an absence.

Determinism (ANL-R30) is checked by repeated evaluation. Generators follow TEST-R2
(realistic CR-10 reports and session durations, with shrinking).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason
from wattwise_core.analytics.srpe import LOAD_MODEL_SRPE, srpe_load

pytestmark = pytest.mark.property

_TOL = 1e-9

_rpe = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_duration = st.floats(min_value=1.0, max_value=12 * 3600.0, allow_nan=False, allow_infinity=False)


# SRPE-T1 independent oracle: hand-computed (RPE, seconds) -> (srpe_load, foster_au) literals,
# NOT a re-statement of the production formula. Each load is computed by hand as
# (RPE/10)^2 * (seconds/3600) * 100 and each Foster AU as RPE * (seconds/60), then pinned as a
# constant. Because these are fixed numbers rather than the same squared expression the impl
# evaluates, a squared->linear regression of the production code (which would, e.g., turn the
# (7, 3600) case from 49.0 into 70.0) makes this table FAIL — the redundant formula-mirror oracle
# could not catch that. Verified independently against the implementation before pinning.
_SRPE_ORACLE: tuple[tuple[float, float, float, float], ...] = (
    # rpe, duration_s, expected_srpe_load, expected_foster_au
    (7.0, 3600.0, 49.0, 420.0),  # one hour at RPE 7 — the doc-40 golden
    (5.0, 3600.0, 25.0, 300.0),  # linear regression would read 50.0
    (10.0, 3600.0, 100.0, 600.0),  # full scale — squared and linear coincide here only
    (1.0, 3600.0, 1.0, 60.0),  # linear regression would read 10.0
    (7.0, 1800.0, 24.5, 210.0),  # half hour
    (6.0, 7200.0, 72.0, 720.0),  # two hours; linear would read 120.0
    (3.0, 600.0, 1.5, 30.0),  # ten minutes; linear would read 5.0
    (8.0, 5400.0, 96.0, 720.0),  # 1.5 h; linear would read 120.0
    (2.0, 3600.0, 4.0, 120.0),  # linear would read 20.0
    (9.0, 2700.0, 60.75, 405.0),  # 45 min; linear would read 67.5
    (0.0, 3600.0, 0.0, 0.0),  # honest zero
    (4.0, 4500.0, 20.0, 300.0),  # 1.25 h; linear would read 50.0
)


@pytest.mark.parametrize(("rpe", "duration_s", "expected_load", "expected_au"), _SRPE_ORACLE)
def test_srpe_matches_independent_oracle(
    rpe: float, duration_s: float, expected_load: float, expected_au: float
) -> None:
    """SRPE-T1: the computed load equals a hand-computed literal; Foster AU is recorded.

    Independent of the production formula: a squared->linear regression would shift every
    non-degenerate row (e.g. (7, 3600) 49.0 -> 70.0) and fail this table.
    """
    result = srpe_load(rpe, duration_s)
    assert isinstance(result, Computed)
    assert result.value == pytest.approx(expected_load, abs=_TOL)
    assert result.quality.extra["foster_au"] == pytest.approx(expected_au, abs=_TOL)


@settings(max_examples=200)
@given(rpe=_rpe, duration_s=_duration)
def test_srpe_is_deterministic_and_labelled(rpe: float, duration_s: float) -> None:
    """SRPE-T5/ANL-R30: repeated evaluation is identical and always carries its own label."""
    first = srpe_load(rpe, duration_s, sport="strength")
    second = srpe_load(rpe, duration_s, sport="strength")
    assert isinstance(first, Computed)
    assert isinstance(second, Computed)
    assert first.value == second.value
    assert first.quality.extra["load_model"] == LOAD_MODEL_SRPE
    assert first.provenance.reference_params["load_model"] == LOAD_MODEL_SRPE
    assert first.provenance.sport == "strength"
    assert first.provenance.channels == ("perceived_exertion",)


@settings(max_examples=200)
@given(low=_rpe, high=_rpe, duration_s=_duration)
def test_srpe_monotonic_in_rpe(low: float, high: float, duration_s: float) -> None:
    """SRPE-T4: at a fixed duration, a not-lower report never yields a lower load."""
    lo, hi = sorted((low, high))
    lo_res = srpe_load(lo, duration_s)
    hi_res = srpe_load(hi, duration_s)
    assert isinstance(lo_res, Computed)
    assert isinstance(hi_res, Computed)
    assert hi_res.value >= lo_res.value - _TOL


@settings(max_examples=200)
@given(rpe=_rpe, short=_duration, long=_duration)
def test_srpe_monotonic_in_duration(rpe: float, short: float, long: float) -> None:
    """SRPE-T4: at a fixed report, a not-shorter session never yields a lower load."""
    lo, hi = sorted((short, long))
    lo_res = srpe_load(rpe, lo)
    hi_res = srpe_load(rpe, hi)
    assert isinstance(lo_res, Computed)
    assert isinstance(hi_res, Computed)
    assert hi_res.value >= lo_res.value - _TOL


def test_srpe_missing_inputs_fail_closed() -> None:
    """SRPE-T3: an absent report or duration is MISSING_REQUIRED_INPUT, never a default."""
    for rpe, duration in ((None, 3600.0), (7.0, None), (None, None)):
        result = srpe_load(rpe, duration)
        assert isinstance(result, Unavailable)
        assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


@settings(max_examples=100)
@given(
    rpe=st.one_of(
        st.floats(max_value=-1e-9, min_value=-1e6, allow_nan=False),
        st.floats(min_value=10.0 + 1e-6, max_value=1e6, allow_nan=False),
        st.just(math.nan),
        st.just(math.inf),
    ),
    duration_s=_duration,
)
def test_srpe_out_of_scale_rpe_is_out_of_domain(rpe: float, duration_s: float) -> None:
    """SRPE-T2: a present-but-invalid report is OUT_OF_DOMAIN — never clamped into [0, 10]."""
    result = srpe_load(rpe, duration_s)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


@settings(max_examples=100)
@given(
    rpe=_rpe,
    duration_s=st.one_of(
        st.floats(max_value=0.0, min_value=-1e6, allow_nan=False),
        st.just(math.nan),
        st.just(math.inf),
    ),
)
def test_srpe_invalid_duration_is_out_of_domain(rpe: float, duration_s: float) -> None:
    """SRPE-T2: a non-positive or non-finite duration is OUT_OF_DOMAIN (never divided into)."""
    result = srpe_load(rpe, duration_s)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


def test_srpe_zero_report_is_honest_zero_load() -> None:
    """SRPE-T5: RPE 0 over a valid duration is a real Computed(0.0) report, not an absence."""
    result = srpe_load(0.0, 3600.0)
    assert isinstance(result, Computed)
    assert result.value == 0.0
    assert result.quality.extra["load_model"] == LOAD_MODEL_SRPE
