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

from wattwise_core.analytics.constants import (
    SRPE_LOAD_PER_HOUR_AT_FULL_SCALE,
    SRPE_RPE_FULL_SCALE,
)
from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason
from wattwise_core.analytics.srpe import LOAD_MODEL_SRPE, srpe_load

pytestmark = pytest.mark.property

_TOL = 1e-9

_rpe = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_duration = st.floats(min_value=1.0, max_value=12 * 3600.0, allow_nan=False, allow_infinity=False)


def _oracle(rpe: float, duration_s: float) -> float:
    """Independent closed-form oracle for the declared RPE-as-intensity mapping."""
    intensity = rpe / SRPE_RPE_FULL_SCALE
    return intensity * intensity * (duration_s / 3600.0) * SRPE_LOAD_PER_HOUR_AT_FULL_SCALE


@settings(max_examples=200)
@given(rpe=_rpe, duration_s=_duration)
def test_srpe_matches_independent_oracle(rpe: float, duration_s: float) -> None:
    """SRPE-T1: the computed load equals the declared mapping; Foster AU is recorded."""
    result = srpe_load(rpe, duration_s)
    assert isinstance(result, Computed)
    assert result.value == pytest.approx(_oracle(rpe, duration_s), abs=_TOL)
    assert result.quality.extra["foster_au"] == pytest.approx(rpe * duration_s / 60.0, abs=_TOL)


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
