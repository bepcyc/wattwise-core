"""Property-based tests for W' balance (doc 40 §6; WBAL-T1..T7, TEST-R3).

Covered property IDs (Section 11.1):

* **WBAL-T1** — ``P == CP`` constant ⇒ ``W'bal == W'`` for every t.
* **WBAL-T2** — upper bound ``W'bal(t) <= W'`` always, and every value finite.
* **WBAL-T3** — expenditure is monotone: a constant ``P > CP`` segment strictly
  decreases by ``(P - CP)`` each step.
* **WBAL-T4** — full recovery: after a long sub-CP tail the balance converges back
  to ``W'`` (using the per-second instantaneous ``τ_W``).
* **WBAL-T7** — negative allowed by default vs ``floor=True`` ⇒ ``max(0, raw)``.

Plus the fail-closed degenerate cases (TEST-R3): empty stream, all-``null``,
missing CP, missing W', non-finite/non-positive params each map to the exact typed
reason.

(WBAL-T5 — integral↔differential parity — and WBAL-T6 — published golden — live in
the golden suite; the integral form is not exposed here, so parity has no second
implementation to compare against and is covered by the golden differential.)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.constants import DEFAULT_CLOSED_FORM_ABS_TOL
from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason
from wattwise_core.analytics.series import FloatArray
from wattwise_core.analytics.wbal import wbal

pytestmark = pytest.mark.property

_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Finite, sensible cycling reference params.
_cp = st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_wprime = st.floats(min_value=1_000.0, max_value=50_000.0, allow_nan=False, allow_infinity=False)
_power_sample = st.floats(min_value=0.0, max_value=2_000.0, allow_nan=False, allow_infinity=False)


def _abs_tol(value: float) -> float:
    return DEFAULT_CLOSED_FORM_ABS_TOL * max(1.0, abs(value))


@_SETTINGS
@given(cp=_cp, w_prime=_wprime, n=st.integers(min_value=1, max_value=400))
def test_wbal_t1_p_equals_cp_constant_stays_at_w_prime(cp: float, w_prime: float, n: int) -> None:
    """WBAL-T1: constant P == CP ⇒ series is exactly W' at every second."""
    power = np.full(n, cp, dtype=np.float64)
    result = wbal(power, cp, w_prime)
    assert isinstance(result, Computed)
    series = result.value.series
    assert series.shape == (n,)
    assert np.all(np.abs(series - w_prime) <= _abs_tol(w_prime))


@_SETTINGS
@given(
    cp=_cp,
    w_prime=_wprime,
    powers=st.lists(_power_sample, min_size=1, max_size=300),
    floor=st.booleans(),
)
def test_wbal_t2_upper_bound_and_finite(
    cp: float, w_prime: float, powers: list[float], floor: bool
) -> None:
    """WBAL-T2: W'bal(t) <= W' for all t (within tol) and every value finite."""
    power = np.array(powers, dtype=np.float64)
    result = wbal(power, cp, w_prime, floor=floor)
    assert isinstance(result, Computed)
    series = result.value.series
    assert np.all(np.isfinite(series))
    # Upper bound (tol for float recovery arithmetic).
    assert np.all(series <= w_prime + _abs_tol(w_prime))
    if floor:
        assert np.all(series >= 0.0)
    # Reported minimum equals the series minimum.
    assert math.isclose(
        result.value.w_prime_balance_min,
        float(np.min(series)),
        abs_tol=_abs_tol(w_prime),
    )


@_SETTINGS
@given(
    cp=_cp,
    w_prime=_wprime,
    above=st.floats(min_value=1.0, max_value=800.0, allow_nan=False),
    n=st.integers(min_value=2, max_value=200),
)
def test_wbal_t3_expenditure_monotone_decreasing(
    cp: float, w_prime: float, above: float, n: int
) -> None:
    """WBAL-T3: constant P>CP decreases the raw balance by exactly (P-CP)/step."""
    p = cp + above
    power = np.full(n, p, dtype=np.float64)
    result = wbal(power, cp, w_prime, floor=False)
    assert isinstance(result, Computed)
    series = result.value.series
    diffs = np.diff(series)
    # Each step drops by (P - CP) (constant expenditure), strictly monotone down.
    assert np.all(diffs < 0.0)
    assert np.all(np.abs(diffs - (-(p - cp))) <= _abs_tol(w_prime))


@_SETTINGS
@given(cp=_cp, w_prime=_wprime, deficit=st.floats(min_value=20.0, max_value=300.0))
def test_wbal_t4_full_recovery_to_w_prime(cp: float, w_prime: float, deficit: float) -> None:
    """WBAL-T4: a long sub-CP tail reconstitutes W' back toward W'.

    Deplete with a short hard burst, then recover for a long time below CP; the
    final balance must approach W' (monotonically rising during the recovery tail).
    """
    assume(cp - deficit >= 0.0)
    burst = np.full(5, cp + 400.0, dtype=np.float64)
    # Long recovery tail well below CP.
    recover = np.full(20_000, cp - deficit, dtype=np.float64)
    power = np.concatenate([burst, recover])
    result = wbal(power, cp, w_prime, floor=False)
    assert isinstance(result, Computed)
    series = result.value.series
    # Recovery tail is monotonically non-decreasing toward W'.
    tail = series[5:]
    assert np.all(np.diff(tail) >= -_abs_tol(w_prime))
    # Converged close to W' after a very long tail.
    assert series[-1] <= w_prime + _abs_tol(w_prime)
    assert math.isclose(series[-1], w_prime, rel_tol=1e-6, abs_tol=1.0)


@_SETTINGS
@given(
    cp=_cp,
    w_prime=_wprime,
    powers=st.lists(_power_sample, min_size=1, max_size=300),
)
def test_wbal_t7_floor_is_max0_of_raw(cp: float, w_prime: float, powers: list[float]) -> None:
    """WBAL-T7: floor=True yields exactly max(0, raw) of the floor=False series."""
    power = np.array(powers, dtype=np.float64)
    raw = wbal(power, cp, w_prime, floor=False)
    floored = wbal(power, cp, w_prime, floor=True)
    assert isinstance(raw, Computed)
    assert isinstance(floored, Computed)
    expected = np.maximum(raw.value.series, 0.0)
    assert np.allclose(floored.value.series, expected, atol=_abs_tol(w_prime), rtol=0.0)
    # Raw is allowed to be negative; floored never is.
    assert np.all(floored.value.series >= 0.0)


@_SETTINGS
@given(
    powers=st.lists(_power_sample, min_size=1, max_size=50),
    w_prime=_wprime,
)
def test_wbal_fail_closed_missing_cp(powers: list[float], w_prime: float) -> None:
    """TEST-R3: missing CP ⇒ MISSING_REQUIRED_INPUT."""
    power = np.array(powers, dtype=np.float64)
    result = wbal(power, None, w_prime)
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.MISSING_REQUIRED_INPUT


@_SETTINGS
@given(
    powers=st.lists(_power_sample, min_size=1, max_size=50),
    cp=_cp,
)
def test_wbal_fail_closed_missing_w_prime(powers: list[float], cp: float) -> None:
    """TEST-R3: missing W' ⇒ MISSING_REQUIRED_INPUT."""
    power = np.array(powers, dtype=np.float64)
    result = wbal(power, cp, None)
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.MISSING_REQUIRED_INPUT


@_SETTINGS
@given(cp=_cp, w_prime=_wprime, n=st.integers(min_value=0, max_value=50))
def test_wbal_fail_closed_empty_and_all_gap(cp: float, w_prime: float, n: int) -> None:
    """TEST-R3: empty stream and all-``null`` stream ⇒ MISSING_REQUIRED_INPUT."""
    empty: FloatArray = np.array([], dtype=np.float64)
    result_empty = wbal(empty, cp, w_prime)
    assert isinstance(result_empty, Unavailable)
    assert result_empty.reason == UnavailableReason.MISSING_REQUIRED_INPUT

    all_gap = np.full(max(n, 1), np.nan, dtype=np.float64)
    result_gap = wbal(all_gap, cp, w_prime)
    assert isinstance(result_gap, Unavailable)
    assert result_gap.reason == UnavailableReason.MISSING_REQUIRED_INPUT


@_SETTINGS
@given(
    powers=st.lists(_power_sample, min_size=1, max_size=50),
    w_prime=_wprime,
    bad_cp=st.sampled_from([float("nan"), float("inf"), float("-inf"), 0.0, -10.0]),
)
def test_wbal_fail_closed_bad_cp_out_of_domain(
    powers: list[float], w_prime: float, bad_cp: float
) -> None:
    """TEST-R3/WBAL-R6: non-finite or non-positive CP ⇒ OUT_OF_DOMAIN."""
    power = np.array(powers, dtype=np.float64)
    result = wbal(power, bad_cp, w_prime)
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.OUT_OF_DOMAIN


@_SETTINGS
@given(
    powers=st.lists(_power_sample, min_size=1, max_size=50),
    cp=_cp,
    bad_w=st.sampled_from([float("nan"), float("inf"), float("-inf"), 0.0, -10.0]),
)
def test_wbal_fail_closed_bad_w_prime_out_of_domain(
    powers: list[float], cp: float, bad_w: float
) -> None:
    """TEST-R3/WBAL-R6: non-finite or non-positive W' ⇒ OUT_OF_DOMAIN."""
    power = np.array(powers, dtype=np.float64)
    result = wbal(power, cp, bad_w)
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.OUT_OF_DOMAIN


@_SETTINGS
@given(
    cp=_cp,
    w_prime=_wprime,
    powers=st.lists(_power_sample, min_size=1, max_size=200),
)
def test_wbal_determinism(cp: float, w_prime: float, powers: list[float]) -> None:
    """ANL-R30: identical inputs ⇒ bit-identical series across repeated calls."""
    power = np.array(powers, dtype=np.float64)
    r1 = wbal(power, cp, w_prime)
    r2 = wbal(power, cp, w_prime)
    assert isinstance(r1, Computed)
    assert isinstance(r2, Computed)
    assert np.array_equal(r1.value.series, r2.value.series)
