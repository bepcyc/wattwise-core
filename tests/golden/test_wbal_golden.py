"""Golden-reference tests for W' balance (doc 40 §6; WBAL-T6 + boundary/floor).

Fixture origin (TEST-R4)
------------------------
All expected values below are hand-derived from the **Skiba (2012) per-second
differential** recurrence (NOT a mean-deficit closed form), seeded
``W'bal(0) = W'`` with ``Δt = 1 s``, using the published Skiba constants
``A=546, B=-0.01, C=316``::

    expenditure (P >= CP):  W'bal(t) = W'bal(t-1) - (P - CP)
    recovery    (P <  CP):  W'bal(t) = W'bal(t-1) + (W' - W'bal(t-1))*(1 - e^(-1/τ_W))
    τ_W = 546 * e^(-0.01 * (CP - P)) + 316

The arithmetic for the primary case (CP=250 W, W'=20000 J, P=[300,400,100,0]) was
worked out by hand and cross-checked with an independent stand-alone reference
script (see the derivation in the docstring of each case):

* t0: P=300 ≥ CP → 20000 - (300-250)          = 19950.0
* t1: P=400 ≥ CP → 19950 - (400-250)          = 19800.0
* t2: P=100 < CP → d_cp=150, τ=546·e^-1.5+316 = 437.8290674410427,
      19800 + (20000-19800)·(1 - e^(-1/τ))    = 19800.456278006666
* t3: P=0   < CP → d_cp=250, τ=546·e^-2.5+316 = 360.8184092486487,
      19800.456… + (20000-19800.456…)·(1 - e^(-1/τ)) = 19801.008543236825

Citation: Skiba, Chidnok, Vanhatalo & Jones (2012), "Modeling the Expenditure and
Reconstitution of Work Capacity above Critical Power," *Med. Sci. Sports Exerc.*
44(8):1526-1532.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wattwise_core.analytics.constants import (
    DEFAULT_CLOSED_FORM_ABS_TOL,
    SKIBA_TAU_A,
    SKIBA_TAU_B,
    SKIBA_TAU_C,
)
from wattwise_core.analytics.result import Computed, UnavailableReason
from wattwise_core.analytics.wbal import WBalResult, wbal

pytestmark = pytest.mark.golden

# Declared tolerance for this closed-form recurrence (ANL-R31): abs 1e-9·max(1,|x|).
# Values are ~2e4, so the effective abs tolerance is ~2e-5.
_TOL = DEFAULT_CLOSED_FORM_ABS_TOL


def _abs_tol(expected: float) -> float:
    return DEFAULT_CLOSED_FORM_ABS_TOL * max(1.0, abs(expected))


def test_wbal_golden_skiba_differential_t6() -> None:
    """WBAL-T6: per-second Skiba differential golden (expenditure + recovery)."""
    cp = 250.0
    w_prime = 20000.0
    power = np.array([300.0, 400.0, 100.0, 0.0], dtype=np.float64)

    result = wbal(power, cp, w_prime)
    assert isinstance(result, Computed)
    val = result.value
    assert isinstance(val, WBalResult)

    expected = [
        19950.0,
        19800.0,
        19800.456278006666,
        19801.008543236825,
    ]
    assert val.series.shape == (4,)
    for got, exp in zip(val.series.tolist(), expected, strict=True):
        assert math.isclose(got, exp, abs_tol=_abs_tol(exp), rel_tol=0.0)

    # Minimum is the deepest depletion across the series.
    assert math.isclose(val.w_prime_balance_min, 19800.0, abs_tol=_abs_tol(19800.0))

    # Provenance/quality sanity (ANL-R5/R33).
    assert result.provenance.sport == "cycling"
    assert result.provenance.channels == ("power",)
    assert result.provenance.reference_params["cp_w"] == cp
    assert result.provenance.reference_params["w_prime_j"] == w_prime
    assert result.quality.extra["model"] == "skiba_2012_differential"
    assert result.quality.extra["floor_policy"] == "raw"


def test_wbal_golden_tau_w_uses_published_constants() -> None:
    """τ_W drives a NON-zero recovery delta, computed with the published Skiba constants.

    The recovery delta is ``(W' - W'bal(t-1))·(1 - e^(-1/τ_W))``; from a FULL tank that
    delta is identically zero, so τ_W (and the SKIBA_TAU constants) would be dead. We
    first DEPLETE the tank with one expenditure second, then recover, so the delta is
    non-zero and τ_W is load-bearing.

    Hand derivation (CP=250, W'=1000, P=[700, 100]):
      t0: P=700 ≥ CP → 1000 - (700-250)                    = 550.0
      t1: P=100 < CP → d_cp=150, τ = 546·e^(-0.01·150)+316 = 437.8290674410427,
          550 + (1000-550)·(1 - e^(-1/τ))                  = 551.0266255149958
    The τ oracle uses the literal published Skiba values (546 / -0.01 / 316), NOT the
    imported SKIBA_TAU_* symbols, so a wrong tau constant breaks production while this
    oracle stays correct and the assertion fails (delta is ~1.03 J ≫ the ~5.5e-7 tol).
    """
    cp = 250.0
    w_prime = 1000.0
    power = np.array([700.0, 100.0], dtype=np.float64)
    result = wbal(power, cp, w_prime)
    assert isinstance(result, Computed)

    # Independent oracle from the PUBLISHED Skiba constants (literals, not the imports).
    prev = w_prime - (700.0 - cp)  # depleted to 550.0 by the expenditure second
    tau = 546.0 * math.exp(-0.01 * (cp - 100.0)) + 316.0
    expected = prev + (w_prime - prev) * (1.0 - math.exp(-1.0 / tau))
    assert expected < w_prime  # recovery stays below W', so the W' clamp is inactive
    assert math.isclose(result.value.series[1], expected, abs_tol=_abs_tol(expected))

    # The published constants the production code imports must equal the Skiba literals.
    assert (SKIBA_TAU_A, SKIBA_TAU_B, SKIBA_TAU_C) == (546.0, -0.01, 316.0)


def test_wbal_golden_boundary_p_equals_cp_is_expenditure() -> None:
    """WBAL-R5: P == CP is the EXPENDITURE branch with (P-CP)·Δt = 0 ⇒ unchanged.

    Hand derivation (CP=250, W'=1000, P=[600,600,250,250]):
      t0: 1000-(600-250)=650 ; t1: 650-350=300 ;
      t2: P==CP expenditure → 300-(250-250)=300 (UNCHANGED, proves boundary) ;
      t3: P==CP → 300 unchanged.
    If P==CP were (wrongly) recovery, t2/t3 would rise toward W' instead.
    """
    cp = 250.0
    w_prime = 1000.0
    power = np.array([600.0, 600.0, 250.0, 250.0], dtype=np.float64)
    result = wbal(power, cp, w_prime)
    assert isinstance(result, Computed)
    expected = [650.0, 300.0, 300.0, 300.0]
    for got, exp in zip(result.value.series.tolist(), expected, strict=True):
        assert math.isclose(got, exp, abs_tol=_abs_tol(exp))


def test_wbal_golden_negative_allowed_and_floor() -> None:
    """WBAL-R2/R5: raw goes negative on over-exhaustion; floor=True ⇒ max(0, raw).

    Hand derivation (CP=250, W'=1000, P=[700,700,700], all expenditure 450/s):
      550, 100, -350 raw → floored 550, 100, 0.
    """
    cp = 250.0
    w_prime = 1000.0
    power = np.array([700.0, 700.0, 700.0], dtype=np.float64)

    raw = wbal(power, cp, w_prime, floor=False)
    assert isinstance(raw, Computed)
    raw_expected = [550.0, 100.0, -350.0]
    for got, exp in zip(raw.value.series.tolist(), raw_expected, strict=True):
        assert math.isclose(got, exp, abs_tol=_abs_tol(exp))
    assert math.isclose(raw.value.w_prime_balance_min, -350.0, abs_tol=_abs_tol(350.0))
    assert raw.quality.extra["floor_policy"] == "raw"

    floored = wbal(power, cp, w_prime, floor=True)
    assert isinstance(floored, Computed)
    floored_expected = [550.0, 100.0, 0.0]
    for got, exp in zip(floored.value.series.tolist(), floored_expected, strict=True):
        assert math.isclose(got, exp, abs_tol=_abs_tol(exp))
    assert math.isclose(floored.value.w_prime_balance_min, 0.0, abs_tol=_TOL)
    assert floored.quality.extra["floor_policy"] == "max_0"


def test_wbal_golden_seeded_at_w_prime() -> None:
    """Seed W'bal(0)=W': a single sub-CP second from a full tank stays at W'."""
    cp = 250.0
    w_prime = 20000.0
    # P below CP from a full tank: recovery toward W' cannot exceed W'.
    result = wbal(np.array([200.0], dtype=np.float64), cp, w_prime)
    assert isinstance(result, Computed)
    assert math.isclose(result.value.series[0], w_prime, abs_tol=_abs_tol(w_prime))


def test_wbal_missing_inputs_fail_closed() -> None:
    """WBAL-R4: missing power / CP / W' ⇒ MISSING_REQUIRED_INPUT."""
    cp = 250.0
    w_prime = 20000.0
    good = np.array([300.0, 100.0], dtype=np.float64)

    empty = wbal(np.array([], dtype=np.float64), cp, w_prime)
    assert getattr(empty, "reason", None) == UnavailableReason.MISSING_REQUIRED_INPUT

    all_gap = wbal(np.array([np.nan, np.nan], dtype=np.float64), cp, w_prime)
    assert getattr(all_gap, "reason", None) == UnavailableReason.MISSING_REQUIRED_INPUT

    no_cp = wbal(good, None, w_prime)
    assert getattr(no_cp, "reason", None) == UnavailableReason.MISSING_REQUIRED_INPUT

    no_wprime = wbal(good, cp, None)
    assert getattr(no_wprime, "reason", None) == UnavailableReason.MISSING_REQUIRED_INPUT


def test_wbal_non_finite_params_out_of_domain() -> None:
    """WBAL-R6/ANL-R32: non-finite or non-positive CP/W' ⇒ OUT_OF_DOMAIN."""
    good = np.array([300.0, 100.0], dtype=np.float64)
    nan_cp = wbal(good, float("nan"), 20000.0)
    assert getattr(nan_cp, "reason", None) == UnavailableReason.OUT_OF_DOMAIN

    inf_w = wbal(good, 250.0, float("inf"))
    assert getattr(inf_w, "reason", None) == UnavailableReason.OUT_OF_DOMAIN

    zero_cp = wbal(good, 0.0, 20000.0)
    assert getattr(zero_cp, "reason", None) == UnavailableReason.OUT_OF_DOMAIN

    neg_w = wbal(good, 250.0, -1.0)
    assert getattr(neg_w, "reason", None) == UnavailableReason.OUT_OF_DOMAIN
