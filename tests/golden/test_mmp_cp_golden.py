"""Golden-reference tests for MMP / CP-W' / best-efforts (doc 40, TEST-R1/R4).

Each golden carries an explicit fixture-origin / derivation note (TEST-R4): the
expected values are computed *independently* of the module under test -- either by
hand or via an independent closed-form derivation written out in the docstring -- so
the test pins the metric to an externally-verifiable truth, not to its own output.

Covered metrics:
  * MMP sliding-window maximum (MMP-R1) + non-increasing-in-duration (MMP-R3).
  * best_effort == MMP exactly (BEST-R1/R4).
  * CP / W' linear work-time regression (CP-R1/R3): oracle recovery (R2 -> 1) and a
    non-perfect noisy case with hand-derived slope/intercept/R2/SE.
  * Long-duration-bias flag (CP-R6): >1200 s trips, the 1200 s endpoint does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from wattwise_core.analytics.mmp_cp import best_effort, cp_wprime, mmp
from wattwise_core.analytics.result import Computed, UnavailableReason, is_computed

GOLDEN_ABS_TOL = 1e-6  # ANL-R31 -- regression tolerance, looser than closed-form 1e-9


@pytest.mark.golden
def test_mmp_sliding_window_max_golden() -> None:
    """MMP(d) = max over offsets of the mean of any contiguous valid d-s window.

    Fixture origin: hand-enumerated. Power = [100, 200, 300, 400, 500] W at 1 Hz.
      MMP(1) = max single sample                 = 500  (window [4,4])
      MMP(2) = max mean of 2 adjacent            = mean(400,500) = 450 (window [3,4])
      MMP(3) = mean(300,400,500)                 = 400  (window [2,4])
      MMP(5) = mean(100,200,300,400,500)         = 300  (window [0,4])
    """
    power = np.array([100, 200, 300, 400, 500], dtype=np.float64)
    res = mmp(power, (1, 2, 3, 5))

    expected = {1: (500.0, 4, 4), 2: (450.0, 3, 4), 3: (400.0, 2, 4), 5: (300.0, 0, 4)}
    for d, (val, start, end) in expected.items():
        r = res[d]
        assert is_computed(r), f"MMP({d}) should be Computed"
        assert isinstance(r, Computed)
        assert r.value.mean_power_w == pytest.approx(val, abs=GOLDEN_ABS_TOL)
        assert (r.value.start_index_s, r.value.end_index_s) == (start, end)
        assert r.value.duration_s == d


@pytest.mark.golden
def test_mmp_non_increasing_golden() -> None:
    """MMP is non-increasing in duration (MMP-R3): MMP(1) >= MMP(2) >= ... (no clamp).

    Fixture origin: derived from the same hand-enumerated curve above.
    """
    power = np.array([100, 200, 300, 400, 500], dtype=np.float64)
    grid = (1, 2, 3, 5)
    res = mmp(power, grid)
    vals = [res[d].value.mean_power_w for d in grid]  # type: ignore[union-attr]
    assert vals == sorted(vals, reverse=True)
    assert vals == [500.0, 450.0, 400.0, 300.0]


@pytest.mark.golden
def test_mmp_gap_window_golden() -> None:
    """A valid d-s window cannot straddle a NaN gap (MMP-R1).

    Fixture origin: hand-enumerated. Power = [100, 200, NaN, 400, 500, 600].
    For d=3 the only contiguous 3-s gap-free windows are [3,5] (400,500,600 -> 500);
    windows [0,2] and [1,3] and [2,4] each touch the index-2 gap and are invalid.
    """
    power = np.array([100, 200, np.nan, 400, 500, 600], dtype=np.float64)
    res = mmp(power, (3,))
    r = res[3]
    assert is_computed(r)
    assert isinstance(r, Computed)
    assert r.value.mean_power_w == pytest.approx(500.0, abs=GOLDEN_ABS_TOL)
    assert (r.value.start_index_s, r.value.end_index_s) == (3, 5)


@pytest.mark.golden
def test_best_effort_equals_mmp_golden() -> None:
    """best_effort(d) is derived from MMP and equals MMP(d) exactly (BEST-R1/R4)."""
    power = np.array([100, 200, 300, 400, 500], dtype=np.float64)
    full = mmp(power, (2,))[2]
    be = best_effort(power, 2)
    assert is_computed(full) and is_computed(be)
    assert isinstance(full, Computed) and isinstance(be, Computed)
    assert be.value.mean_power_w == full.value.mean_power_w
    assert (be.value.start_index_s, be.value.end_index_s) == (
        full.value.start_index_s,
        full.value.end_index_s,
    )


@pytest.mark.golden
def test_cp_oracle_recovery_golden() -> None:
    """Synthetic (CP0, W'0) points recover slope/intercept exactly, R2 -> 1 (CP-R5).

    Fixture origin: independent closed-form derivation.
    Construct W(t) = W'0 + CP0*t exactly with CP0 = 250 W, W'0 = 20000 J at
    durations {120, 300, 600, 1200} s, then P(t) = W(t)/t. Since the points lie on
    the work-time line by construction, OLS recovers CP = 250, W' = 20000, R2 = 1,
    and all residuals are 0 (verified by hand: Stt-based slope = 250.0).
    """
    cp0, w0 = 250.0, 20000.0
    durations = [120, 300, 600, 1200]
    points = {t: (w0 + cp0 * t) / t for t in durations}  # P(t) = W(t)/t

    fit = cp_wprime(points)
    assert is_computed(fit)
    assert isinstance(fit, Computed)
    assert fit.value.cp_w == pytest.approx(250.0, abs=1e-6)
    assert fit.value.w_prime_j == pytest.approx(20000.0, abs=1e-3)
    assert fit.value.r2 == pytest.approx(1.0, abs=1e-9)
    assert max(abs(r) for r in fit.value.residuals) == pytest.approx(0.0, abs=1e-6)
    assert fit.quality.extra["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit.quality.extra["long_duration_bias"] is False  # 1200 endpoint, no trip


@pytest.mark.golden
def test_cp_noisy_regression_golden() -> None:
    """Non-perfect fit recovers a hand-derived slope/intercept/R2/SE (CP-R3).

    Fixture origin: independent statistics derivation (not the module). Measured
    powers (W) at durations {120, 240, 600, 1200} s:
        P(120)=420, P(240)=360, P(600)=300, P(1200)=270  ->  W = P*t:
        W = {120:50400, 240:86400, 600:180000, 1200:324000}
    OLS on (t, W) by the closed-form normal equations gives:
        CP   = 251.6326530612245 W
        W'   = 24318.367346938787 J
        R2   = 0.99894736219611
        SE(CP)= 5.77590681554485 ,  SE(W')= 3951.328205248115
    (All four numbers reproduced by hand via Stt/Stw and s2 = SSR/(n-2).)
    """
    measured = {120: 420.0, 240: 360.0, 600: 300.0, 1200: 270.0}
    fit = cp_wprime(measured)
    assert is_computed(fit)
    assert isinstance(fit, Computed)
    assert fit.value.cp_w == pytest.approx(251.6326530612245, abs=GOLDEN_ABS_TOL)
    assert fit.value.w_prime_j == pytest.approx(24318.367346938787, abs=1e-3)
    assert fit.value.r2 == pytest.approx(0.99894736219611, abs=1e-9)
    assert fit.value.se_cp == pytest.approx(5.77590681554485, abs=1e-6)
    assert fit.value.se_wprime == pytest.approx(3951.328205248115, abs=1e-3)


@pytest.mark.golden
def test_cp_long_duration_bias_flag_golden() -> None:
    """Any contributing duration STRICTLY > 1200 s trips the bias flag (CP-R6).

    Fixture origin: oracle line CP0=250, W'0=20000 extended with a 1800 s point and
    the domain widened to admit it. The 1800 s point (> 1200 s) must set the flag and
    list itself as offending; the fit itself is still returned (non-blocking).
    """
    cp0, w0 = 250.0, 20000.0
    durations = [120, 300, 600, 1200, 1800]
    points = {t: (w0 + cp0 * t) / t for t in durations}

    fit = cp_wprime(points, domain_max_s=3600)
    assert is_computed(fit)
    assert isinstance(fit, Computed)
    assert fit.quality.extra["long_duration_bias"] is True
    detail = fit.quality.extra["long_duration_bias_detail"]
    assert isinstance(detail, dict)
    assert detail["offending_durations_s"] == [1800]
    assert detail["threshold_s"] == 1200
    assert fit.quality.confidence < 1.0  # downgraded but still Computed


@pytest.mark.golden
def test_cp_1200_endpoint_does_not_trip_bias_golden() -> None:
    """The 1200 s domain endpoint does NOT trip the long-duration-bias flag (CP-R6)."""
    cp0, w0 = 250.0, 20000.0
    points = {t: (w0 + cp0 * t) / t for t in (120, 300, 600, 1200)}
    fit = cp_wprime(points)
    assert is_computed(fit)
    assert isinstance(fit, Computed)
    assert fit.quality.extra["long_duration_bias"] is False
    assert "long_duration_bias_detail" not in fit.quality.extra
    assert fit.quality.confidence == 1.0


@pytest.mark.golden
def test_cp_too_few_points_unavailable_golden() -> None:
    """Fewer than CP_MIN_POINTS distinct in-domain durations -> INSUFFICIENT_DATA."""
    fit = cp_wprime({120: 400.0, 360: 320.0})
    assert not is_computed(fit)
    assert fit.reason == UnavailableReason.INSUFFICIENT_DATA  # type: ignore[union-attr]


@pytest.mark.golden
def test_cp_clustered_points_unavailable_golden() -> None:
    """max/min duration ratio < 3 -> clustered -> INSUFFICIENT_DATA (CP-R3/R4)."""
    fit = cp_wprime({400: 300.0, 500: 290.0, 600: 285.0})
    assert not is_computed(fit)
    assert fit.reason == UnavailableReason.INSUFFICIENT_DATA  # type: ignore[union-attr]
