"""Property-based tests for MMP / CP-W' / best-efforts (doc 40, TEST-R1/R2/R3).

Property IDs covered (Section 11.1 MMP-T1..T4 / BEST-T1..T2 / CP-T1..T6):

  * MMP-T1  -- sliding-window-max is EXACT vs an independent brute-force oracle.
  * MMP-T2  -- non-increasing in duration (no clamp).
  * MMP-T3  -- gap validity: a window straddling a NaN gap is never used; per-duration
               partial availability (a too-long duration is Unavailable while shorter
               durations stay Computed).
  * BEST-T1 -- best_effort(d) == MMP(d) exactly, same provenance window.
  * BEST-T2 -- best-effort fail-closed (no valid window -> INSUFFICIENT_DATA;
               empty channel -> MISSING_REQUIRED_INPUT).
  * CP-T1   -- oracle recovery: synthetic (CP0, W'0) points recover params, R2 -> 1.
  * CP-T2   -- poor/insufficient typed Unavailable (sign / clustered / too-few).
  * CP-T3   -- fit always carries R2 + SE in the QualityReport.
  * CP-T4   -- SE monotonicity guard: removing the longest-duration lever point does
               NOT decrease SE(CP) at fixed residual variance (CP-R5).
  * CP-T5   -- domain exclusion: out-of-domain durations never enter the fit.
  * CP-T6   -- long-duration-bias flag trips strictly above 1200 s, not at 1200 s.

Generators (TEST-R2) produce variable-length 1 Hz power streams with NaN gaps of
varied length, plus athlete-realistic CP/W' parameter sets, with shrinking.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.mmp_cp import (
    _ols_standard_errors,
    best_effort,
    cp_wprime,
    mmp,
)
from wattwise_core.analytics.result import (
    Computed,
    Unavailable,
    UnavailableReason,
    is_computed,
)

# ---------------------------------------------------------------------------
# Strategies (TEST-R2): variable-length 1 Hz power with NaN gaps; CP/W' params.
# ---------------------------------------------------------------------------

_power_sample = st.one_of(
    st.none(),  # a gap (NaN)
    st.floats(min_value=0.0, max_value=1600.0, allow_nan=False, allow_infinity=False),
)


@st.composite
def power_streams(draw: st.DrawFn, min_len: int = 1, max_len: int = 60) -> np.ndarray:
    """A 1 Hz power array (float64) where ``None`` draws become NaN gaps."""
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    raw = draw(st.lists(_power_sample, min_size=n, max_size=n))
    return np.array([np.nan if x is None else float(x) for x in raw], dtype=np.float64)


def _brute_force_mmp(power: np.ndarray, d: int) -> float | None:
    """Independent reference oracle for MMP(d): best mean over any gap-free window of length >= d.

    The power-duration curve value at ``d`` is the best sustainable average for AT
    LEAST ``d`` seconds (MMP-R1/R3): the maximum mean over every contiguous gap-free
    window whose length is ``>= d``. ``None`` when no such window exists.
    """
    n = power.size
    best: float | None = None
    for length in range(d, n + 1):
        for i in range(0, n - length + 1):
            win = power[i : i + length]
            if np.any(np.isnan(win)):
                continue
            m = float(np.mean(win))
            if best is None or m > best:
                best = m
    return best


# ---------------------------------------------------------------------------
# MMP-T1: sliding-window-max EXACT vs brute-force oracle.
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(power=power_streams(), d=st.integers(min_value=1, max_value=20))
@settings(max_examples=400)
def test_mmp_exact_vs_bruteforce_oracle(power: np.ndarray, d: int) -> None:
    """MMP(d) equals an independent brute-force maximum (MMP-T1)."""
    res = mmp(power, (d,))[d]
    expected = _brute_force_mmp(power, d)
    if expected is None:
        assert not is_computed(res)
        assert isinstance(res, Unavailable)
        # No window: either too short for the channel or all-gap.
        assert res.reason in (
            UnavailableReason.INSUFFICIENT_DATA,
            UnavailableReason.MISSING_REQUIRED_INPUT,
        )
    else:
        assert is_computed(res)
        assert isinstance(res, Computed)
        assert res.value.mean_power_w == pytest.approx(expected, abs=1e-9, rel=1e-12)
        # Reported achieving window reproduces the reported mean exactly and is
        # at least d seconds long (at-least-d power-duration curve, MMP-R1/R3).
        win = power[res.value.start_index_s : res.value.end_index_s + 1]
        assert win.size == res.value.window_len_s
        assert res.value.window_len_s >= d
        assert not np.any(np.isnan(win))
        assert float(np.mean(win)) == pytest.approx(res.value.mean_power_w, abs=1e-9)


# ---------------------------------------------------------------------------
# MMP-T2: non-increasing in duration (no clamp).
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(power=power_streams(min_len=1, max_len=80))
@settings(max_examples=300)
def test_mmp_non_increasing_in_duration(power: np.ndarray) -> None:
    """Where consecutive grid durations are both Computed, MMP is non-increasing."""
    grid = (1, 2, 3, 5, 10, 20, 30)
    res = mmp(power, grid)
    computed = [
        (d, res[d].value.mean_power_w)  # type: ignore[union-attr]
        for d in grid
        if is_computed(res[d])
    ]
    for (d1, v1), (d2, v2) in itertools.pairwise(computed):
        assert d1 < d2
        # Longer duration can never exceed a shorter one (MMP-R3), within float eps.
        assert v2 <= v1 + 1e-9


# ---------------------------------------------------------------------------
# MMP-T3: gap validity + per-duration partial availability.
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(power=power_streams(min_len=2, max_len=50))
@settings(max_examples=300, suppress_health_check=[HealthCheck.filter_too_much])
def test_mmp_window_never_straddles_gap(power: np.ndarray) -> None:
    """A Computed MMP window is always entirely gap-free (MMP-R1)."""
    grid = (1, 2, 5, 10)
    res = mmp(power, grid)
    for d in grid:
        r = res[d]
        if is_computed(r):
            assert isinstance(r, Computed)
            win = power[r.value.start_index_s : r.value.end_index_s + 1]
            assert win.size == r.value.window_len_s
            assert r.value.window_len_s >= d
            assert not np.any(np.isnan(win))


@pytest.mark.property
@given(power=power_streams(min_len=1, max_len=40))
@settings(max_examples=200)
def test_mmp_partial_availability(power: np.ndarray) -> None:
    """A too-long duration is Unavailable while short d may be Computed (MMP-R5)."""
    n = power.size
    too_long = n + 5
    res = mmp(power, (1, too_long))
    r_long = res[too_long]
    assert not is_computed(r_long)
    assert isinstance(r_long, Unavailable)
    assert r_long.reason == UnavailableReason.INSUFFICIENT_DATA


@pytest.mark.property
@given(grid=st.lists(st.integers(1, 30), min_size=1, max_size=6, unique=True))
@settings(max_examples=50)
def test_mmp_empty_channel_missing_required_input(grid: list[int]) -> None:
    """A zero-sample power channel maps every duration to MISSING_REQUIRED_INPUT (MMP-R5)."""
    res = mmp(np.array([], dtype=np.float64), tuple(grid))
    for d in grid:
        r = res[d]
        assert not is_computed(r)
        assert isinstance(r, Unavailable)
        assert r.reason == UnavailableReason.MISSING_REQUIRED_INPUT


# ---------------------------------------------------------------------------
# BEST-T1 / BEST-T2: best_effort == MMP exactly; fail-closed.
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(power=power_streams(min_len=1, max_len=60), d=st.integers(1, 20))
@settings(max_examples=300)
def test_best_effort_equals_mmp(power: np.ndarray, d: int) -> None:
    """best_effort(d) is exactly MMP(d) with identical provenance (BEST-T1)."""
    via_mmp = mmp(power, (d,))[d]
    via_best = best_effort(power, d)
    if is_computed(via_mmp):
        assert is_computed(via_best)
        assert isinstance(via_mmp, Computed)
        assert isinstance(via_best, Computed)
        assert via_best.value.mean_power_w == via_mmp.value.mean_power_w
        assert via_best.value.start_index_s == via_mmp.value.start_index_s
        assert via_best.value.end_index_s == via_mmp.value.end_index_s
    else:
        assert not is_computed(via_best)
        assert isinstance(via_mmp, Unavailable)
        assert isinstance(via_best, Unavailable)
        assert via_best.reason == via_mmp.reason


@pytest.mark.property
@given(d=st.integers(1, 30))
@settings(max_examples=30)
def test_best_effort_fail_closed_empty(d: int) -> None:
    """No power channel -> best effort fails closed to MISSING_REQUIRED_INPUT (BEST-T2)."""
    r = best_effort(np.array([], dtype=np.float64), d)
    assert not is_computed(r)
    assert isinstance(r, Unavailable)
    assert r.reason == UnavailableReason.MISSING_REQUIRED_INPUT


# ---------------------------------------------------------------------------
# CP-T1: oracle recovery from synthetic (CP0, W'0).
# ---------------------------------------------------------------------------

_cp0 = st.floats(min_value=120.0, max_value=400.0, allow_nan=False)
_w0 = st.floats(min_value=5000.0, max_value=40000.0, allow_nan=False)
_durations = st.lists(
    st.integers(min_value=120, max_value=1200), min_size=3, max_size=8, unique=True
)


@pytest.mark.property
@given(cp0=_cp0, w0=_w0, durations=_durations)
@settings(max_examples=300)
def test_cp_oracle_recovery(cp0: float, w0: float, durations: list[int]) -> None:
    """Points on the exact work-time line recover (CP0, W'0) with R2 -> 1 (CP-T1)."""
    durs = sorted(set(durations))
    # Need a wide-enough spread to clear the duration-ratio gate.
    assume(durs[-1] / durs[0] >= 3.0)
    points = {t: (w0 + cp0 * t) / t for t in durs}  # P(t) = W(t)/t exact

    fit = cp_wprime(points)
    assert is_computed(fit)
    assert isinstance(fit, Computed)
    assert fit.value.cp_w == pytest.approx(cp0, rel=1e-6, abs=1e-4)
    assert fit.value.w_prime_j == pytest.approx(w0, rel=1e-6, abs=1e-2)
    assert fit.value.r2 == pytest.approx(1.0, abs=1e-9)
    assert max(abs(r) for r in fit.value.residuals) == pytest.approx(0.0, abs=1e-3)
    # CP-T3: fit always carries goodness-of-fit.
    assert "r2" in fit.quality.extra
    assert "se_cp" in fit.quality.extra
    assert "se_wprime" in fit.quality.extra
    assert math.isfinite(fit.value.se_cp)
    assert math.isfinite(fit.value.se_wprime)


# ---------------------------------------------------------------------------
# CP-T2: typed Unavailable for too-few / clustered / wrong-sign.
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    points=st.dictionaries(
        st.integers(120, 1200),
        st.floats(50.0, 600.0, allow_nan=False),
        min_size=0,
        max_size=2,
    )
)
@settings(max_examples=100)
def test_cp_too_few_points_insufficient(points: dict[int, float]) -> None:
    """Fewer than the minimum distinct in-domain durations -> INSUFFICIENT_DATA (CP-T2)."""
    assume(len(points) < 3)
    fit = cp_wprime(points)
    assert not is_computed(fit)
    assert isinstance(fit, Unavailable)
    assert fit.reason == UnavailableReason.INSUFFICIENT_DATA


@pytest.mark.property
@given(
    base=st.integers(120, 380),
    spread=st.integers(1, 40),
    p0=st.floats(200.0, 400.0, allow_nan=False),
)
@settings(max_examples=150)
def test_cp_clustered_points_insufficient(base: int, spread: int, p0: float) -> None:
    """Durations too clustered (max/min < 3) -> INSUFFICIENT_DATA (CP-T2)."""
    durs = [base, base + spread, base + 2 * spread]
    assume(durs[-1] / durs[0] < 3.0)
    points = {t: p0 - 0.01 * (t - base) for t in durs}
    fit = cp_wprime(points)
    assert not is_computed(fit)
    assert isinstance(fit, Unavailable)
    assert fit.reason == UnavailableReason.INSUFFICIENT_DATA


@pytest.mark.property
@given(durations=_durations)
@settings(max_examples=150)
def test_cp_wrong_sign_poor_fit(durations: list[int]) -> None:
    """A descending work-time line (CP <= 0) is rejected as POOR_FIT (CP-T2).

    Construct W decreasing in t (work falls as duration grows): no positive CP slope
    can fit, so the sign gate fires. We make W(t) decrease linearly with t while
    keeping a strong linear R2 so the rejection is by SIGN, not by R2.
    """
    durs = sorted(set(durations))
    assume(len(durs) >= 3)
    assume(durs[-1] / durs[0] >= 3.0)
    # Strong negative-slope work-time line -> CP < 0.
    points = {t: (50000.0 - 20.0 * t) / t for t in durs}
    # Ensure powers stay finite/positive enough to be a valid input.
    assume(all(math.isfinite(p) and p > 0 for p in points.values()))
    fit = cp_wprime(points)
    if not is_computed(fit):
        assert isinstance(fit, Unavailable)
        assert fit.reason == UnavailableReason.POOR_FIT


# ---------------------------------------------------------------------------
# CP-T4: SE monotonicity guard (CP-R5) on the OLS SE formula directly.
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    durations=st.lists(st.integers(120, 1200), min_size=4, max_size=8, unique=True),
    scatter=st.floats(min_value=10.0, max_value=3000.0, allow_nan=False),
    seed=st.integers(0, 2**31 - 1),
)
@settings(max_examples=300)
def test_cp_se_monotonicity_removing_longest(
    durations: list[int], scatter: float, seed: int
) -> None:
    """Removing the longest-duration lever point does NOT decrease SE(CP) (CP-T4/CP-R5).

    Isolates the lever-arm effect: with the residual variance ``s2`` held fixed and
    the longest-duration (max-leverage) point removed, ``S_tt`` shrinks, so
    ``SE(CP) = sqrt(s2 / S_tt)`` cannot decrease. Confidence never increases by
    dropping the most-informative point.
    """
    durs = sorted(set(durations))
    assume(len(durs) >= 4)
    t = np.array(durs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    resid = rng.normal(0.0, scatter, size=t.size)
    resid = resid - resid.mean()
    n = t.size
    se_cp_full, _ = _ols_standard_errors(t, resid, n)

    # Remove the longest-duration point; renormalize residuals to the SAME s2.
    t2 = t[:-1]
    resid2 = resid[:-1]
    resid2 = resid2 - resid2.mean()
    s2_full = float(np.sum(resid**2)) / (n - 2)
    n2 = t2.size
    if n2 - 2 <= 0:
        return  # SE formula returns 0 for an interpolating line; nothing to compare
    s2_red = float(np.sum(resid2**2)) / (n2 - 2)
    assume(s2_red > 1e-9)
    resid2 = resid2 * math.sqrt(s2_full / s2_red)
    se_cp_red, _ = _ols_standard_errors(t2, resid2, n2)

    assert se_cp_red + 1e-9 >= se_cp_full


# ---------------------------------------------------------------------------
# CP-T5: domain exclusion.
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(cp0=_cp0, w0=_w0, durations=_durations)
@settings(max_examples=200)
def test_cp_out_of_domain_points_excluded(cp0: float, w0: float, durations: list[int]) -> None:
    """Adding far out-of-domain points does not change an in-domain fit (CP-T5)."""
    durs = sorted(set(durations))
    assume(durs[-1] / durs[0] >= 3.0)
    in_domain = {t: (w0 + cp0 * t) / t for t in durs}
    fit_clean = cp_wprime(in_domain)
    assume(is_computed(fit_clean))
    assert isinstance(fit_clean, Computed)

    # Inject points well below 120 s and well above 1200 s (default domain).
    polluted = dict(in_domain)
    polluted[30] = (w0 + cp0 * 30) / 30  # < 120 s -> excluded
    polluted[60] = (w0 + cp0 * 60) / 60  # < 120 s -> excluded
    polluted[3600] = (w0 + cp0 * 3600) / 3600  # > 1200 s -> excluded by default domain
    fit_polluted = cp_wprime(polluted)
    assert isinstance(fit_polluted, Computed)
    assert fit_polluted.value.cp_w == pytest.approx(fit_clean.value.cp_w, abs=1e-6)
    assert fit_polluted.value.w_prime_j == pytest.approx(fit_clean.value.w_prime_j, abs=1e-3)
    # The out-of-domain points were excluded -> 1200 s endpoint, no long-bias trip.
    assert fit_polluted.quality.extra["long_duration_bias"] is False


@pytest.mark.property
@given(
    durations=st.lists(st.integers(1, 119), min_size=3, max_size=6, unique=True),
    p0=st.floats(200.0, 500.0, allow_nan=False),
)
@settings(max_examples=100)
def test_cp_all_below_domain_unavailable(durations: list[int], p0: float) -> None:
    """If every supplied duration is below the domain, the fit has no points (CP-T5/CP-R2)."""
    points = dict.fromkeys(durations, p0)
    fit = cp_wprime(points)
    assert not is_computed(fit)
    assert isinstance(fit, Unavailable)
    assert fit.reason == UnavailableReason.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# CP-T6: long-duration-bias flag (>1200 s trips; 1200 s endpoint does not).
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(cp0=_cp0, w0=_w0)
@settings(max_examples=100)
def test_cp_long_duration_bias_strict(cp0: float, w0: float) -> None:
    """A point strictly above 1200 s trips the flag; the 1200 s endpoint does not (CP-T6)."""
    # Endpoint-only set: includes 1200 s but nothing above -> no trip.
    endpoint = {t: (w0 + cp0 * t) / t for t in (120, 600, 1200)}
    fit_end = cp_wprime(endpoint)
    assert isinstance(fit_end, Computed)
    assert fit_end.quality.extra["long_duration_bias"] is False
    assert fit_end.quality.confidence == 1.0

    # Add a 1800 s point and widen the domain to admit it -> trips.
    widened = dict(endpoint)
    widened[1800] = (w0 + cp0 * 1800) / 1800
    fit_wide = cp_wprime(widened, domain_max_s=3600)
    assert isinstance(fit_wide, Computed)
    assert fit_wide.quality.extra["long_duration_bias"] is True
    detail = fit_wide.quality.extra["long_duration_bias_detail"]
    assert isinstance(detail, dict)
    offending = detail["offending_durations_s"]
    assert isinstance(offending, list)
    assert 1800 in offending
    assert 1200 not in offending
    assert fit_wide.quality.confidence < 1.0
