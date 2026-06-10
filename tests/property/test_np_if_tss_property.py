"""Property-based tests for the cycling-power load family (doc 40 §11.1).

Covers the per-metric property IDs:

- **NP-T1** — Jensen: ``NP >= mean(R(t))`` over the analysis window, for all streams.
- **NP-T2** — constant power ``c`` ⇒ ``NP == avg_power == mean(R) == c`` to 1e-6.
- **NP-T3** — seeding: ``<30`` contiguous valid seconds ⇒ ``INSUFFICIENT_DATA``.
- **NP-T4** — translation invariance: shifting every ``t_seconds`` by a constant offset
  does not change NP (NP is defined on relative time / the 1 Hz grid).
- **TSS-T2** — TSS is quadratic in IF (via NP) and linear in ``duration_valid_s``.
- **IF-T1** — IF propagates the NP Unavailable; never an average-power fallback.
- **Fail-closed (TEST-R3)** — empty stream / all-``null`` ⇒ ``MISSING_REQUIRED_INPUT``;
  sub-30-s ⇒ ``INSUFFICIENT_DATA``; missing/non-positive FTP ⇒ the correct typed reason;
  no NaN/Inf ever escapes into a ``Computed`` (ANL-R32).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.constants import NP_ROLLING_WINDOW_S
from wattwise_core.analytics.np_if_tss import (
    intensity_factor,
    normalized_power,
    power_tss,
)
from wattwise_core.analytics.result import (
    Computed,
    Unavailable,
    UnavailableReason,
    is_computed,
)
from wattwise_core.analytics.series import Stream

NP_TOL = 1e-6

CI_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# A finite, non-degenerate cycling-power sample (watts). Bounded to keep R^4 finite.
powers = st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False)


@st.composite
def power_value_lists(draw: st.DrawFn, *, min_size: int, max_size: int) -> list[float | None]:
    """A 1 Hz power list of valid samples (no gaps), length in [min_size, max_size]."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return [draw(powers) for _ in range(n)]


@st.composite
def power_value_lists_with_gaps(draw: st.DrawFn) -> list[float | None]:
    """A 1 Hz power list that MAY contain ``None`` gaps."""
    n = draw(st.integers(min_value=0, max_value=120))
    out: list[float | None] = []
    for _ in range(n):
        if draw(st.booleans()):
            out.append(None)
        else:
            out.append(draw(powers))
    return out


# --- NP-T1: Jensen guarantee -------------------------------------------------


@given(values=power_value_lists(min_size=NP_ROLLING_WINDOW_S, max_size=400))
@CI_SETTINGS
def test_np_jensen_ge_mean_r(values: list[float | None]) -> None:
    """NP-T1: NP >= mean(R(t)) over the analysis window, universally."""
    result = normalized_power(Stream.from_values(values))
    assume(is_computed(result))
    assert isinstance(result, Computed)
    # Jensen's inequality for the convex x^4: (mean(R^4))^(1/4) >= mean(R).
    assert result.value.np_w >= result.value.mean_r_w - NP_TOL


# --- NP-T2: constant-power identity ------------------------------------------


@given(
    c=st.floats(min_value=1.0, max_value=1500.0, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=NP_ROLLING_WINDOW_S, max_value=400),
)
@CI_SETTINGS
def test_np_constant_identity(c: float, n: int) -> None:
    """NP-T2: constant power c ⇒ NP == avg_power == mean(R) == c to 1e-6."""
    result = normalized_power(Stream.from_values([c] * n))
    assert isinstance(result, Computed)
    assert result.value.np_w == pytest.approx(c, abs=NP_TOL, rel=NP_TOL)
    assert result.value.avg_power_w == pytest.approx(c, abs=NP_TOL, rel=NP_TOL)
    assert result.value.mean_r_w == pytest.approx(c, abs=NP_TOL, rel=NP_TOL)


# --- NP-T4: translation invariance -------------------------------------------


@given(
    values=power_value_lists(min_size=NP_ROLLING_WINDOW_S, max_size=300),
    offset=st.integers(min_value=0, max_value=100_000),
)
@CI_SETTINGS
def test_np_translation_invariant(values: list[float | None], offset: int) -> None:
    """NP-T4: an integer-second time-translation leaves NP unchanged.

    NP is defined on relative time / the 1 Hz grid; shifting every sample by a whole
    number of seconds keeps every sample aligned to the same integer grid, so the
    resampled series (and thus NP) is identical. (A fractional shift re-aligns samples
    against the 1 Hz grid and legitimately changes the interpolated series — that is a
    resampling effect, not an NP-invariance violation, so it is not asserted here.)
    """
    base = Stream.from_values(values)
    shifted = Stream(t_seconds=base.t_seconds + float(offset), values=base.values.copy())
    r_base = normalized_power(base)
    r_shift = normalized_power(shifted)
    assert is_computed(r_base) == is_computed(r_shift)
    if isinstance(r_base, Computed) and isinstance(r_shift, Computed):
        assert r_shift.value.np_w == pytest.approx(r_base.value.np_w, abs=NP_TOL, rel=NP_TOL)


# --- NP-T3 / fail-closed: seeding & absent input -----------------------------


@given(values=power_value_lists(min_size=0, max_size=NP_ROLLING_WINDOW_S - 1))
@CI_SETTINGS
def test_np_sub_window_insufficient_or_missing(values: list[float | None]) -> None:
    """<30 contiguous valid s ⇒ INSUFFICIENT_DATA; empty ⇒ MISSING_REQUIRED_INPUT."""
    result = normalized_power(Stream.from_values(values))
    assert isinstance(result, Unavailable)
    if len(values) == 0:
        assert result.reason == UnavailableReason.MISSING_REQUIRED_INPUT
    else:
        assert result.reason == UnavailableReason.INSUFFICIENT_DATA


def test_np_empty_stream_missing_required_input() -> None:
    """Empty power stream ⇒ MISSING_REQUIRED_INPUT (absent channel, §6)."""
    result = normalized_power(Stream.from_values([]))
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.MISSING_REQUIRED_INPUT


def test_np_all_null_stream_missing_required_input() -> None:
    """All-null power stream ⇒ MISSING_REQUIRED_INPUT, never INSUFFICIENT_DATA."""
    result = normalized_power(Stream.from_values([None] * 50))
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.MISSING_REQUIRED_INPUT


def test_np_gap_breaks_contiguity_insufficient() -> None:
    """A long gap that prevents 30 contiguous valid seconds ⇒ INSUFFICIENT_DATA (NP-R3).

    The gap must exceed ``max_interp_gap_s`` (3 s) so the resampler does NOT bridge it;
    a >3 s hole leaves the longest contiguous valid run at 20 < 30 on each side.
    """
    # 20 valid, a 5-second hole (> max_interp_gap_s), 20 valid.
    values: list[float | None] = [200.0] * 20 + [None] * 5 + [200.0] * 20
    result = normalized_power(Stream.from_values(values))
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.INSUFFICIENT_DATA


@given(values=power_value_lists_with_gaps())
@CI_SETTINGS
def test_np_no_naninf_escapes(values: list[float | None]) -> None:
    """ANL-R32: a Computed NP is always finite; otherwise typed Unavailable."""
    result = normalized_power(Stream.from_values(values))
    if isinstance(result, Computed):
        assert math.isfinite(result.value.np_w)
        assert math.isfinite(result.value.avg_power_w)
        assert math.isfinite(result.value.mean_r_w)
    else:
        assert isinstance(result, Unavailable)
        assert result.reason in {
            UnavailableReason.MISSING_REQUIRED_INPUT,
            UnavailableReason.INSUFFICIENT_DATA,
            UnavailableReason.OUT_OF_DOMAIN,
        }


# --- IF-T1: propagation, no avg-power fallback -------------------------------


@given(
    values=power_value_lists(min_size=0, max_size=NP_ROLLING_WINDOW_S - 1),
    ftp=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)
@CI_SETTINGS
def test_if_propagates_np_unavailable(values: list[float | None], ftp: float) -> None:
    """IF-T1: when NP is Unavailable, IF is the SAME Unavailable (no fallback)."""
    np_result = normalized_power(Stream.from_values(values))
    assume(isinstance(np_result, Unavailable))
    if_result = intensity_factor(np_result, ftp_w=ftp)
    assert isinstance(if_result, Unavailable)
    assert isinstance(np_result, Unavailable)
    assert if_result.reason == np_result.reason


@given(
    c=st.floats(min_value=50.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    ftp=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)
@CI_SETTINGS
def test_if_equals_np_over_ftp(c: float, ftp: float) -> None:
    """IF-T1: IF == NP/FTP exactly (constant ride ⇒ NP == c)."""
    np_result = normalized_power(Stream.from_values([c] * 60))
    if_result = intensity_factor(np_result, ftp_w=ftp)
    assert isinstance(if_result, Computed)
    assert if_result.value == pytest.approx(c / ftp, abs=NP_TOL, rel=NP_TOL)


def test_if_missing_ftp_missing_required_input() -> None:
    """Absent FTP ⇒ MISSING_REQUIRED_INPUT (§6 absent reference param)."""
    np_result = normalized_power(Stream.from_values([200.0] * 60))
    if_result = intensity_factor(np_result, ftp_w=None)
    assert isinstance(if_result, Unavailable)
    assert if_result.reason == UnavailableReason.MISSING_REQUIRED_INPUT


@pytest.mark.parametrize("bad_ftp", [0.0, -1.0, float("nan"), float("inf")])
def test_if_nonpositive_or_nonfinite_ftp_out_of_domain(bad_ftp: float) -> None:
    """Non-positive / non-finite FTP ⇒ OUT_OF_DOMAIN (present-but-invalid, §6)."""
    np_result = normalized_power(Stream.from_values([200.0] * 60))
    if_result = intensity_factor(np_result, ftp_w=bad_ftp)
    assert isinstance(if_result, Unavailable)
    assert if_result.reason == UnavailableReason.OUT_OF_DOMAIN


# --- TSS-T2: quadratic in IF (NP), linear in duration ------------------------


@given(
    c=st.floats(min_value=50.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    ftp=st.floats(min_value=80.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    duration=st.integers(min_value=60, max_value=20000),
    k=st.floats(min_value=1.1, max_value=4.0, allow_nan=False, allow_infinity=False),
)
@CI_SETTINGS
def test_tss_quadratic_in_if_linear_in_duration(
    c: float, ftp: float, duration: int, k: float
) -> None:
    """TSS-T2: scaling NP by k scales TSS by k^2; scaling duration by m scales TSS by m."""
    np_result = normalized_power(Stream.from_values([c] * 60))
    assert isinstance(np_result, Computed)

    tss = power_tss(np_result, ftp_w=ftp, duration_valid_s=duration)
    assert isinstance(tss, Computed)

    # Quadratic in NP: a ride at power k*c (so NP scales by k) has k^2 the TSS.
    np_scaled = normalized_power(Stream.from_values([c * k] * 60))
    assert isinstance(np_scaled, Computed)
    tss_scaled = power_tss(np_scaled, ftp_w=ftp, duration_valid_s=duration)
    assert isinstance(tss_scaled, Computed)
    assert tss_scaled.value == pytest.approx(tss.value * k * k, rel=1e-6, abs=1e-6)

    # Linear in duration.
    tss_2x = power_tss(np_result, ftp_w=ftp, duration_valid_s=duration * 2)
    assert isinstance(tss_2x, Computed)
    assert tss_2x.value == pytest.approx(tss.value * 2.0, rel=1e-9, abs=1e-9)


def test_tss_propagates_np_unavailable() -> None:
    """TSS propagates the NP Unavailable; never fabricates a number."""
    np_result = normalized_power(Stream.from_values([]))
    tss = power_tss(np_result, ftp_w=250.0, duration_valid_s=3600)
    assert isinstance(tss, Unavailable)
    assert tss.reason == UnavailableReason.MISSING_REQUIRED_INPUT


def test_tss_nonpositive_duration_out_of_domain() -> None:
    """Non-positive duration_valid_s ⇒ OUT_OF_DOMAIN, never 0/clamp."""
    np_result = normalized_power(Stream.from_values([200.0] * 60))
    tss = power_tss(np_result, ftp_w=250.0, duration_valid_s=0)
    assert isinstance(tss, Unavailable)
    assert tss.reason == UnavailableReason.OUT_OF_DOMAIN
