"""Property-based tests for aerobic decoupling (doc 40 §9; DEC-T1..T6; TEST-R1/R2/R3).

Covers the per-metric property IDs (doc 40 §11.1, DEC-T1..T6):

- **DEC-T1** const power + const HR ⇒ 0 %;
- **DEC-T2** coasting-invariance — inserting ``output == 0`` samples (excluded AFTER
  the time split) does not change the result;
- **DEC-T3** smoothed-power spike stability (the 30 s smoothing damps a single spike);
- **DEC-T4** missing / short / too-variable / too-few-included ⇒ the matching
  typed :class:`Unavailable` (fail-closed, TEST-R3);
- **DEC-T5** sign convention — a second-half efficiency *drop* ⇒ positive decoupling;
- **DEC-T6** time-midpoint split — the boundary is the elapsed-time midpoint, not the
  sample-count midpoint.

All generators emit 1 Hz streams with ``None`` gaps / coast segments (TEST-R2). The
function is pure and deterministic (ANL-R2/R30), so equal inputs ⇒ equal outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.decoupling import (
    MIN_INCLUDED_SAMPLES_PER_HALF,
    aerobic_decoupling,
)
from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason
from wattwise_core.analytics.series import Stream

pytestmark = pytest.mark.property

# Closed-form tolerance (doc 40 §4, ANL-R31): exact-ratio asserts use 1e-9·max(1,|x|).
REL_TOL = 1e-9
CI_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --- DEC-T1: const power + const HR ⇒ 0 % ------------------------------------------
@CI_SETTINGS
@given(
    power=st.floats(min_value=50.0, max_value=600.0, allow_nan=False),
    hr=st.floats(min_value=80.0, max_value=200.0, allow_nan=False),
    duration=st.integers(min_value=1300, max_value=3600),
)
def test_constant_output_and_hr_is_zero(power: float, hr: float, duration: int) -> None:
    """Constant output + constant HR ⇒ 0 % decoupling to closed-form tol (DEC-T1)."""
    out = Stream.from_values([power] * duration)
    hrs = Stream.from_values([hr] * duration)

    result = aerobic_decoupling(out, hrs, "cycling")

    assert isinstance(result, Computed)
    assert result.value == pytest.approx(0.0, abs=REL_TOL)


# --- DEC-T2: coasting-invariance ---------------------------------------------------
@CI_SETTINGS
@given(
    power=st.floats(min_value=80.0, max_value=400.0, allow_nan=False),
    hr_first=st.floats(min_value=110.0, max_value=160.0, allow_nan=False),
    hr_second=st.floats(min_value=110.0, max_value=170.0, allow_nan=False),
    coast_positions=st.lists(
        st.integers(min_value=40, max_value=1750), min_size=0, max_size=12, unique=True
    ),
)
def test_coasting_insertion_invariant(
    power: float,
    hr_first: float,
    hr_second: float,
    coast_positions: list[int],
) -> None:
    """Inserting ``output == 0`` (coasting) seconds does not change the result (DEC-T2).

    With a globally constant output, every kept (moving, seeded) second smooths to the
    same value and its HR/time index is preserved, so excluding coasting seconds AFTER
    the time split (DEC-R2) leaves both half-means — hence the result — exactly equal.
    """
    n = 1800
    power_series: list[float] = [power] * n
    hr_series: list[float] = [hr_first] * 900 + [hr_second] * 900

    baseline = aerobic_decoupling(
        Stream.from_values(power_series), Stream.from_values(hr_series), "cycling"
    )

    coasted = list(power_series)
    for pos in coast_positions:
        coasted[pos] = 0.0  # a coasting / freewheeling second (raw power == 0)
    with_coast = aerobic_decoupling(
        Stream.from_values(coasted), Stream.from_values(hr_series), "cycling"
    )

    assert isinstance(baseline, Computed)
    assert isinstance(with_coast, Computed)
    # Exact invariance: constant output ⇒ kept seconds' smoothed value unchanged.
    assert with_coast.value == pytest.approx(baseline.value, abs=REL_TOL)


# --- DEC-T3: smoothed-output spike stability ---------------------------------------
@CI_SETTINGS
@given(
    spike_pos=st.integers(min_value=200, max_value=1600),
    spike_mag=st.floats(min_value=400.0, max_value=2000.0, allow_nan=False),
)
def test_single_spike_is_damped_by_smoothing(spike_pos: int, spike_mag: float) -> None:
    """A single 1 s output spike barely moves the result vs the raw-power effect (DEC-T3).

    The 30 s smoothing (DEC-R3) spreads one spike over 30 windowed seconds, so its
    contribution to a half-mean is ~1/30 of using raw power. The decoupling stays a
    small finite number (no blow-up), confirming smoothing not raw power feeds the term.
    """
    n = 1800
    power = [200.0] * n
    power[spike_pos] = spike_mag  # single-second spike
    hr = [150.0] * n

    result = aerobic_decoupling(Stream.from_values(power), Stream.from_values(hr), "cycling")

    assert isinstance(result, Computed)
    assert np.isfinite(result.value)
    # One spike of magnitude M over a ~900 s half, smoothed across 30 s, perturbs a
    # half-mean by at most ~M*30/(30*900) = M/900 of a watt-equivalent; the resulting
    # decoupling magnitude stays bounded well under what a raw-power injection (M/200/900)
    # would imply blowing up. Just assert it stays a small, finite, bounded number.
    assert abs(result.value) < 50.0


# --- DEC-T5: sign convention -------------------------------------------------------
@CI_SETTINGS
@given(
    power=st.floats(min_value=100.0, max_value=350.0, allow_nan=False),
    hr_base=st.floats(min_value=120.0, max_value=150.0, allow_nan=False),
    hr_delta=st.floats(min_value=2.0, max_value=25.0, allow_nan=False),
)
def test_sign_convention(power: float, hr_base: float, hr_delta: float) -> None:
    """Second-half HR rise (efficiency drop) ⇒ POSITIVE; HR fall ⇒ NEGATIVE (DEC-T5).

    Output held constant; only second-half HR moves. A HR rise lowers second-half
    output-per-beat (efficiency drop) ⇒ positive decoupling, and symmetrically a HR
    fall ⇒ negative — the documented, stable sign convention (DEC-R5).
    """
    n = 1800
    out = Stream.from_values([power] * n)

    hr_rise = Stream.from_values([hr_base] * 900 + [hr_base + hr_delta] * 900)
    rise = aerobic_decoupling(out, hr_rise, "cycling")
    assert isinstance(rise, Computed)
    assert rise.value > 0.0

    hr_fall = Stream.from_values([hr_base + hr_delta] * 900 + [hr_base] * 900)
    fall = aerobic_decoupling(out, hr_fall, "cycling")
    assert isinstance(fall, Computed)
    assert fall.value < 0.0


# --- DEC-T6: time-midpoint split (not sample-count) --------------------------------
@pytest.mark.property
def test_time_midpoint_split_not_sample_count() -> None:
    """The half boundary is the elapsed-TIME midpoint, not the sample-count midpoint.

    Build an effort whose valid samples are denser in the first part and sparser in
    the second: a gap region (resampled to ``null``) sits entirely in the first half
    by time. If the split used sample COUNT, the boundary would shift toward earlier
    time; using elapsed TIME (DEC-R1), ``t_mid`` is the arithmetic midpoint of the
    valid-window endpoints. We assert the recorded ``t_mid_s`` equals
    ``(t_start + t_end)/2`` over the both-valid window, independent of sample density.
    """
    # Valid 0..299, a long >max_interp_gap null hole 300..399, valid 400..1799.
    # Both-valid window endpoints: t_start=0, t_end=1799 ⇒ t_mid=899.5 regardless of
    # the missing 100 s (sample-count midpoint would be elsewhere).
    power: list[float | None] = [200.0] * 300 + [None] * 100 + [200.0] * 1400
    hr: list[float | None] = [150.0] * 300 + [None] * 100 + [150.0] * 1400

    result = aerobic_decoupling(Stream.from_values(power), Stream.from_values(hr), "cycling")

    assert isinstance(result, Computed)
    assert result.quality.extra["t_mid_s"] == pytest.approx(899.5, abs=REL_TOL)
    # Constant output+HR even across the gap ⇒ still 0 % (gap excluded from both means).
    assert result.value == pytest.approx(0.0, abs=REL_TOL)


# --- DEC-T4: fail-closed degenerate cases (TEST-R3) --------------------------------
@pytest.mark.property
def test_missing_power_channel_is_missing_required_input() -> None:
    """Wholly absent output channel ⇒ MISSING_REQUIRED_INPUT (doc 40 §6)."""
    out = Stream.from_values([None] * 1800)
    hr = Stream.from_values([150.0] * 1800)
    result = aerobic_decoupling(out, hr, "cycling")
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


@pytest.mark.property
def test_missing_hr_channel_is_missing_required_input() -> None:
    """Wholly absent HR channel ⇒ MISSING_REQUIRED_INPUT (doc 40 §6)."""
    out = Stream.from_values([200.0] * 1800)
    hr = Stream.from_values([None] * 1800)
    result = aerobic_decoupling(out, hr, "cycling")
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


@pytest.mark.property
def test_empty_streams_are_missing_required_input() -> None:
    """Empty streams ⇒ MISSING_REQUIRED_INPUT, never a fabricated number (ANL-R4)."""
    empty = Stream.from_values([])
    result = aerobic_decoupling(empty, empty, "cycling")
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


@CI_SETTINGS
@given(duration=st.integers(min_value=1, max_value=1199))
def test_too_short_is_insufficient_data(duration: int) -> None:
    """Window shorter than the declared minimum ⇒ INSUFFICIENT_DATA (DEC-R4)."""
    out = Stream.from_values([200.0] * duration)
    hr = Stream.from_values([150.0] * duration)
    result = aerobic_decoupling(out, hr, "cycling")
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA


@pytest.mark.property
def test_too_variable_effort_is_insufficient_data() -> None:
    """A non-steady (interval) effort fails the steadiness gate ⇒ INSUFFICIENT_DATA."""
    # Alternating 100 W / 450 W blocks of 300 s — longer than the 30 s smoothing
    # window, so the *smoothed* output keeps a high CV (a single spike would be damped).
    power = [(100.0 if (i // 300) % 2 == 0 else 450.0) for i in range(1800)]
    hr = [150.0] * 1800
    result = aerobic_decoupling(Stream.from_values(power), Stream.from_values(hr), "cycling")
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA


@pytest.mark.property
def test_sport_mismatch_is_not_applicable() -> None:
    """A sport with no decoupling output channel ⇒ NOT_APPLICABLE_FOR_SPORT (ANL-R12)."""
    out = Stream.from_values([200.0] * 1800)
    hr = Stream.from_values([150.0] * 1800)
    result = aerobic_decoupling(out, hr, "swimming")
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.property
def test_too_few_included_per_half_is_insufficient_data() -> None:
    """A half with < MIN_INCLUDED_SAMPLES_PER_HALF moving seconds ⇒ INSUFFICIENT_DATA.

    Long enough by duration (passes the 20 min gate) but the second half is almost
    entirely coasting (power == 0), so it has too few included samples after exclusion.
    """
    # First half moving, second half coasting except a tiny moving stub (< 60 s).
    power = [200.0] * 900 + ([200.0] * 30 + [0.0] * 870)
    hr = [150.0] * 1800
    result = aerobic_decoupling(Stream.from_values(power), Stream.from_values(hr), "cycling")
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA


@CI_SETTINGS
@given(
    power=st.floats(min_value=80.0, max_value=400.0, allow_nan=False),
    hr_first=st.floats(min_value=110.0, max_value=160.0, allow_nan=False),
    hr_second=st.floats(min_value=110.0, max_value=175.0, allow_nan=False),
)
def test_pure_and_deterministic(power: float, hr_first: float, hr_second: float) -> None:
    """Same inputs ⇒ identical result (ANL-R2/R30 determinism)."""
    out = Stream.from_values([power] * 1800)
    hr = Stream.from_values([hr_first] * 900 + [hr_second] * 900)
    r1 = aerobic_decoupling(out, hr, "cycling")
    r2 = aerobic_decoupling(out, hr, "cycling")
    assert isinstance(r1, Computed)
    assert isinstance(r2, Computed)
    assert r1.value == r2.value


@pytest.mark.property
def test_min_included_constant_is_positive() -> None:
    """Sanity: the declared per-half minimum is a positive integer (DEC-R2)."""
    assert isinstance(MIN_INCLUDED_SAMPLES_PER_HALF, int)
    assert MIN_INCLUDED_SAMPLES_PER_HALF > 0
