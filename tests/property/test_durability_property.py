"""Property-based tests for durability / fatigue resistance (doc 40 §10, DUR-T1..T8).

Covers the per-metric property IDs proposed with the requirement family (issue #26):

- **DUR-T1** identical fresh & fatigued efforts ⇒ ~0 % decrement (retained == 1);
- **DUR-T2** a lower fatigued effort at fixed fresh ⇒ a not-lower decrement (monotone);
- **DUR-T3** the fatigue axis ``accumulated_work_above_cp_j`` is non-decreasing, gaps
  contribute zero, and the total equals ``Σ max(0, P - CP)``;
- **DUR-T4** threshold never reached / a segment too short for the target duration ⇒
  ``Unavailable(INSUFFICIENT_DATA)`` (the sufficiency default, DUR-R5);
- **DUR-T5** sport / missing-input / domain gates fail closed with the exact reason;
- **DUR-T6** a fresh "best" below CP raises the non-blocking ``fresh_effort_below_cp``
  quality flag (DUR-R6);
- **DUR-T7** the per-athlete threshold helper is ``multiple · W'`` and fails closed on
  absent / non-positive inputs (DUR-R7);
- **DUR-T8** purity / determinism — equal inputs yield equal outputs (ANL-R2/R30).

All generators emit uniform 1 Hz power arrays (``np.nan`` = gap, TEST-R2).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.durability import (
    accumulated_work_above_cp_j,
    durability_decrement,
    fatigue_threshold_from_wprime,
)
from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason

pytestmark = pytest.mark.property

REL_TOL = 1e-9
CP_W = 250.0
CI_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _ride(fresh_w: float, fatigued_w: float) -> tuple[np.ndarray, float]:
    """A 300 s fresh effort, a 300 s fatigued effort, and a sub-CP cooldown (all > CP).

    Returns the 1 Hz power array AND a fixture-derived ``fatigue_threshold_j`` placed so
    the fresh→fatigued split lands exactly at the boundary between the two efforts
    (second 300): the threshold sits between the cumulative work-above-CP at the last
    fresh second and the first fatigued second. The fresh segment is then exactly the
    fresh block (best 300 s == ``fresh_w``) and the fatigued segment the fatigued block
    + cooldown (best 300 s == ``fatigued_w``, since the 200 W cooldown is lower). Both
    efforts must be above CP (250 W) for the split to be well-defined.
    """
    power = np.concatenate([np.full(300, fresh_w), np.full(300, fatigued_w), np.full(300, 200.0)])
    fresh_work = 300.0 * (fresh_w - CP_W)  # cumulative work-above-CP at second 299
    threshold = fresh_work + (fatigued_w - CP_W) / 2.0  # half a step into second 300
    return power, threshold


# --- DUR-T1: identical efforts ⇒ ~0 % decrement -----------------------------------
@CI_SETTINGS
@given(effort_w=st.floats(min_value=260.0, max_value=600.0, allow_nan=False))
def test_identical_efforts_zero_decrement(effort_w: float) -> None:
    """Fresh and fatigued best efforts equal ⇒ retained == 1, decrement == 0 (DUR-T1)."""
    power, threshold = _ride(effort_w, effort_w)
    result = durability_decrement(power, CP_W, fatigue_threshold_j=threshold, target_duration_s=300)
    assert isinstance(result, Computed)
    assert result.value.retained_fraction == pytest.approx(1.0, abs=REL_TOL)
    assert result.value.decrement_pct == pytest.approx(0.0, abs=REL_TOL)


# --- DUR-T2: monotonic in the fatigued effort -------------------------------------
@CI_SETTINGS
@given(
    fresh_w=st.floats(min_value=320.0, max_value=600.0, allow_nan=False),
    fatigued_w=st.floats(min_value=265.0, max_value=315.0, allow_nan=False),
    drop=st.floats(min_value=1.0, max_value=10.0, allow_nan=False),
)
def test_lower_fatigued_power_means_not_lower_decrement(
    fresh_w: float, fatigued_w: float, drop: float
) -> None:
    """A lower fatigued effort at fixed fresh ⇒ a not-lower decrement (DUR-T2)."""
    power_h, thr_h = _ride(fresh_w, fatigued_w)
    power_l, thr_l = _ride(fresh_w, fatigued_w - drop)
    higher = durability_decrement(power_h, CP_W, fatigue_threshold_j=thr_h, target_duration_s=300)
    lower = durability_decrement(power_l, CP_W, fatigue_threshold_j=thr_l, target_duration_s=300)
    assert isinstance(higher, Computed) and isinstance(lower, Computed)
    assert lower.value.decrement_pct >= higher.value.decrement_pct - REL_TOL


# --- DUR-T3: the fatigue axis is a well-formed cumulative work-above-CP signal ------
@CI_SETTINGS
@given(
    powers=st.lists(
        st.one_of(st.none(), st.floats(min_value=0.0, max_value=700.0, allow_nan=False)),
        min_size=1,
        max_size=400,
    )
)
def test_accumulated_work_axis_properties(powers: list[float | None]) -> None:
    """Non-decreasing, gaps contribute zero, total == Σ max(0, P - CP) (DUR-T3)."""
    arr = np.array([np.nan if p is None else p for p in powers], dtype=np.float64)
    axis = accumulated_work_above_cp_j(arr, CP_W)
    assert axis.shape == arr.shape
    # Non-decreasing by construction.
    assert np.all(np.diff(axis) >= -REL_TOL)
    expected_total = float(np.nansum(np.maximum(0.0, arr - CP_W)))
    assert float(axis[-1]) == pytest.approx(expected_total, abs=1e-6)


# --- DUR-T4: sufficiency is the default path --------------------------------------
@CI_SETTINGS
@given(threshold=st.floats(min_value=1e6, max_value=1e9, allow_nan=False))
def test_threshold_not_reached_is_insufficient(threshold: float) -> None:
    """An unreachable threshold ⇒ INSUFFICIENT_DATA, never a fabricated decrement (DUR-T4)."""
    power, _ = _ride(330.0, 300.0)
    result = durability_decrement(power, CP_W, fatigue_threshold_j=threshold, target_duration_s=300)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA


@pytest.mark.property
def test_fresh_segment_too_short_is_insufficient() -> None:
    """A target longer than the fresh segment ⇒ INSUFFICIENT_DATA (DUR-T4)."""
    # A tiny threshold crosses within the first fresh second, leaving < 300 s of fresh data.
    power, _ = _ride(330.0, 300.0)
    result = durability_decrement(power, CP_W, fatigue_threshold_j=1.0, target_duration_s=300)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA


@pytest.mark.property
def test_fatigued_segment_too_short_is_insufficient() -> None:
    """A fatigued segment shorter than the target ⇒ INSUFFICIENT_DATA (DUR-T4).

    The threshold is crossed half a step into a 5-second tail, so the fatigued segment
    has only 5 samples for a 300 s target — a partial window must never be averaged as
    if it were a full one (the symmetric twin of the fresh-side case above).
    """
    power = np.concatenate([np.full(350, 330.0), np.full(5, 300.0)])
    threshold = 350.0 * (330.0 - CP_W) + (300.0 - CP_W) / 2.0  # crosses at second 350
    result = durability_decrement(power, CP_W, fatigue_threshold_j=threshold, target_duration_s=300)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA
    assert "fatigued" in result.detail


# --- DUR-T5: fail-closed gates -----------------------------------------------------
@pytest.mark.property
def test_non_cycling_sport_not_applicable() -> None:
    """A non-cycling sport fails closed with NOT_APPLICABLE_FOR_SPORT (DUR-T5/R8)."""
    power, threshold = _ride(330.0, 300.0)
    result = durability_decrement(
        power,
        CP_W,
        fatigue_threshold_j=threshold,
        target_duration_s=300,
        sport="running",
    )
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.property
def test_missing_and_domain_inputs_fail_closed() -> None:
    """Missing power ⇒ MISSING_REQUIRED_INPUT; bad CP / threshold ⇒ OUT_OF_DOMAIN (DUR-T5)."""
    empty = durability_decrement(np.array([], dtype=np.float64), CP_W, fatigue_threshold_j=20_000.0)
    assert isinstance(empty, Unavailable)
    assert empty.reason is UnavailableReason.MISSING_REQUIRED_INPUT

    all_gap = durability_decrement(np.full(600, np.nan), CP_W, fatigue_threshold_j=20_000.0)
    assert isinstance(all_gap, Unavailable)
    assert all_gap.reason is UnavailableReason.MISSING_REQUIRED_INPUT

    power, _ = _ride(330.0, 300.0)
    bad_cp = durability_decrement(power, 0.0, fatigue_threshold_j=20_000.0)
    assert isinstance(bad_cp, Unavailable)
    assert bad_cp.reason is UnavailableReason.OUT_OF_DOMAIN

    bad_threshold = durability_decrement(power, CP_W, fatigue_threshold_j=-1.0)
    assert isinstance(bad_threshold, Unavailable)
    assert bad_threshold.reason is UnavailableReason.OUT_OF_DOMAIN


@pytest.mark.property
def test_non_positive_target_duration_raises() -> None:
    """A non-positive target duration is a programmer error (ValueError), not a metric state."""
    power, _ = _ride(330.0, 300.0)
    with pytest.raises(ValueError, match="target_duration_s"):
        durability_decrement(power, CP_W, fatigue_threshold_j=20_000.0, target_duration_s=0)


# --- DUR-T6: the non-maximal-fresh quality flag ------------------------------------
@pytest.mark.property
def test_fresh_effort_below_cp_is_flagged() -> None:
    """A fresh best below CP raises the non-blocking fresh_effort_below_cp flag (DUR-T6).

    The fresh segment here is a clean 300 s at 245 W (< CP 250, contributing zero work),
    then an above-CP fatigued effort. The split sits at the boundary (threshold 25 J,
    crossed half a second into the 300 W block), so the fresh best is exactly 245 W and
    the flag fires while the metric still computes.
    """
    power = np.concatenate(
        [
            np.full(300, 245.0),  # fresh "best" below CP — not maximal
            np.full(300, 300.0),  # fatigued effort above CP
            np.full(300, 200.0),  # cooldown
        ]
    )
    result = durability_decrement(power, CP_W, fatigue_threshold_j=25.0, target_duration_s=300)
    assert isinstance(result, Computed)
    assert result.value.fresh_best_power_w == pytest.approx(245.0, abs=REL_TOL)
    assert result.quality.extra["fresh_effort_below_cp"] is True


# --- DUR-T7: per-athlete threshold helper ------------------------------------------
@CI_SETTINGS
@given(
    w_prime=st.floats(min_value=5_000.0, max_value=40_000.0, allow_nan=False),
    multiple=st.floats(min_value=1.0, max_value=30.0, allow_nan=False),
)
def test_threshold_helper_is_multiple_of_wprime(w_prime: float, multiple: float) -> None:
    """fatigue_threshold_from_wprime == multiple · W' for valid inputs (DUR-T7)."""
    result = fatigue_threshold_from_wprime(w_prime, multiple=multiple)
    assert isinstance(result, Computed)
    assert result.value == pytest.approx(multiple * w_prime, rel=REL_TOL)


@pytest.mark.property
def test_threshold_helper_fails_closed() -> None:
    """Absent W' ⇒ MISSING_REQUIRED_INPUT; non-positive W'/multiple ⇒ OUT_OF_DOMAIN (DUR-T7)."""
    assert fatigue_threshold_from_wprime(None).reason is (  # type: ignore[union-attr]
        UnavailableReason.MISSING_REQUIRED_INPUT
    )
    assert fatigue_threshold_from_wprime(-1.0).reason is (  # type: ignore[union-attr]
        UnavailableReason.OUT_OF_DOMAIN
    )
    assert fatigue_threshold_from_wprime(20_000.0, multiple=0.0).reason is (  # type: ignore[union-attr]
        UnavailableReason.OUT_OF_DOMAIN
    )


# --- DUR-T8: purity / determinism --------------------------------------------------
@CI_SETTINGS
@given(
    fresh_w=st.floats(min_value=300.0, max_value=600.0, allow_nan=False),
    fatigued_w=st.floats(min_value=255.0, max_value=295.0, allow_nan=False),
)
def test_deterministic(fresh_w: float, fatigued_w: float) -> None:
    """Equal inputs ⇒ equal outputs — the metric is a pure function (DUR-T8, ANL-R2/R30)."""
    ride, threshold = _ride(fresh_w, fatigued_w)
    a = durability_decrement(ride, CP_W, fatigue_threshold_j=threshold, target_duration_s=300)
    b = durability_decrement(ride, CP_W, fatigue_threshold_j=threshold, target_duration_s=300)
    assert isinstance(a, Computed) and isinstance(b, Computed)
    assert a.value == b.value
