"""Golden-reference tests for durability / fatigue resistance (doc 40 §10, DUR-R1..R8).

Fixture origin / derivation (TEST-R4) — every expected value is hand-derived from the
DOCUMENTED definition (the work-conditioned power decrement), independent of the
implementation. The canonical fixture is a single synthetic 1 Hz ride with CP = 250 W:

    segment        seconds   power_w   work_above_cp contribution (W·s = J)
    fresh effort     300       320      300·(320-250) =  21 000
    accumulator      900       260      900·(260-250) =   9 000   (cum reaches 30 000)
    fatigued effort  300       288      300·(288-250) =  11 400
    cooldown         300       200      below CP      =        0
                                                       total  =  41 400 J

  With ``fatigue_threshold_j = 30 000`` the fresh→fatigued split is the second at which
  cumulative work-above-CP first reaches 30 000 J: that is the end of the accumulator,
  index 1199 (21 000 after the fresh block + 9 000 over 900 accumulator seconds).

  Best 300 s power fresh (before 1199): the 320 W effort ⇒ 320 W.
  Best 300 s power fatigued (at/after 1199): the 288 W effort ⇒ 288 W (the 260 / 200 W
  stretches are lower).

  retained = 288 / 320 = 0.9  ⇒  decrement = 100·(1 - 0.9) = 10.0 %.
"""

from __future__ import annotations

import numpy as np
import pytest

from wattwise_core.analytics.durability import (
    DurabilityDecrement,
    accumulated_work_above_cp_j,
    durability_decrement,
    fatigue_threshold_from_wprime,
)
from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason

TOL = 1e-9
CP_W = 250.0


def _canonical_ride() -> np.ndarray:
    """The hand-derived fixture ride described in the module docstring (1 Hz watts)."""
    return np.concatenate(
        [
            np.full(300, 320.0),  # fresh maximal 300 s effort
            np.full(900, 260.0),  # accumulator (just above CP)
            np.full(300, 288.0),  # fatigued 300 s effort (10 % below fresh)
            np.full(300, 200.0),  # cooldown below CP
        ]
    )


@pytest.mark.golden
def test_durability_decrement_canonical() -> None:
    """DUR-R1..R4: the canonical ride yields a 10.0 % decrement, fresh 320 / fatigued 288."""
    result = durability_decrement(
        _canonical_ride(), CP_W, fatigue_threshold_j=30_000.0, target_duration_s=300
    )
    assert isinstance(result, Computed)
    value = result.value
    assert isinstance(value, DurabilityDecrement)
    assert value.fresh_best_power_w == pytest.approx(320.0, abs=TOL)
    assert value.fatigued_best_power_w == pytest.approx(288.0, abs=TOL)
    assert value.retained_fraction == pytest.approx(0.9, abs=TOL)
    assert value.decrement_pct == pytest.approx(10.0, abs=TOL)
    assert value.split_elapsed_s == 1199
    assert value.work_above_cp_total_j == pytest.approx(41_400.0, abs=TOL)
    # The fresh best (320 W) is well above CP, so it reads as a genuine maximal effort.
    assert result.quality.extra["fresh_effort_below_cp"] is False
    assert result.provenance.reference_params["cp_w"] == pytest.approx(CP_W, abs=TOL)


@pytest.mark.golden
def test_accumulated_work_axis_is_intensity_weighted() -> None:
    """DUR-R1: the fatigue axis is cumulative work ABOVE CP; gaps and P == CP add zero."""
    power = np.array([300.0, np.nan, 240.0, 250.0, 260.0], dtype=np.float64)  # cp = 250
    axis = accumulated_work_above_cp_j(power, CP_W)
    # increments: 50, 0 (gap), 0 (below CP), 0 (exactly AT CP: max(0, 0) == 0, no
    # floating-point residual), 10  ⇒ cumulative 50, 50, 50, 50, 60.
    np.testing.assert_allclose(axis, [50.0, 50.0, 50.0, 50.0, 60.0], atol=TOL)
    assert float(axis[-1]) == pytest.approx(60.0, abs=TOL)


@pytest.mark.golden
def test_fatigue_threshold_from_wprime_is_multiple_of_wprime() -> None:
    """DUR-R7: the per-athlete threshold is ``multiple · W'`` joules of work-above-CP.

    The multiple (3.0) is deliberately NOT the packaged default (10.0), so an
    implementation that ignores the argument and reads the config constant fails here
    (non-vacuity): 3 · 20 000 = 60 000, whereas the default would give 200 000.
    """
    result = fatigue_threshold_from_wprime(20_000.0, multiple=3.0)
    assert isinstance(result, Computed)
    assert result.value == pytest.approx(60_000.0, abs=TOL)


@pytest.mark.golden
def test_threshold_never_reached_fails_closed() -> None:
    """DUR-R5: an easy ride that never fatigues the athlete is INSUFFICIENT_DATA, not 100 %."""
    easy = np.full(3600, 200.0)  # never exceeds CP ⇒ zero work above CP
    result = durability_decrement(easy, CP_W, fatigue_threshold_j=30_000.0, target_duration_s=300)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA
