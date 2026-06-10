"""Property-based cross-sport fabrication sweep for the cycling-power family.

- **SPORT-T3 (no cross-sport fabrication sweep)** — across generated multi-sport power
  traces, no cycling-power-family metric emits a plausible-but-unfounded number for an
  inapplicable sport: the only outcomes are ``Computed`` (cycling, applicable) or
  ``Unavailable(NOT_APPLICABLE_FOR_SPORT)`` for a non-power sport — never ``0`` and never
  a cross-sport surrogate (ANL-R12 / ANL-R4).
- **TEST-R3 (fail-closed is itself tested)** — the sport-mismatch case for EVERY metric in
  the cycling-power family maps to ``NOT_APPLICABLE_FOR_SPORT`` (not a different reason and
  not a number), DISTINCT from ``MISSING_REQUIRED_INPUT`` for an absent channel. This now
  includes the mean-maximal-power curve (``mmp``, every grid duration) and its derived
  ``best_effort`` — the power-curve family, not only NP/IF/TSS/W'balance.

The traces are valid power series (no NaN-only / empty inputs) so the gate's effect is
isolated to ``sport``: a mutation that dropped the gate would let a non-power sport return
a ``Computed`` number and break these properties.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.constants import MMP_DURATION_GRID_S
from wattwise_core.analytics.mmp_cp import best_effort, mmp
from wattwise_core.analytics.np_if_tss import (
    intensity_factor,
    load_metrics_bundle,
    normalized_power,
    power_tss,
)
from wattwise_core.analytics.result import Unavailable, UnavailableReason, is_computed
from wattwise_core.analytics.series import Stream
from wattwise_core.analytics.wbal import wbal

CI_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Sports that lack a true mechanical-power channel (or are unknown): the cycling-power
# family is NOT applicable to any of these.
NON_POWER_SPORTS = st.sampled_from(
    ["running", "swimming", "xc_ski", "strength", "other", "ski", "hike", "made_up"]
)

# A valid, fully-populated 1 Hz power series long enough to seed NP (>= 30 s).
powers = st.floats(min_value=1.0, max_value=1500.0, allow_nan=False, allow_infinity=False)
power_lists = st.lists(powers, min_size=30, max_size=200)


@given(values=power_lists, sport=NON_POWER_SPORTS)
@CI_SETTINGS
def test_np_never_fabricates_for_non_power_sport(values: list[float], sport: str) -> None:
    """SPORT-T3/TEST-R3: NP on a non-power sport is ONLY NOT_APPLICABLE_FOR_SPORT."""
    result = normalized_power(Stream.from_values(values), sport=sport)
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@given(values=power_lists, sport=NON_POWER_SPORTS)
@CI_SETTINGS
def test_if_tss_wbal_never_fabricate_for_non_power_sport(values: list[float], sport: str) -> None:
    """SPORT-T3/TEST-R3: IF/TSS/W'bal never emit a number for an inapplicable sport."""
    np_res = normalized_power(Stream.from_values(values), sport=sport)
    if_res = intensity_factor(np_res, 250.0)
    tss_res = power_tss(np_res, 250.0, len(values))
    wbal_res = wbal(Stream.from_values(values).values, 240.0, 20000.0, sport=sport)
    for res in (if_res, tss_res, wbal_res):
        assert isinstance(res, Unavailable)
        assert res.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@given(values=power_lists, sport=NON_POWER_SPORTS)
@CI_SETTINGS
def test_bundle_power_fields_never_fabricate_for_non_power_sport(
    values: list[float], sport: str
) -> None:
    """SPORT-T3: the bundle's power fields are all NOT_APPLICABLE_FOR_SPORT (no 0/number)."""
    bundle = load_metrics_bundle(Stream.from_values(values), None, 250.0, 200.0, 140.0, sport=sport)
    for field in (bundle.np, bundle.if_, bundle.tss, bundle.intensity_class):
        assert isinstance(field, Unavailable)
        assert field.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@given(values=power_lists, sport=NON_POWER_SPORTS)
@CI_SETTINGS
def test_mmp_curve_never_fabricates_for_non_power_sport(values: list[float], sport: str) -> None:
    """SPORT-T3/TEST-R3: the WHOLE MMP curve on a non-power sport is per-duration gated.

    For a valid power trace, EVERY grid duration maps to ``NOT_APPLICABLE_FOR_SPORT``
    (never a Computed peak, never ``INSUFFICIENT_DATA``). A mutation that dropped the
    ``mmp`` sport gate would let a long-enough duration return a ``Computed`` number.
    """
    power = np.asarray(values, dtype=np.float64)
    results = mmp(power, MMP_DURATION_GRID_S, sport=sport)
    for res in results.values():
        assert isinstance(res, Unavailable)
        assert res.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@given(values=power_lists, sport=NON_POWER_SPORTS, d=st.sampled_from(MMP_DURATION_GRID_S))
@CI_SETTINGS
def test_best_effort_never_fabricates_for_non_power_sport(
    values: list[float], sport: str, d: int
) -> None:
    """SPORT-T3/TEST-R3: best_effort (derived from MMP) inherits the sport gate."""
    res = best_effort(np.asarray(values, dtype=np.float64), d, sport=sport)
    assert isinstance(res, Unavailable)
    assert res.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@given(values=power_lists)
@CI_SETTINGS
def test_cycling_is_applicable_and_returns_a_real_or_typed_value(
    values: list[float],
) -> None:
    """The gate does not over-block: cycling yields Computed or a NON-sport Unavailable.

    For cycling the sport is applicable, so NP is either Computed (a finite number) or an
    Unavailable whose reason is NOT the sport-mismatch reason — proving the gate fires on
    sport alone, never on the applicable sport.
    """
    result = normalized_power(Stream.from_values(values), sport="cycling")
    if is_computed(result):
        assert math.isfinite(result.value.np_w)
    else:
        assert isinstance(result, Unavailable)
        assert result.reason != UnavailableReason.NOT_APPLICABLE_FOR_SPORT
