"""Golden-reference tests for aerobic decoupling (doc 40 §9, DEC-R5; TEST-R1/R4).

Each golden carries an *independent* hand derivation in its docstring (TEST-R4): the
expected value is computed from the closed-form decoupling definition, not by
re-running the implementation.

Fixture origin / citation
--------------------------
The decoupling definition is the standard Pw:Hr / Pa:Hr "aerobic decoupling" used in
endurance training (Allen & Coggan, *Training and Racing with a Power Meter*; the
metric popularized by TrainingPeaks / Joe Friel as "aerobic decoupling"), encoded in
doc 40 §9 DEC-R1::

    t_mid       = (t_start + t_end) / 2
    eff_half    = mean(included smoothed output) / mean(included HR)   # per half
    decoupling% = ((eff_first - eff_second) / eff_first) * 100

These goldens are constructed so the 30 s output smoothing (DEC-R3) is identically
flat over every included second (the output is globally constant), which makes the
expected value a pure ratio of constants — derivable by hand to closed form.
"""

from __future__ import annotations

import pytest

from wattwise_core.analytics.decoupling import aerobic_decoupling
from wattwise_core.analytics.result import Computed
from wattwise_core.analytics.series import Stream

pytestmark = pytest.mark.golden

# Doc 40 §4: constant-power tol (NP family) is 1e-6; closed-form default is 1e-9.
ABS_TOL = 1e-9


@pytest.mark.golden
def test_constant_power_and_hr_is_zero_decoupling() -> None:
    """Constant power + constant HR ⇒ 0 % decoupling (DEC-R5, golden).

    Hand derivation: with power ≡ 200 W and HR ≡ 150 bpm over a 1800 s (30 min) ride,
    every included second has smoothed output 200 and HR 150. So
    ``eff_first = eff_second = 200/150`` and
    ``decoupling% = ((200/150 - 200/150)/(200/150))·100 = 0``. Exact to ``ABS_TOL``.
    """
    out = Stream.from_values([200.0] * 1800)
    hr = Stream.from_values([150.0] * 1800)

    result = aerobic_decoupling(out, hr, "cycling")

    assert isinstance(result, Computed)
    assert result.value == pytest.approx(0.0, abs=ABS_TOL)
    assert result.quality.sample_rate_hz == 1.0
    assert result.provenance.channels == ("power", "heart_rate")
    assert result.provenance.sport == "cycling"


@pytest.mark.golden
def test_step_hr_rise_gives_known_positive_decoupling() -> None:
    """Power held, HR steps 150→160 at the half boundary ⇒ exactly 6.25 % (DEC-R1/R5).

    Hand derivation (independent of the implementation):

    - 1800 s ride, valid seconds 0..1799 ⇒ ``t_mid = (0 + 1799)/2 = 899.5``; first half
      is seconds 0..899, second half 900..1799 (DEC-R1, time midpoint).
    - Power ≡ 200 W everywhere ⇒ every included (seeded, moving) second smooths to 200,
      so both half-means of smoothed output are exactly 200.
    - HR = 150 bpm for seconds 0..899, 160 bpm for seconds 900..1799 ⇒ first-half mean
      HR = 150, second-half mean HR = 160 (each half's HR is constant within the half).
    - ``eff_first = 200/150 = 4/3``; ``eff_second = 200/160 = 5/4``.
    - ``decoupling% = ((4/3 - 5/4)/(4/3))·100 = ((1/12)/(16/12))·100 = (1/16)·100 = 6.25``.
    """
    out = Stream.from_values([200.0] * 1800)
    hr = Stream.from_values([150.0] * 900 + [160.0] * 900)

    result = aerobic_decoupling(out, hr, "cycling")

    assert isinstance(result, Computed)
    assert result.value == pytest.approx(6.25, abs=ABS_TOL)
    # Sign convention (DEC-R5): second-half efficiency drop ⇒ POSITIVE decoupling.
    assert result.value > 0.0
    q = result.quality.extra
    assert q["eff_first_half"] == pytest.approx(4.0 / 3.0, abs=ABS_TOL)
    assert q["eff_second_half"] == pytest.approx(5.0 / 4.0, abs=ABS_TOL)
    assert q["t_mid_s"] == pytest.approx(899.5, abs=ABS_TOL)


@pytest.mark.golden
def test_step_hr_drop_gives_known_negative_decoupling() -> None:
    """Power held, HR steps 150→140 ⇒ exactly -7.142857… % (sign convention, DEC-R5).

    Hand derivation: same construction as above but HR drops 150→140 in the second
    half (efficiency *improves*). ``eff_first = 200/150 = 4/3``,
    ``eff_second = 200/140 = 10/7``. ``decoupling% = ((4/3 - 10/7)/(4/3))·100``.
    ``4/3 - 10/7 = (28 - 30)/21 = -2/21``; divided by ``4/3`` ⇒ ``(-2/21)·(3/4) = -1/14``;
    ``·100 = -7.142857142857…``. Negative ⇒ second-half efficiency gain (DEC-R5 sign).
    """
    out = Stream.from_values([200.0] * 1800)
    hr = Stream.from_values([150.0] * 900 + [140.0] * 900)

    result = aerobic_decoupling(out, hr, "cycling")

    assert isinstance(result, Computed)
    assert result.value == pytest.approx(-100.0 / 14.0, abs=ABS_TOL)
    assert result.value < 0.0


@pytest.mark.golden
def test_pace_sport_uses_speed_channel() -> None:
    """Pace sport (running) decouples speed:HR; same closed form (sport-parameterized).

    Hand derivation: speed ≡ 4.0 m/s, HR 150→160 at the midpoint over 1800 s ⇒
    identical algebra to the power golden ⇒ exactly 6.25 %. Confirms the output
    channel is selected by ``sport`` (Pa:Hr for running), not by a source-name branch
    (ANL-R11/R13), and the value is channel-agnostic given the same numbers.
    """
    out = Stream.from_values([4.0] * 1800)
    hr = Stream.from_values([150.0] * 900 + [160.0] * 900)

    result = aerobic_decoupling(out, hr, "running")

    assert isinstance(result, Computed)
    assert result.value == pytest.approx(6.25, abs=ABS_TOL)
    assert result.provenance.channels == ("speed", "heart_rate")
    assert result.quality.extra["output_channel"] == "speed"
