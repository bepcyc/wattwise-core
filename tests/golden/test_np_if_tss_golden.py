"""Golden-reference tests for the cycling-power load family (doc 40 §4 / §7B).

Implements the named golden cases NP-T2 (constant-power identity), TSS-T1 / TSS-T1a
(3600 valid-moving seconds @ IF=1.0 ⇒ TSS==100 and duration_valid_s==3600 even though
NP is first valid at 30 s), IF-T1 (IF == NP/FTP), and an LM-Golden whole-bundle check.

Fixture origin / derivation (TEST-R4):
  All expected values are hand-derived from the closed-form definitions in doc 40 §3
  with no dependence on the implementation under test:

  Case A — constant power P=c for N=3600 s, FTP=c (c=250 W):
    Resample to 1 Hz ⇒ 3600 valid seconds, no gaps.
    R(t) = 30 s trailing arithmetic mean of a constant series = c for every seeded
      second; seeded for t in [29, 3599] ⇒ analysis window = 3600 - 29 = 3571 seconds.
    mean(R^4) = c^4  ⇒  NP = (c^4)^(1/4) = c = 250.0  exactly.
    avg_power = c = 250; mean(R) = c = 250  ⇒  Jensen equality NP == mean(R) == c.
    duration_valid_s = count of valid 1 Hz seconds over the WHOLE effort = 3600
      (NOT the 3571-second NP analysis window — TSS-R1 / doc 40 §7 note 7).
    IF = NP/FTP = 250/250 = 1.0  (IF-T1).
    TSS = duration_valid_s * NP^2 / (FTP^2 * 3600) * 100
        = 3600 * 250^2 / (250^2 * 3600) * 100 = 100.0  (TSS-T1 / TSS-R2), tol 1e-6.

  Case B — short ramp then steady, hand-checked NP:
    A 4-second ramp [100, 200, 300, 400] padded out to a 30-sample window is NOT used
    for NP (window must be fully seeded, NP-R2); instead we verify the seeding rule by
    a 60-second constant ride giving NP == c with analysis window == 31.

  Case C — LM-Golden bundle: same 3600 s constant ride, FTP=250, avg_power=250,
    avg_hr=125 ⇒ EF = NP/avg_hr = 250/125 = 2.0; VI = NP/avg_power = 250/250 = 1.0;
    tss_per_hour = TSS / (3600/3600) = 100.0; intensity_class(IF=1.0) = "threshold"
    (0.90 <= 1.0 < 1.05); load_model = "power_tss".

Citation for the formulas: Coggan & Allen, *Training and Racing with a Power Meter*
(Normalized Power, Intensity Factor, TSS definitions); doc 40 §3 (NP-R1, IF-R1, TSS-R1).
"""

from __future__ import annotations

import pytest

from wattwise_core.analytics.np_if_tss import (
    LOAD_MODEL_POWER_TSS,
    intensity_factor,
    load_metrics_bundle,
    normalized_power,
    power_tss,
)
from wattwise_core.analytics.result import Computed, is_computed
from wattwise_core.analytics.series import Stream

TSS_TOL = 1e-6
NP_TOL = 1e-6

CONST_POWER_W = 250.0
RIDE_SECONDS = 3600


def _constant_power_stream(power_w: float, seconds: int) -> Stream:
    return Stream.from_values([power_w] * seconds)


@pytest.mark.golden
def test_np_constant_power_identity() -> None:
    """NP-T2: constant power c ⇒ NP == c == avg_power == mean(R) to 1e-6."""
    stream = _constant_power_stream(CONST_POWER_W, RIDE_SECONDS)
    result = normalized_power(stream)

    assert is_computed(result)
    assert isinstance(result, Computed)
    assert result.value.np_w == pytest.approx(CONST_POWER_W, abs=NP_TOL)
    assert result.value.avg_power_w == pytest.approx(CONST_POWER_W, abs=NP_TOL)
    assert result.value.mean_r_w == pytest.approx(CONST_POWER_W, abs=NP_TOL)
    # Analysis window drops the first 29 s of warm-up (R(t) not yet seeded).
    assert result.value.analysis_window_s == RIDE_SECONDS - 29
    # duration_valid_s (whole effort) is carried in the quality report, == 3600.
    assert result.quality.extra["duration_valid_s"] == RIDE_SECONDS
    assert result.quality.extra["analysis_window_s"] == RIDE_SECONDS - 29


@pytest.mark.golden
def test_seeding_short_constant_ride() -> None:
    """NP-R2 seeding: 60 s constant ride ⇒ NP==c, analysis window == 31 (60-29)."""
    stream = _constant_power_stream(CONST_POWER_W, 60)
    result = normalized_power(stream)
    assert is_computed(result)
    assert isinstance(result, Computed)
    assert result.value.np_w == pytest.approx(CONST_POWER_W, abs=NP_TOL)
    assert result.value.analysis_window_s == 60 - 29


@pytest.mark.golden
def test_intensity_factor_equals_np_over_ftp() -> None:
    """IF-T1: IF == NP/FTP exactly; at FTP=NP ⇒ IF == 1.0."""
    stream = _constant_power_stream(CONST_POWER_W, RIDE_SECONDS)
    np_result = normalized_power(stream)
    if_result = intensity_factor(np_result, ftp_w=CONST_POWER_W)
    assert is_computed(if_result)
    assert isinstance(if_result, Computed)
    assert if_result.value == pytest.approx(1.0, abs=NP_TOL)

    # IF == NP/FTP for an off-threshold FTP too (no avg-power fallback).
    if_result2 = intensity_factor(np_result, ftp_w=200.0)
    assert isinstance(if_result2, Computed)
    assert if_result2.value == pytest.approx(CONST_POWER_W / 200.0, abs=NP_TOL)


@pytest.mark.golden
def test_tss_t1_3600s_at_if_one_equals_100() -> None:
    """TSS-T1 / TSS-T1a: 3600 valid-moving s @ IF=1.0 ⇒ TSS==100, duration==3600.

    The headline golden: TSS uses duration_valid_s=3600 (the whole-effort valid count),
    NOT the 3571-second NP analysis window, so the first-29-s ramp does not deflate it.
    """
    stream = _constant_power_stream(CONST_POWER_W, RIDE_SECONDS)
    np_result = normalized_power(stream)
    assert isinstance(np_result, Computed)

    # duration_valid_s is exactly 3600 even though NP is first valid at 30 s.
    duration_valid_s = int(np_result.quality.extra["duration_valid_s"])
    assert duration_valid_s == 3600

    tss_result = power_tss(np_result, ftp_w=CONST_POWER_W, duration_valid_s=duration_valid_s)
    assert is_computed(tss_result)
    assert isinstance(tss_result, Computed)
    assert tss_result.value == pytest.approx(100.0, abs=TSS_TOL)


@pytest.mark.golden
def test_lm_golden_full_bundle() -> None:
    """LM-Golden: hand-derived whole bundle for the 3600 s constant ride.

    EF = NP/avg_hr = 250/125 = 2.0; VI = NP/avg_power = 250/250 = 1.0;
    tss_per_hour = 100.0; intensity_class = "threshold"; load_model = "power_tss".
    """
    stream = _constant_power_stream(CONST_POWER_W, RIDE_SECONDS)
    bundle = load_metrics_bundle(
        power_stream=stream,
        hr_stream=None,
        ftp_w=CONST_POWER_W,
        avg_power_w=CONST_POWER_W,
        avg_hr_bpm=125.0,
    )

    assert isinstance(bundle.duration_valid_s, Computed)
    assert bundle.duration_valid_s.value == 3600

    assert isinstance(bundle.np, Computed)
    assert bundle.np.value.np_w == pytest.approx(CONST_POWER_W, abs=NP_TOL)

    assert isinstance(bundle.if_, Computed)
    assert bundle.if_.value == pytest.approx(1.0, abs=NP_TOL)

    assert isinstance(bundle.tss, Computed)
    assert bundle.tss.value == pytest.approx(100.0, abs=TSS_TOL)

    assert isinstance(bundle.tss_per_hour, Computed)
    assert bundle.tss_per_hour.value == pytest.approx(100.0, abs=TSS_TOL)

    assert isinstance(bundle.efficiency_factor, Computed)
    assert bundle.efficiency_factor.value == pytest.approx(2.0, abs=NP_TOL)

    assert isinstance(bundle.variability_index, Computed)
    assert bundle.variability_index.value == pytest.approx(1.0, abs=NP_TOL)

    assert isinstance(bundle.intensity_class, Computed)
    assert bundle.intensity_class.value == "threshold"

    assert bundle.load_model == LOAD_MODEL_POWER_TSS


@pytest.mark.golden
def test_lm_golden_internal_consistency() -> None:
    """LM-R3: if == np/FTP and tss == duration*np^2/(FTP^2*3600)*100 in the bundle."""
    ftp = 240.0
    stream = _constant_power_stream(CONST_POWER_W, RIDE_SECONDS)
    bundle = load_metrics_bundle(
        power_stream=stream,
        hr_stream=None,
        ftp_w=ftp,
        avg_power_w=CONST_POWER_W,
        avg_hr_bpm=130.0,
    )
    assert isinstance(bundle.np, Computed)
    assert isinstance(bundle.if_, Computed)
    assert isinstance(bundle.tss, Computed)

    np_w = bundle.np.value.np_w
    duration = 3600
    expected_if = np_w / ftp
    expected_tss = duration * np_w * np_w / (ftp * ftp * 3600.0) * 100.0
    assert bundle.if_.value == pytest.approx(expected_if, abs=1e-9)
    assert bundle.tss.value == pytest.approx(expected_tss, abs=1e-9)


def _gapped_power_values(power_w: float, seconds: int, gap: range) -> list[float | None]:
    """Constant-power sample list with a genuine recording gap (``None``) over ``gap``."""
    return [None if i in gap else power_w for i in range(seconds)]


@pytest.mark.golden
def test_tss_t3_genuine_non_moving_seconds_reduce_duration_and_tss() -> None:
    """TSS-T3: an hour of wall-clock time with genuine non-moving seconds ⇒ TSS < 100.

    Hand derivation (TEST-R4): c=250 W over a 3600 s wall-clock hour with a single
    600 s stop — a long (> max_interp_gap_s) typed-null gap per GBO-R22, seconds
    1800..2399. Valid moving seconds = 3600 - 600 = 3000 (TSS-R1: stops/pauses/long
    gaps are NOT exercise time). NP over its own valid windows of a constant series
    is exactly c, so at FTP=c:
        TSS = duration_valid_s * NP^2 / (FTP^2 * 3600) * 100
            = 3000 * c^2 / (c^2 * 3600) * 100 = 250/3 = 83.333333...
    This is the intended valid-moving-time semantics, not a defect (TSS-R2).
    """
    values = _gapped_power_values(CONST_POWER_W, RIDE_SECONDS, range(1800, 2400))
    np_result = normalized_power(Stream.from_values(values))
    assert isinstance(np_result, Computed)
    assert np_result.value.np_w == pytest.approx(CONST_POWER_W, abs=NP_TOL)

    duration_valid_s = int(np_result.quality.extra["duration_valid_s"])
    assert duration_valid_s == 3000  # < 3600: only the genuine stop reduces it

    tss_result = power_tss(np_result, ftp_w=CONST_POWER_W, duration_valid_s=duration_valid_s)
    assert isinstance(tss_result, Computed)
    assert tss_result.value == pytest.approx(250.0 / 3.0, abs=TSS_TOL)
    assert tss_result.value < 100.0


@pytest.mark.golden
def test_tss_t3_np_warmup_exclusion_alone_does_not_reduce_duration() -> None:
    """TSS-T3 control: NP's 29 s rolling-window warmup drop NEVER reduces duration_valid_s.

    The same gap-free hour whose NP analysis window is 3571 s (the first 29 s are
    rolling-window warmup, NP-R2) still counts all 3600 s as valid moving time
    (TSS-R1/TSS-R2): duration_valid_s < 3600 may arise ONLY from genuine non-moving
    seconds, never from the NP analysis-window exclusion.
    """
    np_result = normalized_power(_constant_power_stream(CONST_POWER_W, RIDE_SECONDS))
    assert isinstance(np_result, Computed)
    assert np_result.value.analysis_window_s == RIDE_SECONDS - 29  # NP window IS clipped
    assert int(np_result.quality.extra["duration_valid_s"]) == RIDE_SECONDS  # duration is NOT


@pytest.mark.golden
def test_tss_t3_zero_watt_coast_seconds_count_as_moving() -> None:
    """TSS-T3 boundary: VALID 0 W samples are coasting — exercise time, not a stop.

    Canonical grounding (TSS-R1 + data-model GBO-R22): stops/pauses arrive as typed
    null samples, so a present, valid ``0 W`` sample is a recorded coasting second
    (descending / soft-pedalling) and MUST count in duration_valid_s. 100 contiguous
    0 W seconds in an otherwise constant-power hour leave duration_valid_s == 3600;
    only the NP value (and hence TSS) drops below the at-threshold 100.
    """
    values: list[float | None] = [
        0.0 if 1000 <= i < 1100 else CONST_POWER_W for i in range(RIDE_SECONDS)
    ]
    np_result = normalized_power(Stream.from_values(values))
    assert isinstance(np_result, Computed)

    duration_valid_s = int(np_result.quality.extra["duration_valid_s"])
    assert duration_valid_s == RIDE_SECONDS  # coasting is moving time

    tss_result = power_tss(np_result, ftp_w=CONST_POWER_W, duration_valid_s=duration_valid_s)
    assert isinstance(tss_result, Computed)
    assert tss_result.value < 100.0  # NP-driven, not duration-driven
