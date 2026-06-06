"""Golden-reference tests for HRV (doc 40 §8, TEST-R1/R4).

Origin / derivation note (TEST-R4)
----------------------------------
The time-domain golden is HAND-DERIVED from a tiny five-beat NN sequence so the
expected RMSSD/SDNN/pNN50/meanNN can be checked by elementary arithmetic, with no
appeal to the implementation under test:

    NN (ms)                = [800, 850, 820, 810, 805]
    successive diffs (ms)  = [ +50,  -30,  -10,   -5]      (NN[i+1] - NN[i])
    squared diffs          = [2500,  900,  100,   25]
    RMSSD  = sqrt(mean(squared diffs)) = sqrt((2500+900+100+25)/4)
           = sqrt(881.25)              = 29.685855217594792 ms   (HRV-R3)
    meanNN = (800+850+820+810+805)/5   = 817.0 ms
    SDNN   = sqrt( mean((NN - meanNN)^2) )  (population SD, ddof=0)
           = sqrt( (289 + 1089 + 9 + 49 + 144)/5 ) = sqrt(316.0)
           = 17.776388834631177 ms
    pNN50  = % of |successive diff| > 50 ms = 0/4 = 0.0 %  (|+50| is NOT > 50)

Every interval lies within 20% of its local-median baseline, so the mandatory
artifact-correction stage (HRV-R1) flags ZERO beats (corrected_fraction == 0.0);
the NN series equals the input RR series and the golden values are exact.

The freq-domain Parseval sanity (HRV-R7) is asserted against a synthetically
generated tachogram (a single sinusoid in the HF band), where the total integrated
spectral power must equal the variance of the detrended tachogram (Parseval's
theorem) and the band power must concentrate in HF — a self-evident DSP identity,
not a value copied from the implementation.
"""

from __future__ import annotations

import math

import pytest

from wattwise_core.analytics.hrv import (
    FreqDomainHrv,
    HrvFidelity,
    TimeDomainHrv,
    dsp_available,
    freq_domain_hrv,
    time_domain_hrv,
)
from wattwise_core.analytics.result import Computed

# Hand-derived golden NN sequence and expected scalars (see module docstring).
GOLDEN_NN_MS = [800.0, 850.0, 820.0, 810.0, 805.0]
EXPECTED_RMSSD = 29.685855217594792
EXPECTED_SDNN = 17.776388834631177
EXPECTED_MEAN_NN = 817.0
EXPECTED_PNN50 = 0.0
GOLDEN_TOL = 1e-9


@pytest.mark.golden
def test_time_domain_rmssd_golden() -> None:
    """RMSSD/SDNN/pNN50/meanNN match the hand-derived values (HRV-R3, TEST-R4)."""
    # min_duration_s lowered to admit the 5-beat fixture; the formula is unchanged.
    result = time_domain_hrv(rr_intervals_ms=GOLDEN_NN_MS, min_duration_s=0.0)
    assert isinstance(result, Computed)
    value = result.value
    assert isinstance(value, TimeDomainHrv)
    assert value.rmssd_ms == pytest.approx(EXPECTED_RMSSD, abs=GOLDEN_TOL)
    assert value.sdnn_ms == pytest.approx(EXPECTED_SDNN, abs=GOLDEN_TOL)
    assert value.mean_nn_ms == pytest.approx(EXPECTED_MEAN_NN, abs=GOLDEN_TOL)
    assert value.pnn50_pct == pytest.approx(EXPECTED_PNN50, abs=GOLDEN_TOL)


@pytest.mark.golden
def test_time_domain_artifact_correction_is_zero_on_clean_series() -> None:
    """The clean golden series triggers no artifact correction (HRV-R1)."""
    result = time_domain_hrv(rr_intervals_ms=GOLDEN_NN_MS, min_duration_s=0.0)
    assert isinstance(result, Computed)
    assert result.quality.extra["corrected_fraction"] == 0.0
    assert result.quality.extra["fidelity"] == HrvFidelity.RAW_STREAM.value


@pytest.mark.golden
def test_rmssd_independent_recomputation() -> None:
    """A second, independent RMSSD computation (pure Python) agrees (TEST-R4)."""
    diffs = [
        GOLDEN_NN_MS[i + 1] - GOLDEN_NN_MS[i] for i in range(len(GOLDEN_NN_MS) - 1)
    ]
    rmssd = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    result = time_domain_hrv(rr_intervals_ms=GOLDEN_NN_MS, min_duration_s=0.0)
    assert isinstance(result, Computed)
    assert result.value.rmssd_ms == pytest.approx(rmssd, abs=GOLDEN_TOL)


def _hf_sinusoid_tachogram(
    *, mean_rr_ms: float = 1000.0, amp_ms: float = 40.0, f_hz: float = 0.25,
    duration_s: float = 300.0,
) -> list[float]:
    """Synthesize an NN series whose RR is sinusoidally modulated at ``f_hz`` (HF band).

    The modulation frequency 0.25 Hz lies inside the HF band (0.15-0.40 Hz), so a
    correct spectral estimate must place essentially all power in HF.
    """
    t = 0.0
    nn: list[float] = []
    while t < duration_s:
        rr = mean_rr_ms + amp_ms * math.sin(2.0 * math.pi * f_hz * t)
        nn.append(rr)
        t += rr / 1000.0
    return nn


@pytest.mark.golden
@pytest.mark.skipif(not dsp_available(), reason="freq-domain requires scipy.signal DSP")
def test_freq_domain_parseval_and_hf_band_golden() -> None:
    """Parseval identity holds and HF sinusoid lands in the HF band (HRV-R7)."""
    nn = _hf_sinusoid_tachogram()
    result = freq_domain_hrv(rr_intervals_ms=nn, min_duration_s=120.0)
    assert isinstance(result, Computed)
    value = result.value
    assert isinstance(value, FreqDomainHrv)

    # Band powers are non-negative (HRV-R7) and the HF sinusoid dominates the HF band.
    assert value.lf_power >= 0.0
    assert value.hf_power >= 0.0
    assert value.hf_power > 100.0 * value.lf_power  # >>; HF-only modulation

    # Parseval (HRV-R7): total integrated spectral power ~= variance of detrended
    # tachogram. Welch with a Hann window introduces small windowing leakage, hence
    # a loose relative tolerance (spectral, not a closed-form recurrence — ANL-R31).
    total_power = result.quality.extra["total_power"]
    variance = result.quality.extra["detrended_variance"]
    assert isinstance(total_power, float)
    assert isinstance(variance, float)
    assert total_power == pytest.approx(variance, rel=0.10)

    # hrv_spectral_method recorded (HRV-R6) and total_power NOT on the API value.
    assert value.hrv_spectral_method == "welch"
    assert not hasattr(value, "total_power")
