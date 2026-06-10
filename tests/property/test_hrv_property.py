"""Property-based tests for HRV (doc 40 §8, TEST-R1/R2/R3).

Covers the per-metric HRV property IDs (digest §2, HRV-T0a..T0c / T1..T6):

* T0a  RR/NN full pipeline produces a typed result, never a bare number.
* T0b  summary-only surfaces at ``fidelity=summary_only``; freq-domain ->
        ``Unavailable(MISSING_REQUIRED_INPUT)`` (series-requiring metric).
* T0c  neither input path -> ``Unavailable(MISSING_REQUIRED_INPUT)``, NEVER zeros.
* T1   artifact-correction runs FIRST: NN derives from RR before any metric (HRV-R1).
* T2   over-artifact (> ceiling) -> ``Unavailable(INSUFFICIENT_DATA)`` (HRV-R2).
* T3   freq-domain missing DSP -> ``Unavailable(MISSING_DEPENDENCY)``, never LF=0/HF=0
        (HRV-R5).
* T4   synthetic sinusoid lands power in the correct band + Parseval ~= variance;
        band powers >= 0 (HRV-R7).
* T5   sub-minimum duration -> ``Unavailable(INSUFFICIENT_DATA)`` (HRV-R4).
* T6   determinism: identical inputs -> identical results (ANL-R30).

Property generators (TEST-R2) produce variable-length NN streams across realistic
physiological ranges and degenerate fail-closed cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from wattwise_core.analytics import hrv as hrv_mod
from wattwise_core.analytics.constants import HRV_ARTIFACT_CEILING_FRAC
from wattwise_core.analytics.hrv import (
    FreqDomainHrv,
    HrvFidelity,
    TimeDomainHrv,
    dsp_available,
    freq_domain_hrv,
    time_domain_hrv,
)
from wattwise_core.analytics.result import (
    Computed,
    Unavailable,
    UnavailableReason,
)

# Physiologically plausible NN intervals: ~300 ms (200 bpm) .. ~1500 ms (40 bpm).
_NN_MS = st.floats(min_value=300.0, max_value=1500.0, allow_nan=False, allow_infinity=False)


def _nn_streams(min_size: int = 2, max_size: int = 400) -> st.SearchStrategy[list[float]]:
    return st.lists(_NN_MS, min_size=min_size, max_size=max_size)


def _long_steady_nn(*, mean_rr_ms: float, jitter_ms: float, n: int, seed: int) -> list[float]:
    """A steady NN series long enough to clear the 2-min gate (deterministic)."""
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-jitter_ms, jitter_ms, size=n)
    return [float(mean_rr_ms + j) for j in jitter]


# --- HRV-T0a: full pipeline returns a typed envelope, never a bare number --------
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(nn=_nn_streams())
def test_time_domain_returns_typed_envelope(nn: list[float]) -> None:
    result = time_domain_hrv(rr_intervals_ms=nn, min_duration_s=0.0)
    assert isinstance(result, Computed | Unavailable)
    if isinstance(result, Computed):
        v = result.value
        assert isinstance(v, TimeDomainHrv)
        # No NaN/Inf escapes into a Computed (ANL-R32).
        assert math.isfinite(v.rmssd_ms)
        assert math.isfinite(v.sdnn_ms)
        assert math.isfinite(v.mean_nn_ms)
        assert math.isfinite(v.pnn50_pct)
        assert v.rmssd_ms >= 0.0
        assert v.sdnn_ms >= 0.0
        assert 0.0 <= v.pnn50_pct <= 100.0


# --- HRV-T1: artifact-correction-first (NN derived from RR before metrics) --------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(nn=_nn_streams(min_size=3))
def test_artifact_correction_runs_first(nn: list[float]) -> None:
    """Computed time-domain always reports a corrected_fraction <= ceiling (HRV-R1/R2)."""
    result = time_domain_hrv(rr_intervals_ms=nn, min_duration_s=0.0)
    if isinstance(result, Computed):
        frac = result.quality.extra["corrected_fraction"]
        assert isinstance(frac, float)
        assert 0.0 <= frac <= HRV_ARTIFACT_CEILING_FRAC
        assert result.quality.extra["fidelity"] == HrvFidelity.RAW_STREAM.value


# --- HRV-T2: over-artifact -> Unavailable(INSUFFICIENT_DATA) ----------------------
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(n_pairs=st.integers(min_value=20, max_value=120))
def test_over_artifact_yields_unavailable(n_pairs: int) -> None:
    """A heavily corrupted RR series (every other beat ectopic) fails closed (HRV-R2)."""
    # Alternate normal/extreme beats so the percentage filter flags ~50% -> > 5% ceiling.
    nn: list[float] = []
    for _ in range(n_pairs):
        nn.append(1000.0)
        nn.append(300.0)  # gross ectopic spike vs ~1000 ms baseline
    result = time_domain_hrv(rr_intervals_ms=nn, min_duration_s=0.0)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA


# --- HRV-T5: sub-minimum duration -> Unavailable(INSUFFICIENT_DATA) ---------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(nn=_nn_streams(min_size=2, max_size=30))
def test_sub_minimum_duration_unavailable(nn: list[float]) -> None:
    """A short recording below the 2-min minimum fails closed, never zeros (HRV-R4)."""
    total_s = sum(nn) / 1000.0
    assume(total_s < 120.0)
    result = time_domain_hrv(rr_intervals_ms=nn, min_duration_s=120.0)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA


# --- HRV-T0c: neither input path -> MISSING_REQUIRED_INPUT, never zeros -----------
def test_neither_path_unavailable_never_zeros() -> None:
    result = time_domain_hrv()
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT
    fresult = freq_domain_hrv()
    assert isinstance(fresult, Unavailable)
    assert fresult.reason is UnavailableReason.MISSING_REQUIRED_INPUT


@settings(max_examples=50)
@given(empty=st.just([]))
def test_empty_series_unavailable(empty: list[float]) -> None:
    """An empty RR list is treated as no input, never an empty-numeric result."""
    result = time_domain_hrv(rr_intervals_ms=empty)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


# --- HRV-T0b: summary-only surfaces; freq-domain -> MISSING_REQUIRED_INPUT ---------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(rmssd=st.floats(min_value=0.0, max_value=300.0, allow_nan=False))
def test_summary_only_surfaces_no_freq(rmssd: float) -> None:
    td = time_domain_hrv(summary_rmssd_ms=rmssd)
    assert isinstance(td, Computed)
    assert td.value.rmssd_ms == pytest.approx(rmssd)
    assert td.quality.extra["fidelity"] == HrvFidelity.SUMMARY_ONLY.value
    assert td.quality.extra["corrected_fraction"] is None  # no correction on summary

    # Any series-requiring metric (freq-domain) -> MISSING_REQUIRED_INPUT.
    fd = freq_domain_hrv(summary_present=True)
    assert isinstance(fd, Unavailable)
    assert fd.reason is UnavailableReason.MISSING_REQUIRED_INPUT


def test_summary_negative_rmssd_out_of_domain() -> None:
    """A present-but-invalid summary scalar -> OUT_OF_DOMAIN, never a zero (ANL-R32)."""
    result = time_domain_hrv(summary_rmssd_ms=-1.0)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


# --- present-but-invalid RR series -> OUT_OF_DOMAIN -------------------------------
@settings(max_examples=50)
@given(bad=st.sampled_from([0.0, -10.0, math.inf, math.nan]))
def test_invalid_rr_values_out_of_domain(bad: float) -> None:
    nn = [800.0, 810.0, bad, 805.0, 800.0]
    result = time_domain_hrv(rr_intervals_ms=nn, min_duration_s=0.0)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


# --- HRV-T3: freq-domain missing DSP -> MISSING_DEPENDENCY, never LF=0/HF=0 --------
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_freq_missing_dsp_yields_missing_dependency(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With DSP capability OFF, freq-domain fails MISSING_DEPENDENCY, never zero bands."""
    nn = _long_steady_nn(mean_rr_ms=900.0, jitter_ms=30.0, n=200, seed=seed)
    monkeypatch.setattr(hrv_mod, "_DSP_AVAILABLE", False)
    assert hrv_mod.dsp_available() is False
    result = freq_domain_hrv(rr_intervals_ms=nn, min_duration_s=120.0)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_DEPENDENCY


def test_dsp_capability_check_precedes_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing capability is reported BEFORE any spectral attempt (ANL-R34/DEP-R2)."""
    nn = _long_steady_nn(mean_rr_ms=1000.0, jitter_ms=20.0, n=180, seed=1)
    monkeypatch.setattr(hrv_mod, "_DSP_AVAILABLE", False)
    result = freq_domain_hrv(rr_intervals_ms=nn, min_duration_s=120.0)
    assert isinstance(result, Unavailable)
    # MISSING_DEPENDENCY (capability), NOT MISSING_REQUIRED_INPUT (the input IS present).
    assert result.reason is UnavailableReason.MISSING_DEPENDENCY


# --- HRV-T4: band powers >= 0 and Parseval ~= variance on real spectra ------------
@pytest.mark.skipif(not dsp_available(), reason="freq-domain requires scipy.signal DSP")
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(
    mean_rr=st.floats(min_value=700.0, max_value=1100.0, allow_nan=False),
    f_hz=st.floats(min_value=0.16, max_value=0.38, allow_nan=False),  # inside HF band
)
def test_freq_band_powers_nonneg_and_hf_dominates(mean_rr: float, f_hz: float) -> None:
    """An HF sinusoid: band powers >= 0 and HF dominates LF; Parseval holds (HRV-R7)."""
    t = 0.0
    nn: list[float] = []
    amp = 35.0
    while t < 300.0:
        rr = mean_rr + amp * math.sin(2.0 * math.pi * f_hz * t)
        nn.append(rr)
        t += rr / 1000.0
    result = freq_domain_hrv(rr_intervals_ms=nn, min_duration_s=120.0)
    assert isinstance(result, Computed)
    v = result.value
    assert isinstance(v, FreqDomainHrv)
    assert v.lf_power >= 0.0
    assert v.hf_power >= 0.0
    assert math.isfinite(v.lf_hf_ratio)
    # Power injected only in HF -> HF strictly dominates LF.
    assert v.hf_power > v.lf_power
    # Parseval (HRV-R7): total integrated power ~= variance of detrended tachogram.
    total_power = result.quality.extra["total_power"]
    variance = result.quality.extra["detrended_variance"]
    assert isinstance(total_power, float)
    assert isinstance(variance, float)
    assert total_power == pytest.approx(variance, rel=0.15)
    # total_power is internal-only; never on the exposed value object (HRV-R7).
    assert not hasattr(v, "total_power")


# --- HRV-T6: determinism (ANL-R30) ------------------------------------------------
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(nn=_nn_streams(min_size=3))
def test_time_domain_deterministic(nn: list[float]) -> None:
    r1 = time_domain_hrv(rr_intervals_ms=nn, min_duration_s=0.0)
    r2 = time_domain_hrv(rr_intervals_ms=nn, min_duration_s=0.0)
    assert type(r1) is type(r2)
    if isinstance(r1, Computed) and isinstance(r2, Computed):
        assert r1.value == r2.value
    elif isinstance(r1, Unavailable) and isinstance(r2, Unavailable):
        assert r1.reason is r2.reason


@pytest.mark.skipif(not dsp_available(), reason="freq-domain requires scipy.signal DSP")
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=5_000))
def test_freq_domain_deterministic(seed: int) -> None:
    nn = _long_steady_nn(mean_rr_ms=950.0, jitter_ms=40.0, n=220, seed=seed)
    r1 = freq_domain_hrv(rr_intervals_ms=nn, min_duration_s=120.0)
    r2 = freq_domain_hrv(rr_intervals_ms=nn, min_duration_s=120.0)
    assert type(r1) is type(r2)
    if isinstance(r1, Computed) and isinstance(r2, Computed):
        assert r1.value == r2.value


pytestmark = pytest.mark.property
