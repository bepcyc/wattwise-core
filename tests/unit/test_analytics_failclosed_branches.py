"""Fail-closed / edge branches across the analytics package (ANL-R4/R32, doc 40 §6).

Targets the uncovered defensive branches of the pure metric functions: non-finite
inputs and overflowing intermediates must become a typed :class:`Unavailable`
(``OUT_OF_DOMAIN``), never a NaN/Inf inside a ``Computed``; degenerate inputs hit the
exact typed reason (``INSUFFICIENT_DATA`` / ``MISSING_REQUIRED_INPUT`` / ``NOT_SEEDED``);
caller defects (bad shapes, bad windows, bad keys) raise. Also exercises the
``to_jsonable`` envelopes the API serializes. Everything here is pure and offline.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pytest

from wattwise_core.analytics import hrv as hrv_mod
from wattwise_core.analytics import hrv_freq as hf
from wattwise_core.analytics import load_resolution as lr
from wattwise_core.analytics._hrv_core import _correct_artifacts
from wattwise_core.analytics.cp import _fit_cp_model, _ols_standard_errors, cp_wprime
from wattwise_core.analytics.decoupling import (
    _decoupling_pct,
    _half_efficiency,
    _HalfEfficiencies,
    aerobic_decoupling,
)
from wattwise_core.analytics.hrv import TimeDomainHrv, _time_domain_from_series
from wattwise_core.analytics.load_resolution import resolve_hr_load
from wattwise_core.analytics.mmp_cp import _exact_window_peak
from wattwise_core.analytics.np_if_tss import (
    NormalizedPowerValue,
    intensity_factor,
    normalized_power,
    power_tss,
)
from wattwise_core.analytics.pmc import (
    PmcSeed,
    _align_coverage,
    _normalize_input,
    _pmc_day_result,
    _resolve_seed,
    _resolve_window,
    _validate_loads,
    pmc,
    windowed_equiv_tol,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    QualityReport,
    Unavailable,
    UnavailableReason,
)
from wattwise_core.analytics.series import Stream, trailing_rolling_mean
from wattwise_core.analytics.wbal import wbal

pytestmark = pytest.mark.unit

_HUGE = 1.7e308


# ----------------------------------------------------------------------- result.py


def test_result_envelopes_serialize_to_jsonable_dicts() -> None:
    """ANL-R3/R5: Computed/Unavailable/Quality/Lineage all serialize without recompute."""
    quality = QualityReport(coverage_fraction=0.5, gap_count=2, extra={"r2": 0.9})
    lineage = InputLineage(sport="cycling", activity_ids=("a1",), channels=("power",))
    computed = Computed(value=42.0, quality=quality, provenance=lineage)
    out = computed.to_jsonable()
    assert out["available"] is True
    assert out["value"] == 42.0
    quality_json = out["quality"]
    assert isinstance(quality_json, dict)
    assert quality_json["coverage_fraction"] == 0.5
    assert quality_json["r2"] == 0.9  # extra is flattened in
    lineage_json = out["provenance"]
    assert isinstance(lineage_json, dict)
    assert lineage_json["activity_ids"] == ["a1"]
    missing = Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, "no ftp")
    assert missing.to_jsonable() == {
        "available": False,
        "reason": "missing_required_input",
        "detail": "no ftp",
    }


# ----------------------------------------------------------------------- series.py


def test_stream_validation_rejects_malformed_arrays() -> None:
    """ANL-R7: shape mismatch, non-1-D arrays, and time regressions are caller defects."""
    t = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="same shape"):
        Stream(t_seconds=t, values=np.array([1.0]))
    two_d = np.zeros((2, 2))
    with pytest.raises(ValueError, match="1-D"):
        Stream(t_seconds=two_d, values=two_d)
    with pytest.raises(ValueError, match="non-decreasing"):
        Stream(t_seconds=np.array([1.0, 0.0]), values=np.array([1.0, 2.0]))


def test_trailing_rolling_mean_rejects_non_positive_window() -> None:
    """NP-R2: a non-positive rolling window is a caller defect, not a silent no-op."""
    with pytest.raises(ValueError, match="positive"):
        trailing_rolling_mean(np.array([1.0, 2.0]), 0)


# ------------------------------------------------------------------- _hrv_core.py


def test_artifact_correction_degenerate_series_never_divide_by_zero() -> None:
    """HRV-R1: empty / single-beat / all-zero RR series correct nothing and never crash."""
    empty = _correct_artifacts(np.array([], dtype=np.float64))
    assert empty.total_beats == 0
    assert empty.corrected_fraction == 0.0
    single = _correct_artifacts(np.array([800.0]))
    assert single.corrected_count == 0  # no neighbours -> nothing to test against
    zeros = _correct_artifacts(np.zeros(5))
    assert zeros.corrected_count == 0  # zero baseline -> comparison skipped


# ------------------------------------------------------------------------- hrv.py


def test_time_domain_hrv_serializes_and_overflow_fails_closed() -> None:
    """ANL-R32: an overflowing RMSSD becomes OUT_OF_DOMAIN, never an Inf in a Computed."""
    td = TimeDomainHrv(rmssd_ms=40.0, sdnn_ms=50.0, pnn50_pct=10.0, mean_nn_ms=800.0)
    assert td.to_jsonable()["rmssd_ms"] == 40.0
    result = _time_domain_from_series(
        [_HUGE, 1.0, _HUGE, 1.0],
        sport=None,
        artifact_ceiling_frac=1.0,
        min_duration_s=0.0,
        ectopic_threshold_frac=_HUGE,
    )
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


# -------------------------------------------------------------------- hrv_freq.py


def test_freq_domain_envelope_serializes() -> None:
    """HRV-R6: the freq-domain value payload serializes its spectral-method tag."""
    fd = hf.FreqDomainHrv(lf_power=1.0, hf_power=2.0, lf_hf_ratio=0.5, hrv_spectral_method="welch")
    assert fd.to_jsonable()["hrv_spectral_method"] == "welch"


def test_resample_tachogram_sub_grid_recording_is_empty() -> None:
    """HRV-R6: a recording shorter than one resample step yields an empty grid."""
    assert hf._resample_tachogram(np.array([1000.0]), 0.5).size == 0


def test_band_power_without_dsp_capability_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """HRV-R5: the spectral kernel refuses to run without scipy.signal (fail closed)."""
    monkeypatch.setattr(hrv_mod, "_scipy_signal", None)
    with pytest.raises(RuntimeError, match="fail closed"):
        hf._band_power(np.zeros(8), 4.0, (0.04, 0.15), (0.15, 0.4))


def test_band_power_tiny_signal_and_out_of_range_band_integrate_to_zero() -> None:
    """HRV-R5/R6: nperseg degrades to n for tiny signals; an empty band mask is 0.0."""
    spec = hf._band_power(np.array([1.0, -1.0, 1.0]), 4.0, (10.0, 11.0), (11.0, 12.0))
    assert spec.lf_power == 0.0
    assert spec.hf_power == 0.0


def test_corrected_nn_for_spectrum_gates_fail_closed() -> None:
    """HRV-R1/R2/R4: bad RR, over-artifact, and too-few-NN each hit their typed reason."""
    bad = hf._corrected_nn_for_spectrum(
        [-1.0, 800.0],
        artifact_ceiling_frac=0.1,
        min_duration_s=0.0,
        ectopic_threshold_frac=0.2,
    )
    assert isinstance(bad, Unavailable)
    assert bad.reason is UnavailableReason.OUT_OF_DOMAIN
    over = hf._corrected_nn_for_spectrum(
        [800.0] * 10 + [3000.0] * 3,
        artifact_ceiling_frac=0.0,
        min_duration_s=0.0,
        ectopic_threshold_frac=0.2,
    )
    assert isinstance(over, Unavailable)
    assert over.reason is UnavailableReason.INSUFFICIENT_DATA
    few = hf._corrected_nn_for_spectrum(
        [800.0, 810.0, 805.0],
        artifact_ceiling_frac=1.0,
        min_duration_s=0.0,
        ectopic_threshold_frac=0.2,
    )
    assert isinstance(few, Unavailable)
    assert few.reason is UnavailableReason.INSUFFICIENT_DATA


def test_band_powers_validation_fails_closed() -> None:
    """HRV-R7/ANL-R32: negative power, zero HF, and an overflowing ratio fail closed."""
    negative = hf._band_powers_to_result(
        hf._SpectralResult(lf_power=-1.0, hf_power=1.0, total_power=0.0)
    )
    assert isinstance(negative, Unavailable)
    assert negative.reason is UnavailableReason.OUT_OF_DOMAIN
    zero_hf = hf._band_powers_to_result(
        hf._SpectralResult(lf_power=1.0, hf_power=0.0, total_power=1.0)
    )
    assert isinstance(zero_hf, Unavailable)
    assert zero_hf.reason is UnavailableReason.INSUFFICIENT_DATA
    overflow = hf._band_powers_to_result(
        hf._SpectralResult(lf_power=_HUGE, hf_power=1e-308, total_power=1.0)
    )
    assert isinstance(overflow, Unavailable)
    assert overflow.reason is UnavailableReason.OUT_OF_DOMAIN


def test_freq_domain_public_gates_propagate_typed_reasons() -> None:
    """HRV-R2/R4: the public freq-domain entry propagates each gate's exact reason."""
    bad_rr = hf.freq_domain_hrv(rr_intervals_ms=[-1.0, 800.0])
    assert isinstance(bad_rr, Unavailable)
    assert bad_rr.reason is UnavailableReason.OUT_OF_DOMAIN
    # 4 short beats resample to fewer than 4 grid samples -> spectral gate fails closed.
    short = hf.freq_domain_hrv(rr_intervals_ms=[100.0, 110.0, 105.0, 100.0], min_duration_s=0.0)
    assert isinstance(short, Unavailable)
    assert short.reason is UnavailableReason.INSUFFICIENT_DATA
    # A perfectly constant tachogram has zero HF power: the ratio is undefined.
    flat = hf.freq_domain_hrv(rr_intervals_ms=[800.0] * 300, min_duration_s=0.0)
    assert isinstance(flat, Unavailable)
    assert flat.reason is UnavailableReason.INSUFFICIENT_DATA


# ----------------------------------------------------------------- load_bundle.py


def test_ratio_metric_denominator_domain_and_overflow_fail_closed() -> None:
    """LM-R1/ANL-R32: non-positive denominators and overflowing ratios fail closed."""
    from wattwise_core.analytics.load_bundle import _ratio_metric  # noqa: PLC0415 - cycle-break

    base = Computed(
        NormalizedPowerValue(np_w=_HUGE, avg_power_w=200.0, mean_r_w=200.0, analysis_window_s=60)
    )
    negative = _ratio_metric(base, -1.0, denom_name="ftp_w")
    assert isinstance(negative, Unavailable)
    assert negative.reason is UnavailableReason.OUT_OF_DOMAIN
    overflow = _ratio_metric(base, 1e-308, denom_name="ftp_w")
    assert isinstance(overflow, Unavailable)
    assert overflow.reason is UnavailableReason.OUT_OF_DOMAIN


def test_tss_per_hour_duration_domain_and_overflow_fail_closed() -> None:
    """LM-R1/ANL-R32: a non-positive duration and an overflowing rate fail closed."""
    from wattwise_core.analytics.load_bundle import _tss_per_hour  # noqa: PLC0415 - cycle-break

    zero_duration = _tss_per_hour(Computed(100.0), 0)
    assert isinstance(zero_duration, Unavailable)
    assert zero_duration.reason is UnavailableReason.OUT_OF_DOMAIN
    overflow = _tss_per_hour(Computed(_HUGE), 1)
    assert isinstance(overflow, Unavailable)
    assert overflow.reason is UnavailableReason.OUT_OF_DOMAIN


# ------------------------------------------------------------- load_resolution.py


def test_resolve_hr_load_without_hr_channel_is_absent() -> None:
    """LM-R2: no HR channel at all -> None (an absent HR load, not an Unavailable)."""
    assert resolve_hr_load(None, 190.0, 50.0, "m", preferred_load_model=None) is None


def test_resolve_hr_load_prefers_an_applicable_zonal_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOAD-R4: when the zonal model IS applicable the preferred result is returned."""
    zonal = Computed(123.0, quality=QualityReport(extra={"load_model": "hr_load_zonal"}))
    monkeypatch.setattr(lr, "_zonal_hr_load", lambda: zonal)
    hr = Stream.from_values([120.0] * 120)
    result = resolve_hr_load(hr, 190.0, 50.0, "m", preferred_load_model="hr_load_zonal")
    assert result is zonal


# ------------------------------------------------------------------------ mmp_cp.py


def test_exact_window_peak_degenerate_durations_yield_none() -> None:
    """MMP-R1: a non-positive duration or a window longer than the series is None."""
    csum_valid = np.array([0.0, 1.0, 2.0])
    csum_vals = np.array([0.0, 100.0, 200.0])
    assert _exact_window_peak(csum_valid, csum_vals, 2, 0) is None
    assert _exact_window_peak(csum_valid, csum_vals, 2, 3) is None


# --------------------------------------------------------------------- np_if_tss.py


def test_normalized_power_overflow_fails_closed() -> None:
    """ANL-R32: a fourth-power overflow becomes OUT_OF_DOMAIN, never an Inf NP."""
    result = normalized_power(Stream.from_values([1e100] * 60))
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


def test_intensity_factor_and_tss_overflow_fail_closed() -> None:
    """IF-R1/TSS-R1/ANL-R32: a denormal FTP overflows IF/TSS into a typed refusal."""
    np_result = Computed(
        NormalizedPowerValue(np_w=200.0, avg_power_w=180.0, mean_r_w=190.0, analysis_window_s=600)
    )
    if_overflow = intensity_factor(np_result, 1e-308)
    assert isinstance(if_overflow, Unavailable)
    assert if_overflow.reason is UnavailableReason.OUT_OF_DOMAIN
    tss_overflow = power_tss(np_result, 1e-150, 10**9)
    assert isinstance(tss_overflow, Unavailable)
    assert tss_overflow.reason is UnavailableReason.OUT_OF_DOMAIN


# -------------------------------------------------------------------------- cp.py


def test_cp_fit_gates_fail_closed() -> None:
    """CP-R2/R3/R4: non-finite points are excluded; degenerate/noisy fits are typed."""
    # Degenerate OLS spread: 2 interpolating points have zero estimable error.
    se = _ols_standard_errors(np.array([180.0, 720.0]), np.array([0.0, 0.0]), 2)
    assert se == (0.0, 0.0)
    # A NaN/Inf power must never enter the fit (the remaining set is then too small).
    too_few = cp_wprime({180: math.inf, 360: 300.0, 720: float("nan")})
    assert isinstance(too_few, Unavailable)
    assert too_few.reason is UnavailableReason.INSUFFICIENT_DATA
    # Identical work at every duration: no determinable slope -> POOR_FIT.
    flat = _fit_cp_model([180, 360, 720], {180: 400.0, 360: 200.0, 720: 100.0}, r2_min=0.95)
    assert isinstance(flat, Unavailable)
    assert flat.reason is UnavailableReason.POOR_FIT
    # Overflowing work values produce non-finite fit parameters -> OUT_OF_DOMAIN.
    overflow = _fit_cp_model([180, 360, 720], {180: 1e306, 360: 5e305, 720: 9e305}, r2_min=0.95)
    assert isinstance(overflow, Unavailable)
    assert overflow.reason is UnavailableReason.OUT_OF_DOMAIN
    # A wildly non-linear point set fails the R-squared gate -> POOR_FIT.
    noisy = _fit_cp_model(
        [180, 360, 540, 720], {180: 100.0, 360: 900.0, 540: 50.0, 720: 800.0}, r2_min=0.95
    )
    assert isinstance(noisy, Unavailable)
    assert noisy.reason is UnavailableReason.POOR_FIT


# ------------------------------------------------------------------------ wbal.py


def test_wbal_input_gates_fail_closed() -> None:
    """WBAL-R4/R6: an absent or non-1-D power stream hits its exact typed reason."""
    missing = wbal(None, 250.0, 20000.0, sport="cycling")
    assert isinstance(missing, Unavailable)
    assert missing.reason is UnavailableReason.MISSING_REQUIRED_INPUT
    two_d = wbal(np.zeros((2, 2)), 250.0, 20000.0, sport="cycling")
    assert isinstance(two_d, Unavailable)
    assert two_d.reason is UnavailableReason.OUT_OF_DOMAIN


def test_wbal_gap_seconds_carry_the_balance_forward() -> None:
    """WBAL-R1: a NaN (gap) second contributes no work and no recovery."""
    power = np.array([300.0, np.nan, 300.0])
    result = wbal(power, 250.0, 20000.0, sport="cycling")
    assert not isinstance(result, Unavailable)
    series = result.value.series
    assert series[1] == series[0]  # the gap second carried the balance forward
    assert result.quality.gap_count == 1


def test_wbal_overflowing_expenditure_fails_closed() -> None:
    """WBAL-R6/ANL-R32: an overflowing deficit never escapes as -Inf in a Computed."""
    result = wbal(np.array([_HUGE, _HUGE, _HUGE]), 250.0, 20000.0, sport="cycling")
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


# ------------------------------------------------------------------ decoupling.py


def test_decoupling_rejects_non_positive_smoothing_window() -> None:
    """DEC-R3: a non-positive smoothing window is a caller defect (ValueError)."""
    s = Stream.from_values([200.0] * 10)
    with pytest.raises(ValueError, match="positive"):
        aerobic_decoupling(s, s, "cycling", smoothing_window_s=0)


def test_decoupling_channels_that_never_overlap_fail_closed() -> None:
    """DEC-R1: output and HR valid on disjoint windows -> MISSING_REQUIRED_INPUT."""
    out_vals: list[float | None] = [200.0] * 100 + [None] * 200
    hr_vals: list[float | None] = [None] * 200 + [150.0] * 100
    result = aerobic_decoupling(
        Stream.from_values(out_vals), Stream.from_values(hr_vals), "cycling", min_duration_s=1
    )
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


def test_decoupling_all_coasting_effort_fails_closed() -> None:
    """DEC-R2: an effort that is 100% coasting has no included second."""
    out = Stream.from_values([0.0] * 600)
    hr = Stream.from_values([140.0] * 600)
    result = aerobic_decoupling(out, hr, "cycling", min_duration_s=60)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.INSUFFICIENT_DATA


def test_decoupling_negative_output_window_fails_closed() -> None:
    """ANL-R32: a non-positive mean smoothed output is OUT_OF_DOMAIN, never a ratio."""
    out = Stream.from_values([-100.0] * 600)
    hr = Stream.from_values([140.0] * 600)
    result = aerobic_decoupling(out, hr, "cycling", min_duration_s=60)
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


def test_half_efficiency_undefined_ratios_yield_none() -> None:
    """DEC-R1/R2: a zero HR mean or an overflowing efficiency is None (not a number)."""
    n = 120
    included = np.ones(n, dtype=bool)
    in_half = np.ones(n, dtype=bool)
    zero_hr = _half_efficiency(np.full(n, 200.0), np.zeros(n), included, in_half)
    assert zero_hr is None
    overflow = _half_efficiency(np.full(n, _HUGE), np.full(n, 1e-308), included, in_half)
    assert overflow is None


def test_decoupling_pct_undefined_or_overflowing_ratio_fails_closed() -> None:
    """DEC-R1/ANL-R32: zero first-half efficiency or an overflowing percent fails closed."""
    halves = _HalfEfficiencies(
        eff_first=0.0, eff_second=1.0, n_first=60, n_second=60, cv=0.1, t_mid=10.0, n_coasting=0
    )
    zero = _decoupling_pct(halves)
    assert isinstance(zero, Unavailable)
    assert zero.reason is UnavailableReason.OUT_OF_DOMAIN
    overflow = _decoupling_pct(
        _HalfEfficiencies(
            eff_first=1e-308,
            eff_second=-_HUGE,
            n_first=60,
            n_second=60,
            cv=0.1,
            t_mid=10.0,
            n_coasting=0,
        )
    )
    assert isinstance(overflow, Unavailable)
    assert overflow.reason is UnavailableReason.OUT_OF_DOMAIN


# ------------------------------------------------------------------------- pmc.py


def test_pmc_empty_mapping_yields_empty_series() -> None:
    """PMC-R6: an empty date-keyed load mapping produces an empty (not fabricated) series."""
    assert pmc({}) == []


def test_pmc_input_validation_raises_on_caller_defects() -> None:
    """PMC-R6/ANL-R32: non-date keys, non-finite loads, and bad windows are defects."""
    with pytest.raises(TypeError, match="must be datetime"):
        _normalize_input({"2026-06-01": 50.0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite"):
        _validate_loads([math.inf])
    with pytest.raises(ValueError, match="out of range"):
        _resolve_window((0, 5), 3)


def test_pmc_coverage_alignment_requires_a_date_keyed_grid() -> None:
    """DEGR-R2: date-keyed coverage needs date-keyed loads; lengths must match the grid."""
    d = _dt.date(2026, 6, 1)
    with pytest.raises(ValueError, match="date-keyed"):
        _align_coverage({d: None}, dates=None, n=1)
    with pytest.raises(ValueError, match="grid length"):
        _align_coverage([None, None], dates=None, n=3)
    aligned = _align_coverage({d: None}, dates=[d, d + _dt.timedelta(days=1)], n=2)
    assert aligned == [None, None]


def test_pmc_seed_resolution_tuple_and_non_finite() -> None:
    """PMC-R3/R5: a tuple seed resolves; a non-finite seed fails closed NOT_SEEDED."""
    assert _resolve_seed((10.0, 5.0), 3) == (10.0, 5.0)
    bad = _resolve_seed(PmcSeed(ctl_prev=math.nan, atl_prev=1.0), 3)
    assert isinstance(bad, Unavailable)
    assert bad.reason is UnavailableReason.NOT_SEEDED


def test_pmc_day_result_non_finite_values_fail_closed() -> None:
    """ANL-R32: a non-finite CTL/ATL/TSB never escapes into a Computed PmcDay."""
    result = _pmc_day_result(
        ctl=math.inf,
        atl=1.0,
        tsb=0.0,
        provisional=False,
        day_index=0,
        local_date=None,
        tau_ctl=42.0,
        tau_atl=7.0,
        sport=None,
        load_coverage=None,
    )
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


def test_windowed_equiv_tol_scales_with_magnitude() -> None:
    """PMC-R4: the windowed-equivalence tolerance is 1e-9 * max(1, |v|)."""
    assert windowed_equiv_tol(0.5) == pytest.approx(1e-9)
    assert windowed_equiv_tol(-2000.0) == pytest.approx(2e-6)
