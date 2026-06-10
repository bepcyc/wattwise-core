"""Freq-domain HRV: tachogram resample -> Welch PSD -> LF/HF band powers (doc 40 §8).

This is the frequency-domain half of the HRV pipeline (HRV-R5/R6/R7), split out of
:mod:`wattwise_core.analytics.hrv` for the module size ceiling (QUAL-R9). All public
names remain importable from :mod:`wattwise_core.analytics.hrv` (re-exported there),
so callers and tests are unaffected.

The DSP capability (``scipy.signal``) is owned by :mod:`wattwise_core.analytics.hrv`
(``_DSP_AVAILABLE`` / :func:`~wattwise_core.analytics.hrv.dsp_available`); this module
reads it through the ``hrv`` module object at call time so a runtime patch of
``hrv._DSP_AVAILABLE`` is honoured (ANL-R34 / DEP-R2/R3, HRV-R5). Band powers are
never zero-filled on a missing capability — they fail closed with
``MISSING_DEPENDENCY``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from wattwise_core.analytics._hrv_core import (
    HRV_ECTOPIC_THRESHOLD_FRAC,
    FloatArray,
    HrvFidelity,
    _ArtifactCorrection,
    _correct_artifacts,
)
from wattwise_core.analytics.constants import (
    HRV_ARTIFACT_CEILING_FRAC,
    HRV_HF_BAND_HZ,
    HRV_LF_BAND_HZ,
    HRV_MIN_DURATION_S,
    HRV_TACHOGRAM_RESAMPLE_HZ,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)

# ANL-R11: sport-agnostic — same declaration as the time-domain HRV family (hrv.py).
APPLICABLE_SPORTS: None = None


class SpectralMethod(StrEnum):
    """Spectral estimator recorded in ``hrv_spectral_method`` (HRV-R6).

    Disjoint vocabulary from the canonical daily-wellness ``hrv_method`` time-domain
    variant tag (``rmssd``|``sdnn``|``pnn50``); different owners, never confused.
    """

    WELCH = "welch"
    LOMB_SCARGLE = "lomb_scargle"


@dataclass(frozen=True, slots=True)
class FreqDomainHrv:
    """Freq-domain HRV band powers (HRV-R5/R6). Total power is NOT exposed (HRV-R7)."""

    lf_power: float
    hf_power: float
    lf_hf_ratio: float
    hrv_spectral_method: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "lf_power": self.lf_power,
            "hf_power": self.hf_power,
            "lf_hf_ratio": self.lf_hf_ratio,
            "hrv_spectral_method": self.hrv_spectral_method,
        }


def _resample_tachogram(nn_ms: FloatArray, fs_hz: float) -> FloatArray:
    """Resample the NN tachogram onto an evenly-spaced ``fs_hz`` grid (HRV-R6).

    Cumulative beat-occurrence times are the abscissae; the instantaneous NN value
    (in ms) is the ordinate; a uniform grid is interpolated linearly between beats.
    Pure & deterministic. Returns the detrended (mean-removed) evenly-spaced signal
    so a band integral is a variance contribution (Parseval, HRV-R7).
    """
    # Beat occurrence times (s): the i-th NN spans [t_i, t_{i+1}); place each NN
    # value at the cumulative end-time of that interval.
    t_beats_s = np.cumsum(nn_ms) / 1000.0
    t_beats_s = t_beats_s - t_beats_s[0]  # start at 0
    duration_s = float(t_beats_s[-1])
    grid = np.arange(0.0, duration_s, 1.0 / fs_hz, dtype=np.float64)
    if grid.size == 0:
        return grid
    resampled = np.interp(grid, t_beats_s, nn_ms)
    return resampled - float(np.mean(resampled))  # detrend (remove DC)


@dataclass(frozen=True, slots=True)
class _SpectralResult:
    lf_power: float
    hf_power: float
    total_power: float  # INTERNAL Parseval-check quantity (HRV-R7), never exposed


def _band_power(
    detrended_1d: FloatArray,
    fs_hz: float,
    lf_band: tuple[float, float],
    hf_band: tuple[float, float],
) -> _SpectralResult:
    """Welch PSD integrated over the LF/HF bands (HRV-R5/R6).

    Returns LF, HF and TOTAL band power. Total power is the integral of the PSD over
    the whole frequency axis and is used ONLY for the internal Parseval check
    (HRV-R7: total spectral power ~= variance of the detrended tachogram); it is
    never surfaced in the API result. Requires the DSP capability (HRV-R5).
    """
    # Read the DSP handle from the ``hrv`` module at call time (lazy import avoids an
    # import cycle and honours a runtime patch of the capability).
    from wattwise_core.analytics import hrv as _hrv  # noqa: PLC0415 - cycle-break (intentional)

    scipy_signal = _hrv._dsp_signal_module()
    if scipy_signal is None:  # capability-gated by caller (ANL-R34); defensive
        raise RuntimeError("scipy.signal unavailable; freq-domain must fail closed")
    n = detrended_1d.size
    # Welch with a segment length bounded by signal length; deterministic config
    # (ANL-R30): fixed Hann window, 50% overlap, no random anything.
    nperseg = min(n, 256)
    if nperseg < 4:
        nperseg = n
    freqs, psd = scipy_signal.welch(
        detrended_1d,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    def _integrate(band: tuple[float, float]) -> float:
        mask = (freqs >= band[0]) & (freqs < band[1])
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    lf = _integrate(lf_band)
    hf = _integrate(hf_band)
    total = float(np.trapezoid(psd, freqs))
    return _SpectralResult(lf_power=lf, hf_power=hf, total_power=total)


def _missing_series(summary_present: bool) -> Unavailable:
    """Typed ``MISSING_REQUIRED_INPUT`` for an absent RR series (HRV-R0/DEP-R2).

    Freq-domain is a series-requiring metric (HRV-R0): a summary-only input has no
    spectrum. The detail distinguishes "summary present but no series" from "no input
    at all" so the caller can act on which path failed.
    """
    detail = (
        "freq-domain HRV requires the rr_intervals_ms series; summary_only carries "
        "no spectrum (HRV-R0)"
        if summary_present
        else "no rr_intervals_ms series for freq-domain HRV (HRV-R0)"
    )
    return Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, detail)


def _missing_dsp_capability() -> Unavailable | None:
    """DSP-capability gate (ANL-R34 / HRV-R5), or ``None`` when the capability exists.

    A missing DSP capability fails closed with ``MISSING_DEPENDENCY``, NEVER
    ``LF=0, HF=0`` (HRV-R5). Reads ``hrv._DSP_AVAILABLE`` at call time so a runtime
    patch of the capability is honoured.
    """
    # Lazy import (call time) avoids an import cycle and reads the *current*
    # ``hrv._DSP_AVAILABLE`` so a monkeypatch of the capability is honoured.
    from wattwise_core.analytics import hrv as _hrv  # noqa: PLC0415 - cycle-break (intentional)

    if not _hrv.dsp_available():
        return Unavailable(
            UnavailableReason.MISSING_DEPENDENCY,
            "freq-domain HRV requires scipy.signal (DSP capability absent) — HRV-R5",
        )
    return None


def _corrected_nn_for_spectrum(
    rr_intervals_ms: list[float],
    *,
    artifact_ceiling_frac: float,
    min_duration_s: float,
    ectopic_threshold_frac: float,
) -> tuple[_ArtifactCorrection, FloatArray, float] | Unavailable:
    """Validate, artifact-correct and duration-gate the RR series (HRV-R1/R2/R4).

    Returns ``(correction, nn, usable_s)`` on success, else a typed ``Unavailable``:
    non-finite/non-positive RR ⇒ ``OUT_OF_DOMAIN``; over-artifact ⇒
    ``INSUFFICIENT_DATA`` (HRV-R2); too few NN / too-short recording ⇒
    ``INSUFFICIENT_DATA`` (HRV-R4). Freq-domain needs at least 4 NN samples.
    """
    rr = np.asarray(rr_intervals_ms, dtype=np.float64)
    if not np.all(np.isfinite(rr)) or np.any(rr <= 0.0):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "rr_intervals_ms must be finite and strictly positive",
        )

    correction = _correct_artifacts(rr, threshold_frac=ectopic_threshold_frac)
    if correction.corrected_fraction > artifact_ceiling_frac:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"corrected interval fraction {correction.corrected_fraction:.3f} exceeds "
            f"ceiling {artifact_ceiling_frac:.3f} (HRV-R2)",
        )
    nn = correction.nn_ms
    usable_s = float(np.sum(nn)) / 1000.0
    if nn.size < 4 or usable_s < min_duration_s:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"usable recording {usable_s:.1f}s below minimum {min_duration_s:.1f}s (HRV-R4)",
        )
    return correction, nn, usable_s


def _band_powers_to_result(spec: _SpectralResult) -> tuple[float, float, float] | Unavailable:
    """Validate band powers and the LF/HF ratio (HRV-R7 / ANL-R32), or fail closed.

    Returns ``(lf, hf, lf_hf)``; a non-finite/negative band power ⇒ ``OUT_OF_DOMAIN``;
    a zero HF (undefined ratio) ⇒ ``INSUFFICIENT_DATA``; a non-finite ratio ⇒
    ``OUT_OF_DOMAIN``.
    """
    lf = spec.lf_power
    hf = spec.hf_power
    if not (math.isfinite(lf) and math.isfinite(hf)) or lf < 0.0 or hf < 0.0:
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, "non-finite or negative band power")
    if hf <= 0.0:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            "HF band power is zero; LF/HF ratio undefined",
        )
    lf_hf = lf / hf
    if not math.isfinite(lf_hf):
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, "non-finite LF/HF ratio")
    return lf, hf, lf_hf


def _build_freq_result(
    lf: float,
    hf: float,
    lf_hf: float,
    *,
    detrended: FloatArray,
    spec: _SpectralResult,
    correction: _ArtifactCorrection,
    usable_s: float,
    resample_hz: float,
    lf_band: tuple[float, float],
    hf_band: tuple[float, float],
    sport: str | None,
) -> Computed[FreqDomainHrv]:
    """Assemble the freq-domain ``Computed`` envelope (QualityReport + InputLineage).

    Records the Parseval-consistency quantities (HRV-R7): total band power and the
    variance of the detrended tachogram (the two should agree).
    """
    fd = FreqDomainHrv(
        lf_power=lf,
        hf_power=hf,
        lf_hf_ratio=lf_hf,
        hrv_spectral_method=SpectralMethod.WELCH.value,
    )
    variance = float(np.var(detrended))
    quality = QualityReport(
        coverage_fraction=1.0,
        sample_rate_hz=resample_hz,
        confidence=1.0 - correction.corrected_fraction,
        extra={
            "fidelity": HrvFidelity.RAW_STREAM.value,
            "corrected_fraction": correction.corrected_fraction,
            "usable_duration_s": usable_s,
            "hrv_spectral_method": SpectralMethod.WELCH.value,
            "lf_band_hz": list(lf_band),
            "hf_band_hz": list(hf_band),
            "total_power": spec.total_power,  # internal Parseval-check, not API value
            "detrended_variance": variance,
        },
    )
    lineage = InputLineage(
        sport=sport,
        channels=("rr_intervals_ms",),
        reference_params={
            "fidelity": HrvFidelity.RAW_STREAM.value,
            "hrv_spectral_method": SpectralMethod.WELCH.value,
        },
    )
    return Computed(value=fd, quality=quality, provenance=lineage)


def freq_domain_hrv(
    *,
    rr_intervals_ms: list[float] | None = None,
    summary_present: bool = False,
    sport: str | None = None,
    artifact_ceiling_frac: float = HRV_ARTIFACT_CEILING_FRAC,
    min_duration_s: float = HRV_MIN_DURATION_S,
    ectopic_threshold_frac: float = HRV_ECTOPIC_THRESHOLD_FRAC,
    resample_hz: float = HRV_TACHOGRAM_RESAMPLE_HZ,
    lf_band: tuple[float, float] = HRV_LF_BAND_HZ,
    hf_band: tuple[float, float] = HRV_HF_BAND_HZ,
) -> MetricResult[FreqDomainHrv]:
    """Freq-domain HRV band powers (LF/HF/LF:HF) via Welch on the tachogram (HRV-R5/R6).

    Requires the RR/NN series path (freq-domain is a series-requiring metric, HRV-R0):
    a ``summary_only`` input -> ``Unavailable(MISSING_REQUIRED_INPUT)``, no series at
    all -> ``MISSING_REQUIRED_INPUT``. The DSP capability is checked FIRST after input
    presence (ANL-R34 / HRV-R5): if ``scipy.signal`` is unavailable ->
    ``Unavailable(MISSING_DEPENDENCY)``, NEVER ``LF=0, HF=0``. Artifact-correction
    (HRV-R1) and the HRV-R2/R4 gates run before the spectrum. Band powers are >= 0;
    total power is an internal Parseval check (HRV-R7), never exposed. Pure (ANL-R2).
    """
    # Input presence FIRST (DEP-R2: distinguish missing input from missing capability).
    if rr_intervals_ms is None or len(rr_intervals_ms) == 0:
        return _missing_series(summary_present)

    # DSP capability (ANL-R34 / HRV-R5): never attempt-and-crash, never zero-fill.
    no_dsp = _missing_dsp_capability()
    if no_dsp is not None:
        return no_dsp

    prepared = _corrected_nn_for_spectrum(
        rr_intervals_ms,
        artifact_ceiling_frac=artifact_ceiling_frac,
        min_duration_s=min_duration_s,
        ectopic_threshold_frac=ectopic_threshold_frac,
    )
    if isinstance(prepared, Unavailable):
        return prepared
    correction, nn, usable_s = prepared

    detrended = _resample_tachogram(nn, resample_hz)
    if detrended.size < 4:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            "resampled tachogram too short for spectral estimation (HRV-R4)",
        )

    spec = _band_power(detrended, resample_hz, lf_band, hf_band)
    bands = _band_powers_to_result(spec)
    if isinstance(bands, Unavailable):
        return bands
    lf, hf, lf_hf = bands

    return _build_freq_result(
        lf,
        hf,
        lf_hf,
        detrended=detrended,
        spec=spec,
        correction=correction,
        usable_s=usable_s,
        resample_hz=resample_hz,
        lf_band=lf_band,
        hf_band=hf_band,
        sport=sport,
    )


__all__ = [
    "APPLICABLE_SPORTS",
    "FreqDomainHrv",
    "SpectralMethod",
    "freq_domain_hrv",
]
