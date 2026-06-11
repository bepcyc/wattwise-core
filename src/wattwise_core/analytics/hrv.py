"""HRV pipeline: RR/NN -> artifact-correct -> time-domain + freq-domain (doc 40 §8).

Heart-rate-variability analysis is a pure, deterministic, fail-closed DSP pipeline
governed by HRV-R0..HRV-R7 (digest §3 / §6). It is **sport-agnostic** (ANL-R11):
RR/NN intervals exist for any sport with a beat-to-beat capture, so HRV is never
``NOT_APPLICABLE_FOR_SPORT``.

Three-way fidelity path (HRV-R0), fixed priority:

1. ``rr_intervals_ms`` series present -> FULL pipeline (mandatory artifact correction
   HRV-R1 -> time-domain HRV-R3 -> freq-domain HRV-R5/R6 where the DSP capability
   exists). Fidelity ``raw_stream``.
2. summary-only scalars (``rmssd``/``sdnn``/``pnn50``, no series) -> surface at
   ``fidelity=summary_only``; NO fabricated intervals, NO artifact correction, NO
   freq-domain. Any series-requiring metric -> ``Unavailable(MISSING_REQUIRED_INPUT)``.
3. neither -> ``Unavailable(MISSING_REQUIRED_INPUT)``, NEVER 0 / empty-numeric / placeholder.

Mandatory artifact-correction stage runs FIRST on the RR series (HRV-R1): a
percentage/adaptive-threshold filter flags ectopic/missed/extra beats and replaces
them, producing the artifact-corrected normal-to-normal (NN) series. The corrected
fraction is reported in the QualityReport; if it exceeds the declared ceiling
(default 5%, HRV-R2) the time-/freq-domain results are ``Unavailable(INSUFFICIENT_DATA)``.
A usable recording of at least the declared minimum duration (default 2 min, HRV-R4)
is required, else ``Unavailable(INSUFFICIENT_DATA)``.

Freq-domain band powers (LF 0.04-0.15 Hz, HF 0.15-0.40 Hz, LF/HF) require a DSP stack
(``scipy.signal``). That capability is detected at init (ANL-R34 / DEP-R1/R2/R3): if
``scipy.signal`` is unavailable the freq-domain result is ``Unavailable(MISSING_DEPENDENCY)``
and NEVER ``LF=0, HF=0`` (HRV-R5). The tachogram is resampled to an evenly-spaced grid
(default 4 Hz, HRV-R6) and a Welch periodogram is integrated over the bands; total
spectral power is an INTERNAL Parseval-consistency quantity (HRV-R7), not API-exposed.
The spectral method + detrending are recorded in ``hrv_spectral_method``
(``welch``|``lomb_scargle``) — disjoint from the canonical daily-wellness ``hrv_method``
time-domain tag (HRV-R6 name-collision rule).

Requirement IDs implemented here: HRV-R0, HRV-R1, HRV-R2, HRV-R3, HRV-R4, HRV-R5,
HRV-R6, HRV-R7; cross-cutting ANL-R2/R3/R4/R5/R30/R32/R33/R34, DEP-R1/R2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from wattwise_core.analytics._hrv_core import (
    HRV_ECTOPIC_THRESHOLD_FRAC,
    FloatArray,
    HrvFidelity,
    _correct_artifacts,
)
from wattwise_core.analytics.constants import (
    HRV_ARTIFACT_CEILING_FRAC,
    HRV_MIN_DURATION_S,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)

# ANL-R11: sport-agnostic — HRV is computed from canonical RR/NN (or summary) data for
# a recording, independent of any particular sport; ``None`` is the declared
# sport-agnostic marker (never NOT_APPLICABLE_FOR_SPORT; only HRV-R0 fidelity paths).
APPLICABLE_SPORTS: None = None

# DSP capability is detected ONCE at import (ANL-R34 / DEP-R3): if scipy.signal is
# importable the engine has the freq-domain capability; otherwise freq-domain
# metrics fail closed with MISSING_DEPENDENCY (HRV-R5), never zero-filled bands.
try:  # pragma: no cover - environment-dependent capability probe
    import scipy.signal as _scipy_signal  # type: ignore[import-untyped]

    _DSP_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without scipy
    _scipy_signal = None
    _DSP_AVAILABLE = False


def dsp_available() -> bool:
    """True iff the runtime DSP capability (``scipy.signal``) is present (ANL-R34)."""
    return _DSP_AVAILABLE


def _dsp_signal_module() -> Any:
    """The once-probed ``scipy.signal`` module, or ``None`` (ANL-R34 / DEP-R3).

    Accessor for the sibling freq-domain module so it shares this module's single
    capability probe rather than re-importing scipy. Returns ``Any`` because scipy is
    an untyped dependency; the caller capability-gates on :func:`dsp_available` first.
    """
    return _scipy_signal


@dataclass(frozen=True, slots=True)
class TimeDomainHrv:
    """Time-domain HRV metrics from artifact-corrected NN intervals, ms (HRV-R3)."""

    rmssd_ms: float
    sdnn_ms: float
    pnn50_pct: float
    mean_nn_ms: float

    def to_jsonable(self) -> dict[str, object]:
        return {
            "rmssd_ms": self.rmssd_ms,
            "sdnn_ms": self.sdnn_ms,
            "pnn50_pct": self.pnn50_pct,
            "mean_nn_ms": self.mean_nn_ms,
        }


def _time_domain(nn_ms: FloatArray) -> TimeDomainHrv:
    """Compute RMSSD / SDNN / pNN50 / meanNN from NN intervals (HRV-R3).

    ``RMSSD = sqrt(mean((NN[i+1] - NN[i])^2))``; SDNN = population stdev of NN;
    pNN50 = % of successive NN differences > 50 ms; meanNN = mean NN. Pure.
    """
    diffs = np.diff(nn_ms)
    rmssd = float(np.sqrt(np.mean(diffs * diffs)))
    sdnn = float(np.std(nn_ms))  # population SD (ddof=0), the canonical SDNN
    pnn50 = float(np.mean(np.abs(diffs) > 50.0) * 100.0)
    mean_nn = float(np.mean(nn_ms))
    return TimeDomainHrv(rmssd_ms=rmssd, sdnn_ms=sdnn, pnn50_pct=pnn50, mean_nn_ms=mean_nn)


def _time_domain_from_series(
    rr_intervals_ms: list[float],
    *,
    sport: str | None,
    artifact_ceiling_frac: float,
    min_duration_s: float,
    ectopic_threshold_frac: float,
) -> MetricResult[TimeDomainHrv]:
    """Path 1: RR/NN series -> full pipeline (HRV-R1/R2/R3/R4), highest fidelity.

    Mandatory artifact correction runs FIRST (HRV-R1); gate on corrected fraction
    (HRV-R2) and usable duration / >= 2 NN samples (HRV-R4); then compute the
    time-domain metrics from the NN series. Fails closed (ANL-R4): non-finite/
    non-positive RR ⇒ ``OUT_OF_DOMAIN``; over-artifact / too-short ⇒
    ``INSUFFICIENT_DATA``; non-finite metric ⇒ ``OUT_OF_DOMAIN``.
    """
    rr = np.asarray(rr_intervals_ms, dtype=np.float64)
    if not np.all(np.isfinite(rr)) or np.any(rr <= 0.0):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "rr_intervals_ms must be finite and strictly positive",
        )

    correction = _correct_artifacts(rr, threshold_frac=ectopic_threshold_frac)
    # HRV-R2: over-artifact -> INSUFFICIENT_DATA (never silently compute).
    if correction.corrected_fraction > artifact_ceiling_frac:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"corrected interval fraction {correction.corrected_fraction:.3f} "
            f"exceeds ceiling {artifact_ceiling_frac:.3f} (HRV-R2)",
        )
    nn = correction.nn_ms
    # HRV-R4: usable duration = sum of NN intervals (ms -> s) and >= 2 NN samples.
    usable_s = float(np.sum(nn)) / 1000.0
    if nn.size < 2 or usable_s < min_duration_s:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"usable recording {usable_s:.1f}s below minimum {min_duration_s:.1f}s (HRV-R4)",
        )

    td = _time_domain(nn)
    if not _all_finite(td):
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, "non-finite time-domain HRV value")
    quality = QualityReport(
        coverage_fraction=1.0,
        confidence=1.0 - correction.corrected_fraction,
        extra={
            "fidelity": HrvFidelity.RAW_STREAM.value,
            "corrected_fraction": correction.corrected_fraction,
            "corrected_count": correction.corrected_count,
            "total_beats": correction.total_beats,
            "usable_duration_s": usable_s,
            "artifact_ceiling_frac": artifact_ceiling_frac,
        },
    )
    lineage = InputLineage(
        sport=sport,
        channels=("rr_intervals_ms",),
        reference_params={"fidelity": HrvFidelity.RAW_STREAM.value},
    )
    return Computed(value=td, quality=quality, provenance=lineage)


def _time_domain_from_summary(
    *,
    summary_rmssd_ms: float | None,
    summary_sdnn_ms: float | None,
    summary_pnn50_pct: float | None,
    summary_mean_nn_ms: float | None,
    sport: str | None,
) -> MetricResult[TimeDomainHrv]:
    """Path 2: device-computed summary scalars -> surface at ``summary_only`` fidelity.

    NO artifact correction and NO fabricated intervals (HRV-R0). WHICHEVER summary variant
    the source supplied per ``hrv_method`` (``hrv_rmssd_ms``, ``hrv_sdnn_ms``, or
    ``hrv_pnn50_pct``) is surfaced in its OWN statistic/unit — an SDNN-only or pNN50-only
    summary is a valid ``summary_only`` tier, never forced unavailable (ANL-T-R1.8).
    Every SUPPLIED scalar must be finite & non-negative, else ``OUT_OF_DOMAIN``; absent
    scalars are surfaced as NaN (not zero / placeholder).
    """
    supplied = {
        "hrv_rmssd_ms": summary_rmssd_ms,
        "hrv_sdnn_ms": summary_sdnn_ms,
        "hrv_pnn50_pct": summary_pnn50_pct,
    }
    channels = tuple(name for name, value in supplied.items() if value is not None)
    if not channels:  # defensive: time_domain_hrv only enters Path 2 with >=1 scalar
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT, "no hrv summary scalar supplied"
        )
    for name, value in supplied.items():
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            return Unavailable(
                UnavailableReason.OUT_OF_DOMAIN, f"summary {name} must be finite and >= 0"
            )
    rmssd = float(summary_rmssd_ms) if summary_rmssd_ms is not None else math.nan
    sdnn = float(summary_sdnn_ms) if summary_sdnn_ms is not None else math.nan
    pnn50 = float(summary_pnn50_pct) if summary_pnn50_pct is not None else math.nan
    mean_nn = float(summary_mean_nn_ms) if summary_mean_nn_ms is not None else math.nan
    td = TimeDomainHrv(rmssd_ms=rmssd, sdnn_ms=sdnn, pnn50_pct=pnn50, mean_nn_ms=mean_nn)
    quality = QualityReport(
        coverage_fraction=1.0,
        confidence=1.0,
        extra={
            "fidelity": HrvFidelity.SUMMARY_ONLY.value,
            "corrected_fraction": None,
        },
    )
    lineage = InputLineage(
        sport=sport,
        channels=channels,
        reference_params={"fidelity": HrvFidelity.SUMMARY_ONLY.value},
    )
    return Computed(value=td, quality=quality, provenance=lineage)


def time_domain_hrv(
    *,
    rr_intervals_ms: list[float] | None = None,
    summary_rmssd_ms: float | None = None,
    summary_sdnn_ms: float | None = None,
    summary_pnn50_pct: float | None = None,
    summary_mean_nn_ms: float | None = None,
    sport: str | None = None,
    artifact_ceiling_frac: float = HRV_ARTIFACT_CEILING_FRAC,
    min_duration_s: float = HRV_MIN_DURATION_S,
    ectopic_threshold_frac: float = HRV_ECTOPIC_THRESHOLD_FRAC,
) -> MetricResult[TimeDomainHrv]:
    """Time-domain HRV (RMSSD/SDNN/pNN50/meanNN) via the three-way path (HRV-R0/R3).

    Path 1 (``rr_intervals_ms`` present): mandatory artifact correction FIRST
    (HRV-R1), gate on corrected fraction (HRV-R2) and usable duration (HRV-R4), then
    compute the time-domain metrics from the NN series. Path 2 (summary scalars,
    no series): surface the device-computed summary at ``fidelity=summary_only`` with
    NO correction. Path 3 (neither): ``Unavailable(MISSING_REQUIRED_INPUT)``,
    never zeros. Fail-closed (ANL-R4); pure (ANL-R2).
    """
    # --- Path 1: RR/NN series -> full pipeline (highest fidelity) ---
    if rr_intervals_ms is not None and len(rr_intervals_ms) > 0:
        return _time_domain_from_series(
            rr_intervals_ms,
            sport=sport,
            artifact_ceiling_frac=artifact_ceiling_frac,
            min_duration_s=min_duration_s,
            ectopic_threshold_frac=ectopic_threshold_frac,
        )

    # --- Path 2: summary-only scalars -> surface, NO correction/freq-domain ---
    # ANY supplied variant (RMSSD, SDNN, or pNN50) enters the summary tier (ANL-T-R1.8):
    # an SDNN-only or pNN50-only summary is surfaced in its own statistic, never forced
    # to MISSING_REQUIRED_INPUT just because RMSSD is absent.
    if any(s is not None for s in (summary_rmssd_ms, summary_sdnn_ms, summary_pnn50_pct)):
        return _time_domain_from_summary(
            summary_rmssd_ms=summary_rmssd_ms,
            summary_sdnn_ms=summary_sdnn_ms,
            summary_pnn50_pct=summary_pnn50_pct,
            summary_mean_nn_ms=summary_mean_nn_ms,
            sport=sport,
        )

    # --- Path 3: neither -> MISSING_REQUIRED_INPUT, never zeros (HRV-R0) ---
    return Unavailable(
        UnavailableReason.MISSING_REQUIRED_INPUT,
        "no rr_intervals_ms series and no hrv summary scalar (HRV-R0)",
    )


def _all_finite(td: TimeDomainHrv) -> bool:
    return all(math.isfinite(x) for x in (td.rmssd_ms, td.sdnn_ms, td.pnn50_pct, td.mean_nn_ms))


# Freq-domain HRV (HRV-R5/R6/R7) lives in the sibling :mod:`hrv_freq` module for the
# module size ceiling (QUAL-R9); its public names are re-exported here so callers and
# tests can keep importing them from ``wattwise_core.analytics.hrv`` unchanged. The
# import is at the bottom (after the DSP probe + ``dsp_available`` are defined) so the
# sibling can read this module's capability state without a circular-import hazard.
from wattwise_core.analytics.hrv_freq import (  # noqa: E402
    FreqDomainHrv,
    SpectralMethod,
    freq_domain_hrv,
)

__all__ = [
    "APPLICABLE_SPORTS",
    "FreqDomainHrv",
    "HrvFidelity",
    "SpectralMethod",
    "TimeDomainHrv",
    "dsp_available",
    "freq_domain_hrv",
    "time_domain_hrv",
]
