"""Cycling-power load family: Normalized Power, Intensity Factor, TSS, bundle.

Pure, deterministic, fail-closed metrics (doc 40 §4, Section 7B) returning the typed
:data:`~wattwise_core.analytics.result.MetricResult` envelope, never a bare number.

Requirement IDs implemented here:

- **NP-R1** — ``NP = (mean(R(t)^4))^(1/4)`` where ``R(t)`` is the seeded 30 s trailing
  arithmetic mean of power.
- **NP-R2** — the 30 s rolling mean is seeded (first valid output only once 30 contiguous
  valid seconds exist); a partial window is never treated as a full-window mean.
- **NP-R3** — gaps (``null``/NaN) never contribute as zeros; a window straddling a gap is
  not valid until 30 contiguous valid seconds re-accumulate.
- **NP-R4** — analysis-window invariants: Jensen ``NP >= mean(R(t))``; constant power
  ``c`` ⇒ ``NP == c`` to ``NP_CONSTANT_POWER_TOL``; translation-invariant.
- **NP-R5** — ``<30`` contiguous valid power seconds ⇒ ``INSUFFICIENT_DATA``; no power
  channel ⇒ ``MISSING_REQUIRED_INPUT``.
- **IF-R1** — ``IF = NP / FTP`` (canonical time-effective FTP, ANL-R9); propagate
  Unavailable from NP; never fall back to average power.
- **TSS-R1/R2** — ``TSS = duration_valid_s * NP^2 / (FTP^2 * 3600) * 100``;
  3600 valid-moving seconds @ IF=1.0 ⇒ ``TSS == 100`` to ``TSS_GOLDEN_TOL`` even though NP
  first becomes valid at 30 s, because ``duration_valid_s`` is the WHOLE-effort valid-moving
  count (incl. the first 29 s ramp), distinct from the NP analysis window.
- **LM-R1/R2/R3** — the per-activity load-metrics bundle (``duration_valid_s``, ``np``,
  ``if``, ``tss``, ``tss_per_hour``, ``efficiency_factor`` = NP/avg_hr, ``variability_index``
  = NP/avg_power, ``intensity_class``, ``load_model``); each field an independent
  ``MetricResult`` that propagates Unavailable and is internally consistent.

Engine-wide contract honoured: pure functions, no I/O / wall-clock / global state
(ANL-R2/R30); typed envelope (ANL-R3); fail-closed with the exact reason (ANL-R4, §6
reason-mapping); no source-name branching (ANL-R1); no NaN/Inf in a ``Computed`` value
(ANL-R32 ⇒ ``OUT_OF_DOMAIN``); cycling-power-specific applicability (ANL-R11, §5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wattwise_core.analytics.constants import (
    INTENSITY_CLASS_CUTS,
    INTENSITY_CLASS_LABELS,
    MAX_INTERP_GAP_S,
    NP_ROLLING_WINDOW_S,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)
from wattwise_core.analytics.series import (
    FloatArray,
    Stream,
    longest_contiguous_valid,
    resample_to_1hz,
    trailing_rolling_mean,
)

# This family is cycling-power-specific (doc 40 §5, ANL-R11): it requires a true
# mechanical-power channel. Declared as metadata, never branched on in formula code.
APPLICABLE_SPORTS: tuple[str, ...] = ("cycling",)

POWER_CHANNEL = "power"
LOAD_MODEL_POWER_TSS = "power_tss"


def _not_applicable_for_sport(metric: str, sport: str) -> Unavailable:
    """Typed sport-mismatch unavailable for a cycling-power-family metric (ANL-R12).

    A power-family metric requested for an ``activity`` whose ``sport`` is not in
    :data:`APPLICABLE_SPORTS` MUST fail closed with ``NOT_APPLICABLE_FOR_SPORT`` —
    distinct from ``MISSING_REQUIRED_INPUT`` (a channel that COULD exist for this sport
    is simply absent) — never a fabricated cross-sport number (ANL-R12, SPORT-T2/T3).
    """
    return Unavailable(
        UnavailableReason.NOT_APPLICABLE_FOR_SPORT,
        f"{metric} is a cycling-power-family metric, not defined for sport {sport!r}",
    )


@dataclass(frozen=True, slots=True)
class NormalizedPowerValue:
    """The value payload of a Computed Normalized Power result.

    ``np_w`` is the Normalized Power in watts (NP-R1). ``avg_power_w`` /
    ``mean_r_w`` are carried for the analysis window so consumers can verify the
    Jensen guarantee and the constant-power identity without recomputation, and so
    the bundle can derive the variability index. ``analysis_window_s`` is the number
    of seconds with a valid (fully-seeded) ``R(t)`` (distinct from
    ``duration_valid_s``; doc 40 §7 note 7).
    """

    np_w: float
    avg_power_w: float
    mean_r_w: float
    analysis_window_s: int


def _resampled_power(power_stream: Stream) -> FloatArray:
    """Resample the power stream to a uniform 1 Hz grid (ANL-R8)."""
    return resample_to_1hz(power_stream, max_interp_gap_s=MAX_INTERP_GAP_S)


def _valid_moving_seconds(power_1hz: FloatArray) -> int:
    """Count valid (non-NaN) 1 Hz power seconds over the WHOLE effort (TSS-R1).

    This is ``duration_valid_s`` — the valid-moving (exercise) duration including the
    first 29 s ramp and any post-gap re-accumulation. It is DISTINCT from the NP
    analysis window (which drops the warm-up seconds where ``R(t)`` is not yet seeded).
    Gaps (NaN) never count.
    """
    if power_1hz.size == 0:
        return 0
    return int(np.count_nonzero(~np.isnan(power_1hz)))


def normalized_power(
    power_stream: Stream, *, sport: str = "cycling"
) -> MetricResult[NormalizedPowerValue]:
    """Normalized Power from a cycling-power stream (NP-R1..R5).

    Pipeline: resample to 1 Hz (ANL-R8) → seeded 30 s trailing mean ``R(t)``
    (NP-R1/R2, gaps never zero per NP-R3) → ``NP = (mean(R^4))^(1/4)`` over the valid
    analysis window.

    ``sport`` is the canonical sport of the activity (ANL-R11): NP is a cycling-power
    metric, so a ``sport`` outside :data:`APPLICABLE_SPORTS` fails closed with
    ``NOT_APPLICABLE_FOR_SPORT`` BEFORE any computation — a power channel carried by a
    non-power sport (e.g. a running power-pod) is never turned into a cycling NP number
    (ANL-R12, SPORT-T2/T3). The gate is on the canonical ``sport`` value, never a source
    name (ANL-R11).

    Fail-closed (ANL-R4, §6):
    - sport outside the cycling-power family ⇒ ``NOT_APPLICABLE_FOR_SPORT`` (ANL-R12);
    - no power channel (empty or all-``null`` stream) ⇒ ``MISSING_REQUIRED_INPUT``;
    - ``<30`` contiguous valid power seconds (so ``R(t)`` is never seeded) ⇒
      ``INSUFFICIENT_DATA`` (NP-R5);
    - a non-finite NP value ⇒ ``OUT_OF_DOMAIN`` (ANL-R32).
    """
    if sport not in APPLICABLE_SPORTS:
        return _not_applicable_for_sport("normalized power", sport)

    power_1hz = _resampled_power(power_stream)

    # No power channel at all: empty stream or every sample is a gap (ANL-R7 ⇒ §6
    # absent-input). Never confuse this with INSUFFICIENT_DATA.
    if power_1hz.size == 0 or not np.any(~np.isnan(power_1hz)):
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "no valid power channel",
        )

    # NP-R3/R5: the 30 s trailing mean is only seeded across 30 *contiguous* valid
    # seconds; a gap > max_interp_gap_s breaks the run. If no run reaches the window
    # length there is no valid R(t) anywhere.
    if longest_contiguous_valid(power_1hz) < NP_ROLLING_WINDOW_S:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"fewer than {NP_ROLLING_WINDOW_S} contiguous valid power seconds",
        )

    rolling = trailing_rolling_mean(power_1hz, NP_ROLLING_WINDOW_S)
    r_valid = rolling[~np.isnan(rolling)]
    if r_valid.size == 0:  # pragma: no cover - guarded by the contiguity check above
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            "no seeded 30 s rolling-mean samples",
        )

    mean_r4 = float(np.mean(np.power(r_valid, 4)))
    np_w = float(mean_r4 ** 0.25)

    if not math.isfinite(np_w):  # ANL-R32: never a NaN/Inf in a Computed value.
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "non-finite normalized power",
        )

    # Average power over the FULL valid-moving window (Coggan VI denominator, doc 40
    # §7B); mean of the seeded R(t) over the analysis window (for the Jensen check).
    power_valid = power_1hz[~np.isnan(power_1hz)]
    avg_power_w = float(np.mean(power_valid))
    mean_r_w = float(np.mean(r_valid))

    valid_moving_s = _valid_moving_seconds(power_1hz)
    quality = QualityReport(
        coverage_fraction=valid_moving_s / power_1hz.size,
        sample_rate_hz=1.0,
        gap_count=int(np.count_nonzero(np.isnan(power_1hz))),
        extra={
            "analysis_window_s": int(r_valid.size),
            "duration_valid_s": valid_moving_s,
            "rolling_window_s": NP_ROLLING_WINDOW_S,
        },
    )
    provenance = InputLineage(channels=(POWER_CHANNEL,))
    return Computed(
        NormalizedPowerValue(
            np_w=np_w,
            avg_power_w=avg_power_w,
            mean_r_w=mean_r_w,
            analysis_window_s=int(r_valid.size),
        ),
        quality=quality,
        provenance=provenance,
    )


def _validated_ftp(ftp_w: float | None) -> float | Unavailable:
    """Return the validated positive FTP, or a typed Unavailable.

    Absent FTP is ``MISSING_REQUIRED_INPUT`` (ANL-R9, §6 absent-input); a present but
    non-positive/non-finite FTP violates the domain precondition ⇒ ``OUT_OF_DOMAIN`` (§6).
    Returning the narrowed ``float`` lets callers use it without a runtime ``assert``.
    """
    if ftp_w is None:
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "no effective FTP",
        )
    if not math.isfinite(ftp_w) or ftp_w <= 0.0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "FTP must be a positive finite value",
        )
    return ftp_w


def intensity_factor(
    np_result: MetricResult[NormalizedPowerValue], ftp_w: float | None
) -> MetricResult[float]:
    """Intensity Factor ``IF = NP / FTP`` (IF-R1).

    Propagates the NP Unavailable verbatim (never recomputed from average power, IF-R1).
    Absent FTP ⇒ ``MISSING_REQUIRED_INPUT``; non-positive FTP ⇒ ``OUT_OF_DOMAIN`` (§6).
    """
    if isinstance(np_result, Unavailable):
        return np_result  # propagate the exact NP Unavailable (IF-R1)

    ftp = _validated_ftp(ftp_w)
    if isinstance(ftp, Unavailable):
        return ftp

    if_value = np_result.value.np_w / ftp
    if not math.isfinite(if_value):  # ANL-R32
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, "non-finite intensity factor")

    quality = QualityReport(
        coverage_fraction=np_result.quality.coverage_fraction,
        sample_rate_hz=np_result.quality.sample_rate_hz,
        gap_count=np_result.quality.gap_count,
        extra={"np_w": np_result.value.np_w, "ftp_w": ftp_w},
    )
    provenance = InputLineage(
        channels=(POWER_CHANNEL,),
        reference_params={"ftp_w": ftp_w},
    )
    return Computed(if_value, quality=quality, provenance=provenance)


def power_tss(
    np_result: MetricResult[NormalizedPowerValue],
    ftp_w: float | None,
    duration_valid_s: int,
) -> MetricResult[float]:
    """Power-based Training Stress Score (TSS-R1/R2).

    ``TSS = duration_valid_s * NP^2 / (FTP^2 * 3600) * 100``.

    ``duration_valid_s`` is the engine-derived valid-moving duration over the WHOLE
    effort (TSS-R1, == LM-R1 ``duration_valid_s``), NOT the NP analysis window: a clean
    3600 s ride yields ``TSS == 100`` even though NP is first valid at 30 s.

    Propagates the NP Unavailable (TSS-R1 family). Absent FTP ⇒ ``MISSING_REQUIRED_INPUT``;
    non-positive FTP or non-positive duration ⇒ ``OUT_OF_DOMAIN``.
    """
    if isinstance(np_result, Unavailable):
        return np_result  # propagate the exact NP Unavailable

    ftp = _validated_ftp(ftp_w)
    if isinstance(ftp, Unavailable):
        return ftp

    if duration_valid_s <= 0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "duration_valid_s must be a positive number of seconds",
        )

    np_w = np_result.value.np_w
    tss = duration_valid_s * np_w * np_w / (ftp * ftp * 3600.0) * 100.0
    if not math.isfinite(tss):  # ANL-R32
        return Unavailable(UnavailableReason.OUT_OF_DOMAIN, "non-finite TSS")

    quality = QualityReport(
        coverage_fraction=np_result.quality.coverage_fraction,
        sample_rate_hz=np_result.quality.sample_rate_hz,
        gap_count=np_result.quality.gap_count,
        extra={
            "np_w": np_w,
            "ftp_w": ftp_w,
            "duration_valid_s": duration_valid_s,
            "load_model": LOAD_MODEL_POWER_TSS,
        },
    )
    provenance = InputLineage(
        channels=(POWER_CHANNEL,),
        reference_params={"ftp_w": ftp_w},
    )
    return Computed(tss, quality=quality, provenance=provenance)


def _intensity_class(if_value: float) -> str:
    """Band the Intensity Factor into the ordered intensity class (LM-R1).

    Cut-points ``INTENSITY_CLASS_CUTS`` (0.55/0.75/0.90/1.05) are half-open lower
    bounds: ``recovery`` for ``IF < 0.55`` up to ``vo2`` for ``IF >= 1.05``. Monotone
    non-decreasing in IF by construction.
    """
    idx = int(np.searchsorted(np.asarray(INTENSITY_CLASS_CUTS), if_value, side="right"))
    return INTENSITY_CLASS_LABELS[idx]


def intensity_class(if_result: MetricResult[float]) -> MetricResult[str]:
    """Intensity class banded from IF (LM-R1); propagates the IF Unavailable."""
    if isinstance(if_result, Unavailable):
        return if_result
    label = _intensity_class(if_result.value)
    return Computed(
        label,
        quality=if_result.quality,
        provenance=if_result.provenance,
    )


# The per-activity load-metrics bundle assembly (LM-R1/R2/R3, LOAD-R4) lives in the focused
# sibling module to keep BOTH modules within the QUAL-R9 size ceiling; re-exported here so the
# canonical ``np_if_tss.load_metrics_bundle`` / ``np_if_tss.LoadMetricsBundle`` entry points are
# unchanged for the service facade and the golden/property tests. Imported at the BOTTOM (after
# every NP/IF/TSS definition above exists) so the one-directional dependency never forms a cycle.
from wattwise_core.analytics.load_bundle import (  # noqa: E402
    LoadMetricsBundle,
    load_metrics_bundle,
)

__all__ = [
    "APPLICABLE_SPORTS",
    "LOAD_MODEL_POWER_TSS",
    "LoadMetricsBundle",
    "NormalizedPowerValue",
    "intensity_class",
    "intensity_factor",
    "load_metrics_bundle",
    "normalized_power",
    "power_tss",
]
