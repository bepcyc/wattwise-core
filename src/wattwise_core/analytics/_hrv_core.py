"""Shared low-level HRV primitives: artifact correction + fidelity tags (doc 40 §8).

Dependency-free core (numpy + stdlib only) imported by BOTH
:mod:`wattwise_core.analytics.hrv` (time-domain) and
:mod:`wattwise_core.analytics.hrv_freq` (freq-domain). Splitting these primitives out
keeps each metric module under the size ceiling (QUAL-R9) AND breaks the import cycle
between the two siblings: this module imports neither of them, so either can be
imported first without a partially-initialised-module hazard.

The public HRV API is unchanged — callers/tests import everything from
:mod:`wattwise_core.analytics.hrv`, which re-exports the names defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# --- artifact-correction config (HRV-R1) -------------------------------------
# Percentage / adaptive-threshold ectopic-beat filter. A beat is flagged when the
# RR interval deviates from the local running median (adaptive baseline) by more
# than this fraction; this is the standard percentage criterion used by
# Kubios-style artifact correction. Default 0.20 (= "very low" Kubios threshold's
# permissive cousin; a common 20% literature value). Externalized (open-core).
HRV_ECTOPIC_THRESHOLD_FRAC: Final = 0.20  # HRV-R1
# Window (in beats) of the running-median baseline the percentage criterion uses.
HRV_ECTOPIC_MEDIAN_WINDOW_BEATS: Final = 5  # HRV-R1


class HrvFidelity(StrEnum):
    """How the HRV inputs were sourced (HRV-R0). Recorded in lineage/quality."""

    RAW_STREAM = "raw_stream"
    """Beat-to-beat RR/NN series -> full pipeline."""
    SUMMARY_ONLY = "summary_only"
    """Device-computed scalar summary only; no series -> no correction/freq-domain."""


@dataclass(frozen=True, slots=True)
class _ArtifactCorrection:
    """Result of the mandatory artifact-correction stage (HRV-R1)."""

    nn_ms: FloatArray
    corrected_fraction: float
    corrected_count: int
    total_beats: int


def _correct_artifacts(
    rr_ms: FloatArray,
    *,
    threshold_frac: float = HRV_ECTOPIC_THRESHOLD_FRAC,
    median_window_beats: int = HRV_ECTOPIC_MEDIAN_WINDOW_BEATS,
) -> _ArtifactCorrection:
    """Percentage / adaptive-threshold ectopic-beat correction RR -> NN (HRV-R1).

    Each RR interval is compared against an adaptive baseline (the running median of
    a small surrounding window of intervals). An interval deviating from that
    baseline by more than ``threshold_frac`` is flagged as an artifact (ectopic /
    missed / extra beat) and REPLACED by the baseline (interpolated normal value),
    yielding the normal-to-normal (NN) series. The corrected fraction is reported so
    HRV-R2 can gate on it. Pure & deterministic (ANL-R2/R30).
    """
    n = rr_ms.size
    nn = rr_ms.astype(np.float64).copy()
    if n == 0:
        return _ArtifactCorrection(nn, 0.0, 0, 0)

    half = max(1, median_window_beats // 2)
    corrected = 0
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        # Baseline excludes the interval under test to avoid self-masking.
        neighbours = np.concatenate((rr_ms[lo:i], rr_ms[i + 1 : hi]))
        if neighbours.size == 0:
            continue
        baseline = float(np.median(neighbours))
        if baseline <= 0.0:
            continue
        if abs(rr_ms[i] - baseline) > threshold_frac * baseline:
            nn[i] = baseline
            corrected += 1
    return _ArtifactCorrection(
        nn_ms=nn,
        corrected_fraction=corrected / n,
        corrected_count=corrected,
        total_beats=n,
    )


__all__ = [
    "HRV_ECTOPIC_MEDIAN_WINDOW_BEATS",
    "HRV_ECTOPIC_THRESHOLD_FRAC",
    "FloatArray",
    "HrvFidelity",
]
