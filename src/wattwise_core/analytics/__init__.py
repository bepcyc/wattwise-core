"""Analytics engine: pure, deterministic, fail-closed endurance metrics (doc 40).

Every metric is a pure function of declared canonical inputs returning a typed
:data:`~wattwise_core.analytics.result.MetricResult` envelope — never a bare number,
never a fabricated value (ANL-R2/R3/R4). The shared contract (result envelope,
1 Hz resampler, constants) is exported here; the per-metric functions live in their
own modules and are aggregated by :mod:`wattwise_core.analytics.service`.
"""

from __future__ import annotations

from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
    is_computed,
)
from wattwise_core.analytics.series import Stream, resample_to_1hz

__all__ = [
    "Computed",
    "InputLineage",
    "MetricResult",
    "QualityReport",
    "Stream",
    "Unavailable",
    "UnavailableReason",
    "is_computed",
    "resample_to_1hz",
]
