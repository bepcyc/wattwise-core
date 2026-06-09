"""Coverage / value / label helpers for the performance chart projections (SCHEMA-R9).

Extracted from the performance router so the route handlers stay within the module-size
ceiling (QUAL-R9). These map a computed/unavailable :class:`MetricResult` into the
chart-ready ``coverage`` descriptor, the typed-null scalar (never a fabricated ``0``,
API-R29), and the jargon-free X-tick labels (API-R21).
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Final

from wattwise_core.analytics.result import (
    MetricResult,
    Unavailable,
    UnavailableReason,
    is_computed,
)
from wattwise_core.api.chart_schemas import CoverageDescriptor

#: Reasons whose absence is a FAILED computation (vs. a true no-data absence, ANL-R4).
_FAILED_REASONS: Final = frozenset(
    {
        UnavailableReason.MISSING_DEPENDENCY,
        UnavailableReason.POOR_FIT,
        UnavailableReason.OUT_OF_DOMAIN,
    }
)


def present_coverage(quality: Any) -> CoverageDescriptor:
    """Map a computed metric's ``QualityReport`` to a present coverage (PMC-R6 provisional).

    When the metric's load was re-resolved to a lower-fidelity equivalence-class member
    (DEGR-R2), the ``QualityReport.extra`` carries ``substitution_class`` + ``from_fidelity``;
    these populate the consumer-visible ``substitution:{class, from_fidelity}`` so a client can
    retrieve the displaced top tier from the API response (API-R29 / SUB-R1(c)). Without a
    substitution the field stays ``None``.
    """
    extra = getattr(quality, "extra", {}) or {}
    fidelity = str(extra.get("fidelity", "raw_stream"))
    sub_class = extra.get("substitution_class")
    from_fid = extra.get("from_fidelity")
    substitution = (
        {"class": sub_class, "from_fidelity": from_fid} if sub_class and from_fid else None
    )
    return CoverageDescriptor(
        present=True,
        fidelity=fidelity,
        gap_fraction=1.0 - float(getattr(quality, "coverage_fraction", 1.0)),
        provisional=bool(extra.get("provisional", False)),
        substitution=substitution,
    )


def absent_coverage(result: Unavailable) -> CoverageDescriptor:
    """Map a typed :class:`Unavailable` to typed-absence coverage (ANL-R4; no reason leak)."""
    fidelity = "absent_failed" if result.reason in _FAILED_REASONS else "absent_true"
    return CoverageDescriptor(present=False, fidelity=fidelity, gap_fraction=1.0)


def coverage_for(result: MetricResult[Any]) -> CoverageDescriptor:
    """Coverage for either branch of a :class:`MetricResult`."""
    return present_coverage(result.quality) if is_computed(result) else absent_coverage(result)


def value_of(result: MetricResult[float]) -> float | None:
    """The scalar value of a numeric result, or typed ``null`` (never ``0``)."""
    return float(result.value) if is_computed(result) else None


def opt_float(value: Any) -> float | None:
    """Coerce a finite numeric (e.g. a quality ``extra`` stat) to ``float | None``."""
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def empty_coverage() -> CoverageDescriptor:
    """The summary-level coverage descriptor for a present series (SCHEMA-R9)."""
    return CoverageDescriptor(present=True, fidelity="raw_stream")


def day_label(day: _dt.date) -> str:
    """Jargon-free X-tick label for a calendar day (API-R21)."""
    return day.strftime("%b ") + str(day.day)


def duration_label(seconds: int) -> str:
    """Jargon-free X-tick label for a power-duration grid point (API-R21)."""
    return f"{seconds // 60} min" if seconds % 60 == 0 else f"{seconds} sec"


def now() -> _dt.datetime:
    """Server timestamp for the precomputed ``computed_at`` (wall-clock at edge)."""
    return _dt.datetime.now(tz=_dt.UTC)


__all__ = [
    "absent_coverage",
    "coverage_for",
    "day_label",
    "duration_label",
    "empty_coverage",
    "now",
    "opt_float",
    "present_coverage",
    "value_of",
]
