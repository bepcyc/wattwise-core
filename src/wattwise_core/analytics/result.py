"""The typed analytic result envelope (doc 40 §1, ANL-R3/R4/R5/R33).

Every metric is a pure function returning a :data:`MetricResult` — either
:class:`Computed` (value + :class:`QualityReport` + :class:`InputLineage`) or
:class:`Unavailable` (a typed reason). A metric that cannot be correctly computed
**fails closed** (ANL-R4): it returns :class:`Unavailable`, never a 0, clamped,
default, extrapolated, or otherwise fabricated number. A wrong-but-plausible number
is the highest-severity defect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeAlias, TypeIs, TypeVar

T = TypeVar("T")


class UnavailableReason(StrEnum):
    """Why a metric is unavailable (doc 40 §1 — the exact closed set)."""

    INSUFFICIENT_DATA = "insufficient_data"
    """Not enough samples / too-short window / too-clustered / too-artifact-laden."""
    MISSING_REQUIRED_INPUT = "missing_required_input"
    """A required canonical channel OR reference param is absent or stale."""
    MISSING_DEPENDENCY = "missing_dependency"
    """A required runtime numeric/DSP capability is unavailable (input present)."""
    POOR_FIT = "poor_fit"
    """A model fit failed its goodness-of-fit gate."""
    OUT_OF_DOMAIN = "out_of_domain"
    """Inputs violate a domain precondition (e.g. HR_max <= HR_rest; non-finite)."""
    NOT_SEEDED = "not_seeded"
    """Required pre-window seed state is unavailable for an exact windowed query."""
    NOT_APPLICABLE_FOR_SPORT = "not_applicable_for_sport"
    """The metric is not defined/meaningful for the activity's sport (ANL-R11/R12)."""


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Coverage/quality of a computed value, readable without recomputation (ANL-R5).

    ``coverage_fraction`` is valid required samples / window. ``extra`` carries
    metric-specific quality stats (fit R²/SE, corrected-interval fraction, gap
    structure, sample rate, long-duration-bias flags, substituted fidelity, …).
    """

    coverage_fraction: float = 1.0
    sample_rate_hz: float | None = None
    gap_count: int = 0
    confidence: float = 1.0
    extra: dict[str, object] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, object]:
        out: dict[str, object] = {
            "coverage_fraction": self.coverage_fraction,
            "sample_rate_hz": self.sample_rate_hz,
            "gap_count": self.gap_count,
            "confidence": self.confidence,
        }
        out.update(self.extra)
        return out


@dataclass(frozen=True, slots=True)
class InputLineage:
    """Which canonical records/streams + athlete params (effective-dated) fed a result.

    Carries the resolved ``sport`` (ANL-R13). Never carries a source NAME — formula
    code is provenance-blind (ANL-R1/R33).
    """

    sport: str | None = None
    activity_ids: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    reference_params: dict[str, object] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "sport": self.sport,
            "activity_ids": list(self.activity_ids),
            "channels": list(self.channels),
            "reference_params": self.reference_params,
        }


@dataclass(frozen=True, slots=True)
class Computed(Generic[T]):  # noqa: UP046 - keep explicit Generic API surface (frozen public contract)
    """A successfully computed metric value with its quality + lineage."""

    value: T
    quality: QualityReport = field(default_factory=QualityReport)
    provenance: InputLineage = field(default_factory=InputLineage)

    available: bool = field(default=True, init=False)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "available": True,
            "value": self.value,
            "quality": self.quality.to_jsonable(),
            "provenance": self.provenance.to_jsonable(),
        }


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A typed fail-closed result (ANL-R4). Explicitly NOT a number."""

    reason: UnavailableReason
    detail: str = ""

    available: bool = field(default=False, init=False)

    def to_jsonable(self) -> dict[str, object]:
        return {"available": False, "reason": self.reason.value, "detail": self.detail}


# A metric result is one or the other (ANL-R3): never a bare number.
MetricResult: TypeAlias = Computed[T] | Unavailable  # noqa: UP040 - keep TypeAlias public API surface


def is_computed(result: MetricResult[T]) -> TypeIs[Computed[T]]:  # noqa: UP047 - keep explicit TypeVar API surface
    """Type-narrowing guard: True iff the result is a :class:`Computed`.

    As a :class:`typing.TypeIs`, a truthy result narrows ``result`` to
    ``Computed[T]`` for the type checker, so consumers can read ``.value`` after an
    ``if is_computed(r):`` guard without an ``Unavailable`` union-attr error.
    """
    return isinstance(result, Computed)


__all__ = [
    "Computed",
    "InputLineage",
    "MetricResult",
    "QualityReport",
    "Unavailable",
    "UnavailableReason",
    "is_computed",
]
