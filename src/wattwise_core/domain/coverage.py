"""Typed coverage descriptor and typed-absence (GAP-R2, GAP-R4).

Every physiological signal carries a typed, schema-validated coverage descriptor.
Absence is *typed*, never zero-filled or fabricated (GAP-R1). Analytic engines read
coverage and **fail closed**: when inputs are absent they return a typed
:class:`Unavailable`, not a plausible-but-wrong number (GAP-R4).

Invariant: a coverage object MUST NOT contain any source identity — no
``source_descriptor_id``, no source name (GAP-R2, GBO-AC-3). That lives only in the
candidate/lineage store, never on a consumer-visible record.
"""

from __future__ import annotations

from dataclasses import dataclass

from wattwise_core.domain.enums import Fidelity


@dataclass(frozen=True, slots=True)
class Substitution:
    """Records that a value came from a lower-fidelity equivalence-class member."""

    equivalence_class: str
    from_fidelity: Fidelity


@dataclass(frozen=True, slots=True)
class Coverage:
    """Per-channel / per-metric coverage descriptor (GAP-R2).

    Carries no source identity. ``fidelity`` ranks via
    :func:`wattwise_core.domain.enums.trust_rank`.
    """

    present: bool
    fidelity: Fidelity
    gap_fraction: float = 0.0
    disputed: bool = False
    provisional: bool = False
    substitution: Substitution | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.gap_fraction <= 1.0:
            raise ValueError(f"gap_fraction must be in [0,1], got {self.gap_fraction}")
        absent = self.fidelity in (Fidelity.ABSENT_TRUE, Fidelity.ABSENT_FAILED)
        if absent and self.present:
            raise ValueError("present=True is inconsistent with an absent_* fidelity")

    @classmethod
    def absent(cls, *, failed: bool = False) -> Coverage:
        """A typed absence (GAP-R1): no usable value.

        ``failed`` distinguishes ``absent_failed`` (a source should have supplied
        it but the fetch failed) from ``absent_true`` (no source supplies it).
        """
        return cls(
            present=False,
            fidelity=Fidelity.ABSENT_FAILED if failed else Fidelity.ABSENT_TRUE,
            gap_fraction=1.0,
        )

    def to_jsonable(self) -> dict[str, object]:
        """Serialize for the canonical ``coverage`` JSON column (no source identity)."""
        out: dict[str, object] = {
            "present": self.present,
            "fidelity": self.fidelity.value,
            "gap_fraction": self.gap_fraction,
            "disputed": self.disputed,
            "provisional": self.provisional,
            "substitution": (
                None
                if self.substitution is None
                else {
                    "class": self.substitution.equivalence_class,
                    "from_fidelity": self.substitution.from_fidelity.value,
                }
            ),
        }
        return out


# NOTE: the analytic fail-closed result envelope (MetricResult / Computed /
# Unavailable / UnavailableReason) is owned by doc 40 and lives in
# wattwise_core.analytics.result — not here. This module owns only the GBO
# coverage descriptor (doc 20).
