"""Training-load equivalence-class substitution carriers (DM-SUB-R1 / DEGR-R2).

The canonical ``training_load`` channel is resolved through an ORDERED equivalence class
(DM-SUB-R1 worked example) via the LOAD-R3 fallback: power-based TSS is the top member
(``raw_stream``); the HR-derived Banister load (TRIMP) is the lowest ``modeled`` member.

When the top source is withdrawn and the load is recomputed from the HR member, the value
is a SUBSTITUTION: its coverage MUST carry :data:`Fidelity.SUBSTITUTED` with
``substitution:{class:training_load, from_fidelity:raw_stream}`` (DEGR-R2 / DM-SUB-R4) — the
in-class downgrade token, NEVER the displaced member's own ``modeled`` tier — so a client
badges reduced precision rather than reading an HR load as full-fidelity power-TSS. This
module owns those small, pure carriers; the per-activity resolution (LOAD-R3 fallback) lives
in :mod:`wattwise_core.analytics.service`, the PMC surfacing in
:mod:`wattwise_core.analytics.pmc`.
"""

from __future__ import annotations

from dataclasses import dataclass

from wattwise_core.analytics.constants import TRAINING_LOAD_CLASS
from wattwise_core.domain.coverage import Coverage, Substitution
from wattwise_core.domain.enums import Fidelity

__all__ = [
    "LOAD_SUBSTITUTED_COVERAGE",
    "LOAD_TOP_COVERAGE",
    "LoadContribution",
    "day_load_coverage",
]


@dataclass(frozen=True, slots=True)
class LoadContribution:
    """One activity's resolved training load + the coverage of the class member it came from.

    ``coverage.fidelity`` is :data:`Fidelity.RAW_STREAM` for the top (power-TSS) member, or
    :data:`Fidelity.SUBSTITUTED` (with a populated ``substitution``) when the load came from
    the lower-fidelity HR member because the power source was withdrawn (DEGR-R2).
    """

    value: float
    coverage: Coverage


# The TOP member of the ``training_load`` class is power-TSS at raw_stream; the HR member is
# modeled. When the HR member wins because the power source is absent (withdrawn), the load is
# SUBSTITUTED and carries the displaced top tier in ``from_fidelity`` (DEGR-R2 / DM-SUB-R4).
LOAD_TOP_COVERAGE = Coverage(present=True, fidelity=Fidelity.RAW_STREAM)
LOAD_SUBSTITUTED_COVERAGE = Coverage(
    present=True,
    fidelity=Fidelity.SUBSTITUTED,
    substitution=Substitution(
        equivalence_class=TRAINING_LOAD_CLASS, from_fidelity=Fidelity.RAW_STREAM
    ),
)


def day_load_coverage(*, has_load: bool, substituted: bool) -> Coverage | None:
    """The coverage for one calendar day's aggregated load (DEGR-R2).

    A day fed (in whole or part) by a substituted lower-fidelity member is SUBSTITUTED — a
    partially substituted day is never presented at full fidelity; a day all of whose
    contributions are top-tier carries the top-member coverage; a no-load day (rest or
    unknown-load) carries ``None``, never a fabricated fidelity.
    """
    if not has_load:
        return None
    return LOAD_SUBSTITUTED_COVERAGE if substituted else LOAD_TOP_COVERAGE
