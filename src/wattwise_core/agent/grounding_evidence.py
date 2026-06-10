"""Sync grounding-evidence wrapper + canonical workout-name library (GROUND-R2/R7).

The focused sibling of :mod:`wattwise_core.agent.engine_services` (QUAL-R9 size split) that owns the
deterministic grounding-evidence plumbing the ``ClaimGrounder``
runs on: the canonical training-prescription NAME library a prescribed workout grounds against
(GROUND-R2), the pre-resolved sync :class:`_SnapshotEvidence` adapter (so the synchronous
fail-closed grounder verifies NUMBER claims VERBATIM against canonical analytics without awaiting,
GROUND-R7), and the async snapshot resolver that fills it. Behaviour is identical to the prior
inline definitions; this is purely a size decomposition that keeps ``engine_services`` under the
QUAL-R9 module ceiling.

Cited requirements: GROUND-R2, GROUND-R3, GROUND-R4, GROUND-R7, COACH-R2, QUAL-R9.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from wattwise_core.agent.capabilities import CanonicalEvidence
from wattwise_core.agent.contracts import Claim, ClaimKind

# The canonical training-prescription workout NAME library (GROUND-R2). A prescribed workout NAME
# in a multi-day PLAN deliverable grounds ONLY if it normalizes to one of these canonical
# training-prescription names — the deterministic, fixed vocabulary the engine recognizes (not
# athlete-specific data). An invented/free-text name ("magic super workout") resolves to None and
# is scrubbed (GROUND-R3, "when in doubt, scrub"). This is the minimal canonical name-allow path
# the PLAN deliverable needs so a prescribed name is NOT auto-scrubbed like a free-form answer's
# NAME claim (the free-form answer passes NO library, so its NAME claims still fail closed).
CANONICAL_WORKOUT_NAMES: frozenset[str] = frozenset(
    {
        "rest day",
        "recovery ride",
        "recovery spin",
        "endurance ride",
        "long ride",
        "tempo intervals",
        "sweet spot intervals",
        "threshold intervals",
        "vo2max intervals",
        "anaerobic intervals",
        "sprint intervals",
    }
)


def _normalize_workout_name(name: str) -> str:
    """Normalize a workout name for canonical-library comparison (case/whitespace-folded)."""
    return " ".join(name.casefold().split())


class _SnapshotEvidence:
    """Sync grounding evidence: pre-resolved canonical snapshots + first-party URL gate.

    The deterministic grounder (GROUND-R*) is synchronous and reads canonical values via a sync
    ``metric_snapshot``; the canonical :class:`CanonicalEvidence` exposes only the async
    ``metric_value``. This wrapper carries the snapshots an async pass resolved ahead of time over
    the extracted claims, so a NUMBER claim is verified VERBATIM against canonical analytics
    (GROUND-R7) WITHOUT the grounder ever awaiting. ``url_allowed`` / ``metric_value`` delegate to
    the wrapped evidence.

    A NAME claim grounds via :meth:`canonical_name` ONLY when an explicit canonical workout-name
    library is supplied (the PLAN path, COACH-R2); with no library (``allow_names`` empty — the
    free-form answer/digest default) NAME claims fail closed (GROUND-R3), since Phase-1 ships no
    open canonical workout library for free-form prose.
    """

    def __init__(
        self,
        evidence: CanonicalEvidence,
        snapshots: Mapping[tuple[str, str | None], float | None],
        *,
        allow_names: frozenset[str] = frozenset(),
    ) -> None:
        self._evidence = evidence
        self._snapshots = snapshots
        self._allow_names = allow_names

    def metric_snapshot(self, metric: str, as_of: str | None) -> float | None:
        """The pre-resolved canonical value for ``(metric, as_of)``, or ``None`` (GROUND-R7)."""
        return self._snapshots.get((metric, as_of))

    async def metric_value(self, metric: str, as_of: str | None) -> float | None:
        """Satisfy the async :class:`GroundingEvidence` contract by delegating (GROUND-R2)."""
        return await self._evidence.metric_value(metric, as_of)

    def url_allowed(self, url: str) -> bool:
        """First-party URL allow-list, delegated to the canonical evidence (GROUND-R4)."""
        return self._evidence.url_allowed(url)

    def canonical_name(self, name: str) -> str | None:
        """Resolve a prescribed workout NAME against the supplied canonical library (GROUND-R2).

        Returns a stable canonical id (``workout:{normalized}``) when ``name`` normalizes to an
        allowed canonical training-prescription name, else ``None`` so the grounder scrubs the
        claim (fail-closed, GROUND-R3). With an EMPTY ``allow_names`` (the free-form default) every
        name resolves to ``None`` — preserving the Phase-1 "no canonical workout library" behaviour
        for non-plan deliverables.
        """
        if not self._allow_names:
            return None
        normalized = _normalize_workout_name(name)
        if normalized in self._allow_names:
            return f"workout:{normalized}"
        return None


async def _resolve_snapshots(
    evidence: CanonicalEvidence, claims: Sequence[Claim]
) -> dict[tuple[str, str | None], float | None]:
    """Resolve each NUMBER claim's canonical value ahead of the synchronous grounder.

    Reads the canonical analytic VERBATIM via the async ``metric_value`` for every distinct
    ``(metric, as_of)`` a NUMBER claim points at (GROUND-R7); the grounder then verifies against
    this snapshot without awaiting. A metric the service cannot compute resolves to ``None`` so the
    grounder scrubs the claim (fail-closed), never a placeholder.
    """
    snapshots: dict[tuple[str, str | None], float | None] = {}
    for claim in claims:
        if claim.kind is not ClaimKind.NUMBER or claim.metric is None:
            continue
        key = (claim.metric, claim.ref)
        if key not in snapshots:
            snapshots[key] = await evidence.metric_value(claim.metric, claim.ref)
    return snapshots


__all__ = [
    "CANONICAL_WORKOUT_NAMES",
    "_SnapshotEvidence",
    "_normalize_workout_name",
    "_resolve_snapshots",
]
