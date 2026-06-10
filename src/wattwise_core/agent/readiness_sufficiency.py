"""Record-freshness POLICY for the readiness deliverable (GROUND-R6, the issue #12 axis).

The focused sibling of :mod:`wattwise_core.agent.readiness_deliverable` (QUAL-R9 size split) that
owns the deterministic, fail-closed policy over a :class:`~wattwise_core.analytics.sufficiency.
RecordSufficiency`: the honest stale sentences/clauses, the verdict-blocking decision, and the
projection of the sufficiency envelope into the OUTCOME-R4 coverage caveat.

It is PURE and sits strictly BELOW the deliverable in the import graph: it depends only on the
typed sufficiency envelope and the :class:`~wattwise_core.domain.enums.ReadinessVerdict`, never on
the deliverable's :class:`Readiness` shape — so the deliverable imports DOWNWARD from here with no
cycle, and re-exports :data:`STALE_DATA_CLAUSE` so existing import paths stay stable.

Cited requirements: GROUND-R6 (fail closed on insufficient evidence), OUTCOME-R4 (typed coverage
caveat), VOICE-R7 (no athlete-facing numbers; precise staleness stays in the structured caveat).
"""

from __future__ import annotations

from typing import Any

from wattwise_core.analytics.sufficiency import RecordSufficiency
from wattwise_core.domain.enums import ReadinessVerdict

#: The truthful abstain lead when the form number EXISTS but the record behind it has gone stale
#: (GROUND-R6, sufficiency axis): the most recent OBSERVED data is old enough that the verdict would
#: be read off an EWMA tail of assumed-rest days, which can be real rest OR a silently-broken sync —
#: data alone cannot tell (MNAR). Honest under BOTH branches: it asks the athlete to check sync
#: without asserting it, and emits no verdict/number. Carries no digit (VOICE-R7).
STALE_ABSTAIN_SENTENCE = (
    "I haven't seen any recent training data, so I can't read your readiness right now — "
    "if you've been training, it's worth checking that your data sync is still connected."
)

#: The honest staleness clause appended to a delivered verdict in the disclose zone (FRESH < gap <=
#: MAX): the verdict still ships (and is only ever a less-aggressive, safe-side call there — GO is
#: blocked upstream) but its currency is disclosed. Digit-free so it never trips the number cap and
#: keeps the precise staleness only in the structured coverage caveat (VOICE-R7).
STALE_DATA_CLAUSE = "I haven't seen new training data in a few days, so this may lag where you are."


def freshness_blocks_verdict(sufficiency: RecordSufficiency, verdict: ReadinessVerdict) -> bool:
    """True iff record staleness forbids emitting ``verdict`` at all (asymmetric fail-closed).

    Two fail-closed conditions, mirroring the one-directional HRV nudge (which may only push toward
    caution): the record is INSUFFICIENT (past the hard floor / never observed) so no current-state
    verdict can be read; OR the oracle's call is the most-aggressive ``go`` on a merely STALE record
    — telling a fatigued athlete to go hard off a record that cannot see the last several days is
    exactly the manufactured-freshness failure, so ``go`` is never emitted on stale data. A
    less-aggressive verdict on a stale record is safe-side and still ships (DEGRADED + caveated).
    """
    return sufficiency.insufficient or (sufficiency.stale and verdict is ReadinessVerdict.GO)


def sufficiency_coverage_fields(sufficiency: RecordSufficiency) -> dict[str, Any]:
    """Project the sufficiency envelope into the OUTCOME-R4 coverage-caveat fields.

    The source-agnostic record-freshness caveat: the machine-readable ``staleness_days`` + the
    ``stale``/``substituted`` flags + the resulting ``fidelity`` (the precise day count lives ONLY
    here, never in athlete-facing prose, VOICE-R7). The deliverable merges these into its coverage
    map alongside the oracle's inputs/override fields.
    """
    fields: dict[str, Any] = {
        "fidelity": sufficiency.fidelity,
        "staleness_days": sufficiency.staleness_days,
    }
    if sufficiency.stale:
        fields["stale"] = True
    if sufficiency.substituted:
        fields["substituted"] = True
    return fields


__all__ = [
    "STALE_ABSTAIN_SENTENCE",
    "STALE_DATA_CLAUSE",
    "freshness_blocks_verdict",
    "sufficiency_coverage_fields",
]
