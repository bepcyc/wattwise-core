"""Evidence *sufficiency* — is the canonical record current/complete enough to advise on?

The grounding gate verifies that a stated number AGREES with the canonical record (GROUND-R2/R7).
It is silent on a second, equally load-bearing question: is the record ITSELF sufficient — recent
and complete enough — to support the advice? A connector that silently stopped delivering (an
expired ``intervals.icu`` token, a withdrawn OAuth grant) leaves an UNOBSERVED tail of days that
the load pipeline cannot distinguish from genuine rest, so the EWMA decays ATL toward zero and
recorded form (TSB) drifts UP — manufacturing a "you're fresh" signal precisely when data stopped.
A faithfully-verified number can therefore still carry false advice; an entailment/faithfulness
check cannot catch it because the (stale) record genuinely entails the sentence. The missing axis
is the record's epistemic STATUS, which no layer inspected.

This pure module is that axis: a typed :class:`RecordSufficiency` distilled from the freshness of
the observed canonical data (the gap between the reference day and the most recent OBSERVED
activity) plus its load fidelity (was any recent day fed by a SUBSTITUTED lower-fidelity member,
DEGR-R2). It is a deterministic, side-effect-free function of dates and flags (ANL-R2/R30): no DB,
no wall-clock, no RNG, no imports from ``agent/`` or ``persistence/``. The freshness signal is
inherently MISSING-NOT-AT-RANDOM (sync tends to break during travel/illness — exactly when
training and recovery are atypical), so the honest treatment is to WIDEN caution and abstain, never
to impute the gap into a confident number; the consuming deliverable applies that fail-closed policy
asymmetrically (insufficiency may only lower aggressiveness or abstain, never raise it).

Cited requirements: GROUND-R6/R7 (fail closed on insufficient/unavailable evidence), DEGR-R2
(substituted-fidelity surfacing), OUTCOME-R3/-R4 (degraded outcome + typed coverage caveat),
ANL-R2/R30 (pure module). Scientific basis: the unobserved tail is treated as a *missing
measurement* whose uncertainty grows over the gap rather than an asserted ``L = 0`` rest day — the
state-space/Kalman framing of the fitness-fatigue model (Kolossa et al. 2017) and the
sufficient-context lens on grounded generation (Joren et al. 2025).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Fidelity",
    "RecordSufficiency",
    "assess_record_sufficiency",
]

#: The athlete-facing fidelity of a record-backed answer (mirrors the OUTCOME-R4 caveat shape).
Fidelity = Literal["full", "partial", "degraded"]


@dataclass(frozen=True, slots=True)
class RecordSufficiency:
    """How current/complete the canonical record is behind a metric read (the sufficiency axis).

    ``staleness_days`` is the whole-day gap between the reference day and the most recent OBSERVED
    activity, or ``None`` when nothing was ever observed. ``fresh_within_days`` /
    ``max_staleness_days`` are the configured zone edges the consumer's policy reads.
    ``substituted`` flags a recent load from a lower-fidelity equivalence-class member (DEGR-R2).

    ``sync_suspect`` resolves the MNAR ambiguity: an old observation is the EXPECTED state during a
    taper/rest block when the pipeline is healthy, so the staleness zones below only "bite" when a
    connector that should be delivering is broken/silent. The three derived properties classify the
    record without re-deriving it:

    * :attr:`stale` — a SUSPECTED-sync gap beyond the caveat-free freshness zone: the read is usable
      but its currency must be disclosed and the most aggressive verdict withheld.
    * :attr:`insufficient` — never observed, or a SUSPECTED-sync gap beyond the hard floor: the
      unobserved tail now dominates the EWMA-relevant window, so a current-state read cannot be
      trusted at all (fail-closed → the consumer abstains, GROUND-R6).
    * :attr:`fidelity` — the source-agnostic ``full | partial | degraded`` label the OUTCOME-R4
      coverage caveat renders.
    """

    staleness_days: int | None
    fresh_within_days: int
    max_staleness_days: int
    substituted: bool = False
    sync_suspect: bool = False

    @property
    def observed(self) -> bool:
        """True iff the record carries an observed activity to anchor freshness against."""
        return self.staleness_days is not None

    @property
    def gap_days(self) -> int | None:
        """The raw whole-day gap to the most recent observation (the measurement, not a verdict)."""
        return self.staleness_days

    @property
    def stale(self) -> bool:
        """True iff the read should disclose staleness AND withhold the most aggressive verdict.

        Crucially gated on :attr:`sync_suspect`: an old observation is the EXPECTED, trustworthy
        state during a legitimate taper or rest block when the sync pipeline is healthy — only when
        a connector that SHOULD be delivering is broken/silent does the same gap mean data is likely
        MISSING. Without that corroboration we never block a real taper's fresh-form ``go`` (the
        false-abstain failure that a raw activity-gap signal would cause).
        """
        return (
            self.staleness_days is not None
            and self.staleness_days > self.fresh_within_days
            and self.sync_suspect
        )

    @property
    def insufficient(self) -> bool:
        """True iff the record cannot support a current-state verdict (fail-closed, GROUND-R6).

        Either nothing was ever observed, or a SUSPECTED-sync gap has run past the hard floor — past
        which the assumed-rest decay has moved the EWMA far enough that the read is dominated by the
        unobserved (fabricated-rest) tail rather than real data. A long gap with a HEALTHY sync is a
        genuine detraining/off-season read, not insufficiency, so it is not abstained here.
        """
        if self.staleness_days is None:
            return True
        return self.staleness_days > self.max_staleness_days and self.sync_suspect

    @property
    def fidelity(self) -> Fidelity:
        """The source-agnostic fidelity label for the OUTCOME-R4 coverage caveat."""
        if self.insufficient:
            return "degraded"
        if self.stale or self.substituted:
            return "partial"
        return "full"


def assess_record_sufficiency(
    *,
    reference_date: _dt.date,
    last_observed_date: _dt.date | None,
    fresh_within_days: int,
    max_staleness_days: int,
    substituted: bool = False,
    sync_suspect: bool = False,
) -> RecordSufficiency:
    """Assess record sufficiency from observed-data freshness + sync health + load fidelity (pure).

    ``last_observed_date`` is the most recent day the athlete actually has OBSERVED activity data
    for (``None`` when none exists). Staleness is the whole-day gap to ``reference_date``, clamped
    at ``0`` so a future-dated observation (clock skew across sources) never reads as negative
    staleness. ``substituted`` marks a recent lower-fidelity (HR-modeled) load (DEGR-R2).
    ``sync_suspect`` is the MISSING-NOT-AT-RANDOM disambiguator (the heart of issue #12): a gap only
    implies MISSING data — rather than a legitimate taper/rest — when a connector that should be
    delivering is broken or silently not syncing. Pure (ANL-R2): it measures the evidence and
    decides nothing about verdicts, leaving the fail-closed policy to the consuming deliverable.
    """
    if last_observed_date is None:
        staleness: int | None = None
    else:
        staleness = max(0, (reference_date - last_observed_date).days)
    return RecordSufficiency(
        staleness_days=staleness,
        fresh_within_days=fresh_within_days,
        max_staleness_days=max_staleness_days,
        substituted=substituted,
        sync_suspect=sync_suspect,
    )
