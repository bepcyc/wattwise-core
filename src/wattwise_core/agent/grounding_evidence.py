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

import datetime as _dt
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from wattwise_core.agent import grounding as _grounding
from wattwise_core.agent import grounding_sweep as _sweep
from wattwise_core.agent.capabilities import CanonicalEvidence, MetricEquivalence
from wattwise_core.agent.contracts import ChatModel, Claim, ClaimKind, GroundingResult
from wattwise_core.agent.structured import StructuredOutputError, run_structured
from wattwise_core.analytics.service import AnalyticsService

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


class WorkoutEquivalence:
    """Resolve a localized workout NAME to a canonical workout id (#17 / GROUND-R2).

    The multilingual sibling of :class:`~wattwise_core.agent.metric_equivalence.MetricEquivalence`
    for workout names. A PLAN deliverable prescribes its workout names in the run's LANGUAGE
    (a ``ru`` plan writes "восстановительная езда", not "recovery ride"), so binding the canonical
    concept to the ENGLISH spelling scrubbed every localized NAME claim and DEGRADED the plan (#17).
    This layer maps a localized (or English-synonym) surface name to the STABLE canonical English
    name, so grounding resolves the concept by a language-independent id while the surface form
    varies by language (the ICU/Fluent message-ID principle).

    The alias map is loaded CONTENT (``[agent.workout_aliases]``, CFG-R1a), NOT hardcoded here. The
    canonical English names in ``allowed`` (``CANONICAL_WORKOUT_NAMES``) ALWAYS resolve to themselves
    (the English FLOOR), so the English plan path is unchanged. A name that is neither a loaded alias
    nor a canonical English name resolves to ``None`` so the grounder still fails closed (GROUND-R3),
    and an alias whose configured value is not itself a canonical English name is rejected (a
    misconfigured alias never invents a new canonical concept).
    """

    def __init__(self, aliases: Mapping[str, str], allowed: frozenset[str]) -> None:
        # The canonical English names (the FLOOR), pre-folded so an English plan resolves with no
        # alias entry; a non-English plan resolves through the folded alias map onto one of these.
        self._allowed = frozenset(_normalize_workout_name(name) for name in allowed)
        self._aliases: dict[str, str] = {}
        for surface, canonical in aliases.items():
            folded_canonical = _normalize_workout_name(canonical)
            # A misconfigured alias pointing at a non-canonical name fails closed (GROUND-R3).
            if folded_canonical in self._allowed:
                self._aliases[_normalize_workout_name(surface)] = folded_canonical

    def canonical_name(self, name: str) -> str | None:
        """Return the canonical workout id for ``name`` (``workout:{canonical}``), or ``None``.

        Resolution order, all against the SAME canonical English-name set (fail-closed):
        1. the folded name is itself a canonical English name (the FLOOR — English plan unchanged);
        2. the folded name is a loaded alias whose value is a canonical English name (#17).
        Anything else resolves to ``None`` so the grounder scrubs the claim (GROUND-R3).
        """
        folded = _normalize_workout_name(name)
        if folded in self._allowed:
            return f"workout:{folded}"
        mapped = self._aliases.get(folded)
        if mapped is not None:
            return f"workout:{mapped}"
        return None


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
        workout_equivalence: WorkoutEquivalence | None = None,
    ) -> None:
        self._evidence = evidence
        self._snapshots = snapshots
        self._allow_names = allow_names
        # The multilingual workout-name resolver (#17): when supplied it owns NAME resolution so a
        # localized plan name grounds via the loaded alias table; when absent the deliverable keeps
        # the historical English-only frozenset behaviour (the FakeModel suite / no-bundle seam).
        self._workout_equivalence = workout_equivalence

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
        claim (fail-closed, GROUND-R3). With an EMPTY ``allow_names`` and no workout-equivalence
        (the free-form default) every name resolves to ``None`` — preserving the Phase-1 "no
        canonical workout library" behaviour for non-plan deliverables.

        When a :class:`WorkoutEquivalence` is wired (the PLAN path, #17) it owns resolution: a
        localized name grounds via the loaded ``[agent.workout_aliases]`` table onto the canonical
        English id, so a non-English plan no longer scrubs every prescription. Without it, the
        historical English-only frozenset lookup is used (the FakeModel / no-bundle behaviour).
        """
        if self._workout_equivalence is not None:
            return self._workout_equivalence.canonical_name(name)
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


class _ExtractedClaim(BaseModel):
    """One candidate claim the model points at (STRUCT-R5); code verifies it, not the model."""

    model_config = {"extra": "forbid"}
    kind: ClaimKind = ClaimKind.NUMBER
    text: str = ""
    metric: str | None = None
    value: float | None = None
    as_of: str | None = None


class _ClaimSchema(BaseModel):
    """The structured claim-extraction output (GROUND-R2/STRUCT-R5)."""

    model_config = {"extra": "forbid"}
    claims: list[_ExtractedClaim] = Field(default_factory=list)


class ClaimGrounder:
    """Model-extract + code-verify grounder over canonical evidence (GROUND-R1/R2/R7).

    ``allow_names`` is the canonical workout-NAME library a NAME claim may ground against
    (GROUND-R2): the free-form answer/digest grounder passes none (NAME claims fail closed, the
    Phase-1 default), while a PLAN grounder passes :data:`CANONICAL_WORKOUT_NAMES` so a prescribed
    workout name can ground rather than being auto-scrubbed (COACH-R2).

    ``equivalence`` is the config-loaded metric-equivalence layer (§16): the canonical evidence
    resolves a natural metric label a real model emits ("fitness", "Chronic Training Load (CTL)")
    to its canonical key before reading the value (GROUND-R2). With none injected the evidence
    degenerates to canonical-key-only resolution (the prior behaviour). ``reference_date`` anchors
    the latest-available-date fallback for a claim that carries no as-of date.
    """

    def __init__(
        self,
        model: ChatModel,
        svc: AnalyticsService,
        *,
        allow_names: frozenset[str] = frozenset(),
        equivalence: MetricEquivalence | None = None,
        workout_equivalence: WorkoutEquivalence | None = None,
        reference_date: _dt.date | None = None,
        tolerance: _grounding.NumericTolerance | None = None,
        allowed_hosts: frozenset[str] | None = None,
        lookback_days: int | None = None,
        claim_system: str = "",
    ) -> None:
        self._model = model
        self._svc = svc
        self._allow_names = allow_names
        # The multilingual workout-name resolver (#17): the PLAN grounder injects this so a
        # localized prescription name grounds against the loaded alias table -> canonical id.
        # None preserves the English-only ``allow_names`` frozenset behaviour for every other path.
        self._workout_equivalence = workout_equivalence
        self._equivalence = equivalence
        self._reference_date = reference_date
        # None -> the grounder's own default band (preserves the prior behaviour for any seam
        # that injects no coach-config); the engine wires the config-loaded threshold in.
        self._tolerance = tolerance if tolerance is not None else _grounding.NumericTolerance()
        # Config-loaded GROUND-R4 URL allow-list + §16 dateless-claim lookback (CFG-R1a). None ->
        # the canonical evidence's no-config fallbacks (empty host set, default lookback); the
        # engine wires the loaded CoachBundle values in for EVERY grounder path (incl. edits).
        self._allowed_hosts = allowed_hosts
        self._lookback_days = lookback_days
        # The loaded claim-extraction system prompt (§16 / SKILL-R1): the engine embeds NO prompt
        # inline (CFG-R3 / ARCH-R29). Empty default preserves the prior FakeModel-suite behaviour
        # (the suite scripts the extracted claims, so the prompt text is immaterial offline).
        self._claim_system = claim_system

    async def ground(
        self,
        *,
        athlete_id: str,
        draft: str,
        retrieved: Mapping[str, Any],
        request_text: str | None = None,
    ) -> GroundingResult:
        try:
            extracted = await run_structured(
                self._model, system=self._claim_system, data=draft, schema=_ClaimSchema
            )
            claims = [
                Claim(kind=c.kind, text=c.text, metric=c.metric, value=c.value, ref=c.as_of)
                for c in extracted.claims
            ]
        except (StructuredOutputError, NotImplementedError):
            claims = []
        evidence = CanonicalEvidence(
            self._svc,
            athlete_id,
            equivalence=self._equivalence,
            reference_date=self._reference_date,
            allowed_hosts=self._allowed_hosts,
            lookback_days=self._lookback_days,
        )
        snapshots = await _resolve_snapshots(evidence, claims)
        snapshot_evidence = _SnapshotEvidence(
            evidence,
            snapshots,
            allow_names=self._allow_names,
            workout_equivalence=self._workout_equivalence,
        )
        # Numbers the ATHLETE supplied in their own request are sayable echoes (a plan's
        # "5-7 hours a week" is the user's constraint, not a canonical-data claim): collect
        # their tokens so the grounder/sweep can verify an echo instead of scrubbing it.
        # Tokens are sign-stripped: in "5-7 hours" the dash is a RANGE separator, not a minus,
        # so the echo set must carry "7", never "-7".
        request_numbers = (
            frozenset(tok.lstrip("-") for tok in _sweep.NUMBER_RE.findall(request_text))
            if request_text
            else frozenset()
        )
        return _grounding.ground(
            draft,
            claims,
            snapshot_evidence,
            allow_urls=(),
            tolerance=self._tolerance,
            request_numbers=request_numbers,
        )


__all__ = [
    "CANONICAL_WORKOUT_NAMES",
    "ClaimGrounder",
    "WorkoutEquivalence",
    "_SnapshotEvidence",
    "_normalize_workout_name",
    "_resolve_snapshots",
]
