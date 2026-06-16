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

from wattwise_core.agent import grounding as _grounding
from wattwise_core.agent import grounding_sweep as _sweep
from wattwise_core.agent.capabilities import CanonicalEvidence, MetricEquivalence
from wattwise_core.agent.contracts import ChatModel, Claim, ClaimKind, GroundingResult
from wattwise_core.agent.grounding_binding import BindingGuard, BindingMode
from wattwise_core.agent.grounding_claim_schema import (
    _WORKOUT_TYPE_TO_NAME,
    CanonicalWorkoutType,
    _ClaimSchema,
    _ExtractedClaim,
    source_claims,
)
from wattwise_core.agent.grounding_constraints import ActiveConstraint, ConstraintGate
from wattwise_core.agent.grounding_entailment import EntailmentGate
from wattwise_core.agent.grounding_factsheet import render_fact_sheet
from wattwise_core.agent.memory import ConstraintSeverity
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.observability import metrics as obs_metrics
from wattwise_core.observability.logging import get_logger
from wattwise_core.persistence.types import utcnow

_logger = get_logger(__name__)

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


def _coerce_severity(value: object) -> ConstraintSeverity:
    """Coerce a recalled constraint's ``severity`` projection to the enum, defaulting SOFT.

    The recalled core tier (MEM-R6) carries ``severity`` as a plain string token; an unknown /
    missing value falls back closed to SOFT (caution, never an unintended HARD veto, ADR 0008 §4).
    """
    if value is None:
        return ConstraintSeverity.SOFT
    try:
        return ConstraintSeverity(str(value))
    except ValueError:
        return ConstraintSeverity.SOFT


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

        This is the LEGACY surface-name path (English canonical names): it is the fallback the
        grounder uses ONLY when a NAME claim carries no structured ``workout_type`` (issue #18) — a
        non-English plan resolves through :meth:`canonical_workout_type` instead, language-free.
        """
        if not self._allow_names:
            return None
        normalized = _normalize_workout_name(name)
        if normalized in self._allow_names:
            return f"workout:{normalized}"
        return None

    def canonical_workout_type(self, workout_type: str) -> str | None:
        """Resolve a STRUCTURED canonical workout TYPE to its stable citation id (COACH-R2, #18).

        The LANGUAGE-INDEPENDENT grounding path: a prescribed-workout NAME claim that carries a
        typed ``workout_type`` (the model's structured prescription, STRUCT-R1) grounds by that
        enum value, NOT by re-matching its translated surface name — so a plan in ANY language
        resolves identically. The returned id is byte-identical to the legacy surface-name path's
        ``workout:{normalized}`` (the enum maps 1:1 to :data:`CANONICAL_WORKOUT_NAMES`), so the
        GROUND-R5 citation is stable across languages. With an EMPTY ``allow_names`` (the free-form
        default) every type resolves to ``None`` (fail-closed, GROUND-R3); an out-of-vocabulary
        value resolves to ``None`` too (the enum is the only accepted set, STRUCT-R3).
        """
        if not self._allow_names:
            return None
        try:
            member = CanonicalWorkoutType(workout_type)
        except ValueError:
            return None
        normalized = _WORKOUT_TYPE_TO_NAME[member]
        # Defensive: the type's canonical name must be in the allowed library (they are kept in
        # lockstep); if a deployment narrows the library, an absent name still fails closed.
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
        reference_date: _dt.date | None = None,
        tolerance: _grounding.NumericTolerance | None = None,
        allowed_hosts: frozenset[str] | None = None,
        lookback_days: int | None = None,
        claim_system: str = "",
        binding: BindingGuard | None = None,
        entailment: EntailmentGate | None = None,
        constraint_gate: ConstraintGate | None = None,
    ) -> None:
        self._model = model
        self._svc = svc
        self._allow_names = allow_names
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
        # Issue #10 binding-faithful layers, both OPTIONAL so every existing seam/test keeps its
        # prior value-only behaviour: ``binding`` is the deterministic GROUND-R10 guard (its mode
        # decides enforce/shadow/off), ``entailment`` the decorrelated GROUND-R11 sentence gate.
        self._binding = binding
        self._entailment = entailment
        # The deterministic constraint gate (proposed GROUND-R13/R14, ADR 0008): runs AFTER the
        # value + entailment layers, vetoing/cautioning a prescription that contradicts an active
        # athlete constraint threaded in via ``active_constraints``. ``None`` (mode off, or empty
        # bundle) preserves the prior behaviour. The gate is PURE; this grounder does the wiring.
        self._constraint_gate = constraint_gate

    async def ground(
        self,
        *,
        athlete_id: str,
        draft: str,
        retrieved: Mapping[str, Any],
        request_text: str | None = None,
        active_constraints: Sequence[Mapping[str, Any]] | None = None,
        evidence_claims: Sequence[Mapping[str, Any]] | None = None,
    ) -> GroundingResult:
        # COMPOSE-R3 point 2 (slice 3): the candidate-claim source is resolved by
        # :func:`source_claims` — a POPULATED two-layer evidence layer is authoritative (grounding
        # verifies THOSE claims, never re-extracting from the draft); an absent/empty layer falls
        # back to draft extraction (the transitional bridge until compose always populates it).
        claims = await source_claims(
            model=self._model,
            claim_system=self._claim_system,
            draft=draft,
            evidence_claims=evidence_claims,
        )
        guard = self._anchored_guard()
        claims = self._rebind_claims(guard, draft, claims)
        evidence = CanonicalEvidence(
            self._svc,
            athlete_id,
            equivalence=self._equivalence,
            reference_date=self._reference_date,
            allowed_hosts=self._allowed_hosts,
            lookback_days=self._lookback_days,
        )
        snapshots = await _resolve_snapshots(evidence, claims)
        snapshot_evidence = _SnapshotEvidence(evidence, snapshots, allow_names=self._allow_names)
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
        result = _grounding.ground(
            draft,
            claims,
            snapshot_evidence,
            allow_urls=(),
            tolerance=self._tolerance,
            request_numbers=request_numbers,
            binding=guard if guard is not None and guard.mode is BindingMode.ENFORCE else None,
        )
        result = await self._apply_entailment(result, snapshots, retrieved, request_text)
        return self._apply_constraints(result, active_constraints)

    def _apply_constraints(
        self, result: GroundingResult, active_constraints: Sequence[Mapping[str, Any]] | None
    ) -> GroundingResult:
        """Run the deterministic constraint gate over the grounded result (GROUND-R13/R14).

        The athlete's ACTIVE constraints are threaded in from graph state (the recalled core tier,
        MEM-R6) as plain ``{content, severity}`` projections; this wiring builds the typed
        :class:`ActiveConstraint` set (deriving each forbidden-activity token set) and runs the PURE
        gate. A HARD violation scrubs the contradicting prescription and forces the decision off
        ``proceed``; a SOFT one surfaces a caution (the prescription stays). Both are RECORDED on
        observability counters (ADR 0008 §7), mirroring the entailment recorder. With no gate
        configured (mode off / empty bundle) or no active constraints, the result is unchanged.
        """
        if self._constraint_gate is None or not active_constraints:
            return result
        constraints = [
            ActiveConstraint.from_content(content, _coerce_severity(item.get("severity")))
            for item in active_constraints
            if (content := str(item.get("content") or "").strip())
        ]
        if not constraints:
            return result
        gated = self._constraint_gate.apply(result, constraints)
        registry = obs_metrics.get_registry()
        for sentence in gated.hard_violations:  # noqa: B007  count, not the value
            registry.increment(
                obs_metrics.CONSTRAINT_VIOLATIONS,
                labels={"severity": ConstraintSeverity.HARD.value},
            )
        if gated.cautions:
            registry.increment(obs_metrics.CONSTRAINT_CAUTIONS, amount=float(len(gated.cautions)))
        if gated.hard_violations:
            _logger.warning("grounding_constraint_violations", count=len(gated.hard_violations))
        if gated.cautions:
            _logger.info("grounding_constraint_cautions", count=len(gated.cautions))
        return gated.result

    def _anchored_guard(self) -> BindingGuard | None:
        """The run's GROUND-R10 guard, anchored ONCE to the evidence's reference date.

        Anchoring to the SAME clock the canonical evidence uses keeps the temporal rule
        and the value reads on one reference date, and keeps ``ground`` a deterministic
        function of its inputs (GRAPH-R4). ``None`` when unconfigured or mode ``off``.
        """
        if self._binding is None or self._binding.mode is BindingMode.OFF:
            return None
        return self._binding.anchored(self._reference_date or utcnow().date())

    def _rebind_claims(
        self, guard: BindingGuard | None, draft: str, claims: list[Claim]
    ) -> list[Claim]:
        """Re-derive each claim's canonical cell out of its own sentence (issue #10, R10).

        Runs BEFORE snapshot resolution, so the values fetched and verified are the cells
        the SENTENCES assert — the model's extracted binding cannot route verification.
        Every rebind is recorded (alertable counter + log: a drifting extractor is an
        operational signal, AGT-OBS-R7). SHADOW records what WOULD change and applies
        nothing; residual non-rebindable inconsistencies are recorded here too and fail
        closed inside ``ground`` when the mode is ENFORCE.
        """
        if guard is None:
            return claims
        rebound, events = guard.rebind(draft, claims)
        registry = obs_metrics.get_registry()
        for event in (*events, *guard.assess(draft, rebound)):
            registry.increment(
                obs_metrics.GROUNDING_BINDING_EVENTS,
                labels={"event": event.value, "mode": guard.mode.value},
            )
        if events:
            _logger.warning(
                "grounding_binding_rebound",
                mode=guard.mode.value,
                events=[e.value for e in events],
            )
        return list(rebound) if guard.mode is BindingMode.ENFORCE else claims

    async def _apply_entailment(
        self,
        result: GroundingResult,
        snapshots: Mapping[tuple[str, str | None], float | None],
        retrieved: Mapping[str, Any],
        request_text: str | None,
    ) -> GroundingResult:
        """Run the optional GROUND-R11 sentence gate over the grounded result (issue #10).

        The fact sheet is rendered by CODE from the same snapshots the value gate verified
        against plus the turn's retrieved records (and the athlete's own request, so an
        echoed constraint is entailed rather than vetoed). A verifier failure degrades to
        the deterministic layers and is RECORDED (counter + log) — never silently open.
        """
        if self._entailment is None or not result.scrubbed_text.strip():
            return result
        facts = render_fact_sheet(snapshots, retrieved, request_text=request_text)
        gated, report = await self._entailment.apply(result, facts=facts)
        registry = obs_metrics.get_registry()
        if report.unavailable:
            registry.increment(obs_metrics.ENTAILMENT_UNAVAILABLE)
            _logger.warning("grounding_entailment_unavailable")
            return gated
        if report.checked:
            registry.increment(obs_metrics.ENTAILMENT_CHECKS, amount=float(report.checked))
        if report.vetoed:
            registry.increment(obs_metrics.ENTAILMENT_VETOES, amount=float(len(report.vetoed)))
            _logger.warning("grounding_entailment_vetoes", count=len(report.vetoed))
        return gated


__all__ = [
    "CANONICAL_WORKOUT_NAMES",
    "CanonicalWorkoutType",
    "ClaimGrounder",
    "_ClaimSchema",
    "_ExtractedClaim",
    "_SnapshotEvidence",
    "_normalize_workout_name",
    "_resolve_snapshots",
    "source_claims",
]
