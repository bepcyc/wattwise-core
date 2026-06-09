"""Concrete production agent services: planner, gateway, coverage, grounder (doc 50).

The focused sibling of :mod:`wattwise_core.agent.engine` (QUAL-R9 size split) that owns the
CONCRETE production implementations of the injected agent seams the graph runs on — a model-driven
retrieval planner (PLAN-R1/R2), the canonical capability gateway (TOOL-R1), a deterministic
coverage assessor, and a model-extract + code-verify grounder over canonical evidence
(GROUND-R1/R2/R7) — plus the closed structured-output schemas the model fills and the
``_build_services`` bundle assembler. ``engine`` imports these and re-exports the public ones
(``ModelPlanner`` / ``RegistryGateway`` / ``DeterministicCoverage`` / ``ClaimGrounder``) so every
historical ``from wattwise_core.agent.engine import ...`` path stays stable.

The model NEVER self-certifies (OUTCOME-R5): it emits only the structured retrieval plan and the
candidate claims; deterministic code resolves capabilities and verifies every claim against
canonical data, then fail-closed grounds (unverifiable numbers/names/URLs scrubbed, GROUND-R3).

Cited requirements: PLAN-R1/R2/R3/R5, TOOL-R1, STRUCT-R5, GROUND-R1/R2/R3/R5/R7, GRAPH-R5,
OUTCOME-R5.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from wattwise_core.agent import grounding as _grounding
from wattwise_core.agent.capabilities import (
    CAPABILITY_BY_KEY,
    CanonicalEvidence,
    MetricEquivalence,
    gather,
)
from wattwise_core.agent.contracts import (
    ChatModel,
    Claim,
    ClaimKind,
    GroundingResult,
    RetrievalRequest,
)
from wattwise_core.agent.seams import AgentServices
from wattwise_core.agent.structured import StructuredOutputError, run_structured
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.persistence.types import utcnow

# Date-range capabilities the headline planner can request without an activity id; the
# per-activity/per-day capabilities need an id the planner does not have at plan time.
_DATE_RANGE_CAPABILITIES = ("weekly_load", "critical_power", "power_curve")
_DEFAULT_WINDOW_DAYS = 42


class _PlanSchema(BaseModel):
    """Provider-enforced retrieval plan (PLAN-R2): which canonical capabilities to gather."""

    model_config = {"extra": "forbid"}
    capabilities: list[str] = Field(default_factory=list)
    window_days: int = Field(default=_DEFAULT_WINDOW_DAYS, ge=1, le=365)


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


_PLAN_SYSTEM = (
    "You are the coaching agent's retrieval planner. Choose which canonical analytics "
    "capabilities to gather to answer the athlete, from the closed set "
    f"{_DATE_RANGE_CAPABILITIES}, plus a window in days. Return ONLY the structured plan."
)
_CLAIM_SYSTEM = (
    "Extract every factual numeric claim in the draft as a candidate claim with its "
    "metric name, value, and the local date (ISO 8601) it is as-of when the draft states "
    "one. Do NOT judge correctness — only point at candidates."
)


class ModelPlanner:
    """Model-driven retrieval planner (PLAN-R1/R2): the structured plan IS the selection."""

    def __init__(self, model: ChatModel, *, reference_date: _dt.date | None = None) -> None:
        self._model = model
        self._today = reference_date or utcnow().date()

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        """Emit the next batch of capability requests; fail-closed to a default on error."""
        try:
            plan = await run_structured(
                self._model,
                system=_PLAN_SYSTEM,
                data=f"question: {request_text}\nopen_gaps: {list(gaps)}\nalready: {list(already)}",
                schema=_PlanSchema,
            )
            keys = [k for k in plan.capabilities if k in _DATE_RANGE_CAPABILITIES]
            window = plan.window_days
        except (StructuredOutputError, NotImplementedError):
            keys, window = ["weekly_load"], _DEFAULT_WINDOW_DAYS
        if not keys:
            keys = ["weekly_load"]
        frm = self._today - _dt.timedelta(days=window)
        params = {"from_date": frm.isoformat(), "to_date": self._today.isoformat()}
        seen = set(already)
        return [
            RetrievalRequest(capability=k, params=dict(params))
            for k in keys
            if k in CAPABILITY_BY_KEY and k not in seen
        ]


class RegistryGateway:
    """Resolves capability requests to canonical evidence via the one registry (TOOL-R1)."""

    def __init__(self, svc: AnalyticsService) -> None:
        self._svc = svc

    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        result = await gather(self._svc, athlete_id, list(requests))
        return result.records


class DeterministicCoverage:
    """Reports planned capabilities that resolved to no canonical evidence (pure)."""

    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        # A turn with no retrieved evidence at all is the only structural gap the headline
        # flow reports; per-capability emptiness is surfaced by the gather records.
        return set() if retrieved else {"no_canonical_evidence"}


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

    async def ground(
        self, *, athlete_id: str, draft: str, retrieved: Mapping[str, Any]
    ) -> GroundingResult:
        try:
            extracted = await run_structured(
                self._model, system=_CLAIM_SYSTEM, data=draft, schema=_ClaimSchema
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
        snapshot_evidence = _SnapshotEvidence(evidence, snapshots, allow_names=self._allow_names)
        return _grounding.ground(
            draft, claims, snapshot_evidence, allow_urls=(), tolerance=self._tolerance
        )


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


class CoachBundle:
    """The loaded OSS coach-config: compose prompt + metric-equivalence + tolerance (§16/SKILL-R1).

    DATA the engine consumes (COACH-CFG-R3), loaded from external config (``[agent.coach]`` +
    ``[agent.metric_aliases]`` in ``defaults.toml``, overridable by the operator/private bundle) —
    the engine embeds NO persona/prompt/alias/threshold literal inline (CFG-R1a / SKILL-R6). The
    empty default bundle (no prompt, empty equivalence, default tolerance) preserves the prior
    FakeModel-test behaviour for any seam that injects none. :meth:`services` / :meth:`grounder` are
    the construction seam that wires this config-loaded equivalence + tolerance into the graph.
    """

    __slots__ = ("allowed_hosts", "equivalence", "lookback_days", "system_prompt", "tolerance")

    def __init__(
        self,
        system_prompt: str = "",
        equivalence: MetricEquivalence | None = None,
        tolerance: _grounding.NumericTolerance | None = None,
        allowed_hosts: frozenset[str] = frozenset(),
        lookback_days: int | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.equivalence = equivalence if equivalence is not None else MetricEquivalence({})
        self.tolerance = tolerance if tolerance is not None else _grounding.NumericTolerance()
        # GROUND-R4 first-party URL allow-list + §16 dateless-claim lookback, loaded content
        # (CFG-R1a). The empty default bundle ships no hosts (fail-closed) and no lookback override.
        self.allowed_hosts = allowed_hosts
        self.lookback_days = lookback_days

    @classmethod
    def from_settings(cls, settings: Any) -> CoachBundle:
        """Build the coach bundle from resolved settings (the loaded §16 config)."""
        return cls(
            system_prompt=settings.agent__coach__system_prompt,
            equivalence=MetricEquivalence(settings.agent__metric_aliases),
            tolerance=_grounding.NumericTolerance(
                rel=settings.agent__coach__grounding_rel_tolerance,
                abs_=settings.agent__coach__grounding_abs_tolerance,
                display_decimals=settings.agent__coach__grounding_display_decimals,
            ),
            allowed_hosts=frozenset(settings.agent__allowed_hosts),
            lookback_days=settings.agent__coach__latest_lookback_days,
        )

    def services(
        self, model: ChatModel, svc: AnalyticsService, *, allow_names: frozenset[str] = frozenset()
    ) -> AgentServices:
        """Production service bundle wiring this coach-config's equivalence + tolerance + URL."""
        return build_services(
            model,
            svc,
            allow_names=allow_names,
            equivalence=self.equivalence,
            tolerance=self.tolerance,
            allowed_hosts=self.allowed_hosts,
            lookback_days=self.lookback_days,
        )

    def grounder(self, model: ChatModel, svc: AnalyticsService) -> ClaimGrounder:
        """A grounder carrying this coach-config's equivalence + tolerance + URL/lookback (§16)."""
        return ClaimGrounder(
            model,
            svc,
            equivalence=self.equivalence,
            tolerance=self.tolerance,
            allowed_hosts=self.allowed_hosts,
            lookback_days=self.lookback_days,
        )


def build_services(
    model: ChatModel,
    svc: AnalyticsService,
    *,
    allow_names: frozenset[str] = frozenset(),
    equivalence: MetricEquivalence | None = None,
    reference_date: _dt.date | None = None,
    tolerance: _grounding.NumericTolerance | None = None,
    allowed_hosts: frozenset[str] | None = None,
    lookback_days: int | None = None,
) -> AgentServices:
    """Assemble the concrete production service bundle for the graph (GRAPH-R5).

    ``allow_names`` is the canonical workout-NAME library the grounder may ground a prescribed NAME
    against (empty for the free-form answer/digest; :data:`CANONICAL_WORKOUT_NAMES` for a PLAN
    deliverable so its prescriptions are not auto-scrubbed, COACH-R2 / GROUND-R2). ``equivalence``
    is the config-loaded metric-equivalence layer (§16) the grounder resolves a natural metric
    label through (GROUND-R2); ``reference_date`` anchors the latest-available-date fallback for a
    dateless claim; ``tolerance`` is the config-loaded numeric-match band (GROUND-R7);
    ``allowed_hosts`` is the config-loaded first-party URL allow-list (GROUND-R4) and
    ``lookback_days`` the §16 dateless-claim window. All default to ``None`` (canonical-key-only,
    today, default band, no-config host/lookback fallbacks) for callers that inject no coach-config
    — the engine wires the loaded bundle in for EVERY grounder path.
    """
    return AgentServices(
        planner=ModelPlanner(model, reference_date=reference_date),
        gateway=RegistryGateway(svc),
        coverage=DeterministicCoverage(),
        grounder=ClaimGrounder(
            model,
            svc,
            allow_names=allow_names,
            equivalence=equivalence,
            reference_date=reference_date,
            tolerance=tolerance,
            allowed_hosts=allowed_hosts,
            lookback_days=lookback_days,
        ),
    )


__all__ = [
    "CANONICAL_WORKOUT_NAMES",
    "ClaimGrounder",
    "CoachBundle",
    "DeterministicCoverage",
    "ModelPlanner",
    "RegistryGateway",
    "build_services",
]
