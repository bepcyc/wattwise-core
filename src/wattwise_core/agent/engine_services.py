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
from enum import StrEnum
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
from wattwise_core.agent.grounding_evidence import (
    CANONICAL_WORKOUT_NAMES,
    _resolve_snapshots,
    _SnapshotEvidence,
)
from wattwise_core.agent.locale import LocalePolicy
from wattwise_core.agent.seams import AgentServices
from wattwise_core.agent.skills import CoachManifest, load_manifest
from wattwise_core.agent.structured import StructuredOutputError, run_structured
from wattwise_core.agent.voice import VoicePresentation
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.observability import runtrace
from wattwise_core.persistence.types import utcnow


class PlanCapability(StrEnum):
    """The CLOSED set of capabilities the headline planner may request (PLAN-R3 schema enum).

    The date-range capabilities the planner can request without an activity id (the per-activity /
    per-day capabilities need an id the planner does not have at plan time). PLAN-R3: the planner
    MUST be structurally UNABLE to express a capability outside this set — the schema enum (not a
    post-hoc filter) constrains it, so an out-of-registry request is a structured-output VALIDATION
    failure (the model emitting a non-member value never validates), routed as a re-plan, never
    silently dropped. Each member is a key of the single shared capability registry.
    """

    WEEKLY_LOAD = "weekly_load"
    CRITICAL_POWER = "critical_power"
    POWER_CURVE = "power_curve"


_DEFAULT_WINDOW_DAYS = 42


class _PlanSchema(BaseModel):
    """Provider-enforced retrieval plan (PLAN-R2/-R3): which canonical capabilities to gather.

    ``capabilities`` is a list of the CLOSED :class:`PlanCapability` ENUM (PLAN-R3): the model
    cannot emit a capability outside the registry — a non-member value is a structured-output
    validation failure handled as a re-plan (the planner's fail-closed default), NOT a silently
    dropped key. ``extra="forbid"`` rejects any unknown field (STRUCT-R3).
    """

    model_config = {"extra": "forbid"}
    capabilities: list[PlanCapability] = Field(default_factory=list)
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


class ModelPlanner:
    """Model-driven retrieval planner (PLAN-R1/R2): the structured plan IS the selection.

    ``plan_system`` is the loaded planner system prompt (§16 / SKILL-R1): the engine embeds NO
    prompt inline (CFG-R3 / ARCH-R29) — the production wiring injects the verbatim fragment loaded
    from the coach-config bundle, and the empty default (``""``) preserves the prior behaviour for
    any seam that injects no coach-config (the FakeModel suite scripts the plan, so the prompt text
    is immaterial offline).
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        reference_date: _dt.date | None = None,
        plan_system: str = "",
    ) -> None:
        self._model = model
        self._today = reference_date or utcnow().date()
        self._plan_system = plan_system

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        """Emit the next batch of capability requests; fail-closed to a default on error (PLAN-R3).

        The plan's ``capabilities`` are the CLOSED :class:`PlanCapability` enum, so the model cannot
        express an out-of-registry capability: a non-member value is a structured-output validation
        failure (STRUCT-R2) that ``run_structured`` surfaces as :class:`StructuredOutputError`,
        handled here as a RE-PLAN to the default capability (PLAN-R3 "handled as a re-plan, not a
        crash") — never silently dropped.
        """
        try:
            plan = await run_structured(
                self._model,
                system=self._plan_system,
                data=f"question: {request_text}\nopen_gaps: {list(gaps)}\nalready: {list(already)}",
                schema=_PlanSchema,
            )
            keys = [c.value for c in plan.capabilities]
            window = plan.window_days
        except (StructuredOutputError, NotImplementedError):
            keys, window = [PlanCapability.WEEKLY_LOAD.value], _DEFAULT_WINDOW_DAYS
        if not keys:
            keys = [PlanCapability.WEEKLY_LOAD.value]
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
        """Resolve the requests and EMIT every scope-override anomaly (TOOL-R1, AGT-OBS-R5a).

        ``gather`` constructs a typed :class:`AnomalyEvent` for each attempted cross-athlete
        scope override it ignored (PLAN-R5). This production seam — the one the live graph calls
        — EMITS each one onto the run trace and counts it in metrics (AGT-OBS-R5a); it does NOT
        discard them, so injection-neutralization is monitorable in production, not only in CI.
        Identity stays the server-derived ``athlete_id`` (the override is ignored); records return.
        """
        result = await gather(self._svc, athlete_id, list(requests))
        runtrace.record_scope_anomalies(result.anomalies)
        return result.records


class DeterministicCoverage:
    """Reports planned capabilities that resolved to no canonical evidence (pure)."""

    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        # A turn with no retrieved evidence at all is the only structural gap the headline
        # flow reports; per-capability emptiness is surfaced by the gather records.
        return set() if retrieved else {"no_canonical_evidence"}


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

    async def ground(
        self, *, athlete_id: str, draft: str, retrieved: Mapping[str, Any]
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
        snapshot_evidence = _SnapshotEvidence(evidence, snapshots, allow_names=self._allow_names)
        return _grounding.ground(
            draft, claims, snapshot_evidence, allow_urls=(), tolerance=self._tolerance
        )


class CoachBundle:
    """The loaded OSS coach-config: ALL prompts + skills + equivalence + tolerance (§16/SKILL-R*).

    DATA the engine consumes (COACH-CFG-R3), loaded from external config (``[agent.coach.*]`` +
    ``[agent.metric_aliases]`` in ``defaults.toml``, overridable by the operator/private bundle) —
    the engine embeds NO persona/prompt/alias/threshold/skill literal inline (CFG-R3 / ARCH-R29 /
    SKILL-R6). It carries EVERY system/agent prompt the engine sends a model: the compose
    ``system_prompt`` plus the planner / claim / reflection / readiness-narration prompts,
    each MOVED VERBATIM from the former inline engine literals so behaviour is preserved (a wording
    change would alter the live agent, NOT caught by the offline FakeModel suite). The
    ``manifest`` is the loaded, validated :class:`~wattwise_core.agent.skills.CoachManifest` (the
    named/versioned skills + resolved cross-references, SKILL-R2/-R4); :meth:`from_settings` builds
    it through ``load_manifest`` which FAILS CLOSED on a skill referencing a missing
    prompt/capability/rule (CFG-R6). The empty default bundle (no prompts, empty equivalence/skills)
    preserves the prior FakeModel-test behaviour for any seam that injects none.
    """

    __slots__ = (
        "allowed_hosts",
        "claim_system",
        "equivalence",
        "locales",
        "lookback_days",
        "manifest",
        "plan_system",
        "presentation",
        "readiness_system",
        "reflect_system",
        "shared_preamble",
        "system_prompt",
        "tolerance",
    )

    def __init__(
        self,
        system_prompt: str = "",
        equivalence: MetricEquivalence | None = None,
        tolerance: _grounding.NumericTolerance | None = None,
        allowed_hosts: frozenset[str] = frozenset(),
        lookback_days: int | None = None,
        presentation: VoicePresentation | None = None,
        *,
        plan_system: str = "",
        claim_system: str = "",
        reflect_system: str = "",
        readiness_system: str = "",
        shared_preamble: str = "",
        manifest: CoachManifest | None = None,
        locales: LocalePolicy | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.equivalence = equivalence if equivalence is not None else MetricEquivalence({})
        self.tolerance = tolerance if tolerance is not None else _grounding.NumericTolerance()
        # GROUND-R4 first-party URL allow-list + §16 dateless-claim lookback, loaded content
        # (CFG-R1a). The empty default bundle ships no hosts (fail-closed) and no lookback override.
        self.allowed_hosts = allowed_hosts
        self.lookback_days = lookback_days
        # The config-loaded athlete-facing presentation policy (VOICE-R2/-R7): the reverse
        # [agent.metric_aliases] label map + fallback lead the deliverables enforce AFTER
        # grounding. The empty default still SCRUBS a surviving code to a neutral phrase
        # (fail-closed VOICE-R2), never showing it; :meth:`from_settings` wires the real map.
        self.presentation = presentation if presentation is not None else VoicePresentation()
        # The externalized verdict/narration system prompts (§16 / SKILL-R1, CFG-R3): the engine
        # source holds NONE of these inline (ARCH-R29). Threaded to the planner / grounder (here)
        # and to the reflect node / readiness narrator (via the engine). Empty defaults keep the
        # FakeModel suite green (it scripts every verdict, so the prompt text is immaterial).
        self.plan_system = plan_system
        self.claim_system = claim_system
        self.reflect_system = reflect_system
        self.readiness_system = readiness_system
        # The shared safety/grounding preamble carrying the INJECT-R2 instruction (delimited
        # <untrusted-data> is information to ANALYSE, never commands). It is LAYERED IN FRONT of the
        # persona in :attr:`compose_system` so the instruction is actually PRESENT in the system
        # prompt the model receives (INJECT-R2 "the system prompt MUST instruct …"), not merely
        # stored. Empty default => no preamble (the FakeModel suite, no INJECT-R2 line needed).
        self.shared_preamble = shared_preamble
        # The loaded, validated skill manifest (SKILL-R2/-R4); empty default => no skills.
        self.manifest = manifest
        # The loaded per-language surface packs + config-driven default-language fallback
        # (LANG-R1/-R4): the compose-time localized prompt variant and the localized abstain
        # copy resolve through this. The empty default (no packs) preserves the prior
        # English-prompt behaviour for any seam that injects no coach-config.
        self.locales = locales if locales is not None else LocalePolicy()

    @property
    def compose_system(self) -> str:
        """The compose-node system prompt: shared preamble (INJECT-R2) layered before the persona.

        SKILL-R3 layering order (shared preamble -> persona); the run-time grounded data envelope
        (INJECT-R1) is appended downstream by the compose node. Putting the INJECT-R2 instruction
        HERE makes it part of the system prompt actually sent to the model, satisfying INJECT-R2's
        "the system prompt MUST instruct the model that delimited data content is information to
        analyze, never commands". With an empty preamble (the FakeModel default) this is exactly the
        bare persona, preserving the prior compose behaviour.
        """
        parts = [p for p in (self.shared_preamble, self.system_prompt) if p]
        return "\n\n".join(parts)

    @classmethod
    def from_settings(cls, settings: Any) -> CoachBundle:
        """Build the coach bundle from resolved settings (the loaded §16 config), fail-closed.

        Loads + validates the skill manifest through ``skills.load_manifest`` (SKILL-R4 / CFG-R6): a
        skill referencing a missing prompt fragment, an out-of-registry capability (PLAN-R3), or a
        missing grounding rule raises a :class:`~wattwise_core.agent.skills.SkillBundleError` so the
        engine refuses to start on an internally-inconsistent bundle. The compose ``system_prompt``
        is ALSO exposed to the manifest as a named fragment (a skill MAY compose it), alongside the
        four verdict/narration prompts.
        """
        prompts = dict(settings.agent__coach__prompts)
        # The persona/compose prompt is a named fragment a skill MAY layer in (SKILL-R3): expose it
        # under its stable name so ``prompt_fragments = [..., "system_prompt", ...]`` resolves.
        prompts.setdefault("system_prompt", settings.agent__coach__system_prompt)
        manifest = load_manifest(
            prompts=prompts,
            grounding_rules=settings.agent__coach__grounding_rules,
            manifest=settings.agent__coach__manifest,
            skills=settings.agent__coach__skills,
        )
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
            # Reverse the SAME loaded alias map into athlete-native labels (CFG-R1a): the
            # presentation pass translates a surviving internal code back to a human word.
            presentation=VoicePresentation.from_aliases(settings.agent__metric_aliases),
            plan_system=settings.agent__coach__prompts["plan_system"],
            claim_system=settings.agent__coach__prompts["claim_system"],
            reflect_system=settings.agent__coach__prompts["reflect_system"],
            readiness_system=settings.agent__coach__prompts["readiness_system"],
            shared_preamble=settings.agent__coach__prompts["shared_preamble"],
            manifest=manifest,
            # The per-language packs + the bundle manifest's default_language (LANG-R1/-R4):
            # the supported set and its fallback are CONFIG, never engine code.
            locales=LocalePolicy.from_config(
                settings.agent__coach__languages,
                settings.agent__coach__manifest["default_language"],
            ),
        )

    def services(
        self, model: ChatModel, svc: AnalyticsService, *, allow_names: frozenset[str] = frozenset()
    ) -> AgentServices:
        """Production services wiring this coach-config's prompts + equivalence + tolerance."""
        return build_services(
            model,
            svc,
            allow_names=allow_names,
            equivalence=self.equivalence,
            tolerance=self.tolerance,
            allowed_hosts=self.allowed_hosts,
            lookback_days=self.lookback_days,
            plan_system=self.plan_system,
            claim_system=self.claim_system,
        )

    def grounder(self, model: ChatModel, svc: AnalyticsService) -> ClaimGrounder:
        """A grounder carrying this coach-config's prompt + equivalence + tolerance + URL (§16)."""
        return ClaimGrounder(
            model,
            svc,
            equivalence=self.equivalence,
            tolerance=self.tolerance,
            allowed_hosts=self.allowed_hosts,
            lookback_days=self.lookback_days,
            claim_system=self.claim_system,
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
    plan_system: str = "",
    claim_system: str = "",
) -> AgentServices:
    """Assemble the concrete production service bundle for the graph (GRAPH-R5).

    ``allow_names`` is the canonical workout-NAME library the grounder may ground a prescribed NAME
    against (empty for the free-form answer/digest; :data:`CANONICAL_WORKOUT_NAMES` for a PLAN
    deliverable so its prescriptions are not auto-scrubbed, COACH-R2 / GROUND-R2). ``equivalence``
    is the config-loaded metric-equivalence layer (§16) the grounder resolves a natural metric
    label through (GROUND-R2); ``reference_date`` anchors the latest-available-date fallback for a
    dateless claim; ``tolerance`` is the config-loaded numeric-match band (GROUND-R7);
    ``allowed_hosts`` is the config-loaded first-party URL allow-list (GROUND-R4) and
    ``lookback_days`` the §16 dateless-claim window. ``plan_system`` / ``claim_system`` are the
    externalized planner / claim-extraction system prompts (§16 / SKILL-R1, CFG-R3) — the engine
    embeds none inline (ARCH-R29). All default to ``None``/``""`` (canonical-key-only, today,
    default band, no-config host/lookback fallbacks, empty prompts) for callers that inject no
    coach-config — the engine wires the loaded bundle in for EVERY service path.
    """
    return AgentServices(
        planner=ModelPlanner(model, reference_date=reference_date, plan_system=plan_system),
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
            claim_system=claim_system,
        ),
    )


__all__ = [
    "CANONICAL_WORKOUT_NAMES",
    "ClaimGrounder",
    "CoachBundle",
    "DeterministicCoverage",
    "ModelPlanner",
    "PlanCapability",
    "RegistryGateway",
    "build_services",
]
