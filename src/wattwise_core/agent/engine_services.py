"""Concrete production agent services: planner, gateway, coverage, grounder (doc 50).

The focused sibling of :mod:`wattwise_core.agent.engine` (QUAL-R9 size split) that owns the
CONCRETE production implementations of the injected agent seams the graph runs on — the canonical
capability gateway (TOOL-R1), a deterministic coverage assessor, and a model-extract + code-verify
grounder over canonical evidence (GROUND-R1/R2/R7) — plus the claim structured-output schemas the
model fills and the ``build_services`` bundle assembler. The model-driven retrieval planner +
its closed plan schema live in the focused :mod:`engine_planner` sibling (QUAL-R9 size split)
and are re-exported here. ``engine`` imports these and re-exports the public ones
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

from wattwise_core.agent import grounding as _grounding
from wattwise_core.agent.capabilities import (
    MetricEquivalence,
    gather,
)
from wattwise_core.agent.contracts import ChatModel, RetrievalRequest
from wattwise_core.agent.engine_planner import ModelPlanner, PlanCapability
from wattwise_core.agent.engine_planner import (
    _PlanSchema as _PlanSchema,  # noqa: PLC0414  explicit re-export (historical import path)
)
from wattwise_core.agent.grounding_evidence import (
    CANONICAL_WORKOUT_NAMES,
    ClaimGrounder,
    WorkoutEquivalence,
)
from wattwise_core.agent.locale import LocalePolicy
from wattwise_core.agent.seams import AgentServices
from wattwise_core.agent.skills import CoachManifest, load_manifest
from wattwise_core.agent.voice import VoicePresentation
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.observability import runtrace


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
        "detailed_compose_directive",
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
        "workout_equivalence",
    )

    def __init__(
        self,
        system_prompt: str = "",
        equivalence: MetricEquivalence | None = None,
        workout_equivalence: WorkoutEquivalence | None = None,
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
        detailed_compose_directive: str = "",
    ) -> None:
        self.system_prompt = system_prompt
        self.equivalence = equivalence if equivalence is not None else MetricEquivalence({})
        # The config-loaded multilingual workout-name equivalence (#17 / GROUND-R2): resolves a
        # localized prescription NAME claim onto the canonical English id so a non-English PLAN
        # grounds instead of scrubbing every workout name. The empty default (no aliases, only the
        # CANONICAL_WORKOUT_NAMES floor) preserves the prior English-only behaviour for no-bundle
        # seams; :meth:`from_settings` wires the loaded [agent.workout_aliases] table.
        self.workout_equivalence = (
            workout_equivalence
            if workout_equivalence is not None
            else WorkoutEquivalence({}, CANONICAL_WORKOUT_NAMES)
        )
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
        # The DETAILED-length compose steering fragment (VOICE-R7/-R8): layered after the
        # localized system prompt ONLY when the run asked for a detailed answer, steering the
        # model to weave up to the detailed number cap of grounded figures into the prose
        # (a detailed deep-dive with zero cited numbers is under-informative). Loaded content
        # (CFG-R1a); the empty default keeps the prior compose behaviour for no-bundle seams.
        self.detailed_compose_directive = detailed_compose_directive

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
            # The loaded multilingual workout-name table (#17): localized name -> canonical English
            # name, validated against the CANONICAL_WORKOUT_NAMES floor (a misconfigured value is
            # dropped, fail-closed). The PLAN grounder resolves prescription names through this.
            workout_equivalence=WorkoutEquivalence(
                settings.agent__workout_aliases, CANONICAL_WORKOUT_NAMES
            ),
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
                # The config-gated generic any-language pass-through (accepted deviation from
                # LANG-R1 packs-only; LANG-R4 fallback still recorded): template + gate are
                # loaded content (CFG-R1a), code only interpolates the language tag/locale.
                passthrough_enabled=settings.agent__coach__language_passthrough,
                passthrough_directive=settings.agent__coach__language_passthrough_directive,
            ),
            detailed_compose_directive=settings.agent__coach__prompts["detailed_compose_directive"],
        )

    def services(
        self, model: ChatModel, svc: AnalyticsService, *, allow_names: frozenset[str] = frozenset()
    ) -> AgentServices:
        """Production services wiring this coach-config's prompts + equivalence + tolerance."""
        return build_services(
            model,
            svc,
            allow_names=allow_names,
            # The PLAN path passes allow_names=CANONICAL_WORKOUT_NAMES; threading the loaded
            # workout-equivalence here is what lets a localized prescription NAME ground (#17). A
            # free-form (empty-allow_names) path still scrubs every NAME (the equivalence only
            # resolves canonical concepts, never invents one), so behaviour there is unchanged.
            workout_equivalence=self.workout_equivalence if allow_names else None,
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
    workout_equivalence: WorkoutEquivalence | None = None,
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
            workout_equivalence=workout_equivalence,
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
