"""The externalized skill/prompt bundle: named, versioned, composable behavior units (§16).

This module owns the engine-side CLOSED schema + fail-closed loader for the coach-config behavior
assets the engine consumes as DATA (never as embedded code): the named system/agent prompt
fragments, the named grounding/abstention rules, the bundle manifest, and the array of named,
versioned **skills**. The engine source embeds NONE of these (CFG-R3 / ARCH-R29); they are loaded
at runtime from the config bundle (``[agent.coach.*]`` in ``defaults.toml``, overridable by the
operator file / a private bundle).

What this module adds on top of the raw settings maps (which are SHAPE-only, ``config/settings``):

* :class:`Skill` — the SKILL-R2/-R3a closed skill record: ``name`` (free-form stable id),
  ``version``, ``deliverable_type`` (the closed §12 enum), ``inputs``, ``capability_refs`` (subset
  of the single shared capability registry, PLAN-R3), ``tier_preference``/``effort_preference``
  (MODEL-R1), ``grounding_refs`` (named rules), and ``prompt_fragments`` (ordered NAMED fragment
  references — inline prompt text is a load error, SKILL-R3a).
* :class:`CoachManifest` — the loaded, validated bundle (SKILL-R3b): the manifest record + the
  resolved prompt fragments + grounding rules + the skill registry, with a :meth:`compose` helper
  that layers a skill's named fragments into a final prompt (persona -> shared preamble -> skill
  body, SKILL-R3).
* :func:`load_manifest` — SKILL-R4 / CFG-R6: validates every skill/prompt/manifest record against
  the closed schema and resolves every named reference (``capability_refs`` against the registry,
  ``grounding_refs`` against the loaded rules, ``prompt_fragments`` against the loaded prompts). A
  malformed record, an unknown field, or an UNRESOLVED reference (a skill citing a missing prompt,
  capability, or rule) FAILS CLOSED with a clear :class:`SkillBundleError` — the engine never
  starts on a partially-loaded or silently-defaulted behavior bundle.

Cited requirements: SKILL-R1, SKILL-R2, SKILL-R3, SKILL-R3a, SKILL-R3b, SKILL-R4, SKILL-R5,
SKILL-R6, PLAN-R3, CFG-R3, CFG-R4, CFG-R6, INJECT-R2, MODEL-R1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wattwise_core.agent.capabilities import CAPABILITY_BY_KEY


class SkillBundleError(RuntimeError):
    """A coach-config behavior bundle failed closed at load (CFG-R6 / SKILL-R4).

    Raised when a skill/prompt/manifest record is malformed, carries an unknown field, or — the
    central fail-closed case — references a prompt fragment, capability, or grounding rule that the
    bundle does not define. The engine MUST refuse to start rather than run a partially-loaded or
    silently-defaulted behavior bundle (SKILL-R4); it MUST NOT fall back to an embedded default that
    would mask a missing private bundle.
    """


class DeliverableType(StrEnum):
    """The closed deliverable kind a skill serves (SKILL-R3a; the COACH-R1 §12 deliverable set)."""

    WEEKLY_DIGEST = "weekly_digest"
    READINESS_FORM_ASSESSMENT = "readiness_form_assessment"
    MULTI_DAY_PLAN = "multi_day_plan"
    INSIGHT = "insight"
    BRIEFING = "briefing"


class TierPreference(StrEnum):
    """The model-tier preference a skill declares (MODEL-R1; bounded by the escalation ceiling)."""

    FLASH = "flash"
    PRO = "pro"
    FRONTIER = "frontier"


class EffortPreference(StrEnum):
    """The reasoning-effort preference a skill declares (MODEL-R1 orthogonal effort axis)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Skill(BaseModel):
    """One named, versioned, composable skill record (SKILL-R2 / SKILL-R3a, closed schema).

    The unit of composable agent behavior: a prompt/policy bundle declaring its identity + version,
    the deliverable it serves, the capability-registry entries it may request (PLAN-R3), its
    tier/effort preference, the grounding rules it applies, and the ordered prompt fragments it
    composes BY NAME. ``extra="forbid"`` rejects an unknown field at load (STRUCT-R3 fail-closed);
    cross-reference resolution (registry / rules / fragments) is enforced by :func:`load_manifest`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    deliverable_type: DeliverableType
    inputs: tuple[str, ...] = ()
    capability_refs: tuple[str, ...] = ()
    tier_preference: TierPreference = TierPreference.FLASH
    effort_preference: EffortPreference = EffortPreference.LOW
    grounding_refs: tuple[str, ...] = ()
    prompt_fragments: tuple[str, ...] = ()


class _ManifestRecord(BaseModel):
    """The closed manifest identity record (SKILL-R3b): bundle identity + schema-version target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_name: str = Field(min_length=1)
    bundle_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    default_language: str = Field(min_length=1)


#: The skill/manifest schema version THIS engine implements. A bundle whose manifest
#: ``schema_version`` does not match is rejected fail-closed (SKILL-R3b / CFG-R6) — the engine
#: refuses a bundle authored for an incompatible schema rather than silently coercing it.
SUPPORTED_SCHEMA_VERSION = "1"


class CoachManifest(BaseModel):
    """The loaded, validated coach behavior bundle (SKILL-R3b / SKILL-R4).

    Holds the manifest identity record, the resolved prompt-fragment map (name -> verbatim prompt
    body), the resolved grounding-rule map (name -> policy text), and the skill registry (name ->
    :class:`Skill`). Every cross-reference was resolved at load (:func:`load_manifest`); :meth:`get`
    selects a skill by name and :meth:`compose` layers a skill's named fragments into the final
    prompt (SKILL-R3 layering order).
    """

    model_config = ConfigDict(frozen=True)

    manifest: _ManifestRecord
    prompts: Mapping[str, str]
    grounding_rules: Mapping[str, str]
    skills: tuple[Skill, ...]

    def get(self, name: str) -> Skill | None:
        """Select the skill registered under ``name`` (SKILL-R5 selection), or ``None``."""
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def fragment(self, name: str) -> str:
        """Resolve a prompt fragment by name (every reference was validated at load)."""
        return self.prompts[name]

    def compose(self, skill_name: str) -> str:
        """Compose a skill's ordered prompt fragments into one final prompt (SKILL-R3 layering).

        The fragments are joined in the skill's declared order (the bundle authors the layering —
        persona -> shared preamble -> skill body, SKILL-R3); the run-time grounded data envelope is
        appended by the caller, not here. Every named fragment was resolved at load, so this never
        raises on a missing reference (CFG-R6 guaranteed the resolution).
        """
        skill = self.get(skill_name)
        if skill is None:
            raise SkillBundleError(f"no skill named {skill_name!r} in the loaded bundle")
        return "\n\n".join(self.prompts[name] for name in skill.prompt_fragments)


def load_manifest(
    *,
    prompts: Mapping[str, str],
    grounding_rules: Mapping[str, str],
    manifest: Mapping[str, str],
    skills: Sequence[Mapping[str, Any]],
) -> CoachManifest:
    """Load + validate the coach behavior bundle, failing closed on any flaw (SKILL-R4 / CFG-R6).

    Validates the manifest record + every skill record against the closed schema (unknown fields
    rejected, STRUCT-R3) and resolves EVERY named reference: ``capability_refs`` against the single
    shared capability registry (PLAN-R3), ``grounding_refs`` against ``grounding_rules``, and
    ``prompt_fragments`` against ``prompts``. A malformed record, an unknown field, an unsupported
    ``schema_version``, a duplicate skill name, or an UNRESOLVED reference (a skill citing a missing
    prompt/capability/rule) raises :class:`SkillBundleError` — the engine never starts on a
    partially-loaded or silently-defaulted bundle (SKILL-R4), and never falls back to an embedded
    default (SKILL-R6).
    """
    try:
        record = _ManifestRecord.model_validate(dict(manifest))
    except ValidationError as exc:
        raise SkillBundleError(f"invalid bundle manifest: {exc}") from exc
    if record.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise SkillBundleError(
            f"bundle targets skill/manifest schema_version {record.schema_version!r}, "
            f"engine supports {SUPPORTED_SCHEMA_VERSION!r} (CFG-R6 fail-closed)"
        )

    loaded: list[Skill] = []
    seen: set[str] = set()
    for raw in skills:
        try:
            skill = Skill.model_validate(dict(raw))
        except ValidationError as exc:
            raise SkillBundleError(f"invalid skill record {dict(raw)!r}: {exc}") from exc
        if skill.name in seen:
            raise SkillBundleError(
                f"duplicate skill name {skill.name!r} in the bundle (names MUST be unique)"
            )
        seen.add(skill.name)
        _resolve_references(skill, prompts=prompts, grounding_rules=grounding_rules)
        loaded.append(skill)

    return CoachManifest(
        manifest=record,
        prompts=dict(prompts),
        grounding_rules=dict(grounding_rules),
        skills=tuple(loaded),
    )


def _resolve_references(
    skill: Skill,
    *,
    prompts: Mapping[str, str],
    grounding_rules: Mapping[str, str],
) -> None:
    """Fail closed when a skill references a prompt/capability/rule the bundle lacks (CFG-R6).

    The central SKILL-R4 / CFG-R6 invariant: a skill MUST resolve every named reference. An
    out-of-registry ``capability_refs`` entry (PLAN-R3), a ``grounding_refs`` name with no loaded
    rule, or a ``prompt_fragments`` name with no loaded fragment is an internally-inconsistent
    bundle that MUST NOT boot — raise :class:`SkillBundleError`.
    """
    for capability in skill.capability_refs:
        if capability not in CAPABILITY_BY_KEY:
            raise SkillBundleError(
                f"skill {skill.name!r} references capability {capability!r} not in the "
                f"capability registry (PLAN-R3); the bundle is internally inconsistent"
            )
    for rule in skill.grounding_refs:
        if rule not in grounding_rules:
            raise SkillBundleError(
                f"skill {skill.name!r} references grounding rule {rule!r} with no loaded "
                f"rule of that name (CFG-R6 fail-closed)"
            )
    for fragment in skill.prompt_fragments:
        if fragment not in prompts:
            raise SkillBundleError(
                f"skill {skill.name!r} references prompt fragment {fragment!r} with no loaded "
                f"fragment of that name (CFG-R6 fail-closed)"
            )


__all__ = [
    "SUPPORTED_SCHEMA_VERSION",
    "CoachManifest",
    "DeliverableType",
    "EffortPreference",
    "Skill",
    "SkillBundleError",
    "TierPreference",
    "load_manifest",
]
