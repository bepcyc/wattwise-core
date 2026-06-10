"""Language selection + the localized prompt-variant / fallback policy (LANG-R1/-R3/-R4).

This LEAF module owns the engine side of multilingual voice: resolving the athlete's selected
``locale`` against the SUPPORTED language set (LANG-R1), choosing the localized prompt VARIANT
layered into the compose system prompt at composition time (LANG-R3 — language drives only the
surface rendering, never grounding/identity/registry/verdicts), and the config-driven
``default_language`` fallback when a requested language has no loaded variant (LANG-R4), with the
fallback RECORDED for observability (a log line + a metrics counter, §15) and never a mixed-language
or untranslated-internal-string deliverable.

ALL language content is loaded config (LANG-R1: adding a language is a config/content concern, not
an engine code change; CFG-R3/ARCH-R29: no prompt body inline): the per-language packs come from the
``[agent.coach.languages.<lang>]`` tables of the coach bundle, each carrying the localized compose
directive (the per-language prompt variant) and the localized fail-closed limitation copy
(GROUND-R6). The only in-code copy is the deterministic English LIMITATION FLOOR — the safety net an
isolated caller with NO loaded bundle (the FakeModel test seam) degrades to, so an abstaining run
can never ship an empty body; a loaded bundle's packs always override it.

This module depends only on the observability layer (no agent sibling imports, ARCH-R21).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from wattwise_core.observability import metrics as obs_metrics
from wattwise_core.observability.logging import get_logger

_logger = get_logger(__name__)

# The deterministic fail-closed limitation FLOOR for a caller with NO loaded language packs
# (GROUND-R6, VOICE-R2/R3): jargon-free, warm, truthful. A loaded bundle's per-language
# ``limitation`` copy (config content, LANG-R1) always takes precedence; this exists only so an
# abstaining run can never ship an empty body when no bundle is wired (the FakeModel test seam).
_LIMITATION_FLOOR = {
    "en": "I don't have enough confirmed data to answer that reliably yet. "
    "Sync your sources and I'll take another look.",
    "de": "Mir fehlen noch genug gesicherte Daten, um das verlaesslich zu beantworten. "
    "Synchronisiere deine Quellen und ich schaue noch einmal.",
    "ru": "Poka nedostatochno podtverzhdyonnyh dannyh, chtoby otvetit' nadyozhno. "
    "Sinhroniziruj istochniki i ya posmotryu snova.",
}

#: The engine-baseline default language LANG-R4 pins for OSS ("en"); the LOADED bundle's
#: ``default_language`` (config, operator-overridable to de/ru) governs whenever a bundle is wired.
_BASELINE_DEFAULT_LANGUAGE = "en"


def _primary_subtag(locale: str | None) -> str:
    """The lowercase primary language subtag of a locale ("de-AT" -> "de"; empty -> "")."""
    return (locale or "").split("-", 1)[0].strip().lower()


@dataclass(frozen=True, slots=True)
class LanguagePack:
    """One language's loaded surface content (LANG-R1; config, never engine code).

    ``compose_directive`` is the localized prompt VARIANT layered into the compose system prompt
    at composition time (LANG-R3) — written in the target language so the model answers in it
    end-to-end; ``limitation`` is the localized fail-closed abstain copy (GROUND-R6).
    """

    compose_directive: str = ""
    limitation: str = ""


@dataclass(frozen=True, slots=True)
class LocalePolicy:
    """The loaded language set + the config-driven fallback resolution (LANG-R1/-R4).

    ``packs`` maps a primary language subtag to its loaded :class:`LanguagePack`; the supported
    set IS the keys (LANG-R1 owns the set). ``default_language`` is the config-driven surface
    fallback an unsupported request resolves to (LANG-R4). The EMPTY policy (no packs — a caller
    wiring no bundle) resolves everything to the baseline default and falls back to the in-code
    limitation floor, preserving the prior FakeModel-suite behaviour.
    """

    packs: Mapping[str, LanguagePack] = field(default_factory=dict)
    default_language: str = _BASELINE_DEFAULT_LANGUAGE

    @classmethod
    def from_config(
        cls, languages: Mapping[str, Mapping[str, Any]], default_language: str
    ) -> LocalePolicy:
        """Build from the loaded ``[agent.coach.languages.*]`` tables (config content, LANG-R1).

        ``default_language`` (the bundle's LANG-R4 fallback) MUST itself be a loaded language —
        otherwise the fallback path would resolve to a language with no variant, so the bundle
        fails closed at load (SKILL-R4 spirit: never a partially-loaded behavior bundle).
        """
        packs = {
            _primary_subtag(lang): LanguagePack(
                compose_directive=str(table.get("compose_directive", "")),
                limitation=str(table.get("limitation", "")),
            )
            for lang, table in languages.items()
        }
        default = _primary_subtag(default_language) or _BASELINE_DEFAULT_LANGUAGE
        if packs and default not in packs:
            raise ValueError(f"default_language {default!r} has no loaded language pack (LANG-R4)")
        return cls(packs=packs, default_language=default)

    def resolve(self, requested: str | None) -> tuple[str, bool]:
        """Resolve a requested locale to ``(language, fallback_used)`` (LANG-R4).

        A requested language with a loaded pack resolves to itself; anything else (an
        unsupported language, or no request at all) resolves to the config-driven
        ``default_language``. ``fallback_used`` is True ONLY for a non-empty request that had no
        loaded variant — an absent/unset locale is the documented presentation default
        (LANG-R4: the engine surface default), not a fallback event.
        """
        lang = _primary_subtag(requested)
        if lang and lang in self.packs:
            return lang, False
        return self.default_language, bool(lang) and bool(self.packs)

    def compose_system(self, persona: str, requested: str | None) -> str:
        """The compose system prompt with the localized variant layered in (LANG-R3).

        Resolves the run's language, RECORDS a fallback when the requested language had no
        loaded variant (LANG-R4 observability: log + metric — never surfaced to the athlete),
        and appends ONE resolved language's compose directive after the persona (SKILL-R3
        layering; one language per deliverable — never mixed). With no pack for the resolved
        language (the empty policy) the persona is returned unchanged.
        """
        lang = self.resolve_recorded(requested, surface="compose")
        directive = self.packs.get(lang, LanguagePack()).compose_directive
        parts = [p for p in (persona, directive) if p]
        return "\n\n".join(parts)

    def resolve_recorded(self, requested: str | None, *, surface: str) -> str:
        """Resolve a locale and RECORD any fallback for observability (LANG-R4, §15)."""
        lang, fallback = self.resolve(requested)
        if fallback:
            _logger.info(
                "language fallback",
                requested_language=_primary_subtag(requested),
                resolved_language=lang,
                source=surface,
            )
            obs_metrics.get_registry().increment(
                obs_metrics.LANGUAGE_FALLBACKS,
                labels={"requested": _primary_subtag(requested), "resolved": lang},
            )
        return lang

    def limitation(self, requested: str | None) -> str:
        """The localized fail-closed limitation copy for an abstaining run (GROUND-R6).

        Resolves the language through the same LANG-R4 fallback (recorded), preferring the
        loaded pack's config copy and degrading to the deterministic in-code floor only when no
        pack carries one (the no-bundle seam) — never an empty or mixed-language body. The
        EMPTY policy (no packs) keys the floor on the REQUESTED language directly, preserving
        the engine's historical localized floor for an isolated caller.
        """
        if not self.packs:
            requested_lang = _primary_subtag(requested)
            return _LIMITATION_FLOOR.get(requested_lang, _LIMITATION_FLOOR["en"])
        lang = self.resolve_recorded(requested, surface="limitation")
        loaded = self.packs.get(lang, LanguagePack()).limitation
        return loaded or _LIMITATION_FLOOR.get(lang, _LIMITATION_FLOOR["en"])


#: The shared no-bundle policy (module-level so isolated callers get ONE stable instance).
EMPTY_LOCALE_POLICY = LocalePolicy()


__all__ = [
    "EMPTY_LOCALE_POLICY",
    "LanguagePack",
    "LocalePolicy",
]
