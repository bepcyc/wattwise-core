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

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from wattwise_core.observability import metrics as obs_metrics
from wattwise_core.observability.logging import get_logger

#: Strict IETF BCP-47-shaped tag gate for prompt interpolation (INJECT-R1): primary
#: subtag + optional alphanumeric subtags only — nothing else reaches the directive.
_IETF_TAG_RE = re.compile(r"[a-zA-Z]{2,8}(?:-[a-zA-Z0-9]{1,8})*")

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

# The deterministic readiness-sentence FLOOR for a caller with NO loaded language packs (#18 /
# GROUND-R6, COACH-R7, OUTCOME-R4): the per-verdict state leads, the truthful abstain/stale-abstain
# leads, and the HRV-missing / stale disclosure clauses — the SAME English text the readiness
# deliverable used to hold as code constants, now relocated so a loaded pack's localized
# ``readiness_*`` copy always wins (mirroring ``_LIMITATION_FLOOR``). Only the English floor lives
# in code; DE/RU surface only through loaded packs (CFG-R3 / LANG-R1). The keys mirror the
# ``[agent.coach.languages.<lang>]`` config keys (sans the ``readiness_`` prefix).
_READINESS_FLOOR: dict[str, str] = {
    "state_go": "You're fresh and ready for a hard day.",
    "state_maintain": "You're in a steady place — keep things as planned.",
    "state_ease": "You're carrying some fatigue, so ease off a little today.",
    "state_rest": "You're deep in fatigue right now, so today is for rest.",
    "abstain": "There isn't enough recent data to read your readiness yet.",
    "stale_abstain": (
        "I haven't seen any recent training data, so I can't read your readiness right now — "
        "if you've been training, it's worth checking that your data sync is still connected."
    ),
    "hrv_unavailable": "I don't have a recent HRV reading, so this is from your form.",
    "stale_data": "I haven't seen new training data in a few days, so this may lag where you are.",
}

#: The engine-baseline default language LANG-R4 pins for OSS ("en"); the LOADED bundle's
#: ``default_language`` (config, operator-overridable to de/ru) governs whenever a bundle is wired.
_BASELINE_DEFAULT_LANGUAGE = "en"


def _primary_subtag(locale: str | None) -> str:
    """The lowercase primary language subtag of a locale ("de-AT" -> "de"; empty -> "")."""
    return (locale or "").split("-", 1)[0].strip().lower()


@dataclass(frozen=True, slots=True)
class ReadinessCopy:
    """The readiness deliverable's localized surface sentences (#18 / COACH-R7, GROUND-R6).

    The catalog-keyed readiness strings — per-verdict state leads, the abstain / stale-abstain
    leads, and the HRV-missing / stale disclosure clauses — resolved for ONE language (loaded pack
    copy, else the English code floor). The readiness deliverable composes its fail-closed lead from
    THIS object instead of the former in-code English constants, so a localized run narrates its
    deterministic sentences in the requested language too. ``state_for`` maps a verdict name (the
    lowercase :class:`~wattwise_core.domain.enums.ReadinessVerdict` value) to its state lead.
    """

    state_go: str = ""
    state_maintain: str = ""
    state_ease: str = ""
    state_rest: str = ""
    abstain: str = ""
    stale_abstain: str = ""
    hrv_unavailable: str = ""
    stale_data: str = ""

    def state_for(self, verdict_value: str) -> str:
        """The per-verdict state lead for a ``ReadinessVerdict`` value (``go``/``maintain``/…)."""
        return {
            "go": self.state_go,
            "maintain": self.state_maintain,
            "ease": self.state_ease,
            "rest": self.state_rest,
        }[verdict_value]


@dataclass(frozen=True, slots=True)
class LanguagePack:
    """One language's loaded surface content (LANG-R1; config, never engine code).

    ``compose_directive`` is the localized prompt VARIANT layered into the compose system prompt
    at composition time (LANG-R3) — written in the target language so the model answers in it
    end-to-end; ``limitation`` is the localized fail-closed abstain copy (GROUND-R6).
    ``readiness`` carries this language's loaded readiness sentences (#18; empty fields fall back
    to the English code floor at resolution time).
    """

    compose_directive: str = ""
    limitation: str = ""
    readiness: ReadinessCopy = field(default_factory=ReadinessCopy)


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
    #: Config-gated generic any-language pass-through (accepted deviation from LANG-R1's
    #: packs-only reading): when enabled AND a requested language has no loaded pack, the compose
    #: prompt layers ``passthrough_directive`` (a config TEMPLATE; code interpolates the
    #: ``{language_tag}``/``{locale}`` placeholders) instead of the default pack's variant, so the
    #: coach answers IN the requested language. The LANG-R4 fallback is still RECORDED, loaded
    #: packs stay authoritative, and grounding/abstention copy are untouched.
    passthrough_enabled: bool = False
    passthrough_directive: str = ""

    @classmethod
    def from_config(
        cls,
        languages: Mapping[str, Mapping[str, Any]],
        default_language: str,
        *,
        passthrough_enabled: bool = False,
        passthrough_directive: str = "",
    ) -> LocalePolicy:
        """Build from the loaded ``[agent.coach.languages.*]`` tables (config content, LANG-R1).

        ``default_language`` (the bundle's LANG-R4 fallback) MUST itself be a loaded language —
        otherwise the fallback path would resolve to a language with no variant, so the bundle
        fails closed at load (SKILL-R4 spirit: never a partially-loaded behavior bundle).
        ``passthrough_enabled``/``passthrough_directive`` are the config-gated generic
        any-language pass-through (see the class attributes; both default OFF/empty so a caller
        wiring no pass-through keeps the strict packs-only behaviour).
        """
        packs = {
            _primary_subtag(lang): LanguagePack(
                compose_directive=str(table.get("compose_directive", "")),
                limitation=str(table.get("limitation", "")),
                # The localized readiness sentences (#18): loaded by the SAME catalog-key convention
                # as compose_directive/limitation — ``readiness_<key>`` in the language table. An
                # absent key stays "" and falls back to the English floor at resolution time.
                readiness=ReadinessCopy(
                    state_go=str(table.get("readiness_state_go", "")),
                    state_maintain=str(table.get("readiness_state_maintain", "")),
                    state_ease=str(table.get("readiness_state_ease", "")),
                    state_rest=str(table.get("readiness_state_rest", "")),
                    abstain=str(table.get("readiness_abstain", "")),
                    stale_abstain=str(table.get("readiness_stale_abstain", "")),
                    hrv_unavailable=str(table.get("readiness_hrv_unavailable", "")),
                    stale_data=str(table.get("readiness_stale_data", "")),
                ),
            )
            for lang, table in languages.items()
        }
        default = _primary_subtag(default_language) or _BASELINE_DEFAULT_LANGUAGE
        if packs and default not in packs:
            raise ValueError(f"default_language {default!r} has no loaded language pack (LANG-R4)")
        return cls(
            packs=packs,
            default_language=default,
            passthrough_enabled=passthrough_enabled,
            passthrough_directive=passthrough_directive,
        )

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

        When the config-gated generic pass-through is ON and the requested language has no
        loaded pack, the templated pass-through directive (interpolated with the requested
        tag/locale) is layered INSTEAD of the default pack's variant — the coach answers in the
        requested language while the fallback is still recorded and everything else (grounding,
        abstention copy, loaded packs) stays authoritative (accepted deviation from LANG-R1's
        packs-only reading).
        """
        directive = self._passthrough_directive(requested)
        if directive is None:
            lang = self.resolve_recorded(requested, surface="compose")
            directive = self.packs.get(lang, LanguagePack()).compose_directive
        else:
            # Still a LANG-R4 fallback event (no loaded pack for the request): record it for
            # observability even though the SURFACE answers in the requested language.
            self.resolve_recorded(requested, surface="compose")
        parts = [p for p in (persona, directive) if p]
        return "\n\n".join(parts)

    def _passthrough_directive(self, requested: str | None) -> str | None:
        """The interpolated pass-through directive, or ``None`` when the gate does not apply.

        Applies ONLY when: the gate is on, a template is loaded, the request names a language,
        packs are loaded (a no-bundle caller keeps the historical behaviour), and the requested
        language has NO loaded pack (loaded packs stay authoritative). A template whose
        placeholders fail to interpolate yields ``None`` (fail-closed to the default-pack path,
        never a half-rendered prompt).
        """
        lang = _primary_subtag(requested)
        if (
            not self.passthrough_enabled
            or not self.passthrough_directive
            or not lang
            or not self.packs
            or lang in self.packs
        ):
            return None
        # INJECT-R1/AGT-SEC: the requested tag is UNTRUSTED input headed for the system
        # prompt — only a strictly-valid IETF language tag is ever interpolated; anything
        # else (embedded newlines, prose, control text) collapses to the safe primary
        # subtag, so no caller-controlled instruction line can enter the directive.
        full_tag = (requested or "").strip()
        safe_locale = full_tag if _IETF_TAG_RE.fullmatch(full_tag) else lang
        try:
            return self.passthrough_directive.format(language_tag=lang, locale=safe_locale)
        except (KeyError, IndexError, ValueError):
            return None

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

        PASS-THROUGH languages (no pack) deliberately receive the DEFAULT language's copy: the
        abstain path is deterministic and model-free by design (GROUND-R6 — a failed run may
        not call the model again), so an any-language limitation cannot be generated here. The
        abstaining deliverable body is the limitation alone (no localized prose beside it), so
        no single deliverable mixes languages; the fallback is RECORDED (LANG-R4) like every
        other one.
        """
        if not self.packs:
            requested_lang = _primary_subtag(requested)
            return _LIMITATION_FLOOR.get(requested_lang, _LIMITATION_FLOOR["en"])
        lang = self.resolve_recorded(requested, surface="limitation")
        loaded = self.packs.get(lang, LanguagePack()).limitation
        return loaded or _LIMITATION_FLOOR.get(lang, _LIMITATION_FLOOR["en"])

    def readiness_copy(self, requested: str | None) -> ReadinessCopy:
        """The resolved localized readiness sentences for the run (#18 / COACH-R7, GROUND-R6).

        Resolves the run's language through the SAME LANG-R4 fallback (recorded), then returns a
        :class:`ReadinessCopy` whose every field is the loaded pack's localized sentence when
        present, else the deterministic English code FLOOR (``_READINESS_FLOOR``) — so a readiness
        run NEVER ships an empty or English-by-construction sentence on a supported localized run,
        and a no-bundle caller keeps the historical English sentences. The EMPTY policy (no packs)
        keys the floor on the REQUESTED language directly, mirroring :meth:`limitation`; since only
        the English floor exists in code, an unsupported/no-bundle locale resolves to English (the
        deterministic abstain path is model-free and cannot be generated per-language here).
        """
        if not self.packs:
            return _readiness_floor_copy()
        lang = self.resolve_recorded(requested, surface="readiness")
        loaded = self.packs.get(lang, LanguagePack()).readiness
        floor = _READINESS_FLOOR
        return ReadinessCopy(
            state_go=loaded.state_go or floor["state_go"],
            state_maintain=loaded.state_maintain or floor["state_maintain"],
            state_ease=loaded.state_ease or floor["state_ease"],
            state_rest=loaded.state_rest or floor["state_rest"],
            abstain=loaded.abstain or floor["abstain"],
            stale_abstain=loaded.stale_abstain or floor["stale_abstain"],
            hrv_unavailable=loaded.hrv_unavailable or floor["hrv_unavailable"],
            stale_data=loaded.stale_data or floor["stale_data"],
        )


def _readiness_floor_copy() -> ReadinessCopy:
    """The English readiness FLOOR as a :class:`ReadinessCopy` (the no-bundle / unsupported path)."""
    return ReadinessCopy(
        state_go=_READINESS_FLOOR["state_go"],
        state_maintain=_READINESS_FLOOR["state_maintain"],
        state_ease=_READINESS_FLOOR["state_ease"],
        state_rest=_READINESS_FLOOR["state_rest"],
        abstain=_READINESS_FLOOR["abstain"],
        stale_abstain=_READINESS_FLOOR["stale_abstain"],
        hrv_unavailable=_READINESS_FLOOR["hrv_unavailable"],
        stale_data=_READINESS_FLOOR["stale_data"],
    )


#: The shared no-bundle policy (module-level so isolated callers get ONE stable instance).
EMPTY_LOCALE_POLICY = LocalePolicy()


__all__ = [
    "EMPTY_LOCALE_POLICY",
    "LanguagePack",
    "LocalePolicy",
    "ReadinessCopy",
]
