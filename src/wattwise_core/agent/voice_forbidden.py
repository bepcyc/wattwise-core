"""VOICE-R2/-R2a internal-vocabulary SCHEMA + the closed-vocabulary deny pass (#98).

The focused leaf of the voice layer (ARCH-R21 / QUAL-R9 size split) that owns the engine's
internal-vocabulary SCHEMA (the metric codes + the jargon/enum words the code is structurally not
allowed to read out) AND the deterministic "scrub anything the allow-list does not keep" machinery
:mod:`wattwise_core.agent.voice` runs inside ``enforce_presentation`` after metric-code translation:

1. the engine-emittable internal enum/jargon SCHEMA (:data:`INTERNAL_ENUM_TOKENS`) plus the
   deployment-configured per-language forbidden terms — removed as exact word-bounded tokens
   (:func:`scrub_forbidden`);
2. internal-identifier-SHAPED tokens of the ``snake_case`` / requirement-id forms ONLY
   (:func:`scrub_identifier_shapes`).

It imports NOTHING from :mod:`wattwise_core.agent.voice` — the scrub functions take the
deployment ``forbidden_terms`` frozenset directly, not the ``VoicePresentation`` — so it sits
strictly BELOW ``voice`` (no cycle). The realization is KEEP-by-default (the visible layer is a
projection of grounded canonical content per COMPOSE-R3); this is the fail-closed deny BACKSTOP,
never a literal multilingual allow-list. Cited: VOICE-R2/-R2a, COMPOSE-R3, VOICE-R4 (abbrevs kept).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

# The closed set of INTERNAL metric CODES the athlete-facing voice must NEVER surface in prose
# (VOICE-R2: "ctl/atl/tsb are internal tokens"). Engine SCHEMA (the names the code is structurally
# not allowed to read out), NOT a config value. Matched case-folded, word-bounded; the athlete-
# NATIVE label each CODE translates TO is config-loaded (the reverse [agent.metric_aliases] map).
#
# DELIBERATELY EXCLUDED — athlete-native words the spec WANTS surfaced, NOT codes (VOICE-R4):
# "form"/"freshness" (the word for TSB), "hrv" (VOICE-R4 surfaces "your HRV"), "if" (the English
# word is far too common to scrub safely).
INTERNAL_METRIC_TOKENS: frozenset[str] = frozenset(
    {
        "ctl",
        "atl",
        "tsb",
        "tss",
        "np",
        "cp",
        "critical_power_w",
        "w_prime_j",
        "wprime",
        "w'",
        "hrv_rmssd_ms",
        "rmssd",
    }
)

# The closed set of INTERNAL ENGINE jargon / enum-value words the athlete-facing voice must NEVER
# surface (VOICE-R2 / VOICE-R2a): the implementation/architecture vocabulary the engine can itself
# emit. Engine SCHEMA (so spec<->code correspondence is checkable), NOT config. Scrubbed as EXACT
# word-bounded tokens (incl. an English plural and the multi-word/hyphen spellings the spec lists),
# never by shape. A NOVEL internal word the engine begins emitting MUST be added here in the same
# change that lets it emit it (VOICE-R2a residual edict).
#
# DELIBERATELY EXCLUDED — the VOICE-R2 forbidden words that are ALSO everyday EN/DE coach prose
# ("weekly cap", "a model of consistency", "pro Woche" = German "per week", "flash of speed", "the
# tool that makes you faster", "thread of consistency", "the endpoint of your interval", "completed
# a great block"): cap, tool, model, tier, flash, pro, frontier, cost, budget, coverage, fidelity,
# thread, checkpoint, endpoint, completed, degraded. Hardcoding these would mangle real coach
# answers (adversarial + review finding, #98), so they are scrubbed ONLY when a deployment opts in
# via ``VoicePresentation.forbidden_terms`` (empty default), otherwise bounded by the COMPOSE-R3
# grounded-projection invariant. Snake_case members are also caught by the identifier-shape pass —
# listing them here is belt-and-suspenders for the bare form AND covers the spaced/hyphen spelling.
INTERNAL_ENUM_TOKENS: frozenset[str] = frozenset(
    {
        "api",
        "database",
        "schema",
        "mcp",
        "rag",
        "adapter",
        "grounding",
        "scrub",
        "token",
        "gbo",
        "redraft",
        "replan",
        "regenerate",
        "technical_proof",
        "vector_store",
        "vector store",
        "canonical_store",
        "canonical-store",
        "canonical store",
        "source_descriptor",
        "source descriptor",
        "reasoning_effort",
        "deliverable_type",
        "visible_answer",
        "capability_registry",
        "coverage_caveat",
        "from_fidelity",
        "reveal_numbers",
        "awaiting_approval",
        "budget_exceeded",
    }
)

# Internal-identifier SHAPES that NEVER occur in legitimate EN/DE/RU coach prose (VOICE-R2a, #98):
# a snake_case identifier (>=2 ASCII-lowercase[+digit] segments joined by ``_``, e.g.
# ``weekly_digest``/``thread_id``) and a requirement-id (``GROUND-R3``/``VOICE-R7``). DELIBERATELY
# NOT ALL_CAPS / camelCase / dotted-path: an ALL_CAPS rule re-bans the athlete-native abbreviations
# VOICE-R4 surfaces (HRV/FTP/PR/SST/HIT) and DE/RU sport abbreviations (HFV/ПАНО/МПК/ЧСС), and
# camelCase fires on names. A hyphen-compound ("warm-up"), a date ("2026-06-17"), a time ("7:30"),
# a decimal, and a possessive ("athlete's") are all left intact.
_IDENTIFIER_SHAPE_RE = re.compile(
    r"(?<![\w'])(?:[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z]{2,}-R\d+[a-z]?)(?![\w])"
)
# Tidy passes after a removal so a scrub never leaves a visible gap or orphan punctuation.
_EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
_ORPHAN_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_DOUBLE_SPACE_RE = re.compile(r"[ \t]{2,}")


@lru_cache(maxsize=64)
def _compile_forbidden(extra: frozenset[str]) -> re.Pattern[str] | None:
    """Word-bounded alternation over ``INTERNAL_ENUM_TOKENS`` + the deployment ``extra`` terms.

    Longest-phrase-first (a multi-word term wins over a substring), case-insensitive, with the same
    markdown-emphasis guard as the metric-token regex and an optional English plural so ``token`` ->
    ``tokens`` is caught (non-English inflections need explicit enumeration in ``extra``). Cached on
    the (frozen, hashable) ``extra`` set; returns ``None`` only for an empty union (never in
    practice — the schema set is non-empty).
    """
    terms = INTERNAL_ENUM_TOKENS | extra
    if not terms:
        return None
    alt = "|".join(sorted((re.escape(t) for t in terms), key=len, reverse=True))
    return re.compile(
        rf"(?<![\w'])(?:\*\*|__|[*_`])?(?:{alt})(?:s|es)?(?:\*\*|__|[*_`])?(?![\w])",
        flags=re.IGNORECASE,
    )


def scrub_forbidden(
    body: str,
    forbidden_terms: frozenset[str],
    on_scrub: Callable[[str, str], None] | None = None,
) -> str:
    """REMOVE every internal-enum / configured-forbidden term (VOICE-R2/-R2a).

    Unlike a metric CODE (which translates to an athlete word), these have no athlete-native
    synonym, so a match is removed outright ("when in doubt, omit", VOICE-R2). The caller runs this
    AFTER metric translation (so a translated word is never in the set) and BEFORE lead repair (so a
    lead emptied by a scrub falls through to the warm fallback opener). ``forbidden_terms`` is the
    deployment's optional per-language extension. ``on_scrub`` is the VOICE-R2a/AGT-OBS-R4
    observability seam — the engine MAY pass it to record each removal.
    """
    pattern = _compile_forbidden(forbidden_terms)
    if pattern is None:
        return body
    if on_scrub is not None:
        for match in pattern.finditer(body):
            on_scrub(match.group(0).strip("*_` "), "voice-r2-forbidden")
    return pattern.sub("", body)


def scrub_identifier_shapes(body: str, on_scrub: Callable[[str, str], None] | None = None) -> str:
    """REMOVE internal-identifier-SHAPED tokens (snake_case / requirement-id) (VOICE-R2a)."""
    if on_scrub is not None:
        for match in _IDENTIFIER_SHAPE_RE.finditer(body):
            on_scrub(match.group(0), "internal-identifier-shape")
    return _IDENTIFIER_SHAPE_RE.sub("", body)


def normalize_after_scrub(body: str) -> str:
    """Tidy whitespace/punctuation a removal left behind (no visible gap, no orphan punctuation)."""
    body = _EMPTY_PARENS_RE.sub("", body)
    body = _ORPHAN_PUNCT_RE.sub(r"\1", body)
    body = _DOUBLE_SPACE_RE.sub(" ", body)
    return body.strip()


__all__ = [
    "INTERNAL_ENUM_TOKENS",
    "INTERNAL_METRIC_TOKENS",
    "normalize_after_scrub",
    "scrub_forbidden",
    "scrub_identifier_shapes",
]
