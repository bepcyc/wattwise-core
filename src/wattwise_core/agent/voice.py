"""Leaf voice/projection layer: the shared coach-voice primitives (VOICE-R7/-R8).

This is the LEAF module of the deliverables family (ARCH-R21 / QUAL-R9): it owns the
voice-contract primitives that BOTH :mod:`wattwise_core.agent.deliverables` (the
free-form answer + weekly digest) and :mod:`wattwise_core.agent.readiness_deliverable`
(the readiness/form deliverable) build on — the grounded-citation shape
(:class:`Citation`), the per-turn observation (:class:`Observation`), the response-length
verbosity knob (:data:`ResponseLength` + :func:`number_cap`), and the DETERMINISTIC
presentation checks/enforcement (leads-with-state, foregrounded-number count, number-cap
demotion, citation projection).

It imports NOTHING from any sibling deliverable / engine / api / persistence module — only
stdlib (and, by contract, ``agent/contracts``/pydantic when needed) — so it sits strictly
BELOW both deliverable modules in the import graph. Hoisting these shared primitives here
(rather than into one deliverable that the other imports back) is what breaks the former
``deliverables`` <-> ``readiness_deliverable`` cycle: both now depend DOWNWARD on this leaf,
and ``deliverables`` re-exports these names so every historical import path stays stable.

The voice contract is a PRESENTATION layer over the graph's fail-closed grounding, never a
relaxation of it (VOICE-R7): this module rewrites no number and certifies no groundedness —
it projects what the graph grounded and runs the deterministic leads-with-state /
number-count checks that gate the two presentation properties (EVAL-R5b.1).

Cited requirements: COACH-R7, COACH-R8, GROUND-R5/-R7, VOICE-R7/-R8/-R9, EVAL-R5b.1.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

# Athlete-facing verbosity (VOICE-R8); the persisted default is ``standard``.
ResponseLength = Literal["short", "standard", "detailed"]

# The closed set of INTERNAL metric CODES the athlete-facing voice must NEVER surface in prose
# (VOICE-R2: "ctl/atl/tsb are internal tokens"). This is the engine's metric SCHEMA — the code
# names it is structurally not allowed to read out — not a config value (CFG-R1a is about
# models/URLs/budgets/ports/paths). Matched case-folded, word-bounded.
#
# DELIBERATELY EXCLUDED — athlete-native words the spec WANTS surfaced, NOT codes (VOICE-R4):
#   * "form" / "freshness" — the athlete-native word for TSB (VOICE-R4 "freshness or form");
#   * "hrv" — VOICE-R4 surfaces "your HRV" to the athlete directly;
#   * "if" — the English word "if" is far too common to scrub safely (false positives).
# The athlete-NATIVE label each canonical CODE translates TO is config-loaded (the reverse
# [agent.metric_aliases] map on :class:`VoicePresentation`), so a deployment re-words
# "fitness"/"fatigue"/"freshness" without an engine change; only the forbidden-CODE vocabulary
# lives here as schema.
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

# Number-density CAP per response length (VOICE-R7 defaults; exact ceilings live in
# the loaded persona config, so callers MAY override via ``number_cap``).
_NUMBER_CAP: Mapping[ResponseLength, int] = {"short": 2, "standard": 3, "detailed": 4}

# Foregrounded-number caps by response length and coach numeric-detail level. Level 3 preserves
# the historical response-length-only caps; lower levels make the answer more human-first, while
# higher levels permit pro-style metric density without relaxing grounding.
_NUMERIC_DETAIL_CAPS: Mapping[ResponseLength, Mapping[int, int]] = {
    "short": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    "standard": {1: 0, 2: 1, 3: 3, 4: 5, 5: 7},
    "detailed": {1: 1, 2: 2, 3: 4, 4: 7, 5: 10},
}

# Matches a foregrounded explicit numeric value in athlete-facing prose for the
# deterministic number-density count (VOICE-R7 / EVAL-R5b.1). Plain integers and
# decimals, optionally signed; standalone, so dates/words are not miscounted.
#
# The trailing guard ``(?!\.?\d)(?!\w)`` rejects only a CONTINUATION of the number — a
# following digit, or a decimal point that itself precedes a digit (so ``3`` inside
# ``3.14`` / ``v1.2.3`` is not counted as a bare ``3``) — and any word char. It deliberately
# does NOT reject a following SENTENCE PERIOD: the prior guard ``(?![\w.])`` did, which
# silently UNDERCOUNTED a number ending a sentence (``"your fitness is 62."`` counted 0),
# making the number-cap gate too lenient (a real foregrounded number escaped the count).
# This fix makes the deterministic cap check (EVAL-R5b.1) non-vacuous for sentence-final numbers.
_NUMBER_RE = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?!\.?\d)(?!\w)")

# Tags stripped to read the LEADING athlete-facing sentence out of grounded HTML for
# the deterministic leads-with-state check (the body is sanitized later by the API).
_TAG_RE = re.compile(r"<[^>]+>")

# Report-frame lead patterns that FAIL the leads-with-state gate (COACH-R7 / VOICE-R7): a
# leading sentence that introduces a data/metrics readout ("here is your training-load
# picture", "here are your latest metrics/numbers", "your current numbers are", a bare
# "metrics:" / "numbers:" intro, or a colon-terminated list intro) is NOT a state read — it
# reads like a cited-metrics report, which VOICE-R7 forbids. These match what a LEAD SENTENCE
# may not be; a normal warm state sentence that merely happens to contain one of these words
# elsewhere is unaffected because the gate inspects only the FIRST sentence.
_REPORT_FRAME_RE = re.compile(
    r"\b(?:here\s+(?:is|are|'?s)|this\s+is)\b[^.!?]*?\b"
    r"(?:picture|metrics?|numbers?|values?|stats?|figures?|breakdown|snapshot|readout|"
    r"summary|overview|data|report|rundown)\b"
    r"|^\s*(?:metrics?|numbers?|values?|stats?|figures?)\s*:"
    r"|\byour\s+(?:current|latest)\b[^.!?]*?\b"
    r"(?:metrics?|numbers?|values?|stats?|figures?|picture|snapshot|readout|breakdown)\b",
    flags=re.IGNORECASE,
)

# A lead that ends in a colon (a "...:" list intro, the bullet-list report frame) is a
# metrics-report frame, not a state read (COACH-R7). The first "sentence" of such a body has
# no . ! ? terminator, so :func:`first_sentence` returns the whole pre-list run ending in ":".
_COLON_LIST_INTRO_RE = re.compile(r":\s*$")


@dataclass(frozen=True, slots=True)
class Citation:
    """A surviving grounded claim's pointer to its canonical record (GROUND-R5).

    Shape ``{metric, value, as_of}`` referencing a canonical record id (activity /
    analytic-computation / workout / plan), NEVER a source/provider id. ``value`` is
    taken VERBATIM from canonical analytics (GROUND-R7); this layer never recomputes.
    """

    record_id: str
    metric: str | None = None
    value: float | None = None
    as_of: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """One distinct athlete-facing observation carrying a STABLE id (COACH-R8).

    The stable ``observation_id`` is the expand/drill handle a later follow-up turn
    targets without re-stating the original question. ``citations`` are the grounded
    numbers behind the observation, surfaced on demand (VOICE-R9), never as a hero
    metrics dump.
    """

    observation_id: str
    text: str
    citations: tuple[Citation, ...] = ()


# --- deterministic presentation checks (the GATE of EVAL-R5b.1) ---


def first_sentence(html_or_text: str) -> str:
    """Return the leading athlete-facing sentence with markup/whitespace stripped.

    Reads the lead out of the (later-sanitized) grounded body so the leads-with-state
    check (COACH-R7) inspects what the athlete actually sees first.
    """
    plain = _TAG_RE.sub(" ", html_or_text)
    plain = " ".join(plain.split())
    for end in (". ", "! ", "? "):
        idx = plain.find(end)
        if idx != -1:
            return plain[: idx + 1].strip()
    return plain.strip()


def count_foregrounded_numbers(html_or_text: str) -> int:
    """Count explicit foregrounded numeric values in athlete-facing prose (VOICE-R7).

    The deterministic number-density measurement; the caller compares it against the
    per-length cap. Markup is stripped first so attribute digits are not counted.
    """
    plain = _TAG_RE.sub(" ", html_or_text)
    return len(_NUMBER_RE.findall(plain))


def leads_with_state(html_or_text: str) -> bool:
    """True iff the leading sentence reads as a STATE/trend phrase, not a metrics report.

    Deterministic gate for COACH-R7 / EVAL-R5b.1. A lead FAILS when it is:

    * empty;
    * a bare number / metric token with no plain-language words around it (the original
      check — fewer than two real words after numbers are stripped);
    * a metrics-report FRAME — "here is/are your … picture/metrics/numbers/…", "your
      current/latest … numbers", a "metrics:"/"numbers:" intro (:data:`_REPORT_FRAME_RE`);
    * a colon-terminated list intro (the "…:" that opens a bullet list of values,
      :data:`_COLON_LIST_INTRO_RE`); or
    * a raw internal metric token (ctl/atl/tsb/…) used AS the lead's subject
      (:func:`_lead_is_raw_metric_token`).

    A normal warm state sentence — even one that mentions a grounded number in passing —
    PASSES, because it carries sentence words around the value and opens with a human read
    of how the athlete is doing, not a data readout. This requirement governs PRESENTATION
    ORDER only; it never changes which numbers are grounded (VOICE-R7).
    """
    lead = first_sentence(html_or_text)
    if not lead:
        return False
    if _COLON_LIST_INTRO_RE.search(lead) or _REPORT_FRAME_RE.search(lead):
        return False
    if _lead_is_raw_metric_token(lead):
        return False
    stripped = _NUMBER_RE.sub(" ", lead)
    words = [w for w in re.findall(r"[^\W\d_]+", stripped, flags=re.UNICODE) if len(w) > 1]
    return len(words) >= 2


def _lead_is_raw_metric_token(lead: str) -> bool:
    """True iff the lead's content is dominated by raw internal metric tokens (VOICE-R2).

    A lead such as ``"ctl: 6.7, atl: 30.2, tsb: -28"`` carries >= 2 "words" (ctl/atl/tsb) so
    the word-count check alone would pass it; but those words ARE the internal codes the
    athlete must never see, so the lead is a metric-token readout, not a state read. The gate
    fails a lead whose only multi-letter words are internal metric tokens (no real
    plain-language word remains once the tokens are removed). The token set is the canonical
    schema (:data:`INTERNAL_METRIC_TOKENS`, the forbidden metric CODES), not a config value —
    it is what the engine is structurally not allowed to surface in prose.
    """
    stripped = _NUMBER_RE.sub(" ", lead)
    words = [w.lower() for w in re.findall(r"[^\W\d_]+", stripped, flags=re.UNICODE) if len(w) > 1]
    if not words:
        return False
    non_token = [w for w in words if w not in INTERNAL_METRIC_TOKENS]
    return len(non_token) < 2


# --- citation projection ---


def _to_citation(raw: Mapping[str, Any]) -> Citation:
    """Project one graph citation mapping into the typed :class:`Citation` (GROUND-R5).

    Reads the canonical ``{metric, value, as_of}`` + record-id shape; a citation with
    no resolvable record id is dropped by the caller (no claim without a citation).
    """
    value = raw.get("value")
    return Citation(
        record_id=str(raw.get("record_id", "")),
        metric=_opt_str(raw.get("metric")),
        value=float(value) if isinstance(value, (int, float)) else None,
        as_of=_opt_str(raw.get("as_of")),
    )


def _opt_str(value: Any) -> str | None:
    """Coerce an optional graph field to ``str | None`` without inventing a value."""
    return None if value is None else str(value)


def _project_citations(raw: Sequence[Mapping[str, Any]]) -> tuple[Citation, ...]:
    """Project + filter graph citations: keep only those with a resolvable record id."""
    out = (_to_citation(c) for c in raw)
    return tuple(c for c in out if c.record_id)


# --- number-density cap ---


def number_cap(response_length: ResponseLength, coach_numeric_detail_level: int = 3) -> int:
    """Return the foregrounded-number ceiling for length + numeric-detail preference.

    The default level ``3`` intentionally matches the historical response-length caps, so
    existing callers keep their behavior until they pass a resolved preference.
    """
    level = coach_numeric_detail_level if coach_numeric_detail_level in (1, 2, 3, 4, 5) else 3
    return _NUMERIC_DETAIL_CAPS[response_length][level]


def _enforce_number_cap(html: str, text: str, cap: int) -> tuple[str, str]:
    """Deterministically hold the body to the foregrounded-number cap (VOICE-R7).

    If the projected body foregrounds more explicit numbers than the per-length ceiling,
    the surplus foregrounded numbers (keeping the first ``cap``) are demoted to a plain
    "(value omitted)" token so the cap is ENFORCED on what ships — not merely test-asserted
    (EVAL-R5b.1). The grounded numbers themselves remain available via the citations /
    reveal-numbers follow-up; only the in-prose density is bounded.
    """
    if count_foregrounded_numbers(text) <= cap:
        return html, text
    return _demote_numbers(html, cap), _demote_numbers(text, cap)


def _demote_numbers(body: str, cap: int) -> str:
    """Keep the first ``cap`` foregrounded numbers; replace the rest with a token."""
    seen = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        return match.group(0) if seen <= cap else "(value omitted)"

    return _NUMBER_RE.sub(_sub, body)


# --- athlete-facing presentation enforcement (VOICE-R2/-R7 / COACH-R7; EVAL-R5b.1) ---


@dataclass(frozen=True, slots=True)
class VoicePresentation:
    """Config-loaded presentation policy for the athlete-facing voice (VOICE-R2/-R7).

    Built from the loaded ``[agent.metric_aliases]`` config (CFG-R1a): ``labels`` is the
    canonical-key -> athlete-native label map (the REVERSE of the alias map, e.g.
    ``{"ctl": "fitness", "atl": "fatigue", "tsb": "freshness"}``) used to TRANSLATE any
    internal metric code that survives into prose into athlete-native language; ``fallback_lead``
    is the warm, jargon-free, number-light opener prepended when the model's lead is a
    metrics-report frame and cannot be salvaged (fail-closed lead, mirroring the readiness
    deliverable's per-verdict fallback). The empty default (no labels, the OSS fallback opener)
    preserves the prior test seam behaviour: with no labels a surviving internal token is still
    SCRUBBED to a neutral phrase rather than shown, never left as a code (fail-closed VOICE-R2).

    This is a PRESENTATION policy only — it rewrites no grounded number and certifies no
    groundedness (VOICE-R7): translation/scrub/lead-repair touch ONLY the prose; the grounded
    ``{metric, value, as_of}`` citations the graph produced are untouched and remain the
    on-demand reveal-numbers backing (GROUND-R5/-R7).
    """

    labels: Mapping[str, str] = field(default_factory=dict)
    fallback_lead: str = "Here's where your training stands right now."
    #: Neutral athlete-native phrase a code with no configured label scrubs TO (never a code).
    #: Carries NO leading article/possessive so it reads cleanly after one ("your training
    #: load") AND on its own ("training load is high") without doubling ("your your …").
    neutral_term: str = "training load"

    @classmethod
    def from_aliases(
        cls,
        aliases: Mapping[str, str],
        *,
        fallback_lead: str | None = None,
        neutral_term: str | None = None,
        preferred: Mapping[str, str] | None = None,
    ) -> VoicePresentation:
        """Build the policy by REVERSING the loaded ``[agent.metric_aliases]`` map (CFG-R1a).

        The alias map is ``natural-label -> canonical-key`` (e.g. ``"fitness" -> "ctl"``); this
        reverses it to ``canonical-key -> athlete-native label`` so a surviving code in prose
        translates back to a human word. When several natural labels map to one canonical key
        (``"fitness"``/``"chronic training load"`` both -> ``ctl``) a single SHORT, code-free
        preferred label is chosen (the shortest alias that is not itself a code and carries no
        parenthetical gloss), so the translation reads like a coach, not a glossary. ``preferred``
        lets the config pin an explicit label per key; ``fallback_lead``/``neutral_term`` default
        to the warm OSS copy when the config supplies none.
        """
        labels: dict[str, str] = {}
        for natural, canonical in aliases.items():
            key = str(canonical).strip().lower()
            label = str(natural).strip()
            if not key or not label:
                continue
            cand = label.lower()
            if cand in INTERNAL_METRIC_TOKENS or "(" in label:
                continue  # a code or a glossed alias is not an athlete-native label
            existing = labels.get(key)
            if existing is None or len(label) < len(existing):
                labels[key] = label
        if preferred:
            labels.update({str(k).strip().lower(): str(v) for k, v in preferred.items()})
        kwargs: dict[str, Any] = {"labels": labels}
        if fallback_lead is not None:
            kwargs["fallback_lead"] = fallback_lead
        if neutral_term is not None:
            kwargs["neutral_term"] = neutral_term
        return cls(**kwargs)


# Alternation of the forbidden codes, longest-first so a multi-word code (``critical_power_w``)
# is tried before a substring. Word-bounded at use so it never mangles a real word that merely
# contains the letters.
_TOKEN_ALTERNATION = "|".join(
    sorted((re.escape(t) for t in INTERNAL_METRIC_TOKENS), key=len, reverse=True)
)

# A PARENTHETICAL GLOSS whose content contains a forbidden code, e.g. ``"fitness (ctl)"`` or
# ``"freshness (training stress balance / tsb)"``: the whole ``(...)`` run is DROPPED (the
# preceding athlete word already carries the meaning), so a real word is never doubled with its
# code gloss. Run BEFORE the standalone-token pass so ``"fitness (ctl)"`` -> ``"fitness"`` rather
# than ``"fitness (fitness)"``. Non-nested parens only (the model never nests a metric gloss).
_GLOSS_PAREN_RE = re.compile(
    rf"\s*\((?:[^()]*?(?<![\w'])(?:{_TOKEN_ALTERNATION})(?![\w])[^()]*?)\)",
    flags=re.IGNORECASE,
)

# A raw internal metric CODE used as a standalone word in prose (case-insensitive), with any
# surrounding markdown emphasis (``**``/``__``/``*``/``_``/`` ` ``) and an OPTIONAL trailing
# parenthetical gloss (``"ctl (chronic training load / fitness)"``) so the whole code+gloss run
# is replaced by the athlete-native label in ONE substitution. Word-bounded so it never mangles
# a real word that merely contains the letters; a leading apostrophe (a possessive) is excluded.
_RAW_TOKEN_RE = re.compile(
    rf"(?<![\w'])(?:\*\*|__|[*_`])?(?P<tok>{_TOKEN_ALTERNATION})(?:\*\*|__|[*_`])?"
    r"(?:\s*\((?:[^()]*)\))?(?![\w])",
    flags=re.IGNORECASE,
)


def enforce_presentation(
    html: str,
    text: str,
    *,
    response_length: ResponseLength,
    presentation: VoicePresentation,
    coach_numeric_detail_level: int = 3,
) -> tuple[str, str]:
    """Hold the athlete-facing body to the voice contract AFTER grounding (VOICE-R2/-R7).

    A deterministic PRESENTATION pass layered over the graph's fail-closed grounding; it
    rewrites no grounded number and changes no citation (VOICE-R7). In order:

    1. **Translate / scrub raw internal metric tokens (VOICE-R2).** Any ``ctl``/``atl``/``tsb``/
       ``tss``/… code that survived into the prose (with markdown emphasis and/or a trailing
       gloss) is replaced by its athlete-native label from the config-loaded map, or by a neutral
       phrase when none is configured — NEVER left as a code.
    2. **Repair a report-frame lead (COACH-R7).** If the leading sentence is not a state read
       (a metrics-report frame, a colon list-intro, or a bare metric token, per
       :func:`leads_with_state`),
       the offending lead sentence is dropped if a later state sentence can lead; otherwise the
       warm, number-light fallback opener is prepended (fail-closed, mirroring the readiness
       deliverable's per-verdict fallback). The grounded body content is preserved.
    3. **Hold the number-density cap (VOICE-R7/-R8).** Surplus foregrounded numbers beyond the
       per-length ceiling are demoted (the existing :func:`_enforce_number_cap`).

    The HTML body is re-derived from the repaired TEXT via the caller's sanitizer is NOT done
    here (this leaf has no HTML escaper); instead both bodies are transformed in lockstep so the
    caller's already-sanitized HTML stays consistent. Numbers that survive are exactly the
    grounded ones; demotion only bounds in-prose density (citations are the reveal backing).
    """
    text = _translate_tokens(text, presentation)
    html = _translate_tokens(html, presentation)
    text, html = _repair_lead(text, html, presentation)
    return _enforce_number_cap(html, text, number_cap(response_length, coach_numeric_detail_level))


def _translate_tokens(body: str, presentation: VoicePresentation) -> str:
    """Replace every raw internal metric code with an athlete word (VOICE-R2).

    Two passes: (1) DROP a parenthetical gloss that merely restates a code after a real word
    (``"fitness (ctl)"`` -> ``"fitness"``), so an athlete word is never doubled with its code;
    (2) replace a STANDALONE code (with emphasis / trailing gloss) by its config-loaded
    athlete-native label, or the neutral phrase when none is configured — never left as a code.
    """
    body = _GLOSS_PAREN_RE.sub("", body)

    def _sub(match: re.Match[str]) -> str:
        token = match.group("tok").lower()
        return presentation.labels.get(token, presentation.neutral_term)

    return _RAW_TOKEN_RE.sub(_sub, body)


def _repair_lead(text: str, html: str, presentation: VoicePresentation) -> tuple[str, str]:
    """Make the body LEAD with a state read, dropping/prepending as needed (COACH-R7).

    Returns ``(text, html)``. If the text already leads with a state phrase it is returned
    unchanged. Otherwise the first (report-frame) sentence is dropped when a later sentence can
    lead with a state read; if no salvageable lead remains, the warm fallback opener is
    prepended. The HTML is updated by the SAME edit (drop/prepend) so both bodies stay aligned;
    the caller re-sanitizes the HTML, so this only needs to keep the visible prose in step.
    """
    if leads_with_state(text):
        return text, html
    salvaged = _drop_report_lead(text)
    if salvaged is not None and leads_with_state(salvaged):
        return salvaged, _wrap_html(salvaged, html)
    repaired = f"{presentation.fallback_lead} {text}".strip()
    return repaired, _wrap_html(repaired, html)


def _drop_report_lead(text: str) -> str | None:
    """Drop the leading report-frame sentence, returning the remainder (or None if none left).

    Splits off the first sentence (the report frame / colon intro) and returns what follows,
    so a body like ``"Here is your picture: <state sentence>."`` promotes the real state read to
    the lead. A colon list-intro lead has no sentence terminator, so the split is on the colon.
    """
    lead = first_sentence(text)
    remainder = text[len(lead) :].strip() if text.startswith(lead) else ""
    if not remainder:
        # No sentence terminator (a "...:" list intro): promote what follows the colon.
        _head, sep, rest = text.partition(":")
        remainder = rest.strip() if sep else ""
    return remainder or None


def _wrap_html(text: str, original_html: str) -> str:
    """Reflect a repaired TEXT body back into the HTML body, escaping the new text.

    The graph's ``grounded_html`` was server-sanitized from ``grounded_text``; after a prose
    repair the HTML must mirror the repaired text. This leaf has no HTML escaper, so it escapes
    the minimal set (``&``/``<``/``>``) and wraps in a single paragraph — the SAME shape
    ``graph_state.safe_html`` produces (AGT-SEC-R2). An empty repair yields the original HTML.
    """
    if not text:
        return original_html
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<p>{escaped}</p>"


__all__ = [
    "INTERNAL_METRIC_TOKENS",
    "Citation",
    "Observation",
    "ResponseLength",
    "VoicePresentation",
    "count_foregrounded_numbers",
    "enforce_presentation",
    "first_sentence",
    "leads_with_state",
    "number_cap",
]
