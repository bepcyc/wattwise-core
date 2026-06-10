"""Deterministic, extraction-independent text sweeps for the grounder (GROUND-R3/R4/R7).

The focused sibling of :mod:`wattwise_core.agent.grounding` (QUAL-R9 size split) that owns the
SECOND, model-independent nets the grounder runs over the scrubbed draft AFTER per-claim
verification: a URL sweep (GROUND-R4) and a NUMERIC-coverage sweep (GROUND-R7 / H4). Both exist
because fail-closure must not depend on the LLM claim-extractor surfacing every span — an invented
URL or a fabricated number the extractor missed must still never reach athlete-facing text.

* :func:`scrub_unverified_urls` — remove every URL not first-party allow-listed / accepted by the
  evidence (GROUND-R4: invented URLs scrubbed unconditionally, even unextracted ones).
* :func:`scrub_uncovered_numbers` — remove every numeric PHRASE not POSITIONALLY covered by a
  character range the grounder verified/rewrote (:func:`span_covered`, issue #4) and not a
  structurally safe NON-metric token (a date, a "Day/Week/Zone N" ordinal, an "NxM" interval, or a
  duration/percentage with an attached unit). Anything else — a bare or metric-attached figure the
  grounder did not verify — is removed (fail-closed, H4). Removal is PHRASE-wise
  (:data:`NUMBER_PHRASE_RE`): a "5-7" range goes as ONE span, with any orphan leading dash and
  now-empty spaced unit token, so no dangling punctuation survives (VOICE-R2 clean copy).

Everything here is a pure, synchronous function of its text inputs (GRAPH-R4): it calls no model
and no service, so the same inputs always yield the same result.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from wattwise_core.agent.contracts import GroundingEvidence

# A numeric literal token in athlete-facing prose (the numeric-coverage sweep target).
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# A whole numeric PHRASE: one number optionally chained into a range by hyphen/en-dash/em-dash
# ("5-7", "5 - 7", an en/em-dash range). The sweep removes WHOLE phrases, never a lone member
# of a range, so a scrub can not leave a dangling range dash between two vanished numbers
# (VOICE-R2 clean copy).
NUMBER_PHRASE_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*[-\u2013\u2014]\s*\d+(?:\.\d+)?)*")

# An orphan range/punctuation dash left DIRECTLY before a removed numeric phrase (an
# "... - 5 h"-shaped lead with the 5 scrubbed): removed together with the phrase so no
# dangling dash survives.
_LEADING_ORPHAN_DASH_RE = re.compile(r"(?:^|(?<=\s))[-\u2013\u2014]\s*\Z")

# A spaced duration/distance/percentage unit token DIRECTLY after a removed numeric phrase
# ("5-7 h" -> removing the range must also take the now-empty unit " h"), mirroring the attached
# _UNIT_SUFFIX_RE vocabulary. Only the unit itself — never a following word.
_TRAILING_ORPHAN_UNIT_RE = re.compile(
    r"\A\s*(?:(?:m|min|mins|h|hr|hrs|s|sec|km|mi)(?![A-Za-z])|%)", re.IGNORECASE
)

# A URL token in athlete-facing prose; the deterministic URL sweep checks every match against the
# allow-list / matched-record destinations regardless of model extraction (GROUND-R4).
URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)

# Structural words that legitimately precede a small NON-metric ordinal in a plan/answer body
# ("Day 1", "Week 2", "Zone 3"); the number after one of these is a structural index, not a
# grounding-checkable metric magnitude, so the numeric sweep leaves it (still NEVER a metric word
# like ctl/atl/tsb/load/power — those carry no free pass and must ground as a claim).
_STRUCTURAL_PREFIXES: frozenset[str] = frozenset(
    {"day", "week", "phase", "block", "zone", "set", "rep", "reps", "round", "interval", "lap"}
)
# A number immediately bound to one of these unit suffixes is a duration/distance/percentage token
# (``45m``, ``90min``, ``2h``, ``30s``, ``5km``, ``20%``), not a bare metric magnitude. The unit is
# valid only when it is the END of a word — followed by end-of-string or a NON-alphanumeric — so
# ``60kg`` etc. is still ``kg`` (kept) but a metric magnitude run into a letter is not falsely
# excused. ``%`` is non-word so it uses an explicit non-letter lookahead rather than ``\b``.
_UNIT_SUFFIX_RE = re.compile(
    r"\A(?:(?:m|min|mins|h|hr|hrs|s|sec|km|mi)(?![A-Za-z])|%)", re.IGNORECASE
)
# ISO dates and NxM interval structures are SAFE non-metric spans: a digit-group inside one of these
# is part of a date/structure, not a free-floating metric. Spans are pre-computed over the whole
# text (like URL spans) so EVERY digit-group within them is skipped — a narrow per-token window
# would miss the trailing group of a date (``-08`` of ``2026-06-08``).
_SAFE_SPAN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d+\s*x\s*\d+", re.IGNORECASE)


def span_covered(ranges: Sequence[tuple[int, int]], start: int, end: int) -> bool:
    """True iff ``[start, end)`` lies WITHIN a verified/rewritten character range.

    The POSITIONAL coverage test of the numeric sweep (GROUND-R7 / H4, issue #4): a numeric
    token is covered only when the grounder actually verified/rewrote THOSE characters of
    the draft — string equality with some other claim's published value covers nothing.
    """
    return any(s <= start and end <= e for s, e in ranges)


def normalize_urls(urls: Iterable[str]) -> frozenset[str]:
    """Normalize an allow-list into a comparable set (GROUND-R4)."""
    return frozenset(normalize_url(u) for u in urls)


def normalize_url(url: str) -> str:
    """Normalize a URL for allow-list comparison: strip whitespace, lowercase scheme/host.

    A conservative normalization: trims surrounding whitespace and a trailing slash, and
    lowercases the scheme+host portion. Anything ambiguous compares unequal and is
    scrubbed (fail-closed, GROUND-R4) rather than guessed into the allow-list.
    """
    stripped = url.strip().rstrip("/")
    if "://" not in stripped:
        return stripped.casefold()
    scheme, rest = stripped.split("://", 1)
    if "/" in rest:
        host, path = rest.split("/", 1)
        return f"{scheme.casefold()}://{host.casefold()}/{path}"
    return f"{scheme.casefold()}://{rest.casefold()}"


def scrub_unverified_urls(
    text: str, evidence: GroundingEvidence, allow_list: frozenset[str]
) -> str:
    """Remove every URL in the body not on the allow-list / a matched record (GROUND-R4).

    A SECOND, extraction-independent net: even a URL the model never surfaced as a claim is
    scrubbed unless it is first-party allow-listed or accepted by the evidence — so a
    model-invented link can never reach the athlete just because it went unextracted. A body
    with no URL is returned untouched; removals collapse the whitespace they leave behind.
    """
    if not URL_RE.search(text):
        return text

    def _keep(match: re.Match[str]) -> str:
        url = match.group(0)
        if normalize_url(url) in allow_list or evidence.url_allowed(url):
            return url
        return ""

    cleaned = URL_RE.sub(_keep, text)
    if cleaned == text:
        return text
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return re.sub(r"\s+([.,;:!?])", r"\1", cleaned).strip()


def scrub_uncovered_numbers(
    text: str, covered_ranges: Sequence[tuple[int, int]]
) -> tuple[str, int]:
    """Scrub every numeric phrase not covered by a grounded claim / safe token (GROUND-R7 / H4).

    The deterministic NUMERIC analogue of :func:`scrub_unverified_urls`: numeric fail-closure must
    not depend on the LLM claim-extractor catching every span. "CTL is 60 and TSB is 999" where the
    extractor returns only the CTL claim leaves the fabricated 999 in the body — so this second,
    extraction-independent net sweeps the text and removes any numeric PHRASE that is neither (a)
    made of tokens whose character ranges the grounder actually VERIFIED/REWROTE
    (``covered_ranges``, positions in ``text`` — coverage is POSITIONAL, never string membership: a
    leftover wrong token that merely EQUALS another claim's published value is still swept, issue
    #4) nor (b) a structurally safe NON-metric token (a date, a "Day/Week/Phase/Block/Zone N"
    ordinal, an "NxM" interval structure, or a duration/percentage with an attached unit, per
    :func:`_is_safe_numeric_context`). Anything else — a bare or metric-attached figure the
    grounder did not verify — is removed (fail-closed, "when in doubt, scrub").

    Removal is PHRASE-wise (:data:`NUMBER_PHRASE_RE`): a range like "5-7" is taken as one span, an
    orphan dash directly before it and a now-empty spaced unit token directly after it are removed
    with it, so the surrounding sentence stays grammatical — never a dangling range dash or a bare
    unit (VOICE-R2 clean copy). A range with ANY uncovered member is removed whole (fail-closed).
    Returns the cleaned text and the number of phrases removed (the caller downgrades the decision
    if any was). A body with no uncovered number is returned untouched.
    """
    # Pre-compute the spans EVERY digit-group inside which is safe regardless of per-token context:
    #  - a surviving URL (``/activity/42``): an already-verified first-party link (URL sweep keeps
    #    only allow-listed links), never a free-floating metric;
    #  - an ISO date / NxM interval (``2026-06-08``, ``3x12``): a digit-group within a structure.
    # Computing whole spans (not a narrow per-token window) is what keeps the TRAILING group of a
    # date (``-08`` of ``2026-06-08``) from being scrubbed (the per-token window missed it).
    safe_spans = [(m.start(), m.end()) for m in URL_RE.finditer(text)]
    safe_spans += [(m.start(), m.end()) for m in _SAFE_SPAN_RE.finditer(text)]

    removed = 0
    out: list[str] = []
    cursor = 0
    for match in NUMBER_PHRASE_RE.finditer(text):
        if match.start() < cursor:  # already swallowed by a widened earlier removal
            continue
        if _phrase_covered(text, match, safe_spans, covered_ranges):
            continue
        start, end = _expand_removal_span(text, match.start(), match.end())
        out.append(text[cursor:start])
        cursor = end
        removed += 1
    if not removed:
        return text, 0
    out.append(text[cursor:])
    cleaned = "".join(out)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return re.sub(r"\s+([.,;:!?])", r"\1", cleaned).strip(), removed


def _phrase_covered(
    text: str,
    match: re.Match[str],
    safe_spans: list[tuple[int, int]],
    covered_ranges: Sequence[tuple[int, int]],
) -> bool:
    """True iff a numeric PHRASE may stay: safe span / structural context / every token covered.

    A phrase inside a pre-computed safe span (URL / ISO date / NxM structure) stays whole; a phrase
    whose FIRST token sits in a safe structural context ("Week 1-4", an attached unit "45m") stays
    whole; otherwise EVERY numeric token of the phrase must lie within a POSITIONALLY covered
    character range (:func:`span_covered` — a range the grounder actually verified/rewrote, issue
    #4) — a range with any unverified member is removed whole (fail-closed, GROUND-R3).
    """
    if any(start <= match.start() < end for start, end in safe_spans):
        return True
    if _is_safe_numeric_context(text, match):
        return True
    base = match.start()
    for tok in NUMBER_RE.finditer(match.group(0)):
        start, end = base + tok.start(), base + tok.end()
        if tok.group(0).startswith("-"):
            # Inside a phrase the dash is a RANGE separator, not a minus sign: the member's
            # covered range is the digits only ("7" of "5-7"), so the sign is excluded here.
            start += 1
        if not span_covered(covered_ranges, start, end):
            return False
    return True


def _expand_removal_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a removal span over an orphan leading dash / trailing empty unit (VOICE-R2).

    Removing a numeric phrase must not strand the punctuation/unit that existed only to carry it:
    a dash directly before the phrase and a spaced unit token directly after it
    ("5-7 h") are removed with the phrase so the remaining prose reads cleanly — no dangling dash,
    no bare empty unit. Deterministic and purely lexical (GRAPH-R4).
    """
    lead = _LEADING_ORPHAN_DASH_RE.search(text[:start])
    if lead is not None:
        start = lead.start()
    trail = _TRAILING_ORPHAN_UNIT_RE.match(text[end:])
    if trail is not None:
        end += trail.end()
    return start, end


def numeric_phrase_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand one numeric token's span to its WHOLE containing phrase (GROUND-R3 / VOICE-R2).

    The per-claim scrub path's analogue of the phrase-wise sweep: the grounder removes an
    ungrounded number by its FULL numeric phrase (the whole "5-7" range, plus an orphan leading
    dash / trailing empty unit), so a claim-level scrub can never leave a dangling dash or a bare
    unit behind. Pure span arithmetic — the caller performs the actual (seam-local, delta-tracked)
    removal so positional coverage stays consistent (issue #4).
    """
    for match in NUMBER_PHRASE_RE.finditer(text):
        if match.start() <= start < match.end():
            return _expand_removal_span(text, match.start(), match.end())
    return _expand_removal_span(text, start, end)


def _is_safe_numeric_context(text: str, match: re.Match[str]) -> bool:
    """True iff a numeric span is a structurally safe NON-metric token (H4 allow-list).

    Conservative + fail-closed: only a number with an attached duration/distance/percentage unit
    (``45m``, ``20%``, ``2h``) or preceded by a structural ordinal word (``Day 1``, ``Week 2``) is
    safe here. (Dates and NxM intervals are handled span-wise by the caller.) A BARE number, or a
    number attached to a metric word, is NOT safe and is swept. Purely lexical over the surrounding
    characters (deterministic, GRAPH-R4).
    """
    start, end = match.start(), match.end()
    after = text[end:]
    # A unit suffix immediately after the number (``45m``, ``20%``, ``2h``).
    if _UNIT_SUFFIX_RE.match(after):
        return True
    # A structural prefix word immediately before the number (``Day 1``, ``Week 2``).
    preceding = text[:start].rstrip()
    last_word = re.search(r"([A-Za-z]+)\W*\Z", preceding)
    if last_word is not None:
        return last_word.group(1).casefold() in _STRUCTURAL_PREFIXES
    return False


__all__ = [
    "NUMBER_PHRASE_RE",
    "NUMBER_RE",
    "URL_RE",
    "normalize_url",
    "normalize_urls",
    "numeric_phrase_span",
    "scrub_uncovered_numbers",
    "scrub_unverified_urls",
    "span_covered",
]
