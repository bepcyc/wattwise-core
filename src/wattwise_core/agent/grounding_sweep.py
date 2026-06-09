"""Deterministic, extraction-independent text sweeps for the grounder (GROUND-R3/R4/R7).

The focused sibling of :mod:`wattwise_core.agent.grounding` (QUAL-R9 size split) that owns the
SECOND, model-independent nets the grounder runs over the scrubbed draft AFTER per-claim
verification: a URL sweep (GROUND-R4) and a NUMERIC-coverage sweep (GROUND-R7 / H4). Both exist
because fail-closure must not depend on the LLM claim-extractor surfacing every span — an invented
URL or a fabricated number the extractor missed must still never reach athlete-facing text.

* :func:`scrub_unverified_urls` — remove every URL not first-party allow-listed / accepted by the
  evidence (GROUND-R4: invented URLs scrubbed unconditionally, even unextracted ones).
* :func:`scrub_uncovered_numbers` — remove every number-like span not covered by a GROUNDED claim's
  published canonical value or a structurally safe NON-metric token (a date, a "Day/Week/Zone N"
  ordinal, an "NxM" interval, or a duration/percentage with an attached unit). Anything else — a
  bare or metric-attached figure the grounder did not verify — is removed (fail-closed, H4).

Everything here is a pure, synchronous function of its text inputs (GRAPH-R4): it calls no model
and no service, so the same inputs always yield the same result.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from wattwise_core.agent.contracts import GroundingEvidence

# A numeric literal token in athlete-facing prose (the numeric-coverage sweep target).
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

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
    text: str, grounded_numbers: frozenset[str] | set[str]
) -> tuple[str, int]:
    """Scrub every number-like span not covered by a grounded claim / safe token (GROUND-R7 / H4).

    The deterministic NUMERIC analogue of :func:`scrub_unverified_urls`: numeric fail-closure must
    not depend on the LLM claim-extractor catching every span. "CTL is 60 and TSB is 999" where the
    extractor returns only the CTL claim leaves the fabricated 999 in the body — so this second,
    extraction-independent net sweeps the text and removes any numeric span that is neither (a) a
    GROUNDED canonical value the grounder already verified (``grounded_numbers``, the values it just
    published) nor (b) a structurally safe NON-metric token (a date, a "Day/Week/Phase/Block/Zone N"
    ordinal, an "NxM" interval structure, or a duration/percentage with an attached unit, per
    :func:`_is_safe_numeric_context`). Anything else — a bare or metric-attached figure the grounder
    did not verify — is removed (fail-closed, "when in doubt, scrub"). Returns the cleaned text and
    the number of spans removed (the caller downgrades the decision if any was). A body with no
    uncovered number is returned untouched.
    """
    removed = 0
    # Pre-compute the spans EVERY digit-group inside which is safe regardless of per-token context:
    #  - a surviving URL (``/activity/42``): an already-verified first-party link (URL sweep ran
    #    first and kept it), never a free-floating metric;
    #  - an ISO date / NxM interval (``2026-06-08``, ``3x12``): a digit-group within a structure.
    # Computing whole spans (not a narrow per-token window) is what keeps the TRAILING group of a
    # date (``-08`` of ``2026-06-08``) from being scrubbed (the per-token window missed it).
    safe_spans = [(m.start(), m.end()) for m in URL_RE.finditer(text)]
    safe_spans += [(m.start(), m.end()) for m in _SAFE_SPAN_RE.finditer(text)]

    def _keep(match: re.Match[str]) -> str:
        token = match.group(0)
        if any(start <= match.start() < end for start, end in safe_spans):
            return token
        if token in grounded_numbers or _is_safe_numeric_context(text, match):
            return token
        nonlocal removed
        removed += 1
        return ""

    cleaned = NUMBER_RE.sub(_keep, text)
    if not removed:
        return text, 0
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return re.sub(r"\s+([.,;:!?])", r"\1", cleaned).strip(), removed


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
    "NUMBER_RE",
    "URL_RE",
    "normalize_url",
    "normalize_urls",
    "scrub_uncovered_numbers",
    "scrub_unverified_urls",
]
