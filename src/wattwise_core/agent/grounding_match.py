"""Numeric-match primitives for the deterministic grounder (doc 50 GROUND-R5/R7).

The focused sibling of :mod:`wattwise_core.agent.grounding` (QUAL-R9 size split) that owns the
numeric side of claim verification: the config-carried :class:`NumericTolerance` band, the
fail-closed canonical-value read (:func:`_canonical_metric` / :func:`_as_float`), the
tolerance comparison (:func:`_within_tolerance`), the grounded-number citation
(:func:`_metric_citation`, GROUND-R5), and the POSITIONAL canonical-display rewrite of the
model's numeric token (:func:`_apply_number_scrub` / :func:`_render_value`, GROUND-R7 verbatim).

The positional primitives (:func:`bounded_number_pattern`, :func:`find_claim_token_span`,
:func:`ranges_overlap`, :func:`shift_ranges_after`, :func:`remove_span_clean`, plus the
sweep-side :func:`~wattwise_core.agent.grounding_sweep.span_covered`) exist because the rewrite
MUST be anchored to the claim's OWN span and numeric coverage MUST be tracked as character
RANGES, never as string membership: with two
claims sharing a token, an unanchored ``str.find`` rewrite lands the second claim's canonical
value on the FIRST claim's span and a flat published-value set then "covers" the leftover wrong
token — publishing BOTH numbers wrong under ``proceed`` (the GROUND-R7/R9 fail-open of issue #4).
All functions are pure/synchronous and deterministic (GRAPH-R4); ``grounding`` imports and
re-exports ``NumericTolerance`` so every historical
``from wattwise_core.agent.grounding import NumericTolerance`` path stays stable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from wattwise_core.agent.contracts import Claim, GroundingEvidence
from wattwise_core.agent.grounding_sweep import NUMBER_RE

# Default numeric tolerance for matching a claimed metric value against the canonical
# analytic (GROUND-R7). A claimed number within this band of the canonical value is treated
# as a verbatim re-statement; anything outside is scrubbed/replaced. The default REL band is
# wide enough to accept a model DISPLAYING the canonical value at sane human precision (e.g.
# "your ctl is 6.7" for a canonical 6.7315, or a whole-number round of a small CTL/ATL/TSB),
# while a FABRICATED number — wrong by tens of percent — is still scrubbed. The exact
# threshold is config-loaded content (§16 / SKILL-R1 metric thresholds); this constant is the
# fallback the OSS engine resolves it from, never a hidden hardcode of policy in the gate.
_DEFAULT_REL_TOLERANCE = 0.02
# Absolute floor so near-zero canonical values (e.g. tsb ≈ 0) still admit a 1-decimal
# display ("0.0" for a canonical 0.004) without the relative band collapsing to nothing.
_DEFAULT_ABS_TOLERANCE = 0.05
# Decimal places the canonical value is rounded to when it REPLACES a recognized numeric span
# in the published text (GROUND-R7). Config-loaded (§16); this is the no-config fallback.
_DEFAULT_DISPLAY_DECIMALS = 1


@dataclass(frozen=True, slots=True)
class NumericTolerance:
    """The numeric-match band the grounder accepts for a metric value (GROUND-R7).

    Carried into :func:`~wattwise_core.agent.grounding.ground` so the threshold is config-loaded
    content (§16 / SKILL-R1), not a literal baked into the gate. ``rel`` is the relative band (a
    fraction of the canonical magnitude) and ``abs_`` the absolute floor for near-zero canonical
    values; a claimed value within EITHER band of the canonical value grounds. The defaults accept
    a human-precision display of the canonical number and scrub a fabrication.

    ``display_decimals`` is the precision the CANONICAL value is rounded to when it is PUBLISHED
    in place of the model's numeric span (GROUND-R7 verbatim): the model's approximation — even
    a within-tolerance one like "102" for a canonical 100 — is NEVER what reaches the athlete;
    the canonical value, rounded for a human display, always is.
    """

    rel: float = _DEFAULT_REL_TOLERANCE
    abs_: float = _DEFAULT_ABS_TOLERANCE
    display_decimals: int = _DEFAULT_DISPLAY_DECIMALS


_DEFAULT_TOLERANCE = NumericTolerance()


def _canonical_metric(evidence: GroundingEvidence, metric: str, ref: str | None) -> float | None:
    """Read the canonical value for a ``(metric, as_of)`` request, fail-closed (GROUND-R7).

    The base :class:`GroundingEvidence` contract's ``metric_value`` is async; the grounder
    is synchronous and deterministic, so the caller resolves snapshots ahead of time onto
    an optional synchronous ``metric_snapshot(metric, as_of)`` accessor (the resolved-ahead
    path). When that accessor is absent, or returns a non-finite/unavailable value, this
    returns ``None`` — the number is then scrubbed, never surfaced as a placeholder. The
    grounder never awaits inside ``ground``.
    """
    accessor = getattr(evidence, "metric_snapshot", None)
    if accessor is None:
        return None
    raw: Any = accessor(metric, ref)
    return _as_float(raw)


def _as_float(raw: Any) -> float | None:
    """Coerce a canonical value to a finite float, or ``None`` (fail-closed)."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    return None


def _within_tolerance(
    claimed: float, canonical: float, tolerance: NumericTolerance = _DEFAULT_TOLERANCE
) -> bool:
    """True iff a claimed number matches the canonical value within tolerance (GROUND-R7)."""
    if not (math.isfinite(claimed) and math.isfinite(canonical)):
        return False
    return math.isclose(
        claimed,
        canonical,
        rel_tol=tolerance.rel,
        abs_tol=tolerance.abs_,
    )


def _metric_citation(claim: Claim, canonical: float) -> dict[str, Any]:
    """Citation for a grounded number: canonical metric id + verbatim value (GROUND-R5/R7).

    ``record_id`` is the stable canonical reference to the analytic the number was read
    from — ``{metric}@{as_of}`` (or just the metric when no date) — so a deliverable layer
    keeps the citation (a grounded number MUST carry a resolvable citation, GROUND-R5; a
    citation with no record id is dropped downstream and the number would ship uncited).
    """
    record_id = f"{claim.metric}@{claim.ref}" if claim.ref else str(claim.metric)
    return {
        "kind": "metric",
        "record_id": record_id,
        "metric": claim.metric,
        "value": canonical,
        "as_of": claim.ref,
    }


def bounded_number_pattern(token: str) -> re.Pattern[str]:
    """Compile a word-bounded pattern for a numeric token (GROUND-R7, issue #4).

    Digit/decimal lookarounds keep the token from ever matching INSIDE a longer number:
    ``102`` matches neither the prefix of ``1029`` nor the tail of ``5102`` / ``0.102``,
    and not the integer part of ``102.5`` — while a sentence-final ``102.`` still matches
    (a trailing dot only blocks when a digit follows it). Every token search in the
    grounder MUST go through this pattern, never a bare ``str.find``.
    """
    return re.compile(rf"(?<![\d.]){re.escape(token)}(?!\.?\d)")


def ranges_overlap(ranges: Sequence[tuple[int, int]], start: int, end: int) -> bool:
    """True iff ``[start, end)`` overlaps any of the half-open character ``ranges``."""
    return any(s < end and start < e for s, e in ranges)


def shift_ranges_after(
    ranges: Sequence[tuple[int, int]], pivot: int, delta: int
) -> list[tuple[int, int]]:
    """Shift every range starting at/after ``pivot`` by ``delta`` (a text edit happened there).

    Coverage ranges are positions in the CURRENT draft; every subsequent edit (a later
    claim's rewrite/removal, a span scrub) must re-base the ranges behind it or the sweep
    would test stale positions. Ranges strictly before the edit are untouched; the caller
    guarantees no tracked range straddles the edited span.
    """
    return [(s + delta, e + delta) if s >= pivot else (s, e) for s, e in ranges]


def remove_span_clean(text: str, start: int, end: int) -> tuple[str, int]:
    """Remove ``[start, end)`` and tidy ONLY the seam it leaves; return (text, length delta).

    Seam-local cleanup (collapse the doubled whitespace, drop a space left hanging before
    punctuation, trim a now-leading/trailing seam) instead of a whole-text regex pass, so
    every character OUTSIDE the seam keeps its position up to the single computable shift
    that :func:`shift_ranges_after` applies — the invariant positional coverage depends on.
    """
    left, right = text[:start], text[end:]
    trimmed_left = left.rstrip()
    trimmed_right = right.lstrip()
    if not trimmed_left or not trimmed_right or trimmed_right[0] in ".,;:!?":
        edited = trimmed_left + trimmed_right
    elif len(trimmed_left) != len(left) or len(trimmed_right) != len(right):
        edited = trimmed_left + " " + trimmed_right
    else:
        edited = left + right
    return edited, len(edited) - len(text)


def find_claim_token_span(
    text: str, claim: Claim, covered: Sequence[tuple[int, int]]
) -> tuple[int, int] | None:
    """Locate the claim's OWN numeric token in ``text``, anchored to the claim's span (R7).

    Resolution order — positional and word-bounded at every step (issue #4):

    1. **anchor**: each verbatim occurrence of ``claim.text`` in the draft, taking the first
       whose word-bounded token does not overlap a range an earlier claim already
       verified/rewrote — so two claims sharing a token each land in their OWN span;
    2. **fallback** (``claim.text`` not verbatim in the draft — case/wording drift): the
       first word-bounded occurrence of the bare token outside every covered range.

    Never a bare ``str.find`` over the whole draft, and never a match inside a longer
    number (:func:`bounded_number_pattern`). Returns the absolute ``(start, end)`` span of
    the token, or ``None`` when it is not present — the numeric-coverage sweep then owns
    fail-closure for any digits that DID reach the draft (GROUND-R3).
    """
    token_match = NUMBER_RE.search(claim.text)
    token = token_match.group(0) if token_match is not None else claim.text
    pattern = bounded_number_pattern(token)
    if token != claim.text:
        for anchor in re.finditer(re.escape(claim.text), text):
            inner = pattern.search(text, anchor.start())
            if inner is None or inner.end() > anchor.end():
                continue
            if not ranges_overlap(covered, inner.start(), inner.end()):
                return inner.span()
    for match in pattern.finditer(text):
        if not ranges_overlap(covered, match.start(), match.end()):
            return match.span()
    return None


def _apply_number_scrub(
    text: str, claim: Claim, replacement: str, covered: Sequence[tuple[int, int]]
) -> tuple[str, list[tuple[int, int]]]:
    """Rewrite the claim's numeric token IN ITS OWN SPAN to the canonical value, or remove it.

    For a NUMBER claim the published figure is ALWAYS the canonical value (``replacement`` =
    its display string, or ``""`` to remove an ungrounded/unavailable number). The token is
    located positionally (:func:`find_claim_token_span`): anchored to the claim's own text
    span, word-bounded, and skipping ranges earlier claims already verified — never the
    first ``str.find`` hit in the whole draft (issue #4, GROUND-R7). Returns the edited text
    and the updated coverage ranges: a published canonical value adds ITS character range
    (only those characters count as covered for the numeric sweep); a removal or a
    not-found token adds nothing, so any leftover digits stay uncovered and the sweep
    fails closed on them (GROUND-R3).
    """
    span = find_claim_token_span(text, claim, covered)
    if span is None:
        # The model's token is not literally in the draft; nothing to publish/remove here. The
        # positional numeric sweep is the backstop for any uncovered number that DID reach it.
        return text, list(covered)
    start, end = span
    if replacement == "":
        edited, delta = remove_span_clean(text, start, end)
        return edited, shift_ranges_after(covered, end, delta)
    edited = text[:start] + replacement + text[end:]
    updated = shift_ranges_after(covered, end, len(replacement) - (end - start))
    updated.append((start, start + len(replacement)))
    return edited, updated


def _render_value(value: float, decimals: int) -> str:
    """Render the canonical value at ``decimals`` precision, dropping a trailing ``.0``.

    Rounds to the configured display precision (so a high-precision canonical value surfaces as
    the human number a coach would say) and, when the rounded value is integral, renders it
    without a trailing ``.0`` for display parity (``84.0`` -> ``"84"``).
    """
    rounded = round(value, decimals)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.{decimals}f}"


__all__ = [
    "NumericTolerance",
    "bounded_number_pattern",
    "find_claim_token_span",
    "ranges_overlap",
    "remove_span_clean",
    "shift_ranges_after",
]
