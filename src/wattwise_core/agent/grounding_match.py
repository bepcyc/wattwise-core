"""Numeric-match primitives for the deterministic grounder (doc 50 GROUND-R5/R7).

The focused sibling of :mod:`wattwise_core.agent.grounding` (QUAL-R9 size split) that owns the
numeric side of claim verification: the config-carried :class:`NumericTolerance` band, the
fail-closed canonical-value read (:func:`_canonical_metric` / :func:`_as_float`), the
tolerance comparison (:func:`_within_tolerance`), the grounded-number citation
(:func:`_metric_citation`, GROUND-R5), and the canonical-display rewrite of the model's numeric
token (:func:`_apply_number_scrub` / :func:`_render_value`, GROUND-R7 verbatim). The rewrite is
anchored to the CLAIM'S OWN span and never matches inside a longer number, and a verified span is
marked with a sentinel (:func:`_sentinel`) so numeric coverage is POSITIONAL — a stray token that
merely EQUALS a published value is never treated as covered. All functions
are pure/synchronous and deterministic (GRAPH-R4); ``grounding`` imports and re-exports
``NumericTolerance`` so every historical
``from wattwise_core.agent.grounding import NumericTolerance`` path stays stable.
"""

from __future__ import annotations

import math
import re
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


# Sentinel characters (Unicode private-use area, U+E000..U+F8FF) temporarily stand in for the
# canonical display values the grounder has already verified. The numeric-coverage sweep then sees
# ONLY unverified digits — coverage is positional, never by string equality, so a stray token that
# merely EQUALS a published value cannot ride a grounded claim's coverage (fail-closed, H4). The
# caller strips this range from the raw draft first so a model can never pre-plant a sentinel.
_SENTINEL_BASE = 0xE000
_SENTINEL_RANGE_RE = re.compile("[\\ue000-\\uf8ff]")


def _sentinel(index: int) -> str:
    """The placeholder character standing in for the ``index``-th published canonical value."""
    return chr(_SENTINEL_BASE + index)


def _strip_sentinels(text: str) -> str:
    """Remove every private-use sentinel character from a raw draft (anti-spoofing guard)."""
    return _SENTINEL_RANGE_RE.sub("", text)


def _bounded_number_search(text: str, token: str, start: int = 0) -> re.Match[str] | None:
    """Find ``token`` as a STANDALONE number from ``start`` — never inside a longer number.

    The lookarounds refuse a match whose digits continue on either side (``102`` inside ``1029``
    or ``100`` inside ``100.5``) and a match that is really the magnitude of a signed/decimal
    neighbour (``4`` inside ``-4``). An unmatched token simply yields no rewrite — the numeric
    sweep then removes the figure (fail-closed), never a corrupted neighbouring number.
    """
    pattern = re.compile(rf"(?<![\d.\-]){re.escape(token)}(?!\.?\d)")
    return pattern.search(text, start)


def _locate_number_token(text: str, claim: Claim, display: str | None) -> tuple[int, int] | None:
    """Locate the claim's OWN numeric token in ``text``, anchored to the claim's span (R7).

    Resolution order, strictest first: (1) find ``claim.text`` in the draft and match the token
    only WITHIN that span — two claims sharing a token (two "100"s) each resolve to their own
    occurrence; (2) a bounded search for the token anywhere (the model did not reproduce
    ``claim.text`` verbatim — case / wording drift); (3) a bounded search for the canonical
    ``display`` string (the model restated the value at a different precision). ``None`` when
    nothing matches — the caller publishes nothing and the numeric sweep scrubs the figure.
    """
    token_match = NUMBER_RE.search(claim.text)
    token = token_match.group(0) if token_match is not None else claim.text
    span_idx = text.find(claim.text)
    if span_idx != -1:
        bounded = _bounded_number_search(text, token, span_idx)
        if bounded is not None and bounded.start() < span_idx + len(claim.text):
            return bounded.span()
    bounded = _bounded_number_search(text, token)
    if bounded is not None:
        return bounded.span()
    if display is not None and display != token:
        bounded = _bounded_number_search(text, display)
        if bounded is not None:
            return bounded.span()
    return None


def _apply_number_scrub(
    text: str, claim: Claim, replacement: str, marker: str
) -> tuple[str, str | None]:
    """Rewrite the claim's numeric token in ``text`` to the canonical value, or remove it (R7).

    For a NUMBER claim the published figure is ALWAYS the canonical value (``replacement`` =
    its display string, or ``""`` to remove an ungrounded/unavailable number). The token is
    located via :func:`_locate_number_token` — anchored to the claim's own span, never inside a
    longer number — and a published value is written as the caller's ``marker`` sentinel (the
    caller restores the display string after the numeric sweep, so coverage stays positional).
    Returns the edited text and the canonical display string actually published, or ``None``
    when nothing was published: removed, or token not found — in which case NOTHING is claimed
    as covered and the numeric sweep removes any stray figure (fail-closed).
    """
    span = _locate_number_token(text, claim, replacement or None)
    if span is None:
        return text, None
    start, end = span
    if replacement == "":
        edited = text[:start] + text[end:]
        edited = re.sub(r"\s{2,}", " ", edited)
        edited = re.sub(r"\s+([.,;:!?])", r"\1", edited).strip()
        return edited, None
    return text[:start] + marker + text[end:], replacement


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


__all__ = ["NumericTolerance"]
