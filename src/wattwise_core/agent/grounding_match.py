"""Numeric-match primitives for the deterministic grounder (doc 50 GROUND-R5/R7).

The focused sibling of :mod:`wattwise_core.agent.grounding` (QUAL-R9 size split) that owns the
numeric side of claim verification: the config-carried :class:`NumericTolerance` band, the
fail-closed canonical-value read (:func:`_canonical_metric` / :func:`_as_float`), the
tolerance comparison (:func:`_within_tolerance`), the grounded-number citation
(:func:`_metric_citation`, GROUND-R5), and the canonical-display rewrite of the model's numeric
token (:func:`_apply_number_scrub` / :func:`_render_value`, GROUND-R7 verbatim). All functions
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


def _apply_number_scrub(text: str, claim: Claim, replacement: str) -> tuple[str, str | None]:
    """Rewrite the model's numeric token in ``text`` to the canonical value, or remove it (R7).

    For a NUMBER claim the published figure is ALWAYS the canonical value (``replacement`` =
    its display string, or ``""`` to remove an ungrounded/unavailable number). This finds the
    model's own numeric token — the first number in ``claim.text`` (e.g. ``"63.50"`` from
    ``"your fitness is at 63.50"``) — and replaces THAT token in the draft, so the rewrite is
    robust to the model not reproducing ``claim.text`` verbatim (case / wording drift). Returns the
    edited text and the canonical display string actually published (so the numeric-coverage sweep
    treats it as covered), or ``None`` when nothing was published (removed / token not found).
    """
    token_match = NUMBER_RE.search(claim.text)
    token = token_match.group(0) if token_match is not None else claim.text
    idx = text.find(token)
    if idx == -1:
        # The model's token is not literally in the draft; nothing to publish/remove here. The
        # numeric sweep is the backstop for any uncovered number that DID reach the draft.
        return text, replacement or None
    edited = text[:idx] + replacement + text[idx + len(token) :]
    if replacement == "":
        edited = re.sub(r"\s{2,}", " ", edited)
        edited = re.sub(r"\s+([.,;:!?])", r"\1", edited).strip()
        return edited, None
    return edited, replacement


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
