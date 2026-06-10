"""Numeric claim verification + the GROUND-R7 number machinery (QUAL-R9 size split).

The focused sibling of :mod:`wattwise_core.agent.grounding` that owns everything NUMERIC in the
fail-closed grounder: the config-loaded :class:`NumericTolerance` band, the per-claim NUMBER
verifier (canonical match within tolerance, the user-request ECHO path, contradiction
correction), the canonical-value display rendering, and the in-draft numeric span rewrite.
Behaviour is identical to the prior inline definitions; this is purely a size decomposition that
keeps ``grounding`` under the QUAL-R9 module ceiling. Everything here is pure, synchronous and
deterministic (GRAPH-R4) — no model call, no awaits.

Cited requirements: GROUND-R1, GROUND-R2, GROUND-R3, GROUND-R5, GROUND-R7, QUAL-R9.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from wattwise_core.agent.contracts import (
    Claim,
    GroundedClaim,
    GroundingEvidence,
    GroundVerdict,
)
from wattwise_core.agent.grounding_sweep import NUMBER_RE, remove_numeric_span

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

    Carried into :func:`ground` so the threshold is config-loaded content (§16 / SKILL-R1),
    not a literal baked into the gate. ``rel`` is the relative band (a fraction of the
    canonical magnitude) and ``abs_`` the absolute floor for near-zero canonical values; a
    claimed value within EITHER band of the canonical value grounds. The defaults accept a
    human-precision display of the canonical number and scrub a fabrication.

    ``display_decimals`` is the precision the CANONICAL value is rounded to when it is PUBLISHED
    in place of the model's numeric span (GROUND-R7 verbatim): the model's approximation — even
    a within-tolerance one like "102" for a canonical 100 — is NEVER what reaches the athlete;
    the canonical value, rounded for a human display, always is.
    """

    rel: float = _DEFAULT_REL_TOLERANCE
    abs_: float = _DEFAULT_ABS_TOLERANCE
    display_decimals: int = _DEFAULT_DISPLAY_DECIMALS


_DEFAULT_TOLERANCE = NumericTolerance()


class _Outcome:
    """Internal per-claim result: the typed verdict + how to edit the draft.

    ``scrub_text`` is ``None`` to leave the span untouched (a grounded survivor), an
    empty string to remove the span (fail-closed scrub, GROUND-R3), or a replacement
    string to substitute a canonical value/library item (GROUND-R7/R3).
    """

    __slots__ = ("grounded", "scrub_text")

    def __init__(self, grounded: GroundedClaim, scrub_text: str | None) -> None:
        self.grounded = grounded
        self.scrub_text = scrub_text


def _scrubbed(claim: Claim, verdict: GroundVerdict) -> _Outcome:
    """Build a scrub outcome: the span is removed and no citation is attached (GROUND-R3)."""
    return _Outcome(GroundedClaim(claim, verdict, None), "")


def _parse_request_values(request_numbers: Iterable[str]) -> frozenset[float]:
    """Parse the user-request numeric tokens into comparable float values (fail-closed).

    A token that does not parse as a finite float is dropped — it can then never excuse a
    claim, so a malformed echo set degenerates to the prior all-canonical behaviour.
    """
    values: set[float] = set()
    for token in request_numbers:
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value):
            values.add(value)
    return frozenset(values)


def _request_echo(claim: Claim) -> _Outcome:
    """Ground a number as an ECHO of the athlete's own request (the plan constraint).

    The published span is left exactly as the user-supplied figure (no canonical rewrite — there
    is no canonical analytic behind it); the citation records the ``user_request`` provenance so a
    deliverable can distinguish an echoed constraint from a canonical metric (GROUND-R5). The
    claim's own numeric token is re-published verbatim so the numeric-coverage sweep treats it as
    covered.
    """
    token_match = NUMBER_RE.search(claim.text)
    token = token_match.group(0) if token_match is not None else _render_value(claim.value or 0, 1)
    citation = {"kind": "user_request", "record_id": "user_request", "value": claim.value}
    return _Outcome(GroundedClaim(claim, GroundVerdict.GROUNDED, citation), token)


def _verify_number(
    claim: Claim,
    evidence: GroundingEvidence,
    tolerance: NumericTolerance,
    request_values: frozenset[float] = frozenset(),
) -> _Outcome:
    """Match a claimed number against the canonical analytic within tolerance (GROUND-R7).

    A claim with no value cannot be checked, so it fails closed (scrubbed). When the
    canonical computation is unavailable the number is scrubbed entirely — never a
    placeholder or zero (GROUND-R7) — UNLESS the number is an ECHO of one the ATHLETE
    supplied in their own request (``request_values``): a user-supplied figure is the
    request's own constraint, not a canonical-data claim, so it stays sayable and is cited
    as a ``user_request`` echo. Canonical verification always runs FIRST: a claim whose
    metric resolves to an available canonical value is verified (and corrected) against
    canonical data even if its value coincides with a request number.

    Tolerance only decides whether a claim is RECOGNIZED as a (rounded/restated) reference to
    the canonical number — it NEVER lets the model's own figure ship. A claimed value within
    tolerance is ``grounded``, but the published span is ALWAYS rewritten to the canonical value
    rounded to display precision (so canonical ctl=100 with a within-band claim of "102" ships
    "100", never "102" — GROUND-R7 verbatim). A value outside tolerance is ``contradicted`` and
    likewise replaced by the canonical value, and NEVER published as stated. Either way the
    number the athlete sees is the canonical analytic, not the model's approximation.
    """
    if claim.value is None:
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    canonical = (
        _canonical_metric(evidence, claim.metric, claim.ref) if claim.metric is not None else None
    )
    if canonical is None:
        # Sign-insensitive: a range's second member ("5-7") extracts with the dash attached,
        # so a claimed 7 (or -7) matches the user's sign-stripped echo token.
        if claim.value in request_values or abs(claim.value) in request_values:
            return _request_echo(claim)
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    # The PUBLISHED figure is ALWAYS the canonical value rounded to display precision — never the
    # model's own number (GROUND-R7). ``scrub_text`` carries that bare canonical display token; the
    # caller (:func:`_apply_number_scrub`) writes it over the model's numeric token in the draft.
    canonical_display = _render_value(canonical, tolerance.display_decimals)
    if _within_tolerance(claim.value, canonical, tolerance):
        citation = _metric_citation(claim, canonical)
        return _Outcome(GroundedClaim(claim, GroundVerdict.GROUNDED, citation), canonical_display)
    return _Outcome(GroundedClaim(claim, GroundVerdict.CONTRADICTED, None), canonical_display)


# --- numeric helpers (GROUND-R7) ---


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
    if replacement == "":
        # Remove by the WHOLE containing numeric phrase (a "5-7" range scrubs as one span, plus
        # any orphan leading dash / trailing empty unit), so a claim-level scrub never leaves a
        # dangling range dash or a bare unit in the prose (VOICE-R2 clean copy).
        return remove_numeric_span(text, idx, idx + len(token)), None
    edited = text[:idx] + replacement + text[idx + len(token) :]
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
