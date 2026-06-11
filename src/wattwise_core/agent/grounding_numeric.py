"""Per-claim NUMBER verification + the user-request ECHO path (doc 50 GROUND-R3/R5/R7).

The focused sibling of :mod:`wattwise_core.agent.grounding` (QUAL-R9 size split) that owns the
per-claim verdict machinery the grounder dispatches on: the :class:`_Outcome` verdict+edit pair,
the NUMBER verifier (canonical match within tolerance, contradiction correction, GROUND-R7) and
the user-request ECHO path — a number the ATHLETE supplied in their own request (a plan's
"5-7 hours a week") is the request's own constraint, not a canonical-data claim, so an echo of it
grounds with a ``user_request`` citation instead of failing closed (GROUND-R3 scope, GROUND-R5).

The POSITIONAL primitives the published rewrite runs on (word-bounded token spans, character-RANGE
coverage — issue #4) live in :mod:`wattwise_core.agent.grounding_match`; an echoed figure is
republished VERBATIM over its own anchored span via the same
:func:`~wattwise_core.agent.grounding_match._apply_number_scrub` path, so echo coverage is a
character range too (never string membership) and an ambiguous bare-token echo fails closed like
any other claim. Everything here is pure, synchronous and deterministic (GRAPH-R4).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from wattwise_core.agent.contracts import (
    Claim,
    GroundedClaim,
    GroundingEvidence,
    GroundVerdict,
)
from wattwise_core.agent.grounding_match import (
    _THOUSANDS_AWARE_NUMBER_RE,
    NumericTolerance,
    _canonical_metric,
    _metric_citation,
    _render_value,
    _within_tolerance,
)


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

    The published figure is the user-supplied token re-published VERBATIM (no canonical rewrite —
    there is no canonical analytic behind it); the citation records the ``user_request``
    provenance so a deliverable can distinguish an echoed constraint from a canonical metric
    (GROUND-R5). Republishing the claim's own token through the positional rewrite path
    (:func:`~wattwise_core.agent.grounding_match._apply_number_scrub`) marks the token's OWN
    character range as covered — never every string-equal occurrence (issue #4): the echo
    legitimizes the value for ITS anchored span only, and an ambiguous bare-token echo (multiple
    uncovered occurrences, no anchor) publishes nothing and is swept (fail-closed).
    """
    token_match = _THOUSANDS_AWARE_NUMBER_RE.search(claim.text)
    token = token_match.group(0) if token_match is not None else _render_value(claim.value or 0, 1)
    citation = {"kind": "user_request", "record_id": "user_request", "value": claim.value}
    return _Outcome(GroundedClaim(claim, GroundVerdict.GROUNDED, citation), token)


def _verify_number(
    claim: Claim,
    evidence: GroundingEvidence,
    tolerance: NumericTolerance,
    request_values: frozenset[float] = frozenset(),
    *,
    echo_blocked: bool = False,
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
        # so a claimed 7 (or -7) matches the user's sign-stripped echo token. ``echo_blocked``
        # is the binding guard's R10d veto (issue #10): a METRIC-SHAPED sentence may never
        # ground as a request echo — it verifies canonically or scrubs (fail-closed).
        if not echo_blocked and (
            claim.value in request_values or abs(claim.value) in request_values
        ):
            return _request_echo(claim)
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    # The PUBLISHED figure is ALWAYS the canonical value rounded to display precision — never the
    # model's own number (GROUND-R7). ``scrub_text`` carries that bare canonical display token; the
    # caller (:func:`~wattwise_core.agent.grounding_match._apply_number_scrub`) writes it over the
    # model's numeric token in the claim's OWN anchored span of the draft (issue #4).
    canonical_display = _render_value(canonical, tolerance.display_decimals)
    if _within_tolerance(claim.value, canonical, tolerance):
        citation = _metric_citation(claim, canonical)
        return _Outcome(GroundedClaim(claim, GroundVerdict.GROUNDED, citation), canonical_display)
    return _Outcome(GroundedClaim(claim, GroundVerdict.CONTRADICTED, None), canonical_display)


__all__ = ["_Outcome", "_parse_request_values", "_request_echo", "_scrubbed", "_verify_number"]
