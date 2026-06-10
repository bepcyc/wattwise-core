"""Deterministic, fail-closed grounding — the trust core (doc 50 GROUND-R1..R9).

This module owns the ONE place where an athlete-facing draft's factual claims are
verified by CODE before the deliverable may leave the ``ground`` node. The model never
self-certifies (GROUND-R1): it may only extract candidate claims (STRUCT-R5); here,
deterministic logic decides each claim's groundedness against canonical evidence
(GROUND-R2) and scrubs anything that does not match (GROUND-R3, "when in doubt, scrub").

Cited requirements (doc 50): GROUND-R1 (code verifies every number/name/date/URL claim,
model never self-certifies); GROUND-R2 (match by exact id, canonical-name library, or
numeric tolerance); GROUND-R3 (fail closed — an unmatched claim is removed or replaced,
never shipped); GROUND-R4 (URLs restricted to the allow-list + matched-record
destinations; invented URLs scrubbed unconditionally); GROUND-R5 (each survivor cites
its canonical id); GROUND-R7 (numbers taken verbatim from the canonical analytic; an
out-of-tolerance or unavailable value is scrubbed, never a placeholder/zero); GROUND-R9
(typed per-claim verdict grounded/ungrounded/contradicted/complementary aggregating to a
bounded proceed/regenerate/replan/abstain decision; contradicted is never published;
complementary publishes only when non-prescriptive and carrying no unverified
number/name/URL); GROUND-R8 (golden + property tests: planted hallucinations scrubbed,
known-good drafts unchanged).

:func:`ground` is a pure, synchronous, deterministic function of (draft, extracted
claims, canonical evidence, url allow-list) returning a
:class:`~wattwise_core.agent.contracts.GroundingResult`. It never calls a model, awaits,
or mutates state in place (GRAPH-R4); it emits no athlete-facing prose, only scrubs the
draft.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from wattwise_core.agent.contracts import (
    Claim,
    ClaimKind,
    GroundDecision,
    GroundedClaim,
    GroundingEvidence,
    GroundingResult,
    GroundVerdict,
)
from wattwise_core.agent.grounding_sweep import (
    NUMBER_RE,
    URL_RE,
    normalize_url,
    normalize_urls,
    scrub_uncovered_numbers,
    scrub_unverified_urls,
)

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


@runtime_checkable
class NameLibrary(Protocol):
    """Optional canonical-name resolver an evidence object MAY also implement (GROUND-R2).

    The base :class:`~wattwise_core.agent.contracts.GroundingEvidence` contract exposes
    only numeric and URL verification. A NAME claim (a workout/plan item) must match a
    canonical library item; an evidence object that can resolve names implements this
    structural protocol. When the evidence object does NOT implement it, every name
    claim fails closed and is scrubbed (GROUND-R3) — code never trusts a name it cannot
    resolve to a real canonical id.
    """

    def canonical_name(self, name: str) -> str | None:
        """Return the canonical record id for ``name`` if it is a real library item.

        Returns ``None`` when ``name`` matches no canonical workout/plan item; the
        grounder then scrubs the claim (fail-closed). Implementations MUST be
        deterministic and side-effect-free.
        """
        ...


def ground(
    draft_text: str,
    claims: Sequence[Claim],
    evidence: GroundingEvidence,
    allow_urls: Iterable[str],
    *,
    tolerance: NumericTolerance = _DEFAULT_TOLERANCE,
) -> GroundingResult:
    """Verify every claim against canonical evidence and scrub the draft (GROUND-R1/R3/R9).

    Each claim is classified by deterministic CODE into a typed
    :class:`~wattwise_core.agent.contracts.GroundVerdict` — the model never certifies the
    final verdict (GROUND-R1). A claim that does not match is scrubbed from
    ``draft_text`` (GROUND-R3); a number is replaced by its canonical value when one
    exists (GROUND-R7); a surviving claim carries a citation to its canonical id
    (GROUND-R5). The per-claim verdicts aggregate to a bounded
    :class:`~wattwise_core.agent.contracts.GroundDecision` (GROUND-R9): ``proceed`` when
    everything publishable grounds, ``regenerate`` / ``replan`` when there is something
    to recover, ``abstain`` when nothing grounds.

    This function is synchronous and deterministic. Numeric verification reads canonical
    values from the metric snapshot the caller resolved (see :func:`_verify_number`); it
    does not itself call a model or a live service, so the same inputs always yield the
    same result (GRAPH-R4).
    """
    allow_list = normalize_urls(allow_urls)
    name_library = evidence if isinstance(evidence, NameLibrary) else None
    grounded: list[GroundedClaim] = []
    text = draft_text
    grounded_numbers: set[str] = set()
    for claim in claims:
        outcome = _verify_claim(claim, evidence, name_library, allow_list, tolerance)
        grounded.append(outcome.grounded)
        if outcome.scrub_text is None:
            continue
        if claim.kind is ClaimKind.NUMBER:
            # A NUMBER's published figure is rewritten to the CANONICAL value at display precision
            # (verbatim for grounded, corrected for contradicted, removed for ungrounded). Rewrite
            # the model's NUMERIC TOKEN within the draft directly, not the whole claim.text span, so
            # display rewriting is robust to the model not reproducing claim.text verbatim (case /
            # wording drift). The actual canonical display string is recorded so the numeric sweep
            # below does NOT scrub the very value the grounder verified (GROUND-R7).
            text, published = _apply_number_scrub(text, claim, outcome.scrub_text)
            if published is not None:
                grounded_numbers.add(published)
        else:
            text = _scrub_span(text, claim.text, outcome.scrub_text)
    decision = _decide(grounded)
    text = scrub_unverified_urls(text, evidence, allow_list)
    text, swept = scrub_uncovered_numbers(text, grounded_numbers)
    if swept:
        # The draft carried a number the claim extractor never surfaced and the deterministic
        # sweep had to remove (GROUND-R3, mirroring the URL sweep). An unverified number reaching
        # athlete-facing text is a grounding failure even if every EXTRACTED claim grounded, so the
        # run must NOT proceed: re-draft if anything grounded survives, else abstain (fail-closed).
        decision = _downgrade_for_sweep(decision, grounded)
    return GroundingResult(decision=decision, claims=tuple(grounded), scrubbed_text=text)


def _downgrade_for_sweep(
    decision: GroundDecision, grounded: Sequence[GroundedClaim]
) -> GroundDecision:
    """Force a non-``proceed`` decision when the numeric sweep removed an uncovered number (H4).

    A swept number means an unverified figure was about to ship despite every EXTRACTED claim
    grounding — a grounding failure (GROUND-R3). ``proceed`` is downgraded to ``regenerate`` when
    some grounded claim survives (re-draft without the offending span); when NOTHING publishable
    survives the draft can no longer answer (GROUND-R6) and the swept figure was a metric the draft
    cited — a re-gatherable gap — so it downgrades to ``replan`` (recover via re-gather, bounded by
    ``reflection_count``), never an immediate abstain. An already-recovering/abstaining decision is
    left as-is (still not ``proceed``).
    """
    if decision is not GroundDecision.PROCEED:
        return decision
    if any(_is_publishable(c) for c in grounded):
        return GroundDecision.REGENERATE
    return GroundDecision.REPLAN


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


def _verify_claim(
    claim: Claim,
    evidence: GroundingEvidence,
    name_library: NameLibrary | None,
    allow_list: frozenset[str],
    tolerance: NumericTolerance,
) -> _Outcome:
    """Dispatch one claim to its kind-specific verifier (GROUND-R2)."""
    if claim.kind is ClaimKind.NUMBER:
        return _verify_number(claim, evidence, tolerance)
    if claim.kind is ClaimKind.NAME:
        return _verify_name(claim, name_library)
    if claim.kind is ClaimKind.URL:
        return _verify_url(claim, evidence, allow_list)
    return _verify_statement(claim)


def _verify_number(
    claim: Claim, evidence: GroundingEvidence, tolerance: NumericTolerance
) -> _Outcome:
    """Match a claimed number against the canonical analytic within tolerance (GROUND-R7).

    A claim with no metric/value cannot be checked, so it fails closed (scrubbed). When
    the canonical computation is unavailable the number is scrubbed entirely — never a
    placeholder or zero (GROUND-R7).

    Tolerance only decides whether a claim is RECOGNIZED as a (rounded/restated) reference to
    the canonical number — it NEVER lets the model's own figure ship. A claimed value within
    tolerance is ``grounded``, but the published span is ALWAYS rewritten to the canonical value
    rounded to display precision (so canonical ctl=100 with a within-band claim of "102" ships
    "100", never "102" — GROUND-R7 verbatim). A value outside tolerance is ``contradicted`` and
    likewise replaced by the canonical value, and NEVER published as stated. Either way the
    number the athlete sees is the canonical analytic, not the model's approximation.
    """
    if claim.metric is None or claim.value is None:
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    canonical = _canonical_metric(evidence, claim.metric, claim.ref)
    if canonical is None:
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    # The PUBLISHED figure is ALWAYS the canonical value rounded to display precision — never the
    # model's own number (GROUND-R7). ``scrub_text`` carries that bare canonical display token; the
    # caller (:func:`_apply_number_scrub`) writes it over the model's numeric token in the draft.
    canonical_display = _render_value(canonical, tolerance.display_decimals)
    if _within_tolerance(claim.value, canonical, tolerance):
        citation = _metric_citation(claim, canonical)
        return _Outcome(GroundedClaim(claim, GroundVerdict.GROUNDED, citation), canonical_display)
    return _Outcome(GroundedClaim(claim, GroundVerdict.CONTRADICTED, None), canonical_display)


def _verify_name(claim: Claim, name_library: NameLibrary | None) -> _Outcome:
    """Match a named workout/plan item against the canonical library (GROUND-R2/R3).

    A name resolves only if the evidence object can look it up AND returns a canonical
    id; otherwise it fails closed and is removed (GROUND-R3). A resolved name carries a
    citation to its canonical id (GROUND-R5).
    """
    candidate = claim.ref if claim.ref is not None else claim.text
    if name_library is None:
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    canonical_id = name_library.canonical_name(candidate)
    if canonical_id is None:
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    citation = {"kind": "name", "record": "workout", "canonical_id": canonical_id}
    return _Outcome(GroundedClaim(claim, GroundVerdict.GROUNDED, citation), None)


def _verify_url(claim: Claim, evidence: GroundingEvidence, allow_list: frozenset[str]) -> _Outcome:
    """Restrict a URL to the allow-list / matched-record destinations (GROUND-R4).

    A URL passes only if it is on the caller's allow-list OR the evidence object accepts
    it (a destination already on a matched canonical record). Anything else — including
    every model-invented URL — is scrubbed unconditionally (GROUND-R4).
    """
    url = claim.ref if claim.ref is not None else claim.text
    normalized = normalize_url(url)
    if normalized in allow_list or evidence.url_allowed(url):
        citation = {"kind": "url", "canonical_id": url}
        return _Outcome(GroundedClaim(claim, GroundVerdict.GROUNDED, citation), None)
    return _scrubbed(claim, GroundVerdict.UNGROUNDED)


def _verify_statement(claim: Claim) -> _Outcome:
    """Classify a non-factual statement (GROUND-R9 ``complementary`` rule, fail-closed).

    A statement MAY publish as ``complementary`` only when it is non-prescriptive AND
    carries NO checkable token of its own (GROUND-R9: "a statement carries no checkable
    number/name/URL"). This is ENFORCED deterministically here, not trusted from the
    model's claim-kind label: a statement smuggling a numeric literal or a URL (e.g. a
    ``STATEMENT`` whose text is "Your CTL is 999") is treated as ``ungrounded`` and scrubbed,
    so a factual span can never reach the athlete unverified by being mislabeled non-factual.
    A prescriptive statement (a target/instruction) without a backing grounded prescription
    is likewise ungrounded (fail-closed default).
    """
    if claim.prescriptive or _carries_checkable_token(claim.text):
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    return _Outcome(GroundedClaim(claim, GroundVerdict.COMPLEMENTARY, None), None)


def _carries_checkable_token(text: str) -> bool:
    """True if ``text`` contains a numeric literal or a URL (a checkable factual token).

    The deterministic guard for GROUND-R9: a complementary statement must be purely
    non-factual; a number or URL in its span makes it a checkable claim that MUST be
    verified by its kind-specific verifier, never published on a statement's free pass.
    """
    return bool(NUMBER_RE.search(text) or URL_RE.search(text))


def _scrubbed(claim: Claim, verdict: GroundVerdict) -> _Outcome:
    """Build a scrub outcome: the span is removed and no citation is attached (GROUND-R3)."""
    return _Outcome(GroundedClaim(claim, verdict, None), "")


def _decide(claims: Sequence[GroundedClaim]) -> GroundDecision:
    """Aggregate per-claim verdicts into a bounded recovery decision (GROUND-R6/R9).

    - ``regenerate`` when a checkable claim is ``contradicted`` — the canonical value EXISTS
      and was already substituted in place by :func:`_verify_number` (GROUND-R3/R7), so the
      correct move is a bounded re-draft with the corrected value, NOT a coverage re-plan.
      ``contradicted`` still carries the strongest penalty: it is NEVER published (already
      enforced) and never yields ``proceed``.
    - ``replan`` when scrubbing left NOTHING publishable (GROUND-R6) AND a scrubbed claim is a
      RE-GATHERABLE metric gap (:func:`_has_regatherable_metric_gap`): GROUND-R9 routes
      ``ground -> reflect -> plan_retrieval`` to re-gather, BOUNDED by ``reflection_count``
      (REFLECT-R4) — the bound is the fail-closed floor (degrades to a truthful limitation if
      re-gather still fails), so this only ATTEMPTS recovery first.
    - ``abstain`` when nothing publishable survives and there is nothing to RE-GATHER — every claim
      is a fabrication a replan could never ground (NAME/URL/prescription), or there are no claims.
    - ``regenerate`` when a claim is ``ungrounded`` but a grounded claim survives — the answer is
      still producible, re-draft with the offending span removed.
    - ``proceed`` when every claim is publishable (``grounded`` or a publishable ``complementary``).
    """
    verdicts = [c.verdict for c in claims]
    has_grounded = any(v is GroundVerdict.GROUNDED for v in verdicts)
    has_publishable = any(_is_publishable(c) for c in claims)
    has_contradicted = any(v is GroundVerdict.CONTRADICTED for v in verdicts)
    has_ungrounded = any(v is GroundVerdict.UNGROUNDED for v in verdicts)
    if has_contradicted:
        # The contradicted number was replaced by the canonical value in place; re-draft
        # with the corrected text rather than re-planning for different evidence.
        return GroundDecision.REGENERATE
    if not has_publishable:
        # Nothing publishable survived — the deliverable cannot answer (GROUND-R6). If the loss
        # was to a MISSING metric (re-gatherable), recover via ``replan``; otherwise abstain.
        return GroundDecision.REPLAN if _has_regatherable_metric_gap(claims) else (
            GroundDecision.ABSTAIN
        )
    if has_ungrounded:
        return GroundDecision.REGENERATE if has_grounded else GroundDecision.ABSTAIN
    return GroundDecision.PROCEED


def _has_regatherable_metric_gap(claims: Sequence[GroundedClaim]) -> bool:
    """True iff a scrubbed claim is a missing-metric gap that re-gathering could close (GROUND-R6).

    The deterministic signal that distinguishes a RECOVERABLE under-grounding (route ``replan``)
    from a pure fabrication (route ``abstain``): an ungrounded NUMBER is a real metric the draft
    cited whose canonical value was MISSING at grounding time (``_verify_number`` returns
    ``UNGROUNDED`` only when the canonical value was ``None`` — never retrieved); that gap is what
    re-gathering closes (GROUND-R9 ``replan`` = "missing evidence"). An ungrounded NAME/URL or
    scrubbed prescription is a fabrication retrieval can never ground, so it does NOT replan.
    """
    return any(
        c.verdict is GroundVerdict.UNGROUNDED and c.claim.kind is ClaimKind.NUMBER
        for c in claims
    )


def _is_publishable(claim: GroundedClaim) -> bool:
    """A claim may publish iff it grounded, or is a non-prescriptive complementary (GROUND-R9).

    A ``contradicted`` claim is NEVER publishable (GROUND-R9); an ``ungrounded`` claim is
    scrubbed; a ``complementary`` claim publishes only when non-prescriptive (already
    enforced in :func:`_verify_statement`, re-checked here as a fail-closed guard).
    """
    if claim.verdict is GroundVerdict.GROUNDED:
        return True
    if claim.verdict is GroundVerdict.COMPLEMENTARY:
        return not claim.claim.prescriptive
    return False


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


# --- span helpers (the URL/number sweep primitives live in ``grounding_sweep``) ---


def _scrub_span(text: str, span: str, replacement: str) -> str:
    """Replace one occurrence of ``span`` in ``text`` with ``replacement`` (GROUND-R3).

    An empty ``replacement`` removes the span and collapses the doubled whitespace it
    leaves behind, so a scrubbed draft reads cleanly without the unverified fragment.
    The match is literal (no regex) and bounded to the first occurrence so unrelated
    repeats of common words are left intact.
    """
    if not span:
        return text
    idx = text.find(span)
    if idx == -1:
        return text
    edited = text[:idx] + replacement + text[idx + len(span) :]
    if replacement == "":
        edited = re.sub(r"\s{2,}", " ", edited)
        edited = re.sub(r"\s+([.,;:!?])", r"\1", edited)
    return edited.strip() if replacement == "" else edited


__all__ = ["NameLibrary", "NumericTolerance", "ground"]
