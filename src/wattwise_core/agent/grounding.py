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

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from wattwise_core.agent.contracts import (
    Claim,
    ClaimKind,
    GroundDecision,
    GroundedClaim,
    GroundingEvidence,
    GroundingResult,
    GroundVerdict,
)

# The numeric-match primitives (the tolerance band, the fail-closed canonical-value read, the
# GROUND-R5 citation, the POSITIONAL GROUND-R7 canonical-display rewrite) live in the focused
# :mod:`grounding_match` sibling and the per-claim NUMBER verifier (canonical match + user-request
# echo) in :mod:`grounding_numeric` (QUAL-R9 size split); ``NumericTolerance`` is re-exported here
# so every historical ``from wattwise_core.agent.grounding import NumericTolerance`` path stays.
from wattwise_core.agent.grounding_match import (
    _DEFAULT_TOLERANCE,
    NumericTolerance,
    _apply_number_scrub,
    remove_span_clean,
    shift_ranges_after,
)
from wattwise_core.agent.grounding_numeric import (
    _Outcome,
    _parse_request_values,
    _scrubbed,
    _verify_number,
)
from wattwise_core.agent.grounding_sweep import (
    NUMBER_RE,
    URL_RE,
    normalize_url,
    normalize_urls,
    scrub_uncovered_numbers,
    scrub_unverified_urls,
)


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
    request_numbers: frozenset[str] = frozenset(),
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

    ``request_numbers`` are the numeric tokens the ATHLETE supplied in their own request text. A
    number the user supplied is not a canonical-data claim — it is the user's own constraint (e.g.
    "5-7 hours a week" on a plan request) — so an ECHO of it is sayable: it verifies as a
    user-request echo (cited ``user_request``) instead of failing closed against canonical
    analytics, and the numeric sweep treats it as covered. Canonical verification still wins
    whenever the claim's metric resolves to an available canonical value (a real metric claim is
    never excused by a coincidental request echo).
    """
    allow_list = normalize_urls(allow_urls)
    name_library = evidence if isinstance(evidence, NameLibrary) else None
    request_values = _parse_request_values(request_numbers)
    grounded: list[GroundedClaim] = []
    text = draft_text
    covered: list[tuple[int, int]] = []
    for claim in claims:
        outcome = _verify_claim(
            claim, evidence, name_library, allow_list, tolerance, request_values
        )
        grounded.append(outcome.grounded)
        if outcome.scrub_text is None:
            continue
        if claim.kind is ClaimKind.NUMBER:
            # A NUMBER's published figure is rewritten to the CANONICAL value at display precision
            # (verbatim for grounded, corrected for contradicted, removed for ungrounded). The
            # rewrite is POSITIONAL (issue #4): the token is located in the claim's OWN span,
            # word-bounded, skipping ranges earlier claims already verified — never the first
            # ``str.find`` hit, so two claims sharing a token each land in their own span. The
            # character range actually published is tracked in ``covered`` so the numeric sweep
            # below does NOT scrub the very value the grounder verified (GROUND-R7) — and ONLY
            # that range: a leftover equal-looking token elsewhere is still swept.
            text, covered = _apply_number_scrub(text, claim, outcome.scrub_text, covered)
        else:
            text, covered = _scrub_span(text, claim.text, outcome.scrub_text, covered)
    decision = _decide(grounded)
    # The numeric sweep consumes the positional coverage FIRST, while the ranges are still
    # valid; the URL sweep edits text afterwards (its removals would shift the ranges).
    text, swept = scrub_uncovered_numbers(text, covered)
    if swept:
        # The draft carried a number the claim extractor never surfaced and the deterministic
        # sweep had to remove (GROUND-R3, mirroring the URL sweep). An unverified number reaching
        # athlete-facing text is a grounding failure even if every EXTRACTED claim grounded, so the
        # run must NOT proceed: re-draft if anything grounded survives, else abstain (fail-closed).
        decision = _downgrade_for_sweep(decision, grounded)
    text = scrub_unverified_urls(text, evidence, allow_list)
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


def _verify_claim(
    claim: Claim,
    evidence: GroundingEvidence,
    name_library: NameLibrary | None,
    allow_list: frozenset[str],
    tolerance: NumericTolerance,
    request_values: frozenset[float] = frozenset(),
) -> _Outcome:
    """Dispatch one claim to its kind-specific verifier (GROUND-R2)."""
    if claim.kind is ClaimKind.NUMBER:
        return _verify_number(claim, evidence, tolerance, request_values)
    if claim.kind is ClaimKind.NAME:
        return _verify_name(claim, name_library)
    if claim.kind is ClaimKind.URL:
        return _verify_url(claim, evidence, allow_list)
    return _verify_statement(claim)


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
        return (
            GroundDecision.REPLAN
            if _has_regatherable_metric_gap(claims)
            else (GroundDecision.ABSTAIN)
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
        c.verdict is GroundVerdict.UNGROUNDED and c.claim.kind is ClaimKind.NUMBER for c in claims
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


# --- span helpers (the URL/number sweep primitives live in ``grounding_sweep``;
# --- the numeric primitives in ``grounding_numeric``) ---


def _scrub_span(
    text: str, span: str, replacement: str, covered: Sequence[tuple[int, int]]
) -> tuple[str, list[tuple[int, int]]]:
    """Replace one occurrence of ``span`` in ``text`` with ``replacement`` (GROUND-R3).

    An empty ``replacement`` removes the span and tidies the seam it leaves behind
    (seam-local, :func:`~wattwise_core.agent.grounding_match.remove_span_clean`), so a
    scrubbed draft reads cleanly without the unverified fragment. The match is literal (no
    regex) and bounded to the first occurrence so unrelated repeats of common words are
    left intact. Returns the edited text and the numeric-coverage ranges re-based across
    the edit (issue #4: every edit must shift the positional coverage behind it, or the
    numeric sweep would test stale positions).
    """
    if not span:
        return text, list(covered)
    idx = text.find(span)
    if idx == -1:
        return text, list(covered)
    end = idx + len(span)
    if replacement == "":
        edited, delta = remove_span_clean(text, idx, end)
        return edited, shift_ranges_after(covered, end, delta)
    edited = text[:idx] + replacement + text[end:]
    return edited, shift_ranges_after(covered, end, len(replacement) - len(span))


__all__ = ["NameLibrary", "NumericTolerance", "ground"]
