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

# Default numeric tolerance (relative) for matching a claimed metric value against the
# canonical analytic (GROUND-R7). A claimed number within this fraction of the canonical
# value is treated as a verbatim re-statement; anything outside is scrubbed/replaced.
_DEFAULT_REL_TOLERANCE = 1e-3
# Absolute floor so near-zero canonical values do not make the relative band vanish.
_DEFAULT_ABS_TOLERANCE = 1e-6


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
    allow_list = _normalize_urls(allow_urls)
    name_library = evidence if isinstance(evidence, NameLibrary) else None
    grounded: list[GroundedClaim] = []
    text = draft_text
    for claim in claims:
        outcome = _verify_claim(claim, evidence, name_library, allow_list)
        grounded.append(outcome.grounded)
        if outcome.scrub_text is not None:
            text = _scrub_span(text, claim.text, outcome.scrub_text)
    decision = _decide(grounded)
    text = _scrub_unverified_urls(text, evidence, allow_list)
    return GroundingResult(decision=decision, claims=tuple(grounded), scrubbed_text=text)


def _scrub_unverified_urls(
    text: str, evidence: GroundingEvidence, allow_list: frozenset[str]
) -> str:
    """Remove every URL in the body not on the allow-list / a matched record (GROUND-R4).

    A SECOND, extraction-independent net: even a URL the model never surfaced as a claim is
    scrubbed unless it is first-party allow-listed or accepted by the evidence — so a
    model-invented link can never reach the athlete just because it went unextracted. A body
    with no URL is returned untouched; removals collapse the whitespace they leave behind.
    """
    if not _URL_RE.search(text):
        return text

    def _keep(match: re.Match[str]) -> str:
        url = match.group(0)
        if _normalize_url(url) in allow_list or evidence.url_allowed(url):
            return url
        return ""

    cleaned = _URL_RE.sub(_keep, text)
    if cleaned == text:
        return text
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return re.sub(r"\s+([.,;:!?])", r"\1", cleaned).strip()


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
) -> _Outcome:
    """Dispatch one claim to its kind-specific verifier (GROUND-R2)."""
    if claim.kind is ClaimKind.NUMBER:
        return _verify_number(claim, evidence)
    if claim.kind is ClaimKind.NAME:
        return _verify_name(claim, name_library)
    if claim.kind is ClaimKind.URL:
        return _verify_url(claim, evidence, allow_list)
    return _verify_statement(claim)


def _verify_number(claim: Claim, evidence: GroundingEvidence) -> _Outcome:
    """Match a claimed number against the canonical analytic within tolerance (GROUND-R7).

    A claim with no metric/value cannot be checked, so it fails closed (scrubbed). When
    the canonical computation is unavailable the number is scrubbed entirely — never a
    placeholder or zero (GROUND-R7). A claimed value within tolerance is ``grounded``; a
    value outside tolerance is ``contradicted`` and replaced by the canonical value
    (and NEVER published as stated).
    """
    if claim.metric is None or claim.value is None:
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    canonical = _canonical_metric(evidence, claim.metric, claim.ref)
    if canonical is None:
        return _scrubbed(claim, GroundVerdict.UNGROUNDED)
    if _within_tolerance(claim.value, canonical):
        citation = _metric_citation(claim, canonical)
        return _Outcome(GroundedClaim(claim, GroundVerdict.GROUNDED, citation), None)
    replacement = _format_number(canonical, claim.text)
    return _Outcome(GroundedClaim(claim, GroundVerdict.CONTRADICTED, None), replacement)


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
    normalized = _normalize_url(url)
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
    return bool(_NUMBER_RE.search(text) or _URL_RE.search(text))


def _scrubbed(claim: Claim, verdict: GroundVerdict) -> _Outcome:
    """Build a scrub outcome: the span is removed and no citation is attached (GROUND-R3)."""
    return _Outcome(GroundedClaim(claim, verdict, None), "")


def _decide(claims: Sequence[GroundedClaim]) -> GroundDecision:
    """Aggregate per-claim verdicts into a bounded recovery decision (GROUND-R9).

    - ``regenerate`` when a checkable claim is ``contradicted`` — the canonical value EXISTS
      and was already substituted in place by :func:`_verify_number` (GROUND-R3/R7), so the
      correct move is a bounded re-draft with the corrected value, NOT a coverage re-plan.
      ``contradicted`` still carries the strongest penalty: it is NEVER published (already
      enforced) and never yields ``proceed``.
    - ``abstain`` when nothing publishable survives and there is nothing to recover (every
      claim ``ungrounded``/scrubbed, no grounded survivor) — cannot answer (GROUND-R6).
    - ``regenerate`` when a claim is ``ungrounded`` but at least one grounded claim
      survives — re-draft with the offending span removed/corrected.
    - ``proceed`` when every claim is publishable (``grounded`` or a publishable
      ``complementary``) — publish.

    ``replan`` is reserved for contradictions that the in-place canonical substitution
    cannot resolve (none arise here, since every contradicted number is replaced verbatim);
    a future kind of unrecoverable contradiction would route to ``replan``.
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
        return GroundDecision.ABSTAIN
    if has_ungrounded:
        return GroundDecision.REGENERATE if has_grounded else GroundDecision.ABSTAIN
    return GroundDecision.PROCEED


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


def _within_tolerance(claimed: float, canonical: float) -> bool:
    """True iff a claimed number matches the canonical value within tolerance (GROUND-R7)."""
    if not (math.isfinite(claimed) and math.isfinite(canonical)):
        return False
    return math.isclose(
        claimed,
        canonical,
        rel_tol=_DEFAULT_REL_TOLERANCE,
        abs_tol=_DEFAULT_ABS_TOLERANCE,
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


def _format_number(canonical: float, original: str) -> str:
    """Render the canonical value into the original span, preserving its surround.

    The first numeric token in ``original`` is replaced by the canonical value so units
    and surrounding words survive (e.g. ``"CTL is 99 today"`` -> ``"CTL is 84 today"``).
    When no numeric token is found the canonical value is rendered bare.
    """
    rendered = _render_value(canonical)
    replaced, count = _NUMBER_RE.subn(rendered, original, count=1)
    return replaced if count else rendered


def _render_value(value: float) -> str:
    """Render a float without a trailing ``.0`` for integral values (display parity)."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


# --- URL + span helpers ---


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# A URL token in athlete-facing prose; the deterministic URL sweep checks every match
# against the allow-list / matched-record destinations regardless of model extraction
# (GROUND-R4: invented URLs are scrubbed unconditionally).
_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)


def _normalize_urls(urls: Iterable[str]) -> frozenset[str]:
    """Normalize an allow-list into a comparable set (GROUND-R4)."""
    return frozenset(_normalize_url(u) for u in urls)


def _normalize_url(url: str) -> str:
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


__all__ = ["NameLibrary", "ground"]
