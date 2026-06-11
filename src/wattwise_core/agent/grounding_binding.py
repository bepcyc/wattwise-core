"""Deterministic claim-binding: the SENTENCE selects the cell (issue #10, GROUND-R10).

The trust hole this closes: the value gate verifies a NUMBER against the canonical
``(metric, as_of)`` cell the CLAIM names — but that binding was extracted by the SAME
model that wrote the draft, so the defendant routed its own cross-examination ("your
fatigue is 71" checked against ``ctl``; "your CTL is 71" checked at a cherry-picked past
date; a metric-shaped sentence excused as a user-request echo). Generator and extractor
share one set of weights, so their failure modes correlate — verification confirmed the
hallucination instead of catching it.

The fix is an AUTHORITY INVERSION, not another referee: :meth:`BindingGuard.rebind`
re-derives each claim's binding FROM ITS OWN SENTENCE — the only text the athlete will
read — through the SAME config-loaded
:class:`~wattwise_core.agent.metric_equivalence.MetricEquivalence` vocabulary the value
verifier uses (GROUND-R2). The model's extracted ``(metric, as_of)`` degrades to a span
pointer with no routing power: a sentence naming one metric IS bound to that metric; a
sentence stating one explicit date IS pinned to that date; a present-tense undated
sentence with a stale extracted date verifies against the LATEST value. A mis-extraction
therefore stops being a scrub and becomes a CORRECTION — the ordinary GROUND-R7 machinery
substitutes the right cell's canonical value in place, so the athlete ends up with the
true figure instead of a hole.

:meth:`BindingGuard.check_number` is the residual fail-closed floor for what rebinding
must not guess at (an AMBIGUOUS multi-metric sentence whose claim matches none of its
labels; the strict no-label rule; the stale-date rule for callers that skip rebinding),
and :meth:`BindingGuard.echo_blocked` keeps a metric-shaped sentence off the user-request
echo pass (R10d). Everything is pure and deterministic over ``(draft, claim, config,
reference date)`` — no model call, no service read (GRAPH-R4); rebinding can only point
verification at the sentence's OWN cell, never invent a value. SHADOW mode is owned by
the caller (:class:`~wattwise_core.agent.grounding_evidence.ClaimGrounder` records
would-be rebinds/violations without applying them); the guard itself only computes.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from wattwise_core.agent.contracts import Claim, ClaimKind
from wattwise_core.agent.grounding_match import (
    _THOUSANDS_AWARE_NUMBER_RE,
    bounded_number_pattern,
)
from wattwise_core.agent.metric_equivalence import MetricEquivalence


class BindingMode(StrEnum):
    """Rollout mode for the binding guards (issue #10 Phase 4 rollout discipline).

    ``off`` skips the guards entirely (the pre-guard behaviour); ``shadow`` assesses and
    records violations on the observability surface WITHOUT scrubbing (the safe first
    rollout step); ``enforce`` fails closed on every violation. Loaded config content
    (CFG-R1a), never a code-baked policy.
    """

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class BindingViolation(StrEnum):
    """A deterministic per-claim binding violation (proposed GROUND-R10)."""

    #: The claim's sentence names a canonical metric DIFFERENT from the claim's own
    #: ``metric`` — the wrong-cell mis-attribution (issue #10 scenario 2). Never publishable.
    METRIC_MISMATCH = "metric_mismatch"
    #: A present-tense, undated sentence carries a PAST ``as_of`` — the stale-as-current
    #: temporal cherry-pick (issue #10 scenario 1, the inverse-H2 hole).
    STALE_AS_OF = "stale_as_of"
    #: The sentence names NO resolvable metric label at all while the claim cites one —
    #: enforced only under ``require_metric_label`` (strict R10a; high over-scrub risk,
    #: so it ships config-gated and off by default).
    UNLABELED_METRIC = "unlabeled_metric"


class BindingEvent(StrEnum):
    """A deterministic REBIND the guard applied to a claim (the authority inversion).

    Where a violation can only remove sayability, a rebind RESTORES it correctly: the
    sentence's own words re-select the canonical cell and the ordinary value machinery
    then verifies — and, for a contradicted figure, CORRECTS — against the right cell
    (GROUND-R7). Recorded per event so a drifting extractor is alertable (AGT-OBS-R7).
    """

    #: The sentence names exactly ONE canonical metric and the claim cited a different
    #: (or no) one — the claim is re-bound to the sentence's metric.
    METRIC_REBOUND = "metric_rebound"
    #: A present-tense, undated sentence carried a PAST ``as_of`` — the date is dropped
    #: so the claim verifies against the LATEST canonical value (what "is" asserts).
    AS_OF_REBOUND = "as_of_rebound"
    #: The sentence states exactly ONE explicit ISO date — the claim's ``as_of`` is
    #: pinned to it (closing the dated-sentence-with-absent-ref half of H2).
    AS_OF_PINNED = "as_of_pinned"


# No-config fallback deixis lexicon (CFG-R1a: production loads ``[agent.binding]``; this
# is the conservative default for seams that inject no policy). Present-tense copulas are
# deliberately included: "your CTL is 71" asserts NOW, so a past as_of contradicts it.
_DEFAULT_PRESENT_DEIXIS: tuple[str, ...] = (
    "is",
    "are",
    "sits at",
    "stands at",
    "today",
    "currently",
    "right now",
    "at the moment",
)

# An EXPLICIT date marker inside the sentence makes the dated reading explicit — the H2
# path then governs (verify AT that date), so the present-deixis rule stands down. "may"
# is deliberately absent from the month list: as a modal verb it would silently suppress
# the guard ("you may want to...") — fail-closed, a "May 1" sentence over-scrubs instead.
_DATE_MARKER_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\bas of\b|\b(?:january|february|march|april|june|july|august"
    r"|september|october|november|december)\b",
    re.IGNORECASE,
)

# Sentence boundary characters for the enclosing-sentence scan (deterministic, lexical).
_SENTENCE_BOUNDARY = frozenset(".!?\n")

# An explicit ISO date stated IN a sentence: the one temporal binding the prose can state
# unambiguously, so it is the one the rebind may pin ``as_of`` to (month-name dates are
# ambiguous without a year and stay with the entailment layer).
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# The athlete-facing FORM metric reads the canonical ``tsb`` field (the same PmcDay.tsb,
# see ``CanonicalEvidence._pmc_scalar``), so for BINDING comparison the two keys are one
# quantity — "your form is +5" bound to ``tsb`` is consistent, never a mismatch.
_CANONICAL_FOLD: dict[str, str] = {"form": "tsb"}


def _fold_key(key: str) -> str:
    """Fold canonical keys that read the same underlying quantity (form == tsb)."""
    return _CANONICAL_FOLD.get(key, key)


@dataclass(frozen=True, slots=True)
class BindingPolicy:
    """Config-carried binding policy (CFG-R1a; loaded from ``[agent.binding]``).

    ``present_deixis`` is the lexicon marking a sentence as asserting the PRESENT;
    ``freshness_days`` is how far behind the reference date an ``as_of`` may sit and
    still count as current (canonical series can lag a day, e.g. an HRV sample recorded
    yesterday); ``require_metric_label`` enables the strict R10a no-label rule.
    """

    present_deixis: tuple[str, ...] = _DEFAULT_PRESENT_DEIXIS
    freshness_days: int = 1
    require_metric_label: bool = False


class BindingGuard:
    """Pure per-claim binding checks over the ORIGINAL draft (proposed GROUND-R10).

    Construction folds the loaded metric surface forms (alias keys + canonical enum
    values) into word-bounded patterns once; every check is then a deterministic lexical
    pass over the claim's own enclosing sentence. ``reference_date`` anchors the temporal
    rule; the caller (the grounder seam) resolves it ONCE per run via :meth:`anchored` so
    :func:`~wattwise_core.agent.grounding.ground` stays deterministic (GRAPH-R4).
    """

    def __init__(
        self,
        equivalence: MetricEquivalence,
        *,
        policy: BindingPolicy | None = None,
        mode: BindingMode = BindingMode.ENFORCE,
        reference_date: _dt.date | None = None,
    ) -> None:
        self._equivalence = equivalence
        self._policy = policy if policy is not None else BindingPolicy()
        self.mode = mode
        self._reference_date = reference_date
        self._label_patterns: tuple[tuple[re.Pattern[str], str], ...] = tuple(
            (re.compile(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])"), form)
            for form in sorted(equivalence.surface_forms())
        )
        self._deixis_patterns: tuple[re.Pattern[str], ...] = tuple(
            re.compile(rf"(?<![a-z0-9]){re.escape(_fold_text(term))}(?![a-z0-9])")
            for term in self._policy.present_deixis
            if term.strip()
        )

    def anchored(self, reference_date: _dt.date) -> BindingGuard:
        """A copy of this guard pinned to ``reference_date`` (one anchor per run).

        The grounder resolves the anchor once (the same reference date its canonical
        evidence uses) so all temporal checks inside one ``ground`` call share one clock
        and the call stays a deterministic function of its inputs (GRAPH-R4).
        """
        if self._reference_date is not None:
            return self
        guard = BindingGuard(
            self._equivalence,
            policy=self._policy,
            mode=self.mode,
            reference_date=reference_date,
        )
        return guard

    def check_number(self, draft: str, claim: Claim) -> BindingViolation | None:
        """The binding verdict for one NUMBER claim, or ``None`` when consistent.

        Fail-closed direction only: a violation can only remove sayability. A claim whose
        sentence cannot be located in the draft returns ``None`` — the positional rewrite
        and the numeric-coverage sweep already own fail-closure for unanchorable spans.
        """
        if claim.kind is not ClaimKind.NUMBER:
            return None
        sentence = self._sentence_for(draft, claim)
        if sentence is None:
            return None
        labels = self._sentence_metric_keys(sentence)
        claim_key = self._claim_key(claim)
        if claim_key is not None:
            if labels and claim_key not in labels:
                return BindingViolation.METRIC_MISMATCH
            if not labels and self._policy.require_metric_label:
                return BindingViolation.UNLABELED_METRIC
        if self._is_stale_as_of(sentence, claim):
            return BindingViolation.STALE_AS_OF
        return None

    def echo_blocked(self, draft: str, claim: Claim) -> bool:
        """True iff the claim's sentence is metric-shaped (R10d: no echo free pass).

        A number in a sentence that names a canonical metric is a metric claim; the
        athlete's own request numbers may excuse a restated CONSTRAINT ("7 hours a
        week"), never a sentence that reads as canonical data ("your TSB is 100").
        """
        sentence = self._sentence_for(draft, claim)
        if sentence is None:
            return False
        return bool(self._sentence_metric_keys(sentence))

    def assess(self, draft: str, claims: tuple[Claim, ...]) -> tuple[BindingViolation, ...]:
        """Every violation across ``claims`` in claim order (the SHADOW-mode reading).

        Used by the grounder seam to record would-be enforcement on the observability
        surface without scrubbing (issue #10 Phase 4: shadow before enforce).
        """
        found: list[BindingViolation] = []
        for claim in claims:
            violation = self.check_number(draft, claim)
            if violation is not None:
                found.append(violation)
        return tuple(found)

    def rebind(
        self, draft: str, claims: Sequence[Claim]
    ) -> tuple[tuple[Claim, ...], tuple[BindingEvent, ...]]:
        """Re-derive each NUMBER claim's canonical cell out of its OWN sentence (R10).

        The authority inversion that closes issue #10 by construction: the model's
        extracted ``(metric, as_of)`` could route verification to whichever cell agreed
        with the draft, so here the SENTENCE — the only thing the athlete will read —
        re-selects the cell deterministically and the extraction degrades to a span
        pointer. Verification (and the GROUND-R7 in-place correction) then runs against
        the cell the prose actually asserts:

        * the sentence names exactly ONE canonical metric -> the claim is bound to it
          (a wrong or missing extracted metric is OVERRIDDEN, never trusted);
        * the sentence states exactly ONE explicit ISO date -> ``as_of`` is pinned to it
          (a dated sentence can no longer silently verify against the latest day);
        * a present-tense, undated sentence with a stale past ``as_of`` -> the date is
          dropped, so the claim verifies against TODAY's value — a cherry-picked stale
          figure becomes a CONTRADICTED claim and is corrected in place, turning the
          old fail-open into a published true value.

        Ambiguity never guesses (a multi-metric or multi-date sentence leaves the claim
        as extracted — :meth:`check_number` still fails it closed on mismatch), and a
        claim without a locatable sentence is untouched. Pure and deterministic.
        """
        rebound: list[Claim] = []
        events: list[BindingEvent] = []
        for claim in claims:
            sentence = self._sentence_for(draft, claim) if claim.kind is ClaimKind.NUMBER else None
            if sentence is None:
                rebound.append(claim)
                continue
            metric_bound, metric_event = self._rebind_metric(sentence, claim)
            fully_bound, as_of_event = self._rebind_as_of(sentence, metric_bound)
            rebound.append(fully_bound)
            events.extend(e for e in (metric_event, as_of_event) if e is not None)
        return tuple(rebound), tuple(events)

    def _rebind_metric(self, sentence: str, claim: Claim) -> tuple[Claim, BindingEvent | None]:
        """Bind the claim to the sentence's single named metric (sentence wins, R10a)."""
        resolutions = self._sentence_metric_resolutions(sentence)
        if len(resolutions) != 1:
            return claim, None
        ((folded, canonical),) = resolutions.items()
        if self._claim_key(claim) == folded:
            return claim, None
        return replace(claim, metric=canonical), BindingEvent.METRIC_REBOUND

    def _rebind_as_of(self, sentence: str, claim: Claim) -> tuple[Claim, BindingEvent | None]:
        """Bind ``as_of`` to the sentence's explicit date, or drop a stale one (R10b)."""
        stated = {match.group(0) for match in _ISO_DATE_RE.finditer(sentence)}
        if len(stated) == 1:
            (date_token,) = stated
            if claim.ref == date_token:
                return claim, None
            return replace(claim, ref=date_token), BindingEvent.AS_OF_PINNED
        if not stated and self._is_stale_as_of(sentence, claim):
            return replace(claim, ref=None), BindingEvent.AS_OF_REBOUND
        return claim, None

    # --- internals (pure lexical helpers) ---

    def _claim_key(self, claim: Claim) -> str | None:
        """The claim's folded canonical metric key, or ``None`` when unresolvable."""
        if claim.metric is None:
            return None
        key = self._equivalence.canonical_key(claim.metric)
        return _fold_key(key) if key is not None else None

    def _sentence_metric_keys(self, sentence: str) -> frozenset[str]:
        """The folded canonical keys of every metric surface form the sentence names."""
        return frozenset(self._sentence_metric_resolutions(sentence))

    def _sentence_metric_resolutions(self, sentence: str) -> dict[str, str]:
        """folded key -> canonical key for every metric surface form the sentence names.

        Folding (form == tsb) dedupes labels that read one quantity, so a sentence saying
        "form" yields exactly one resolution and :meth:`_rebind_metric` can bind to its
        canonical key unambiguously.
        """
        folded_sentence = _fold_text(sentence)
        resolutions: dict[str, str] = {}
        for pattern, form in self._label_patterns:
            if pattern.search(folded_sentence):
                resolved = self._equivalence.canonical_key(form)
                if resolved is not None:
                    resolutions.setdefault(_fold_key(resolved), resolved)
        return resolutions

    def _is_stale_as_of(self, sentence: str, claim: Claim) -> bool:
        """True iff a present-tense, undated sentence cites a past ``as_of`` (R10b)."""
        as_of = _parse_iso_date(claim.ref)
        if as_of is None:
            return False
        anchor = self._reference_date
        if anchor is None or as_of >= anchor - _dt.timedelta(days=self._policy.freshness_days):
            return False
        if _DATE_MARKER_RE.search(sentence):
            return False
        folded = _fold_text(sentence)
        return any(pattern.search(folded) for pattern in self._deixis_patterns)

    def _sentence_for(self, draft: str, claim: Claim) -> str | None:
        """The enclosing sentence of the claim's own span in the ORIGINAL draft."""
        anchor = draft.find(claim.text) if claim.text else -1
        if anchor == -1:
            token_match = _THOUSANDS_AWARE_NUMBER_RE.search(claim.text)
            if token_match is None:
                return None
            found = bounded_number_pattern(token_match.group(0)).search(draft)
            if found is None:
                return None
            anchor = found.start()
        return _enclosing_sentence(draft, anchor)


def _fold_text(text: str) -> str:
    """Casefold + collapse whitespace, mirroring the equivalence-layer normalization."""
    return " ".join(text.casefold().split())


def _parse_iso_date(token: str | None) -> _dt.date | None:
    """Parse an ISO ``as_of`` token, or ``None`` (an unparseable date is the H2 rule's job)."""
    if token is None or not token.strip():
        return None
    try:
        return _dt.date.fromisoformat(token.strip())
    except ValueError:
        return None


def _enclosing_sentence(text: str, anchor: int) -> str:
    """The sentence of ``text`` containing position ``anchor`` (lexical boundaries)."""
    start = anchor
    while start > 0 and text[start - 1] not in _SENTENCE_BOUNDARY:
        start -= 1
    end = anchor
    while end < len(text) and text[end] not in _SENTENCE_BOUNDARY:
        end += 1
    return text[start:end].strip()


def policy_from_config(
    present_deixis: list[str], freshness_days: int, require_metric_label: bool
) -> BindingPolicy:
    """Build the loaded :class:`BindingPolicy` from resolved settings values (CFG-R1a).

    An EMPTY configured deixis list falls back to the conservative default lexicon — a
    blank lexicon would silently disable the temporal rule, which is the fail-open
    direction this guard exists to close.
    """
    cleaned = tuple(term for term in present_deixis if term.strip())
    policy = BindingPolicy(freshness_days=freshness_days, require_metric_label=require_metric_label)
    if cleaned:
        policy = replace(policy, present_deixis=cleaned)
    return policy


def guard_from_settings(settings: Any, equivalence: MetricEquivalence) -> BindingGuard:
    """Build the configured guard from resolved ``[agent.binding]`` settings (CFG-R1a).

    The mode string is validated through the closed :class:`BindingMode` enum so a
    misconfigured value fails the boot closed rather than silently running unguarded.
    The guard shares the caller's loaded ``equivalence`` — guard and value verifier must
    read ONE metric vocabulary (GROUND-R2).
    """
    return BindingGuard(
        equivalence,
        policy=policy_from_config(
            list(settings.agent__binding__present_deixis),
            settings.agent__binding__freshness_days,
            settings.agent__binding__require_metric_label,
        ),
        mode=BindingMode(settings.agent__binding__mode),
    )


__all__ = [
    "BindingEvent",
    "BindingGuard",
    "BindingMode",
    "BindingPolicy",
    "BindingViolation",
    "guard_from_settings",
    "policy_from_config",
]
