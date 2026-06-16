"""Unit tests for the deterministic claim-BINDING guards (issue #10, proposed GROUND-R10).

The issue's acceptance criterion 1: every demonstrated mis-binding scenario (temporal
cherry-pick, metric mis-binding, echo laundering) is a planted golden that must scrub,
while known-good consistently-bound drafts pass unchanged. Criterion 2: metamorphic
binding-flip properties — perturbing ONLY the binding of a grounded draft must flip the
verdict off ``proceed``. All tests exercise the pure :func:`ground` function with an
explicit :class:`BindingGuard` — no live service, no model call.
"""

from __future__ import annotations

import datetime as dt

import pytest

from wattwise_core.agent.contracts import (
    Claim,
    ClaimKind,
    GroundDecision,
    GroundVerdict,
)
from wattwise_core.agent.grounding import ground
from wattwise_core.agent.grounding_binding import (
    BindingEvent,
    BindingGuard,
    BindingMode,
    BindingPolicy,
    BindingViolation,
    policy_from_config,
)
from wattwise_core.agent.metric_equivalence import MetricEquivalence

pytestmark = pytest.mark.unit

_ALIASES = {"fitness": "ctl", "fatigue": "atl", "form": "tsb", "freshness": "tsb"}
_TODAY = dt.date(2026, 6, 10)


class _FakeEvidence:
    """Keyed canonical snapshots: ``(metric, as_of)`` resolves like the production path."""

    def __init__(self, cells: dict[tuple[str, str | None], float]) -> None:
        self._cells = cells

    async def metric_value(self, metric: str, as_of: str | None) -> float | None:
        return self._cells.get((metric, as_of))

    def metric_snapshot(self, metric: str, as_of: str | None) -> float | None:
        return self._cells.get((metric, as_of))

    def url_allowed(self, url: str) -> bool:
        return False


def _guard(**kwargs: object) -> BindingGuard:
    policy = BindingPolicy(
        freshness_days=int(kwargs.pop("freshness_days", 1)),
        require_metric_label=bool(kwargs.pop("require_metric_label", False)),
    )
    return BindingGuard(
        MetricEquivalence(_ALIASES),
        policy=policy,
        mode=BindingMode.ENFORCE,
        reference_date=_TODAY,
    )


def _number(text: str, metric: str | None, value: float, *, ref: str | None = None) -> Claim:
    return Claim(kind=ClaimKind.NUMBER, text=text, metric=metric, value=value, ref=ref)


# --- issue #10 scenario 1: the temporal cherry-pick (inverse-H2) -------------------------


def test_present_tense_claim_with_past_as_of_is_scrubbed() -> None:
    """A present-tense sentence may not ground at a model-chosen PAST date (R10b).

    Canonical CTL today is 55; six weeks ago it was 71. The draft asserts NOW ("is ...
    today") while the extracted claim cites the old date whose value matches — the exact
    stale-as-current cherry-pick of issue #10. The value-only gate published this with a
    citation; the binding guard must scrub it and block ``proceed``.
    """
    evidence = _FakeEvidence({("ctl", "2026-04-29"): 71.0, ("ctl", None): 55.0})
    draft = "Your CTL is 71 today, right where it needs to be."
    claims = [_number("CTL is 71", "ctl", 71.0, ref="2026-04-29")]
    result = ground(draft, claims, evidence, [], binding=_guard())
    assert result.claims[0].verdict is GroundVerdict.UNGROUNDED
    assert "71" not in result.scrubbed_text
    assert result.decision is not GroundDecision.PROCEED


def test_same_claim_without_binding_guard_was_fail_open() -> None:
    """The pre-guard behaviour pins the bug: the cherry-picked claim grounded (issue #10).

    Without the guard the model-chosen ``(metric, as_of)`` cell verifies the stale value
    and the draft PROCEEDS — the fail-open this change closes. Kept as a contrast pin so
    a regression that silently disables the guard is visible in review.
    """
    evidence = _FakeEvidence({("ctl", "2026-04-29"): 71.0, ("ctl", None): 55.0})
    draft = "Your CTL is 71 today, right where it needs to be."
    claims = [_number("CTL is 71", "ctl", 71.0, ref="2026-04-29")]
    result = ground(draft, claims, evidence, [])
    assert result.decision is GroundDecision.PROCEED


def test_explicitly_dated_sentence_still_grounds_at_its_date() -> None:
    """A sentence that STATES its date keeps the H2 dated-read path (no false positive).

    "as of 2026-04-29" makes the dated reading explicit, so the past ``as_of`` is the
    sentence's own binding — verified AT that date, exactly as before.
    """
    evidence = _FakeEvidence({("ctl", "2026-04-29"): 71.0})
    draft = "As of 2026-04-29 your CTL is 71."
    claims = [_number("CTL is 71", "ctl", 71.0, ref="2026-04-29")]
    result = ground(draft, claims, evidence, [], binding=_guard())
    assert result.decision is GroundDecision.PROCEED
    assert "71" in result.scrubbed_text


def test_freshness_window_admits_yesterdays_sample() -> None:
    """An ``as_of`` one day behind the anchor still counts as current (data lag, R10b).

    Canonical series can lag a day (an HRV sample recorded yesterday IS today's latest);
    the configured freshness window keeps that legitimate claim sayable.
    """
    evidence = _FakeEvidence({("hrv_rmssd_ms", "2026-06-09"): 65.0})
    draft = "Your HRV is 65 right now."
    claims = [_number("HRV is 65", "hrv_rmssd_ms", 65.0, ref="2026-06-09")]
    result = ground(draft, claims, evidence, [], binding=_guard())
    assert result.decision is GroundDecision.PROCEED


# --- issue #10 scenario 2: the metric mis-binding ----------------------------------------


def test_sentence_labelled_fatigue_never_verifies_against_ctl() -> None:
    """A sentence naming one canonical metric may not verify against another (R10a).

    The draft says "fatigue" (canonical ``atl``) while the claim binds ``ctl`` — the
    correlated extractor mix-up of issue #10. The wrong-cell value matched (CTL is 71),
    so the value-only gate shipped an inverted fatigue picture; the guard must treat the
    mis-binding as contradicted-class (never publishable) and re-draft.
    """
    evidence = _FakeEvidence({("ctl", None): 71.0, ("atl", None): 92.0})
    draft = "Your fatigue is sitting at 71, so push hard tomorrow."
    claims = [_number("fatigue is sitting at 71", "ctl", 71.0)]
    result = ground(draft, claims, evidence, [], binding=_guard())
    assert result.claims[0].verdict is GroundVerdict.CONTRADICTED
    assert "71" not in result.scrubbed_text
    assert result.decision is GroundDecision.REGENERATE


def test_consistently_bound_alias_label_passes_unchanged() -> None:
    """A known-good draft passes the guard untouched (GROUND-R8: no over-scrub).

    "fitness" resolves to the SAME canonical key the claim verifies against, so the guard
    is silent and the value gate publishes the canonical figure exactly as before.
    """
    evidence = _FakeEvidence({("ctl", None): 84.0})
    draft = "Your fitness sits at 84 today."
    claims = [_number("fitness sits at 84", "ctl", 84.0)]
    result = ground(draft, claims, evidence, [], binding=_guard())
    assert result.decision is GroundDecision.PROCEED
    assert result.scrubbed_text == draft


def test_form_label_binding_tsb_is_one_quantity_not_a_mismatch() -> None:
    """``form`` and ``tsb`` read the same canonical field, so the pair never mismatches.

    Guards the false-positive class: the athlete-facing FORM alias resolves through the
    enum while the claim cites ``tsb`` — the binding comparison must fold them together
    (mirroring the canonical evidence's own FORM->tsb read).
    """
    evidence = _FakeEvidence({("tsb", None): 5.0})
    draft = "Your form is 5 today."
    claims = [_number("form is 5", "tsb", 5.0)]
    result = ground(draft, claims, evidence, [], binding=_guard())
    assert result.decision is GroundDecision.PROCEED


def test_multi_metric_sentence_allows_each_consistent_claim() -> None:
    """A sentence naming several metrics admits a claim bound to ANY of them (R10a).

    Mismatch fires only when the claim's key is named by NONE of the sentence's labels —
    a two-metric summary sentence stays publishable.
    """
    evidence = _FakeEvidence({("ctl", None): 84.0, ("atl", None): 70.0})
    draft = "Your fitness is 84 and your fatigue is 70."
    claims = [
        _number("fitness is 84", "ctl", 84.0),
        _number("fatigue is 70", "atl", 70.0),
    ]
    result = ground(draft, claims, evidence, [], binding=_guard())
    assert result.decision is GroundDecision.PROCEED


def test_unlabeled_sentence_passes_by_default_and_scrubs_under_strict_rule() -> None:
    """The strict no-label rule is config-gated (R10a strict; off by default).

    Default policy: a label-free sentence ("you're at 84 now") carries no binding signal
    and stays publishable on the value gate. With ``require_metric_label`` on, the same
    claim fails closed — the operator's strictness knob.
    """
    evidence = _FakeEvidence({("ctl", None): 84.0})
    draft = "You're at 84 now."
    claims = [_number("at 84", "ctl", 84.0)]
    relaxed = ground(draft, claims, evidence, [], binding=_guard())
    assert relaxed.decision is GroundDecision.PROCEED
    strict = ground(draft, claims, evidence, [], binding=_guard(require_metric_label=True))
    assert strict.decision is not GroundDecision.PROCEED
    assert "84" not in strict.scrubbed_text


# --- issue #10 scenario 3: echo laundering ------------------------------------------------


def test_metric_shaped_sentence_cannot_pass_as_request_echo() -> None:
    """A sentence naming a canonical metric never grounds as a user-request echo (R10d).

    The athlete asked about "100" (a TSS target), the draft answers "your TSB is 100"
    with no resolvable metric binding — the value-only gate grounded it as the user's own
    number. The guard blocks the echo pass for metric-shaped sentences; the claim scrubs.
    """
    evidence = _FakeEvidence({})
    draft = "Sure thing: your TSB is 100."
    claims = [_number("TSB is 100", None, 100.0)]
    result = ground(
        draft, claims, evidence, [], request_numbers=frozenset({"100"}), binding=_guard()
    )
    assert result.claims[0].verdict is GroundVerdict.UNGROUNDED
    assert "100" not in result.scrubbed_text
    assert result.decision is not GroundDecision.PROCEED


def test_genuine_constraint_echo_stays_sayable() -> None:
    """A metric-free restatement of the athlete's own constraint still echoes (R10d scope).

    "7 hours a week" names no canonical metric, so the echo path keeps the user's own
    constraint sayable — the guard narrows the pass, it does not remove it.
    """
    evidence = _FakeEvidence({})
    draft = "Planning around 7 hours a week."
    claims = [_number("7 hours", None, 7.0)]
    result = ground(draft, claims, evidence, [], request_numbers=frozenset({"7"}), binding=_guard())
    assert result.decision is GroundDecision.PROCEED
    assert "7 hours" in result.scrubbed_text


# --- metamorphic binding-flip properties (issue #10 acceptance criterion 2) ----------------


@pytest.mark.parametrize(
    ("draft", "metric", "ref"),
    [
        # base sentence label flipped to a different metric's word
        ("Your fatigue is 84 today.", "ctl", None),
        # as_of shifted into the stale past under present tense
        ("Your fitness is 84 today.", "ctl", "2026-01-01"),
    ],
)
def test_binding_perturbation_flips_a_grounded_draft_off_proceed(
    draft: str, metric: str, ref: str | None
) -> None:
    """Perturbing ONLY the binding of a grounded draft flips the verdict (metamorphic).

    The unperturbed base ("Your fitness is 84 today." bound to latest ctl=84) proceeds;
    each perturbation keeps the VALUE verifiable somewhere but breaks the binding, so the
    guard must take the run off ``proceed`` — binding sensitivity as an invariant.
    """
    evidence = _FakeEvidence({("ctl", None): 84.0, ("ctl", "2026-01-01"): 84.0})
    base = ground(
        "Your fitness is 84 today.",
        [_number("fitness is 84", "ctl", 84.0)],
        evidence,
        [],
        binding=_guard(),
    )
    assert base.decision is GroundDecision.PROCEED
    perturbed = ground(
        draft, [_number("is 84", metric, 84.0, ref=ref)], evidence, [], binding=_guard()
    )
    assert perturbed.decision is not GroundDecision.PROCEED


# --- the authority inversion: rebinding (the sentence selects the cell) -------------------


def test_rebind_overrides_the_extracted_metric_with_the_sentences_label() -> None:
    """The sentence's single named metric WINS over the extracted binding (rebind, R10a).

    "fatigue" is the sentence's only metric label, so the claim is re-bound from the
    extractor's ``ctl`` to ``atl`` BEFORE any value is fetched — the model's extraction
    keeps no routing power over which cell verifies the figure.
    """
    guard = _guard()
    draft = "Your fatigue is sitting at 71."
    (claim,), events = guard.rebind(draft, [_number("fatigue is sitting at 71", "ctl", 71.0)])
    assert claim.metric == "atl"
    assert events == (BindingEvent.METRIC_REBOUND,)


def test_rebound_misattribution_is_corrected_in_place_not_just_scrubbed() -> None:
    """End to end, a mis-bound figure becomes the TRUE value of the sentence's own cell.

    The rebind points verification at ``atl`` (the sentence's metric); the stated 71 is
    out of tolerance against the canonical 92, so the GROUND-R7 correction machinery
    substitutes the truth in place and the run re-drafts — the athlete ends up with the
    real fatigue figure instead of a hole (usefulness preserved, hallucination gone).
    """
    guard = _guard()
    draft = "Your fatigue is sitting at 71, so push hard tomorrow."
    rebound, _ = guard.rebind(draft, [_number("fatigue is sitting at 71", "ctl", 71.0)])
    evidence = _FakeEvidence({("ctl", None): 71.0, ("atl", None): 92.0})
    result = ground(draft, list(rebound), evidence, [], binding=guard)
    assert result.claims[0].verdict is GroundVerdict.CONTRADICTED
    assert "92" in result.scrubbed_text
    assert "71" not in result.scrubbed_text
    assert result.decision is GroundDecision.REGENERATE


def test_rebind_drops_a_cherry_picked_stale_date_so_today_wins() -> None:
    """A present-tense sentence sheds a stale extracted date and verifies against NOW.

    The cherry-picked April cell agreed with the draft's 71; after the rebind the claim
    reads the LATEST canonical value (55), contradicts, and is corrected in place — the
    stale-as-current fail-open becomes a published true figure.
    """
    guard = _guard()
    draft = "Your CTL is 71 today, right where it needs to be."
    rebound, events = guard.rebind(draft, [_number("CTL is 71", "ctl", 71.0, ref="2026-04-29")])
    assert rebound[0].ref is None
    assert events == (BindingEvent.AS_OF_REBOUND,)
    evidence = _FakeEvidence({("ctl", "2026-04-29"): 71.0, ("ctl", None): 55.0})
    result = ground(draft, list(rebound), evidence, [], binding=guard)
    assert result.claims[0].verdict is GroundVerdict.CONTRADICTED
    assert "55" in result.scrubbed_text
    assert "71" not in result.scrubbed_text


def test_rebind_pins_as_of_to_the_sentences_explicit_date() -> None:
    """A sentence stating one ISO date pins ``as_of`` to it (the missing half of H2).

    With an ABSENT extracted date, a dated sentence used to verify against the latest
    day; pinning makes the sentence's own date the verification target.
    """
    guard = _guard()
    draft = "Back on 2026-04-29 your CTL was 71."
    rebound, events = guard.rebind(draft, [_number("CTL was 71", "ctl", 71.0)])
    assert rebound[0].ref == "2026-04-29"
    assert events == (BindingEvent.AS_OF_PINNED,)


def test_rebind_never_guesses_on_an_ambiguous_sentence() -> None:
    """A multi-metric sentence rebinds nothing; the residual check still fails it closed.

    Two labels mean the lexical layer cannot know which cell the figure belongs to —
    rebinding abstains (no swap-by-guess) and the claim, bound to NEITHER label, is
    flagged as a mismatch for the enforce path.
    """
    guard = _guard()
    draft = "Your fitness and fatigue tell one story: 71 says hold back."
    claims = [_number("71 says hold back", "tsb", 71.0)]
    rebound, events = guard.rebind(draft, claims)
    assert rebound[0].metric == "tsb"
    assert events == ()
    assert guard.check_number(draft, rebound[0]) is BindingViolation.METRIC_MISMATCH


def test_rebind_leaves_consistent_claims_untouched() -> None:
    """A consistently-bound draft rebinds nothing — known-good behaviour is preserved."""
    guard = _guard()
    draft = "Your fitness sits at 84 today."
    claims = [_number("fitness sits at 84", "ctl", 84.0)]
    rebound, events = guard.rebind(draft, claims)
    assert rebound == tuple(claims)
    assert events == ()


# --- guard plumbing: shadow assessment, config policy, sentence location ------------------


def test_assess_reports_violations_without_scrubbing() -> None:
    """SHADOW-mode reading: ``assess`` names the violations, the text is untouched.

    The grounder seam uses this to record would-be enforcement on the observability
    surface before turning the mode to enforce (issue #10 Phase 4 rollout).
    """
    guard = _guard()
    draft = "Your fatigue is sitting at 71."
    claims = (_number("fatigue is sitting at 71", "ctl", 71.0),)
    assert guard.assess(draft, claims) == (BindingViolation.METRIC_MISMATCH,)


def test_policy_from_config_keeps_default_lexicon_when_blank() -> None:
    """An empty configured deixis list falls back to the default lexicon (fail-closed).

    A blank lexicon would silently disable the temporal rule — the fail-open direction —
    so the loader restores the conservative default instead.
    """
    policy = policy_from_config([], 1, False)
    assert "today" in policy.present_deixis
    explicit = policy_from_config(["today"], 2, True)
    assert explicit.present_deixis == ("today",)
    assert explicit.freshness_days == 2
    assert explicit.require_metric_label is True


def test_claim_text_absent_from_draft_yields_no_binding_signal() -> None:
    """An unanchorable claim is the positional machinery's job, not the guard's.

    When neither the claim text nor its numeric token occurs in the draft the guard
    abstains (no violation) — the rewrite/sweep layers already fail closed on whatever
    DID reach the text.
    """
    guard = _guard()
    claim = _number("CTL is 71", "ctl", 71.0, ref="2026-01-01")
    assert guard.check_number("A draft about something else entirely.", claim) is None


def test_anchored_guard_pins_the_reference_date_once() -> None:
    """``anchored`` pins an un-anchored guard's clock; an anchored guard is unchanged.

    The grounder resolves the anchor once per run so every temporal check inside one
    ``ground`` call shares one reference date (deterministic, GRAPH-R4).
    """
    floating = BindingGuard(MetricEquivalence(_ALIASES), mode=BindingMode.SHADOW)
    pinned = floating.anchored(_TODAY)
    claim = _number("CTL is 71", "ctl", 71.0, ref="2026-04-29")
    assert pinned.check_number("Your CTL is 71 today.", claim) is not None
    assert pinned.mode is BindingMode.SHADOW
    assert (
        _guard().anchored(dt.date(2030, 1, 1)).check_number("Your CTL is 71 today.", claim)
        is BindingViolation.STALE_AS_OF
    )


# --- issue #25: prescriptions stay safe under the production binding-ENFORCE path -----------


def _prescription(text: str, metric: str | None, value: float) -> Claim:
    """A NUMBER claim marked as a future TARGET (issue #25)."""
    return Claim(kind=ClaimKind.NUMBER, text=text, metric=metric, value=value, prescriptive=True)


def test_prescription_under_binding_enforce_is_never_rewritten_upward() -> None:
    """A prescribed recovery week is never loaded UP to 7xCTL on the DEFAULT (ENFORCE) path.

    The issue #25 unit goldens run with ``binding=None``, but production wires the binding guard
    in ENFORCE mode. A prescription must still bypass canonical correction with the guard live:
    scrubbed, never rewritten to the maintenance value, decision off ``proceed``.
    """
    evidence = _FakeEvidence({("weekly_load_target", None): 420.0})
    draft = "Recovery week: aim for 320 TSS."
    claims = [_prescription("320", "weekly_load_target", 320.0)]
    result = ground(draft, claims, evidence, [], binding=_guard())
    assert result.claims[0].verdict is not GroundVerdict.GROUNDED
    assert "420" not in result.scrubbed_text  # never loaded up to maintenance
    assert "320" not in result.scrubbed_text  # unverifiable target scrubbed, fail-closed
    assert result.decision is not GroundDecision.PROCEED
