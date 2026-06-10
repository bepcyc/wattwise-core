"""Unit tests for deterministic fail-closed grounding (doc 50 GROUND-R1..R9, GROUND-R8).

GROUND-R8 mandates golden + property coverage proving: a planted hallucinated
number/name/URL is scrubbed (100%), a known-good draft passes unchanged (100%), a
``contradicted`` number is never published, and the aggregate ``abstain``s when nothing
grounds. These tests exercise the pure :func:`ground` function against a fake canonical
evidence object — no live service, no model call.
"""

from __future__ import annotations

from wattwise_core.agent.contracts import (
    Claim,
    ClaimKind,
    GroundDecision,
    GroundVerdict,
)
from wattwise_core.agent.grounding import ground

# --- fake canonical evidence (GROUND-R2/R4/R7) ---


class _FakeEvidence:
    """A canonical evidence object for tests (GroundingEvidence + NameLibrary seams).

    Implements the async ``metric_value`` / sync ``url_allowed`` of the
    :class:`~wattwise_core.agent.contracts.GroundingEvidence` contract, PLUS the optional
    synchronous ``metric_snapshot`` the grounder reads (resolved-ahead path) and the
    optional ``canonical_name`` of the NameLibrary protocol. Anything not pre-loaded is
    unavailable, so the grounder fails closed.
    """

    def __init__(
        self,
        *,
        metrics: dict[str, float] | None = None,
        names: dict[str, str] | None = None,
        allowed_urls: frozenset[str] | None = None,
    ) -> None:
        self._metrics = metrics or {}
        self._names = names or {}
        self._allowed_urls = allowed_urls or frozenset()

    async def metric_value(self, metric: str, as_of: str | None) -> float | None:
        return self._metrics.get(metric)

    def metric_snapshot(self, metric: str, as_of: str | None) -> float | None:
        return self._metrics.get(metric)

    def url_allowed(self, url: str) -> bool:
        return url in self._allowed_urls

    def canonical_name(self, name: str) -> str | None:
        return self._names.get(name)


def _number(text: str, metric: str, value: float, *, ref: str | None = None) -> Claim:
    return Claim(kind=ClaimKind.NUMBER, text=text, metric=metric, value=value, ref=ref)


def _name(text: str, *, ref: str | None = None) -> Claim:
    return Claim(kind=ClaimKind.NAME, text=text, ref=ref)


def _url(text: str) -> Claim:
    return Claim(kind=ClaimKind.URL, text=text, ref=text)


def _statement(text: str, *, prescriptive: bool = False) -> Claim:
    return Claim(kind=ClaimKind.STATEMENT, text=text, prescriptive=prescriptive)


# --- known-good draft unchanged (GROUND-R8) ---


def test_known_good_draft_passes_unchanged() -> None:
    """A draft whose every claim grounds is published verbatim, decision proceed (GROUND-R8)."""
    evidence = _FakeEvidence(
        metrics={"ctl": 84.0},
        names={"Sweet Spot 3x12": "wkt-1"},
        allowed_urls=frozenset({"https://wattwise.app/activity/42"}),
    )
    draft = (
        "Your fitness sits at 84 today. Try Sweet Spot 3x12 next. "
        "See https://wattwise.app/activity/42"
    )
    claims = [
        _number("84", "ctl", 84.0),
        _name("Sweet Spot 3x12"),
        _url("https://wattwise.app/activity/42"),
    ]
    result = ground(draft, claims, evidence, allow_urls=[])
    assert result.decision is GroundDecision.PROCEED
    assert result.scrubbed_text == draft
    assert all(c.verdict is GroundVerdict.GROUNDED for c in result.claims)


def test_grounded_number_carries_metric_citation() -> None:
    """A surviving number cites its canonical metric + verbatim value (GROUND-R5/R7)."""
    evidence = _FakeEvidence(metrics={"ctl": 84.0})
    result = ground("CTL is 84", [_number("84", "ctl", 84.0, ref="2026-06-06")], evidence, [])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.GROUNDED
    assert claim.citation == {
        "kind": "metric",
        "record_id": "ctl@2026-06-06",
        "metric": "ctl",
        "value": 84.0,
        "as_of": "2026-06-06",
    }


def test_number_within_tolerance_grounds() -> None:
    """A claimed number within relative tolerance of canonical is grounded (GROUND-R7)."""
    evidence = _FakeEvidence(metrics={"tss": 100.0})
    result = ground("TSS 100.05", [_number("100.05", "tss", 100.05)], evidence, [])
    assert result.decision is GroundDecision.PROCEED
    assert result.claims[0].verdict is GroundVerdict.GROUNDED


# --- ADVERSARIAL FABRICATION TESTS (must FAIL on the pre-fix behaviour) --------------------


def test_within_tolerance_publishes_canonical_value_not_the_model_number() -> None:
    """H1 fabrication: a within-band but WRONG number publishes the CANONICAL value, not model's.

    Canonical ctl=100; the model says 102 (within the 2% band). The OLD behaviour marked it GROUNDED
    and shipped "102" — a fabrication the athlete sees while canonical is 100 (GROUND-R7 violation).
    The published number MUST be the canonical value; "102" must NEVER reach the body.
    """
    evidence = _FakeEvidence(metrics={"ctl": 100.0})
    result = ground("Your CTL is 102 today.", [_number("102", "ctl", 102.0)], evidence, [])
    # Recognized as a (rounded) restatement -> grounded, but the PUBLISHED number is canonical.
    assert result.claims[0].verdict is GroundVerdict.GROUNDED
    assert "102" not in result.scrubbed_text, "the model's within-band approximation must NOT ship"
    assert "100" in result.scrubbed_text, "the canonical value must be published (GROUND-R7)"
    assert result.decision is GroundDecision.PROCEED


def test_unextracted_number_is_swept_and_does_not_proceed() -> None:
    """H4 fabrication: a number the extractor MISSED ("TSB is 999") is swept; run does not proceed.

    The draft states two numbers but the claim extractor returns ONLY the CTL claim — the OLD
    behaviour left the fabricated 999 in the body because numeric fail-closure depended on the
    extractor. The deterministic numeric-coverage sweep removes the uncovered 999 and the decision
    is forced off ``proceed`` (a grounded survivor -> ``regenerate``), so the 999 cannot ship.
    """
    evidence = _FakeEvidence(metrics={"ctl": 60.0})
    # Only the CTL number is extracted; the "TSB is 999" span is NOT a claim.
    result = ground("Your CTL is 60 and TSB is 999.", [_number("60", "ctl", 60.0)], evidence, [])
    assert "999" not in result.scrubbed_text, "the unextracted fabricated number must be swept (H4)"
    assert "60" in result.scrubbed_text, "the grounded canonical value survives the sweep"
    assert result.decision is not GroundDecision.PROCEED, "an uncovered number must block proceed"


def test_shared_token_claims_each_rewrite_their_own_span() -> None:
    """Two claims sharing a numeric token each rewrite their OWN occurrence (GROUND-R7).

    Canonical ctl=100, tss=98; the draft says "100" for both. The OLD rewrite anchored on the
    FIRST substring occurrence of the token in the whole draft, so the TSS correction landed on
    the CTL's span — publishing "CTL is 98 and TSS was 100" (both wrong, swapped) under a
    ``proceed`` decision. Span-anchored rewriting must publish each claim's canonical value in
    its own place.
    """
    evidence = _FakeEvidence(metrics={"ctl": 100.0, "tss": 98.0})
    draft = "Your CTL is 100 and your weekly TSS was 100."
    claims = [_number("CTL is 100", "ctl", 100.0), _number("TSS was 100", "tss", 100.0)]
    result = ground(draft, claims, evidence, [])
    assert result.scrubbed_text == "Your CTL is 100 and your weekly TSS was 98."
    assert result.decision is GroundDecision.PROCEED
    assert all(c.verdict is GroundVerdict.GROUNDED for c in result.claims)


def test_contradicted_shared_token_corrects_its_own_span_only() -> None:
    """A contradicted claim sharing a token with a grounded one corrects ITS span (GROUND-R9).

    Canonical ctl=100, atl=80; the draft claims "100" for both. The OLD rewrite corrupted the
    grounded CTL span to "80" and left the CONTRADICTED model figure "100" in the ATL span —
    publishing the one value GROUND-R9 forbids. The correction must land on the ATL occurrence.
    """
    evidence = _FakeEvidence(metrics={"ctl": 100.0, "atl": 80.0})
    draft = "Your CTL is 100 and your ATL is 100."
    claims = [_number("CTL is 100", "ctl", 100.0), _number("ATL is 100", "atl", 100.0)]
    result = ground(draft, claims, evidence, [])
    assert result.scrubbed_text == "Your CTL is 100 and your ATL is 80."
    assert result.decision is GroundDecision.REGENERATE  # contradicted never proceeds


def test_token_never_rewrites_inside_a_longer_number() -> None:
    """A claim token is matched as a standalone number, never inside a longer one (GROUND-R7).

    The claim's "102" must not anchor inside the unrelated "1029": the claim rewrites its own
    span, and the uncovered "1029" is swept as an unverified figure (H4) — blocking proceed.
    """
    evidence = _FakeEvidence(metrics={"ctl": 102.0})
    draft = "Over 1029 TSS this block, your CTL is 102."
    result = ground(draft, [_number("CTL is 102", "ctl", 102.0)], evidence, [])
    assert "1029" not in result.scrubbed_text
    assert "your CTL is 102" in result.scrubbed_text
    assert result.decision is not GroundDecision.PROCEED


def test_repeated_grounded_number_is_not_string_covered() -> None:
    """A stray duplicate of a grounded VALUE is still an uncovered span (positional coverage).

    The string-set coverage let ANY token equal to a published value ship unverified ("TSS 100"
    riding the grounded CTL's "100"). Coverage is positional now: only the span the grounder
    rewrote is covered, the duplicate is swept, and the run re-drafts (fail-closed, H4).
    """
    evidence = _FakeEvidence(metrics={"ctl": 100.0})
    draft = "Your CTL is 100. Yes - 100 TSS planned tomorrow."
    result = ground(draft, [_number("CTL is 100", "ctl", 100.0)], evidence, [])
    assert result.scrubbed_text.count("100") == 1
    assert result.decision is not GroundDecision.PROCEED


def test_numeric_sweep_keeps_dates_units_and_structural_tokens() -> None:
    """H4: the numeric sweep keeps safe NON-metric tokens (dates, units, ordinals), only metrics go.

    The sweep must NOT corrupt legitimate grounded prose: an ISO date, a "Day N" ordinal, an "NxM"
    interval, and a unit-bearing duration/percentage are structurally safe and survive, while only
    the unverified metric-magnitude figure would be removed. Guards the over-scrub regressions
    (``2026-06-08`` -> ``2026-06`` and ``20%`` -> ``%``) the per-token window caused.
    """
    evidence = _FakeEvidence(metrics={"ctl": 60.0})
    draft = "On 2026-06-08, Day 1: 3x12 efforts, 45m easy, improved 20%. CTL is 60."
    result = ground(draft, [_number("60", "ctl", 60.0)], evidence, [])
    for token in ("2026-06-08", "Day 1", "3x12", "45m", "20%", "60"):
        assert token in result.scrubbed_text, f"safe/grounded token {token!r} must survive"


def test_numeric_sweep_keeps_grounded_value_inside_a_url() -> None:
    """H4: a digit inside a SURVIVING first-party URL is never reached by the numeric sweep."""
    url = "https://wattwise.app/activity/42"
    evidence = _FakeEvidence(metrics={"ctl": 60.0}, allowed_urls=frozenset({url}))
    draft = f"Your CTL is 60. See {url}"
    result = ground(draft, [_number("60", "ctl", 60.0), _url(url)], evidence, [url])
    assert url in result.scrubbed_text, "URL path digits must not be scrubbed by the numeric sweep"
    assert result.decision is GroundDecision.PROCEED


# --- planted hallucinated NUMBER scrubbed / contradicted dropped (GROUND-R3/R7/R9) ---


def test_planted_hallucinated_number_is_contradicted_and_replaced() -> None:
    """A wrong number is contradicted, replaced by canonical, never published (GROUND-R7/R9)."""
    evidence = _FakeEvidence(metrics={"ctl": 84.0})
    result = ground("Your form is 99 today", [_number("99", "ctl", 99.0)], evidence, [])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.CONTRADICTED
    # contradicted number NEVER published as stated; replaced by the canonical value.
    assert "99" not in result.scrubbed_text
    assert "84" in result.scrubbed_text
    # GROUND-R9 (corrected): a contradicted number replaced IN PLACE by the canonical
    # value is a bounded re-draft (regenerate), NOT a coverage re-plan — the corrected
    # value already exists, so no different evidence needs fetching.
    assert result.decision is GroundDecision.REGENERATE


def test_unavailable_metric_number_is_scrubbed_not_placeholder() -> None:
    """A number whose canonical computation is unavailable is removed, never zeroed (GROUND-R7).

    The unavailable number is scrubbed (never a placeholder/zero). The aggregate is the recovery
    ``replan`` (GROUND-R6): a missing metric is re-gatherable, so the run attempts retrieval before
    the ``reflection_count`` bound forces an abstain (REFLECT-R4) — see
    :func:`test_unavailable_metric_alone_replans_then_router_bounds_to_abstain`.
    """
    evidence = _FakeEvidence(metrics={})  # hrv unavailable
    result = ground("Your HRV is 65 today", [_number("65", "hrv", 65.0)], evidence, [])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.UNGROUNDED
    assert "65" not in result.scrubbed_text
    assert "0" not in result.scrubbed_text.replace("today", "")  # no placeholder/zero
    assert result.decision is GroundDecision.REPLAN


def test_number_claim_missing_metric_fails_closed() -> None:
    """A number claim with no metric/value cannot be checked, so it fails closed."""
    evidence = _FakeEvidence(metrics={"ctl": 84.0})
    bad = Claim(kind=ClaimKind.NUMBER, text="42", metric=None, value=None)
    result = ground("Mystery 42", [bad], evidence, [])
    assert result.claims[0].verdict is GroundVerdict.UNGROUNDED
    assert "42" not in result.scrubbed_text


# --- planted hallucinated NAME scrubbed (GROUND-R2/R3) ---


def test_planted_hallucinated_name_is_scrubbed() -> None:
    """A workout name with no canonical library match is removed (GROUND-R3)."""
    evidence = _FakeEvidence(names={"Sweet Spot 3x12": "wkt-1"})
    result = ground(
        "Do the Galaxy Brain Destroyer tomorrow", [_name("Galaxy Brain Destroyer")], evidence, []
    )
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.UNGROUNDED
    assert "Galaxy Brain Destroyer" not in result.scrubbed_text
    assert result.decision is GroundDecision.ABSTAIN


def test_grounded_name_carries_canonical_citation() -> None:
    """A resolved name cites its canonical workout id (GROUND-R5)."""
    evidence = _FakeEvidence(names={"Sweet Spot 3x12": "wkt-1"})
    result = ground("Try Sweet Spot 3x12", [_name("Sweet Spot 3x12")], evidence, [])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.GROUNDED
    assert claim.citation == {"kind": "name", "record": "workout", "canonical_id": "wkt-1"}


def test_name_without_library_fails_closed() -> None:
    """Evidence that cannot resolve names scrubs every name claim (fail-closed GROUND-R3)."""

    class _NumbersOnly:
        async def metric_value(self, metric: str, as_of: str | None) -> float | None:
            return None

        def url_allowed(self, url: str) -> bool:
            return False

    result = ground("Try Sweet Spot 3x12", [_name("Sweet Spot 3x12")], _NumbersOnly(), [])
    assert result.claims[0].verdict is GroundVerdict.UNGROUNDED
    assert "Sweet Spot 3x12" not in result.scrubbed_text


# --- planted hallucinated URL scrubbed (GROUND-R4) ---


def test_model_invented_url_is_scrubbed_unconditionally() -> None:
    """A URL not on the allow-list nor accepted by evidence is removed (GROUND-R4)."""
    evidence = _FakeEvidence(allowed_urls=frozenset({"https://wattwise.app/ok"}))
    result = ground(
        "Read more at http://evil.example/phish", [_url("http://evil.example/phish")], evidence, []
    )
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.UNGROUNDED
    assert "evil.example" not in result.scrubbed_text


def test_allow_listed_url_passes() -> None:
    """A URL on the caller's allow-list grounds and is cited (GROUND-R4/R5)."""
    evidence = _FakeEvidence()
    url = "https://wattwise.app/activity/7"
    result = ground(f"See {url}", [_url(url)], evidence, allow_urls=[url])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.GROUNDED
    assert claim.citation == {"kind": "url", "canonical_id": url}


def test_url_accepted_by_evidence_record_passes() -> None:
    """A URL on a matched canonical record (evidence.url_allowed) grounds (GROUND-R4)."""
    evidence = _FakeEvidence(allowed_urls=frozenset({"https://wattwise.app/r/1"}))
    url = "https://wattwise.app/r/1"
    result = ground(f"Open {url}", [_url(url)], evidence, [])
    assert result.claims[0].verdict is GroundVerdict.GROUNDED


def test_url_allow_list_is_normalized() -> None:
    """Trailing-slash / case differences still match the allow-list (GROUND-R4 normalization)."""
    evidence = _FakeEvidence()
    result = ground(
        "Go to https://WattWise.app/Activity/9/",
        [_url("https://WattWise.app/Activity/9/")],
        evidence,
        allow_urls=["https://wattwise.app/Activity/9"],
    )
    assert result.claims[0].verdict is GroundVerdict.GROUNDED


# --- complementary statements (GROUND-R9 fail-closed default) ---


def test_non_prescriptive_statement_is_complementary_and_published() -> None:
    """A non-prescriptive statement with no checkable token publishes complementary (GROUND-R9)."""
    evidence = _FakeEvidence()
    draft = "You're trending in the right direction."
    result = ground(draft, [_statement(draft)], evidence, [])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.COMPLEMENTARY
    assert result.scrubbed_text == draft
    assert result.decision is GroundDecision.PROCEED


def test_statement_smuggling_a_number_is_scrubbed_not_complementary() -> None:
    """A non-prescriptive STATEMENT carrying a numeric literal is ungrounded (GROUND-R9).

    Closes the mislabel path: code never lets a factual number ship on a statement's
    free pass just because the model tagged the span ``STATEMENT`` instead of ``NUMBER``.
    """
    evidence = _FakeEvidence()
    draft = "Your CTL is 999 and climbing."
    result = ground(draft, [_statement(draft)], evidence, [])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.UNGROUNDED
    assert result.decision is GroundDecision.ABSTAIN
    assert "999" not in result.scrubbed_text


def test_unextracted_url_is_scrubbed_by_the_sweep() -> None:
    """A URL the model never surfaced as a claim is still scrubbed (GROUND-R4 unconditional)."""
    evidence = _FakeEvidence()
    draft = "Nice, steady week. More at http://evil.example/x"
    # The only extracted claim is the non-factual lead; the URL is NOT extracted.
    result = ground(draft, [_statement("Nice, steady week.")], evidence, [])
    assert result.decision is GroundDecision.PROCEED  # the statement still publishes
    assert "evil.example" not in result.scrubbed_text  # the invented URL is swept out


def test_allowlisted_url_survives_the_sweep() -> None:
    """A first-party allow-listed URL is preserved by the deterministic URL sweep (GROUND-R4)."""
    url = "https://wattwise.app/guide"
    evidence = _FakeEvidence(allowed_urls=frozenset({url}))
    draft = f"Here's a tip. See {url}"
    result = ground(draft, [_statement("Here's a tip.")], evidence, [url])
    assert url in result.scrubbed_text


def test_prescriptive_statement_without_grounding_is_scrubbed() -> None:
    """A prescriptive statement is treated as ungrounded and scrubbed (GROUND-R9 fail-closed)."""
    evidence = _FakeEvidence()
    draft = "Ride 5 hours at threshold tomorrow."
    result = ground(draft, [_statement(draft, prescriptive=True)], evidence, [])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.UNGROUNDED
    assert result.scrubbed_text == ""
    assert result.decision is GroundDecision.ABSTAIN


# --- aggregate decision matrix (GROUND-R9) ---


def test_replan_when_a_missing_metric_is_among_the_scrubbed_claims() -> None:
    """GROUND-R6: nothing publishable, but a MISSING metric is present -> replan, not abstain.

    The draft mixes a fabricated workout NAME and URL with a NUMBER about a real canonical metric
    (``ctl``) that was NOT retrieved. Everything is scrubbed and nothing publishable survives, but
    the missing metric is RECOVERABLE by re-gathering (GROUND-R9 ``replan``), so the aggregate is
    ``replan`` — the engine attempts the ``ground -> reflect -> plan_retrieval`` recovery (bounded
    by ``reflection_count``, REFLECT-R4) before degrading. The fabrications are still scrubbed.
    """
    evidence = _FakeEvidence()  # ctl unavailable -> a re-gatherable metric gap
    claims = [_number("99", "ctl", 99.0), _name("Fake Workout"), _url("http://x.invalid")]
    result = ground("99 Fake Workout http://x.invalid", claims, evidence, [])
    assert all(c.verdict is GroundVerdict.UNGROUNDED for c in result.claims)
    assert result.decision is GroundDecision.REPLAN


def test_abstain_on_empty_claims() -> None:
    """No claims at all means nothing publishable -> abstain (fail-closed default).

    With ZERO claims there is no missing-metric signal to re-gather, so a ``replan`` would loop
    pointlessly; the aggregate is a truthful ``abstain``.
    """
    result = ground("", [], _FakeEvidence(), [])
    assert result.decision is GroundDecision.ABSTAIN
    assert result.claims == ()


# --- GROUND-R6 recovery-replan boundary (the dead-edge fix) -----------------------------


def test_replan_when_missing_metric_evidence_could_be_regathered() -> None:
    """GROUND-R6: scrub left NO answer but the gap is a MISSING metric -> replan, not abstain.

    The athlete asked about a metric (``ctl``) whose canonical value was NOT in the retrieved
    evidence, so the only claim is an ungrounded NUMBER and nothing publishable survives. The
    deliverable can no longer answer (GROUND-R6), but the gap is RECOVERABLE by re-gathering the
    missing metric (GROUND-R9 ``replan`` = "return to gather/plan for missing evidence"), so the
    aggregate decision MUST be ``replan`` — routing ``ground -> reflect -> plan_retrieval`` within
    the ``reflection_count`` bound (REFLECT-R4) rather than abstaining immediately. The bound makes
    this safe: if re-gathering still cannot ground, the run degrades to a truthful limitation at
    the bound (the fail-closed floor is the router's, not an early abstain). MUTATION-GUARD: if the
    aggregator drops REPLAN (the dead-edge regression), this asserts != ABSTAIN and fails.
    """
    evidence = _FakeEvidence()  # ctl unavailable -> the metric was not retrieved
    result = ground("Your fitness is at 84 today.", [_number("84", "ctl", 84.0)], evidence, [])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.UNGROUNDED
    assert "84" not in result.scrubbed_text  # the unverified number is still scrubbed
    assert result.decision is GroundDecision.REPLAN
    assert result.decision is not GroundDecision.ABSTAIN


def test_missing_metric_does_not_replan_when_a_grounded_survivor_exists() -> None:
    """A missing metric BESIDE a grounded survivor regenerates, not replans (GROUND-R9 boundary).

    REPLAN is reserved for the case where scrubbing left NOTHING publishable (the deliverable can
    no longer answer, GROUND-R6). When a grounded survivor remains the answer is still producible
    with the offending span removed, so the correct recovery is a bounded re-DRAFT (``regenerate``),
    not a coverage re-plan — even though one scrubbed claim was a missing metric.
    """
    evidence = _FakeEvidence(metrics={"ctl": 84.0})  # ctl available, atl missing
    claims = [_number("84", "ctl", 84.0), _number("70", "atl", 70.0)]
    result = ground("CTL 84 and ATL 70", claims, evidence, [])
    assert any(c.verdict is GroundVerdict.GROUNDED for c in result.claims)
    assert any(c.verdict is GroundVerdict.UNGROUNDED for c in result.claims)
    assert result.decision is GroundDecision.REGENERATE
    assert "84" in result.scrubbed_text


def test_regenerate_when_some_ground_and_some_ungrounded() -> None:
    """A grounded survivor beside an ungrounded scrub yields regenerate (GROUND-R9)."""
    evidence = _FakeEvidence(metrics={"ctl": 84.0})
    claims = [_number("84", "ctl", 84.0), _name("Fake Workout")]
    result = ground("CTL 84 and do Fake Workout", claims, evidence, [])
    verdicts = {c.verdict for c in result.claims}
    assert GroundVerdict.GROUNDED in verdicts
    assert GroundVerdict.UNGROUNDED in verdicts
    assert result.decision is GroundDecision.REGENERATE
    assert "Fake Workout" not in result.scrubbed_text
    assert "84" in result.scrubbed_text


def test_contradicted_regenerates_with_canonical_value_not_replan() -> None:
    """A contradicted number regenerates with the canonical value, never published (GROUND-R9).

    Corrected GROUND-R9: a contradicted number is replaced in place by the canonical value
    and routes to a bounded re-draft (``regenerate``), NOT a coverage re-plan — the right
    value already exists. The contradicted value is still never published.
    """
    evidence = _FakeEvidence(metrics={"ctl": 84.0, "atl": 70.0})
    claims = [_number("84", "ctl", 84.0), _number("120", "atl", 120.0)]
    result = ground("CTL 84, ATL 120", claims, evidence, [])
    assert result.decision is GroundDecision.REGENERATE
    assert "120" not in result.scrubbed_text
    assert "70" in result.scrubbed_text


def test_survivors_helper_returns_only_grounded() -> None:
    """The GroundingResult.survivors property exposes grounded claims only (GROUND-R5)."""
    evidence = _FakeEvidence(metrics={"ctl": 84.0})
    claims = [_number("84", "ctl", 84.0), _name("Fake Workout")]
    result = ground("CTL 84 do Fake Workout", claims, evidence, [])
    assert len(result.survivors) == 1
    assert result.survivors[0].verdict is GroundVerdict.GROUNDED


# --- contract-evidence async path (GROUND-R2) ---


async def test_contract_metric_value_is_async_seam() -> None:
    """The fake honors the async GroundingEvidence.metric_value contract (GROUND-R2)."""
    evidence = _FakeEvidence(metrics={"ctl": 84.0})
    assert await evidence.metric_value("ctl", None) == 84.0
    assert await evidence.metric_value("missing", None) is None


def test_evidence_without_sync_accessor_scrubs_numbers() -> None:
    """Evidence exposing only the async contract (no sync accessor) scrubs numbers (fail-closed)."""

    class _AsyncOnly:
        async def metric_value(self, metric: str, as_of: str | None) -> float | None:
            return 84.0

        def url_allowed(self, url: str) -> bool:
            return False

    result = ground("CTL 84", [_number("84", "ctl", 84.0)], _AsyncOnly(), [])
    # No synchronous metric_snapshot -> grounder never awaits -> number unavailable.
    assert result.claims[0].verdict is GroundVerdict.UNGROUNDED
    assert "84" not in result.scrubbed_text  # the unverified number is scrubbed (fail-closed)
    # The metric is missing -> re-gatherable, so the aggregate is the recovery replan (GROUND-R6);
    # the reflection_count bound is what forces the eventual abstain (REFLECT-R4).
    assert result.decision is GroundDecision.REPLAN
