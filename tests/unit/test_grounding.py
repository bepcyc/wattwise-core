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
    """A number whose canonical computation is unavailable is removed, never zeroed (GROUND-R7)."""
    evidence = _FakeEvidence(metrics={})  # hrv unavailable
    result = ground("Your HRV is 65 today", [_number("65", "hrv", 65.0)], evidence, [])
    (claim,) = result.claims
    assert claim.verdict is GroundVerdict.UNGROUNDED
    assert "65" not in result.scrubbed_text
    assert "0" not in result.scrubbed_text.replace("today", "")  # no placeholder/zero
    assert result.decision is GroundDecision.ABSTAIN


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


def test_abstain_when_nothing_grounds() -> None:
    """With every claim scrubbed and none publishable, the decision is abstain (GROUND-R6/R9)."""
    evidence = _FakeEvidence()
    claims = [_number("99", "ctl", 99.0), _name("Fake Workout"), _url("http://x.invalid")]
    result = ground("99 Fake Workout http://x.invalid", claims, evidence, [])
    assert all(c.verdict is GroundVerdict.UNGROUNDED for c in result.claims)
    assert result.decision is GroundDecision.ABSTAIN


def test_abstain_on_empty_claims() -> None:
    """No claims at all means nothing publishable -> abstain (fail-closed default)."""
    result = ground("", [], _FakeEvidence(), [])
    assert result.decision is GroundDecision.ABSTAIN
    assert result.claims == ()


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
    assert result.decision is GroundDecision.ABSTAIN
