"""The two-layer composed-answer types + the inline-tag parser (COMPOSE-R3, #87).

Asserts the schema shape (visible prose + typed evidence layer), the ``EvidenceClaim → Claim``
projection the grounder reuses, and :func:`parse_tagged_answer` — the deterministic simple-regex
split of a ``<technical_proof>``-tagged model answer into the visible prose (``draft``) plus the
parsed evidence-claim layer, fail-closed at every tag edge (unclosed/duplicate/stray/malformed).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wattwise_core.agent.contracts import (
    Claim,
    ClaimKind,
    ComposedAnswer,
    EvidenceClaim,
    parse_tagged_answer,
)

pytestmark = pytest.mark.unit


def test_composed_answer_carries_visible_prose_and_evidence_layer() -> None:
    """A ComposedAnswer holds the visible prose plus the typed evidence-claim layer."""
    answer = ComposedAnswer(
        visible_answer="Your fitness is climbing steadily.",
        evidence_claims=(
            EvidenceClaim(kind=ClaimKind.NUMBER, text="fitness 5.7", metric="ctl", value=5.7),
        ),
    )
    assert answer.visible_answer == "Your fitness is climbing steadily."
    assert len(answer.evidence_claims) == 1
    assert answer.evidence_claims[0].value == 5.7


def test_composed_answer_evidence_layer_defaults_empty() -> None:
    """The evidence layer defaults to empty so a number-free answer is representable."""
    answer = ComposedAnswer(visible_answer="Nice work staying consistent.")
    assert answer.evidence_claims == ()


def test_composed_answer_requires_visible_answer() -> None:
    """visible_answer is required: a structured output without it is a validation failure.

    This is the COMPOSE-R3 'structured-only, no flat-blob fallback' guarantee at the type
    layer — an output carrying no typed visible prose cannot be constructed.
    """
    with pytest.raises(ValidationError):
        ComposedAnswer.model_validate({"evidence_claims": []})


def test_evidence_claim_projects_onto_grounding_claim() -> None:
    """EvidenceClaim.to_claim yields the internal Claim the grounder already verifies."""
    ec = EvidenceClaim(
        kind=ClaimKind.NAME,
        text="Sweet Spot 3x12",
        prescriptive=True,
        workout_type="sweet_spot",
    )
    claim = ec.to_claim()
    assert isinstance(claim, Claim)
    assert claim.kind is ClaimKind.NAME
    assert claim.text == "Sweet Spot 3x12"
    assert claim.prescriptive is True
    assert claim.workout_type == "sweet_spot"


# --- parse_tagged_answer: inline <technical_proof> tags -> ComposedAnswer (COMPOSE-R3, tags) ---


def test_parse_prose_only_has_empty_evidence() -> None:
    """No tag at all: the whole (stripped) text is the visible prose, evidence empty."""
    out = parse_tagged_answer("  You're building steadily.\n")
    assert out.visible_answer == "You're building steadily."
    assert out.evidence_claims == ()


def test_parse_splits_block_from_visible_and_extracts_number_claim() -> None:
    """A closed block is stripped from the visible prose and parsed into a NUMBER claim.

    Mirrors the owner's framing: technical proof inside the tag, warm prose outside.
    The parenthetical ``as_of`` date maps onto EvidenceClaim.ref (no as_of field).
    """
    text = (
        "<technical_proof>fitness is 5.7 (ctl, as_of 2026-06-15); "
        "fatigue 4.8 (atl) — basis for the read</technical_proof>\n"
        "You've been building steadily — fitness is climbing while fatigue stays manageable."
    )
    out = parse_tagged_answer(text)
    assert "technical_proof" not in out.visible_answer
    assert out.visible_answer.startswith("You've been building steadily")
    metrics = {(c.metric, c.value, c.ref) for c in out.evidence_claims}
    assert ("ctl", 5.7, "2026-06-15") in metrics
    assert ("atl", 4.8, None) in metrics
    assert all(c.kind is ClaimKind.NUMBER for c in out.evidence_claims)


def test_parse_unclosed_block_consumes_to_end_failclosed() -> None:
    """An UNCLOSED opening tag consumes to end-of-text — the tail never leaks as visible."""
    text = "You're recovered.\n<technical_proof>freshness is 6.0 (tsb) and the rest is cut off"
    out = parse_tagged_answer(text)
    assert out.visible_answer == "You're recovered."
    assert "technical_proof" not in out.visible_answer
    assert ("tsb", 6.0, None) in {(c.metric, c.value, c.ref) for c in out.evidence_claims}


def test_parse_strips_every_block_and_stray_fragment() -> None:
    """Duplicate blocks + an orphan closing fragment are all removed from the visible prose."""
    text = (
        "<technical_proof>load is 42 (tss)</technical_proof>"
        "Lead prose.</technical_proof> more prose "
        "<technical_proof>ramp is 1.1 (ramp_rate)</technical_proof>tail."
    )
    out = parse_tagged_answer(text)
    assert "technical_proof" not in out.visible_answer
    assert "<" not in out.visible_answer and ">" not in out.visible_answer
    vals = {c.value for c in out.evidence_claims}
    assert {42.0, 1.1} <= vals


def test_parse_skips_malformed_claim_lines_failsoft() -> None:
    """A segment with no parseable number is skipped (fail-soft), never raises."""
    text = (
        "<technical_proof>you are doing great; consistency matters; "
        "load is 100 (tss)</technical_proof>Keep it up."
    )
    out = parse_tagged_answer(text)
    assert out.visible_answer == "Keep it up."
    assert {(c.metric, c.value) for c in out.evidence_claims} == {("tss", 100.0)}


def test_parse_does_not_split_a_spaced_numeric_range() -> None:
    """The em-dash/hyphen free-prose tail strip must NOT eat a spaced numeric range.

    A range like ``5 - 7`` is not a clean single NUMBER claim, so the segment is skipped
    rather than mis-parsed into the value ``5`` (which would silently drop the ``7``).
    """
    out = parse_tagged_answer("<technical_proof>weekly hours 5 - 7 (target)</technical_proof>ok")
    # No claim fabricated with value 5 from the truncated range.
    assert all(c.value != 5.0 for c in out.evidence_claims)


def test_parse_output_round_trips_through_evidence_claim_model_validate() -> None:
    """Each parsed claim's model_dump round-trips back through EvidenceClaim (slice-3 gate)."""
    out = parse_tagged_answer(
        "<technical_proof>fitness is 5.7 (ctl, as_of 2026-06-15)</technical_proof>hi"
    )
    for c in out.evidence_claims:
        again = EvidenceClaim.model_validate(c.model_dump())
        assert again.to_claim().metric == c.metric


def test_parse_block_only_yields_empty_visible() -> None:
    """A block with no surrounding prose yields empty visible prose (caller routes to abstain)."""
    out = parse_tagged_answer("<technical_proof>form 4.8 (tsb)</technical_proof>")
    assert out.visible_answer == ""
    assert out.evidence_claims  # evidence still parsed


def test_parse_per_ride_tss_maps_activity_to_ref() -> None:
    """#47: a per-ride TSS claim maps its ``activity <id>`` paren token onto EvidenceClaim.ref.

    The technical block writes ``each ride 21 (activity_tss, activity act-123)``; the parser keys
    the claim by the ACTIVITY id in ref (not a date), with metric overridden to ``activity_tss``,
    and the ``activity`` ref marker is NOT mis-picked as the metric override.
    """
    out = parse_tagged_answer(
        "<technical_proof>each ride 21 (activity_tss, activity act-123)</technical_proof>"
        "Your ride was hard."
    )
    assert out.visible_answer == "Your ride was hard."
    perride = [c for c in out.evidence_claims if c.metric == "activity_tss"]
    assert perride and perride[0].value == 21.0 and perride[0].ref == "act-123"


def test_parse_per_ride_tss_without_activity_ref_has_none_ref() -> None:
    """#47 fail-closed: a per-ride claim with no ``activity <id>`` token carries ref=None.

    Per-ride TSS is per-day ambiguous, so a claim that names no activity must NOT fall back to a
    date — it keeps ref=None and the grounder scrubs it (an empty-ref activity_tss resolves None).
    """
    out = parse_tagged_answer("<technical_proof>each ride 21 (activity_tss)</technical_proof>hi")
    perride = [c for c in out.evidence_claims if c.metric == "activity_tss"]
    assert perride and perride[0].ref is None


def test_parse_dated_claim_still_maps_as_of_not_activity() -> None:
    """#47 mutual-exclusion: a dated claim still maps as_of -> ref, never an activity id."""
    out = parse_tagged_answer(
        "<technical_proof>fitness is 5.7 (ctl, as_of 2026-06-15)</technical_proof>hi"
    )
    assert ("ctl", 5.7, "2026-06-15") in {(c.metric, c.value, c.ref) for c in out.evidence_claims}


def test_parse_attributed_opener_does_not_leak_block_body_as_prose() -> None:
    """An attributed opener (`<technical_proof foo>`) must still have its body stripped, not leaked.

    The model is instructed to emit a bare tag, but a deviating attributed opener MUST NOT spill the
    evidence reasoning into the visible prose (the fail-closed premise defends against deviation).
    """
    text = (
        '<technical_proof lang="en">secret internal reasoning; ctl 5.7'
        "</technical_proof>You are fresh."
    )
    out = parse_tagged_answer(text)
    assert "technical_proof" not in out.visible_answer
    assert "secret internal reasoning" not in out.visible_answer
    assert out.visible_answer == "You are fresh."
