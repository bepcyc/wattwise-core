"""The two-layer composed-answer types + structured-compose seam (COMPOSE-R3, #87).

Slice 1 of the COMPOSE-R3 epic: the typed evidence/visible layers and the
``compose_structured`` helper exist and behave, with NO node wired yet. Asserts the
schema shape, the ``EvidenceClaim → Claim`` projection the grounder reuses, and that
the helper drives a model's schema-constrained ``structured`` decoding (never a flat
blob).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from wattwise_core.agent.contracts import (
    Claim,
    ClaimKind,
    ComposedAnswer,
    EvidenceClaim,
    compose_structured,
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


class _StructuredOnlyModel:
    """A ChatModel fake whose ``structured`` decodes the requested schema; records the call."""

    def __init__(self, answer: ComposedAnswer) -> None:
        self._answer = answer
        self.seen_schema: type[BaseModel] | None = None
        self.seen_data: str | None = None

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        self.seen_schema = schema
        self.seen_data = data
        assert schema is ComposedAnswer
        return self._answer  # type: ignore[return-value]

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        raise AssertionError("compose_structured must use structured decoding, not flat compose")


async def test_compose_structured_drives_schema_constrained_decoding() -> None:
    """compose_structured asks the model for a ComposedAnswer via structured decoding.

    It must route through ``structured`` (schema-constrained), never the flat ``compose`` —
    the visible + evidence layers arrive as one validated unit (COMPOSE-R3).
    """
    expected = ComposedAnswer(
        visible_answer="You're fresh and ready.",
        evidence_claims=(
            EvidenceClaim(kind=ClaimKind.NUMBER, text="form 4.8", metric="tsb", value=4.8),
        ),
    )
    model = _StructuredOnlyModel(expected)
    got = await compose_structured(model, system="coach", context="fact sheet")
    assert got is expected
    assert model.seen_schema is ComposedAnswer
    assert model.seen_data == "fact sheet"
