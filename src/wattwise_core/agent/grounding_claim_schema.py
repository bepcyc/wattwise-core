"""Candidate-claim vocabulary, schema, and SOURCING for the grounder (GROUND-R2/STRUCT-R5).

Factored out of :mod:`wattwise_core.agent.grounding_evidence` (QUAL-R9 module-size ceiling) as a
focused leaf holding the claim-extraction VOCABULARY and the candidate-claim SOURCE decision:

* :class:`CanonicalWorkoutType` / :data:`_WORKOUT_TYPE_TO_NAME` — the closed, language-independent
  canonical workout-type vocabulary a prescribed-workout NAME claim grounds by structure (COACH-R2).
* :class:`_ExtractedClaim` / :class:`_ClaimSchema` — the structured-output shape the model emits.
* :func:`source_claims` — COMPOSE-R3 point 2: a POPULATED two-layer evidence layer is the
  authoritative candidate source; an absent/empty layer falls back to draft extraction.

It depends only on leaf contracts (``Claim``/``ClaimKind``/``EvidenceClaim``) and the structured
seam, never on ``grounding_evidence``, so the import is one-directional (no cycle).
``grounding_evidence`` re-exports these names so existing import sites (engine, grounding, tests)
are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from wattwise_core.agent.compose_contracts import EvidenceClaim
from wattwise_core.agent.contracts import ChatModel, Claim, ClaimKind
from wattwise_core.agent.structured import StructuredOutputError, run_structured


class CanonicalWorkoutType(StrEnum):
    """The closed, LANGUAGE-INDEPENDENT canonical workout-type vocabulary (COACH-R2, STRUCT-R1/R3).

    The typed prescription the model emits as STRUCTURED output for a prescribed-workout NAME
    claim: a plan in ANY language carries the SAME enum member, so grounding resolves the workout
    by its STRUCTURE — never by re-matching a translated surface name (the language-enumeration the
    owner rejected). Each member maps 1:1 to a canonical training-prescription name in
    :data:`CANONICAL_WORKOUT_NAMES`, so a structured-type ground yields the SAME stable
    ``workout:{name}`` canonical id as the legacy surface-name path (GROUND-R5 citation stability).
    The enum is exhaustive for the prescription decision space (STRUCT-R3): an out-of-vocabulary
    value is a structured-output validation failure, never a new type.
    """

    REST_DAY = "rest_day"
    RECOVERY_RIDE = "recovery_ride"
    RECOVERY_SPIN = "recovery_spin"
    ENDURANCE_RIDE = "endurance_ride"
    LONG_RIDE = "long_ride"
    TEMPO_INTERVALS = "tempo_intervals"
    SWEET_SPOT_INTERVALS = "sweet_spot_intervals"
    THRESHOLD_INTERVALS = "threshold_intervals"
    VO2MAX_INTERVALS = "vo2max_intervals"
    ANAEROBIC_INTERVALS = "anaerobic_intervals"
    SPRINT_INTERVALS = "sprint_intervals"


#: The structured canonical workout TYPE -> the normalized canonical NAME its ``workout:{name}``
#: citation id is keyed on. The type's identifier IS the canonical name with spaces (so the
#: structured-type ground and the legacy surface-name ground produce a byte-identical canonical id).
_WORKOUT_TYPE_TO_NAME: Mapping[CanonicalWorkoutType, str] = {
    member: member.value.replace("_", " ") for member in CanonicalWorkoutType
}


class _ExtractedClaim(BaseModel):
    """One candidate claim the model points at (STRUCT-R5); code verifies it, not the model.

    ``workout_type`` is the typed canonical prescription the model emits for a prescribed-workout
    NAME claim (COACH-R2, language-independent, issue #18): a closed :class:`CanonicalWorkoutType`
    enum so the plan-structure verdict is provider-enforced over the canonical vocabulary (STRUCT-R1
    /R3), not the translated surface name. It is optional (``None``) — a NUMBER/URL/STATEMENT claim
    carries none, and a NAME claim without it falls back to the legacy surface-name match.

    ``prescriptive`` (GROUND-R9) distinguishes a directive statement ("do threshold work tomorrow")
    from a descriptive one ("you did a threshold workout yesterday"). A prescriptive statement
    with no verifiable number is classified COMPLEMENTARY and may be scrubbed, while a descriptive
    one is published under the complementary free pass.
    """

    model_config = {"extra": "forbid"}
    kind: ClaimKind = ClaimKind.NUMBER
    text: str = ""
    metric: str | None = None
    value: float | None = None
    as_of: str | None = None
    prescriptive: bool = False
    workout_type: CanonicalWorkoutType | None = None


class _ClaimSchema(BaseModel):
    """The structured claim-extraction output (GROUND-R2/STRUCT-R5)."""

    model_config = {"extra": "forbid"}
    claims: list[_ExtractedClaim] = Field(default_factory=list)


def _extracted_to_claim(c: _ExtractedClaim) -> Claim:
    """Project a model-extracted candidate onto the internal grounding ``Claim``."""
    return Claim(
        kind=c.kind,
        text=c.text,
        metric=c.metric,
        value=c.value,
        ref=c.as_of,
        prescriptive=c.prescriptive,
        # The typed canonical prescription (language-independent, #18): carried onto the claim so
        # the NAME verifier grounds by STRUCTURE, not the translated surface name.
        workout_type=c.workout_type.value if c.workout_type is not None else None,
    )


async def source_claims(
    *,
    model: ChatModel,
    claim_system: str,
    draft: str,
    evidence_claims: Sequence[Mapping[str, Any]] | None,
) -> list[Claim]:
    """Resolve the candidate-claim list the grounder verifies (COMPOSE-R3 point 2 / GROUND-R2).

    When the two-layer answer carries a POPULATED evidence layer it is the authoritative source:
    each entry (the ``model_dump()`` of an :class:`EvidenceClaim`) is RE-VALIDATED through
    ``EvidenceClaim`` — never trusted as a raw dict — and projected via ``to_claim``, so a malformed
    entry fails closed at the type boundary rather than smuggling an unvalidated claim into
    grounding. A missing (``None``) or EMPTY layer falls back to extracting candidate claims from
    the visible ``draft`` — the transitional bridge until compose always populates the layer, so
    every fake/legacy caller emitting no evidence claims keeps its prior draft-extraction behaviour.
    """
    if evidence_claims:
        return [EvidenceClaim.model_validate(ec).to_claim() for ec in evidence_claims]
    try:
        extracted = await run_structured(
            model, system=claim_system, data=draft, schema=_ClaimSchema
        )
    except (StructuredOutputError, NotImplementedError):
        return []
    return [_extracted_to_claim(c) for c in extracted.claims]


__all__ = [
    "_WORKOUT_TYPE_TO_NAME",
    "CanonicalWorkoutType",
    "_ClaimSchema",
    "_ExtractedClaim",
    "source_claims",
]
