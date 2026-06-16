"""Two-layer composed-answer contracts (COMPOSE-R3, #87).

Factored out of :mod:`wattwise_core.agent.contracts` (QUAL-R9 module-size ceiling) as a
focused leaf holding ONLY the COMPOSE-R3 vocabulary: the visible/evidence layer types and
the structured-compose helper. It depends on the base claim + model seam from ``contracts``
(``Claim``/``ClaimKind``/``ChatModel``); ``contracts`` re-exports these names from its own
surface so existing import sites are unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel

from wattwise_core.agent.contracts import ChatModel, Claim, ClaimKind


class EvidenceClaim(BaseModel):
    """One candidate claim in the evidence layer of a two-layer answer (COMPOSE-R3 / STRUCT-R5).

    The provider-enforced structured-output mirror of :class:`Claim`: it carries the same
    candidate-claim fields a model may emit, but as a schema-constrained ``BaseModel`` so the
    evidence layer is decoded under STRUCT-R1, never free-text parsed. :meth:`to_claim` projects
    it onto the internal :class:`Claim` the grounding pipeline (GROUND-R2) already verifies, so
    the new layer reuses the existing grounder unchanged.
    """

    kind: ClaimKind
    text: str
    metric: str | None = None
    value: float | None = None
    ref: str | None = None
    prescriptive: bool = False
    workout_type: str | None = None

    def to_claim(self) -> Claim:
        """Project this schema-constrained evidence claim onto the internal grounding ``Claim``."""
        return Claim(
            kind=self.kind,
            text=self.text,
            metric=self.metric,
            value=self.value,
            ref=self.ref,
            prescriptive=self.prescriptive,
            workout_type=self.workout_type,
        )


class ComposedAnswer(BaseModel):
    """The two-layer ``compose`` output (COMPOSE-R3): a visible prose layer + an evidence layer.

    ``visible_answer`` is the warm, observation-first coach prose the athlete reads (VOICE-R1/-R7),
    carried downstream as the STATE-R2 ``draft``. ``evidence_claims`` is the internal evidence
    layer — every supporting candidate claim and canonical number — the GROUND-R2/STRUCT-R5
    extraction consumes as its authoritative source. The evidence layer is NEVER shown to the
    athlete (VOICE-R2) and NEVER serialized into an API response (OUTCOME-R2). A model that cannot
    emit a typed ``visible_answer`` is a STRUCT-R2 validation failure, not a flat-blob fallback.
    """

    visible_answer: str
    evidence_claims: tuple[EvidenceClaim, ...] = ()


async def compose_structured(
    model: ChatModel, *, system: str, context: str, max_tokens: int = 1024
) -> ComposedAnswer:
    """Compose the two-layer answer (COMPOSE-R3) over any ``ChatModel`` via its structured seam.

    A free function (not a ``ChatModel`` protocol member) so every existing model/fake satisfies
    the seam unchanged: it drives the provider's schema-constrained decoding (``structured``) to
    return a validated :class:`ComposedAnswer` — the visible prose and the evidence layer as ONE
    unit, never a flat blob split heuristically. A model that cannot enforce structured output
    raises here (STRUCT-R2), which the compose node handles as a validation failure, not a
    free-text fallback. ``max_tokens`` is accepted for signature parity with
    :meth:`ChatModel.compose` and reserved for a future budget-aware structured call.
    """
    return await model.structured(system=system, data=context, schema=ComposedAnswer)


__all__ = ["ComposedAnswer", "EvidenceClaim", "compose_structured"]
