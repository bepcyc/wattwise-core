"""Two-layer composed-answer contracts (COMPOSE-R3, #87).

Factored out of :mod:`wattwise_core.agent.contracts` (QUAL-R9 module-size ceiling) as a
focused leaf holding ONLY the COMPOSE-R3 vocabulary: the visible/evidence layer types and
the structured-compose helper. It depends on the base claim + model seam from ``contracts``
(``Claim``/``ClaimKind``/``ChatModel``); ``contracts`` re-exports these names from its own
surface so existing import sites are unchanged.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from wattwise_core.agent.contracts import Claim, ClaimKind


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


# --- inline <technical_proof> tag parsing (COMPOSE-R3, owner's tag framing, #87) -----------
#
# The model emits ONE plain-text answer: a `<technical_proof>…</technical_proof>` block carrying
# the evidence layer (numbers + reasoning), and the warm visible prose OUTSIDE it. A SIMPLE regex
# (owner-approved) splits that into the same internal ComposedAnswer the grounder already consumes.
# Fail-closed at every edge: an unclosed tag consumes to end-of-text (its tail never leaks as
# visible prose); duplicate/nested blocks are all removed; a stray tag fragment is scrubbed; a
# malformed claim line is skipped (the downstream grounder + deterministic sweep are the final
# authority). The function NEVER raises and NEVER returns raw tag text as the visible layer.

#: A `<technical_proof>` block. The opener TOLERATES attributes (``<technical_proof foo="x">``) —
#: the model is instructed to emit the bare tag, but a deviating attributed opener MUST still have
#: its body captured-and-stripped (never leaked as prose), so the opener is ``\btechnical_proof\b``
#: with an optional ``[^>]*`` attribute run. An unclosed opener matches to end-of-text via the
#: ``|$`` alternation (fail-closed: the unterminated tail is treated as evidence and stripped).
_TECH_PROOF_BLOCK_RE = re.compile(
    r"<\s*technical_proof\b[^>]*>(.*?)(?:<\s*/\s*technical_proof\s*>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
#: Any residual opening/closing tag fragment the block regex did not consume (orphan close tag,
#: an inner nested tag left inside a captured body, a malformed opener).
_TECH_PROOF_FRAGMENT_RE = re.compile(r"<\s*/?\s*technical_proof\b[^>]*>?", flags=re.IGNORECASE)
#: A single NUMBER claim: an optional metric label, an optional ``is``, then a signed decimal,
#: with an optional ``(canonical_metric, as_of YYYY-MM-DD)`` parenthetical.
_CLAIM_RE = re.compile(
    r"(?P<metric>[A-Za-z][A-Za-z ]*?)\s+(?:is\s+)?(?P<value>-?\d+(?:\.\d+)?)\s*"
    r"(?:\((?P<paren>[^)]*)\))?",
)
_AS_OF_RE = re.compile(r"as_of\s+(?P<as_of>\d{4}-\d{2}-\d{2})", flags=re.IGNORECASE)
#: A per-ACTIVITY reference token in the parenthetical (``activity <id>``) for a per-ride TSS
#: claim (#47): the activity id maps onto :attr:`EvidenceClaim.ref` (the grounder resolves the
#: single ride's TSS by activity id). Mutually exclusive with ``as_of`` — a claim is dated OR
#: activity-scoped, never both.
_ACTIVITY_REF_RE = re.compile(r"activity\s+(?P<aid>[A-Za-z0-9][\w.-]*)", flags=re.IGNORECASE)
#: A free-prose tail after an em/en-dash or spaced hyphen (" — basis for the read") that must be
#: dropped before claim parsing — but ONLY when the tail is non-numeric prose, so a spaced numeric
#: range ("5 - 7") is left intact (and then skipped as a non-single-number segment, never halved).
_PROSE_TAIL_RE = re.compile(r"\s+[—–-]\s+(?=\D*$)")  # noqa: RUF001 (em/en dash intended)


def _parse_claim_lines(block_body: str) -> list[EvidenceClaim]:
    """Extract NUMBER EvidenceClaims from a technical-proof block body (SIMPLE regex, fail-soft).

    The block is a ``;``-separated list of claims in the owner's style
    ``"fitness is 5.7 (ctl, as_of 2026-06-15); fatigue 4.8 (atl) — basis for the read"``. Each
    segment is reduced to its claim head (a trailing free-prose ``— …`` tail is dropped) and matched
    for ``metric value (paren)``. The parenthetical's leading non-``as_of`` token is the canonical
    metric override (e.g. ``ctl``); ``as_of <ISO>`` maps onto :attr:`EvidenceClaim.ref`. A segment
    that is not a single clean NUMBER (no number, a spaced range, multiple numbers) is SKIPPED —
    the grounder re-validates and the deterministic sweep still governs the visible draft, so a
    skipped line only means that claim is absent from the evidence layer (fail-closed: when in
    doubt, drop the claim). Never raises.
    """
    claims: list[EvidenceClaim] = []
    for raw_segment in block_body.split(";"):
        head = _PROSE_TAIL_RE.split(raw_segment.strip(), maxsplit=1)[0].strip()
        if not head:
            continue
        # Reject a spaced numeric range ("5 - 7") outright: more than one number OUTSIDE the
        # parenthetical is not a single clean NUMBER claim, so skip rather than mis-parse it into
        # the first value. The parenthetical (which may carry an ``as_of`` ISO date full of digits)
        # is removed before counting so a dated claim is not mistaken for a multi-number segment.
        head_outside_paren = re.sub(r"\([^)]*\)", "", head)
        if len(re.findall(r"-?\d+(?:\.\d+)?", head_outside_paren)) != 1:
            continue
        match = _CLAIM_RE.search(head)
        if match is None:
            continue
        paren = match.group("paren") or ""
        # ref is a DATE (``as_of <ISO>``) OR an ACTIVITY id (``activity <id>``, per-ride TSS,
        # #47) — mutually exclusive; the date takes precedence if (wrongly) both appear.
        as_of_match = _AS_OF_RE.search(paren)
        activity_match = _ACTIVITY_REF_RE.search(paren)
        if as_of_match:
            ref: str | None = as_of_match.group("as_of")
        elif activity_match:
            ref = activity_match.group("aid")
        else:
            ref = None
        # Canonical-metric override: the first parenthetical token that is neither the ``as_of``
        # date marker NOR the ``activity <id>`` ref marker (e.g. "ctl"/"activity_tss"), preferred
        # over the prose label; MetricEquivalence canonicalizes it downstream. Skipping ``activity``
        # prevents the ref marker from being mis-picked as the metric.
        override = ""
        for raw_token in paren.split(","):
            token = raw_token.strip()
            low = token.lower()
            if token and not low.startswith("as_of") and not low.startswith("activity "):
                override = token.split()[0]
                break
        metric = (override or match.group("metric")).strip()
        try:
            value = float(match.group("value"))
        except ValueError:  # pragma: no cover - regex already constrains the shape
            continue
        claims.append(
            EvidenceClaim(
                kind=ClaimKind.NUMBER, text=head, metric=metric or None, value=value, ref=ref
            )
        )
    return claims


def parse_tagged_answer(text: str) -> ComposedAnswer:
    """Split a tagged model answer into the two-layer :class:`ComposedAnswer` (COMPOSE-R3, tags).

    ``visible_answer`` is the text with EVERY ``<technical_proof>`` block (closed or unclosed-to-
    end) and any stray tag fragment removed, whitespace-normalized. ``evidence_claims`` is union of
    NUMBER claims parsed from every block body. Totality: this never raises and never returns raw
    tag text as the visible layer — a block-only answer yields an EMPTY ``visible_answer`` (the
    caller routes empty-visible to the fail-closed degrade/abstain path, never ships it).
    """
    blocks = [m.group(1) for m in _TECH_PROOF_BLOCK_RE.finditer(text)]
    visible = _TECH_PROOF_BLOCK_RE.sub("", text)
    visible = _TECH_PROOF_FRAGMENT_RE.sub("", visible)
    visible = re.sub(r"[ \t]{2,}", " ", visible)
    visible = re.sub(r"\n{3,}", "\n\n", visible).strip()
    claims: list[EvidenceClaim] = []
    for body in blocks:
        claims.extend(_parse_claim_lines(body))
    return ComposedAnswer(visible_answer=visible, evidence_claims=tuple(claims))


__all__ = ["ComposedAnswer", "EvidenceClaim", "parse_tagged_answer"]
