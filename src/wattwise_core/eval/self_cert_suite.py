"""No-self-certification suite (QA-EVAL-R2.10 / EVAL-R5a).

This suite is the testing-side home of the agentic-engine "no self-grade / no
self-certification" rule. Each case plants a model self-claim ("this answer is fully
grounded / verified / approved") ALONGSIDE an ungrounded fabricated number, then drives the
PRODUCTION deterministic grounder (:func:`wattwise_core.agent.grounding.ground`) — the same
shipped grounding identity the graph runs — to assert, programmatically:

* (a) the self-claim does NOT substitute for the grounding verdict — adding the self-cert
  STATEMENT claim to the draft changes NEITHER the aggregate
  :class:`~wattwise_core.agent.contracts.GroundDecision` NOR any per-claim verdict the
  grounder produces for the real (number/url) claims; and
* (b) the grounded/approved state is taken from the engine's grounding signals, not the
  model's stated confidence — the fabricated number is SCRUBBED (verdict ungrounded /
  contradicted, never published) regardless of the self-claim.

A self-certified-but-ungrounded answer therefore scores zero. Network-free and
deterministic (TIER-R1, QA-EVAL-R9).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wattwise_core.agent.contracts import (
    Claim,
    ClaimKind,
    GroundedClaim,
    GroundVerdict,
)
from wattwise_core.agent.grounding import ground
from wattwise_core.eval.grading import SelfCertGrade

_DATASETS_DIR = Path(__file__).parent / "datasets"


class _Evidence:
    """Minimal sync grounding evidence for the self-cert suite (GROUND-R2/-R7).

    Exposes the synchronous ``metric_snapshot`` the production grounder resolves numbers
    against and an empty URL allow-list. Numbers come VERBATIM from the case's canonical
    metrics; a value the case does not list resolves to ``None`` so the grounder scrubs the
    claim (fail-closed) — exactly the path a fabricated number takes.
    """

    def __init__(self, metrics: Mapping[str, float]) -> None:
        self._metrics = dict(metrics)

    def metric_snapshot(self, metric: str, as_of: str | None) -> float | None:
        return self._metrics.get(metric)

    async def metric_value(self, metric: str, as_of: str | None) -> float | None:
        return self._metrics.get(metric)

    def url_allowed(self, url: str) -> bool:
        return False


def _load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (_DATASETS_DIR / f"{name}.json").read_text(encoding="utf-8")
    )
    return loaded


def _fabricated_claim(case: dict[str, Any]) -> Claim:
    raw = case["fabricated_claim"]
    value = raw.get("value")
    return Claim(
        kind=ClaimKind(str(raw["kind"])),
        text=str(raw["text"]),
        metric=raw.get("metric"),
        value=float(value) if value is not None else None,
        ref=raw.get("ref"),
    )


def _self_cert_claim(case: dict[str, Any]) -> Claim:
    """The model's self-certification, modelled as a STATEMENT claim the grounder sees."""
    return Claim(kind=ClaimKind.STATEMENT, text=str(case["self_cert_text"]))


def _verdict_map(claims: tuple[GroundedClaim, ...]) -> dict[str, GroundVerdict]:
    """Per-claim verdict keyed by claim text, restricted to NON-statement claims.

    The self-cert STATEMENT is excluded so the comparison is over the REAL (number/url)
    claims only: adding the self-claim must not change the verdict of any real claim.
    """
    return {
        gc.claim.text: gc.verdict
        for gc in claims
        if gc.claim.kind is not ClaimKind.STATEMENT
    }


def _number_scrubbed(claims: tuple[GroundedClaim, ...], fabricated: Claim) -> bool:
    """True iff the fabricated number did NOT survive as a grounded published claim."""
    for gc in claims:
        if gc.claim.text == fabricated.text and gc.verdict is GroundVerdict.GROUNDED:
            return False
    return True


def _run_case(case: dict[str, Any]) -> list[str]:
    """Drive the production grounder with/without the self-claim; return failure reasons."""
    cid = str(case["id"])
    metrics = {str(k): float(v) for k, v in case.get("evidence", {}).get("metrics", {}).items()}
    evidence = _Evidence(metrics)
    fabricated = _fabricated_claim(case)
    draft = str(case["draft_prose"])
    # Run 1: the real claim ALONE (no self-cert claim present).
    baseline = ground(draft, (fabricated,), evidence, ())
    # Run 2: the SAME real claim PLUS the model's self-certification statement.
    with_self_cert = ground(draft, (fabricated, _self_cert_claim(case)), evidence, ())

    failures: list[str] = []
    # (b) the engine's grounding signal — not the self-claim — decides: the fabricated
    # number is scrubbed in the self-certified run.
    if not _number_scrubbed(with_self_cert.claims, fabricated):
        failures.append(f"{cid}: a self-certified fabricated number survived grounding")
    # (a) the self-claim does NOT move the verdict: aggregate decision unchanged AND every
    # real claim's verdict unchanged versus the run without the self-claim.
    if with_self_cert.decision is not baseline.decision:
        failures.append(
            f"{cid}: self-cert claim changed the GroundDecision "
            f"{baseline.decision} -> {with_self_cert.decision}"
        )
    if _verdict_map(with_self_cert.claims) != _verdict_map(baseline.claims):
        failures.append(f"{cid}: self-cert claim changed a real claim's verdict")
    return failures


async def grade_self_certification() -> SelfCertGrade:
    """Run the no-self-certification fixtures through the production grounder (QA-EVAL-R2.10).

    Async to match the engine-suite grader signature the runner awaits; the production
    grounder is synchronous and deterministic so this performs no I/O.
    """
    cases = _load("self_certification")["cases"]
    failures: list[str] = []
    passed = 0
    for case in cases:
        case_failures = _run_case(case)
        if case_failures:
            failures.extend(case_failures)
        else:
            passed += 1
    return SelfCertGrade(len(cases), passed, tuple(failures))


__all__ = ["grade_self_certification"]
