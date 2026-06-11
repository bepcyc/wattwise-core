"""Unit tests for the decorrelated entailment gate + fact sheet (issue #10, GROUND-R11).

The gate re-checks each published sentence against a CODE-rendered canonical fact sheet
and can only VETO (fail-closed): a non-entailed sentence is removed and the run re-drafts;
an unavailable verifier degrades to the deterministic layers and is reported. Tests drive
the gate with a scripted fake verifier — no model download, no network.
"""

from __future__ import annotations

import pytest

from wattwise_core.agent.contracts import (
    Claim,
    ClaimKind,
    GroundDecision,
    GroundedClaim,
    GroundingResult,
    GroundVerdict,
)
from wattwise_core.agent.grounding_entailment import (
    EntailmentGate,
    EntailmentThresholds,
    EntailmentVerifier,
)
from wattwise_core.agent.grounding_factsheet import render_fact_sheet

pytestmark = pytest.mark.unit

_THRESHOLDS = EntailmentThresholds(number=0.5, statement=0.5)


class _ScriptedVerifier:
    """Scores sentences by script: listed sentences are unsupported, the rest entailed."""

    def __init__(self, unsupported: frozenset[str] = frozenset()) -> None:
        self.unsupported = unsupported
        self.calls: list[str] = []

    async def support(self, *, sentence: str, facts: str) -> float:
        self.calls.append(sentence)
        return 0.1 if sentence in self.unsupported else 0.9


class _BrokenVerifier:
    """A verifier whose runtime always fails (the missing-dependency / load-failure path)."""

    async def support(self, *, sentence: str, facts: str) -> float:
        raise RuntimeError("verifier backend is unavailable")


def _grounded_number(text: str, metric: str, value: float) -> GroundedClaim:
    claim = Claim(kind=ClaimKind.NUMBER, text=text, metric=metric, value=value)
    citation = {"kind": "metric", "record_id": metric, "metric": metric, "value": value}
    return GroundedClaim(claim, GroundVerdict.GROUNDED, citation)


def _complementary(text: str) -> GroundedClaim:
    claim = Claim(kind=ClaimKind.STATEMENT, text=text)
    return GroundedClaim(claim, GroundVerdict.COMPLEMENTARY, None)


def _result(text: str, *claims: GroundedClaim) -> GroundingResult:
    return GroundingResult(
        decision=GroundDecision.PROCEED, claims=tuple(claims), scrubbed_text=text
    )


async def test_entailed_sentences_pass_unchanged() -> None:
    """A draft whose every checkable sentence is entailed publishes verbatim (no veto)."""
    gate = EntailmentGate(_ScriptedVerifier(), _THRESHOLDS)
    result = _result("Your fitness is 84.", _grounded_number("84", "ctl", 84.0))
    gated, report = await gate.apply(result, facts="canonical metric ctl (latest value): 84")
    assert gated.scrubbed_text == result.scrubbed_text
    assert gated.decision is GroundDecision.PROCEED
    assert report.checked == 1
    assert report.vetoed == ()


async def test_non_entailed_sentence_is_removed_and_run_redrafts() -> None:
    """A sentence the fact sheet does not entail is vetoed and ``proceed`` is revoked.

    The value gate verified each FIGURE; the entailment gate rejects the SENTENCE whose
    meaning the record does not support (the binding errors rules cannot enumerate) —
    the prose is removed and the run re-drafts (fail-closed, GROUND-R11).
    """
    bad = "Your fitness climbed steadily all month and sits at 84."
    verifier = _ScriptedVerifier(unsupported=frozenset({bad}))
    gate = EntailmentGate(verifier, _THRESHOLDS)
    text = f"{bad} Keep the easy days easy."
    result = _result(text, _grounded_number("84", "ctl", 84.0), _complementary("Keep it easy."))
    gated, report = await gate.apply(result, facts="canonical metric ctl (latest value): 84")
    assert bad not in gated.scrubbed_text
    assert "Keep the easy days easy." in gated.scrubbed_text
    assert gated.decision is GroundDecision.REGENERATE
    assert report.vetoed == (bad,)


async def test_numberless_trend_statement_is_checked_and_vetoed() -> None:
    """The COMPLEMENTARY free pass is closed: a numberless trend claim must be entailed.

    "Your HRV trend is stable." carries no checkable token, so the deterministic gate
    published it unverified; with the entailment gate on, the unsupported direction claim
    is vetoed — nothing publishable remains, so the run abstains (GROUND-R6).
    """
    trend = "Your HRV trend is stable."
    verifier = _ScriptedVerifier(unsupported=frozenset({trend}))
    gate = EntailmentGate(verifier, _THRESHOLDS)
    result = _result(trend, _complementary(trend))
    gated, report = await gate.apply(result, facts="canonical record hrv: falling")
    assert gated.scrubbed_text == ""
    assert gated.decision is GroundDecision.ABSTAIN
    assert report.vetoed == (trend,)


async def test_digitless_text_without_statement_claims_is_not_checked() -> None:
    """With no digits and no published statement claim there is nothing to check."""
    verifier = _ScriptedVerifier()
    gate = EntailmentGate(verifier, _THRESHOLDS)
    result = _result("Nice work this week.", _grounded_number("84", "ctl", 84.0))
    gated, report = await gate.apply(result, facts="canonical metric ctl: 84")
    assert verifier.calls == []
    assert report.checked == 0
    assert gated.scrubbed_text == result.scrubbed_text


async def test_unavailable_verifier_degrades_to_deterministic_layers() -> None:
    """A verifier fault returns the result UNCHANGED and reports the degradation.

    Issue #10 fail-closed rule: a missing/unloadable verifier is a RECORDED degradation
    to the deterministic layers — never a crash of the athlete's turn and never an
    unrecorded fail-open.
    """
    gate = EntailmentGate(_BrokenVerifier(), _THRESHOLDS)
    result = _result("Your fitness is 84.", _grounded_number("84", "ctl", 84.0))
    gated, report = await gate.apply(result, facts="canonical metric ctl: 84")
    assert gated is result
    assert report.unavailable is True


async def test_check_budget_bounds_verifier_calls() -> None:
    """The per-deliverable check budget bounds verifier calls deterministically."""
    verifier = _ScriptedVerifier()
    gate = EntailmentGate(verifier, _THRESHOLDS, max_checks=2)
    text = "CTL is 1. CTL is 2. CTL is 3. CTL is 4."
    result = _result(text, _grounded_number("1", "ctl", 1.0))
    _, report = await gate.apply(result, facts="canonical metric ctl: 1")
    assert len(verifier.calls) == 2
    assert report.checked == 2


async def test_non_proceed_decision_is_kept_after_a_veto() -> None:
    """A veto never WEAKENS the aggregate: an already-recovering decision is kept."""
    bad = "Your fitness is 84."
    gate = EntailmentGate(_ScriptedVerifier(unsupported=frozenset({bad})), _THRESHOLDS)
    result = GroundingResult(
        decision=GroundDecision.REPLAN,
        claims=(_grounded_number("84", "ctl", 84.0),),
        scrubbed_text=f"{bad} More soon.",
    )
    gated, _ = await gate.apply(result, facts="")
    assert gated.decision is GroundDecision.REPLAN
    assert bad not in gated.scrubbed_text


def test_gate_seam_accepts_protocol_implementations() -> None:
    """The scripted fake satisfies the runtime-checkable verifier seam (typed contract)."""
    assert isinstance(_ScriptedVerifier(), EntailmentVerifier)


# --- the code-rendered fact sheet (the verifier's only evidence) ---------------------------


def test_fact_sheet_renders_snapshots_records_and_request_deterministically() -> None:
    """The sheet carries snapshots, retrieved records, and the athlete's request, sorted.

    Code-rendered and deterministic (GRAPH-R4): values verbatim from the resolved
    snapshots (GROUND-R7), records as compact sorted JSON, the request as its own line so
    a user-constraint echo is entailed rather than vetoed. ``None`` snapshots are omitted
    (an unavailable metric adds NO fact).
    """
    sheet = render_fact_sheet(
        {("ctl", None): 84.0, ("atl", "2026-06-09"): 70.5, ("hrv", None): None},
        {"weekly_load": {"tss": 300}},
        request_text="Can I race Sunday?",
    )
    assert sheet.splitlines() == [
        "canonical metric atl (as of 2026-06-09): 70.5",
        "canonical metric ctl (latest value): 84",
        'canonical record weekly_load: {"tss": 300}',
        "the athlete's request says: Can I race Sunday?",
    ]


def test_fact_sheet_truncates_with_a_marker_when_oversized() -> None:
    """An oversized sheet truncates WITH a marker — absent facts can only cause vetoes."""
    snapshots = {(f"metric_{i:03d}", None): float(i) for i in range(400)}
    sheet = render_fact_sheet(snapshots, max_chars=500)
    assert len(sheet) <= 500
    assert sheet.endswith("[fact sheet truncated]")
