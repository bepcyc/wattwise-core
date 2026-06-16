"""Unit tests for the deterministic constraint gate (proposed GROUND-R13/R14, ADR 0008).

Covers the constraint FLOOR (the deterministic, model-free contradiction layer of ADR 0008
§1/§2/§10 — NOT the deferred NLI verifier): a HARD constraint VETOES a contradicting prescription
(scrub + non-proceed) while a SOFT one only CAUTIONS (the prescription stays, flagged); the key
necessity case that a NEUTRAL activity must NOT be blocked (an "easy swim" against "no running"
still proceeds — proving the floor is contradiction-detection, not the over-blocking
support-detection of ADR 0008 §1); and the multilingual floor (a German/Russian constraint gates
an English prescription, ADR 0008 §8). Pure, deterministic — no model, no IO (GRAPH-R4).
"""

from __future__ import annotations

import pytest

from wattwise_core.agent.contracts import GroundDecision, GroundingResult
from wattwise_core.agent.grounding_constraints import (
    ActiveConstraint,
    ConstraintGate,
    ConstraintMode,
    apply_constraint_gate,
    forbidden_activities,
)
from wattwise_core.agent.memory import ConstraintSeverity


def _proceed(text: str) -> GroundingResult:
    """A minimal PROCEED grounding result over ``text`` (no claims needed for the gate)."""
    return GroundingResult(decision=GroundDecision.PROCEED, claims=(), scrubbed_text=text)


def _hard(content: str) -> ActiveConstraint:
    return ActiveConstraint.from_content(content, ConstraintSeverity.HARD)


def _soft(content: str) -> ActiveConstraint:
    return ActiveConstraint.from_content(content, ConstraintSeverity.SOFT)


@pytest.mark.unit
def test_hard_running_constraint_vetoes_running_prescription() -> None:
    """A HARD 'no running' constraint VETOES a 'run 5x4 min' prescription (GROUND-R14 veto)."""
    result = _proceed("Let's run 5x4 min hard on Tuesday. Keep the rest easy.")
    gated = apply_constraint_gate(result, [_hard("No running — doctor's orders.")])

    assert gated.hard_violations  # a HARD violation was detected
    # Scrubbed: the contradicting prescription is removed from the published text.
    assert "run 5x4 min" not in gated.result.scrubbed_text.casefold()
    assert "keep the rest easy" in gated.result.scrubbed_text.casefold()
    # Decision forced off PROCEED (re-draft, since grounded prose survives).
    assert gated.result.decision is GroundDecision.REGENERATE


@pytest.mark.unit
def test_hard_veto_with_nothing_surviving_abstains() -> None:
    """A HARD veto that empties the published text downgrades to ABSTAIN (GROUND-R6)."""
    result = _proceed("Run intervals tomorrow.")
    gated = apply_constraint_gate(result, [_hard("Avoid running for now.")])

    assert gated.hard_violations
    assert gated.result.scrubbed_text.strip() == ""
    assert gated.result.decision is GroundDecision.ABSTAIN


@pytest.mark.unit
def test_soft_constraint_cautions_without_scrubbing() -> None:
    """A SOFT constraint surfaces a CAUTION but does NOT scrub or downgrade (GROUND-R14 caution)."""
    text = "Let's run 5x4 min hard on Tuesday."
    gated = apply_constraint_gate(_proceed(text), [_soft("My knee is a bit sore, avoid running.")])

    assert not gated.hard_violations
    assert gated.cautions  # the contradiction is surfaced, not silenced
    # The prescription STAYS (no silent scrub — the inverse-harm guard, ADR 0008 §2).
    assert gated.result.scrubbed_text == text
    assert gated.result.decision is GroundDecision.PROCEED
    assert gated.cautions[0].activity == "run"


@pytest.mark.unit
def test_neutral_activity_does_not_block() -> None:
    """The KEY case: an 'easy swim' against 'no running' is NEUTRAL — it must PROCEED (ADR 0008 §1).

    Proves the floor detects CONTRADICTION, not mere non-support: a prescription a constraint does
    not endorse but does not contradict is published, never over-blocked (the catastrophic
    over-refusal a support-only gate would cause).
    """
    text = "Let's do an easy 30 min swim on Tuesday."
    gated = apply_constraint_gate(_proceed(text), [_hard("No running — doctor's orders.")])

    assert not gated.hard_violations
    assert not gated.cautions
    assert gated.result.scrubbed_text == text
    assert gated.result.decision is GroundDecision.PROCEED


@pytest.mark.unit
def test_german_constraint_gates_english_prescription() -> None:
    """A German 'keine Intervalle' constraint gates an English 'do intervals' prescription (§8)."""
    text = "Do threshold intervals on Wednesday. Otherwise keep it easy this week."
    gated = apply_constraint_gate(_proceed(text), [_hard("Keine Intervalle diese Woche.")])

    assert gated.hard_violations
    assert "intervals" not in gated.result.scrubbed_text.casefold()
    assert "keep it easy" in gated.result.scrubbed_text.casefold()
    assert gated.result.decision is GroundDecision.REGENERATE


@pytest.mark.unit
def test_russian_constraint_gates_english_prescription() -> None:
    """A Russian 'не бегать' constraint gates an English 'go for a run' prescription (§8)."""
    text = "Go for an easy run on Friday. Swim on Saturday instead."
    constraint = _hard("Не бегать пока колено болит.")  # noqa: RUF001 - Russian (RU) constraint text
    gated = apply_constraint_gate(_proceed(text), [constraint])

    assert gated.hard_violations
    assert "run" not in gated.result.scrubbed_text.casefold()
    assert "swim on saturday" in gated.result.scrubbed_text.casefold()
    assert gated.result.decision is GroundDecision.REGENERATE


@pytest.mark.unit
def test_constraint_without_negation_forbids_nothing() -> None:
    """A constraint with no negation cue near the activity forbids nothing (conservative floor)."""
    assert forbidden_activities("I really enjoy running and long rides.") == frozenset()
    # A clear negation does extract the forbidden activity token.
    assert "run" in forbidden_activities("No running for six weeks.")


@pytest.mark.unit
def test_empty_text_or_no_constraints_is_noop() -> None:
    """No constraints, or empty text, leaves the result untouched (the gate is conservative)."""
    result = _proceed("Run intervals tomorrow.")
    assert apply_constraint_gate(result, []).result is result
    empty = GroundingResult(decision=GroundDecision.PROCEED, claims=(), scrubbed_text="  ")
    assert apply_constraint_gate(empty, [_hard("No running.")]).result is empty


@pytest.mark.unit
def test_shadow_mode_detects_but_does_not_apply() -> None:
    """SHADOW mode reports the would-be veto on the counters but ships the UNMODIFIED text (§7)."""
    text = "Let's run 5x4 min hard on Tuesday."
    gate = ConstraintGate(mode=ConstraintMode.SHADOW)
    gated = gate.apply(_proceed(text), [_hard("No running — doctor's orders.")])

    assert gated.hard_violations  # detection still happens (for the shadow counters)
    # But nothing is applied: the text and decision are unchanged.
    assert gated.result.scrubbed_text == text
    assert gated.result.decision is GroundDecision.PROCEED


@pytest.mark.unit
def test_enforce_mode_applies_veto() -> None:
    """ENFORCE mode applies the HARD veto (scrub + downgrade) (§7)."""
    text = "Let's run 5x4 min hard on Tuesday. Keep the rest easy."
    gate = ConstraintGate(mode=ConstraintMode.ENFORCE)
    gated = gate.apply(_proceed(text), [_hard("No running — doctor's orders.")])

    assert "run 5x4 min" not in gated.result.scrubbed_text.casefold()
    assert gated.result.decision is GroundDecision.REGENERATE
