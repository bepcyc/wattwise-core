"""Non-vacuous-grader tests for the eval harness (doc 80 §5).

These tests pin the four clauses doc 80 §5 makes the gate that proves fail-closed
grounding/abstention is real rather than self-reading:

* **QA-EVAL-R2.6 / QA-EVAL-R6** — structured-output conformance is measured by ACTUAL
  declared-schema validation (the production claim-extraction schema), never a hardcoded
  ``schema_valid`` literal: a recorded structured output that violates the declared schema
  reads as a schema failure.
* **QA-EVAL-R2.9 / EVAL-R3** — intent / retrieval-plan accuracy scores the PRODUCTION
  :class:`~wattwise_core.agent.engine_services.ModelPlanner` (the shipped planner driven by
  a recorded structured plan), AND a labelled intent classification — never an in-module
  keyword matcher hand-tuned to the dataset.
* **QA-EVAL-R2.10 / EVAL-R5a** — a no-self-certification suite asserts a model self-claim
  ("this answer is fully grounded/approved") never substitutes for the engine's
  grounding/approval verdict; a self-certified-but-ungrounded case scores zero.
* **QA-EVAL-R8** — per-case token/cost/latency are recorded and the gate fails when the
  declared median cost-per-task or p95 latency budget is exceeded.

Every test is network-free and deterministic (TIER-R1, QA-EVAL-R9). A test is NON-VACUOUS:
mutating the rule it encodes breaks it.
"""

from __future__ import annotations

import pytest

from wattwise_core.eval.budget import BudgetGrade, CostLatencyBudget, grade_budget
from wattwise_core.eval.runner import EvalMode, EvalRunner, list_suites, run_suite
from wattwise_core.eval.self_cert_suite import grade_self_certification
from wattwise_core.eval.suites import grade_intent_plan

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# QA-EVAL-R2.6 — schema_valid comes from REAL declared-schema validation       #
# --------------------------------------------------------------------------- #


async def test_schema_valid_is_measured_not_a_literal() -> None:
    # A case whose recorded structured claim-extraction output VIOLATES the declared
    # production schema MUST read as schema-invalid — proving schema_valid is a measurement,
    # not the hardcoded ``True`` literal QA-EVAL-R2.6 forbids.
    runner = EvalRunner(mode=EvalMode.RECORDED)
    bad_case = {
        "id": "schema-violation",
        "draft_prose": "Your fitness is steady.",
        # ``value`` must be a number per the declared _ClaimSchema; a string violates it.
        "structured_output": {"claims": [{"kind": "number", "value": "not-a-number"}]},
        "candidate_claims": [{"kind": "number", "text": "x", "metric": "ctl", "value": 1.0}],
        "evidence": {"metrics": {"ctl": 1.0}},
    }
    outcome = await runner.run_case(bad_case, tolerance=0.01)
    assert outcome.schema_valid is False


async def test_schema_valid_true_for_conforming_recorded_output() -> None:
    # A well-formed recorded structured output validates against the declared schema.
    runner = EvalRunner(mode=EvalMode.RECORDED)
    good_case = {
        "id": "schema-ok",
        "draft_prose": "Your fitness is steady.",
        "structured_output": {"claims": [{"kind": "number", "metric": "ctl", "value": 1.0}]},
        "candidate_claims": [{"kind": "number", "text": "ctl 1.0", "metric": "ctl", "value": 1.0}],
        "evidence": {"metrics": {"ctl": 1.0}},
    }
    outcome = await runner.run_case(good_case, tolerance=0.01)
    assert outcome.schema_valid is True


# --------------------------------------------------------------------------- #
# QA-EVAL-R2.9 / EVAL-R3 — score the PRODUCTION planner + a labelled intent     #
# --------------------------------------------------------------------------- #


async def test_intent_plan_scores_production_planner() -> None:
    # The gated path (no explicit ``predicted``) MUST drive the production ModelPlanner over
    # each case's recorded structured plan and score its EMITTED capability requests, plus
    # the labelled intent classification — never the keyword reference matcher.
    grade = await grade_intent_plan()
    assert grade.precision >= 0.9
    assert grade.recall >= 0.9
    assert grade.intent_accuracy >= 0.9
    assert grade.passed


async def test_intent_plan_intent_miss_fails_the_gate() -> None:
    # Teeth: a classified intent that disagrees with the label fails the gate even when the
    # capability plan is perfect — the intent term is real, not a stand-in.
    grade = await grade_intent_plan(predicted_intents={"intent-weekly-load": "decoupling"})
    assert not grade.passed


# --------------------------------------------------------------------------- #
# QA-EVAL-R2.10 / EVAL-R5a — no self-certification substitutes for the verdict  #
# --------------------------------------------------------------------------- #


async def test_self_certification_suite_passes_and_is_listed() -> None:
    assert "self_certification" in list_suites()
    grade = await grade_self_certification()
    assert grade.total >= 1
    assert grade.passed
    assert grade.failures == ()
    card = await run_suite("self_certification")
    assert card.passed


# --------------------------------------------------------------------------- #
# QA-EVAL-R8 — per-case token/cost/latency recorded + budget gate              #
# --------------------------------------------------------------------------- #


def test_budget_gate_passes_within_declared_budgets() -> None:
    budget = CostLatencyBudget(
        median_cost_usd=0.01, p95_latency_ms=2000.0, cost_per_1k_tokens_usd=0.0002
    )
    samples = (
        {"case_id": "a", "total_tokens": 100, "cost_usd": 0.002, "latency_ms": 500.0},
        {"case_id": "b", "total_tokens": 120, "cost_usd": 0.003, "latency_ms": 700.0},
        {"case_id": "c", "total_tokens": 90, "cost_usd": 0.001, "latency_ms": 600.0},
    )
    grade = grade_budget(samples, budget)
    assert grade.passed
    assert grade.median_cost_usd == 0.002
    assert grade.total_tokens == 310


def test_budget_gate_fails_when_median_cost_exceeds_budget() -> None:
    # Teeth: median cost above the declared budget fails the gate (QA-EVAL-R8).
    budget = CostLatencyBudget(
        median_cost_usd=0.001, p95_latency_ms=10000.0, cost_per_1k_tokens_usd=0.0002
    )
    samples = (
        {"case_id": "a", "total_tokens": 1, "cost_usd": 0.01, "latency_ms": 1.0},
        {"case_id": "b", "total_tokens": 1, "cost_usd": 0.02, "latency_ms": 1.0},
    )
    grade = grade_budget(samples, budget)
    assert not grade.passed


def test_budget_gate_fails_when_p95_latency_exceeds_budget() -> None:
    # Teeth: p95 latency above the declared budget fails the gate (QA-EVAL-R8).
    budget = CostLatencyBudget(
        median_cost_usd=1.0, p95_latency_ms=100.0, cost_per_1k_tokens_usd=0.0002
    )
    samples = tuple(
        {"case_id": str(i), "total_tokens": 1, "cost_usd": 0.0, "latency_ms": 1000.0}
        for i in range(20)
    )
    grade = grade_budget(samples, budget)
    assert not grade.passed
    assert isinstance(grade, BudgetGrade)


def test_budget_requires_caller_supplied_price_no_code_default() -> None:
    """CFG-R1a: the token price has NO code-baked default, so a budget cannot be built
    without it — the price is always a caller-supplied number (loaded from config), never a
    silent free-of-charge 0.0 fallback; a supplied price is the one cost_for uses."""
    with pytest.raises(TypeError):
        CostLatencyBudget(median_cost_usd=0.01, p95_latency_ms=2000.0)  # type: ignore[call-arg]
    # And a supplied price is the one used: pricing N tokens is N/1000 * price, not zero.
    budget = CostLatencyBudget(
        median_cost_usd=0.01, p95_latency_ms=2000.0, cost_per_1k_tokens_usd=0.0002
    )
    assert budget.cost_for(1000) == pytest.approx(0.0002)
    assert budget.cost_for(2000) == pytest.approx(0.0004)
