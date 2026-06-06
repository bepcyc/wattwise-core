"""Offline eval-harness tests: run the suites, assert the QA-EVAL-R6 thresholds.

Cited requirements: EVAL-R1 / TIER-R1 (offline, no network — recorded-response mode);
EVAL-R2 / EVAL-R4 / QA-EVAL-R2.1 (grounding faithfulness, planted hallucinations
scrubbed); QA-EVAL-R2.2 (abstention / fail-closed); QA-EVAL-R2.6 (schema conformance);
QA-EVAL-R6 (the hard gate thresholds — grounding >= 99% with zero fabricated, abstention
100%, schema 100%); OUTCOME-R5 (graders are deterministic, never model self-assertion);
EVAL-R9 (machine-readable scorecard). The prompt-injection corpus has its own marked
suite in ``tests/inject/test_injection.py`` (INJ-R2).

Every test here is network-free and deterministic: it loads checked-in datasets and runs
the reference pipeline with the offline :class:`FakeModel` only.
"""

from __future__ import annotations

import pytest

from wattwise_core.eval.grading import (
    ABSTENTION_MIN_RATE,
    GROUNDING_MIN_FAITHFULNESS,
    SCHEMA_MIN_RATE,
    grade_abstention,
    grade_grounding,
    grade_schema,
)
from wattwise_core.eval.runner import (
    EvalMode,
    EvalRunner,
    RunnerOutcome,
    load_dataset,
    run_suite,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Dataset loading — versioned, checked-in (QA-EVAL-R1, EVAL-R8)               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["grounding", "abstention", "injection"])
def test_dataset_is_versioned_and_nonempty(name: str) -> None:
    dataset = load_dataset(name)
    assert dataset.version, "dataset MUST carry a version (QA-EVAL-R1, EVAL-R8)"
    assert dataset.cases, "dataset MUST carry at least one case"
    ids = [c["id"] for c in dataset.cases]
    assert len(ids) == len(set(ids)), "case ids MUST be unique"


def test_runner_rejects_live_mode_in_oss_suite() -> None:
    # The OSS offline suite is recorded-response only (QA-EVAL-R9, TIER-R1).
    with pytest.raises(ValueError, match="recorded-response"):
        EvalRunner(mode=EvalMode.LIVE)


# --------------------------------------------------------------------------- #
# Grounding / faithfulness suite (EVAL-R4, QA-EVAL-R2.1) — gate >= 99%, 0 fab #
# --------------------------------------------------------------------------- #


async def test_grounding_suite_meets_threshold() -> None:
    card = await run_suite("grounding")
    assert card.grades.grounding.passed
    assert card.grades.grounding.faithfulness >= GROUNDING_MIN_FAITHFULNESS
    assert card.grades.grounding.fabricated == 0, "zero fabricated numbers (QA-EVAL-R6)"
    assert card.grades.grounding.failures == ()


async def test_grounding_scrubs_every_planted_hallucination() -> None:
    # Each grounding case's planted hallucination (invented number, wrong value,
    # non-allow-listed URL) MUST be scrubbed; valid claims MUST survive (EVAL-R4).
    dataset = load_dataset("grounding")
    runner = EvalRunner()
    for case in dataset.cases:
        outcome = await runner.run_case(case, tolerance=dataset.tolerance)
        assert outcome.expected_scrubbed <= outcome.actually_scrubbed, case["id"]
        assert outcome.every_surfaced_number_canonical, case["id"]
        assert not outcome.published_non_canonical, case["id"]


async def test_grounding_memory_non_substitution_cites_live_value() -> None:
    # EVAL-R2a / MEM-R3: a stale memory value MUST NOT substitute for the live canonical
    # number; the surfaced number must equal the live evidence value, not memory.
    dataset = load_dataset("grounding")
    case = next(c for c in dataset.cases if c["id"] == "grounding-memory-non-substitution")
    runner = EvalRunner()
    outcome = await runner.run_case(case, tolerance=dataset.tolerance)
    assert outcome.every_surfaced_number_canonical
    expectation = case["expected"]["must_cite_live_not_memory"]
    live = case["evidence"]["metrics"][expectation["metric"]]
    assert live == expectation["live_value"]
    assert expectation["live_value"] != expectation["memory_value"]


def test_grounding_grader_flags_a_fabricated_leak() -> None:
    # A planted leak MUST be caught by the deterministic grader, never silently passed.
    leaked = RunnerOutcome(
        case_id="synthetic-leak",
        suite="grounding",
        abstained=False,
        schema_valid=True,
        every_surfaced_number_canonical=False,
        published_non_canonical=frozenset({"ctl"}),
        expected_scrubbed=frozenset({"ctl"}),
        actually_scrubbed=frozenset(),
    )
    grade = grade_grounding([leaked])
    assert not grade.passed
    assert grade.fabricated == 1
    assert grade.failures


# --------------------------------------------------------------------------- #
# Abstention / fail-closed suite (QA-EVAL-R2.2) — gate 100%, single fab fails #
# --------------------------------------------------------------------------- #


async def test_abstention_suite_meets_threshold() -> None:
    card = await run_suite("abstention")
    assert card.grades.abstention.passed
    assert card.grades.abstention.rate >= ABSTENTION_MIN_RATE
    assert card.grades.abstention.fabrications == 0
    assert card.grades.abstention.failures == ()


async def test_abstention_never_publishes_a_number_when_data_absent() -> None:
    dataset = load_dataset("abstention")
    runner = EvalRunner()
    for case in dataset.cases:
        outcome = await runner.run_case(case, tolerance=dataset.tolerance)
        assert outcome.abstained, f"{case['id']} MUST abstain (QA-EVAL-R2.2)"
        assert not outcome.published_non_canonical, f"{case['id']} fabricated a number"


def test_abstention_grader_trips_on_single_fabrication() -> None:
    fabricated = RunnerOutcome(
        case_id="synthetic-fab",
        suite="abstention",
        abstained=False,
        schema_valid=True,
        every_surfaced_number_canonical=False,
        published_non_canonical=frozenset({"hrv_rmssd_ms"}),
        expected_scrubbed=frozenset(),
        actually_scrubbed=frozenset(),
    )
    grade = grade_abstention([fabricated])
    assert not grade.passed, "a single confident fabrication MUST trip the gate"
    assert grade.fabrications == 1


# --------------------------------------------------------------------------- #
# Structured-output conformance (QA-EVAL-R2.6) — gate 100%                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["grounding", "abstention", "injection"])
async def test_schema_conformance_is_total(name: str) -> None:
    card = await run_suite(name)
    assert card.grades.schema.rate >= SCHEMA_MIN_RATE
    assert card.grades.schema.passed


def test_schema_grader_flags_invalid_verdict() -> None:
    invalid = RunnerOutcome(
        case_id="synthetic-schema",
        suite="grounding",
        abstained=False,
        schema_valid=False,
        every_surfaced_number_canonical=True,
        published_non_canonical=frozenset(),
        expected_scrubbed=frozenset(),
        actually_scrubbed=frozenset(),
    )
    grade = grade_schema([invalid])
    assert not grade.passed
    assert grade.failures


# --------------------------------------------------------------------------- #
# Aggregate scorecard (EVAL-R9) + whole-suite gate                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["grounding", "abstention", "injection"])
async def test_suite_scorecard_passes_and_is_jsonable(name: str) -> None:
    card = await run_suite(name)
    assert card.passed, f"{name} suite MUST clear the QA-EVAL-R6 gate"
    blob = card.to_jsonable()
    # Machine-readable artifact for trend tracking (EVAL-R9).
    assert blob["suite"] == name
    assert blob["mode"] == "recorded"
    assert blob["passed"] is True
    assert blob["total_cases"] == len(load_dataset(name).cases)
    for key in ("grounding", "abstention", "schema", "injection"):
        assert "passed" in blob[key]


async def test_runner_is_deterministic_across_runs() -> None:
    # The same dataset grades identically on every run — the suite is a stable CI gate
    # (EVAL-R1, QA-EVAL-R9). Determinism is what lets recorded-mode gate PRs.
    first = (await run_suite("grounding")).to_jsonable()
    second = (await run_suite("grounding")).to_jsonable()
    assert first == second
