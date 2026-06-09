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

import json
from pathlib import Path

import pytest

from wattwise_core.agent.readiness_deliverable import HRV_UNAVAILABLE_CLAUSE
from wattwise_core.eval.__main__ import main as cli_main
from wattwise_core.eval.baseline import (
    BASELINE_FORMAT_VERSION,
    BASELINE_PATH,
    build_baseline,
    compare_to_baseline,
    load_baseline,
    write_baseline,
)
from wattwise_core.eval.grading import (
    ABSTENTION_MIN_RATE,
    GROUNDING_MIN_FAITHFULNESS,
    SCHEMA_MIN_RATE,
    ReadinessGrade,
    grade_abstention,
    grade_grounding,
    grade_schema,
)
from wattwise_core.eval.runner import (
    EvalMode,
    EvalRunner,
    RunnerOutcome,
    list_suites,
    load_dataset,
    run_suite,
)
from wattwise_core.eval.suites import (
    _consistency_failure as readiness_consistency_failure,
)
from wattwise_core.eval.suites import (
    _is_abstain as readiness_is_abstain,
)
from wattwise_core.eval.suites import (
    _voice_failures as readiness_voice_failures,
)
from wattwise_core.eval.suites import (
    grade_intent_plan,
    grade_judge,
    grade_readiness,
    grade_termination,
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
    # EVAL-R2a / MEM-R3: a STALE memory value is fed into the pipeline as a competing
    # candidate; the PRODUCTION grounder must surface the LIVE canonical value and scrub
    # the memory value (proven through the pipeline, not by literal JSON checks).
    dataset = load_dataset("grounding")
    case = next(c for c in dataset.cases if c["id"] == "grounding-memory-non-substitution")
    runner = EvalRunner()
    outcome = await runner.run_case(case, tolerance=dataset.tolerance)
    # No fabricated/non-canonical number leaked: the memory value (55.0) did NOT survive.
    assert outcome.every_surfaced_number_canonical
    assert not outcome.published_non_canonical
    expectation = case["expected"]["must_cite_live_not_memory"]
    live = case["evidence"]["metrics"][expectation["metric"]]
    assert live == expectation["live_value"]
    assert expectation["live_value"] != expectation["memory_value"]
    # The memory metric was actually scrubbed by the production grounder (memory@value).
    memory_key = f"{expectation['metric']}@{float(expectation['memory_value'])}"
    assert memory_key in outcome.actually_scrubbed


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


async def test_unavailable_metric_is_never_surfaced_as_a_number() -> None:
    # GROUND-R7: a metric whose canonical computation is marked ``unavailable`` MUST be
    # stated as unavailable (a placeholder/zero is forbidden). The production grounder
    # surfaces NO number for any unavailable metric in the abstention dataset.
    dataset = load_dataset("abstention")
    runner = EvalRunner()
    for case in dataset.cases:
        unavailable = set((case.get("evidence", {}) or {}).get("unavailable", {}))
        if not unavailable:
            continue
        outcome = await runner.run_case(case, tolerance=dataset.tolerance)
        # No unavailable metric leaked a (non-canonical) number, and the run abstained.
        assert outcome.every_surfaced_number_canonical, case["id"]
        assert not outcome.published_non_canonical, case["id"]
        assert outcome.abstained, case["id"]


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


# --------------------------------------------------------------------------- #
# Production-grounder gate (EVAL-R4 / GROUND-R8) + 100% faithfulness floor     #
# --------------------------------------------------------------------------- #


def test_grounding_threshold_is_absolute_100_percent() -> None:
    # EVAL-R4 (corrected): the grounding faithfulness gate is the binding 100% mandate,
    # not a 99% band — any planted-hallucination leak OR dropped-valid claim fails CI.
    assert GROUNDING_MIN_FAITHFULNESS == 1.0


async def test_grounding_suite_runs_the_production_grounder() -> None:
    # GROUND-R8 / EVAL-R4: the gate exercises the SHIPPED grounder, not a re-implementation.
    # A 99.x% (single-leak) suite must now FAIL the absolute gate.
    card = await run_suite("grounding")
    assert card.grades.grounding.passed
    assert card.grades.grounding.faithfulness == 1.0


# --------------------------------------------------------------------------- #
# New CI-gated suites: termination, intent_plan, multilingual, judge          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["termination", "intent_plan", "multilingual", "judge"])
async def test_engine_suite_passes_and_is_listed(name: str) -> None:
    assert name in list_suites()
    card = await run_suite(name)
    assert card.passed, f"{name} suite MUST clear its gate"
    assert card.to_jsonable()["passed"] is True


async def test_termination_suite_drives_both_bounds() -> None:
    # EVAL-R7 / REFLECT-R4: both the reflection_count and redraft_count bounds terminate
    # the production graph at a DEGRADED outcome (no unbounded loop, no error).
    grade = await grade_termination()
    assert grade.total == 2
    assert grade.passed
    assert grade.failures == ()


async def test_intent_plan_gate_at_least_point_nine() -> None:
    # EVAL-R3: the planner's emitted capability requests gate at precision AND recall >= 0.9.
    grade = grade_intent_plan()
    assert grade.precision >= 0.9
    assert grade.recall >= 0.9
    assert grade.passed


def test_intent_plan_gate_fails_below_threshold() -> None:
    # A planner that mis-plans every case must FAIL the >= 0.9 gate (no silent pass).
    grade = grade_intent_plan(predicted={})  # planner emitted nothing -> recall 0
    assert not grade.passed


async def test_judge_never_certifies_grounding() -> None:
    # EVAL-R5: the judge is a qualitative rubric (structured output, recorded offline); it
    # scores tone/voice/clarity only and never gates grounding/abstention/injection/status.
    grade = await grade_judge()
    assert grade.total == 2
    assert grade.passed


# --------------------------------------------------------------------------- #
# Readiness / form suite (QA-EVAL-R2.4 / COACH-R7) — deterministic 100% gates  #
# --------------------------------------------------------------------------- #


def test_grade_readiness_passes() -> None:
    # QA-EVAL-R2.4 + COACH-R7: every "good" fixture (each band, the HRV-suppressed nudge,
    # HRV-present-normal, HRV-unavailable, and the form-unavailable abstain) clears BOTH
    # deterministic gates — verdict-direction consistency AND voice-liveness at 100%.
    grade = grade_readiness()
    assert grade.passed
    assert grade.failures == ()
    assert grade.consistency_rate == 1.0
    assert grade.voice_rate == 1.0


def test_grade_readiness_rejects_inconsistent_verdict() -> None:
    # Teeth (QA-EVAL-R2.4 / EVAL-R5): a deep-negative form (-30 => REST band) delivered as
    # a hard "go" MUST be flagged inconsistent by the deterministic certificate — the code
    # decides the band, never the LLM.
    case = {
        "id": "teeth-inconsistent",
        "form": -30.0,
        "hrv_rmssd": None,
        "hrv_baseline": None,
        "delivered_verdict": "go",
        "summary_text": "You should take it easy today and recover.",
        "expects_hrv_unavailable_statement": False,
    }
    reason = readiness_consistency_failure(case)
    assert reason is not None
    assert "inconsistent" in reason


def test_grade_readiness_rejects_number_led_summary() -> None:
    # Teeth (COACH-R7 / QA-EVAL-R2.12): a summary whose first sentence starts with a number
    # demotes the STATE behind a digit — voice-liveness MUST reject it.
    case = {"id": "teeth-number-led", "expects_hrv_unavailable_statement": False}
    failures = readiness_voice_failures(case, "12 is your form today, so push hard.")
    assert failures
    assert any("number-led" in f for f in failures)


def test_readiness_grade_fails_on_nonempty_failures_despite_perfect_rates() -> None:
    # Teeth (FIX 1): a recorded failure MUST fail the gate even when both rates read 1.0.
    # A case can append a real failure (e.g. an abstain case that wrongly delivered a
    # verdict) without lowering consistency_rate/voice_rate, so `.passed` MUST additionally
    # require `failures == ()`.
    grade = ReadinessGrade(
        total=1,
        non_abstain=1,
        consistent=1,
        voice_ok=1,
        failures=("synthetic: abstain case (form null) delivered a verdict",),
    )
    assert grade.consistency_rate == 1.0
    assert grade.voice_rate == 1.0
    assert grade.passed is False, "non-empty failures MUST fail the readiness gate (FIX 1)"


def test_readiness_grade_passes_only_when_failures_empty() -> None:
    # Control for FIX 1: identical rates with NO recorded failures still passes.
    grade = ReadinessGrade(total=1, non_abstain=1, consistent=1, voice_ok=1, failures=())
    assert grade.passed is True


def test_readiness_present_form_null_verdict_is_a_failure() -> None:
    # Teeth (FIX 4): a case WITH a form but NO delivered verdict is NON-abstain and MUST be
    # recorded as a failure ("form present but no verdict delivered"), never silently
    # classified abstain and skipped past readiness_consistent.
    case = {
        "id": "teeth-form-no-verdict",
        "form": 5.0,
        "hrv_rmssd": None,
        "hrv_baseline": None,
        "delivered_verdict": None,
        "summary_text": "You're in a steady place today, so train as planned.",
        "expects_hrv_unavailable_statement": False,
    }
    assert not readiness_is_abstain(case), "form present => NON-abstain (FIX 4)"
    reason = readiness_consistency_failure(case)
    assert reason is not None
    assert "no verdict delivered" in reason


def test_readiness_consistency_failure_keeps_deep_negative_go_teeth() -> None:
    # FIX 4 must not weaken the existing deep-negative-form delivered-"go" teeth: a -30 form
    # (REST band) delivered as a hard "go" is still flagged inconsistent.
    case = {
        "id": "teeth-deep-negative-go",
        "form": -30.0,
        "hrv_rmssd": None,
        "hrv_baseline": None,
        "delivered_verdict": "go",
        "summary_text": "You should take it easy today and recover.",
        "expects_hrv_unavailable_statement": False,
    }
    reason = readiness_consistency_failure(case)
    assert reason is not None
    assert "inconsistent" in reason


def test_readiness_prod_hrv_clause_satisfies_voice_check() -> None:
    # Teeth (FIX 7): the EXACT prod HRV-unavailable clause MUST satisfy the voice grader's
    # HRV-unavailable check, so the gate matches a LIVE narration, not only hand fixtures.
    case = {"id": "teeth-prod-hrv-clause", "expects_hrv_unavailable_statement": True}
    summary = (
        "You're in a steady place today, so train as planned. "
        f"{HRV_UNAVAILABLE_CLAUSE}"
    )
    failures = readiness_voice_failures(case, summary)
    assert failures == [], f"prod HRV clause must clear the voice check, got {failures}"


def test_readiness_positive_hrv_prose_is_not_a_false_unavailable() -> None:
    # Teeth: a summary that mentions HRV POSITIVELY (and happens to say "from your form")
    # MUST NOT satisfy the HRV-unavailable voice check — absence must be STATED, not implied
    # (GROUND-R7). Guards the broadened-regex false-PASS hole the re-verify panel surfaced.
    case = {"id": "teeth-positive-hrv", "expects_hrv_unavailable_statement": True}
    summary = (
        "You're carrying some fatigue, so ease off. "
        "Your HRV is strong, momentum comes from your form."
    )
    failures = readiness_voice_failures(case, summary)
    assert failures, "positive-HRV prose must FAIL the must-state-HRV-unavailable check"


def test_full_scorecard_lists_every_gated_suite() -> None:
    assert set(list_suites()) == {
        "grounding",
        "abstention",
        "injection",
        "termination",
        "intent_plan",
        "multilingual",
        "judge",
        "readiness",
    }


def test_eval_cli_run_gates_green(tmp_path: Path) -> None:
    # EVAL-R1: `python -m wattwise_core.eval run` returns 0 when every suite clears its gate
    # and writes the machine-readable scorecard artifact (EVAL-R9). Synchronous: the CLI
    # owns its own event loop via asyncio.run.
    out = tmp_path / "scorecard.json"
    code = cli_main(["run", "--mode=recorded", f"--scorecard={out}"])
    assert code == 0
    blob = json.loads(out.read_text())
    assert blob["passed"] is True
    assert {s["suite"] for s in blob["suites"]} >= {"termination", "intent_plan", "judge"}


# --------------------------------------------------------------------------- #
# Non-regression baseline (QA-EVAL-R7) — versioned floor + safety regression   #
# --------------------------------------------------------------------------- #


async def _all_cards() -> list[object]:
    """Run every catalogued suite once (the cards the baseline check compares)."""
    return [await run_suite(name, mode=EvalMode.RECORDED) for name in list_suites()]


def test_committed_baseline_exists_and_is_versioned() -> None:
    # QA-EVAL-R7: a versioned baseline artifact MUST be committed beside the eval engine.
    assert BASELINE_PATH.exists(), "the baseline scorecard MUST be committed (QA-EVAL-R7)"
    doc = load_baseline()
    assert doc is not None
    assert doc["baseline_format_version"] == BASELINE_FORMAT_VERSION
    # Every CI-gated suite is represented in the committed baseline.
    assert set(doc["suites"]) == set(list_suites())
    # The committed baseline records a PASSING run (never enshrines a failure as the floor).
    assert all(s["passed"] for s in doc["suites"].values())


async def test_current_run_does_not_regress_committed_baseline() -> None:
    # QA-EVAL-R7: a clean run MUST be >= the committed baseline on every tracked metric.
    cards = await _all_cards()
    report = compare_to_baseline(cards)
    assert report.baseline_present
    assert report.passed, report.summary()
    assert report.regressions == ()


async def test_safety_suite_regression_fails_even_when_absolute_is_high(
    tmp_path: Path,
) -> None:
    # QA-EVAL-R7 teeth: inflate the grounding faithfulness baseline above 1.0. The current
    # run still scores a perfectly-passing 1.0 (absolute gate green), yet the non-regression
    # gate MUST fail it as a SAFETY-SUITE regression — proving the two gates are independent.
    cards = await _all_cards()
    doc = build_baseline(cards)
    doc["suites"]["grounding"]["metrics"]["grounding.faithfulness"] = 1.5
    path = tmp_path / "inflated.json"
    path.write_text(json.dumps(doc))
    report = compare_to_baseline(cards, path=path)
    assert not report.passed
    assert report.has_safety_regression
    assert any(r.suite == "grounding" and r.is_safety for r in report.regressions)


async def test_non_safety_regression_also_fails(tmp_path: Path) -> None:
    # A drop below baseline on a non-safety metric (intent-plan precision) is still a hard
    # fail, but NOT flagged as a safety regression.
    cards = await _all_cards()
    doc = build_baseline(cards)
    doc["suites"]["intent_plan"]["metrics"]["intent_plan.precision"] = 1.25
    path = tmp_path / "inflated.json"
    path.write_text(json.dumps(doc))
    report = compare_to_baseline(cards, path=path)
    assert not report.passed
    assert not report.has_safety_regression
    assert any(r.suite == "intent_plan" for r in report.regressions)


async def test_missing_baseline_does_not_fail_the_gate(tmp_path: Path) -> None:
    # A first run before any baseline is seeded MUST NOT fail on the baseline's absence —
    # there is nothing to regress against; the operator is told to seed one.
    cards = await _all_cards()
    report = compare_to_baseline(cards, path=tmp_path / "nope.json")
    assert not report.baseline_present
    assert report.passed
    assert "eval-update-baseline" in report.summary()


async def test_new_suite_since_baseline_is_reported_not_a_regression(
    tmp_path: Path,
) -> None:
    # QA-EVAL-R7 + I7 interop: a suite added since the baseline (a new I7 dataset) has no
    # prior value, so it is REPORTED as new and never counted as a regression — the gate
    # stays green until the next deliberate `update-baseline`.
    cards = await _all_cards()
    doc = build_baseline(cards)
    del doc["suites"]["readiness"]
    path = tmp_path / "missing-readiness.json"
    path.write_text(json.dumps(doc))
    report = compare_to_baseline(cards, path=path)
    assert report.passed
    assert "readiness" in report.new_suites


async def test_update_baseline_round_trips(tmp_path: Path) -> None:
    # `update-baseline` writes a document a fresh run reads back as a non-regression.
    cards = await _all_cards()
    path = tmp_path / "baseline.json"
    write_baseline(cards, path=path)
    assert path.exists()
    report = compare_to_baseline(cards, path=path)
    assert report.baseline_present
    assert report.passed


def test_update_baseline_cli_rewrites_and_run_stays_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # QA-EVAL-R12(c): `update-baseline` actually rewrites the artifact (no longer a no-op),
    # and a subsequent `run` (with its non-regression gate active) stays green against it.
    target = tmp_path / "baseline-scorecard.json"
    monkeypatch.setattr("wattwise_core.eval.baseline.BASELINE_PATH", target)
    assert cli_main(["update-baseline"]) == 0
    assert target.exists(), "update-baseline MUST write the baseline artifact (not a no-op)"
    out = tmp_path / "scorecard.json"
    assert cli_main(["run", "--mode=recorded", f"--scorecard={out}"]) == 0
