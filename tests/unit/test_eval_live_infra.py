"""Eval-infrastructure contracts: cassettes, live INFRA_ERROR, baseline tag (QA-EVAL-R12).

Pins the three QA-EVAL-R12 clauses: (a) the stale-cassette static check fails when a
dataset's recorded metadata is missing or out of sync with the pinned model/prompt
content; (b) live-mode failures are CLASSIFIED — infrastructure lands as the distinct
``INFRA_ERROR`` status with a configured max rate that alerts and blocks promotion,
never as a silent pass or a quality FAIL; (c) the baseline records a release tag,
refuses to advance over a dirty live run, and a recorded tag absent from git history
fails the check.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from wattwise_core.eval.baseline import (
    live_run_blocks_baseline,
    tag_in_git_history,
)
from wattwise_core.eval.live import (
    LiveRunReport,
    LiveStatus,
    LiveSuiteResult,
    classify_infra,
    classify_infra_text,
)
from wattwise_core.eval.recorded_meta import (
    RECORDED_WITH_KEY,
    current_pins,
    verify_recorded_datasets,
)

pytestmark = pytest.mark.unit

_DATASETS = Path(__file__).parents[2] / "src/wattwise_core/eval/datasets"


# --- (a) stale-cassette static check ------------------------------------------------
def test_committed_datasets_carry_current_pins() -> None:
    """Every committed dataset's cassette metadata matches the pinned model + prompt
    fingerprint — the QA-EVAL-R12(a) static check is green on the committed tree."""
    assert verify_recorded_datasets() == ()


def test_stale_model_pin_fails_the_check(tmp_path: Path) -> None:
    """A dataset recorded under a DIFFERENT model than the pinned one is stale."""
    target = tmp_path / "grounding.json"
    shutil.copy(_DATASETS / "grounding.json", target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload[RECORDED_WITH_KEY]["model"] = "some/other-model"
    target.write_text(json.dumps(payload), encoding="utf-8")
    failures = verify_recorded_datasets(tmp_path)
    assert len(failures) == 1 and "stale" in failures[0]


def test_missing_metadata_fails_the_check(tmp_path: Path) -> None:
    """A dataset with NO recorded_with block fails — metadata is mandatory."""
    target = tmp_path / "abstention.json"
    payload = json.loads((_DATASETS / "abstention.json").read_text(encoding="utf-8"))
    payload.pop(RECORDED_WITH_KEY)
    target.write_text(json.dumps(payload), encoding="utf-8")
    failures = verify_recorded_datasets(tmp_path)
    assert len(failures) == 1 and "missing" in failures[0]


def test_current_pins_carry_model_and_prompt_digest() -> None:
    """The pins cover the model id AND a prompt-content digest (a prompt edit changes it)."""
    pins = current_pins()
    assert set(pins) == {"model", "prompt_sha256"}
    assert pins["model"] and len(pins["prompt_sha256"]) == 64


# --- (b) live-mode INFRA_ERROR classification ----------------------------------------
def test_infra_exceptions_classify_as_infrastructure() -> None:
    """Timeouts, connection drops, and rate-limit (429) responses are INFRA, not quality."""

    class _RateLimited(Exception):
        status_code = 429

    assert classify_infra(TimeoutError("provider timed out"))
    assert classify_infra(ConnectionError("refused"))
    assert classify_infra(_RateLimited())


def test_quality_exceptions_are_not_infrastructure() -> None:
    """A grading/assertion failure stays a QUALITY failure — never excused as infra."""
    assert not classify_infra(AssertionError("grounded number mismatch"))
    assert not classify_infra(ValueError("bad dataset"))


def test_infra_text_classification() -> None:
    """The live smoke's failure TEXT classifies on the same infra taxonomy."""
    assert classify_infra_text("httpx.ConnectTimeout: connection timed out")
    assert classify_infra_text("Error code: 429 - rate limit exceeded")
    assert not classify_infra_text("AssertionError: answer was not grounded")


def test_infra_rate_over_max_blocks_promotion_and_alerts() -> None:
    """Exceeding the configured max INFRA_ERROR rate alerts and blocks promotion."""
    results = (
        LiveSuiteResult("a", LiveStatus.PASS),
        LiveSuiteResult("b", LiveStatus.INFRA_ERROR, detail="429"),
    )
    report = LiveRunReport(results, max_infra_error_rate=0.1)
    assert report.infra_error_rate == 0.5
    assert report.infra_blocked
    assert not report.quality_failed  # the infra blip is NOT a quality regression
    assert any("BLOCKED" in line for line in report.alert_lines())


def test_infra_never_counts_as_pass_and_bars_clean() -> None:
    """An in-budget INFRA_ERROR avoids the blocking alert but still bars a clean run."""
    results = tuple(
        [LiveSuiteResult(f"s{i}", LiveStatus.PASS) for i in range(19)]
        + [LiveSuiteResult("infra", LiveStatus.INFRA_ERROR)]
    )
    report = LiveRunReport(results, max_infra_error_rate=0.1)
    assert not report.infra_blocked
    assert not report.clean  # baseline advancement still requires a NO-infra run


def test_quality_failure_alerts_as_regression() -> None:
    """A genuine quality FAIL alerts as a regression and fails the run."""
    report = LiveRunReport(
        (LiveSuiteResult("grounding", LiveStatus.FAIL, detail="fabricated"),),
        max_infra_error_rate=0.1,
    )
    assert report.quality_failed == ("grounding",)
    assert any("regression" in line for line in report.alert_lines())


def test_max_infra_rate_is_loaded_config() -> None:
    """The max infra rate resolves from [agent.eval] config (CFG-R1a), not a literal."""
    report = LiveRunReport.from_results(())
    assert report.max_infra_error_rate == pytest.approx(0.1)


# --- (c) baseline tag + clean-live advancement gate ----------------------------------
def test_bogus_tag_is_not_in_git_history() -> None:
    """A release tag absent from git history is detected (QA-EVAL-R12(c) CI check)."""
    assert not tag_in_git_history("v999.999.999-nonexistent")


def test_dirty_live_artifact_blocks_baseline_advancement(tmp_path: Path) -> None:
    """update-baseline refuses while the live artifact records infra/quality failures."""
    artifact = tmp_path / "eval-live-scorecard.json"
    artifact.write_text(json.dumps({"clean": False}), encoding="utf-8")
    reason = live_run_blocks_baseline(live_scorecard=artifact)
    assert reason is not None and "refusing to advance" in reason


def test_clean_live_artifact_allows_baseline_advancement(tmp_path: Path) -> None:
    """A clean live run (zero quality + zero infra failures) allows advancement."""
    artifact = tmp_path / "eval-live-scorecard.json"
    artifact.write_text(json.dumps({"clean": True}), encoding="utf-8")
    assert live_run_blocks_baseline(live_scorecard=artifact) is None
