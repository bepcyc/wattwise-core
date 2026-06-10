"""Weekly load-review suite is real and non-vacuous (QA-EVAL-R2.3 / QA-EVAL-R6).

The suite must be CI-gated (listed + passing on the committed dataset) AND its grader
must actually certify numeric consistency against the SHIPPED PMC oracle: a stated
metric, trend, notable-session, or summary number that drifts from canonical analytics
reads as a failure. Each mutation test below applies the exact bug the grader claims to
catch and asserts the grade flips.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from wattwise_core.eval.load_review_suite import (
    LoadReviewGrade,
    _case_failures,
    grade_load_review,
)
from wattwise_core.eval.runner import EvalMode, list_suites, run_suite

pytestmark = pytest.mark.unit

_DATASET = Path(__file__).parents[2] / "src/wattwise_core/eval/datasets/load_review.json"


def _case(idx: int = 0) -> dict[str, Any]:
    payload = json.loads(_DATASET.read_text(encoding="utf-8"))
    return copy.deepcopy(payload["cases"][idx])


def test_load_review_suite_is_listed_and_passes() -> None:
    """load_review is a gated suite (QA-EVAL-R2.3) and the committed dataset passes."""
    assert "load_review" in list_suites()
    grade = grade_load_review()
    assert isinstance(grade, LoadReviewGrade)
    assert grade.total >= 3
    assert grade.passed and grade.failures == ()


async def test_load_review_scorecard_reports_metric() -> None:
    """The suite scorecard carries the gated consistency metric for the baseline."""
    card = await run_suite("load_review", mode=EvalMode.RECORDED)
    blob = card.to_jsonable()
    assert blob["load_review"]["consistency_rate"] == 1.0
    assert card.passed


def test_misstated_ctl_fails() -> None:
    """A review whose stated CTL drifts from the recomputed canonical PMC fails."""
    case = _case()
    case["review"]["ctl"] = float(case["review"]["ctl"]) + 5.0
    assert any("stated ctl" in f for f in _case_failures(case))


def test_misstated_form_fails() -> None:
    """A review whose stated form contradicts canonical CTL/ATL fails."""
    case = _case()
    case["review"]["form"] = float(case["review"]["form"]) + 4.0
    assert any("stated form" in f for f in _case_failures(case))


def test_wrong_trend_direction_fails() -> None:
    """A stated trend opposite to the recomputed CTL movement fails."""
    case = _case()  # the rising build week
    case["review"]["trend"] = "falling"
    assert any("trend" in f for f in _case_failures(case))


def test_invented_notable_session_fails() -> None:
    """Naming a light day as the notable session (missing the heaviest) fails."""
    case = _case(1)  # recovery week; heaviest is 2026-05-30
    case["review"]["notable_sessions"] = ["2026-05-26"]
    failures = _case_failures(case)
    assert any("heaviest session" in f for f in failures)


def test_fabricated_summary_number_fails() -> None:
    """A summary surfacing a number absent from canonical analytics fails (R2.1-style)."""
    case = _case()
    case["summary_text"] += " Your threshold power improved to 311 this week."
    assert any("not canonical" in f for f in _case_failures(case))
