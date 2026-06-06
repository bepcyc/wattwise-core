"""The eval run-mode + the machine-readable aggregate scorecard (EVAL-R9, QA-EVAL-R9).

Factored out of :mod:`wattwise_core.eval.runner` so each module stays under the size
ceiling (QUAL-R9). :class:`Scorecard` is the EVAL-R9 machine-readable artifact the CI gate
writes; :class:`EvalMode` fixes the OSS offline suite to recorded-response mode (TIER-R1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from wattwise_core.eval.grading import SuiteGrades


class EvalMode(StrEnum):
    """Run mode (QA-EVAL-R9). OSS PR gate uses ``RECORDED`` (deterministic, free)."""

    RECORDED = "recorded"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Machine-readable aggregate metrics across one suite (EVAL-R9)."""

    suite: str
    dataset_version: str
    mode: EvalMode
    total_cases: int
    grades: SuiteGrades

    @property
    def passed(self) -> bool:
        return self.grades.passed

    def to_jsonable(self) -> dict[str, Any]:
        g = self.grades
        return {
            "suite": self.suite,
            "dataset_version": self.dataset_version,
            "mode": self.mode.value,
            "total_cases": self.total_cases,
            "passed": self.passed,
            "grounding": {
                "faithfulness": g.grounding.faithfulness,
                "fabricated": g.grounding.fabricated,
                "passed": g.grounding.passed,
                "failures": list(g.grounding.failures),
            },
            "abstention": {
                "rate": g.abstention.rate,
                "fabrications": g.abstention.fabrications,
                "passed": g.abstention.passed,
                "failures": list(g.abstention.failures),
            },
            "schema": {
                "rate": g.schema.rate,
                "passed": g.schema.passed,
                "failures": list(g.schema.failures),
            },
            "injection": {
                "rate": g.injection.rate,
                "passed": g.injection.passed,
                "failures": list(g.injection.failures),
            },
            "termination": {
                "rate": g.termination.rate,
                "passed": g.termination.passed,
                "failures": list(g.termination.failures),
            },
            "intent_plan": {
                "precision": g.intent_plan.precision,
                "recall": g.intent_plan.recall,
                "passed": g.intent_plan.passed,
                "failures": list(g.intent_plan.failures),
            },
            "judge": {
                "passed_cases": g.judge.passed_cases,
                "passed": g.judge.passed,
                "failures": list(g.judge.failures),
            },
        }


__all__ = ["EvalMode", "Scorecard"]
