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
from wattwise_core.eval.passk import PassK


class EvalMode(StrEnum):
    """Run mode (QA-EVAL-R9). OSS PR gate uses ``RECORDED`` (deterministic, free)."""

    RECORDED = "recorded"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Machine-readable aggregate metrics across one suite (EVAL-R9).

    ``pass_k`` carries the QA-EVAL-R10 reliability result (degenerate k=1 in the
    deterministic recorded tier; k>1 only on the env-gated live nightly leg). A SAFETY
    suite's pass^k MUST be 100%, so :attr:`passed` folds the pass^k certificate in.
    """

    suite: str
    dataset_version: str
    mode: EvalMode
    total_cases: int
    grades: SuiteGrades
    pass_k: PassK | None = None

    @property
    def passed(self) -> bool:
        """Suite passes iff every grade passes AND (for a safety suite) pass^k = 100%."""
        return self.grades.passed and (self.pass_k is None or self.pass_k.passed)

    def to_jsonable(self) -> dict[str, Any]:
        card: dict[str, Any] = {
            "suite": self.suite,
            "dataset_version": self.dataset_version,
            "mode": self.mode.value,
            "total_cases": self.total_cases,
            "passed": self.passed,
            **_grades_jsonable(self.grades),
        }
        if self.pass_k is not None:
            card["pass_k"] = {
                "k": self.pass_k.k,
                # ``all_pass_rate`` is the pass^k all-trials certificate (1.0 iff EVERY trial
                # passed) — the metric the non-regression baseline tracks (QA-EVAL-R10). The
                # per-trial single-shot rate is kept as ``trial_pass_rate`` for trend visibility
                # only; the baseline must NOT track it (a flaky 4/5 safety suite reads 0.8 there).
                "all_pass_rate": self.pass_k.all_pass_rate,
                "trial_pass_rate": self.pass_k.trial_pass_rate,
                "pass_k": self.pass_k.pass_k,
                "is_safety": self.pass_k.is_safety,
                "passed": self.pass_k.passed,
            }
        return card


def _grades_jsonable(g: SuiteGrades) -> dict[str, Any]:
    """The per-gate blobs (a module fn so :meth:`Scorecard.to_jsonable` stays small)."""
    return {
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
        "readiness": {
            "consistency_rate": g.readiness.consistency_rate,
            "voice_rate": g.readiness.voice_rate,
            "passed": g.readiness.passed,
            "failures": list(g.readiness.failures),
        },
        "plan": {
            "grounding_rate": g.plan.grounding_rate,
            "progression_rate": g.plan.progression_rate,
            "consistency_rate": g.plan.consistency_rate,
            "passed": g.plan.passed,
            "failures": list(g.plan.failures),
        },
        "voice": {
            "rate": g.voice.rate,
            "passed": g.voice.passed,
            "failures": list(g.voice.failures),
        },
    }


__all__ = ["EvalMode", "Scorecard"]
