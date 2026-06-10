"""Engine-driven eval suites: the dispatch that runs the production graph/planner/judge.

These suites exercise the shipped agent code paths the claim-level grounding/abstention/
injection suites do not (EVAL-R2a/-R3/-R5/-R7, QA-EVAL-R2.*): termination, reflection-
termination, intent/retrieval-plan accuracy (QA-EVAL-R2.9), multilingual, the LLM-as-judge
rubric, readiness/form, the multi-day plan, voice-liveness, and no-self-certification
(QA-EVAL-R2.10). Each builds its typed grades and folds in the QA-EVAL-R8 cost/latency
budget grade. Factored out of :mod:`wattwise_core.eval.runner` so each module stays under
the size ceiling (QUAL-R9).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from wattwise_core.eval import budget as budget_mod
from wattwise_core.eval import reflection_suite, self_cert_suite, suites, voice_suite
from wattwise_core.eval.grading import SuiteGrades, TerminationGrade
from wattwise_core.eval.passk import degenerate_pass_k
from wattwise_core.eval.plan_suite import grade_plan
from wattwise_core.eval.scorecard import EvalMode, Scorecard

_DATASETS_DIR = Path(__file__).parent / "datasets"

# Suites driven by the production graph/planner/judge (EVAL-R2a/-R3/-R5/-R7), in addition to
# the claim-level grounding/abstention/injection suites the runner owns. ``plan``
# (QA-EVAL-R2.5), ``voice`` (QA-EVAL-R2.12), ``reflection_termination`` (QA-EVAL-R2.11) and
# ``self_certification`` (QA-EVAL-R2.10) are the coach-capability + safety suites in the gate.
ENGINE_SUITES: frozenset[str] = frozenset(
    {
        "termination",
        "reflection_termination",
        "intent_plan",
        "multilingual",
        "judge",
        "readiness",
        "plan",
        "voice",
        "self_certification",
    }
)
# The dataset stem each engine suite loads for its dataset_version / case count (some suite
# names differ from their datafile stem, e.g. ``voice`` -> ``voice_liveness``).
_ENGINE_SUITE_DATASET: dict[str, str] = {"voice": "voice_liveness"}


def _sync_engine_grades(name: str) -> SuiteGrades | None:
    """The SYNC engine grades, or ``None`` when ``name`` needs the async path below."""
    if name == "readiness":
        return SuiteGrades(readiness=suites.grade_readiness())
    if name == "plan":
        return SuiteGrades(plan=grade_plan())
    if name == "multilingual":
        # A parity check expressed via the all-or-nothing termination grade shape.
        total, failures = suites.grade_multilingual()
        return SuiteGrades(termination=TerminationGrade(total, total - len(failures), failures))
    return None


async def _async_intent_plan() -> SuiteGrades:
    # QA-EVAL-R2.9 / EVAL-R3: the gated path drives the PRODUCTION ModelPlanner and scores
    # its emitted capability requests AND the labelled intent classification.
    return SuiteGrades(intent_plan=await suites.grade_intent_plan())


async def _async_self_cert() -> SuiteGrades:
    return SuiteGrades(self_cert=await self_cert_suite.grade_self_certification())


async def _async_voice() -> SuiteGrades:
    return SuiteGrades(voice=await voice_suite.grade_voice())


async def _async_reflection_termination() -> SuiteGrades:
    return SuiteGrades(termination=await reflection_suite.grade_reflection_termination())


async def _async_judge() -> SuiteGrades:
    return SuiteGrades(judge=await suites.grade_judge())


async def _async_termination() -> SuiteGrades:
    return SuiteGrades(termination=await suites.grade_termination())


# The async engine-suite graders (the sync ones are in ``_sync_engine_grades``).
_ASYNC_ENGINE_GRADERS: dict[str, Callable[[], Awaitable[SuiteGrades]]] = {
    "intent_plan": _async_intent_plan,
    "self_certification": _async_self_cert,
    "voice": _async_voice,
    "reflection_termination": _async_reflection_termination,
    "judge": _async_judge,
}


async def _engine_grades(name: str) -> SuiteGrades:
    """Compute the typed grades for one engine suite (EVAL-R2a/-R3/-R5/-R7, QA-EVAL-R2.*)."""
    sync = _sync_engine_grades(name)
    if sync is not None:
        return sync
    builder = _ASYNC_ENGINE_GRADERS.get(name, _async_termination)
    return await builder()


def _load_raw(name: str) -> dict[str, Any]:
    path = _DATASETS_DIR / f"{name}.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


async def run_engine_suite(name: str, mode: EvalMode) -> Scorecard:
    """Run an EVAL-R2a/-R3/-R5/-R7 suite that drives the production engine (+ QA-EVAL-R8)."""
    raw = _load_raw(_ENGINE_SUITE_DATASET.get(name, name))
    grades = await _engine_grades(name)
    cases = raw.get("cases", [])
    return Scorecard(
        suite=name,
        dataset_version=str(raw.get("dataset_version", "1.0.0")),
        mode=mode,
        total_cases=len(cases),
        grades=budget_mod.with_budget(grades, cases),
        budget_samples=budget_mod.record_samples(cases),
        pass_k=degenerate_pass_k(name, grades.passed),
    )


__all__ = ["ENGINE_SUITES", "run_engine_suite"]
