"""Intent / retrieval-plan accuracy suite (EVAL-R3 / QA-EVAL-R2.9).

This suite scores the PRODUCTION retrieval planner — never an in-module keyword matcher
(which PLAN-R1 forbids). Each labelled case fixes the correct intent classification AND the
expected retrieval plan; the grader drives the shipped
:class:`~wattwise_core.agent.engine_services.ModelPlanner` over the case's recorded
structured plan (the cassette), scores its EMITTED capability requests for micro-averaged
precision AND recall, and scores the intent classification derived from the emitted plan
against the labelled ``expected_intent`` — all gated at EVAL-R3's >= 0.9 floor.

Network-free and deterministic (TIER-R1, QA-EVAL-R9): the planner is driven by a scripted
:class:`~wattwise_core.agent.model.FakeModel` returning the recorded ``_PlanSchema``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wattwise_core.agent.engine_services import ModelPlanner, _PlanSchema
from wattwise_core.agent.model import FakeModel
from wattwise_core.eval.grading import IntentPlanGrade

_DATASETS_DIR = Path(__file__).parent / "datasets"


def _load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (_DATASETS_DIR / f"{name}.json").read_text(encoding="utf-8")
    )
    return loaded


async def _production_planner_plan(case: dict[str, Any]) -> list[str]:
    """Drive the PRODUCTION ModelPlanner over a case's recorded structured plan (EVAL-R3).

    The recorded ``plan_capabilities`` is the verbatim structured plan a real provider call
    returned (the cassette), fed to the SHIPPED :class:`ModelPlanner` via a scripted
    :class:`FakeModel` returning the closed ``_PlanSchema`` — so the gate scores the
    production planner's EMITTED capability requests (PLAN-R1/R2), never an in-module
    keyword matcher (which PLAN-R1 forbids). The emitted order is preserved.
    """
    schema = _PlanSchema(capabilities=list(case.get("plan_capabilities", [])))
    model = FakeModel(scripted={_PlanSchema.__name__: schema})
    planner = ModelPlanner(model)
    requests = await planner.plan(request_text=str(case["request_text"]), gaps=(), already=())
    return [req.capability for req in requests]


def _intent_of(emitted: list[str]) -> str:
    """The intent classification derived from the planner's emitted plan (QA-EVAL-R2.9).

    The intent IS the planner's leading capability request: the OSS planner realizes intent
    AS the structured retrieval plan (GRAPH-R2.1 — there is no separate intent model call),
    so the classification is read from the production planner's emitted plan, never a keyword
    match on the user text. An empty plan classifies as ``""`` (a mis-plan, scored a miss).
    """
    return emitted[0] if emitted else ""


async def grade_intent_plan(
    predicted: dict[str, set[str]] | None = None,
    predicted_intents: dict[str, str] | None = None,
) -> IntentPlanGrade:
    """Score the PRODUCTION planner's plan + intent against labels (EVAL-R3 / QA-EVAL-R2.9).

    For each labelled case the gated path drives the production :class:`ModelPlanner` over
    the case's recorded structured plan and scores (a) its EMITTED capability requests for
    micro-averaged precision AND recall against ``expected_capabilities`` and (b) the intent
    classification derived from the emitted plan against ``expected_intent`` — all gated at
    EVAL-R3's >= 0.9 floor. ``predicted`` / ``predicted_intents`` override the planner output
    per case (used by the teeth tests to prove a mis-plan / mis-intent fails the gate); when
    absent the production planner output is scored.
    """
    cases = _load("intent_plan")["cases"]
    tp = fp = fn = 0
    intent_hits = 0
    failures: list[str] = []
    for case in cases:
        cid = case["id"]
        emitted = await _production_planner_plan(case)
        # When ``predicted`` is supplied it is AUTHORITATIVE per case (a case absent from it
        # is treated as an empty plan), so the teeth tests can force a mis-plan; otherwise the
        # production planner's emitted requests are scored (QA-EVAL-R2.9 / EVAL-R3).
        got = predicted.get(cid, set()) if predicted is not None else set(emitted)
        expected = {str(k) for k in case["expected_capabilities"]}
        tp += len(expected & got)
        fp += len(got - expected)
        fn += len(expected - got)
        if got != expected:
            failures.append(f"{cid}: expected plan {sorted(expected)} got {sorted(got)}")
        intent = (
            predicted_intents[cid]
            if predicted_intents is not None and cid in predicted_intents
            else _intent_of(emitted)
        )
        if intent == str(case["expected_intent"]):
            intent_hits += 1
        else:
            failures.append(f"{cid}: expected intent {case['expected_intent']!r} got {intent!r}")
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    intent_accuracy = 1.0 if not cases else intent_hits / len(cases)
    return IntentPlanGrade(len(cases), precision, recall, tuple(failures), intent_accuracy)


__all__ = ["grade_intent_plan"]
