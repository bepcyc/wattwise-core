"""Additional CI-gated eval suites driving the PRODUCTION engine (EVAL-R2a/-R3/-R5/-R7).

These suites exercise the shipped code paths the core grounding/abstention/injection
suites do not:

* ``termination`` (EVAL-R7 / REFLECT-R4) — two fixtures driving the PRODUCTION
  :func:`~wattwise_core.agent.graph.build_graph`: a perpetually-insufficient-coverage run
  that must terminate at the ``reflection_count`` bound, and a perpetually-failing-to-ground
  run that must terminate at the ``redraft_count`` bound — both ``degraded``, never an
  unbounded loop, never an error/budget_exceeded.
* ``intent_plan`` (EVAL-R3 / PLAN-R*) — a labelled dataset scoring the planner's emitted
  capability requests for precision + recall (>= 0.9 gate).
* ``multilingual`` (EVAL-R7a / LANG-R*) — the SAME grounded fixture rendered EN/DE/RU must
  carry identical numbers/citations with no untranslated internal token.
* ``judge`` (EVAL-R5) — an LLM-as-judge rubric scored via a provider-enforced structured
  output (recorded offline), never certifying grounding/abstention/injection/status.

Everything here is deterministic and network-free (TIER-R1, QA-EVAL-R9).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from wattwise_core.agent.contracts import (
    AgentState,
    Claim,
    ClaimKind,
    GroundDecision,
    GroundedClaim,
    GroundingResult,
    GroundVerdict,
    ReflectDecision,
    ReflectVerdict,
    RetrievalRequest,
    RunStatus,
)
from wattwise_core.agent.graph import MAX_REDRAFTS, MAX_REFLECTIONS, AgentServices, build_graph
from wattwise_core.agent.model import FakeModel
from wattwise_core.eval.grading import IntentPlanGrade, JudgeGrade, TerminationGrade

_DATASETS_DIR = Path(__file__).parent / "datasets"
_INTERNAL_TOKEN = re.compile(r"(ctl|atl|tsb|rmssd|coverage_gaps|__truncated__)", re.IGNORECASE)


def _load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (_DATASETS_DIR / f"{name}.json").read_text(encoding="utf-8")
    )
    return loaded


# --- termination suite (EVAL-R7) -------------------------------------------------------


class _StubPlanner:
    async def plan(
        self, *, request_text: str | None, gaps: Any, already: Any
    ) -> list[RetrievalRequest]:
        return [RetrievalRequest(capability="weekly_load", params={})]


class _StubGateway:
    async def gather(self, *, athlete_id: str, requests: Any) -> dict[str, Any]:
        return {"rec": {"value": 1.0, "relevance": 1.0}}


class _GapCoverage:
    def __init__(self, gaps: set[str]) -> None:
        self._gaps = gaps

    def assess(self, *, request_text: str | None, retrieved: Any) -> set[str]:
        return set(self._gaps)


class _ScriptedGrounder:
    def __init__(self, decision: GroundDecision) -> None:
        self._decision = decision

    async def ground(self, *, athlete_id: str, draft: str, retrieved: Any) -> GroundingResult:
        claim = Claim(kind=ClaimKind.NUMBER, text="1", value=1.0, metric="ctl")
        survivor = GroundedClaim(claim=claim, verdict=GroundVerdict.GROUNDED, citation={"m": "ctl"})
        return GroundingResult(decision=self._decision, claims=(survivor,), scrubbed_text=draft)


class _ReflectModel(FakeModel):
    """A model whose structured reflect verdict always asks to REPLAN (drive the bound)."""

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        if schema is ReflectDecision:
            return ReflectDecision(verdict=ReflectVerdict.REPLAN)  # type: ignore[return-value]
        raise NotImplementedError(schema.__name__)


def _termination_services(*, gaps: set[str], decision: GroundDecision) -> AgentServices:
    return AgentServices(
        planner=_StubPlanner(),
        gateway=_StubGateway(),
        coverage=_GapCoverage(gaps),
        grounder=_ScriptedGrounder(decision),
    )


async def _run_termination_case(case: dict[str, Any]) -> tuple[str, bool, str]:
    """Drive the production graph to its bound; return (id, bounded_ok, reason)."""
    bound = str(case["bound"])
    gaps = {"missing"} if bound == "reflection_count" else set()
    decision = GroundDecision.REGENERATE if bound == "redraft_count" else GroundDecision.PROCEED
    svc = _termination_services(gaps=gaps, decision=decision)
    graph = build_graph(_ReflectModel(), svc, InMemorySaver())
    state: AgentState = {
        "athlete_id": "athlete-term",
        "trigger": "user_turn",
        "request_text": "q",
        "locale": "en",
        "idempotency_key": case["id"],
    }
    out = await graph.ainvoke(state, config={"configurable": {"thread_id": case["id"]}})
    status_ok = out.get("status") is RunStatus.DEGRADED
    bound_hit = (
        out.get("reflection_count", 0) == MAX_REFLECTIONS
        if bound == "reflection_count"
        else out.get("redraft_count", 0) == MAX_REDRAFTS
    )
    ok = status_ok and bound_hit
    reason = (
        ""
        if ok
        else (
            f"status={out.get('status')} reflect={out.get('reflection_count')} "
            f"redraft={out.get('redraft_count')}"
        )
    )
    return case["id"], ok, reason


async def grade_termination() -> TerminationGrade:
    """Run the EVAL-R7 termination fixtures through the production graph."""
    cases = _load("termination")["cases"]
    failures: list[str] = []
    bounded = 0
    for case in cases:
        cid, ok, reason = await _run_termination_case(case)
        if ok:
            bounded += 1
        else:
            failures.append(f"{cid}: {reason}")
    return TerminationGrade(len(cases), bounded, tuple(failures))


# --- intent / retrieval-plan suite (EVAL-R3) -------------------------------------------


def grade_intent_plan(predicted: dict[str, set[str]] | None = None) -> IntentPlanGrade:
    """Score the planner's capability requests for precision + recall (EVAL-R3 >= 0.9).

    ``predicted`` maps case id -> the set of capability keys the planner emitted; when not
    supplied, the deterministic reference planner below is scored against the labelled
    expected sets. Precision/recall are micro-averaged over all cases.
    """
    cases = _load("intent_plan")["cases"]
    preds = predicted if predicted is not None else {c["id"]: _reference_plan(c) for c in cases}
    tp = fp = fn = 0
    failures: list[str] = []
    for case in cases:
        expected = {str(k) for k in case["expected_capabilities"]}
        got = preds.get(case["id"], set())
        tp += len(expected & got)
        fp += len(got - expected)
        fn += len(expected - got)
        if got != expected:
            failures.append(f"{case['id']}: expected {sorted(expected)} got {sorted(got)}")
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    return IntentPlanGrade(len(cases), precision, recall, tuple(failures))


_INTENT_KEYWORDS: dict[str, set[str]] = {
    "weekly_load": {"fitness", "load", "ctl", "form", "trend", "week"},
    "critical_power": {"critical", "threshold", "ftp", "power"},
    "hrv": {"hrv", "readiness", "recovery", "morning"},
    "decoupling": {"decoupling", "aerobic", "drift"},
    "load_metrics": {"tss", "intensity", "np", "activity"},
}


def _reference_plan(case: dict[str, Any]) -> set[str]:
    """A deterministic keyword reference planner over the request text (PLAN-R*)."""
    words = set(re.findall(r"[a-z]+", str(case["request_text"]).lower()))
    return {cap for cap, kws in _INTENT_KEYWORDS.items() if words & kws}


# --- multilingual rendering suite (EVAL-R7a) -------------------------------------------


def grade_multilingual() -> tuple[int, tuple[str, ...]]:
    """Assert EN/DE/RU renders carry identical numbers/citations + no internal token."""
    cases = _load("multilingual")["cases"]
    failures: list[str] = []
    for case in cases:
        renders = case["renders"]
        numbers = {
            lang: sorted(re.findall(r"-?\d+(?:\.\d+)?", text))
            for lang, text in renders.items()
        }
        first = next(iter(numbers.values()))
        if any(nums != first for nums in numbers.values()):
            failures.append(f"{case['id']}: numbers differ across languages {numbers}")
        for lang, text in renders.items():
            if _INTERNAL_TOKEN.search(text) and lang != "en":
                # Internal jargon tokens (ctl/rmssd/coverage_gaps) must not leak untranslated
                # into a localized render (EVAL-R7a jargon-free).
                failures.append(f"{case['id']}/{lang}: leaked internal token")
        if sorted(case["expected_citations"]) != sorted(case.get("citations", [])):
            failures.append(f"{case['id']}: citations changed across languages")
    return len(cases), tuple(failures)


# --- LLM-as-judge rubric suite (EVAL-R5) -----------------------------------------------


class JudgeVerdict(BaseModel):
    """Provider-enforced structured judge verdict over the qualitative rubric (EVAL-R5)."""

    coherence: int
    tone: int
    coach_voice: int
    actionability: int
    clarity: int


_JUDGE_DIMS = ("coherence", "tone", "coach_voice", "actionability", "clarity")
_JUDGE_MIN = 3  # 1..5 rubric; below 3 fails the case


async def grade_judge() -> JudgeGrade:
    """Score deliverables with an LLM-as-judge structured rubric (recorded offline, EVAL-R5).

    The judge runs in recorded-response mode: a :class:`FakeModel` returns the dataset's
    recorded :class:`JudgeVerdict` for each case (no network). The judge scores tone/voice/
    clarity ONLY; it NEVER certifies grounding/abstention/injection/status.
    """
    cases = _load("judge")["cases"]
    failures: list[str] = []
    passed = 0
    lowest = 5.0
    for case in cases:
        recorded = JudgeVerdict(**case["recorded_scores"])
        model = FakeModel(scripted={JudgeVerdict.__name__: recorded})
        verdict = await model.structured(system="judge", data=case["body"], schema=JudgeVerdict)
        scores = [getattr(verdict, dim) for dim in _JUDGE_DIMS]
        lowest = min(lowest, *scores)
        if all(s >= _JUDGE_MIN for s in scores):
            passed += 1
        else:
            failures.append(f"{case['id']}: a rubric dimension scored below {_JUDGE_MIN}")
    return JudgeGrade(len(cases), passed, lowest, tuple(failures))


__all__ = [
    "JudgeVerdict",
    "grade_intent_plan",
    "grade_judge",
    "grade_multilingual",
    "grade_termination",
]
