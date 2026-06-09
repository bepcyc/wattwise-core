"""Gated reflection/redraft bounded-termination eval suite (QA-EVAL-R2.11 / EVAL-R7).

The richer THREE-loop termination catalog (``datasets/reflection_termination.json``)
promoted into the GATED scorecard. Where the smaller ``termination`` suite drives two
bounds, this suite drives the production graph (:func:`wattwise_core.agent.graph.build_graph`)
through ALL THREE permitted recovery cycles (GRAPH-R3) and certifies each terminates
GRACEFULLY at its monotonic bound with a ``DEGRADED`` status — NEVER an unbounded loop,
NEVER an error, and NEVER ``budget_exceeded`` (reserved for a refused cost admission,
COST-R4; the cost gate always admits in these fixtures, so a ``budget_exceeded`` here is a
real defect, not a designed exit — the central invariant of this suite):

* **F-COVERAGE-BOUND** — ``assess_coverage -> reflect -> plan_retrieval`` (gaps never close)
  spends the reflection budget to ``MAX_REFLECTIONS``.
* **F-REDRAFT-BOUND** — ``ground -> compose`` (grounder always REGENERATEs) spends the
  redraft budget to ``MAX_REDRAFTS``.
* **F-GROUNDLOOP-TERMINATES** — ``ground -> reflect -> plan_retrieval`` (grounder always
  REPLANs) spends the reflection budget to ``MAX_REFLECTIONS``.

The grade reuses :class:`~wattwise_core.eval.grading.TerminationGrade` (a bounded-rate 100%
gate). Each fixture's per-case ``reflect_verdict`` / ``ground_decision`` / ``coverage_gaps``
drive the matching pathological loop. Deterministic and network-free (TIER-R1, QA-EVAL-R9):
every collaborator is an in-suite fake satisfying the public seams, exercised only through
``build_graph`` and the typed state (ARCH-R21) with langgraph's in-memory saver.

Cited requirements: QA-EVAL-R2.11, EVAL-R7, REFLECT-R4, QA-EVAL-R6, OUTCOME-R1/-R3,
GRAPH-R3/-R5, COST-R4.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
from wattwise_core.agent.graph import (
    MAX_REDRAFTS,
    MAX_REFLECTIONS,
    AgentServices,
    build_graph,
)
from wattwise_core.eval.grading import TerminationGrade

_DATASETS_DIR = Path(__file__).parent / "datasets"
# A finite recursion ceiling comfortably above the longest LEGAL path: an unbounded
# recovery cycle would raise GraphRecursionError here instead of settling at finalize, so
# reaching finalize at all is itself proof of termination.
_RECURSION_LIMIT = 50


def _load() -> dict[str, Any]:
    """Load the checked-in QA-EVAL-R2.11 reflection-termination catalog (no network)."""
    raw: dict[str, Any] = json.loads(
        (_DATASETS_DIR / "reflection_termination.json").read_text(encoding="utf-8")
    )
    return raw


class _ReflectModel:
    """Deterministic ``ChatModel`` whose §6 reflect verdict is scripted (REFLECT-R2)."""

    def __init__(self, *, reflect_verdict: ReflectVerdict) -> None:
        self.compose_calls = 0
        self._reflect_verdict = reflect_verdict

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        if schema is ReflectDecision:
            return ReflectDecision(verdict=self._reflect_verdict)  # type: ignore[return-value]
        raise NotImplementedError(f"no scripted structured output for {schema.__name__}")

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_calls += 1
        return f"draft#{self.compose_calls}"


class _StubPlanner:
    def __init__(self) -> None:
        self.calls = 0

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        self.calls += 1
        return [RetrievalRequest(capability="weekly_load", params={"n": self.calls})]


class _StubGateway:
    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        return {f"rec:{r.capability}": {"value": 1.0, "relevance": 1.0} for r in requests}


class _GapCoverage:
    """Coverage assessor reporting a FIXED open-gap set on every pass (drives the bound)."""

    def __init__(self, gaps: set[str]) -> None:
        self._gaps = gaps

    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        return set(self._gaps)


class _ScriptedGrounder:
    """Grounder returning a FIXED aggregate decision on every pass (drives the bound)."""

    def __init__(self, decision: GroundDecision) -> None:
        self._decision = decision

    async def ground(
        self, *, athlete_id: str, draft: str, retrieved: Mapping[str, Any]
    ) -> GroundingResult:
        claim = Claim(kind=ClaimKind.NUMBER, text="1", value=1.0, metric="ctl")
        survivor = GroundedClaim(
            claim=claim, verdict=GroundVerdict.GROUNDED, citation={"metric": "ctl"}
        )
        return GroundingResult(decision=self._decision, claims=(survivor,), scrubbed_text=draft)


def _services(case: dict[str, Any]) -> AgentServices:
    """Build the injected services that force ONE pathological loop forever (per case)."""
    return AgentServices(
        planner=_StubPlanner(),
        gateway=_StubGateway(),
        coverage=_GapCoverage(set(case.get("coverage_gaps", []))),
        grounder=_ScriptedGrounder(GroundDecision(case["ground_decision"])),
    )


def _budget(bound: str) -> int:
    """The configured budget for the named monotonic bound (REFLECT-R4)."""
    return MAX_REFLECTIONS if bound == "reflection_count" else MAX_REDRAFTS


async def _run_case(case: dict[str, Any]) -> tuple[str, bool, str]:
    """Drive one fixture through the PRODUCTION graph; return (id, bounded_ok, reason).

    Bounded-ok iff the run ended ``DEGRADED`` (never ``budget_exceeded``, never a forbidden
    status) with the named monotonic counter sitting EXACTLY at its budget.
    """
    cid = str(case["id"])
    model = _ReflectModel(reflect_verdict=ReflectVerdict(case["reflect_verdict"]))
    graph = build_graph(model, _services(case), InMemorySaver())
    state = AgentState(
        athlete_id="athlete-term",
        trigger="user_turn",
        request_text="how is my fitness trending?",
        locale="en",
        idempotency_key=cid,
    )
    out = await graph.ainvoke(
        state, config={"configurable": {"thread_id": cid}, "recursion_limit": _RECURSION_LIMIT}
    )
    status = out.get("status")
    bound = str(case["expected_bound_counter"])
    forbidden = set(case.get("forbidden_statuses", ()))
    not_budget = status is not RunStatus.BUDGET_EXCEEDED
    degraded = status is RunStatus.DEGRADED
    at_budget = int(out.get(bound, 0)) == _budget(bound)
    not_forbidden = status is not None and status.value not in forbidden
    ok = not_budget and degraded and at_budget and not_forbidden
    reason = (
        ""
        if ok
        else (
            f"status={status} reflect={out.get('reflection_count')} "
            f"redraft={out.get('redraft_count')} bound={bound}"
        )
    )
    return cid, ok, reason


async def grade_reflection_termination() -> TerminationGrade:
    """Run the THREE-loop QA-EVAL-R2.11 catalog through the production graph (100% gate)."""
    cases = _load()["cases"]
    failures: list[str] = []
    bounded = 0
    for case in cases:
        cid, ok, reason = await _run_case(case)
        if ok:
            bounded += 1
        else:
            failures.append(f"{cid}: {reason}")
    return TerminationGrade(len(cases), bounded, tuple(failures))


__all__ = ["grade_reflection_termination"]
