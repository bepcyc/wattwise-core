"""Reflection/redraft bounded-termination eval fixtures (D-P2 step 9).

Cited requirements: QA-EVAL-R2.11 (the reflection-termination fixture catalog —
versioned, checked-in); EVAL-R7 / REFLECT-R4 (the bounded monotonic recovery counters
terminate every pathological loop GRACEFULLY); QA-EVAL-R6 (100% of termination fixtures
terminate at their bound and degrade); OUTCOME-R3 / GRAPH-R5 (a pathological run settles
at ``finalize`` with a single typed status, never a ``GraphRecursionError`` and never an
unbounded loop); OUTCOME-R1 (``finalize`` emits exactly one :class:`RunStatus`).

These three fixtures (F-COVERAGE-BOUND, F-REDRAFT-BOUND, F-GROUNDLOOP-TERMINATES) drive the
PRODUCTION graph through its only public surface (:func:`wattwise_core.agent.graph.build_graph`)
into each of the three permitted recovery cycles (GRAPH-R3):

* F-COVERAGE-BOUND — ``assess_coverage -> reflect -> plan_retrieval`` (gaps never close).
* F-REDRAFT-BOUND — ``ground -> compose`` (the grounder always REGENERATEs).
* F-GROUNDLOOP-TERMINATES — ``ground -> reflect -> plan_retrieval`` (the grounder always REPLANs).

Each must terminate at its matching monotonic bound (``reflection_count``/``redraft_count``)
with a :data:`RunStatus.DEGRADED` outcome, and MUST NOT yield :data:`RunStatus.BUDGET_EXCEEDED`
(reserved for a refused cost admission, COST-R4 — the cost gate ALWAYS admits in these
fixtures, so a budget_exceeded outcome would be a real defect masquerading as a designed
exit). The catalog is checked-in and versioned (QA-EVAL-R2.11); this module is the gate.

Offline and self-contained: every collaborator is an in-test fake satisfying the public
:mod:`wattwise_core.agent.seams` protocols, exercised purely through ``build_graph`` and
the typed state — no sibling agent in-flight module is imported (ARCH-R21). The
checkpointer is langgraph's in-memory saver (a bounded-termination assertion does not
need a durable saver; durable resume has its own fixtures).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
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

pytestmark = pytest.mark.unit

_CATALOG = Path(__file__).parents[2] / "src" / "wattwise_core" / "eval" / "datasets"


def _load_catalog() -> dict[str, Any]:
    """Load the checked-in QA-EVAL-R2.11 reflection-termination catalog (no network)."""
    raw: dict[str, Any] = json.loads(
        (_CATALOG / "reflection_termination.json").read_text(encoding="utf-8")
    )
    return raw


# --- fakes (satisfy the public seams only; mirror tests/unit/test_graph.py) -------------


class _ReflectModel:
    """Deterministic ``ChatModel`` whose §6 reflect verdict is scripted (REFLECT-R2).

    ``compose`` is counted so a redraft loop's extra draft passes are observable; the
    default reflect verdict is REPLAN so the coverage/replan cycles exercise to their
    budget rather than short-circuiting via ``give_up_gracefully``.
    """

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
        self.calls = 0

    async def ground(
        self,
        *,
        athlete_id: str,
        draft: str,
        retrieved: Mapping[str, Any],
        request_text: str | None = None,
        active_constraints: object = None,
    ) -> GroundingResult:
        self.calls += 1
        claim = Claim(kind=ClaimKind.NUMBER, text="1", value=1.0, metric="ctl")
        survivor = GroundedClaim(
            claim=claim, verdict=GroundVerdict.GROUNDED, citation={"metric": "ctl"}
        )
        return GroundingResult(decision=self._decision, claims=(survivor,), scrubbed_text=draft)


def _services(case: dict[str, Any]) -> tuple[_ReflectModel, AgentServices]:
    """Build the injected services that force ONE pathological loop forever (per case)."""
    model = _ReflectModel(reflect_verdict=ReflectVerdict(case["reflect_verdict"]))
    svc = AgentServices(
        planner=_StubPlanner(),
        gateway=_StubGateway(),
        coverage=_GapCoverage(set(case.get("coverage_gaps", []))),
        grounder=_ScriptedGrounder(GroundDecision(case["ground_decision"])),
    )
    return model, svc


def _input() -> AgentState:
    return AgentState(
        athlete_id="athlete-term",
        trigger="user_turn",
        request_text="how is my fitness trending?",
        locale="en",
        idempotency_key="idem-term",
    )


def _config(thread: str) -> RunnableConfig:
    # A finite recursion_limit comfortably above the longest LEGAL path: if any recovery
    # cycle were unbounded the run would raise GraphRecursionError here instead of
    # settling at finalize, so reaching finalize at all is itself proof of termination.
    return {"configurable": {"thread_id": thread}, "recursion_limit": 50}


def _bound_counter(out: Mapping[str, Any], bound: str) -> int:
    """Read the monotonic counter the fixture says must sit at its budget."""
    return int(out.get(bound, 0))


def _budget(bound: str) -> int:
    """The configured budget for the named bound (REFLECT-R4)."""
    return MAX_REFLECTIONS if bound == "reflection_count" else MAX_REDRAFTS


def _cases() -> list[dict[str, Any]]:
    return list(_load_catalog()["cases"])


def _case(fixture_id: str) -> dict[str, Any]:
    return next(c for c in _cases() if c["fixture"] == fixture_id)


# --- catalog hygiene (QA-EVAL-R2.11: versioned, checked-in) ----------------------------


def test_catalog_is_versioned_and_covers_three_loops() -> None:
    """The reflection-termination catalog is versioned and names all three recovery loops.

    QA-EVAL-R2.11: the fixture set is checked-in + versioned and covers each permitted
    recovery cycle (GRAPH-R3) exactly once, so no loop is left unguarded.
    """
    catalog = _load_catalog()
    assert catalog["dataset_version"], "catalog MUST carry a version (QA-EVAL-R1)"
    assert catalog["suite"] == "reflection_termination"
    fixtures = {c["fixture"] for c in catalog["cases"]}
    assert fixtures == {"F-COVERAGE-BOUND", "F-REDRAFT-BOUND", "F-GROUNDLOOP-TERMINATES"}
    loops = {c["loop"] for c in catalog["cases"]}
    assert loops == {"coverage", "redraft", "ground_replan"}
    ids = [c["id"] for c in catalog["cases"]]
    assert len(ids) == len(set(ids)), "case ids MUST be unique"
    # Every case forbids budget_exceeded explicitly (the central invariant of this suite).
    for case in catalog["cases"]:
        assert "budget_exceeded" in case["forbidden_statuses"]


# --- the shared bounded-termination assertion ------------------------------------------


async def _assert_bounded_graceful_termination(case: dict[str, Any]) -> Mapping[str, Any]:
    """Run a fixture through the PRODUCTION graph and assert it degrades at its bound.

    The single load-bearing assertion this whole suite exists for: a perpetual recovery
    loop terminates GRACEFULLY (DEGRADED) at its monotonic budget, never unbounded
    (reaching finalize at all proves no GraphRecursionError), and NEVER budget_exceeded
    (the cost gate always admits; budget_exceeded is COST-R4 only).
    """
    model, svc = _services(case)
    graph = build_graph(model, svc, InMemorySaver())
    out = await graph.ainvoke(_input(), config=_config(case["id"]))

    status = out.get("status")
    # NEVER budget_exceeded: a coverage/redraft/replan loop is not a refused admission
    # (COST-R4). Checked FIRST, against the un-narrowed status, so this stays a real
    # assertion and not a tautology after the DEGRADED narrowing below.
    assert status is not RunStatus.BUDGET_EXCEEDED, (
        f"{case['id']}: a bounded recovery loop MUST NOT surface budget_exceeded (COST-R4)"
    )
    # Terminated GRACEFULLY at finalize with a single typed status (OUTCOME-R1/-R3).
    assert status is RunStatus.DEGRADED, (
        f"{case['id']}: expected degraded, got {status} "
        f"(reflect={out.get('reflection_count')} redraft={out.get('redraft_count')})"
    )
    # Bounded: the monotonic counter sits EXACTLY at its budget, neither short of it (an
    # early give-up) nor — impossibly — past it (the reducer is monotonic). Sitting at the
    # budget is the proof the loop spent its whole allowance and then stopped, not looped.
    bound = case["expected_bound_counter"]
    assert _bound_counter(out, bound) == _budget(bound), (
        f"{case['id']}: {bound} must reach its budget {_budget(bound)}, "
        f"got {_bound_counter(out, bound)}"
    )
    # No forbidden status leaked (completed / awaiting_approval / budget_exceeded).
    assert status.value not in case["forbidden_statuses"], case["id"]
    return out


async def test_f_coverage_bound_terminates_degraded() -> None:
    """F-COVERAGE-BOUND: perpetual open coverage gaps spend the reflection budget, then degrade.

    The coverage assessor reports an open gap on every pass and reflect always REPLANs, so
    the assess_coverage -> reflect -> plan_retrieval cycle would loop forever; REFLECT-R4
    bounds it to MAX_REFLECTIONS and the run degrades (open gaps remain), never looping,
    never budget_exceeded.
    """
    case = _case("F-COVERAGE-BOUND")
    out = await _assert_bounded_graceful_termination(case)
    assert out.get("reflection_count") == MAX_REFLECTIONS
    # The grounder PROCEEDs, yet the run still degrades because gaps remain open at finalize
    # (terminal_status, OUTCOME-R5): a coverage-bound exit is a degrade, not a completion.
    assert out.get("redraft_count", 0) == 0


async def test_f_redraft_bound_terminates_degraded() -> None:
    """F-REDRAFT-BOUND: a grounder that always REGENERATEs spends BOTH bounds, then degrades.

    The ground -> compose redraft cycle would loop forever; REFLECT-R4 bounds it to MAX_REDRAFTS.
    An exhausted REGENERATE then FALLS THROUGH to ``replan`` while reflection budget remains (spec
    §225/§451), so ``reflection_count`` also reaches MAX_REFLECTIONS before the run degrades — never
    an unbounded ``ground <-> compose`` loop, never budget_exceeded. compose runs the initial draft
    + one per spent redraft + one per spent re-plan cycle.
    """
    case = _case("F-REDRAFT-BOUND")
    model, svc = _services(case)
    graph = build_graph(model, svc, InMemorySaver())
    out = await graph.ainvoke(_input(), config=_config(case["id"]))
    status = out.get("status")
    assert status is not RunStatus.BUDGET_EXCEEDED
    assert status is RunStatus.DEGRADED
    assert out.get("redraft_count") == MAX_REDRAFTS
    # The redraft budget fully spent, then the bounded fall-through to replan spent reflection too.
    assert out.get("reflection_count") == MAX_REFLECTIONS
    assert model.compose_calls == MAX_REDRAFTS + 1 + MAX_REFLECTIONS


async def test_f_groundloop_terminates_degraded() -> None:
    """F-GROUNDLOOP-TERMINATES: a perpetual-REPLAN grounder spends the reflection budget, degrades.

    The ground -> reflect -> plan_retrieval re-plan cycle (the distinct third recovery
    cycle, GRAPH-R3) would loop forever; REFLECT-R4 bounds it to MAX_REFLECTIONS and the
    run degrades (the terminal ground decision is non-PROCEED), never looping, never
    budget_exceeded.
    """
    case = _case("F-GROUNDLOOP-TERMINATES")
    out = await _assert_bounded_graceful_termination(case)
    assert out.get("reflection_count") == MAX_REFLECTIONS


@pytest.mark.parametrize(
    "fixture_id",
    ["F-COVERAGE-BOUND", "F-REDRAFT-BOUND", "F-GROUNDLOOP-TERMINATES"],
)
async def test_no_fixture_ever_reports_budget_exceeded(fixture_id: str) -> None:
    """Every reflection/redraft loop fixture terminates and NEVER reports budget_exceeded.

    The central QA-EVAL-R6 invariant of this suite, asserted once per fixture: budget_exceeded
    is reserved for a refused cost admission (COST-R4). The cost gate admits in all three
    fixtures, so a budget_exceeded here would be the reverted force-degrade defect re-emerging.
    The run also reaches a terminal status at all (proof of termination, OUTCOME-R3).
    """
    case = _case(fixture_id)
    model, svc = _services(case)
    graph = build_graph(model, svc, InMemorySaver())
    out = await graph.ainvoke(_input(), config=_config(f"{case['id']}-budget"))
    status = out.get("status")
    assert status is not None, "run MUST reach a terminal status (OUTCOME-R3)"
    assert status is not RunStatus.BUDGET_EXCEEDED
    assert status.value == case["expected_status"]


async def test_catalog_drives_all_fixtures_through_the_production_graph() -> None:
    """Iterating the whole checked-in catalog degrades every fixture at its bound (QA-EVAL-R6 100%).

    The suite-level gate: every catalog case, driven through the PRODUCTION build_graph,
    terminates gracefully at its declared bound with its declared status and no forbidden
    status — the 100% termination mandate (QA-EVAL-R6), evaluated straight from the
    versioned datafile so adding a future loop fixture is auto-gated.
    """
    cases = _cases()
    assert len(cases) == 3
    for case in cases:
        await _assert_bounded_graceful_termination(case)
