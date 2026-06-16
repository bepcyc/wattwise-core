"""Unit tests for the compiled agent graph (doc 50 GRAPH-R*, REFLECT-R*, OUTCOME-R1).

Offline and self-contained: every collaborator is an in-test fake that satisfies the
public seams in :mod:`wattwise_core.agent.contracts` (a ``ChatModel`` and the four
injected services bundled in :class:`AgentServices`). No sibling agent module is
imported (ARCH-R21) — the graph is exercised purely through ``build_graph`` and the
typed state. The checkpointer is langgraph's in-memory saver.

Asserted behaviours:

* GRAPH-R2/OUTCOME-R1 happy path: a clean run reaches the single ``finalize`` sink and
  emits exactly one ``RunStatus.COMPLETED``.
* REFLECT-R4 coverage exhaustion: when gaps never close, the reflection budget is spent
  exactly ``MAX_REFLECTIONS`` times and the run **degrades** (not loops, not errors).
* GRAPH-R3 redraft cycle is bounded: a grounder that always asks to REGENERATE spends
  the redraft budget exactly ``MAX_REDRAFTS`` times then settles, never looping forever.
* GROUND-R9/OUTCOME-R1: a grounder ABSTAIN finalizes ``DEGRADED`` (never
  ``AWAITING_APPROVAL``); ``awaiting_approval`` is emitted ONLY by ``interrupt_gate`` for
  an approval-gated PLAN paused at a durable langgraph interrupt (CKPT-R5).
* GRAPH-R5: a node-visit-ceiling breach routes to a ``DEGRADED`` finalize, never raising.
* STATE-R4: a changed write to a write-once identity field is rejected at the reducer.
* COST-R4: a refused cost-admission gate finalizes ``BUDGET_EXCEEDED``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
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
    _write_once,
)
from wattwise_core.agent.graph import (
    MAX_REDRAFTS,
    MAX_REFLECTIONS,
    AgentServices,
    build_graph,
)

pytestmark = pytest.mark.unit


# --- fakes (satisfy the public contracts only) -----------------------------------------


class FakeModel:
    """Deterministic ``ChatModel`` stub: counts compose calls, scripts reflect verdicts.

    ``structured`` returns the scripted :class:`ReflectDecision` so the reflect node's
    REFLECT-R2 verdict is provider-shaped in tests; the default verdict is ``replan`` so
    the bounded recovery cycles still exercise to their budget.
    """

    def __init__(self, *, reflect_verdict: ReflectVerdict = ReflectVerdict.REPLAN) -> None:
        self.compose_calls = 0
        self.structured_calls = 0
        self._reflect_verdict = reflect_verdict

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        self.structured_calls += 1
        if schema is ReflectDecision:
            return ReflectDecision(verdict=self._reflect_verdict)  # type: ignore[return-value]
        raise NotImplementedError(f"no scripted structured output for {schema.__name__}")

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_calls += 1
        return f"draft#{self.compose_calls}: {context.splitlines()[0]}"


class FakePlanner:
    """Returns one capability request per call; records the athlete-free inputs seen."""

    def __init__(self) -> None:
        self.calls = 0

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        self.calls += 1
        return [RetrievalRequest(capability="pmc", params={"n": self.calls})]


class FakeGateway:
    """Resolves each request to a canonical record; asserts a server-derived id flows."""

    def __init__(self) -> None:
        self.seen_athlete_ids: list[str] = []

    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        self.seen_athlete_ids.append(athlete_id)
        return {f"rec:{r.capability}": {"value": 42.0} for r in requests}


class FakeCoverage:
    """Coverage assessor whose open-gap set is scripted to drive routing."""

    def __init__(self, gaps: set[str]) -> None:
        self._gaps = gaps

    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        return set(self._gaps)


class FakeGrounder:
    """Grounder returning a scripted aggregate decision; counts invocations."""

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
        claim = Claim(kind=ClaimKind.NUMBER, text="42", value=42.0)
        survivor = GroundedClaim(
            claim=claim, verdict=GroundVerdict.GROUNDED, citation={"metric": "pmc"}
        )
        return GroundingResult(decision=self._decision, claims=(survivor,), scrubbed_text=draft)


def _services(
    *,
    gaps: set[str],
    decision: GroundDecision,
    reflect_verdict: ReflectVerdict = ReflectVerdict.REPLAN,
) -> tuple[FakeModel, AgentServices]:
    model = FakeModel(reflect_verdict=reflect_verdict)
    svc = AgentServices(
        planner=FakePlanner(),
        gateway=FakeGateway(),
        coverage=FakeCoverage(gaps),
        grounder=FakeGrounder(decision),
    )
    return model, svc


def _input(athlete_id: str = "athlete-1") -> AgentState:
    return AgentState(
        athlete_id=athlete_id,
        trigger="user_turn",
        request_text="how is my fitness trending?",
        locale="en",
        idempotency_key="idem-1",
    )


def _config(thread: str) -> RunnableConfig:
    # recursion_limit bounds the supersteps — if any cycle were unbounded the run
    # would raise GraphRecursionError instead of completing within this budget.
    return {"configurable": {"thread_id": thread}, "recursion_limit": 50}


# --- tests -----------------------------------------------------------------------------


async def test_happy_path_completes() -> None:
    """No gaps, grounder proceeds -> single COMPLETED outcome (OUTCOME-R1)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("happy"))

    assert out["status"] is RunStatus.COMPLETED
    assert out["reflection_count"] == 0
    assert out["redraft_count"] == 0
    assert out["coverage_caveat"] is None
    # draft was produced and grounded exactly once (no recovery cycles).
    assert model.compose_calls == 1
    assert svc.grounder.calls == 1  # type: ignore[attr-defined]
    # citations survived grounding.
    assert out["citations"] == [{"metric": "pmc"}]


async def test_ground_node_writes_stable_id_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    """COACH-R8: the production ``ground`` node writes stable-id observations (drill/reveal handle).

    The grounder survives one grounded claim carrying a citation. The ``ground`` node MUST emit an
    ``observations`` channel where the distinct observation carries a STABLE ``observation_id`` (the
    expand/drill handle a later follow-up targets, COACH-R8) plus the grounded ``{metric,...}``
    citation behind it. If the node writes no observations (the pre-fix gap), the observations list
    is empty and drill/reveal-by-id is vacuous, failing this test. The id is DETERMINISTIC: a second
    run with the SAME grounded claim yields the SAME id so a follow-up can target it across turns.
    """
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("obs"))

    observations = out.get("observations") or []
    assert observations, "the ground node MUST write at least one observation (COACH-R8)"
    obs = observations[0]
    assert obs.get("observation_id"), "every observation MUST carry a stable id (COACH-R8)"
    assert obs.get("text"), "an observation MUST carry athlete-facing text to target"
    # The grounded citation is reachable behind the observation (the reveal-numbers backing).
    cited = [c for c in obs.get("citations", []) if c]
    assert cited and cited[0].get("metric") == "pmc"

    # DETERMINISTIC stable id: a fresh run with the same grounded claim re-derives the SAME id,
    # so a follow-up turn can target the observation without re-stating the question (COACH-R8).
    out2 = await graph.ainvoke(_input(), config=_config("obs-2"))
    assert out2["observations"][0]["observation_id"] == obs["observation_id"]


async def test_coverage_exhaustion_degrades() -> None:
    """Persistent gaps spend the reflection budget then DEGRADE (REFLECT-R4)."""
    model, svc = _services(gaps={"missing_ftp"}, decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("degrade"))

    assert out["status"] is RunStatus.DEGRADED
    # reflection budget spent exactly to the bound, not beyond.
    assert out["reflection_count"] == MAX_REFLECTIONS
    # the unresolved gap is surfaced as a TYPED coverage caveat (OUTCOME-R4 structured).
    assert list(out["coverage_caveat"]["missing"]) == ["missing_ftp"]
    assert out["coverage_caveat"]["fidelity"] == "partial"


async def test_redraft_cycle_is_bounded() -> None:
    """A grounder stuck on REGENERATE spends BOTH bounds then settles, no loop (REFLECT-R4).

    The redraft cycle spends ``redraft_count`` to ``MAX_REDRAFTS``; per REFLECT-R4 (spec §225/§451)
    an exhausted REGENERATE then FALLS THROUGH to ``replan`` while reflection budget remains, so
    ``reflection_count`` also spends to ``MAX_REFLECTIONS`` before the run degrades — never an
    immediate abstain and never an unbounded loop. The two distinct monotonic counters bound the
    total: compose runs the initial draft + one per redraft + one per re-plan cycle.
    """
    model, svc = _services(gaps=set(), decision=GroundDecision.REGENERATE)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("redraft"))

    # bounded: BOTH monotonic counters sit exactly at their budget, no infinite loop.
    assert out["redraft_count"] == MAX_REDRAFTS
    assert out["reflection_count"] == MAX_REFLECTIONS
    # initial compose + one per spent redraft + one per spent re-plan cycle (the fall-through).
    assert model.compose_calls == MAX_REDRAFTS + 1 + MAX_REFLECTIONS
    # both budgets exhausted with a non-proceed decision -> degraded, never an error.
    assert out["status"] is RunStatus.DEGRADED


async def test_replan_cycle_is_bounded() -> None:
    """A grounder stuck on REPLAN spends the reflection budget then settles (GRAPH-R3)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.REPLAN)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("replan"))

    assert out["reflection_count"] == MAX_REFLECTIONS
    assert out["status"] is RunStatus.DEGRADED


async def test_exhausted_redraft_falls_through_to_replan_not_immediate_abstain() -> None:
    """REFLECT-R4: an exhausted REGENERATE re-plans while reflection budget remains (§225/§451).

    MUTATION-GUARD for the fall-through edge: a perpetual REGENERATE spends ``redraft_count`` to its
    bound, and instead of routing straight to the gate (an immediate abstain — the pre-fix gap) it
    FALLS THROUGH to a bounded ``replan`` so ``reflection_count`` ALSO reaches its bound. If the
    fall-through is removed, ``reflection_count`` stays 0 and this assertion fails. The run still
    terminates (both monotonic counters spent) and degrades — never an unbounded loop.
    """
    model, svc = _services(gaps=set(), decision=GroundDecision.REGENERATE)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("redraft-fallthrough"))

    assert out["redraft_count"] == MAX_REDRAFTS
    # The load-bearing assertion: the exhausted redraft fell through to replan and spent it too.
    assert out["reflection_count"] == MAX_REFLECTIONS
    assert out["status"] is RunStatus.DEGRADED


async def test_abstain_degrades_not_awaiting_approval() -> None:
    """A grounder ABSTAIN finalizes DEGRADED with a caveat, never AWAITING_APPROVAL.

    GROUND-R9 / OUTCOME-R1 / OUTCOME-R3: abstain is a fail-closed grounding outcome that
    ships a typed limitation deliverable (degraded), not a plan awaiting human approval.
    awaiting_approval is reserved strictly for an approval-gated PLAN at interrupt_gate.
    """
    model, svc = _services(gaps=set(), decision=GroundDecision.ABSTAIN)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("abstain"))

    assert out["status"] is RunStatus.DEGRADED
    assert out["status"] is not RunStatus.AWAITING_APPROVAL
    # the degraded outcome carries a typed coverage caveat (OUTCOME-R4).
    assert out["coverage_caveat"] is not None
    assert out["coverage_caveat"]["fidelity"] == "degraded"


async def test_server_derived_identity_flows_not_client() -> None:
    """The gateway only ever sees the server-set athlete id (AGT-SEC-R1)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    await graph.ainvoke(_input(athlete_id="srv-id-99"), config=_config("identity"))

    assert svc.gateway.seen_athlete_ids == ["srv-id-99"]  # type: ignore[attr-defined]


async def test_missing_identity_fails_closed() -> None:
    """A run without a server-derived athlete id fails closed, not silently (AGT-SEC-R1)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    bad = _input()
    del bad["athlete_id"]
    with pytest.raises(ValueError, match="athlete_id"):
        await graph.ainvoke(bad, config=_config("noident"))


async def test_phase1_run_never_awaits_approval() -> None:
    """Phase-1 ships no approval-gated plan, so awaiting_approval never fires (CKPT-R5)."""
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("noplan"))

    assert out["status"] is RunStatus.COMPLETED
    assert out["status"] is not RunStatus.AWAITING_APPROVAL


async def test_approval_gated_plan_pauses_at_durable_interrupt() -> None:
    """An approval-gated PLAN deliverable PAUSES at interrupt_gate with a durable interrupt.

    CKPT-R5: the pause yields ``awaiting_approval`` carrying the grounded plan + a unique
    interrupt_id, and the run does NOT reach finalize until a matching approve decision
    resumes it — at which point finalize emits COMPLETED.
    """
    model, svc = _services(gaps=set(), decision=GroundDecision.PROCEED)
    graph = build_graph(model, svc, InMemorySaver())
    cfg = _config("approval-plan")

    state = _input()
    # Mark the run's grounded deliverable as an approval-gated plan (product policy).
    state["messages"] = [{"role": "system", "kind": "plan_deliverable", "requires_approval": True}]
    paused = await graph.ainvoke(state, config=cfg)

    # The run is HELD at the durable interrupt: no terminal status yet.
    assert paused.get("status") is None or "status" not in paused
    interrupts = paused["__interrupt__"]
    assert interrupts, "interrupt_gate must yield a durable interrupt"
    payload = interrupts[0].value
    assert payload["status"] == RunStatus.AWAITING_APPROVAL.value
    assert payload["interrupt_id"]

    # Resuming with an approve decision runs finalize to a single COMPLETED outcome.
    resumed = await graph.ainvoke(Command(resume={"approved": True}), config=cfg)
    assert resumed["status"] is RunStatus.COMPLETED


async def test_non_proceed_plan_degrades_never_awaits_approval() -> None:
    """A non-PROCEED plan run is NEVER put to a human decision (issue #25, decision-aware gate).

    The ``ground`` node writes ``grounded_text`` on EVERY pass — including an ABSTAIN body the
    grounder ruled unpublishable. Without a decision-aware gate, an approval-gated plan would pause
    and ship that body as AWAITING_APPROVAL, asking the athlete to approve a plan the grounder does
    not stand behind. The gate now requires PROCEED before soliciting approval: a non-PROCEED plan
    falls through to finalize and degrades like every other deliverable.
    """
    model, svc = _services(gaps=set(), decision=GroundDecision.ABSTAIN)
    graph = build_graph(model, svc, InMemorySaver())
    cfg = _config("non-proceed-plan")

    state = _input()
    # An approval-gated PLAN deliverable, but the grounder ABSTAINS on its body.
    state["messages"] = [{"role": "system", "kind": "plan_deliverable", "requires_approval": True}]
    out = await graph.ainvoke(state, config=cfg)

    assert "__interrupt__" not in out  # never paused for a human decision
    assert out["status"] is RunStatus.DEGRADED
    assert out["status"] is not RunStatus.AWAITING_APPROVAL


async def test_write_once_identity_overwrite_is_rejected() -> None:
    """A second write of a DIFFERENT athlete_id is rejected at the reducer (STATE-R4)."""
    assert _write_once("", "athlete-1") == "athlete-1"
    assert _write_once("athlete-1", "athlete-1") == "athlete-1"
    with pytest.raises(ValueError, match="write-once"):
        _write_once("athlete-1", "attacker-2")


async def test_node_visit_ceiling_degrades_never_raises() -> None:
    """A run that would exceed the configured node-visit ceiling degrades, never errors.

    GRAPH-R5 / OUTCOME-R3: with a tiny ceiling the run terminates at finalize with a
    DEGRADED status instead of raising GraphRecursionError or looping.
    """
    model, svc = _services(gaps={"never_closes"}, decision=GroundDecision.PROCEED)
    # A ceiling far below the longest legal path forces an early ceiling-degrade.
    graph = build_graph(model, svc, InMemorySaver(), node_visit_ceiling=3)

    out = await graph.ainvoke(_input(), config=_config("ceiling"))

    assert out["status"] is RunStatus.DEGRADED
    assert out["cost_rollup"]["node_visits"] >= 3


async def test_budget_exceeded_when_admission_refused() -> None:
    """A cost-admission gate that refuses finalizes BUDGET_EXCEEDED (COST-R4)."""

    class _RefusingGate:
        async def admit(self, *, athlete_id: str, state: AgentState) -> bool:
            return False

        async def settle(self, *, athlete_id: str, state: AgentState) -> None:
            return None

    model = FakeModel()
    svc = AgentServices(
        planner=FakePlanner(),
        gateway=FakeGateway(),
        coverage=FakeCoverage(set()),
        grounder=FakeGrounder(GroundDecision.PROCEED),
        cost_gate=_RefusingGate(),
    )
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("budget"))

    assert out["status"] is RunStatus.BUDGET_EXCEEDED
    # the run short-circuited: it never composed a draft.
    assert model.compose_calls == 0


async def test_reflect_emits_structured_verdict_and_routes() -> None:
    """reflect emits a structured §6 verdict; give_up_gracefully terminates (REFLECT-R2)."""
    model, svc = _services(
        gaps={"missing"},
        decision=GroundDecision.PROCEED,
        reflect_verdict=ReflectVerdict.GIVE_UP_GRACEFULLY,
    )
    graph = build_graph(model, svc, InMemorySaver())

    out = await graph.ainvoke(_input(), config=_config("reflect-verdict"))

    # the structured verdict was obtained at least once ...
    assert model.structured_calls >= 1
    # ... and give_up_gracefully routes through compose+ground to a degraded finalize
    # (REFLECT-R3: a caveated graceful-decline draft, never an empty body).
    assert out["status"] is RunStatus.DEGRADED
    assert out["reflection_count"] == 1
