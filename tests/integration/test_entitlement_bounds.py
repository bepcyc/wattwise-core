"""ENT-R1-AC enforcement — every non-monetary local guard is GOVERNED by the resolved entitlement.

The audit's MED-1 finding was that only ``node_visit_ceiling`` was actually consumed while the
other four bounds were carried + validated but NEVER enforced (an over-claim). These tests prove,
NON-VACUOUSLY, that EACH of the five AGT-ENT-R4 bounds now reads FROM the resolved entitlement at a
REAL enforcement point — and each is MUTATION-PROVEN (revert the wiring -> the test fails):

* ``max_output_tokens`` -> the model's per-call output budget (the ENGINE sizes the model).
* ``max_tool_iterations`` -> a real bound on the gather/tool loop (the GRAPH stops re-planning).
* ``wall_clock_seconds`` -> the whole-run deadline (the ENGINE degrades GRACEFULLY on a breach).
* ``request_rate_per_minute`` -> the API RateLimiter's ``agent``-class ceiling.
* ``node_visit_ceiling`` -> the GRAPH ceiling (covered in ``test_cluster_a_security``; re-asserted
  here against the production seam for completeness).

MED-2 is proven separately (the per-request entitlement threads into the engine + governs the run).

Offline + self-contained (TIER-R1): no network, no real LLM — the model/graph collaborators are
in-test fakes satisfying the public seams, and the wall-clock test drives a deliberately slow fake
graph so the timeout path is exercised deterministically (a tiny bound, not a real 120s wait).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from sqlalchemy import create_engine as _create_sync_engine
from starlette.testclient import TestClient

from tests.integration._schema import provision_app_schema
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
from wattwise_core.agent.deliverables import AgentAnswer, Citation, Observation
from wattwise_core.agent.engine import GraphAgentEngine
from wattwise_core.agent.engine_graph import CompiledCoachGraph
from wattwise_core.agent.graph import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_NODE_VISIT_CEILING,
    AgentServices,
    build_graph,
)
from wattwise_core.agent.graph_state import limitation_text
from wattwise_core.agent.model import OpenAICompatibleModel
from wattwise_core.agent.seams import EntitlementCostGate
from wattwise_core.api.app import _build_rate_limiter, create_app
from wattwise_core.api.ratelimit import DEFAULT_LIMITS, LimitClass, RateLimiter
from wattwise_core.api.routers import agent_routes
from wattwise_core.config import Settings, load_settings
from wattwise_core.entitlement import Entitlements, OssEntitlementResolver
from wattwise_core.identity import OWNER_SUBJECT
from wattwise_core.persistence import Database
from wattwise_core.persistence.models import Base
from wattwise_core.security.crypto import EnvelopeCipher

pytestmark = pytest.mark.integration

_STRONG_KEY = "real-ent-bounds-signing-key-0123456789abcdef"


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    """REAL dev settings on a FILE DB (a real multi-connection pool, skill §7 — never :memory:)."""
    base: dict[str, Any] = {
        "app__environment": "development",
        "database_dsn": f"sqlite+aiosqlite:///{tmp_path / 'ent_bounds.db'}",
        "token_signing_key": _STRONG_KEY,
        "encryption_root_key": EnvelopeCipher.generate_root_key(),
        "object_store__local_root": str(tmp_path / "objects"),
        # A dummy LLM key so OpenAICompatibleModel can build its (never-called) async client; the
        # model-budget tests inspect the per-call budget, they make NO network call (TIER-R1).
        "llm_api_key": "test-only-key-not-used-offline",
    }
    base.update(overrides)
    return load_settings(**base)


# --- in-test graph collaborators (satisfy the public seams only, ARCH-R21) ---------------


class _FakeModel:
    """A ``ChatModel`` stub: scripts a reflect verdict, counts compose calls."""

    def __init__(self, *, reflect_verdict: ReflectVerdict = ReflectVerdict.REPLAN) -> None:
        self.compose_calls = 0
        self._verdict = reflect_verdict

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        if schema is ReflectDecision:
            return ReflectDecision(verdict=self._verdict)  # type: ignore[return-value]
        raise NotImplementedError(schema.__name__)

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_calls += 1
        return f"draft#{self.compose_calls}"


class _CountingGateway:
    """Resolves each planned request to a record; counts how many times gather actually ran."""

    def __init__(self) -> None:
        self.gather_calls = 0

    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        self.gather_calls += 1
        return {f"rec:{self.gather_calls}": {"value": 1.0}}


class _AlwaysPlan:
    """A planner that always returns one request, so the gather/tool loop keeps doing work."""

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        return [RetrievalRequest(capability="pmc", params={})]


class _NeverCovers:
    """A coverage assessor that never closes the gap, forcing the recovery cycles to run."""

    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        return {"never_closes"}


class _ProceedGrounder:
    """A grounder that PROCEEDs over a single grounded survivor (no redraft/replan from here)."""

    async def ground(
        self,
        *,
        athlete_id: str,
        draft: str,
        retrieved: Mapping[str, Any],
        request_text: str | None = None,
        active_constraints: Sequence[Mapping[str, Any]] | None = None,
    ) -> GroundingResult:
        survivor = GroundedClaim(
            claim=Claim(kind=ClaimKind.NUMBER, text="1", value=1.0),
            verdict=GroundVerdict.GROUNDED,
            citation={"metric": "pmc"},
        )
        return GroundingResult(
            decision=GroundDecision.PROCEED, claims=(survivor,), scrubbed_text=draft
        )


def _services(gate: EntitlementCostGate, gateway: _CountingGateway) -> AgentServices:
    """An ``AgentServices`` carrying the entitlement-bearing cost gate + the counting gateway."""
    return AgentServices(
        planner=_AlwaysPlan(),
        gateway=gateway,
        coverage=_NeverCovers(),
        grounder=_ProceedGrounder(),
        cost_gate=gate,
    )


def _input() -> AgentState:
    return AgentState(
        athlete_id="athlete-1",
        trigger="user_turn",
        request_text="how am I?",
        locale="en",
        idempotency_key="idem-ent",
        thread_id="t:ent",
        turn_id="turn-ent",
    )


# ---------------------------------------------------------------- max_output_tokens (model budget)


def test_max_output_tokens_sizes_model_from_entitlement(tmp_path: Path) -> None:
    """The model's per-call output budget is the resolved entitlement's token bound (ENT-R1-AC).

    A config override of ``entitlement.max_output_tokens`` is carried on the resolved plan, and the
    engine's model-sizing reads the budget FROM that plan — never the hardcoded config field alone.
    Proven against the SAME ``_sized_model`` the production engine calls per run.
    """
    override = 12345  # a value provably distinct from the defaults.toml 8192
    settings = _settings(tmp_path, **{"entitlement__max_output_tokens": override})
    plan = OssEntitlementResolver.from_settings(settings).resolve(OWNER_SUBJECT)
    assert plan.max_output_tokens == override  # carried on the resolved plan
    # Build the engine's model with a DIFFERENT (wrong) budget than the entitlement so the test is
    # NON-VACUOUS: only if ``_sized_model`` actually re-sizes FROM the entitlement does the per-run
    # model carry ``override`` — a model that ignored the entitlement would keep the wrong 8192.
    base_model = OpenAICompatibleModel(settings=settings, max_output_tokens=8192)
    assert base_model._max_output_tokens == 8192  # the WRONG baseline budget
    engine = GraphAgentEngine(Database(settings), base_model, entitlement=plan)
    sized = engine._sized_model(plan)
    assert isinstance(sized, OpenAICompatibleModel)
    # The model budget came FROM the entitlement, VERBATIM: the seam honors the resolved value
    # exactly (no lower clamp / no floor — there is none; adequate sizing for the reasoning trace
    # is the operator's responsibility, MODEL-R5a). This is the real governance assertion.
    assert sized._max_output_tokens == override


def test_max_output_tokens_ctor_is_authority_else_config_fallback(tmp_path: Path) -> None:
    """An explicit budget is the authority; absent it, the config-loaded budget is the fallback."""
    settings = _settings(tmp_path)
    explicit = OpenAICompatibleModel(settings=settings, max_output_tokens=9001)
    assert explicit._max_output_tokens == 9001  # the passed entitlement bound governs
    fallback = OpenAICompatibleModel(settings=settings)  # no entitlement -> config field
    assert fallback._max_output_tokens == settings.agent__max_output_tokens


# ---------------------------------------------------------------- max_tool_iterations (tool loop)


async def test_max_tool_iterations_bounds_the_gather_loop_from_entitlement() -> None:
    """Entitlement's ``max_tool_iterations`` bounds the gather/tool loop, GRACEFULLY (AGT-ENT-R4).

    With a planner that always plans and a coverage assessor that never closes the gap, the only
    thing that stops the re-plan -> gather loop is the tool-iteration bound carried on the resolved
    entitlement. A tiny bound (2) is read FROM the entitlement: the loop performs at most that many
    real gathers then routes to compose and finalizes — never an infinite loop, never a raise.
    """
    plan = Entitlements(
        node_visit_ceiling=DEFAULT_NODE_VISIT_CEILING,
        max_output_tokens=8192,
        wall_clock_seconds=120,
        max_tool_iterations=2,
        request_rate_per_minute=120,
    )
    gateway = _CountingGateway()
    svc = _services(EntitlementCostGate(plan), gateway)
    graph = build_graph(
        _FakeModel(),
        svc,
        InMemorySaver(),
        node_visit_ceiling=DEFAULT_NODE_VISIT_CEILING,
        max_tool_iterations=DEFAULT_MAX_TOOL_ITERATIONS,
    )
    cfg = {"configurable": {"thread_id": "tool-bound"}, "recursion_limit": 200}
    out = await graph.ainvoke(_input(), config=cfg)
    # The tool loop was bounded by the entitlement value (NOT the generous module default 16, NOT
    # the node-visit ceiling): at most ``max_tool_iterations`` real gathers ran.
    assert out["tool_iterations"] == 2
    assert gateway.gather_calls == 2
    # And the run still finished gracefully (a terminal status, never a GraphRecursionError).
    assert out["status"] in (RunStatus.COMPLETED, RunStatus.DEGRADED)


async def test_max_tool_iterations_larger_bound_allows_more_gathers() -> None:
    """A LARGER entitlement bound permits MORE gathers — proving the entitlement VALUE governs.

    Same fakes, only the carried bound changes (1 -> 2): the loop performs exactly that many real
    gathers before the tool bound stops it, so the behavior tracks the entitlement value, not a
    constant. (Both bounds are below the reflection-budget-driven ceiling, so the tool bound — not
    MAX_REFLECTIONS — is the gate being exercised here.)
    """
    for bound in (1, 2):
        plan = Entitlements(
            node_visit_ceiling=DEFAULT_NODE_VISIT_CEILING,
            max_output_tokens=8192,
            wall_clock_seconds=120,
            max_tool_iterations=bound,
            request_rate_per_minute=120,
        )
        gateway = _CountingGateway()
        svc = _services(EntitlementCostGate(plan), gateway)
        graph = build_graph(
            _FakeModel(),
            svc,
            InMemorySaver(),
            node_visit_ceiling=DEFAULT_NODE_VISIT_CEILING,
            max_tool_iterations=DEFAULT_MAX_TOOL_ITERATIONS,
        )
        cfg = {"configurable": {"thread_id": f"tool-{bound}"}, "recursion_limit": 200}
        out = await graph.ainvoke(_input(), config=cfg)
        assert out["tool_iterations"] == bound
        assert gateway.gather_calls == bound


# ---------------------------------------------------------------- wall_clock_seconds (run deadline)


class _SlowCompiledGraph:
    """A compiled-graph stub whose ``ainvoke`` sleeps longer than the wall-clock budget."""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.invoked = False

    async def ainvoke(self, state: Any, config: Any) -> AgentState:
        self.invoked = True
        await asyncio.sleep(self._delay)
        # If the deadline did NOT fire this would return a COMPLETED-looking state.
        return AgentState(status=RunStatus.COMPLETED, grounded_text="late answer")


async def test_wall_clock_breach_degrades_gracefully_never_raises() -> None:
    """A run exceeding its wall-clock budget DEGRADES gracefully — never raises (AGT-ENT-R4).

    Drives the production ``CompiledCoachGraph.run`` (the single graph-invoke point the engine uses)
    with a tiny wall-clock bound and a deliberately slow fake graph: the ``asyncio.wait_for``
    deadline fires, and the run returns a DEGRADED terminal state with an EMPTY grounded HTML body
    and a typed degraded caveat — NOT the slow graph's late 'completed' answer, NOT a TimeoutError.
    """
    state = _input()
    slow = _SlowCompiledGraph(delay=5.0)
    coach = CompiledCoachGraph(slow, wall_clock_seconds=0.05)  # type: ignore[arg-type]
    final = await coach.run(state)
    assert slow.invoked  # the run actually started
    assert final["status"] is RunStatus.DEGRADED  # the deadline degraded it, not COMPLETED
    assert final.get("grounded_html") == ""  # no partial/ungrounded body escaped (GROUND-R3)
    assert final.get("grounded_text") != "late answer"  # the slow graph's answer was NOT used
    # The degraded body is NON-EMPTY: a wall-clock degrade must still hand the user the localized
    # "couldn't finish in time" limitation copy (wall_clock_degraded -> limitation_text), never a
    # BLANK answer. A regression returning an empty degraded body (user sees nothing) fails here.
    expected = limitation_text(state)
    assert expected  # the floor copy itself is non-empty (guards the fixture, not just the path)
    assert final.get("grounded_text") == expected  # exactly the localized limitation floor
    assert final["coverage_caveat"]["fidelity"] == "degraded"  # the typed graceful caveat


async def test_wall_clock_within_budget_completes_normally() -> None:
    """A run that finishes inside the wall-clock budget completes normally (no spurious degrade)."""
    fast = _SlowCompiledGraph(delay=0.0)
    coach = CompiledCoachGraph(fast, wall_clock_seconds=5.0)  # type: ignore[arg-type]
    final = await coach.run(_input())
    assert final["status"] is RunStatus.COMPLETED  # the deadline did NOT fire
    assert final["grounded_text"] == "late answer"  # the real graph result passed through


# ------------------------------------------- FIX-4: wall-clock must NOT orphan a live interrupt row

#: The marker the plan deliverable stamps onto its run inputs so ``interrupt_gate`` knows this is an
#: approval-gated PLAN that durably pauses (the SAME marker ``plan_requires_approval`` keys on).
_PLAN_MARKER = {"role": "system", "kind": "plan_deliverable", "requires_approval": True}


def _plan_input() -> AgentState:
    """A pausable approval-gated PLAN run input (carries the ``plan_deliverable`` marker)."""
    state = _input()
    state["messages"] = [_PLAN_MARKER]
    return state


class _SlowPausingGraph:
    """A compiled-graph stub that takes longer than the bound, then PAUSES at the approval gate.

    Models the production approval-gated plan: ``interrupt_gate`` commits a ``live`` ledger row and
    then langgraph SUSPENDS, surfacing the pause back through ``CompiledCoachGraph.run`` as an
    ``__interrupt__`` terminal carrying the gate's ``interrupt_id``. The sleep simulates the run
    taking longer than a tiny wall-clock bound; ``finished`` records whether ``ainvoke`` ran to
    completion (i.e. was NOT cancelled by a deadline).
    """

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.finished = False

    async def ainvoke(self, state: Any, config: Any) -> AgentState:
        await asyncio.sleep(self._delay)
        self.finished = True
        # The pause shape langgraph returns when ``interrupt_gate`` suspended after recording the
        # ``live`` row: the accumulated state with an ``__interrupt__`` carrying the gate's payload.
        paused: dict[str, Any] = {
            "status": RunStatus.AWAITING_APPROVAL,
            "thread_id": "t:ent",
            "__interrupt__": [type("I", (), {"value": {"interrupt_id": "01PAUSE"}})()],
        }
        return cast(AgentState, paused)


async def test_wall_clock_not_applied_to_pausable_plan_path_no_orphan() -> None:
    """A pausable PLAN run is NOT wall-clock-degraded; its live interrupt row survives (FIX-4).

    On the approval-gated plan path ``interrupt_gate`` commits a ``live`` ``AgentInterrupt`` row
    just before langgraph suspends; the pause surfaces back through ``CompiledCoachGraph.run``. If
    wall-clock deadline fired in that window it would return :func:`wall_clock_degraded` (no
    ``interrupt_id``) and ORPHAN the live row forever. Mitigation (a): the deadline is skipped for
    the pausable plan path (identified by the ``plan_deliverable`` marker), so a slow-then-pause run
    is NOT degraded — it returns the pause carrying the ``interrupt_id`` the decision endpoint
    consumes (so the row is reachable, never orphaned). The delay far exceeds the tiny bound, so a
    regression that re-applied the deadline here would degrade instead and fail this test.
    """
    graph = _SlowPausingGraph(delay=0.3)
    coach = CompiledCoachGraph(graph, wall_clock_seconds=0.01)  # type: ignore[arg-type]
    final = await coach.run(_plan_input())
    assert graph.finished  # the invoke ran to completion (the deadline did NOT cancel the pause)
    assert final["status"] is RunStatus.AWAITING_APPROVAL  # the pause survived, NOT a degrade
    assert final.get("status") is not RunStatus.DEGRADED  # explicitly: NOT wall_clock_degraded
    payload = cast(Mapping[str, Any], final).get("__interrupt__")
    assert payload, "the pause + its interrupt_id reached the caller (the live row is consumable)"


async def test_wall_clock_still_applied_to_autonomous_non_plan_path() -> None:
    """The SAME slow fake IS wall-clock-degraded WITHOUT the plan marker (FIX-4 discriminator).

    Proves the skip is keyed on the ``plan_deliverable`` marker, NOT an unconditional bypass: the
    identical slow graph driven by a plain (autonomous) ``/ask`` input — which never pauses — keeps
    the wall-clock deadline and degrades. This is the contrast that makes the sibling test
    non-vacuous: only the pausable plan path skips the deadline.
    """
    graph = _SlowPausingGraph(delay=0.3)
    coach = CompiledCoachGraph(graph, wall_clock_seconds=0.01)  # type: ignore[arg-type]
    final = await coach.run(_input())  # autonomous input: NO plan_deliverable marker
    assert final["status"] is RunStatus.DEGRADED  # the deadline fired on the autonomous path
    assert final.get("grounded_html") == ""  # graceful degrade (no partial body, GROUND-R3)


# ---------------------------------------------------------------- request_rate_per_minute (limiter)


def test_request_rate_per_minute_sets_agent_ceiling_from_entitlement(tmp_path: Path) -> None:
    """The RateLimiter's ``agent`` ceiling is the entitlement request-rate bound (LIMIT-R2).

    A config override of ``entitlement.request_rate_per_minute`` flows into the limiter the app
    factory builds — so the agent-class request rate IS the entitlement's non-monetary request-rate
    guard, NOT the hardcoded ``DEFAULT_LIMITS`` literal. The read/mutating ceilings come from the
    ``[ratelimit]`` config table (no code literal either).
    """
    override = 47  # distinct from DEFAULT_LIMITS[AGENT] == 20 and the defaults.toml 20
    settings = _settings(tmp_path, **{"entitlement__request_rate_per_minute": override})
    limiter = _build_rate_limiter(settings)
    assert limiter._limits[LimitClass.AGENT] == override  # entitlement governs the agent ceiling
    assert limiter._limits[LimitClass.AGENT] != DEFAULT_LIMITS[LimitClass.AGENT]  # not the literal
    # The read/mutating ceilings are config-sourced too (CFG-R1a).
    assert limiter._limits[LimitClass.READ] == settings.ratelimit__read_per_minute
    assert limiter._limits[LimitClass.MUTATING] == settings.ratelimit__mutating_per_minute


def test_read_and_mutating_ceilings_track_config(tmp_path: Path) -> None:
    """Overriding the ``[ratelimit]`` read/mutating ceilings flows into the limiter (CFG-R1a)."""
    settings = _settings(
        tmp_path,
        **{"ratelimit__read_per_minute": 77, "ratelimit__mutating_per_minute": 11},
    )
    limiter = _build_rate_limiter(settings)
    assert limiter._limits[LimitClass.READ] == 77
    assert limiter._limits[LimitClass.MUTATING] == 11


def test_production_agent_ceiling_is_the_documented_20_per_minute(tmp_path: Path) -> None:
    """The PRODUCTION agent-class ceiling is the documented 20/min baseline (LIMIT-R2).

    The other request-rate tests INJECT a config override (47) so they exercise the wiring but
    never assert the SHIPPED production value: the agent ceiling is the ``entitlement``
    request-rate bound, whose defaults.toml value MUST be the documented agent baseline of
    ``20/min`` (the ``ratelimit.py`` module docstring + the existing ``test_api_agent``
    ``limit == 20`` assertions). This drives the REAL assembled ``create_app`` limiter — NO
    limiter override, NO request-rate override — so it is the genuine production ceiling, and
    fails if the shipped default ever again drifts above the documented baseline (e.g. to the
    READ-class 120). The READ/MUTATING ceilings ride their own ``[ratelimit]`` config defaults.
    """
    settings = _settings(tmp_path)  # the real shipped config: no request-rate override
    app = create_app(settings)
    limiter = app.state.rate_limiter  # the limiter the factory assembled at boot
    assert isinstance(limiter, RateLimiter)
    assert limiter._limits[LimitClass.AGENT] == 20  # the documented agent baseline (LIMIT-R2)
    assert limiter._limits[LimitClass.AGENT] == settings.entitlement__request_rate_per_minute
    # The agent ceiling is NOT the READ-class 120 (the drift this guards against).
    assert limiter._limits[LimitClass.AGENT] != settings.ratelimit__read_per_minute


# ---------------------------------------------------------------- MED-2: resolve -> attach -> check


class _RecordingEngine:
    """An engine stand-in that RECORDS the ``entitlement`` the route threaded into ``answer``.

    The gate runs BEFORE the handler, so when the run reaches here the entitlement has already been
    resolved + attached to the request and the route passed it through (MED-2). Returns a grounded
    answer so the 200 path is real.
    """

    def __init__(self) -> None:
        self.seen_entitlement: Entitlements | None = None
        self.was_called = False

    async def answer(self, *, athlete_id: str, entitlement: Entitlements | None = None, **_: Any):
        self.was_called = True
        self.seen_entitlement = entitlement
        return AgentAnswer(
            status=RunStatus.COMPLETED,
            thread_id="01THREAD",
            answer_html="<p>You're fresh.</p>",
            answer_text="You're fresh.",
            observations=(Observation(observation_id="01OBS", text="Recovered."),),
            citations=(Citation(record_id="01CIT", metric="tsb", value=6.2, as_of="2026-06-05"),),
        )


def _create_canonical_schema(db_file: Path) -> None:
    """Create the canonical schema on the harness FILE DB (what migrations do in prod).

    ``POST /v1/agent/ask`` now legitimately reads the canonical ``athlete`` row for the
    persisted language default (API-R37), so the harness database must carry the
    canonical schema even when the test seeds no rows.
    """
    sync_engine = _create_sync_engine(f"sqlite:///{db_file}")
    try:
        Base.metadata.create_all(sync_engine)
    finally:
        sync_engine.dispose()


def test_med2_request_resolved_entitlement_threads_into_engine(tmp_path: Path) -> None:
    """The per-request resolved entitlement is threaded from the HTTP gate INTO the engine (MED-2).

    Drives ``POST /v1/agent/ask`` through the REAL ``create_app`` wiring (the real
    ``agent_feature_gate`` -> ``resolve_entitlement`` attaches ``request.state.entitlement``); ONLY
    the engine is faked, and it RECORDS the entitlement the route passed it. The recorded plan is
    SAME server-derived plan the app resolved onto its state — proving resolve -> attach -> check is
    REAL end to end, not a noop that re-derives from config inside the engine.
    """
    settings = _settings(tmp_path)
    _create_canonical_schema(tmp_path / "ent_bounds.db")
    app = create_app(settings)
    provision_app_schema(app)  # the token route persists the refresh credential (SEC-R2.3)
    engine = _RecordingEngine()
    app.dependency_overrides[agent_routes.agent_engine] = lambda: engine  # only the engine is faked
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    try:
        token_resp = client.post("/v1/auth/token", json={"owner_secret": _STRONG_KEY})
        assert token_resp.status_code == 200, token_resp.text
        auth = {"Authorization": f"Bearer {token_resp.json()['access_token']}"}
        resp = client.post("/v1/agent/ask", json={"question": "How am I?"}, headers=auth)
        assert resp.status_code == 200, resp.text
        assert engine.was_called  # the gate admitted and the route ran the engine
        # The route threaded a REAL resolved entitlement (not None) and it is the SAME plan the app
        # resolved onto its state (the resolve -> attach -> check seam carried it end to end).
        assert engine.seen_entitlement is not None
        assert engine.seen_entitlement == app.state.entitlement_plan
        assert engine.seen_entitlement.can_use_agent is True  # OSS all-permissive plan
    finally:
        client.__exit__(None, None, None)
