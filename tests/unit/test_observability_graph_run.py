"""A real graph run is fully traced + rolled-up + counted (AGT-OBS-R1/-R2/-R7, OBS-R4).

Drives the PRODUCTION ``build_graph`` graph inside a bound run trace — exactly as the engine's
``CompiledCoachGraph.run`` does — and asserts the §15 observability contract holds end-to-end on
a real (offline) run:

- AGT-OBS-R1: each node execution emitted a span under the one run trace.
- AGT-OBS-R2: the per-run rollup the finalize node carries on state records total tokens/cost/
  latency, the model-tier mix, the reflection count, the scrub count, and the terminal status.
- AGT-OBS-R7 / OBS-R4: the terminal status, grounding run, and per-run latency/cost land on the
  production metrics surface so a regression is alertable in production, not only in CI.

Mutation-proofing: dropping the cost_rollup trace-merge leaves the AGT-OBS-R2 keys absent;
dropping the finalize terminal recording leaves the RUN_TERMINAL counter flat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
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
    RetrievalRequest,
    RunStatus,
)
from wattwise_core.agent.graph import AgentServices, build_graph
from wattwise_core.observability import metrics as m
from wattwise_core.observability import runtrace

pytestmark = pytest.mark.unit


class _Model:
    """Offline ChatModel stub: composes prose, never opens a provider span (no usage)."""

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        """Return a scripted reflect verdict; no other schema is requested on the happy path."""
        raise NotImplementedError(f"no scripted structured output for {schema.__name__}")

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        """Return canned prose (no network, no provider usage)."""
        return "your fitness is trending up nicely."


class _Planner:
    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        """Plan one capability request per turn."""
        return [RetrievalRequest(capability="pmc", params={})]


class _Gateway:
    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        """Resolve each request to a canonical record under the authenticated athlete."""
        return {f"rec:{r.capability}": {"value": 42.0} for r in requests}


class _Coverage:
    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        """No remaining gaps -> the happy path reaches finalize completed."""
        return set()


class _Grounder:
    async def ground(
        self, *, athlete_id: str, draft: str, retrieved: Mapping[str, Any]
    ) -> GroundingResult:
        """Ground one claim (grounded) + one scrubbed (ungrounded) so scrub-count is exercised."""
        grounded = GroundedClaim(
            claim=Claim(kind=ClaimKind.NUMBER, text="42", value=42.0),
            verdict=GroundVerdict.GROUNDED,
            citation={"metric": "pmc"},
        )
        scrubbed = GroundedClaim(
            claim=Claim(kind=ClaimKind.NUMBER, text="999", value=999.0),
            verdict=GroundVerdict.UNGROUNDED,
        )
        return GroundingResult(
            decision=GroundDecision.PROCEED, claims=(grounded, scrubbed), scrubbed_text=draft
        )


def _input() -> AgentState:
    return AgentState(
        athlete_id="athlete-1",
        trigger="user_turn",
        request_text="how is my fitness trending?",
        locale="en",
        idempotency_key="idem-obs",
    )


async def test_real_run_is_traced_rolled_up_and_counted() -> None:
    """A bound run records spans + the AGT-OBS-R2 rollup + the AGT-OBS-R7 terminal metrics."""
    svc = AgentServices(
        planner=_Planner(), gateway=_Gateway(), coverage=_Coverage(), grounder=_Grounder()
    )
    graph = build_graph(_Model(), svc, InMemorySaver())
    registry = m.get_registry()
    runs_before = registry.counter_value(m.GROUNDING_RUNS)
    terminal_before = registry.counter_value(m.RUN_TERMINAL, labels={"status": "completed"})

    with runtrace.run_trace("athlete-1:conv-obs") as trace:
        out = await graph.ainvoke(
            _input(),
            config={"configurable": {"thread_id": "athlete-1:conv-obs"}, "recursion_limit": 50},
        )

    assert out["status"] is RunStatus.COMPLETED
    # AGT-OBS-R1: each node execution emitted a span under the one run trace.
    span_names = {s.name for s in trace.spans}
    assert {
        "ingest_request",
        "plan_retrieval",
        "gather",
        "compose",
        "ground",
        "finalize",
    } <= span_names
    assert trace.trace_id == "athlete-1:conv-obs"
    # AGT-OBS-R2: the per-run rollup carries the mandated fields (read from the trace).
    rollup = out["cost_rollup"]
    for key in (
        "total_tokens",
        "total_cost_usd",
        "total_latency_ms",
        "model_tier_mix",
        "reflection_count",
        "scrub_count",
        "status",
    ):
        assert key in rollup, key
    assert rollup["status"] == "completed"
    assert rollup["scrub_count"] == 1  # the ungrounded "999" was scrubbed (AGT-OBS-R4)
    # AGT-OBS-R7 / OBS-R4: the terminal status + grounding run landed on the metrics surface.
    assert registry.counter_value(m.GROUNDING_RUNS) == runs_before + 1
    assert (
        registry.counter_value(m.RUN_TERMINAL, labels={"status": "completed"})
        == terminal_before + 1
    )
    assert registry.summary(m.RUN_LATENCY_SECONDS).count >= 1
