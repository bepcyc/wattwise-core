"""Capability-registry parity tests: gather route == MCP-tool route (ARCH-R28, MCP-R8).

Cited requirements (doc 10): ARCH-R28 (a capability-registry test MUST assert that the
deterministic ``gather`` route and the MCP-tool route expose the SAME capability registry
(parity), that each capability resolves to a single L5 entry point, and that ``gather``
issues zero model calls), MCP-R8 (a first-party MCP tool and the equivalent deterministic
``gather`` call MUST dispatch to IDENTICAL service code, in-process, with NO per-request
subprocess spawn), MCP-R5/ARCH-R5 (single L5 entry per capability), and TOOL-R3/PLAN-R5
(identity is the engine-injected, server-derived athlete id — never a model argument).

Offline and network-free: a recording in-memory analytics service captures the exact
canonical (method, args) calls each route makes, so route parity is asserted behaviorally
— not by re-stating the registry literals. The zero-model-call test wires a poison
``ChatModel`` (raises on ANY call) through a gather-only graph run (``interrupt_after``
the ``gather`` node), proving the deterministic plan->gather spine touches no model.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from wattwise_core.agent.capabilities import (
    CAPABILITIES,
    CAPABILITY_BY_KEY,
    RESOLVERS,
    ActivityParams,
    DateRangeParams,
    WellnessDayParams,
    gather,
)
from wattwise_core.agent.contracts import AgentState, RetrievalRequest
from wattwise_core.agent.graph import AgentServices, build_graph
from wattwise_core.agent.tools import ToolContext, ToolRegistry
from wattwise_core.analytics.service import AnalyticsService

pytestmark = pytest.mark.unit

_ATHLETE = "11111111-1111-1111-1111-111111111111"
_OTHER_ATHLETE = "22222222-2222-2222-2222-222222222222"
_DAY = _dt.date(2026, 6, 1)

# Valid typed args per param schema, so every registry capability can be driven through
# BOTH routes with the same inputs (PLAN-R2 typed params; no source/table names).
_ARGS_BY_SCHEMA: dict[type[BaseModel], dict[str, Any]] = {
    DateRangeParams: {"from_date": _DAY.isoformat(), "to_date": _DAY.isoformat()},
    ActivityParams: {"activity_id": "act-1"},
    WellnessDayParams: {"local_date": _DAY.isoformat()},
}


class RecordingAnalytics:
    """An in-memory canonical-service stand-in recording every (method, args) call.

    Both routes (deterministic ``gather`` and ``ToolRegistry.invoke``) are run against
    separate instances; identical recorded call sequences prove identical L5 dispatch
    (MCP-R8: "dispatch to identical service code"), without trusting registry literals.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def current_sport(self, athlete_id: str) -> str | None:
        self.calls.append(("current_sport", (athlete_id,)))
        return "cycling"

    async def pmc(self, athlete_id: str, from_date: _dt.date, to_date: _dt.date) -> Any:
        self.calls.append(("pmc", (athlete_id, from_date, to_date)))
        return {"ctl": 50.0}

    async def critical_power(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date, *, sport: str
    ) -> Any:
        self.calls.append(("critical_power", (athlete_id, from_date, to_date, sport)))
        return {"cp_w": 280.0}

    async def power_curve(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date, *, sport: str
    ) -> Any:
        self.calls.append(("power_curve", (athlete_id, from_date, to_date, sport)))
        return {"60": 300.0}

    async def coggan(self, activity_id: str) -> Any:
        self.calls.append(("coggan", (activity_id,)))
        return {"tss": 88.0}

    async def aerobic_decoupling(self, activity_id: str) -> Any:
        self.calls.append(("aerobic_decoupling", (activity_id,)))
        return {"decoupling_pct": 4.2}

    async def trimp(self, activity_id: str) -> Any:
        self.calls.append(("trimp", (activity_id,)))
        return {"trimp": 77.0}

    async def durability(self, activity_id: str) -> Any:
        self.calls.append(("durability", (activity_id,)))
        return {"decrement_pct": 8.0}

    async def hrv(self, athlete_id: str, local_date: _dt.date) -> Any:
        self.calls.append(("hrv", (athlete_id, local_date)))
        return {"rmssd_ms": 42.0}


def _svc() -> Any:
    """A fresh recording fake duck-typed as the canonical analytics service."""
    return RecordingAnalytics()


# --------------------------------------------------------------------------- #
# Registry parity: ONE registry, two surfaces (ARCH-R28, MCP-R8, PLAN-R3)      #
# --------------------------------------------------------------------------- #


def test_gather_and_mcp_routes_expose_the_same_capability_registry() -> None:
    """ARCH-R28 parity, BOTH directions: tool surface == gather surface == registry.

    Every capability the deterministic ``gather`` can resolve (``RESOLVERS`` /
    ``CAPABILITY_BY_KEY``) is exposed as exactly one MCP tool, and every MCP tool maps
    back to a registry capability — no tool without a capability, no capability without
    a tool, no hand-authored parallel key set (Principle B: one data path).
    """
    registry_keys = {c.key for c in CAPABILITIES}
    tool_keys = ToolRegistry().capabilities
    # MCP -> registry: every exposed tool is a registry capability.
    assert set(tool_keys) <= registry_keys
    # registry -> MCP: every registry capability is exposed as a tool.
    assert registry_keys <= set(tool_keys)
    # The gather route resolves against the SAME key set.
    assert set(RESOLVERS) == registry_keys
    assert set(CAPABILITY_BY_KEY) == registry_keys


def test_each_capability_has_exactly_one_entry_on_every_surface() -> None:
    """ARCH-R28: one L5 capability-registry entry per capability — no duplicates.

    The registry tuple, the gather resolver map, and the derived MCP tool list each
    carry every capability exactly once, so neither surface can shadow or fork an
    entry (single L5 entry point per capability, ARCH-R5/MCP-R5).
    """
    keys = [c.key for c in CAPABILITIES]
    assert len(keys) == len(set(keys)), "registry keys are unique"
    tool_keys = ToolRegistry().capabilities
    assert len(tool_keys) == len(set(tool_keys)), "tool keys are unique"
    assert len(tool_keys) == len(keys), "exactly one tool per capability"
    # Each capability names exactly one canonical service method, and it exists on the
    # REAL AnalyticsService surface (the single L5 entry point).
    for cap in CAPABILITIES:
        assert hasattr(AnalyticsService, cap.service_method), cap.key


async def test_both_routes_dispatch_to_identical_l5_service_calls() -> None:
    """MCP-R8: a first-party tool and the equivalent gather call hit IDENTICAL L5 code.

    For EVERY registry capability, the deterministic ``gather`` route and the
    ``ToolRegistry.invoke`` route are driven with the same typed args against separate
    recording services; the recorded canonical (method, args) sequences must be
    IDENTICAL, and must include the capability's declared ``service_method`` — choosing
    the MCP route never changes which canonical code runs (no parallel data path).
    """
    for cap in CAPABILITIES:
        args = _ARGS_BY_SCHEMA[cap.param_schema]
        gather_svc, tool_svc = _svc(), _svc()
        gathered = await gather(gather_svc, _ATHLETE, [RetrievalRequest(cap.key, dict(args))])
        result = await ToolRegistry().invoke(
            capability=cap.key,
            context=ToolContext(athlete_id=_ATHLETE),
            service=tool_svc,
            arguments=dict(args),
        )
        assert gather_svc.calls == tool_svc.calls, f"{cap.key}: routes diverged"
        assert cap.service_method in {m for m, _ in tool_svc.calls}, cap.key
        # Both routes resolved (the recording fake always computes).
        assert cap.key in gathered.records
        assert result.available is True


async def test_mcp_tool_dispatch_is_in_process_with_no_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP-R8/TOOL-R1b: tool invocation is an in-process await — NO per-request spawn.

    Any subprocess creation (``subprocess.Popen`` is the choke point ``subprocess.run``
    also uses, plus asyncio's exec/shell spawners) is poisoned to raise; a tool call
    must still succeed, proving dispatch is a direct in-process call into the canonical
    service, not a stdio/JSON-RPC subprocess transport.
    """

    def _no_spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("MCP-R8 violated: per-request subprocess spawn attempted")

    monkeypatch.setattr(subprocess, "Popen", _no_spawn)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _no_spawn)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", _no_spawn)
    svc = _svc()
    result = await ToolRegistry().invoke(
        capability="hrv",
        context=ToolContext(athlete_id=_ATHLETE),
        service=svc,
        arguments={"local_date": _DAY.isoformat()},
    )
    assert result.available is True
    assert svc.calls == [("hrv", (_ATHLETE, _DAY))]


async def test_mcp_tool_never_accepts_a_caller_supplied_athlete_id() -> None:
    """TOOL-R3/PLAN-R5: identity is server-derived; a model-supplied athlete id is inert.

    The tool call smuggles another athlete's id in its arguments; the canonical call
    must run under the engine-injected ``ToolContext`` identity ONLY — the override is
    scrubbed, never adopted (fail-closed identity, AGT-SEC-R1).
    """
    svc = _svc()
    result = await ToolRegistry().invoke(
        capability="hrv",
        context=ToolContext(athlete_id=_ATHLETE),
        service=svc,
        arguments={"local_date": _DAY.isoformat(), "athlete_id": _OTHER_ATHLETE},
    )
    assert result.available is True
    assert svc.calls == [("hrv", (_ATHLETE, _DAY))]
    assert all(_OTHER_ATHLETE not in call_args for _, call_args in svc.calls)


# --------------------------------------------------------------------------- #
# gather issues ZERO model calls (ARCH-R28, MCP-R5/R7: deterministic step)     #
# --------------------------------------------------------------------------- #


class PoisonModel:
    """A ``ChatModel`` that FAILS on any call: proves a path is model-free (ARCH-R28)."""

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        raise AssertionError("ARCH-R28 violated: gather path issued a model call")

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        raise AssertionError("ARCH-R28 violated: gather path issued a model call")


class _OneShotPlanner:
    """A deterministic planner emitting one typed registry request (no model)."""

    async def plan(
        self, *, request_text: str | None, gaps: Sequence[str], already: Sequence[str]
    ) -> Sequence[RetrievalRequest]:
        return [
            RetrievalRequest(
                "weekly_load",
                {"from_date": _DAY.isoformat(), "to_date": _DAY.isoformat()},
            )
        ]


class _GatherGateway:
    """The production gather (capabilities.gather) over the recording service."""

    def __init__(self, svc: RecordingAnalytics) -> None:
        self._svc = svc

    async def gather(
        self, *, athlete_id: str, requests: Sequence[RetrievalRequest]
    ) -> Mapping[str, Any]:
        return (await gather(self._svc, athlete_id, list(requests))).records


class _NoCoverage:
    """A coverage assessor reporting no gaps (keeps the run on the happy spine)."""

    def assess(self, *, request_text: str | None, retrieved: Mapping[str, Any]) -> set[str]:
        return set()


class _UnreachedGrounder:
    """A grounder that must never run in a gather-only (interrupted) graph run."""

    async def ground(
        self,
        *,
        athlete_id: str,
        draft: str,
        retrieved: Mapping[str, Any],
        request_text: str | None = None,
        active_constraints: object = None,
        evidence_claims: object = None,
    ) -> Any:
        raise AssertionError("ground must not run in a gather-only run")


async def test_gather_issues_zero_model_calls() -> None:
    """ARCH-R28: the deterministic plan->gather spine makes ZERO model calls.

    A poison model (raises on ANY ``structured``/``compose`` call) is wired into the
    REAL compiled graph, and the run is interrupted immediately AFTER the ``gather``
    node. The run reaches the interrupt with canonical evidence retrieved — purely
    deterministic resolution against the capability registry — so no model was invoked
    anywhere on ingest -> plan_retrieval -> gather (MCP-R5/MCP-R7: gather is a DIRECT
    typed in-process path, never a model/tool-choice path).
    """
    svc_rec = _svc()
    services = AgentServices(
        planner=_OneShotPlanner(),
        gateway=_GatherGateway(svc_rec),
        coverage=_NoCoverage(),
        grounder=_UnreachedGrounder(),
    )
    graph = build_graph(PoisonModel(), services, InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "gather-only"}}
    state = AgentState(
        athlete_id=_ATHLETE,
        trigger="user_turn",
        request_text="how is my fitness trending?",
        locale="en",
        idempotency_key="idem-parity-1",
    )

    out = await graph.ainvoke(state, config=config, interrupt_after=["gather"])

    # gather DID run (one real canonical resolution, scoped to the server athlete)...
    assert out["tool_iterations"] == 1
    assert svc_rec.calls == [("pmc", (_ATHLETE, _DAY, _DAY))]
    assert out["retrieved"], "gather merged canonical records into state"
    # ...and the poison model never fired: zero model calls on the deterministic path
    # (the PoisonModel would have raised out of ainvoke otherwise).
