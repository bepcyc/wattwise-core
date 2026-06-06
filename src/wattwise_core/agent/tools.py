"""The MCP tool layer: thin read-only wrappers over the one capability registry.

Cited requirements (doc 50): TOOL-R1 (every tool = thin typed wrapper over exactly
ONE capability-registry entry, the SAME canonical service call the planner's
deterministic ``gather`` and the REST API use — no business logic, no second data
path, no source-client access), TOOL-R1b (in-process registry; NO per-request
subprocess spawn), TOOL-R2 (typed inputs/outputs; outputs wrapped as untrusted data),
TOOL-R3 (execution scoped by engine-injected authenticated identity, NEVER by a
model-supplied argument), TOOL-R4 (tools are the model's ONLY I/O), TOOL-R5 (a tool
whose canonical service is unavailable returns a typed ``unavailable`` result, never
fabricated success). Also INJECT-R5 (a tool MUST NOT inject instructions into the
model's instruction region via its return value) and AGT-SEC-R1/PLAN-R5 (identity is
structural, not a model-trusted filter).

This module is a MODEL-INTERFACE only (TOOL-R1a): it dispatches to IDENTICAL service
code as the deterministic ``gather`` path. The single canonical service it wraps is
:class:`~wattwise_core.analytics.service.AnalyticsService`; there is no other data
path here.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from wattwise_core.analytics.result import MetricResult, is_computed
from wattwise_core.analytics.service import AnalyticsService

# Argument keys a model is structurally forbidden from supplying: identity/scope is
# injected by the engine, never read from a tool-call argument (TOOL-R3/AGT-SEC-R1).
_SCOPE_KEYS = frozenset({"athlete_id", "athlete", "tenant_id", "tenant", "user_id"})


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Engine-injected execution scope for a tool call (TOOL-R3).

    ``athlete_id`` is derived from the authenticated request context only (AGT-SEC-R1)
    and is the ONLY identity a tool ever uses. It is never set from a model/tool
    argument; an ``athlete_id``-shaped key in the model's arguments is ignored.
    """

    athlete_id: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A tool's return value, wrapped as UNTRUSTED DATA (TOOL-R2/INJECT-R5).

    ``available`` mirrors the canonical result's fail-closed state (TOOL-R5): a missing
    canonical computation yields ``available=False`` with a typed ``reason`` recorded
    in ``coverage_gaps`` by the caller — never a fabricated success. ``untrusted`` is
    a structural marker so the engine wraps the ``payload`` in a delimited data
    envelope and never concatenates it into the instruction region (INJECT-R5).
    """

    capability: str
    available: bool
    payload: dict[str, Any]
    reason: str | None = None
    untrusted: bool = field(default=True, init=False)


# A bound tool handler closes over the injected identity + the canonical service.
_Handler = Callable[["ToolContext", AnalyticsService, Mapping[str, Any]], Awaitable[ToolResult]]


@dataclass(frozen=True, slots=True)
class Tool:
    """One model-facing tool: a thin typed wrapper over a single capability (TOOL-R1).

    ``capability`` is the registry key the planner also selects (one registry, two
    surfaces). ``handler`` dispatches to exactly one :class:`AnalyticsService` call.
    """

    capability: str
    description: str
    handler: _Handler


def _scrub_scope_args(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Drop any identity/scope key a model tried to supply (TOOL-R3/PLAN-R5/INJECT-R3).

    Identity comes from :class:`ToolContext` only; a model-supplied ``athlete_id`` (an
    injection probe attempting cross-athlete scope) is ignored. The caller emits an
    AGT-OBS-R5a anomaly event when a scope key was present.
    """
    return {k: v for k, v in raw.items() if k.casefold() not in _SCOPE_KEYS}


def _wrap_metric(capability: str, result: MetricResult[Any]) -> ToolResult:
    """Wrap a canonical :data:`MetricResult` as an untrusted tool result (TOOL-R2/R5)."""
    if is_computed(result):
        return ToolResult(capability=capability, available=True, payload=result.to_jsonable())
    return ToolResult(
        capability=capability,
        available=False,
        payload=result.to_jsonable(),
        reason=result.reason.value,
    )


def _date(value: Any) -> _dt.date:
    """Coerce a tool argument to a date (ISO string or date); typed-input contract."""
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value))


# --- per-capability handlers (each = ONE AnalyticsService call, TOOL-R1) ---


async def _pmc(ctx: ToolContext, svc: AnalyticsService, args: Mapping[str, Any]) -> ToolResult:
    a = _scrub_scope_args(args)
    series = await svc.pmc(ctx.athlete_id, _date(a["from_date"]), _date(a["to_date"]))
    last = series[-1] if series else None
    if last is None or not is_computed(last):
        return ToolResult(
            capability="training_load_window",
            available=False,
            payload={"available": False},
            reason="insufficient_data",
        )
    return _wrap_metric("training_load_window", last)


async def _power_curve(
    ctx: ToolContext, svc: AnalyticsService, args: Mapping[str, Any]
) -> ToolResult:
    a = _scrub_scope_args(args)
    curve = await svc.power_curve(ctx.athlete_id, _date(a["from_date"]), _date(a["to_date"]))
    payload = {
        str(d): res.to_jsonable() for d, res in curve.items() if is_computed(res)
    }
    return ToolResult(
        capability="power_curve",
        available=bool(payload),
        payload={"windows": payload},
        reason=None if payload else "insufficient_data",
    )


async def _critical_power(
    ctx: ToolContext, svc: AnalyticsService, args: Mapping[str, Any]
) -> ToolResult:
    a = _scrub_scope_args(args)
    fit = await svc.critical_power(ctx.athlete_id, _date(a["from_date"]), _date(a["to_date"]))
    return _wrap_metric("critical_power_fit", fit)


async def _hrv(ctx: ToolContext, svc: AnalyticsService, args: Mapping[str, Any]) -> ToolResult:
    a = _scrub_scope_args(args)
    result = await svc.hrv(ctx.athlete_id, _date(a["local_date"]))
    return _wrap_metric("readiness_hrv", result)


async def _activity_load(
    ctx: ToolContext, svc: AnalyticsService, args: Mapping[str, Any]
) -> ToolResult:
    a = _scrub_scope_args(args)
    result = await svc.coggan(str(a["activity_id"]))
    return _wrap_metric("activity_load_metrics", result)


async def _decoupling(
    ctx: ToolContext, svc: AnalyticsService, args: Mapping[str, Any]
) -> ToolResult:
    a = _scrub_scope_args(args)
    result = await svc.aerobic_decoupling(str(a["activity_id"]))
    return _wrap_metric("activity_decoupling", result)


# --- the in-process tool registry (TOOL-R1b: no subprocess) ---

_TOOLS: tuple[Tool, ...] = (
    Tool("training_load_window", "Recent training-load state (CTL/ATL/form).", _pmc),
    Tool("power_curve", "Mean-maximal power curve over a window.", _power_curve),
    Tool("critical_power_fit", "Critical-power / W' model fit.", _critical_power),
    Tool("readiness_hrv", "Time-domain HRV for a wellness day.", _hrv),
    Tool("activity_load_metrics", "NP/IF/TSS load metrics for one activity.", _activity_load),
    Tool("activity_decoupling", "Aerobic decoupling for one activity.", _decoupling),
)


class ToolRegistry:
    """In-process registry of read-only tools over the one canonical service (TOOL-R1b).

    No per-request subprocess spawn, no stdio/JSON-RPC transport: a tool call is a
    direct in-process ``await`` into :class:`AnalyticsService` (the SAME code the
    deterministic ``gather`` calls). The registry is the model's ONLY I/O surface
    (TOOL-R4); every call is scoped by the engine-injected :class:`ToolContext`
    (TOOL-R3), individually returnable as traced/untrusted output.
    """

    def __init__(self, tools: tuple[Tool, ...] = _TOOLS) -> None:
        self._tools: dict[str, Tool] = {t.capability: t for t in tools}

    @property
    def capabilities(self) -> tuple[str, ...]:
        """The registry keys exposed to the model (shared verbatim with the planner)."""
        return tuple(self._tools)

    async def invoke(
        self,
        *,
        capability: str,
        context: ToolContext,
        service: AnalyticsService,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """Invoke one tool, scoped by the injected identity (TOOL-R3), fail-closed.

        An out-of-registry capability returns a typed ``unavailable`` result (the
        caller re-plans, PLAN-R3), never a crash. The model's ``arguments`` never
        carry identity: scope comes from ``context`` (TOOL-R3); any scope-shaped key
        is scrubbed inside the handler. The result is always wrapped as untrusted data
        (TOOL-R2/INJECT-R5).
        """
        tool = self._tools.get(capability)
        if tool is None:
            return ToolResult(
                capability=capability,
                available=False,
                payload={"available": False},
                reason="unknown_capability",
            )
        return await tool.handler(context, service, arguments or {})


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
]
