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

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from wattwise_core.agent.capabilities import CAPABILITIES, CAPABILITY_BY_KEY, RESOLVERS
from wattwise_core.agent.contracts import Capability
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


def _scope_keys_present(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """Return any identity/scope keys a model tried to supply (TOOL-R3/PLAN-R5)."""
    return tuple(k for k in raw if k.casefold() in _SCOPE_KEYS)


def _scrub_scope_args(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Drop any identity/scope key a model tried to supply (TOOL-R3/PLAN-R5/INJECT-R3).

    Identity comes from :class:`ToolContext` only; a model-supplied ``athlete_id`` (an
    injection probe attempting cross-athlete scope) is ignored. The caller emits an
    AGT-OBS-R5a anomaly event when a scope key was present.
    """
    return {k: v for k, v in raw.items() if k.casefold() not in _SCOPE_KEYS}


def _wrap_result(capability: str, result: Any) -> ToolResult:
    """Wrap a canonical resolver result as an untrusted tool result (TOOL-R2/R5).

    The resolvers (shared with ``gather``) return the canonical analytics envelope, a gap
    marker (``{"available": False, ...}``), or a series/mapping. A gap marker becomes an
    ``available=False`` result; anything else is surfaced as available untrusted data.
    """
    if isinstance(result, Mapping) and result.get("available") is False:
        return ToolResult(
            capability=capability,
            available=False,
            payload=dict(result),
            reason=str(result.get("reason", "unavailable")),
        )
    payload = _to_payload(result)
    return ToolResult(capability=capability, available=True, payload=payload)


def _to_payload(result: Any) -> dict[str, Any]:
    """Normalise a resolver result into a jsonable tool payload (TOOL-R2)."""
    to_jsonable = getattr(result, "to_jsonable", None)
    if callable(to_jsonable):
        out = to_jsonable()
        return out if isinstance(out, dict) else {"value": out}
    if isinstance(result, Mapping):
        return dict(result)
    if isinstance(result, (list, tuple)):
        return {"items": [_to_payload(r) for r in result]}
    return {"value": result}


def _make_handler(capability: Capability) -> _Handler:
    """Build a tool handler that calls the SAME registry resolver as ``gather`` (TOOL-R1).

    The model's arguments are scrubbed of scope keys (TOOL-R3) and validated against the
    capability's ``param_schema`` (PLAN-R2); dispatch is the single ``RESOLVERS`` entry
    keyed verbatim by ``capability.key`` — there is no second data path.
    """
    resolver = RESOLVERS[capability.key]

    async def handler(
        ctx: ToolContext, svc: AnalyticsService, args: Mapping[str, Any]
    ) -> ToolResult:
        scrubbed = _scrub_scope_args(args)
        try:
            params = capability.param_schema.model_validate(scrubbed)
        except ValueError as exc:
            return ToolResult(
                capability=capability.key,
                available=False,
                payload={"available": False},
                reason=f"invalid_params:{type(exc).__name__}",
            )
        result = await resolver(svc, ctx.athlete_id, params)
        return _wrap_result(capability.key, result)

    return handler


# --- the in-process tool registry (TOOL-R1b: no subprocess), DERIVED from CAPABILITIES ---

# ONE registry, two surfaces (PLAN-R3/TOOL-R1): every MCP tool is built FROM the single
# CAPABILITIES tuple so a tool key == a capability key VERBATIM and both resolve the SAME
# canonical service call. There is no hand-authored parallel key set.
_TOOLS: tuple[Tool, ...] = tuple(
    Tool(c.key, c.description, _make_handler(c)) for c in CAPABILITIES
)


class ToolRegistry:
    """In-process registry of read-only tools over the one canonical service (TOOL-R1b).

    No per-request subprocess spawn, no stdio/JSON-RPC transport: a tool call is a
    direct in-process ``await`` into :class:`AnalyticsService` (the SAME code the
    deterministic ``gather`` calls). The registry is built FROM the single CAPABILITIES
    registry so tool keys match the planner's capability keys verbatim (PLAN-R3/TOOL-R1).
    The registry is the model's ONLY I/O surface (TOOL-R4); every call is scoped by the
    engine-injected :class:`ToolContext` (TOOL-R3), individually returnable as
    traced/untrusted output.
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
        (TOOL-R2/INJECT-R5). A scope-shaped argument is detectable via
        :func:`scope_override_attempted` so the engine can emit an AGT-OBS-R5a anomaly.
        """
        if capability not in CAPABILITY_BY_KEY:
            return ToolResult(
                capability=capability,
                available=False,
                payload={"available": False},
                reason="unknown_capability",
            )
        tool = self._tools.get(capability)
        if tool is None:
            return ToolResult(
                capability=capability,
                available=False,
                payload={"available": False},
                reason="unknown_capability",
            )
        return await tool.handler(context, service, arguments or {})


def scope_override_attempted(arguments: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return the scope-shaped keys a model supplied (for AGT-OBS-R5a anomaly emission)."""
    return _scope_keys_present(arguments or {})


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "scope_override_attempted",
]
