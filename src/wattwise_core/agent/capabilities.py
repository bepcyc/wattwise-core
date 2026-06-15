"""The single capability registry + deterministic gather + canonical grounding evidence.

Cited requirements (doc 50): PLAN-R2 (every planner-selected capability carries TYPED
params — date ranges, an activity reference, a closed metric enum — and NEVER a source
name, table name, or raw query string), PLAN-R3 (each capability maps 1:1 to exactly one
canonical-service call; the SAME registry backs the planner's structured plan and the MCP
tool layer — one data path), PLAN-R5 (the athlete scope of every retrieval is the
engine-injected, server-derived ``athlete_id`` argument, NEVER a value read from a
model-selected request), TOOL-R1 (a capability is a thin typed wrapper over one canonical
call — no business logic, no second data path), TOOL-R5 (a capability whose canonical
computation is unavailable records a typed coverage GAP, never a fabricated success), and
GROUND-R7 (the grounder verifies numeric claims against the canonical analytics service
VERBATIM and checks URLs against a first-party allow-list — it never re-derives a number).

This module is the ONE place the Phase-1 capability set is declared. Both the planner's
deterministic :func:`gather` (here) and the MCP tool layer resolve every capability to the
identical :class:`~wattwise_core.analytics.service.AnalyticsService` method, so there is a
single canonical data path and no divergence between what the model can plan and what it
can ground against.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from wattwise_core.agent.capabilities_evidence import CanonicalEvidence
from wattwise_core.agent.capabilities_metrics import MetricName
from wattwise_core.agent.contracts import Capability, RetrievalRequest
from wattwise_core.agent.metric_equivalence import MetricEquivalence
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.observability import runtrace

# --- typed capability parameter schemas (PLAN-R2: typed, never source/table names) ---


class _Params(BaseModel):
    """Base for capability params: closed (extra keys rejected) typed input (PLAN-R2).

    A capability's params describe ONLY analytic intent — date spans, an activity
    reference, a metric selector. They deliberately cannot express a source name, a
    table/column name, or a raw query: those are not fields here, and ``extra='forbid'``
    rejects any attempt to smuggle one in via a model-supplied key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class DateRangeParams(_Params):
    """An inclusive local-date span for an athlete-level metric (PLAN-R2)."""

    from_date: _dt.date = Field(description="Inclusive first local date of the window.")
    to_date: _dt.date = Field(description="Inclusive last local date of the window.")


class ActivityParams(_Params):
    """A reference to one canonical resolved activity (PLAN-R2).

    The id is an opaque canonical activity identifier, never a source-specific key; the
    service resolves it within the engine-scoped athlete (PLAN-R5).
    """

    activity_id: str = Field(min_length=1, description="Canonical activity identifier.")


class WellnessDayParams(_Params):
    """A single wellness local date for a readiness metric (PLAN-R2)."""

    local_date: _dt.date = Field(description="The wellness local date to read.")


# --- the Phase-1 capability registry (one entry == one AnalyticsService method) ---


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        key="weekly_load",
        description="Recent training-load state (CTL/ATL/form) over a date window.",
        service_method="pmc",
        param_schema=DateRangeParams,
    ),
    Capability(
        key="critical_power",
        description="Critical-power / W' model fit over a date window.",
        service_method="critical_power",
        param_schema=DateRangeParams,
    ),
    Capability(
        key="power_curve",
        description="Mean-maximal power curve over a date window.",
        service_method="power_curve",
        param_schema=DateRangeParams,
    ),
    Capability(
        key="load_metrics",
        description="NP/IF/TSS load metrics for one activity.",
        service_method="coggan",
        param_schema=ActivityParams,
    ),
    Capability(
        key="hrv",
        description="Time-domain HRV for one wellness day.",
        service_method="hrv",
        param_schema=WellnessDayParams,
    ),
    Capability(
        key="decoupling",
        description="Aerobic decoupling for one activity.",
        service_method="aerobic_decoupling",
        param_schema=ActivityParams,
    ),
    Capability(
        key="trimp",
        description="Banister HR training-load (TRIMP) for one activity.",
        service_method="trimp",
        param_schema=ActivityParams,
    ),
    Capability(
        key="durability",
        description="Durability / fatigue-resistance: the fresh-vs-fatigued power decrement "
        "(and total work above CP) for one activity.",
        service_method="durability",
        param_schema=ActivityParams,
    ),
)

CAPABILITY_BY_KEY: Mapping[str, Capability] = {c.key: c for c in CAPABILITIES}


# --- deterministic gather (PLAN-R3/R5, TOOL-R5: fail-closed, gaps not fabrication) ---

# A resolver runs ONE canonical call for a capability, scoped by the gather's athlete arg
# (NEVER by the request) and the request's validated typed params. The params object is
# already validated against the capability's ``param_schema`` before dispatch, so each
# resolver narrows it with a checked ``cast`` to its concrete schema.
_Resolver = Callable[[AnalyticsService, str, BaseModel], Awaitable[Any]]


async def _r_weekly_load(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    q = cast(DateRangeParams, p)
    latest_activity_date = getattr(svc, "latest_activity_date", None)
    if callable(latest_activity_date):
        latest = await latest_activity_date(athlete_id)
        if not isinstance(latest, _dt.date):
            return _gap("no_activities", "athlete has no activities to compute training load")
        if latest < q.from_date:
            return _gap(
                "no_recent_activities",
                f"latest activity {latest.isoformat()} predates requested window",
            )
    return await svc.pmc(athlete_id, q.from_date, q.to_date)


async def _r_critical_power(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    # COACH-R6: the deliverable MUST be produced for the athlete's CURRENT sport — the
    # sport-parameterized analytics are consumed for THAT sport, never hardwired cycling.
    # No current sport set ⇒ a typed coverage gap (fail-closed), never a guessed sport.
    sport = await svc.current_sport(athlete_id)
    if sport is None:
        return _gap("no_current_sport", "athlete has no current sport set")
    q = cast(DateRangeParams, p)
    return await svc.critical_power(athlete_id, q.from_date, q.to_date, sport=sport)


async def _r_power_curve(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    # COACH-R6: scope the power curve to the athlete's CURRENT sport (sport-partitioned,
    # ANL-R13); a sport switch must change subsequent deliverables with no engine change.
    sport = await svc.current_sport(athlete_id)
    if sport is None:
        return _gap("no_current_sport", "athlete has no current sport set")
    q = cast(DateRangeParams, p)
    return await svc.power_curve(athlete_id, q.from_date, q.to_date, sport=sport)


async def _r_load_metrics(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    return await svc.coggan(cast(ActivityParams, p).activity_id)


async def _r_hrv(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    return await svc.hrv(athlete_id, cast(WellnessDayParams, p).local_date)


async def _r_decoupling(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    return await svc.aerobic_decoupling(cast(ActivityParams, p).activity_id)


async def _r_trimp(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    return await svc.trimp(cast(ActivityParams, p).activity_id)


async def _r_durability(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    return await svc.durability(cast(ActivityParams, p).activity_id)


RESOLVERS: Mapping[str, _Resolver] = {
    "weekly_load": _r_weekly_load,
    "critical_power": _r_critical_power,
    "power_curve": _r_power_curve,
    "load_metrics": _r_load_metrics,
    "hrv": _r_hrv,
    "decoupling": _r_decoupling,
    "trimp": _r_trimp,
    "durability": _r_durability,
}

# Identity/scope-shaped keys a model-selected request is structurally forbidden from
# carrying: scope is the engine-injected ``athlete_id`` only (PLAN-R5/AGT-SEC-R1). A key
# of this shape in ``req.params`` is an attempted scope override — ignored, and recorded
# as an AGT-OBS-R5a anomaly correlated to the run trace.
_SCOPE_SHAPED_KEYS: frozenset[str] = frozenset(
    {"athlete_id", "athlete", "tenant_id", "tenant", "user_id", "scope", "as_athlete"}
)


@dataclass(frozen=True, slots=True)
class AnomalyEvent:
    """A typed injection/anomaly event for an attempted scope override (AGT-OBS-R5a)."""

    kind: str
    capability: str
    attempted_keys: tuple[str, ...]
    ignored_override: dict[str, Any]
    authenticated_scope: str


@dataclass(frozen=True, slots=True)
class GatherResult:
    """The deterministic gather output: records keyed by capability + any anomalies."""

    records: dict[str, Any]
    anomalies: tuple[AnomalyEvent, ...] = ()


def _gap(reason: str, detail: str = "") -> dict[str, Any]:
    """A recorded coverage GAP (TOOL-R5): an explicit absence, never a fabricated value."""
    return {"available": False, "reason": reason, "detail": detail}


def _scope_override_keys(params: Mapping[str, Any]) -> tuple[str, ...]:
    """Return any scope-shaped keys a model-selected request carried (PLAN-R5)."""
    return tuple(k for k in params if k.casefold() in _SCOPE_SHAPED_KEYS)


async def gather(
    svc: AnalyticsService,
    athlete_id: str,
    requests: list[RetrievalRequest],
) -> GatherResult:
    """Deterministically execute planner-selected capability requests (PLAN-R3/R5).

    Each request resolves to exactly ONE canonical :class:`AnalyticsService` call (the
    SAME call the MCP tool layer makes), scoped to ``athlete_id`` — the engine-injected,
    server-derived identity — NEVER to any athlete-shaped value inside the request
    (PLAN-R5). If a request's params carry an athlete/scope-shaped key, that override is
    IGNORED and an :class:`AnomalyEvent` is emitted (the attempt, the ignored override,
    and the authenticated scope used) for correlation to the run trace (AGT-OBS-R5a). An
    unknown capability or invalid params records a typed coverage GAP (TOOL-R5), never a
    fabricated success, and never raises out of the gather.

    Args:
        svc: the one canonical analytics service (single data path).
        athlete_id: the server-derived athlete scope; the ONLY identity used.
        requests: planner-selected capability requests with typed params.

    Returns:
        A :class:`GatherResult` carrying the capability->record mapping plus any
        scope-override anomaly events.
    """
    out: dict[str, Any] = {}
    anomalies: list[AnomalyEvent] = []
    for req in requests:
        override_keys = _scope_override_keys(req.params)
        if override_keys:
            anomalies.append(
                AnomalyEvent(
                    kind="scope_override_ignored",
                    capability=req.capability,
                    attempted_keys=override_keys,
                    ignored_override={k: req.params[k] for k in override_keys},
                    authenticated_scope=athlete_id,
                )
            )
        # Each capability resolution is one tool call (PLAN-R3, 1:1 to a canonical-service
        # call): open a span so the call is traced with start/end, status, and parent linkage
        # under the run trace (AGT-OBS-R1). The span is a no-op outside a bound run, so the
        # deterministic/offline path is unchanged. Status flips to ``error`` only if the
        # underlying call raises; a fail-closed coverage GAP is a neutralized call, not an error.
        with runtrace.span(req.capability):
            record, anomaly = await _resolve_one(svc, athlete_id, req)
        out[req.capability] = record
        if anomaly is not None:
            anomalies.append(anomaly)
    return GatherResult(records=out, anomalies=tuple(anomalies))


async def _resolve_one(
    svc: AnalyticsService, athlete_id: str, req: RetrievalRequest
) -> tuple[Any, AnomalyEvent | None]:
    """Resolve one request fail-closed: unknown capability / bad params -> typed gap.

    Scope-shaped keys are stripped from the params before validation so a model-supplied
    ``athlete_id`` can never reach the capability schema; identity is ``athlete_id`` only
    (PLAN-R5). The scope-override detection + its anomaly emission happen in :func:`gather`.

    Returns the resolved record (or a typed coverage GAP, TOOL-R5) PLUS an optional
    :class:`AnomalyEvent`: an out-of-registry capability request (PLAN-R3) is a named
    injection/anomaly case the engine MUST emit (AGT-OBS-R5a) — it is not a silent gap. The
    caller threads the returned anomaly onto the run trace + metrics via the production gateway.
    """
    capability = CAPABILITY_BY_KEY.get(req.capability)
    resolver = RESOLVERS.get(req.capability)
    if capability is None or resolver is None:
        anomaly = AnomalyEvent(
            kind="out_of_registry_capability",
            capability=req.capability,
            attempted_keys=(req.capability,),
            ignored_override={"capability": req.capability},
            authenticated_scope=athlete_id,
        )
        return _gap("unknown_capability", req.capability), anomaly
    clean_params = {k: v for k, v in req.params.items() if k.casefold() not in _SCOPE_SHAPED_KEYS}
    try:
        params = capability.param_schema.model_validate(clean_params)
    except ValueError as exc:
        return _gap("invalid_params", type(exc).__name__), None
    return await resolver(svc, athlete_id, params), None


__all__ = [
    "CAPABILITIES",
    "CAPABILITY_BY_KEY",
    "RESOLVERS",
    "ActivityParams",
    "AnomalyEvent",
    "CanonicalEvidence",
    "DateRangeParams",
    "GatherResult",
    "MetricEquivalence",
    "MetricName",
    "WellnessDayParams",
    "gather",
]
