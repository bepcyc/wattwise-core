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
from enum import StrEnum
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from wattwise_core.agent.contracts import Capability, RetrievalRequest
from wattwise_core.analytics.result import MetricResult, is_computed
from wattwise_core.analytics.service import AnalyticsService

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


class MetricName(StrEnum):
    """Closed vocabulary of grounding-checkable scalar metrics (PLAN-R2, GROUND-R7).

    A typed metric SELECTOR — the value-side of a claim the grounder verifies against the
    canonical service. It is a closed enum (never a free string, never a column name) so a
    model can only request a metric this engine actually computes.
    """

    CTL = "ctl"
    ATL = "atl"
    TSB = "tsb"
    # Athlete-facing synonym for canonical TSB (CTL(d-1)-ATL(d-1)); resolves to the SAME
    # PmcDay.tsb value as TSB (a pure alias, not a second computation).
    FORM = "form"
    CRITICAL_POWER_W = "critical_power_w"
    W_PRIME_J = "w_prime_j"
    HRV_RMSSD_MS = "hrv_rmssd_ms"


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
    return await svc.pmc(athlete_id, q.from_date, q.to_date)


async def _r_critical_power(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    q = cast(DateRangeParams, p)
    return await svc.critical_power(athlete_id, q.from_date, q.to_date)


async def _r_power_curve(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    q = cast(DateRangeParams, p)
    return await svc.power_curve(athlete_id, q.from_date, q.to_date)


async def _r_load_metrics(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    return await svc.coggan(cast(ActivityParams, p).activity_id)


async def _r_hrv(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    return await svc.hrv(athlete_id, cast(WellnessDayParams, p).local_date)


async def _r_decoupling(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    return await svc.aerobic_decoupling(cast(ActivityParams, p).activity_id)


async def _r_trimp(svc: AnalyticsService, athlete_id: str, p: BaseModel) -> Any:
    return await svc.trimp(cast(ActivityParams, p).activity_id)


RESOLVERS: Mapping[str, _Resolver] = {
    "weekly_load": _r_weekly_load,
    "critical_power": _r_critical_power,
    "power_curve": _r_power_curve,
    "load_metrics": _r_load_metrics,
    "hrv": _r_hrv,
    "decoupling": _r_decoupling,
    "trimp": _r_trimp,
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
        out[req.capability] = await _resolve_one(svc, athlete_id, req)
    return GatherResult(records=out, anomalies=tuple(anomalies))


async def _resolve_one(svc: AnalyticsService, athlete_id: str, req: RetrievalRequest) -> Any:
    """Resolve one request fail-closed: unknown capability / bad params -> typed gap.

    Scope-shaped keys are stripped from the params before validation so a model-supplied
    ``athlete_id`` can never reach the capability schema; identity is ``athlete_id`` only
    (PLAN-R5). The detection + anomaly emission happens in :func:`gather`.
    """
    capability = CAPABILITY_BY_KEY.get(req.capability)
    resolver = RESOLVERS.get(req.capability)
    if capability is None or resolver is None:
        return _gap("unknown_capability", req.capability)
    clean_params = {k: v for k, v in req.params.items() if k.casefold() not in _SCOPE_SHAPED_KEYS}
    try:
        params = capability.param_schema.model_validate(clean_params)
    except ValueError as exc:
        return _gap("invalid_params", type(exc).__name__)
    return await resolver(svc, athlete_id, params)


# --- canonical grounding evidence (GROUND-R7: VERBATIM numbers + first-party URLs) ---

# First-party hosts whose links the deliverable may keep (GROUND-R7 allow-list). Anything
# not on this exact-host set is scrubbed by the grounder ("when in doubt, scrub").
_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "wattwise.app",
        "www.wattwise.app",
        "docs.wattwise.app",
    }
)


def _latest_pmc_scalar(series: list[MetricResult[Any]], field: str) -> float | None:
    """The named scalar of the latest computed PMC day, or ``None`` (fail-closed)."""
    for day in reversed(series):
        if is_computed(day):
            return float(getattr(day.value, field))
    return None


def _scalar_of(metric: MetricName, value: object) -> float | None:
    """Read the requested scalar from a Computed value object VERBATIM (GROUND-R7)."""
    attr = {
        MetricName.CRITICAL_POWER_W: "cp_w",
        MetricName.W_PRIME_J: "w_prime_j",
        MetricName.HRV_RMSSD_MS: "rmssd_ms",
    }[metric]
    return float(getattr(value, attr))


class CanonicalEvidence:
    """Read-only canonical evidence for the grounder (GROUND-R7, contracts.GroundingEvidence).

    Implements :class:`~wattwise_core.agent.contracts.GroundingEvidence`. Numbers come
    VERBATIM from the canonical :class:`AnalyticsService` for the engine-scoped athlete —
    this layer NEVER re-derives or rounds a value, it only reads what the service computed.
    A metric the service cannot compute returns ``None`` (the grounder scrubs the claim);
    ``url_allowed`` is a first-party exact-host allow-list.
    """

    def __init__(self, svc: AnalyticsService, athlete_id: str) -> None:
        self._svc = svc
        self._athlete_id = athlete_id

    async def metric_value(self, metric: str, as_of: str | None) -> float | None:
        """The canonical value of ``metric`` as-of a date, or ``None`` (GROUND-R7).

        ``metric`` must be a member of :class:`MetricName`; an unknown metric, a missing
        ``as_of`` where one is required, or an uncomputable result all yield ``None`` so
        the grounder scrubs the claim rather than emitting an unverifiable number.
        """
        try:
            name = MetricName(metric)
        except ValueError:
            return None
        if name in (MetricName.CTL, MetricName.ATL, MetricName.TSB, MetricName.FORM):
            return await self._pmc_scalar(name, as_of)
        if name in (MetricName.CRITICAL_POWER_W, MetricName.W_PRIME_J):
            return await self._cp_scalar(name, as_of)
        return await self._hrv_scalar(name, as_of)

    async def _pmc_scalar(self, name: MetricName, as_of: str | None) -> float | None:
        day = self._date(as_of)
        if day is None:
            return None
        # FORM is the athlete-facing alias of TSB: both read the canonical PmcDay.tsb field.
        field = MetricName.TSB.value if name is MetricName.FORM else name.value
        series = await self._svc.pmc(self._athlete_id, day, day)
        return _latest_pmc_scalar(series, field)

    async def _cp_scalar(self, name: MetricName, as_of: str | None) -> float | None:
        day = self._date(as_of)
        if day is None:
            return None
        fit = await self._svc.critical_power(self._athlete_id, day, day)
        if not is_computed(fit):
            return None
        return _scalar_of(name, fit.value)

    async def _hrv_scalar(self, name: MetricName, as_of: str | None) -> float | None:
        day = self._date(as_of)
        if day is None:
            return None
        result = await self._svc.hrv(self._athlete_id, day)
        if not is_computed(result):
            return None
        return _scalar_of(name, result.value)

    @staticmethod
    def _date(as_of: str | None) -> _dt.date | None:
        """Parse an ISO date, or ``None`` (a metric with no usable as-of is unverifiable)."""
        if as_of is None:
            return None
        try:
            return _dt.date.fromisoformat(as_of)
        except ValueError:
            return None

    def url_allowed(self, url: str) -> bool:
        """True iff ``url`` is an https first-party link on the exact-host allow-list."""
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.hostname in _ALLOWED_HOSTS


__all__ = [
    "CAPABILITIES",
    "CAPABILITY_BY_KEY",
    "RESOLVERS",
    "ActivityParams",
    "AnomalyEvent",
    "CanonicalEvidence",
    "DateRangeParams",
    "GatherResult",
    "MetricName",
    "WellnessDayParams",
    "gather",
]
