"""The PRODUCTION engine paths are observable on a real run (AGT-OBS-R1/-R2/-R5a, OBS-R4).

These exercise the SEAMS the live graph actually calls — ``RegistryGateway.gather`` and the
model span recording — not the bypassed low-level helpers, so they catch the GAP_SPEC
"deviates" finding (the gateway DISCARDED ``GatherResult.anomalies`` so a scope override was
ignored but never EMITTED on a live run). A mutation that reverts the gateway to
``return result.records`` (no emit) breaks ``test_registry_gateway_emits_anomaly_on_run``.

Cites: AGT-OBS-R5a (50:475) anomaly emitted on the run trace + counted in metrics; AGT-OBS-R2
(50:467) model span records provider usage + the per-run rollup; OBS-R4 (70:550) grounding
outcomes recorded in production.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from wattwise_core.agent.contracts import RetrievalRequest
from wattwise_core.agent.engine_services import RegistryGateway
from wattwise_core.analytics.pmc import PmcDay
from wattwise_core.analytics.result import Computed, MetricResult
from wattwise_core.observability import metrics as m
from wattwise_core.observability import runtrace

pytestmark = pytest.mark.unit

_ATHLETE = "11111111-1111-1111-1111-111111111111"
_OTHER = "22222222-2222-2222-2222-222222222222"
_DAY = _dt.date(2026, 6, 1)


class _FakeAnalytics:
    """A network-free analytics stand-in that records which athlete it was scoped to."""

    def __init__(self) -> None:
        self.athlete_calls: list[str] = []

    async def pmc(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date, *, seed: Any = None
    ) -> list[MetricResult[PmcDay]]:
        """Return one canonical PMC day, recording the scoping athlete."""
        self.athlete_calls.append(athlete_id)
        return [Computed(value=PmcDay(ctl=50.0, atl=40.0, tsb=10.0))]

    async def critical_power(
        self, athlete_id: str, from_date: _dt.date, to_date: _dt.date, *, sport: str = "cycling"
    ) -> MetricResult[Any]:
        """Return a canonical CP fit (value unused by the gather records), recording the athlete."""
        self.athlete_calls.append(athlete_id)
        return Computed(value={"cp_w": 250.0})

    async def current_sport(self, athlete_id: str) -> str | None:
        """Return a fixed current sport so sport-parameterized resolvers proceed (COACH-R6)."""
        return "cycling"


async def test_registry_gateway_emits_anomaly_on_run() -> None:
    """RegistryGateway.gather EMITS the scope-override anomaly on the trace + metrics (AGT-OBS-R5a).

    The PRODUCTION seam the live graph calls — not the low-level ``capabilities.gather`` the eval
    probes — must surface the neutralized override. A revert to discarding ``result.anomalies``
    leaves the trace empty and the counter flat, failing this test.
    """
    gateway = RegistryGateway(_FakeAnalytics())  # type: ignore[arg-type]
    registry = m.get_registry()
    before = registry.counter_value(
        m.INJECTION_ANOMALIES, labels={"kind": "scope_override_ignored"}
    )
    request = RetrievalRequest(
        "weekly_load",
        {"from_date": _DAY.isoformat(), "to_date": _DAY.isoformat(), "athlete_id": _OTHER},
    )
    with runtrace.run_trace("athlete:conv") as trace:
        records = await gateway.gather(athlete_id=_ATHLETE, requests=[request])
    # The override was ignored: the records resolved under the authenticated athlete.
    assert "weekly_load" in records
    # The anomaly was EMITTED onto the run trace (not discarded) and counted (AGT-OBS-R5a).
    assert len(trace.anomalies) == 1
    anomaly = trace.anomalies[0]
    assert anomaly.kind == "scope_override_ignored"
    assert anomaly.ignored is True
    assert anomaly.authenticated_scope == _ATHLETE
    after = registry.counter_value(m.INJECTION_ANOMALIES, labels={"kind": "scope_override_ignored"})
    assert after == before + 1.0


async def test_clean_gather_emits_no_anomaly() -> None:
    """A request with no scope-shaped key emits no anomaly (only real overrides are flagged)."""
    gateway = RegistryGateway(_FakeAnalytics())  # type: ignore[arg-type]
    request = RetrievalRequest(
        "weekly_load", {"from_date": _DAY.isoformat(), "to_date": _DAY.isoformat()}
    )
    with runtrace.run_trace("athlete:clean") as trace:
        await gateway.gather(athlete_id=_ATHLETE, requests=[request])
    assert trace.anomalies == []


async def test_registry_gateway_traces_each_capability_call() -> None:
    """Each capability resolution emits its own tool-call span through the gateway (AGT-OBS-R1).

    AGT-OBS-R1 (50:465) requires EACH model/tool call to emit a span; a capability resolution is
    one tool call (PLAN-R3, 1:1 to a canonical-service call). The PRODUCTION seam the live graph
    calls (``RegistryGateway.gather`` -> ``capabilities.gather``) must therefore open a span per
    request. A revert that drops the per-capability ``runtrace.span(req.capability)`` leaves no
    capability-named span on the trace, failing this test.
    """
    gateway = RegistryGateway(_FakeAnalytics())  # type: ignore[arg-type]
    window = {"from_date": _DAY.isoformat(), "to_date": _DAY.isoformat()}
    one = RetrievalRequest("weekly_load", dict(window))
    two = RetrievalRequest("critical_power", dict(window))
    with runtrace.run_trace("athlete:spans") as trace:
        await gateway.gather(athlete_id=_ATHLETE, requests=[one, two])
    span_names = [s.name for s in trace.spans]
    # One span per resolved capability, each correctly ended (start/end recorded, status ok).
    assert "weekly_load" in span_names
    assert "critical_power" in span_names
    capability_spans = [s for s in trace.spans if s.name in {"weekly_load", "critical_power"}]
    assert len(capability_spans) == 2
    for span in capability_spans:
        assert span.end is not None  # the span was closed (start/end, AGT-OBS-R1)
        assert span.latency_ms is not None
        assert span.status == "ok"  # a fail-closed gap is a neutralized call, not an error


async def test_registry_gateway_emits_out_of_registry_anomaly() -> None:
    """An out-of-registry capability request emits the AGT-OBS-R5a anomaly (PLAN-R3).

    AGT-OBS-R5a (50:475) explicitly names "an out-of-registry capability request (PLAN-R3)" as a
    case that MUST emit a typed injection/anomaly event correlated to the run trace and counted in
    metrics. The PRODUCTION seam must surface it — not silently return a coverage gap. A revert that
    drops the ``out_of_registry_capability`` anomaly construction leaves the trace empty and the
    counter flat, failing this test.
    """
    gateway = RegistryGateway(_FakeAnalytics())  # type: ignore[arg-type]
    registry = m.get_registry()
    before = registry.counter_value(
        m.INJECTION_ANOMALIES, labels={"kind": "out_of_registry_capability"}
    )
    request = RetrievalRequest("definitely_not_a_capability", {})
    with runtrace.run_trace("athlete:oor") as trace:
        records = await gateway.gather(athlete_id=_ATHLETE, requests=[request])
    # The unknown capability still fails closed to a typed coverage gap (TOOL-R5), never fabricated.
    assert records["definitely_not_a_capability"]["available"] is False
    # AND the engine EMITTED the named injection/anomaly event onto the run trace + metrics.
    assert len(trace.anomalies) == 1
    anomaly = trace.anomalies[0]
    assert anomaly.kind == "out_of_registry_capability"
    assert anomaly.capability == "definitely_not_a_capability"
    assert anomaly.ignored is True
    assert anomaly.authenticated_scope == _ATHLETE
    after = registry.counter_value(
        m.INJECTION_ANOMALIES, labels={"kind": "out_of_registry_capability"}
    )
    assert after == before + 1.0
