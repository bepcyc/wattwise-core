"""The closed grounding-checkable metric vocabulary (PLAN-R2, GROUND-R7).

A tiny LEAF module holding only the :class:`MetricName` enum so the capability registry
(:mod:`wattwise_core.agent.capabilities`) and the metric-equivalence resolver
(:mod:`wattwise_core.agent.metric_equivalence`) can both depend on it WITHOUT importing each
other (no cycle). It carries no logic — just the closed metric selector the grounder verifies a
claim's value against.
"""

from __future__ import annotations

from enum import StrEnum


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


__all__ = ["MetricName"]
