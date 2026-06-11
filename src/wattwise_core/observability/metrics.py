"""Production metrics surface for the engine + the trustworthy agent (OBS-R5/-R4, AGT-OBS-R7).

The trust-critical agent has to be observable in PRODUCTION, not only in the offline eval
(AGT-OBS-R4/-R7, OBS-R4, CI-R8). This module is the single in-process metrics registry the
engine records to and the ``/metrics`` endpoint scrapes:

- **OBS-R5** operational metrics: request rate/latency/error rate per endpoint exposed as
  counters + latency summaries.
- **OBS-R4 / AGT-OBS-R4** agent-quality signals recorded so regressions are observable in
  production and correlate with the offline eval: grounding pass/scrub outcomes,
  self-correction (reflection) iterations, structured-output validation failures, refusals.
- **AGT-OBS-R7** alertable health/quality signals: rolling grounding-scrub rate,
  structured-output validation-failure rate, reflection-exhaustion rate,
  ``degraded``/``budget_exceeded`` rates, p50/p95 latency, and cost per run. The registry
  exposes the raw counters + latency summaries; a sustained regression in any signal is
  alertable from them (the alerting rule lives in the platform, OBS-R8).

The registry is a small, dependency-free Prometheus-text exposition surface (no new runtime
dependency): a counter is created lazily on first increment, so a never-incremented metric
reads ``0`` rather than failing. It is process-local (one per worker) — the standard
Prometheus client-library model; the platform scrapes each worker. Metric NAMES are a fixed
vocabulary declared here (a stable contract the alerting rules bind to), not operator-tunable.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

# Fixed metric-name vocabulary (the stable alerting contract; not operator-tunable, like the
# logging field allowlist). Names follow the Prometheus convention (unit suffix on summaries).
GROUNDING_RUNS = "wattwise_agent_grounding_runs_total"
GROUNDING_SCRUBS = "wattwise_agent_grounding_scrubs_total"
# Issue #10 binding-faithful grounding signals (proposed GROUND-R10/R11): binding-guard
# events (rebinds + residual violations, labelled by event + mode so a SHADOW rollout is
# observable before ENFORCE — a drifting claim extractor shows up HERE), entailment-gate
# checks/vetoes, and verifier-unavailable degradations (the "deterministic-layers-only +
# recorded" fail-closed path — alertable, AGT-OBS-R7).
GROUNDING_BINDING_EVENTS = "wattwise_agent_grounding_binding_events_total"
ENTAILMENT_CHECKS = "wattwise_agent_grounding_entailment_checks_total"
ENTAILMENT_VETOES = "wattwise_agent_grounding_entailment_vetoes_total"
ENTAILMENT_UNAVAILABLE = "wattwise_agent_grounding_entailment_unavailable_total"
VALIDATION_FAILURES = "wattwise_agent_structured_validation_failures_total"
REFLECTIONS = "wattwise_agent_reflections_total"
REFLECTION_EXHAUSTIONS = "wattwise_agent_reflection_exhaustions_total"
REFUSALS = "wattwise_agent_refusals_total"
INJECTION_ANOMALIES = "wattwise_agent_injection_anomalies_total"
TIER_ESCALATIONS = "wattwise_agent_tier_escalations_total"  # labelled by node+tier (MODEL-R2)
LANGUAGE_FALLBACKS = "wattwise_agent_language_fallbacks_total"  # labelled by requested (LANG-R4)
RUN_TERMINAL = "wattwise_agent_run_terminal_total"  # labelled by terminal status
RUN_COST_USD = "wattwise_agent_run_cost_usd"  # summary (per-run cost)
RUN_LATENCY_SECONDS = "wattwise_agent_run_latency_seconds"  # summary (per-run latency)
ENDPOINT_REQUESTS = "wattwise_http_requests_total"  # labelled by endpoint+outcome (OBS-R5)
ENDPOINT_LATENCY_SECONDS = "wattwise_http_request_latency_seconds"  # OBS-R5 per-endpoint
# --- ingestion operational metrics (doc 30 ING-OBS-R2; internal/admin surface only) ---
# Per-source success/failure rate, per-phase latency, record throughput, open/closed gap
# counts by reason, outbound request counts (cost), rate-limit waits, and freshness lag.
# The OSS engine serves the single implicit owner (ING-SEC-R2), so "per athlete" is the
# instance itself; labels carry the source_key, never athlete PII.
INGEST_SOURCE_RUNS = "wattwise_ingest_source_runs_total"  # labels: source_key, outcome
INGEST_PHASE_LATENCY = "wattwise_ingest_phase_latency_seconds"  # labels: source_key, phase
INGEST_RECORDS = "wattwise_ingest_records_total"  # labels: source_key, stage
INGEST_GAPS_OPENED = "wattwise_ingest_gaps_opened_total"  # labels: reason
INGEST_GAPS_CLOSED = "wattwise_ingest_gaps_closed_total"  # (transient self-heal, ING-GAP-R4)
INGEST_OUTBOUND_REQUESTS = "wattwise_ingest_outbound_requests_total"  # labels: source_key
INGEST_RATE_LIMIT_WAIT = "wattwise_ingest_rate_limit_wait_seconds"  # summary, per source
INGEST_FRESHNESS_LAG = "wattwise_ingest_freshness_lag_seconds"  # summary: now - watermark

_LabelKey = tuple[tuple[str, str], ...]


def _label_key(labels: dict[str, str] | None) -> _LabelKey:
    """A hashable, order-stable key for a label set (empty tuple when unlabelled)."""
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


@dataclass(slots=True)
class _Summary:
    """A streaming numeric summary: count, sum, and a bounded reservoir for quantiles.

    The reservoir holds the last ``_RESERVOIR`` samples so p50/p95 (AGT-OBS-R7) are
    computable without unbounded memory — adequate for rolling latency/cost percentiles
    on a single worker.
    """

    _RESERVOIR: int = 1024
    count: int = 0
    total: float = 0.0
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        """Record one sample into the count/sum and the bounded reservoir."""
        self.count += 1
        self.total += value
        self.samples.append(value)
        if len(self.samples) > self._RESERVOIR:
            del self.samples[0 : len(self.samples) - self._RESERVOIR]

    def quantile(self, q: float) -> float:
        """The ``q`` quantile of the reservoir (0.0 when no samples observed)."""
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return ordered[idx]


class MetricsRegistry:
    """In-process counters + summaries exposed in Prometheus text format (OBS-R5/-R4, AGT-OBS-R7).

    Thread-safe (a single lock guards the maps); counters/summaries are created lazily on
    first use so a never-incremented metric reads ``0`` rather than failing. Labelled series
    are keyed by ``(name, sorted-labels)``.
    """

    __slots__ = ("_counters", "_lock", "_summaries")

    def __init__(self) -> None:
        self._counters: dict[tuple[str, _LabelKey], float] = {}
        self._summaries: dict[tuple[str, _LabelKey], _Summary] = {}
        self._lock = threading.Lock()

    def increment(
        self, name: str, *, amount: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Add ``amount`` to the counter ``name`` for the given label set (created lazily)."""
        key = (name, _label_key(labels))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def observe(self, name: str, value: float, *, labels: dict[str, str] | None = None) -> None:
        """Record one sample into the summary ``name`` for the given label set."""
        key = (name, _label_key(labels))
        with self._lock:
            summary = self._summaries.get(key)
            if summary is None:
                summary = _Summary()
                self._summaries[key] = summary
            summary.observe(value)

    def counter_value(self, name: str, *, labels: dict[str, str] | None = None) -> float:
        """Read a counter's current value (``0.0`` if never incremented)."""
        with self._lock:
            return self._counters.get((name, _label_key(labels)), 0.0)

    def summary(self, name: str, *, labels: dict[str, str] | None = None) -> _Summary:
        """Read a summary (an empty one if never observed; not stored)."""
        with self._lock:
            return self._summaries.get((name, _label_key(labels)), _Summary())

    def render(self) -> str:
        """Render every series in Prometheus text exposition format (OBS-R5 scrape target)."""
        with self._lock:
            counters = dict(self._counters)
            summaries = {
                key: (s.count, s.total, s.quantile(0.5), s.quantile(0.95))
                for key, s in self._summaries.items()
            }
        lines: list[str] = []
        for (name, labels), value in sorted(counters.items()):
            lines.append(f"{name}{_render_labels(labels)} {_fmt(value)}")
        for (name, labels), (count, total, p50, p95) in sorted(summaries.items()):
            lines.append(f"{name}_count{_render_labels(labels)} {_fmt(float(count))}")
            lines.append(f"{name}_sum{_render_labels(labels)} {_fmt(total)}")
            lines.append(f"{name}{_render_labels(_with(labels, '0.5'))} {_fmt(p50)}")
            lines.append(f"{name}{_render_labels(_with(labels, '0.95'))} {_fmt(p95)}")
        return "\n".join(lines) + ("\n" if lines else "")


def _with(labels: _LabelKey, quantile: str) -> _LabelKey:
    """Return the label tuple with an extra ``quantile=`` pseudo-label (Prometheus summary)."""
    return tuple(sorted([*labels, ("quantile", quantile)]))


def _render_labels(labels: _LabelKey) -> str:
    """Render a sorted label tuple as a Prometheus ``{k="v",...}`` clause (empty when none)."""
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
    return "{" + inner + "}"


def _escape(value: str) -> str:
    """Escape a label value per the Prometheus text format (backslash, quote, newline)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: float) -> str:
    """Format a metric value: integers without a decimal point, else a plain float."""
    if value == int(value):
        return str(int(value))
    return repr(value)


# Single process-local registry the engine records to and the /metrics endpoint scrapes
# (the standard Prometheus client-library model: one registry per worker, scraped by the
# platform). Constructed eagerly because it holds no config value (CFG-R1a-safe).
REGISTRY = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Return the process-local metrics registry (OBS-R5 scrape source)."""
    return REGISTRY


__all__ = [
    "ENDPOINT_LATENCY_SECONDS",
    "ENDPOINT_REQUESTS",
    "ENTAILMENT_CHECKS",
    "ENTAILMENT_UNAVAILABLE",
    "ENTAILMENT_VETOES",
    "GROUNDING_BINDING_EVENTS",
    "GROUNDING_RUNS",
    "GROUNDING_SCRUBS",
    "INJECTION_ANOMALIES",
    "LANGUAGE_FALLBACKS",
    "REFLECTIONS",
    "REFLECTION_EXHAUSTIONS",
    "REFUSALS",
    "RUN_COST_USD",
    "RUN_LATENCY_SECONDS",
    "RUN_TERMINAL",
    "TIER_ESCALATIONS",
    "VALIDATION_FAILURES",
    "MetricsRegistry",
    "get_registry",
]
