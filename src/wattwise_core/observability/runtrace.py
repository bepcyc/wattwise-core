"""Per-run trace + spans for the trustworthy agent (AGT-OBS-R1/-R2/-R5a, LOG-R3, CI-R8).

Every agent run is fully traced (AGT-OBS-R1): one :class:`RunTrace` carries a single
``trace_id`` tied to the run's ``thread_id``, and each node execution and each model/tool call
opens a :class:`Span` with start/end, status, and parent linkage recorded under that trace.
The active trace lives in a :class:`contextvars.ContextVar` so a node or the model seam can
reach it without threading it through every signature; the run-invoke point binds it.

Model/tool spans record the AGT-OBS-R2 fields (model, tier, reasoning effort, prompt/completion
tokens read from the real provider usage, computed cost, latency, structured-output schema id);
:meth:`RunTrace.rollup` aggregates the per-run totals the finalize node carries on the outcome
(total tokens/cost/latency, model-tier mix, reflection count, scrub count, terminal status).

Binding the trace ALSO realizes LOG-R3: it binds ``trace_id``/``run_id``/``thread_id`` into the
structlog context vars so every log line emitted during the run carries the correlation context
and a run is reconstructable from the log stream; each span binds its own ``span_id`` for its
duration. The trace is exported to a sink at run end so production runs reconcile with the
offline eval (LOG-R6.3); the OSS default sink records to the process metrics surface (OBS-R4).

A typed injection/anomaly event (AGT-OBS-R5a) is recorded on the trace AND counted in metrics
whenever untrusted input attempts to alter identity/scope/tooling/grounding and is neutralized,
so injection-neutralization is monitorable in production, not only in CI.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

from wattwise_core.observability import metrics as _metrics

# The active run trace for the current task (None outside a run). A ContextVar so a node or the
# model seam reaches the trace without a parameter thread-through, and concurrent runs in one
# event loop never cross (each task sees its own bound trace).
_ACTIVE: ContextVar[RunTrace | None] = ContextVar("wattwise_run_trace", default=None)


def _trace_id_for(thread_id: str | None) -> str:
    """Derive a stable ``trace_id`` tied to ``thread_id`` (AGT-OBS-R1).

    The trace id IS the run's durable thread id when one is present, so the trace correlates
    1:1 with the durable run (CKPT-R3); a run with no thread id (an isolated invoke) gets a
    fresh opaque id so it is still a single reconstructable trace.
    """
    return thread_id or uuid.uuid4().hex


@dataclass(slots=True)
class Span:
    """One traced unit of work: a node execution or a model/tool call (AGT-OBS-R1/-R2).

    Records start/end, status, and parent linkage (``parent_id``) under the run trace. A
    model/tool span ALSO records the AGT-OBS-R2 fields (model/tier/effort, prompt/completion
    tokens from the real provider usage, computed cost, latency, schema id).
    """

    name: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    status: str = "ok"
    model: str | None = None
    tier: str | None = None
    reasoning_effort: str | None = None
    schema_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def latency_ms(self) -> float | None:
        """Wall-clock latency in milliseconds once the span has ended."""
        if self.end is None:
            return None
        return (self.end - self.start) * 1000.0

    def record_usage(
        self,
        *,
        model: str | None = None,
        tier: str | None = None,
        reasoning_effort: str | None = None,
        schema_id: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Record the AGT-OBS-R2 model/tool span fields (provider usage + computed cost)."""
        self.model = model if model is not None else self.model
        self.tier = tier if tier is not None else self.tier
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
        self.schema_id = schema_id if schema_id is not None else self.schema_id
        self.prompt_tokens = prompt_tokens if prompt_tokens is not None else self.prompt_tokens
        self.completion_tokens = (
            completion_tokens if completion_tokens is not None else self.completion_tokens
        )
        self.cost_usd = cost_usd if cost_usd is not None else self.cost_usd


@dataclass(slots=True)
class AnomalyRecord:
    """A typed injection/anomaly event recorded on the run trace (AGT-OBS-R5a).

    Records the attempted override/instruction (as redacted untrusted DATA, never executed),
    the fact that it was IGNORED, and the authenticated scope / registry-bound capability /
    allow-listed value actually used. Correlated to the run trace and counted in metrics.
    """

    kind: str
    capability: str
    attempted: Mapping[str, Any]
    ignored: bool
    authenticated_scope: str


@dataclass(slots=True)
class RunTrace:
    """The single trace for one agent run (AGT-OBS-R1): one ``trace_id`` tied to ``thread_id``.

    Owns the ordered span list, the anomaly events, and the per-run rollup aggregation
    (AGT-OBS-R2). It is bound as the active trace for the duration of the run.
    """

    trace_id: str
    thread_id: str | None
    run_id: str
    started: float = field(default_factory=time.monotonic)
    spans: list[Span] = field(default_factory=list)
    anomalies: list[AnomalyRecord] = field(default_factory=list)
    _stack: list[str] = field(default_factory=list)
    _scrubs: int = 0

    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since the run trace was opened (run latency, AGT-OBS-R7)."""
        return time.monotonic() - self.started

    def open_span(self, name: str) -> Span:
        """Open a child span under the current parent (parent linkage, AGT-OBS-R1)."""
        parent_id = self._stack[-1] if self._stack else None
        span = Span(
            name=name, span_id=uuid.uuid4().hex, parent_id=parent_id, start=time.monotonic()
        )
        self.spans.append(span)
        return span

    def record_anomaly(self, anomaly: AnomalyRecord) -> None:
        """Record an injection/anomaly event on the trace + count it in metrics (AGT-OBS-R5a)."""
        self.anomalies.append(anomaly)
        _metrics.get_registry().increment(
            _metrics.INJECTION_ANOMALIES, labels={"kind": anomaly.kind}
        )

    def rollup(self, status: str) -> dict[str, Any]:
        """Aggregate the per-run rollup carried on the outcome + trace (AGT-OBS-R2).

        Totals tokens/cost/latency across model/tool spans, the model-tier mix, the reflection
        and scrub counts, and the terminal status. Read from the spans' recorded provider usage,
        never fabricated; fields with no recorded usage contribute zero.
        """
        prompt = sum(s.prompt_tokens or 0 for s in self.spans)
        completion = sum(s.completion_tokens or 0 for s in self.spans)
        cost = sum(s.cost_usd or 0.0 for s in self.spans)
        latency_ms = sum(s.latency_ms or 0.0 for s in self.spans)
        tier_mix: dict[str, int] = {}
        for span in self.spans:
            if span.tier is not None:
                tier_mix[span.tier] = tier_mix.get(span.tier, 0) + 1
        return {
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "total_tokens": prompt + completion,
            "total_cost_usd": cost,
            "total_latency_ms": latency_ms,
            "model_tier_mix": tier_mix,
            "reflection_count": self.reflection_count(),
            "scrub_count": self.scrub_count(),
            "status": status,
        }

    def reflection_count(self) -> int:
        """The number of reflect spans this run executed (self-correction count, AGT-OBS-R2)."""
        return sum(1 for s in self.spans if s.name == "reflect")

    def scrub_count(self) -> int:
        """The total grounding scrubs recorded on this run (AGT-OBS-R4)."""
        return self._scrubs

    def add_scrubs(self, count: int) -> None:
        """Accumulate the per-scrub count the ground node observed (AGT-OBS-R4)."""
        self._scrubs += count


def active_trace() -> RunTrace | None:
    """The trace bound to the current run, or ``None`` outside a run."""
    return _ACTIVE.get()


@contextlib.contextmanager
def run_trace(thread_id: str | None, *, run_id: str | None = None) -> Iterator[RunTrace]:
    """Bind a fresh run trace tied to ``thread_id`` for the duration of the run (AGT-OBS-R1).

    Binds ``trace_id``/``run_id``/``thread_id`` into the structlog context (LOG-R3) so every log
    line during the run carries the correlation context and the run is reconstructable from the
    log stream. On exit the bindings are reset and the trace is exported to the metrics sink
    (OBS-R4) so production runs are observable, not only in CI.
    """
    rid = run_id or uuid.uuid4().hex
    trace = RunTrace(trace_id=_trace_id_for(thread_id), thread_id=thread_id, run_id=rid)
    token = _ACTIVE.set(trace)
    bound = {"trace_id": trace.trace_id, "run_id": trace.run_id}
    if thread_id:
        bound["thread_id"] = thread_id
    structlog.contextvars.bind_contextvars(**bound)
    try:
        yield trace
    finally:
        structlog.contextvars.unbind_contextvars("trace_id", "run_id", "thread_id")
        _ACTIVE.reset(token)


@contextlib.contextmanager
def span(name: str) -> Iterator[Span | None]:
    """Open a span on the active trace (no-op outside a run), binding its ``span_id`` (LOG-R3).

    Yields the open :class:`Span` so a model/tool call can record its AGT-OBS-R2 usage on it.
    The span's ``span_id`` is bound into the structlog context for its duration so every log line
    inside the span carries it; status flips to ``error`` if the body raises (AGT-OBS-R1).
    """
    trace = _ACTIVE.get()
    if trace is None:
        yield None
        return
    current = trace.open_span(name)
    trace._stack.append(current.span_id)
    structlog.contextvars.bind_contextvars(span_id=current.span_id)
    try:
        yield current
    except BaseException:
        current.status = "error"
        raise
    finally:
        current.end = time.monotonic()
        trace._stack.pop()
        structlog.contextvars.unbind_contextvars("span_id")


def record_anomaly(anomaly: AnomalyRecord) -> None:
    """Record an anomaly on the active trace (no-op outside a run); count it (AGT-OBS-R5a)."""
    trace = _ACTIVE.get()
    if trace is not None:
        trace.record_anomaly(anomaly)


class ScopeAnomalyEvent(Protocol):
    """The structural shape of a neutralized scope-override event (AGT-OBS-R5a).

    Matched structurally (no agent-module import here) so the agent's gather seam can hand its
    typed anomaly events straight to the trace without coupling the trace to agent internals.
    """

    @property
    def kind(self) -> str: ...

    @property
    def capability(self) -> str: ...

    @property
    def ignored_override(self) -> Mapping[str, Any]: ...

    @property
    def authenticated_scope(self) -> str: ...


def record_scope_anomalies(events: Iterable[ScopeAnomalyEvent]) -> None:
    """Record each neutralized scope override on the trace + metrics (AGT-OBS-R5a).

    Each event is an attempted cross-athlete override that was IGNORED (PLAN-R5): the attempted
    override is recorded as redacted untrusted DATA together with the authenticated scope that
    was actually used; outside a run this is a no-op.
    """
    for event in events:
        record_anomaly(
            AnomalyRecord(
                kind=event.kind,
                capability=event.capability,
                attempted=event.ignored_override,
                ignored=True,
                authenticated_scope=event.authenticated_scope,
            )
        )


def record_scrubs(count: int) -> None:
    """Accumulate a per-scrub count on the active trace + count it in metrics (AGT-OBS-R4)."""
    if count <= 0:
        return
    trace = _ACTIVE.get()
    if trace is not None:
        trace.add_scrubs(count)
    _metrics.get_registry().increment(_metrics.GROUNDING_SCRUBS, amount=count)


__all__ = [
    "AnomalyRecord",
    "RunTrace",
    "ScopeAnomalyEvent",
    "Span",
    "active_trace",
    "record_anomaly",
    "record_scope_anomalies",
    "record_scrubs",
    "run_trace",
    "span",
]
