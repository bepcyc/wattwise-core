"""Run-trace, span, anomaly, and metrics surface unit tests (AGT-OBS-R1/-R2/-R5a, OBS-R5/-R4).

These encode the §15 observability acceptance verbatim against the new in-process trace +
metrics substrate:

- AGT-OBS-R1: a run binds ONE ``trace_id`` tied to ``thread_id``; each opened span records
  start/end, status, and parent linkage under that trace.
- AGT-OBS-R2: a model/tool span records model/tier/effort + prompt/completion tokens + computed
  cost + latency + schema id read from provider usage; the per-run rollup totals them plus the
  model-tier mix, reflection count, scrub count, and terminal status.
- AGT-OBS-R5a: an injection/anomaly event recorded on the trace is ALSO counted in metrics.
- LOG-R3: while a run/span is active the structlog context carries ``trace_id``/``run_id``/
  ``thread_id``/``span_id`` so a log line is reconstructable to the run.
- OBS-R5/-R4/AGT-OBS-R7: the metrics registry renders in Prometheus text format with p50/p95.

Mutation-proofing: each assertion fails if the rule is reverted (e.g. drop the thread_id->trace
tie, discard span usage, or stop counting the anomaly).
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest
import structlog

from wattwise_core.observability import metrics as m
from wattwise_core.observability.logging import configure_logging, get_logger
from wattwise_core.observability.runtrace import (
    AnomalyRecord,
    active_trace,
    record_anomaly,
    record_scrubs,
    run_trace,
    span,
)

pytestmark = pytest.mark.unit


def test_run_binds_single_trace_id_tied_to_thread_id() -> None:
    """A run binds exactly one trace whose trace_id is the run's thread_id (AGT-OBS-R1)."""
    with run_trace("athlete-x:conv-7") as trace:
        assert trace.trace_id == "athlete-x:conv-7"
        assert trace.thread_id == "athlete-x:conv-7"
        assert active_trace() is trace
    # The trace is unbound once the run ends.
    assert active_trace() is None


def test_each_span_records_start_end_status_and_parent_linkage() -> None:
    """Each node/model span records start/end, status, and parent linkage (AGT-OBS-R1)."""
    with run_trace("t:1") as trace, span("plan_retrieval") as parent:
        assert parent is not None
        with span("model_call") as child:
            assert child is not None
            assert child.parent_id == parent.span_id
    names = [s.name for s in trace.spans]
    assert names == ["plan_retrieval", "model_call"]
    for s in trace.spans:
        assert s.start is not None and s.end is not None
        assert s.status == "ok"
        assert s.latency_ms is not None and s.latency_ms >= 0.0
    # The root span has no parent; the child links to the parent (parent linkage).
    assert trace.spans[0].parent_id is None
    assert trace.spans[1].parent_id == trace.spans[0].span_id


def test_span_status_flips_to_error_when_body_raises() -> None:
    """A span whose body raises is recorded with status 'error' (AGT-OBS-R1)."""
    with run_trace("t:err") as trace:
        try:
            with span("compose"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    assert trace.spans[0].status == "error"


def test_model_span_records_usage_and_run_rollup_totals_them() -> None:
    """A model span records tokens/cost/latency/tier/schema; the rollup totals them (AGT-OBS-R2)."""
    with run_trace("t:roll") as trace:
        with span("reflect"):
            pass  # a reflect span counts toward the self-correction count
        with span("model_call") as s:
            assert s is not None
            s.record_usage(
                model="deepseek/deepseek-v4-flash",
                tier="flash",
                reasoning_effort="low",
                schema_id="ReflectDecision",
                prompt_tokens=120,
                completion_tokens=40,
                cost_usd=0.0001,
            )
        record_scrubs(2)
        rollup = trace.rollup(status="completed")
    assert rollup["total_prompt_tokens"] == 120
    assert rollup["total_completion_tokens"] == 40
    assert rollup["total_tokens"] == 160
    assert rollup["total_cost_usd"] == 0.0001
    assert rollup["model_tier_mix"] == {"flash": 1}
    assert rollup["reflection_count"] == 1
    assert rollup["scrub_count"] == 2
    assert rollup["status"] == "completed"


def test_anomaly_recorded_on_trace_and_counted_in_metrics() -> None:
    """An injection/anomaly event lands on the trace AND increments the metric (AGT-OBS-R5a)."""
    registry = m.get_registry()
    before = registry.counter_value(
        m.INJECTION_ANOMALIES, labels={"kind": "scope_override_ignored"}
    )
    with run_trace("t:anom") as trace:
        record_anomaly(
            AnomalyRecord(
                kind="scope_override_ignored",
                capability="weekly_load",
                attempted={"athlete_id": "victim"},
                ignored=True,
                authenticated_scope="owner",
            )
        )
    assert len(trace.anomalies) == 1
    assert trace.anomalies[0].authenticated_scope == "owner"
    after = registry.counter_value(m.INJECTION_ANOMALIES, labels={"kind": "scope_override_ignored"})
    assert after == before + 1.0


def test_log_line_carries_trace_and_span_correlation_context() -> None:
    """Log lines emitted in a run/span carry trace_id/run_id/thread_id/span_id (LOG-R3)."""
    with run_trace("athlete:conv") as trace:
        ctx_run = structlog.contextvars.get_contextvars()
        assert ctx_run["trace_id"] == trace.trace_id
        assert ctx_run["thread_id"] == "athlete:conv"
        assert "run_id" in ctx_run
        with span("gather") as s:
            assert s is not None
            ctx_span = structlog.contextvars.get_contextvars()
            assert ctx_span["span_id"] == s.span_id
        # span_id is unbound after the span ends; trace stays for the run.
        assert "span_id" not in structlog.contextvars.get_contextvars()
    assert "trace_id" not in structlog.contextvars.get_contextvars()


def test_log_line_in_run_is_reconstructable_from_the_stream() -> None:
    """A log line emitted during a run/span carries the LOG-R3 correlation ids in the STREAM.

    The merge_contextvars step is no longer empty (the GAP_SPEC LOG-R3 finding): an emitted JSON
    line carries trace_id/run_id/thread_id (and span_id inside a span) so the run is
    reconstructable from the log stream. A revert (no bind in run_trace/span) leaves them absent.
    """
    stream = io.StringIO()
    configure_logging()
    logger = get_logger("test.obs")
    with redirect_stdout(stream), run_trace("athlete:conv-log") as trace, span("gather") as s:
        assert s is not None
        logger.info("gather_done")
        bound_span = s.span_id
    line = next(json.loads(ln) for ln in stream.getvalue().splitlines() if "gather_done" in ln)
    assert line["trace_id"] == trace.trace_id
    assert line["thread_id"] == "athlete:conv-log"
    assert line["run_id"] == trace.run_id
    assert line["span_id"] == bound_span


def test_metrics_registry_renders_prometheus_text_with_quantiles() -> None:
    """The registry renders counters + p50/p95 summaries in Prometheus text (OBS-R5/AGT-OBS-R7)."""
    registry = m.MetricsRegistry()
    registry.increment(m.GROUNDING_RUNS)
    registry.increment(m.RUN_TERMINAL, labels={"status": "completed"})
    for latency in (0.1, 0.2, 0.3, 0.4):
        registry.observe(m.RUN_LATENCY_SECONDS, latency)
    text = registry.render()
    assert f"{m.GROUNDING_RUNS} 1" in text
    assert f'{m.RUN_TERMINAL}{{status="completed"}} 1' in text
    assert f"{m.RUN_LATENCY_SECONDS}_count 4" in text
    assert f'{m.RUN_LATENCY_SECONDS}{{quantile="0.5"}}' in text
    assert f'{m.RUN_LATENCY_SECONDS}{{quantile="0.95"}}' in text
    summary = registry.summary(m.RUN_LATENCY_SECONDS)
    assert summary.quantile(0.5) in (0.2, 0.3)
    assert summary.quantile(0.95) == 0.4
