"""A RESUMED agent run is fully traced, like the initial run (AGT-OBS-R1, LOG-R3).

``CompiledCoachGraph.run`` binds a run trace tied to the durable ``thread_id`` so every node/model
span + log line of the initial run correlates under one ``trace_id``. ``resume`` (the HITL decision
continuation) MUST do the SAME (AGT-OBS-R1: "every run MUST be fully traced") — otherwise the
post-resume nodes emit spans into a ``None`` trace (silent no-op) and the log lines carry no
correlation context. The GAP_SPEC finding: ``resume`` invoked the compiled graph with no
``run_trace`` binding.

Mutation-proofing: a revert that drops the ``run_trace(thread_id)`` wrapper from ``resume`` leaves
the active trace ``None`` and the LOG-R3 context vars unbound during the resumed invoke, failing
``test_resume_binds_run_trace_tied_to_thread_id``. A revert that hardcodes ``run_trace(None)``
(ignoring the config thread id) breaks the ``trace_id == thread_id`` assertion.
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from wattwise_core.agent.engine_graph import CompiledCoachGraph
from wattwise_core.observability import runtrace

pytestmark = pytest.mark.unit

_THREAD = "athlete-resume:conv-9"


class _RecordingCompiled:
    """A fake compiled graph that captures the trace context active during ``ainvoke``.

    Stands in for the real :class:`CompiledStateGraph` so the test exercises ``resume``'s trace
    binding in isolation: during the invoke it reads the active run trace + the LOG-R3 structlog
    context vars and opens one span (a post-resume node/model call) so the test can assert the
    resumed work is correlated under the run trace.
    """

    def __init__(self) -> None:
        self.active_trace: runtrace.RunTrace | None = None
        self.bound_context: dict[str, Any] = {}
        self.opened_span_into_trace = False

    async def ainvoke(self, command: Any, config: RunnableConfig) -> dict[str, Any]:
        """Capture the active trace + context, open a span, and return a terminal state."""
        self.active_trace = runtrace.active_trace()
        self.bound_context = dict(structlog.contextvars.get_contextvars())
        with runtrace.span("post_resume_node") as span:
            self.opened_span_into_trace = span is not None
        return {"status": "completed"}


async def test_resume_binds_run_trace_tied_to_thread_id() -> None:
    """``resume`` binds a run trace tied to the config thread id (AGT-OBS-R1 / LOG-R3)."""
    compiled = _RecordingCompiled()
    graph = CompiledCoachGraph(compiled, wall_clock_seconds=None)  # type: ignore[arg-type]
    config: RunnableConfig = {"configurable": {"thread_id": _THREAD}}

    await graph.resume(Command(resume="approve"), config)

    # A trace was active during the resumed invoke (not None -> spans no longer no-op, AGT-OBS-R1).
    assert compiled.active_trace is not None
    # The trace id is tied to the durable thread id (CKPT-R3), so the resume's trace correlates
    # with the paused run's trace rather than minting an unrelated opaque id.
    assert compiled.active_trace.trace_id == _THREAD
    assert compiled.active_trace.thread_id == _THREAD
    # A post-resume span opened ONTO the trace (proving the binding actually took effect).
    assert compiled.opened_span_into_trace is True
    assert any(s.name == "post_resume_node" for s in compiled.active_trace.spans)
    # LOG-R3: the correlation context was bound during the invoke so log lines are reconstructable.
    assert compiled.bound_context["trace_id"] == _THREAD
    assert compiled.bound_context["thread_id"] == _THREAD
    assert "run_id" in compiled.bound_context
    # On exit the binding is reset (no leak into the surrounding task).
    assert "trace_id" not in structlog.contextvars.get_contextvars()
    assert runtrace.active_trace() is None


async def test_resume_with_no_thread_id_still_binds_a_trace() -> None:
    """A resume with no thread id still binds a fresh trace, never a None no-op (AGT-OBS-R1)."""
    compiled = _RecordingCompiled()
    graph = CompiledCoachGraph(compiled, wall_clock_seconds=None)  # type: ignore[arg-type]

    await graph.resume(Command(resume="approve"), {"configurable": {}})

    # No thread id -> a fresh opaque trace id (still a single reconstructable trace), NOT None.
    assert compiled.active_trace is not None
    assert compiled.active_trace.trace_id  # a real opaque id was minted
    assert compiled.active_trace.thread_id is None
    assert compiled.opened_span_into_trace is True
