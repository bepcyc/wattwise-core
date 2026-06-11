"""Logging contract: NO application-written log files; streams only (QA-LOG-R1).

Exercises the two representative log-emitting paths the gate names — an API REQUEST
through the assembled app (middleware request logging) and an AGENT RUN (the run-trace
span/rollup logging) — from a clean working directory, and asserts:

* NO application-managed log file appears on disk (the app emits ONLY a structured
  event stream to stdout/stderr — it never writes, rotates, or retains its own files);
* the logging tree carries NO file-writing handler (no ``FileHandler`` /
  ``RotatingFileHandler``) after both paths ran;
* the emitted lines are structured JSON, one event per line, carrying the correlation
  context for the agent run (``run_id``).

The central-redaction half of QA-LOG-R1 is gated by ``test_logging_redaction`` /
``test_planted_secret`` in this same ``logging`` tier.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wattwise_core.api.app import create_app
from wattwise_core.config import Settings, load_settings
from wattwise_core.observability.logging import configure_logging, get_logger
from wattwise_core.observability.runtrace import run_trace, span

pytestmark = pytest.mark.logging


def _settings() -> Settings:
    """Dev settings with an in-memory DSN (no real DB; the probe route is /healthz)."""
    return load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="unit-test-signing-key-not-a-real-secret",
    )


def _snapshot(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()}


def _no_file_handlers() -> list[logging.Handler]:
    """Every file-writing handler installed anywhere in the logging tree."""
    handlers: list[logging.Handler] = list(logging.getLogger().handlers)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            handlers.extend(logger.handlers)
    return [h for h in handlers if isinstance(h, logging.FileHandler)]


def test_request_and_agent_run_write_no_log_file_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real API request AND an agent-run trace emit to the standard streams only:
    no application-managed log file is created on disk (QA-LOG-R1 bullet 1)."""
    monkeypatch.chdir(tmp_path)
    before = _snapshot(tmp_path)
    stream = io.StringIO()
    with redirect_stdout(stream):
        configure_logging()
        client = TestClient(create_app(_settings()), raise_server_exceptions=False)
        assert client.get("/healthz").status_code == 200  # the request log path
        with run_trace("athlete:log-contract") as trace, span("gather"):
            get_logger(__name__).info("gather_done", thread_id=trace.thread_id)
    # No app-managed log file was created by either path (sqlite is :memory:).
    created = _snapshot(tmp_path) - before
    assert not created, f"application wrote files during logging: {sorted(created)}"
    assert _no_file_handlers() == [], "a file-writing log handler is installed"


def test_emitted_agent_run_lines_are_structured_json_with_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent-run emission is structured JSON (one event per line) on stdout,
    carrying the run correlation context (QA-LOG-R1 bullet 2)."""
    monkeypatch.chdir(tmp_path)
    stream = io.StringIO()
    with redirect_stdout(stream):
        configure_logging()
        with run_trace("athlete:log-contract-json"):
            get_logger(__name__).info("agent_step_done")
    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    assert lines, "the agent-run path emitted nothing to stdout"
    events = [json.loads(ln) for ln in lines]  # every line is one JSON object
    done = next(e for e in events if e.get("event") == "agent_step_done")
    assert done.get("run_id"), "agent-run line lacks the run_id correlation context"
    assert done.get("timestamp") and done.get("level")
    # And still: nothing on disk.
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]
    assert Path.cwd() == tmp_path
