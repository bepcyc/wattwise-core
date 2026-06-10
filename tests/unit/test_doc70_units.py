"""Unit tests for the doc-70 cross-cutting pieces: the tamper-evident audit chain
(LOG-R6.2/LOG-R8), the per-request log-correlation binding (LOG-R3), the bounded
concurrent sync runner (PERF-R2), and the migration-head probe (RUN-R6).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
from typing import Any

import pytest
import structlog

from wattwise_core.api.middleware import RequestContextMiddleware
from wattwise_core.ingestion.sync import SyncOrchestrator
from wattwise_core.observability.audit import (
    audit_event,
    record_erasure_hook,
    reset_chain_for_tests,
)
from wattwise_core.persistence.migrations_state import expected_head

pytestmark = pytest.mark.unit


# --------------------------------------------------------- LOG-R6.2 tamper-evident audit chain


def test_audit_events_form_a_verifiable_hash_chain() -> None:
    """Consecutive audit events chain: each ``prev_hash`` is the prior ``entry_hash`` (LOG-R6.2).

    The chain is recomputable from the records alone: ``entry_hash`` = SHA-256 over the
    canonicalized payload + ``prev_hash``, so editing or dropping any earlier event
    breaks verification of every later one (tamper evidence).
    """
    reset_chain_for_tests()
    first = audit_event("auth_token_issued", athlete_id="a-1")
    second = audit_event("data_export_started", athlete_id="a-1")
    assert first["audit_seq"] == 1
    assert second["audit_seq"] == 2
    assert second["prev_hash"] == first["entry_hash"]
    # Recompute the second hash independently from the record's own payload fields.
    payload = {"event": "data_export_started", "athlete_id": "a-1"}
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    expected = hashlib.sha256((second["prev_hash"] + canonical).encode()).hexdigest()
    assert second["entry_hash"] == expected
    # A tampered payload no longer verifies against the recorded hash.
    tampered = json.dumps(
        {"event": "data_export_started", "athlete_id": "a-2"},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    assert (
        hashlib.sha256((second["prev_hash"] + tampered).encode()).hexdigest()
        != (second["entry_hash"])
    )


def test_erasure_hook_event_names_the_athlete(  # LOG-R8
) -> None:
    """The LOG-R8 per-athlete erasure hook emits an audit event naming the opaque id."""
    reset_chain_for_tests()
    record = record_erasure_hook("athlete-42")
    assert record["event"] == "log_pii_erasure_hook"
    assert record["athlete_id"] == "athlete-42"
    assert record["stream"] == "audit"


# -------------------------------------------------------------- LOG-R3 request-context binding


async def test_request_context_middleware_binds_and_clears_request_id() -> None:
    """Every request binds a ``request_id`` into the log context and clears it after (LOG-R3).

    An inbound ``X-Request-ID`` is honored (gateway correlation); after the request the
    contextvars are cleared so nothing leaks into the next request on the same task.
    """
    seen: dict[str, Any] = {}

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        seen.update(structlog.contextvars.get_contextvars())

    middleware = RequestContextMiddleware(inner)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/x",
        "headers": [(b"x-request-id", b"gw-123")],
        "query_string": b"",
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        return None

    await middleware(scope, receive, send)
    assert seen["request_id"] == "gw-123"
    assert "request_id" not in structlog.contextvars.get_contextvars()  # cleared after


# ------------------------------------------------------------------ PERF-R2 bounded concurrency


class _StubTarget:
    """A minimal stand-in for the connection target the runner fans out over."""


async def test_sync_runner_executes_sources_concurrently_bounded() -> None:
    """Independent source syncs overlap up to the configured bound (PERF-R2).

    Three sources with ``concurrency=2``: the observed in-flight peak is 2 (bounded,
    concurrent — not serial), and the per-connection results come back in connection
    order regardless of completion order.
    """
    runner = SyncOrchestrator.__new__(SyncOrchestrator)
    runner._concurrency = 2
    state = {"active": 0, "peak": 0}

    async def fake_sync_one(
        athlete_id: str, conn: Any, win: Any, fetched_at: Any, run_id: str, explicit: bool
    ) -> str:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.02)
        state["active"] -= 1
        return f"result-{id(conn)}"

    targets = [_StubTarget(), _StubTarget(), _StubTarget()]

    async def fake_select(athlete_id: str, connection_id: Any, source: Any) -> list[Any]:
        return list(targets)

    runner._sync_one = fake_sync_one  # type: ignore[method-assign, assignment]
    runner._select_connections = fake_select  # type: ignore[method-assign, assignment]
    runner._now = lambda: _dt.datetime.now(_dt.UTC)
    run = await runner.run("athlete-1")
    assert state["peak"] == 2  # concurrent AND bounded (never 3, never serial 1)
    assert run.results == [f"result-{id(t)}" for t in targets]  # connection order kept


async def test_sync_runner_serializes_when_concurrency_is_one() -> None:
    """``concurrency=1`` (the SQLite clamp) degrades gracefully to serial syncs (PERF-R2)."""
    runner = SyncOrchestrator.__new__(SyncOrchestrator)
    runner._concurrency = 1
    state = {"active": 0, "peak": 0}

    async def fake_sync_one(
        athlete_id: str, conn: Any, win: Any, fetched_at: Any, run_id: str, explicit: bool
    ) -> str:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.005)
        state["active"] -= 1
        return "r"

    async def fake_select(athlete_id: str, connection_id: Any, source: Any) -> list[Any]:
        return [_StubTarget(), _StubTarget()]

    runner._sync_one = fake_sync_one  # type: ignore[method-assign, assignment]
    runner._select_connections = fake_select  # type: ignore[method-assign, assignment]
    runner._now = lambda: _dt.datetime.now(_dt.UTC)
    run = await runner.run("athlete-1")
    assert state["peak"] == 1  # strictly serialized (single-writer degradation)
    assert len(run.results) == 2


# ----------------------------------------------------------------- RUN-R6 migration-head probe


def test_expected_head_resolves_the_newest_migration_revision() -> None:
    """The probe parses the versioned scripts and resolves the single head (RUN-R6).

    In the source checkout the migrations directory is locatable, so the head is a
    concrete revision id — the one no other script names as its ``down_revision``.
    """
    head = expected_head()
    assert head is not None
    assert head >= "0014"  # the auth/ops agent-state migration is at or behind the head
