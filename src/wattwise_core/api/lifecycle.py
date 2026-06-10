"""App lifespan: startup retention sweeps + graceful shutdown (RUN-R11, PRIV-R7/R11.2).

RUN-R11: on ``SIGTERM`` a loop signal handler registered at STARTUP immediately marks
the instance NOT-READY (``app.state.draining`` flips the OBS-R6.2 readiness probe to
503 so the orchestrator drains it from rotation WHILE the accept loop is still open —
uvicorn delivers the shutdown lifespan event only after the accept loop stops, which
is too late for a rolling deploy), chains to uvicorn's own handler so its graceful
shutdown drains in-flight requests within the bounded grace period, and the lifespan
finally re-asserts the drain flag (backstop) and closes the connection pools cleanly.
In-flight agent runs need no extra shutdown
work to stay resumable: every step is durably checkpointed at the node boundary by the
agent-state saver (CKPT-R*), so a resumed instance continues from the last committed
checkpoint — nothing here can lose or double a terminal settlement.

Startup additionally runs the retention sweeps once per boot (the OSS engine has no
scheduler; a periodic external trigger is the platform's cron):

* original-file purge (PRIV-R7 / PRIV-R11.2): retained verbatim originals older than
  ``retention__raw_file_days`` are deleted — object bytes and reference row.
* agent-state expiry (PRIV-R7 / CKPT-R8): durable checkpoints/threads older than
  ``retention__agent_state_days`` are expired.

Sweeps are best-effort at boot (a sweep failure logs and never blocks serving — the
window is enforced again next boot), but every purge is recorded on the audit stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from wattwise_core.agent.retention import sweep_expired_checkpoints
from wattwise_core.agent.state_db import build_agent_state_database
from wattwise_core.config import Settings
from wattwise_core.observability.audit import audit_event
from wattwise_core.observability.logging import get_logger
from wattwise_core.persistence import Database
from wattwise_core.privacy.retention import purge_expired_original_files
from wattwise_core.seams import SYSTEM_SUBJECT, EngineSessionProvider
from wattwise_core.storage import create_object_store

_logger = get_logger("wattwise_core.api.lifecycle")


def build_lifespan() -> Callable[[FastAPI], Any]:
    """Build the ASGI lifespan context for :func:`create_app` (RUN-R11)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Startup: sweeps + ready; shutdown: drain-marked, pools closed (RUN-R11)."""
        app.state.draining = False
        restore_handler = _install_sigterm_drain(app)
        await _run_retention_sweeps(app)
        try:
            yield
        finally:
            # RUN-R11 BACKSTOP: the primary drain flip is the SIGTERM handler above,
            # which fires while the server is STILL accepting (so the readiness probe
            # turns 503 and the orchestrator drains the instance from rotation before
            # uvicorn closes the accept loop). This finally re-asserts the flag for
            # shutdown paths that never saw a SIGTERM, then closes the pools. Agent
            # runs are already durably checkpointed per step, so a resume continues
            # from the last committed checkpoint.
            app.state.draining = True
            restore_handler()
            await _dispose_databases(app)

    return lifespan


def _install_sigterm_drain(app: FastAPI) -> Callable[[], None]:
    """Flip ``app.state.draining`` the instant ``SIGTERM`` arrives (RUN-R11).

    The lifespan-shutdown flip alone is TOO LATE for rolling deploys: uvicorn delivers
    the shutdown lifespan event only AFTER the accept loop has closed, so the readiness
    probe would keep answering 200 through the whole drain window. This registers a
    loop signal handler at STARTUP that marks the instance draining (readiness 503,
    OBS-R6.2) while it still serves, then chains to the previously installed handler
    (uvicorn's own, so its graceful shutdown still runs). The drain callback is also
    exposed as ``app.state.begin_drain`` — the SIGTERM-equivalent seam for tests and
    embedders. On platforms/threads without loop signal handlers (Windows Proactor,
    non-main threads e.g. under TestClient) registration is skipped gracefully and the
    lifespan-finally backstop still applies. Returns a restorer for lifespan shutdown.
    """
    previous = None
    with contextlib.suppress(ValueError):  # signal API is main-thread-only
        previous = signal.getsignal(signal.SIGTERM)

    def _begin_drain() -> None:
        app.state.draining = True  # readiness flips 503; in-flight requests still serve
        if callable(previous):  # chain to uvicorn's handler → graceful shutdown proceeds
            previous(signal.SIGTERM, None)

    app.state.begin_drain = _begin_drain
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, _begin_drain)
    except (NotImplementedError, RuntimeError, ValueError):
        return lambda: None  # unsupported platform/thread: finally backstop only

    def _restore() -> None:
        with contextlib.suppress(Exception):
            loop.remove_signal_handler(signal.SIGTERM)
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)

    return _restore


async def _dispose_databases(app: FastAPI) -> None:
    """Close every process-owned connection pool cleanly (RUN-R11)."""
    database = getattr(app.state, "database", None)
    master = getattr(app.state, "master_data_database", None)
    if isinstance(master, Database) and master is not database:
        await master.dispose()
    if isinstance(database, Database):
        await database.dispose()


async def _run_retention_sweeps(app: FastAPI) -> None:
    """Run the per-boot retention sweeps (PRIV-R7 / PRIV-R11.2 / CKPT-R8); never raise."""
    settings = getattr(app.state, "settings", None)
    database = getattr(app.state, "database", None)
    if not isinstance(settings, Settings) or not isinstance(database, Database):
        return
    await _sweep_original_files(settings, database)
    await _sweep_agent_state(settings)


async def _sweep_original_files(settings: Settings, database: Database) -> None:
    """Purge retained originals past ``retention__raw_file_days`` (PRIV-R11.2)."""
    if settings.retention__raw_file_days <= 0:
        return
    try:
        async with EngineSessionProvider(database).session(subject=SYSTEM_SUBJECT) as session:
            purged = await purge_expired_original_files(
                session,
                create_object_store(settings),
                retention_days=settings.retention__raw_file_days,
            )
        if purged:
            audit_event(
                "original_files_purged",
                count=purged,
                retention_days=settings.retention__raw_file_days,
            )
    except Exception:
        _logger.warning("retention_sweep_failed", category="original_files")


async def _sweep_agent_state(settings: Settings) -> None:
    """Expire durable agent state past ``retention__agent_state_days`` (CKPT-R8)."""
    if settings.retention__agent_state_days <= 0:
        return
    state_db = build_agent_state_database(settings)
    try:
        async with state_db.session() as session:
            result = await sweep_expired_checkpoints(
                session, retention_days=settings.retention__agent_state_days
            )
        audit_event(
            "agent_state_expired",
            swept=str(result),
            retention_days=settings.retention__agent_state_days,
        )
    except Exception:
        _logger.warning("retention_sweep_failed", category="agent_state")
    finally:
        await state_db.dispose()


__all__ = ["build_lifespan"]
