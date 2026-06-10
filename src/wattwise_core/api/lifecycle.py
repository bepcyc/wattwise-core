"""App lifespan: startup retention sweeps + graceful shutdown (RUN-R11, PRIV-R7/R11.2).

RUN-R11: on ``SIGTERM`` the service stops accepting new work, marks itself NOT-READY
(``app.state.draining`` flips the OBS-R6.2 readiness probe to 503 so the orchestrator
drains it from rotation), lets the server drain in-flight requests within its bounded
grace period (uvicorn delivers the shutdown lifespan event after the accept loop stops),
and closes the connection pools cleanly. In-flight agent runs need no extra shutdown
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
        await _run_retention_sweeps(app)
        try:
            yield
        finally:
            # RUN-R11: mark not-ready FIRST (readiness 503 → drained from rotation),
            # then close the pools. Agent runs are already durably checkpointed per
            # step, so a resume continues from the last committed checkpoint.
            app.state.draining = True
            await _dispose_databases(app)

    return lifespan


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
