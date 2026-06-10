"""HITL approval-gate interrupt ledger operations (CKPT-R9 / D-P2, API-R12a).

The focused sibling of :mod:`wattwise_core.agent.checkpoint` (QUAL-R9 size split) that owns the
three durable interrupt-ledger operations the approval gate + decision endpoint drive on the
``agent_interrupt`` table: record a ``live`` row when the graph pauses, atomically consume it on a
winning decision, and read its status to split a refused decision into 404 vs 409. The saver
delegates to these so the operations live in one cohesive, identity-scoped place; each takes the
saver's bound ``athlete_id`` so every guard is independently scoped (CKPT-R3) — a cross-athlete
attempt never matches another owner's row.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from sqlalchemy import Table, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from wattwise_core.agent.state_store import AgentInterrupt
from wattwise_core.persistence.types import uuid7
from wattwise_core.persistence.upsert import upsert

# The saver's get-or-create-thread step, injected so ``record_interrupt`` keeps the thread FK
# satisfied without this module importing the saver (no cycle).
EnsureThread = Callable[[AsyncSession, str], Awaitable[Any]]


class DecisionRefused(RuntimeError):
    """A HITL decision could not consume a live interrupt (CKPT-R9; fail-closed).

    Raised by :meth:`~wattwise_core.agent.engine.GraphAgentEngine.decision` when
    ``consume_interrupt`` returns ``False`` — the atomic guarded UPDATE matched no ``live`` row
    owned by the caller (an already-consumed double-decision F-409, an unknown/never-recorded
    interrupt F-404, or a cross-athlete attempt F-XID). The run is NEVER resumed in that case; the
    API router maps this to 404/409. Lives beside the ledger guard it reports on (QUAL-R9 size
    split); re-exported from :mod:`wattwise_core.agent.engine` so the historical import path stays.
    """


async def record_interrupt(
    sessions: async_sessionmaker[AsyncSession],
    ensure_thread: EnsureThread,
    athlete_id: uuid.UUID,
    thread_id: str,
    interrupt_id: str,
) -> None:
    """Record a ``live`` approval-gate interrupt row (CKPT-R9), idempotently.

    Called by the interrupt-gate when the graph pauses an approval-gated plan.
    ``(thread_id, interrupt_id)`` is UNIQUE, so a gate re-raised on the SAME interrupt must neither
    error nor resurrect an already-``consumed`` row: insert-or-ignore through the sanctioned upsert
    seam (empty ``update_columns`` ⇒ existing row untouched), portable and atomic. ``athlete_id`` is
    the saver's bound identity (CKPT-R3), independently scoped.
    """
    async with sessions() as session:
        await ensure_thread(session, thread_id)
        await upsert(
            session,
            cast(Table, AgentInterrupt.__table__),  # ORM table is a Table at runtime
            {
                "id": uuid7(),
                "thread_id": thread_id,
                "athlete_id": athlete_id,
                "interrupt_id": interrupt_id,
                "status": "live",
            },
            conflict_keys=["thread_id", "interrupt_id"],
            update_columns=[],  # insert-or-ignore: never overwrite an existing row
        )
        await session.commit()


async def consume_interrupt(
    sessions: async_sessionmaker[AsyncSession],
    athlete_id: uuid.UUID,
    thread_id: str,
    interrupt_id: str,
) -> bool:
    """Atomically consume a ``live`` interrupt; True ⇒ resume, False ⇒ 404/409 (CKPT-R9).

    Called once per ``POST …/decision``. A single atomic, identity-scoped conditional update — flip
    ``status`` to ``consumed`` only for a still-``live`` row matching
    ``(thread_id, interrupt_id, athlete_id)`` — built through the SQLAlchemy query builder (NOT raw
    SQL), so it renders identically on all three backends and a cross-athlete decision never matches
    another owner's row (CKPT-R3). Under concurrent decisions exactly ONE flip wins (``rowcount==1``
    ⇒ resume); every other — double-decision (F-409), unknown (F-404), cross-identity (F-XID) — sees
    ``rowcount==0`` and is refused (fail-closed).
    """
    stmt = (
        update(AgentInterrupt)
        .where(
            AgentInterrupt.thread_id == thread_id,
            AgentInterrupt.interrupt_id == interrupt_id,
            AgentInterrupt.athlete_id == athlete_id,
            AgentInterrupt.status == "live",
        )
        .values(status="consumed")
    )
    async with sessions() as session:
        result = cast(CursorResult[Any], await session.execute(stmt))
        await session.commit()
        return result.rowcount == 1


async def interrupt_status(
    sessions: async_sessionmaker[AsyncSession],
    athlete_id: uuid.UUID,
    thread_id: str,
    interrupt_id: str,
) -> Literal["unknown", "live", "consumed"]:
    """Read-only, athlete-scoped status of an interrupt ledger row (CKPT-R9, API-R12a 404/409).

    The side-effect-free probe a refused ``POST …/decision`` consults to split ``404`` (unknown)
    from ``409`` (consumed/stale): the ``status`` of the ``(thread_id, interrupt_id, athlete_id)``
    row — ``live`` (a concurrent decision won the race) or ``consumed`` — else ``unknown`` when no
    such row exists FOR THIS ATHLETE. Athlete-scoped like :func:`consume_interrupt` (CKPT-R3) but
    mutates nothing, so a FOREIGN row reads ``unknown`` and is never disclosed.
    """
    stmt = select(AgentInterrupt.status).where(
        AgentInterrupt.thread_id == thread_id,
        AgentInterrupt.interrupt_id == interrupt_id,
        AgentInterrupt.athlete_id == athlete_id,
    )
    async with sessions() as session:
        status = (await session.execute(stmt)).scalar_one_or_none()
    if status == "live":
        return "live"
    return "consumed" if status == "consumed" else "unknown"


__all__ = ["DecisionRefused", "consume_interrupt", "interrupt_status", "record_interrupt"]
