"""Agent-state retention-window sweeper for durable checkpoints (CKPT-R8, PRIV-R7).

CKPT-R8 requires that checkpoints "expire per the configured retention window"; PRIV-R7
names the agent-state store a retention category that MUST have a configurable retention
window covering BOTH (a) durable run checkpoints (threads, interrupts, resumable run state)
AND (b) durable athlete memory (the agent's persisted coaching memory, which MAY hold
special-category / health-adjacent content per MEM-R3). Both sub-categories MUST be retained
no longer than that configured window. Purge on request is already fulfilled by
:func:`wattwise_core.privacy.erasure.erase_athlete`; this module adds the TIME-WINDOW expiry:
a sweeper that deletes agent-state rows older than the configured window so durable state is
retained no longer than that window.

The window is LOADED config (``retention__agent_state_days``, CFG-R1a) — never a code
literal; ``0`` means retain indefinitely (no sweep), mirroring ``retention__raw_file_days``.
The sweep is athlete-AGNOSTIC (it expires by age across all threads) and runs in the caller's
agent-state transaction; it is the time-window complement of the per-athlete erasure.

Deletion is children-first by construction (writes/checkpoints/interrupts before their
thread) so foreign keys hold even under RESTRICT, mirroring the erasure executor's ordering.
A thread is expired only when ITS OWN ``created_at`` precedes the cutoff, so a long-running
conversation that still receives fresh checkpoints is not torn out from under an active run.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import Select, delete, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.memory import MemoryItem
from wattwise_core.agent.state_store import (
    AgentCheckpoint,
    AgentInterrupt,
    AgentThread,
    AgentWrite,
)


@dataclass(frozen=True, slots=True)
class SweepReport:
    """Per-table rows deleted by one retention sweep (auditable, CKPT-R8)."""

    cutoff: _dt.datetime
    deleted_writes: int
    deleted_checkpoints: int
    deleted_interrupts: int
    deleted_threads: int
    deleted_memory: int

    @property
    def total(self) -> int:
        """Total agent-state rows expired in this sweep."""
        return (
            self.deleted_writes
            + self.deleted_checkpoints
            + self.deleted_interrupts
            + self.deleted_threads
            + self.deleted_memory
        )


async def sweep_expired_checkpoints(
    session: AsyncSession, *, retention_days: int, now: _dt.datetime | None = None
) -> SweepReport:
    """Expire durable agent-state rows older than the configured window (CKPT-R8 / PRIV-R7).

    ``retention_days`` is the configured retention window in days (loaded config, CFG-R1a).
    A non-positive window means retain indefinitely: NOTHING is swept (the report is all-zero),
    consistent with ``retention__raw_file_days = 0``. Otherwise every agent-state row whose
    ``created_at`` precedes ``now - retention_days`` is deleted, children-first so foreign keys
    hold: the writes/checkpoints/interrupts of an expired thread go before the thread row.

    Both PRIV-R7 sub-categories are swept: (a) durable run checkpoints (threads/interrupts/
    writes/resumable state) AND (b) durable athlete memory (``MemoryItem``), which MAY hold
    special-category / health-adjacent content (MEM-R3) and so MUST not outlive the window.
    Memory rows carry no foreign key into ``AgentThread`` (athlete-scoped only, like
    ``AgentInterrupt``); each is expired independently by its OWN ``created_at``.

    A thread is expired only when its OWN ``created_at`` precedes the cutoff; its child
    checkpoints/writes/interrupts are also expired individually by their own age, so a still-
    active thread keeps its recent checkpoints. The caller owns the commit boundary.
    """
    reference = now if now is not None else _dt.datetime.now(_dt.UTC)
    if retention_days <= 0:
        return SweepReport(reference, 0, 0, 0, 0, 0)
    cutoff = reference - _dt.timedelta(days=retention_days)

    expired_threads = select(AgentThread.thread_id).where(AgentThread.created_at < cutoff)
    deleted_writes = await _delete_older_than(session, AgentWrite, cutoff)
    deleted_checkpoints = await _delete_older_than(session, AgentCheckpoint, cutoff)
    deleted_interrupts = await _delete_older_than(session, AgentInterrupt, cutoff)
    # Durable athlete memory (PRIV-R7 sub-category (b)) — expired by its own age, no thread FK.
    deleted_memory = await _delete_older_than(session, MemoryItem, cutoff)
    # A thread is removed only when its own children (by age) AND any rows belonging to a
    # still-younger thread are gone; deleting expired threads last keeps FK order correct.
    deleted_threads = await _delete_threads(session, expired_threads_stmt=expired_threads)

    return SweepReport(
        cutoff=cutoff,
        deleted_writes=deleted_writes,
        deleted_checkpoints=deleted_checkpoints,
        deleted_interrupts=deleted_interrupts,
        deleted_threads=deleted_threads,
        deleted_memory=deleted_memory,
    )


async def _delete_older_than(
    session: AsyncSession,
    model: type[AgentCheckpoint] | type[AgentWrite] | type[AgentInterrupt] | type[MemoryItem],
    cutoff: _dt.datetime,
) -> int:
    """Delete (and count) rows of one agent-state table whose ``created_at`` precedes ``cutoff``."""
    result = cast(
        CursorResult[Any],
        await session.execute(delete(model).where(model.created_at < cutoff)),
    )
    return int(result.rowcount or 0)


async def _delete_threads(
    session: AsyncSession, *, expired_threads_stmt: Select[tuple[str]]
) -> int:
    """Delete thread rows that are themselves past the cutoff (children already swept)."""
    thread_ids = list((await session.execute(expired_threads_stmt)).scalars().all())
    if not thread_ids:
        return 0
    # Remove any remaining children of these threads first (a child younger than the cutoff
    # whose parent thread is itself expired) so a RESTRICT foreign key still holds.
    await session.execute(delete(AgentWrite).where(AgentWrite.thread_id.in_(thread_ids)))
    await session.execute(
        delete(AgentCheckpoint).where(AgentCheckpoint.thread_id.in_(thread_ids))
    )
    await session.execute(
        delete(AgentInterrupt).where(AgentInterrupt.thread_id.in_(thread_ids))
    )
    result = cast(
        CursorResult[Any],
        await session.execute(delete(AgentThread).where(AgentThread.thread_id.in_(thread_ids))),
    )
    return int(result.rowcount or 0)


__all__ = ["SweepReport", "sweep_expired_checkpoints"]
