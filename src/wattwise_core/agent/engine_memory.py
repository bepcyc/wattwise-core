"""The athlete-scoped memory READ + ERASE seam the memory router drives (MEM-R3/-R4, API).

The focused sibling of :mod:`wattwise_core.agent.engine` (QUAL-R9 size split) that owns the
list / get / delete-per-id access over durable athlete memory the GET/DELETE ``/v1/agent/memory``
endpoints need (MEM-R3 erasure is a privacy MUST). It complements the recall-oriented
:class:`~wattwise_core.agent.memory.MemoryStore` (``fetch_relevant`` / ``erase``-all) with the
per-row CRUD-read the API surface requires, querying the SAME dedicated agent-state
:class:`~wattwise_core.agent.memory.MemoryItem` table.

EVERY query is scoped to the SERVER-DERIVED owner ``athlete_id`` (MEM-R3 / AGT-SEC-R1): a row
owned by another athlete is NEVER listed, returned, or deleted, and an id that does not belong to
the caller reads as absent (``None`` / a ``0``-row delete) — never disclosed and never a
cross-identity leak (fail-closed). The seam returns the engine-facing
:class:`~wattwise_core.agent.memory.RecalledItem` projection (personalization context only, never a
canonical analytic number, MEM-R1). It writes nothing except the explicit per-id / whole-athlete
deletes the erasure endpoint requests.

Cited requirements: MEM-R1, MEM-R3, MEM-R4, AGT-SEC-R1, PRIV-R8, CKPT-R8.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.memory import MemoryItem, RecalledItem, _to_recalled


def _coerce_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """Coerce an id to UUID at the query boundary, or ``None`` for a non-UUID token.

    A malformed memory-item id (a client-supplied path segment that is not a UUID) is NOT an
    error here — it simply matches no row the athlete owns, so the get/delete reads as absent
    (fail-closed), exactly like a well-formed id belonging to another athlete.
    """
    return value if isinstance(value, uuid.UUID) else uuid.UUID(value)


def _try_uuid(value: str) -> uuid.UUID | None:
    """Parse a path id to UUID, or ``None`` when it is not a UUID (fail-closed absence)."""
    try:
        return _coerce_uuid(value)
    except (ValueError, AttributeError):
        return None


async def list_memory(
    session: AsyncSession, *, athlete_id: str, limit: int, offset: int = 0
) -> Sequence[RecalledItem]:
    """List the athlete's durable memory rows, most-recent first, paginated (MEM-R3/-R4).

    Scoped STRICTLY to the server-derived owner ``athlete_id`` (MEM-R3): another athlete's rows
    are never listed. Deterministic ordering — newest ``created_at`` first, then a stable id
    tiebreak — so the keyset/offset page is reproducible. Returns the engine-facing
    :class:`RecalledItem` projection (personalization context only, never a canonical number,
    MEM-R1).
    """
    stmt = (
        select(MemoryItem)
        .where(MemoryItem.athlete_id == _coerce_uuid(athlete_id))
        .order_by(MemoryItem.created_at.desc(), MemoryItem.memory_item_id.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [_to_recalled(row) for row in rows]


async def get_memory(
    session: AsyncSession, *, athlete_id: str, memory_item_id: str
) -> RecalledItem | None:
    """Fetch ONE memory row by id, scoped to the owner, else ``None`` (MEM-R3, fail-closed).

    The lookup is by BOTH ``memory_item_id`` AND the server-derived ``athlete_id`` (AGT-SEC-R1):
    a row owned by another athlete — or a non-UUID / unknown id — returns ``None`` and is NEVER
    disclosed (the router maps ``None`` to a 404, indistinguishable from truly-absent). Identity
    is never taken from the client.
    """
    item_id = _try_uuid(memory_item_id)
    if item_id is None:
        return None
    stmt = select(MemoryItem).where(
        MemoryItem.memory_item_id == item_id,
        MemoryItem.athlete_id == _coerce_uuid(athlete_id),
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    return _to_recalled(row) if row is not None else None


async def delete_memory(
    session: AsyncSession, *, athlete_id: str, memory_item_id: str
) -> bool:
    """Delete ONE memory row by id, scoped to the owner; True iff a row was erased (MEM-R3).

    The guarded DELETE matches BOTH ``memory_item_id`` AND the server-derived ``athlete_id``, so
    a cross-athlete or unknown/non-UUID id deletes NOTHING and returns ``False`` (the router maps
    that to a 404 — a foreign row is never confirmed to exist, AGT-SEC-R1). Erasure is a privacy
    MUST (PRIV-R8 / CKPT-R8); the caller's transaction commits the delete.
    """
    item_id = _try_uuid(memory_item_id)
    if item_id is None:
        return False
    stmt = delete(MemoryItem).where(
        MemoryItem.memory_item_id == item_id,
        MemoryItem.athlete_id == _coerce_uuid(athlete_id),
    )
    result = cast(CursorResult[Any], await session.execute(stmt))
    await session.flush()
    return bool(result.rowcount)


async def erase_memory(session: AsyncSession, *, athlete_id: str) -> int:
    """Erase ALL of the athlete's memory rows; returns the count erased (MEM-R3 erasure).

    The whole-athlete erasure the privacy endpoint requests (PRIV-R8): a guarded DELETE scoped to
    the server-derived owner only, never widening to another identity. Returns how many rows were
    removed so the endpoint can report the erasure deterministically.
    """
    stmt = delete(MemoryItem).where(MemoryItem.athlete_id == _coerce_uuid(athlete_id))
    result = cast(CursorResult[Any], await session.execute(stmt))
    await session.flush()
    return int(result.rowcount or 0)


__all__ = [
    "delete_memory",
    "erase_memory",
    "get_memory",
    "list_memory",
]
