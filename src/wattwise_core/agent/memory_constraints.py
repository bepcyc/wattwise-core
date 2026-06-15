"""Constraint-lifecycle query helpers over the agent-memory table (MEM-R6/MEM-R7, ADR 0008).

The focused sibling of :mod:`wattwise_core.agent.memory` (QUAL-R9 size split) that owns the
relational reads/writes of the CONSTRAINT lifecycle the :class:`~wattwise_core.agent.memory.
OssMemoryStore` exposes: writing an ACTIVE constraint, lifting one, and fetching the always-resident
active set. Behaviour is identical to the prior inline method bodies; this is purely a size
decomposition. Every query is scoped STRICTLY to the server-derived owner ``athlete_id`` (MEM-R3 /
AGT-SEC-R1).

Cited requirements: MEM-R3, MEM-R6, MEM-R7, GROUND-R14, AGT-SEC-R1, QUAL-R9.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Sequence

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.memory import (
    ConstraintSeverity,
    ConstraintStatus,
    MemoryItem,
    MemoryItemKind,
    RecalledItem,
    _recency_key,
    _to_recalled,
)


def _coerce_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """Coerce a string id to a UUID at the query boundary (portable UUID binds UUIDs)."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(value)


async def add_constraint(
    session: AsyncSession,
    *,
    athlete_id: str,
    content: str,
    severity: ConstraintSeverity = ConstraintSeverity.SOFT,
    inferred: bool = True,
    effective_until: _dt.datetime | None = None,
) -> RecalledItem:
    """Write an ACTIVE CONSTRAINT row in the athlete's own words (MEM-R7 / GROUND-R14, ADR 0008).

    The explicit capture path of the constraint lifecycle (ADR 0008 §4/§5): persists a
    CONSTRAINT-kind row with ``status=ACTIVE`` and the given ``severity`` (default SOFT — an
    inferred constraint is SOFT until the athlete confirms, never an unconfirmed HARD veto, ADR
    0008 §4), scoped to the server-derived owner (MEM-R3). TRUSTED owner-originated content — no
    untrusted-write guard applies (the explicit API path never carries source-synced text).
    ``effective_until`` is the optional self-expiry instant; ``None`` means it never expires.
    """
    row = MemoryItem(
        athlete_id=_coerce_uuid(athlete_id),
        kind=MemoryItemKind.CONSTRAINT,
        content=content,
        inferred=inferred,
        severity=severity,
        status=ConstraintStatus.ACTIVE,
        effective_until=effective_until,
    )
    session.add(row)
    await session.flush()
    return _to_recalled(row)


async def lift_constraint(session: AsyncSession, *, athlete_id: str, memory_item_id: str) -> bool:
    """Mark the owner's CONSTRAINT row LIFTED; ``True`` iff one was lifted (MEM-R7, ADR 0008 §4).

    The athlete is part of the shared decision (StARRT): they may LIFT a constraint they have
    cleared. The guarded update matches BOTH ``memory_item_id`` AND the server-derived owner id
    (AGT-SEC-R1), so a cross-athlete / unknown / non-UUID id lifts nothing and returns ``False``
    (fail-closed, never disclosing a foreign row). A LIFTED constraint stops gating.
    """
    try:
        item_id = _coerce_uuid(memory_item_id)
    except (ValueError, AttributeError):
        return False
    stmt = select(MemoryItem).where(
        MemoryItem.memory_item_id == item_id,
        MemoryItem.athlete_id == _coerce_uuid(athlete_id),
        MemoryItem.kind == MemoryItemKind.CONSTRAINT,
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        return False
    row.status = ConstraintStatus.LIFTED
    await session.flush()
    return True


async def fetch_active_constraints(
    session: AsyncSession, *, athlete_id: str, now: _dt.datetime
) -> Sequence[RecalledItem]:
    """The athlete's ALWAYS-RESIDENT active-constraint set (MEM-R6 / MEM-R7, ADR 0008 §3/§4).

    Returns EVERY CONSTRAINT-kind row for the server-derived owner (MEM-R3) that is both ACTIVE and
    not expired: ``status`` is ACTIVE (a NULL status reads as ACTIVE for backward compatibility with
    rows written before the lifecycle columns existed) AND ``effective_until`` is either NULL or
    strictly after ``now``. This is the non-evictable core tier (MEM-R6): NOT ranked against the
    keyword/recency pool and NOT subject to a ``limit`` — a standing safety constraint is never
    dropped by usage. Ordering is DETERMINISTIC: HARD before SOFT (the veto-bearing severity first),
    then most-recent first, then a stable id tiebreak.
    """
    stmt = select(MemoryItem).where(
        MemoryItem.athlete_id == _coerce_uuid(athlete_id),
        MemoryItem.kind == MemoryItemKind.CONSTRAINT,
        or_(MemoryItem.status.is_(None), MemoryItem.status == ConstraintStatus.ACTIVE),
        or_(MemoryItem.effective_until.is_(None), MemoryItem.effective_until > now),
    )
    rows = list((await session.execute(stmt)).scalars().all())
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if r.severity is ConstraintSeverity.HARD else 1,
            _recency_key(r.created_at),
            str(r.memory_item_id),
        ),
    )
    return [_to_recalled(r) for r in ranked]


__all__ = ["add_constraint", "fetch_active_constraints", "lift_constraint"]
