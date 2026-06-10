"""Original-file retention-window purge (PRIV-R7 / PRIV-R11.2).

PRIV-R11.2: the original-file object store has a CONFIGURABLE retention window; retained
original recording files older than the window MUST be purged — the OBJECT bytes and the
relational reference both — while the canonical typed model derived from a file outlives
it (purging an original NEVER deletes the canonical typed data already derived from it,
which this sweep never touches). PRIV-R7 names the window: ``retention__raw_file_days``,
loaded config (CFG-R1a), where ``0`` means retain indefinitely (no sweep), mirroring the
agent-state window.

The sweep is athlete-AGNOSTIC (it expires by age across the store) and is the time-window
complement of the per-athlete erasure (:func:`wattwise_core.privacy.erasure.erase_athlete`,
which deletes the objects on request, PRIV-R11.3). Object deletion is attempted before the
reference delete and tolerates an already-absent object (idempotent re-run); the row
delete rides the caller's transaction.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
from collections.abc import Callable

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.persistence.models.activity import ActivityFile
from wattwise_core.persistence.types import utcnow
from wattwise_core.storage import ObjectStore


async def purge_expired_original_files(
    session: AsyncSession,
    object_store: ObjectStore,
    *,
    retention_days: int,
    now: Callable[[], _dt.datetime] = utcnow,
) -> int:
    """Purge original files older than the configured window; return rows removed.

    ``retention_days <= 0`` retains indefinitely (no sweep) — the documented sentinel of
    ``retention__raw_file_days``. Otherwise every ``activity_file`` whose ``created_at``
    precedes the cutoff has its object BYTES deleted from the object store (PRIV-R11.2 —
    the object itself, not merely the reference) and its reference row removed. The
    canonical activity row and every derived typed record stay untouched.
    """
    if retention_days <= 0:
        return 0
    cutoff = now() - _dt.timedelta(days=retention_days)
    rows = (
        (
            await session.execute(
                select(ActivityFile.activity_file_id, ActivityFile.object_ref).where(
                    ActivityFile.created_at < cutoff
                )
            )
        )
        .tuples()
        .all()
    )
    if not rows:
        return 0
    for _file_id, object_ref in rows:
        # Idempotent: a re-run after a partial sweep continues past already-gone bytes.
        with contextlib.suppress(FileNotFoundError, KeyError):
            object_store.delete(object_ref)
    await session.execute(
        delete(ActivityFile).where(
            ActivityFile.activity_file_id.in_([file_id for file_id, _ in rows])
        )
    )
    return len(rows)


__all__ = ["purge_expired_original_files"]
