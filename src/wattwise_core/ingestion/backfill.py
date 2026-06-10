"""Backfill window chunking + the persisted resume cursor (SYN-R5/SYN-R6).

The on-demand backfill mechanism's stateless halves: chunk a historical range into
bounded OLDEST-FIRST windows (SYN-R5) and persist/read the per-source resume cursor
so a cancelled or interrupted backfill resumes WITHOUT re-downloading
already-committed windows (SYN-R6). The cursor rides the dedicated ``"backfill"``
stream sub-key of the SYN-R2 watermark entity — advanced only AFTER a window's
landing transaction committed, so the cursor never points past un-committed data
(SYN-R3). The orchestration loop lives on
:meth:`~wattwise_core.ingestion.sync.SyncOrchestrator.backfill`; the automatic
pacing/prioritisation around it is commercial (COMM-R19), not shipped here.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from wattwise_core.domain.enums import GboType
from wattwise_core.ingestion._sync_targets import SessionFactory, SyncWindow
from wattwise_core.ingestion.watermark import advance_watermark, watermark_for

#: The watermark ``stream`` sub-key carrying the per-source backfill resume cursor.
BACKFILL_STREAM = "backfill"


def chunk_windows(window: SyncWindow, chunk_days: int) -> list[SyncWindow]:
    """Chunk an inclusive ISO-date range into bounded OLDEST-FIRST windows (SYN-R5).

    Each window spans at most ``chunk_days`` days; windows are contiguous and
    non-overlapping, oldest first, so a long history lands incrementally with a
    per-window commit + cursor advance.
    """
    oldest = _dt.date.fromisoformat(window.oldest)
    newest = _dt.date.fromisoformat(window.newest)
    if newest < oldest:
        raise ValueError("backfill window newest precedes oldest")
    windows: list[SyncWindow] = []
    start = oldest
    while start <= newest:
        end = min(start + _dt.timedelta(days=chunk_days - 1), newest)
        windows.append(SyncWindow(oldest=start.isoformat(), newest=end.isoformat()))
        start = end + _dt.timedelta(days=1)
    return windows


async def read_backfill_cursor(
    session_factory: SessionFactory,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    *,
    range_oldest: str,
) -> _dt.date | None:
    """The newest date already committed by a prior backfill of THIS range (SYN-R6).

    The cursor is SCOPED to the range it was advancing (``"<range_oldest>/<through>"``):
    only a re-run of the SAME interrupted range resumes from it. A backfill of a
    DIFFERENT (e.g. older) range gets ``None`` — its windows were never downloaded, so
    skipping them would silently lose data (the re-walk of any overlap is idempotent,
    SYN-R4). Returns ``None`` when no prior cursor applies.
    """
    async with session_factory() as session:
        row = await watermark_for(
            session, athlete_id, source_descriptor_id, GboType.ACTIVITY, stream=BACKFILL_STREAM
        )
    if row is None or row.cursor is None:
        return None
    scoped_oldest, _, through = row.cursor.partition("/")
    if not through or scoped_oldest != range_oldest:
        return None
    return _dt.date.fromisoformat(through)


async def advance_backfill_cursor(
    session_factory: SessionFactory,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    *,
    range_oldest: str,
    through: _dt.date,
    ingest_run_id: uuid.UUID | None = None,
) -> None:
    """Persist the resume cursor AFTER a window's landing transaction committed (SYN-R3/R6).

    Advanced strictly after the window's data is durable, so a crash between the
    commit and this advance only re-downloads ONE already-idempotent window (SYN-R4)
    — the cursor never points past un-committed data. The high-water instant is the
    window's end-of-day so the monotonic forward-only guard holds oldest-first.
    """
    end_of_day = _dt.datetime.combine(through, _dt.time(23, 59, 59), tzinfo=_dt.UTC)
    async with session_factory() as session:
        await advance_watermark(
            session,
            athlete_id,
            source_descriptor_id,
            GboType.ACTIVITY,
            high_water_at=end_of_day,
            content_hint=None,
            cursor=f"{range_oldest}/{through.isoformat()}",
            stream=BACKFILL_STREAM,
            ingest_run_id=ingest_run_id,
        )


__all__ = [
    "BACKFILL_STREAM",
    "advance_backfill_cursor",
    "chunk_windows",
    "read_backfill_cursor",
]
