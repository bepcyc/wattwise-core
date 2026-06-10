"""Watermark advance + typed-gap open/close helpers (SYN-R2/R3, ING-GAP-R2..R5).

Stateless functions over a caller-supplied :class:`AsyncSession`, so the watermark
advance (SYN-R3) and any gap open/close (ING-GAP-R4) ride the SAME transaction as the
batch upsert they represent (ING-UPS-R2) — store state, cursor state, and gap state stay
mutually consistent, and a crash mid-run never advances past un-committed data (ING-R6).

These are L3 ingestion-side writers to source-derived canonical entities (ARCH-R3
canonical-write partition); they take no clock and read none — the caller passes the
instants so the path stays deterministic.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import GapReason, GapState, GboType, Severity
from wattwise_core.observability import metrics as _metrics
from wattwise_core.persistence.models.source import IngestionGap, IngestionWatermark

# The empty-string sentinel for "no per-stream sub-key" (SYN-R2 ``[, stream]``), so the
# composite UNIQUE behaves identically across SQLite/PostgreSQL/MariaDB (no NULL key).
_NO_STREAM = ""


@dataclass(frozen=True, slots=True)
class SyncedRange:
    """The time range a run successfully covered, for watermark + gap bookkeeping.

    Drives the SAME-transaction watermark advance (SYN-R3) and transient-gap self-heal
    (ING-GAP-R4): the watermark advances to ``newest`` and any OPEN transient gap fully
    inside ``[oldest, newest]`` for the committed scope is closed. ``now`` stamps the
    closure time (ING-GAP-R4); passed in so the write path takes no clock.
    """

    oldest: _dt.datetime
    newest: _dt.datetime
    now: _dt.datetime


@dataclass(frozen=True, slots=True)
class AdvanceResult:
    """How many watermarks advanced and transient gaps self-healed in one batch."""

    watermarks_advanced: int = 0
    gaps_closed: int = 0


async def advance_and_heal(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    committed: list[GboCandidate],
    synced_range: SyncedRange,
    *,
    ingest_run_id: uuid.UUID | None = None,
) -> AdvanceResult:
    """Advance the watermark + close covered transient gaps in THIS transaction.

    For each ``gbo_type`` that committed a record in ``committed``, advance its watermark
    to the most-recent ingested instant with the content hint (SYN-R2/R3), and close
    every OPEN transient gap fully inside the synced range (ING-GAP-R4). Both writes ride
    the caller's transaction, the SAME transaction as the batch upsert (ING-UPS-R2), so
    store, cursor, and gap state stay mutually consistent (ING-R6).
    """
    high_water: dict[GboType, tuple[_dt.datetime | None, str | None]] = {}
    for cand in committed:
        gbo_type = GboType(cand.gbo_type)
        observed = cand.observed_at
        prior = high_water.get(gbo_type)
        if prior is None or (observed is not None and (prior[0] is None or observed > prior[0])):
            high_water[gbo_type] = (observed, cand.content_hash)
    advanced = 0
    closed = 0
    for gbo_type, (high_water_at, content_hint) in high_water.items():
        await advance_watermark(
            session, athlete_id, source_descriptor_id, gbo_type,
            high_water_at=high_water_at or synced_range.newest,
            content_hint=content_hint,
            ingest_run_id=ingest_run_id,
        )
        advanced += 1
        closed += await close_covering_gaps(
            session, athlete_id, source_descriptor_id, gbo_type,
            range_start_at=synced_range.oldest,
            range_end_at=synced_range.newest,
            closed_at=synced_range.now,
        )
    closed += await close_token_gaps(
        session, athlete_id, source_descriptor_id,
        {c.source_native_id for c in committed},
        closed_at=synced_range.now,
    )
    return AdvanceResult(watermarks_advanced=advanced, gaps_closed=closed)


async def watermark_for(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    gbo_type: GboType,
    *,
    stream: str = _NO_STREAM,
) -> IngestionWatermark | None:
    """Read the watermark for one ingest scope, or ``None`` if never advanced (SYN-R2)."""
    stmt = select(IngestionWatermark).where(
        IngestionWatermark.athlete_id == athlete_id,
        IngestionWatermark.source_descriptor_id == source_descriptor_id,
        IngestionWatermark.gbo_type == gbo_type,
        IngestionWatermark.stream == stream,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def advance_watermark(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    gbo_type: GboType,
    *,
    high_water_at: _dt.datetime | None,
    content_hint: str | None,
    cursor: str | None = None,
    stream: str = _NO_STREAM,
    ingest_run_id: uuid.UUID | None = None,
) -> IngestionWatermark:
    """Advance (or create) the watermark for one ingest scope (SYN-R2/R3).

    MONOTONIC: the high-water timestamp only moves FORWARD — a lower ``high_water_at`` is
    never written, so a re-run never regresses the cursor (SYN-R3/SYN-R4). The content
    hint is always refreshed so a changed-but-not-new record is re-fetched next time
    (SYN-R2). The write rides the caller's transaction; it is the SAME transaction as the
    batch upsert (ING-UPS-R2), so the cursor never advances past un-committed data.
    """
    row = await watermark_for(
        session, athlete_id, source_descriptor_id, gbo_type, stream=stream
    )
    if row is None:
        row = IngestionWatermark(
            athlete_id=athlete_id,
            source_descriptor_id=source_descriptor_id,
            gbo_type=gbo_type,
            stream=stream,
            high_water_at=high_water_at,
            cursor=cursor,
            content_hint=content_hint,
            ingest_run_id=ingest_run_id,
        )
        session.add(row)
        await session.flush()
        return row
    if high_water_at is not None and (
        row.high_water_at is None or high_water_at > row.high_water_at
    ):
        row.high_water_at = high_water_at  # forward-only (SYN-R3 no-regress)
    if cursor is not None:
        row.cursor = cursor
    row.content_hint = content_hint
    row.ingest_run_id = ingest_run_id
    await session.flush()
    return row


async def open_gap(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID | None,
    gbo_type: GboType,
    *,
    reason: GapReason,
    seen_at: _dt.datetime,
    severity: Severity = Severity.WARNING,
    transient: bool = True,
    range_start_at: _dt.datetime | None = None,
    range_end_at: _dt.datetime | None = None,
    range_start_token: str | None = None,
    range_end_token: str | None = None,
    ingest_run_id: uuid.UUID | None = None,
) -> IngestionGap:
    """Open a typed, range-precise gap for a partial failure (ING-GAP-R2/R3/R5).

    The gap covers EXACTLY the un-ingested range (a time range and/or a discovery
    record-token range), never the whole run; successfully ingested records in the same
    run are left committed (ING-GAP-R5). It is opened ``state=open`` with a ``transient``
    flag (auto-retryable vs. terminal) and first/last-seen stamped to ``seen_at``.
    """
    gap = IngestionGap(
        athlete_id=athlete_id,
        source_descriptor_id=source_descriptor_id,
        gbo_type=gbo_type,
        reason=reason,
        severity=severity,
        state=GapState.OPEN,
        transient=transient,
        range_start_at=range_start_at,
        range_end_at=range_end_at,
        range_start_token=range_start_token,
        range_end_token=range_end_token,
        ingest_run_id=ingest_run_id,
        first_seen_at=seen_at,
        last_seen_at=seen_at,
    )
    session.add(gap)
    await session.flush()
    # ING-OBS-R2: open-gap counts are queryable by reason on the metrics surface.
    _metrics.get_registry().increment(
        _metrics.INGEST_GAPS_OPENED, labels={"reason": reason.value}
    )
    return gap


async def close_covering_gaps(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    gbo_type: GboType,
    *,
    range_start_at: _dt.datetime,
    range_end_at: _dt.datetime,
    closed_at: _dt.datetime,
) -> int:
    """Self-heal: close every OPEN transient gap fully covered by a successful range.

    A transient gap is self-healing — a later successful sync covering the same range
    MUST close it automatically and record the closure time (ING-GAP-R4). A gap is closed
    only when the freshly-synced ``[range_start_at, range_end_at]`` time range fully
    covers the gap's own time range; a TERMINAL gap (needs user/operator action) is never
    auto-closed here. Returns the number of gaps closed.
    """
    stmt = select(IngestionGap).where(
        IngestionGap.athlete_id == athlete_id,
        IngestionGap.source_descriptor_id == source_descriptor_id,
        IngestionGap.gbo_type == gbo_type,
        IngestionGap.state == GapState.OPEN,
        IngestionGap.transient.is_(True),
    )
    closed = 0
    for gap in (await session.execute(stmt)).scalars().all():
        if gap.range_start_at is None or gap.range_end_at is None:
            continue
        if range_start_at <= gap.range_start_at and gap.range_end_at <= range_end_at:
            gap.state = GapState.CLOSED
            gap.closed_at = closed_at
            gap.last_seen_at = closed_at
            closed += 1
    if closed:
        await session.flush()
        # ING-OBS-R2: the transient self-heal rate is observable (ING-GAP-R4).
        _metrics.get_registry().increment(_metrics.INGEST_GAPS_CLOSED, amount=float(closed))
    return closed


async def close_token_gaps(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    landed_native_ids: set[str],
    *,
    closed_at: _dt.datetime,
) -> int:
    """Self-heal token-precise transient gaps whose record just landed (ING-GAP-R4).

    A per-record ``fetch_failed`` gap covers exactly one record token (ING-GAP-R5);
    when a later successful sync lands that record, the gap closes automatically with
    the closure time recorded. Terminal gaps are never auto-closed. Returns the count.
    """
    if not landed_native_ids:
        return 0
    stmt = select(IngestionGap).where(
        IngestionGap.athlete_id == athlete_id,
        IngestionGap.source_descriptor_id == source_descriptor_id,
        IngestionGap.state == GapState.OPEN,
        IngestionGap.transient.is_(True),
        IngestionGap.range_start_token.in_(landed_native_ids),
    )
    closed = 0
    for gap in (await session.execute(stmt)).scalars().all():
        if gap.range_end_token != gap.range_start_token:
            continue  # only single-record token gaps heal here
        gap.state = GapState.CLOSED
        gap.closed_at = closed_at
        gap.last_seen_at = closed_at
        closed += 1
    if closed:
        await session.flush()
        _metrics.get_registry().increment(_metrics.INGEST_GAPS_CLOSED, amount=float(closed))
    return closed


__all__ = [
    "AdvanceResult",
    "SyncedRange",
    "advance_and_heal",
    "advance_watermark",
    "close_covering_gaps",
    "close_token_gaps",
    "open_gap",
    "watermark_for",
]
