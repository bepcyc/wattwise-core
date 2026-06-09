"""Per-record isolation + watermark-window + record-gap helpers for the sync engine.

A focused split of the orchestrator's record-level concerns (QUAL-R9): the per-ASBO map
isolation that keeps good records when one fails (ING-GAP-R5 / ING-UPS-R3), the
ADP-R6 incremental-window narrowing that skips an already-watermarked range, and the
range-precise per-record gap open (ING-GAP-R5). These are L3 ingestion-side helpers; they
import only the rankless domain package, the adapter-contract seam, and the watermark
writer — never a consumer layer.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import GapReason, GboType, Severity
from wattwise_core.ingestion.base import FetchContext, SourceAdapter, SourceDescriptorRef
from wattwise_core.ingestion.watermark import open_gap, watermark_for
from wattwise_core.observability.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FailedRecord:
    """One ASBO whose pure map raised — the un-ingested record range (ING-GAP-R5).

    Carries the source native id (the discovery record-token) and the GBO type so a
    range-precise, typed gap can be opened covering EXACTLY this record while every other
    record in the same batch still commits (ING-GAP-R5 / ING-UPS-R3).
    """

    source_native_id: str
    gbo_type: GboType


@dataclass(slots=True)
class MappedBatch:
    """The per-record-isolated result of fetch+map for one source.

    ``candidates`` are the records that mapped cleanly (committed); ``failed`` are the
    records whose pure map raised, each becoming a range-precise gap (ING-GAP-R5) so a
    single bad record never discards the good ones (ING-UPS-R3).
    """

    candidates: list[GboCandidate]
    failed: list[FailedRecord]


def map_records_isolated(
    adapter: SourceAdapter,
    asbos: Iterable[Any],
    ref: SourceDescriptorRef,
    ctx: FetchContext,
    *,
    source_key: str,
) -> MappedBatch:
    """Pure-map each ASBO in ISOLATION (ING-GAP-R5 / ING-UPS-R3).

    A record whose map raises is recorded as a :class:`FailedRecord` (a range-precise gap
    is opened for exactly it at landing) while every other record still maps and commits
    — a single bad record never discards the good ones (ING-UPS-R3).
    """
    candidates: list[GboCandidate] = []
    failed: list[FailedRecord] = []
    for asbo in asbos:
        try:
            candidates.extend(adapter.map(asbo, ref, ctx))
        except Exception as exc:  # per-record isolation: gap-mark only this record
            _log.warning(
                "sync.record_mapping_failed", source_key=source_key, error_type=type(exc).__name__
            )
            failed.append(
                FailedRecord(source_native_id=_asbo_token(asbo), gbo_type=_asbo_gbo_type(asbo))
            )
    return MappedBatch(candidates=candidates, failed=failed)


async def open_record_gaps(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    failed: list[FailedRecord],
    *,
    ingest_run_id: uuid.UUID,
    seen_at: _dt.datetime,
) -> None:
    """Open a typed, range-precise gap for each per-record map failure (ING-GAP-R5).

    Each failed record gets ONE ``mapping_field_missing`` gap covering exactly that
    record token; the caller invokes this inside the batch transaction so it commits with
    the good records (ING-UPS-R2). It is ``terminal`` (``transient=False``): a map raising
    is a mapping code/schema defect, not a flaky fetch — re-running the SAME source through
    the SAME map deterministically re-fails (MAP-R1 purity), so it needs a code/schema fix
    by the operator (ING-GAP-R2 ``terminal``), it is NOT auto-retryable and MUST NOT be
    auto-closed by the transient self-heal (ING-GAP-R4).
    """
    for rec in failed:
        await open_gap(
            session,
            athlete_id,
            source_descriptor_id,
            rec.gbo_type,
            reason=GapReason.MAPPING_FIELD_MISSING,
            seen_at=seen_at,
            severity=Severity.WARNING,
            transient=False,
            range_start_token=rec.source_native_id,
            range_end_token=rec.source_native_id,
            ingest_run_id=ingest_run_id,
        )


async def watermark_floor(
    session: AsyncSession, athlete_id: uuid.UUID, source_descriptor_id: uuid.UUID
) -> _dt.datetime | None:
    """The MOST-CONSERVATIVE (minimum) ``high_water_at`` across a source's scopes (ADP-R6).

    Each ``gbo_type`` carries its OWN independent cursor (SYN-R2 keys the watermark per
    ``(athlete, source, gbo_type[, stream])``). One discover call may yield references for
    several gbo_types, and the caller cannot know which gbo_types this run will produce.
    Advancing the single fetch-window floor to the MAX across types would over-advance any
    LAGGING type's window — its un-fetched range would be silently skipped and never
    re-discovered (the inverse of ADP-R6). So the floor is the MIN high-water across every
    advanced scope: the window starts no later than the least-advanced cursor, guaranteeing
    no lagging gbo_type's un-fetched range is skipped (a type already past the floor simply
    re-converges idempotently per SYN-R4). A scope never advanced contributes no floor.
    """
    floor: _dt.datetime | None = None
    for gbo_type in GboType:
        wm = await watermark_for(session, athlete_id, source_descriptor_id, gbo_type)
        if wm is None or wm.high_water_at is None:
            continue
        if floor is None or wm.high_water_at < floor:
            floor = wm.high_water_at
    return floor


async def incremental_floor_date(
    session_factory: Any,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    oldest_iso: str,
) -> str:
    """The ISO date an incremental fetch must start from, skipping watermarked ranges (ADP-R6).

    Moves the window floor forward to the source's LEAST-advanced high-water cursor across
    its gbo_type scopes (never backward), so an incremental fetch does NOT re-pull a range
    already known current for EVERY scope, yet never over-advances past a lagging scope's
    un-fetched range (SYN-R2 per-gbo_type cursor). The given ``oldest_iso`` is returned
    unchanged when no watermark exists or it is later than the floor.
    """
    async with session_factory() as session:
        high_water = await watermark_floor(session, athlete_id, source_descriptor_id)
    if high_water is None:
        return oldest_iso
    floor = high_water.date().isoformat()
    return oldest_iso if floor <= oldest_iso else floor


def _asbo_token(asbo: Any) -> str:
    """A stable, non-leaking record token for a failed ASBO (the discovery id; ING-GAP-R5).

    Prefers a structural ``native_id`` / ``id`` attribute (the record's own identity, not
    a source name — Principle A holds); falls back to a positional placeholder so a gap is
    still range-bounded to the single failed record rather than the whole batch.
    """
    token = getattr(asbo, "native_id", None) or getattr(asbo, "id", None)
    return str(token) if token is not None else "unknown_record"


def _asbo_gbo_type(asbo: Any) -> GboType:
    """The GBO type a failed ASBO would have produced; defaults to ``activity`` (ING-GAP-R5).

    Uses a structural ``gbo_type`` hint if the ASBO carries one, else the most common
    record type so the gap is typed; the gap range-token pins the exact failed record.
    """
    raw = getattr(asbo, "gbo_type", None)
    if raw is not None:
        try:
            return GboType(str(raw))
        except ValueError:
            return GboType.ACTIVITY
    return GboType.ACTIVITY


__all__ = [
    "FailedRecord",
    "MappedBatch",
    "incremental_floor_date",
    "map_records_isolated",
    "open_record_gaps",
    "watermark_floor",
]
