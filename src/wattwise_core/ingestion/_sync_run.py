"""Per-source sync-run execution: window narrowing, fetch paths, landing (QUAL-R9 split).

The orchestrator's per-source work, extracted as free functions over a small
:class:`RunContext` so :class:`~wattwise_core.ingestion.sync.SyncOrchestrator` stays
a thin dispatcher: the ADP-R6 incremental-window narrowing, the five-phase discover
pipeline drive and the legacy window-fetch seam, and the landing transaction that
commits the upsert + watermark advance + every typed gap together (ING-UPS-R2) —
including the ADP-R3 declared-GBO-type REFUSAL (terminal ``schema_mismatch`` gap,
nothing written) and the ING-R3 degrade-with-persisted-gap path.

Layer: L3 ingestion-side helpers; imported only by the orchestrator (no cycle, no
consumer edge).
"""

from __future__ import annotations

import datetime as _dt
import time as _time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from wattwise_core.domain.enums import GapReason, GboType
from wattwise_core.ingestion._sync_discover import (
    DiscoverFetch,
    DiscoverOutput,
    PhaseStats,
    emit_run_trace,
    open_discover_gaps,
    run_discover_pipeline,
)
from wattwise_core.ingestion._sync_records import (
    MappedBatch,
    map_records_isolated,
    open_record_gaps,
    watermark_floor,
)
from wattwise_core.ingestion._sync_targets import (
    SessionFactory,
    SourceSyncResult,
    SyncOutcome,
    SyncWindow,
    _ConnectionTarget,
    _uid,
    degrade_with_gap,
    degraded,
    resolve_api_key,
    synced_range,
)
from wattwise_core.ingestion.base import FetchContext, SourceAdapter, SourceDescriptorRef
from wattwise_core.ingestion.capability import UndeclaredGboTypeError
from wattwise_core.ingestion.ingest import IngestResult, IngestService, OriginalFile
from wattwise_core.ingestion.watermark import open_gap
from wattwise_core.observability import metrics as _metrics
from wattwise_core.observability.logging import get_logger
from wattwise_core.seams import SessionProvider
from wattwise_core.security.credentials import CredentialStore

_log = get_logger(__name__)


@runtime_checkable
class AdapterFetch(SourceAdapter, Protocol):
    """A :class:`SourceAdapter` that ALSO exposes a window-fetch seam (ADP-R8, legacy).

    The orchestrator drives fetch polymorphically through this structural shape,
    never by naming a source (ARCH-R2). ``fetch`` is the IMPURE side (network/file
    I/O) kept strictly out of the pure :meth:`SourceAdapter.map`. The shipped
    direct-API adapter implements the richer five-phase
    :class:`~wattwise_core.ingestion._sync_discover.DiscoverFetch` contract instead;
    this seam remains for adapters exposing only ``fetch``.
    """

    async def fetch(
        self, *, api_key: str | None, athlete_native_id: str | None, window: SyncWindow
    ) -> Iterable[Any]:
        """Fetch source-shaped objects for the window (impure I/O; ADP-R8)."""
        ...


@runtime_checkable
class OriginalArtifactSource(Protocol):
    """An adapter that acquires VERBATIM original recording files for capture (ING-R8).

    The orchestrator threads any originals a source yielded into the landing
    transaction so the bytes are stored tier-1 and an ``activity_file`` reference is
    created (FIL-R1). A direct-API source (e.g. one returning JSON, not a recording
    file) does NOT implement this, so the orchestrator captures NO file for it.
    """

    def original_files(self) -> list[OriginalFile]:
        """The verbatim originals acquired in the last fetch (empty if none)."""
        ...


@dataclass(frozen=True, slots=True)
class RunContext:
    """The orchestrator's injected seams, threaded to the per-source run functions."""

    sessions: SessionProvider
    credentials: CredentialStore | None
    now: Any

    def factory_for(self, athlete_id: str) -> SessionFactory:
        """A zero-arg session factory over the provider seam, subject-bound (SEAM-R11)."""
        return lambda: self.sessions.session(subject=athlete_id)


async def narrow_incremental(
    ctx: RunContext, athlete_id: str, target: _ConnectionTarget, window: SyncWindow
) -> tuple[SyncWindow, _dt.datetime | None]:
    """Narrow an incremental window forward of the watermark floor (ADP-R6/SYN-R1).

    Incremental mode pulls only-new since the per-source watermark: the window
    floor moves to the LEAST-advanced per-``gbo_type`` cursor (never backward, and
    never past a lagging scope's un-fetched range). Returns the narrowed window
    plus the floor instant for discover-side ref skipping (ADP-R6).
    """
    async with ctx.sessions.session(subject=athlete_id) as session:
        floor_dt = await watermark_floor(
            session, _uid(athlete_id), _uid(target.source_descriptor_id)
        )
    if floor_dt is None:
        return window, None
    floor_date = floor_dt.date().isoformat()
    oldest = window.oldest if floor_date <= window.oldest else floor_date
    return SyncWindow(oldest=oldest, newest=window.newest), floor_dt


def _fetch_inputs(
    target: _ConnectionTarget, fetched_at: _dt.datetime, sync_run_id: str
) -> tuple[FetchContext, SourceDescriptorRef]:
    """The deterministic mapping inputs for one source's run (MAP-R1 purity seam)."""
    return (
        FetchContext(
            ingest_run_id=sync_run_id, fetched_at=fetched_at, connection_id=target.connection_id
        ),
        SourceDescriptorRef(
            source_descriptor_id=target.source_descriptor_id,
            source_key=target.source_key,
            kind=target.kind,
        ),
    )


async def discover_batch(
    ctx: RunContext,
    adapter: DiscoverFetch,
    target: _ConnectionTarget,
    window: SyncWindow,
    fetched_at: _dt.datetime,
    sync_run_id: str,
    since_watermark: _dt.datetime | None,
) -> DiscoverOutput:
    """Drive the five-phase pipeline for one source (ADP-R4..R8)."""
    fetch_ctx, ref = _fetch_inputs(target, fetched_at, sync_run_id)
    return await run_discover_pipeline(
        adapter,
        window,
        ref,
        fetch_ctx,
        api_key=resolve_api_key(ctx.credentials, target),
        athlete_native_id=target.athlete_native_id,
        since_watermark=since_watermark,
    )


async def fetch_and_map(
    ctx: RunContext,
    adapter: AdapterFetch,
    target: _ConnectionTarget,
    window: SyncWindow,
    fetched_at: _dt.datetime,
    sync_run_id: str,
) -> tuple[MappedBatch, PhaseStats]:
    """Legacy window-fetch seam: fetch ASBOs, then pure-map each in ISOLATION.

    Kept for adapters that expose only ``fetch`` (ING-GAP-R5/ING-UPS-R3 isolation
    still holds); the shipped direct-API adapter implements the full
    :class:`DiscoverFetch` contract instead.
    """
    stats = PhaseStats()
    api_key = resolve_api_key(ctx.credentials, target)
    fetch_started = _time.perf_counter()
    asbos = await adapter.fetch(
        api_key=api_key, athlete_native_id=target.athlete_native_id, window=window
    )
    stats.fetch_ms = (_time.perf_counter() - fetch_started) * 1000.0
    fetch_ctx, ref = _fetch_inputs(target, fetched_at, sync_run_id)
    map_started = _time.perf_counter()
    batch = map_records_isolated(adapter, asbos, ref, fetch_ctx, source_key=target.source_key)
    stats.map_ms = (_time.perf_counter() - map_started) * 1000.0
    stats.records_fetched = len(batch.candidates) + len(batch.failed)
    return batch, stats


async def land(
    ctx: RunContext,
    athlete_id: str,
    target: _ConnectionTarget,
    batch: MappedBatch,
    window: SyncWindow,
    sync_run_id: str,
    fetched_at: _dt.datetime,
    original_files: list[OriginalFile],
    *,
    adapter: SourceAdapter,
    discover: DiscoverOutput | None,
    stats: PhaseStats,
) -> SourceSyncResult:
    """Land the batch in ONE transaction with the cursor + gap bookkeeping (ING-UPS-R2).

    The upsert, the watermark advance (SYN-R3), the self-heal of transient gaps the
    synced range covers (ING-GAP-R4), the per-record/per-ref range-precise gaps
    (ING-GAP-R5), and the tier-1 original capture (ING-R8/FIL-R1) ALL commit in the
    SAME transaction. The engine REFUSES an upsert of any GBO type the adapter did
    not declare (ADP-R3): the typed refusal degrades the source with a TERMINAL
    ``schema_mismatch`` gap — never a silent drop.
    """
    gap_count = len(batch.failed) + (
        len(discover.fetch_failed) + (discover.incomplete_cursor is not None) if discover else 0
    )
    if not batch.candidates and not gap_count:
        result = SourceSyncResult.ok(target, candidates_mapped=0)
        emit_run_trace(target.source_key, result.outcome.value, stats)
        return result
    upsert_started = _time.perf_counter()
    try:
        outcome = await _ingest_batch(
            ctx,
            athlete_id,
            target,
            batch,
            window,
            sync_run_id,
            fetched_at,
            original_files,
            adapter=adapter,
            discover=discover,
        )
    except UndeclaredGboTypeError as exc:  # ADP-R3: typed REFUSAL, nothing written
        return await _refuse_undeclared(ctx, athlete_id, target, window, exc, fetched_at, stats)
    except Exception as exc:  # rolled back by the session ctx; degrade + gap (ING-R3)
        _log.warning(
            "sync.ingest_degraded",
            source_key=target.source_key,
            connection_id=target.connection_id,
            error_type=type(exc).__name__,
        )
        result = await degrade_with_gap(
            ctx.factory_for(athlete_id),
            athlete_id,
            target,
            window,
            exc,
            seen_at=fetched_at,
            detail="writing the canonical batch failed",
        )
        emit_run_trace(target.source_key, result.outcome.value, stats, gaps_opened=1)
        return result
    stats.upsert_ms = (_time.perf_counter() - upsert_started) * 1000.0
    return _finish_landed(ctx, target, batch, outcome, stats, gap_count, discover)


async def _ingest_batch(
    ctx: RunContext,
    athlete_id: str,
    target: _ConnectionTarget,
    batch: MappedBatch,
    window: SyncWindow,
    sync_run_id: str,
    fetched_at: _dt.datetime,
    original_files: list[OriginalFile],
    *,
    adapter: SourceAdapter,
    discover: DiscoverOutput | None,
) -> IngestResult:
    """The single landing transaction: upsert + watermark + every typed gap (ING-UPS-R2)."""
    synced = synced_range(window, ctx.now())
    run_uuid: uuid.UUID = _uid(sync_run_id)
    async with ctx.sessions.session(subject=athlete_id) as session:
        outcome = await IngestService(session).ingest(
            athlete_id,
            target.source_descriptor_id,
            batch.candidates,
            connection_id=target.connection_id,
            ingest_run_id=run_uuid,
            original_files=original_files or None,
            synced_range=synced,
            declared_gbo_types=adapter.capability.supported_gbo_types,
            source_key=target.source_key,
        )
        await open_record_gaps(
            session,
            _uid(athlete_id),
            _uid(target.source_descriptor_id),
            batch.failed,
            ingest_run_id=run_uuid,
            seen_at=fetched_at,
        )
        await open_discover_gaps(
            session,
            _uid(athlete_id),
            _uid(target.source_descriptor_id),
            discover,
            window=window,
            ingest_run_id=run_uuid,
            seen_at=fetched_at,
        )
    return outcome


async def _refuse_undeclared(
    ctx: RunContext,
    athlete_id: str,
    target: _ConnectionTarget,
    window: SyncWindow,
    exc: UndeclaredGboTypeError,
    fetched_at: _dt.datetime,
    stats: PhaseStats,
) -> SourceSyncResult:
    """ADP-R3 refusal path: TERMINAL typed gap + degraded result, nothing landed.

    An adapter emitting an undeclared GBO type is a contract defect needing an
    operator fix — re-running deterministically re-fails, so the gap is terminal
    (``transient=False``) and the refusal is never silently dropped or retried.
    """
    _log.warning(
        "sync.undeclared_gbo_type",
        source_key=target.source_key,
        connection_id=target.connection_id,
        error_type=type(exc).__name__,
    )
    covered = synced_range(window, fetched_at)
    async with ctx.factory_for(athlete_id)() as session:
        await open_gap(
            session,
            _uid(athlete_id),
            _uid(target.source_descriptor_id),
            GboType.ACTIVITY,
            reason=GapReason.SCHEMA_MISMATCH,
            seen_at=fetched_at,
            transient=False,
            range_start_at=covered.oldest,
            range_end_at=covered.newest,
        )
    result = degraded(target, "adapter emitted an undeclared record type")
    emit_run_trace(target.source_key, result.outcome.value, stats, gaps_opened=1)
    return result


def _finish_landed(
    ctx: RunContext,
    target: _ConnectionTarget,
    batch: MappedBatch,
    outcome: IngestResult,
    stats: PhaseStats,
    gap_count: int,
    discover: DiscoverOutput | None,
) -> SourceSyncResult:
    """Build the landed result + emit the per-run trace/metrics (ING-OBS-R1/R2)."""
    partial = gap_count > 0
    result = SourceSyncResult(
        source_key=target.source_key,
        connection_id=target.connection_id,
        outcome=SyncOutcome.DEGRADED if partial else SyncOutcome.OK,
        candidates_mapped=len(batch.candidates),
        activities_written=len(outcome.activities_written),
        wellness_written=outcome.wellness_written,
        detail="some records could not be acquired or mapped" if partial else None,
    )
    emit_run_trace(
        target.source_key,
        result.outcome.value,
        stats,
        candidates_mapped=len(batch.candidates),
        records_failed=len(batch.failed) + (len(discover.fetch_failed) if discover else 0),
        activities_written=result.activities_written,
        wellness_written=result.wellness_written,
        gaps_opened=gap_count,
        gaps_closed=outcome.gaps_closed,
        watermarks_advanced=outcome.watermarks_advanced,
    )
    _observe_freshness(ctx, target.source_key, batch)
    return result


def _observe_freshness(ctx: RunContext, source_key: str, batch: MappedBatch) -> None:
    """Record the per-source freshness lag metric (ING-OBS-R2): now minus newest ingested."""
    newest = None
    for cand in batch.candidates:
        if cand.observed_at is not None and (newest is None or cand.observed_at > newest):
            newest = cand.observed_at
    if newest is None:
        return
    lag = (ctx.now() - newest).total_seconds()
    _metrics.get_registry().observe(
        _metrics.INGEST_FRESHNESS_LAG, max(lag, 0.0), labels={"source_key": source_key}
    )


__all__ = [
    "AdapterFetch",
    "OriginalArtifactSource",
    "RunContext",
    "discover_batch",
    "fetch_and_map",
    "land",
    "narrow_incremental",
]
