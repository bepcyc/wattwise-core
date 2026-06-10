"""Per-source sync execution: authorize -> discover -> fetch -> map -> land (doc 30).

The single-source leg :class:`wattwise_core.ingestion.sync.SyncOrchestrator` runs for
each connection, factored to a sibling module so the orchestrator class stays within the
QUAL-R9 size ceilings without changing behavior: adapter selection through the injected
registry (ARCH-R2, source-blind), incremental window narrowing (ADP-R6), the five-phase
or legacy fetch with auth-break -> ``reauth_required`` (AUT-R4) and failure -> typed-gap
degradation (CON-R3 / ARCH-R9 / ING-R3), and the single landing transaction (UPS-R6).
A failure NEVER raises past the source: every path returns a typed
:class:`~wattwise_core.ingestion._sync_targets.SourceSyncResult`.
"""

from __future__ import annotations

import datetime as _dt

from wattwise_core.ingestion._sync_discover import (
    DiscoverFetch,
    DiscoverOutput,
    PhaseStats,
    emit_run_trace,
)
from wattwise_core.ingestion._sync_records import MappedBatch
from wattwise_core.ingestion._sync_run import (
    AdapterFetch,
    OriginalArtifactSource,
    RunContext,
    discover_batch,
    fetch_and_map,
    land,
    narrow_incremental,
)
from wattwise_core.ingestion._sync_targets import (
    SourceSyncResult,
    SyncWindow,
    _ConnectionTarget,
    degrade_with_gap,
    degraded,
    handle_reauth,
    skipped,
)
from wattwise_core.ingestion.base import AuthError
from wattwise_core.ingestion.registry import AdapterRegistry, UnknownSourceError


async def sync_one(
    ctx: RunContext,
    registry: AdapterRegistry,
    athlete_id: str,
    target: _ConnectionTarget,
    window: SyncWindow,
    fetched_at: _dt.datetime,
    sync_run_id: str,
    explicit_window: bool,
) -> SourceSyncResult:
    """Authorize -> discover -> fetch -> map -> land ONE source, never raising past it.

    Prefers the full five-phase :class:`DiscoverFetch` contract (ADP-R4/R5/R7); falls
    back to the legacy window-fetch seam for adapters that expose only ``fetch``. A
    failure degrades WITH a persisted typed gap (ING-R3), never a crash (CON-R3 /
    ARCH-R9) and never a swallowed-into-a-string outcome.
    """
    try:
        adapter = registry.get(target.source_key)
    except UnknownSourceError:
        return degraded(target, "source is not installed")
    if not isinstance(adapter, DiscoverFetch | AdapterFetch):
        # No direct-API fetch seam (e.g. connectionless file upload): nothing to
        # pull on demand. Skipped, not degraded — this is the expected shape.
        return skipped(target, "source has no on-demand fetch")
    if not explicit_window:
        window, since = await narrow_incremental(ctx, athlete_id, target, window)
    else:
        since = None
    fetched = await _fetch_phase(
        ctx, adapter, athlete_id, target, window, fetched_at, sync_run_id, since
    )
    if isinstance(fetched, SourceSyncResult):
        return fetched
    batch, stats, out = fetched
    originals = adapter.original_files() if isinstance(adapter, OriginalArtifactSource) else []
    return await land(
        ctx,
        athlete_id,
        target,
        batch,
        window,
        sync_run_id,
        fetched_at,
        originals,
        adapter=adapter,
        discover=out,
        stats=stats,
    )


async def _fetch_phase(
    ctx: RunContext,
    adapter: DiscoverFetch | AdapterFetch,
    athlete_id: str,
    target: _ConnectionTarget,
    window: SyncWindow,
    fetched_at: _dt.datetime,
    sync_run_id: str,
    since: _dt.datetime | None,
) -> tuple[MappedBatch, PhaseStats, DiscoverOutput | None] | SourceSyncResult:
    """Run the discover/fetch/map phases; a failure returns the TERMINAL typed result.

    An :class:`AuthError` flips the connection to ``reauth_required`` and stops the
    source (AUT-R4); any other failure degrades with a range-precise typed gap
    (ARCH-R9 / ING-R3). Both terminal paths emit the per-run trace (ING-OBS-R1/R2).
    """
    factory = ctx.factory_for(athlete_id)
    stats = PhaseStats()
    out: DiscoverOutput | None = None
    try:
        if isinstance(adapter, DiscoverFetch):
            out = await discover_batch(ctx, adapter, target, window, fetched_at, sync_run_id, since)
            return out.batch, out.stats, out
        batch, stats = await fetch_and_map(ctx, adapter, target, window, fetched_at, sync_run_id)
    except AuthError as exc:  # credential revoked/expired -> reauth, stop the source (AUT-R4)
        result = await handle_reauth(factory, athlete_id, target, exc, seen_at=fetched_at)
        emit_run_trace(target.source_key, result.outcome.value, stats, gaps_opened=1)
        return result
    except Exception as exc:  # isolate the failure; degrade + typed gap (ARCH-R9/ING-R3)
        result = await degrade_with_gap(
            factory,
            athlete_id,
            target,
            window,
            exc,
            seen_at=fetched_at,
            detail="source fetch or mapping failed",
        )
        emit_run_trace(target.source_key, result.outcome.value, stats, gaps_opened=1)
        return result
    return batch, stats, out


__all__ = ["sync_one"]
