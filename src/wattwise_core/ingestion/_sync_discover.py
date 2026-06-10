"""Five-phase discover pipeline + per-run sync observability (ADP-R4..R7, ING-OBS-R1/R2).

The orchestrator drives a fetch-capable adapter through the spec's full
authorize → discover → fetch → map pipeline here (upsert stays in the orchestrator's
landing transaction). Discovery is cursor-paginated (ADP-R7): a mid-pagination
failure stops discovery at the broken cursor — the already-discovered refs still
fetch/map/land, and the caller opens a typed ``discovery_incomplete`` gap covering
exactly the un-discovered remainder (ING-GAP-R5). A per-ref fetch failure isolates to
that ref (a token-precise typed gap), never the batch. Watermark honoring (ADP-R6) is
belt-and-braces: the adapter filters by the passed ``since_watermark`` AND the engine
re-filters refs whose last-modified hint proves them current.

:func:`emit_run_trace` realizes ING-OBS-R1: every sync run emits ONE structured trace
event with per-phase timing and record counts through the redacted log stream, and
records the ING-OBS-R2 operational metrics (per-source run outcome, per-phase
latency, record throughput). Untrusted source content is never logged — the trace
carries only a count-style flag (ING-OBS-R3).
"""

from __future__ import annotations

import datetime as _dt
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.enums import GapReason, GboType, Severity
from wattwise_core.ingestion._sync_records import MappedBatch, map_records_isolated
from wattwise_core.ingestion.base import AuthError, FetchContext, SourceAdapter, SourceDescriptorRef
from wattwise_core.ingestion.capability import AuthContext, DiscoveryPage, DiscoveryRef
from wattwise_core.ingestion.watermark import open_gap
from wattwise_core.observability import metrics as _metrics
from wattwise_core.observability.logging import get_logger

_log = get_logger(__name__)


@runtime_checkable
class DiscoverFetch(SourceAdapter, Protocol):
    """A fetch-capable adapter implementing the full §3.2 phase contract (ADP-R4/R5/R7).

    ``ensure_authorized`` returns a validated :class:`AuthContext` or raises the typed
    ``AuthError`` taxonomy (ADP-R4); ``discover`` yields cursor-paginated lightweight
    refs honoring the watermark (ADP-R5/R6/R7); ``fetch_ref`` returns ONE validated
    typed ASBO (ADP-R8/CLI-R2). The orchestrator prefers this contract over the legacy
    window-fetch seam and drives it purely from the capability descriptor (ADP-R2).
    """

    async def ensure_authorized(
        self, *, api_key: str | None, athlete_native_id: str | None
    ) -> AuthContext:
        """Validate the credential context or raise a typed ``AuthError`` (ADP-R4)."""
        ...

    async def discover(
        self,
        ctx: AuthContext,
        window: Any,
        *,
        cursor: str | None = None,
        since_watermark: _dt.datetime | None = None,
    ) -> DiscoveryPage:
        """One page of lightweight discovery refs + ``next_cursor`` (ADP-R5/R6/R7)."""
        ...

    async def fetch_ref(self, ctx: AuthContext, ref: DiscoveryRef) -> Any:
        """Fetch one discovered record as a validated typed ASBO (ADP-R8)."""
        ...


@dataclass(slots=True)
class PhaseStats:
    """Per-phase timings + record counts for one source's sync run (ING-OBS-R1)."""

    authorize_ms: float = 0.0
    discover_ms: float = 0.0
    fetch_ms: float = 0.0
    map_ms: float = 0.0
    upsert_ms: float = 0.0
    refs_discovered: int = 0
    refs_skipped: int = 0
    records_fetched: int = 0


@dataclass(slots=True)
class DiscoverOutput:
    """What the five-phase pipeline produced for the landing transaction.

    ``fetch_failed`` refs become token-precise typed gaps; a non-``None``
    ``incomplete_cursor`` means pagination broke there and the caller opens a
    ``discovery_incomplete`` gap covering exactly the un-discovered remainder
    (ADP-R7 / ING-GAP-R5). ``last_discovered_at`` bounds that gap's range start.
    """

    batch: MappedBatch
    fetch_failed: list[DiscoveryRef] = field(default_factory=list)
    incomplete_cursor: str | None = None
    last_discovered_at: _dt.datetime | None = None
    stats: PhaseStats = field(default_factory=PhaseStats)


async def run_discover_pipeline(
    adapter: DiscoverFetch,
    window: Any,
    ref: SourceDescriptorRef,
    ctx: FetchContext,
    *,
    api_key: str | None,
    athlete_native_id: str | None,
    since_watermark: _dt.datetime | None,
) -> DiscoverOutput:
    """Drive authorize → discover → fetch → map for one source (ADP-R4..R8).

    An ``AuthError`` propagates (the orchestrator's reauth path owns it, AUT-R4). A
    discovery page failure stops pagination at that cursor; a per-ref fetch failure
    isolates to that ref; mapping is per-record isolated (ING-GAP-R5 / ING-UPS-R3).
    """
    out = DiscoverOutput(batch=MappedBatch(candidates=[], failed=[]))
    started = time.perf_counter()
    auth = await adapter.ensure_authorized(api_key=api_key, athlete_native_id=athlete_native_id)
    out.stats.authorize_ms = (time.perf_counter() - started) * 1000.0
    refs = await _discover_all(adapter, auth, window, since_watermark, out)
    asbos = await _fetch_refs(adapter, auth, refs, out)
    mapping_started = time.perf_counter()
    out.batch = map_records_isolated(adapter, asbos, ref, ctx, source_key=ref.source_key)
    out.stats.map_ms = (time.perf_counter() - mapping_started) * 1000.0
    return out


async def _discover_all(
    adapter: DiscoverFetch,
    auth: AuthContext,
    window: Any,
    since_watermark: _dt.datetime | None,
    out: DiscoverOutput,
) -> list[DiscoveryRef]:
    """Paginate discovery to completion or the first broken cursor (ADP-R7).

    Engine-side belt-and-braces watermark skip (ADP-R6): a ref whose last-modified
    hint proves it current per ``since_watermark`` is dropped even if the adapter
    yielded it.
    """
    refs: list[DiscoveryRef] = []
    cursor: str | None = None
    started = time.perf_counter()
    while True:
        try:
            page = await adapter.discover(
                auth, window, cursor=cursor, since_watermark=since_watermark
            )
        except AuthError:
            raise
        except Exception:  # partial discovery -> typed gap from exactly this cursor
            out.incomplete_cursor = cursor or ""
            break
        for item in page.refs:
            if (
                since_watermark is not None
                and item.last_modified is not None
                and item.last_modified <= since_watermark
            ):
                out.stats.refs_skipped += 1
                continue
            refs.append(item)
            if item.last_modified is not None and (
                out.last_discovered_at is None or item.last_modified > out.last_discovered_at
            ):
                out.last_discovered_at = item.last_modified
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    out.stats.discover_ms = (time.perf_counter() - started) * 1000.0
    out.stats.refs_discovered = len(refs)
    return refs


async def _fetch_refs(
    adapter: DiscoverFetch,
    auth: AuthContext,
    refs: list[DiscoveryRef],
    out: DiscoverOutput,
) -> list[Any]:
    """Fetch each discovered ref in ISOLATION; a failed ref gap-marks only itself."""
    asbos: list[Any] = []
    started = time.perf_counter()
    for item in refs:
        try:
            asbos.append(await adapter.fetch_ref(auth, item))
        except AuthError:
            raise
        except Exception:  # per-ref isolation: token-precise gap, batch continues
            out.fetch_failed.append(item)
    out.stats.fetch_ms = (time.perf_counter() - started) * 1000.0
    out.stats.records_fetched = len(asbos)
    return asbos


async def open_discover_gaps(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    out: DiscoverOutput | None,
    *,
    window: Any,
    ingest_run_id: uuid.UUID,
    seen_at: _dt.datetime,
) -> int:
    """Open the discover-phase typed gaps inside the landing transaction (ING-GAP-R5).

    One TRANSIENT token-precise ``fetch_failed`` gap per un-fetchable ref (a later
    successful re-fetch of that record self-heals it), plus ONE TRANSIENT
    ``discovery_incomplete`` gap when pagination broke — covering exactly the
    un-discovered remainder: from the last discovered instant (or the window start)
    to the window end, with the broken cursor as the resume token (ADP-R7).
    """
    if out is None:
        return 0
    opened = 0
    for ref in out.fetch_failed:
        await open_gap(
            session,
            athlete_id,
            source_descriptor_id,
            ref.gbo_type,
            reason=GapReason.FETCH_FAILED,
            seen_at=seen_at,
            severity=Severity.WARNING,
            transient=True,
            range_start_token=ref.source_native_id,
            range_end_token=ref.source_native_id,
            ingest_run_id=ingest_run_id,
        )
        opened += 1
    if out.incomplete_cursor is not None:
        start = out.last_discovered_at or _window_start(window)
        await open_gap(
            session,
            athlete_id,
            source_descriptor_id,
            GboType.ACTIVITY,
            reason=GapReason.DISCOVERY_INCOMPLETE,
            seen_at=seen_at,
            severity=Severity.WARNING,
            transient=True,
            range_start_at=start,
            range_end_at=_window_end(window),
            range_start_token=out.incomplete_cursor or None,
            ingest_run_id=ingest_run_id,
        )
        opened += 1
    return opened


def _window_start(window: Any) -> _dt.datetime:
    """The window's inclusive start instant (UTC midnight of its oldest date)."""
    return _dt.datetime.fromisoformat(window.oldest).replace(tzinfo=_dt.UTC)


def _window_end(window: Any) -> _dt.datetime:
    """The window's inclusive end instant (UTC end-of-day of its newest date)."""
    end = _dt.datetime.fromisoformat(window.newest).replace(tzinfo=_dt.UTC)
    return end + _dt.timedelta(days=1) - _dt.timedelta(seconds=1)


def emit_run_trace(
    source_key: str,
    outcome: str,
    stats: PhaseStats,
    *,
    candidates_mapped: int = 0,
    records_failed: int = 0,
    activities_written: int = 0,
    wellness_written: int = 0,
    gaps_opened: int = 0,
    gaps_closed: int = 0,
    watermarks_advanced: int = 0,
) -> None:
    """Emit the per-run sync trace + operational metrics (ING-OBS-R1/R2, ING-OBS-R3).

    One structured event through the redacted log stream with per-phase timing and
    record counts — never source content, secrets, or PII (the allowlist drops
    anything else) — plus the per-source run/latency/throughput metrics.
    """
    _log.info(
        "sync.run_trace",
        source_key=source_key,
        outcome=outcome,
        authorize_ms=round(stats.authorize_ms, 3),
        discover_ms=round(stats.discover_ms, 3),
        fetch_ms=round(stats.fetch_ms, 3),
        map_ms=round(stats.map_ms, 3),
        upsert_ms=round(stats.upsert_ms, 3),
        refs_discovered=stats.refs_discovered,
        refs_skipped=stats.refs_skipped,
        records_fetched=stats.records_fetched,
        candidates_mapped=candidates_mapped,
        records_failed=records_failed,
        activities_written=activities_written,
        wellness_written=wellness_written,
        gaps_opened=gaps_opened,
        gaps_closed=gaps_closed,
        watermarks_advanced=watermarks_advanced,
    )
    _emit_run_metrics(
        source_key,
        outcome,
        stats,
        candidates_mapped=candidates_mapped,
        records_failed=records_failed,
        upserted=activities_written + wellness_written,
    )


def _emit_run_metrics(
    source_key: str,
    outcome: str,
    stats: PhaseStats,
    *,
    candidates_mapped: int,
    records_failed: int,
    upserted: int,
) -> None:
    """Record the per-source run/latency/throughput metrics for one sync run (ING-OBS-R2)."""
    registry = _metrics.get_registry()
    registry.increment(
        _metrics.INGEST_SOURCE_RUNS, labels={"source_key": source_key, "outcome": outcome}
    )
    for phase, value in (
        ("authorize", stats.authorize_ms),
        ("discover", stats.discover_ms),
        ("fetch", stats.fetch_ms),
        ("map", stats.map_ms),
        ("upsert", stats.upsert_ms),
    ):
        registry.observe(
            _metrics.INGEST_PHASE_LATENCY,
            value / 1000.0,
            labels={"source_key": source_key, "phase": phase},
        )
    for stage, count in (
        ("discovered", stats.refs_discovered),
        ("fetched", stats.records_fetched),
        ("mapped", candidates_mapped),
        ("failed", records_failed),
        ("upserted", upserted),
    ):
        if count:
            registry.increment(
                _metrics.INGEST_RECORDS,
                amount=float(count),
                labels={"source_key": source_key, "stage": stage},
            )


__all__ = [
    "DiscoverFetch",
    "DiscoverOutput",
    "PhaseStats",
    "emit_run_trace",
    "open_discover_gaps",
    "run_discover_pipeline",
]
