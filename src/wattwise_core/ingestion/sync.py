"""On-demand sync orchestrator — connection -> fetch -> map -> canonical (doc 30).

Owning requirements: SYN-R* (on-demand sync, no scheduler; chunked resumable
backfill SYN-R5/R6), ADP-R* (pluggable adapters driven through the five-phase
authorize -> discover -> fetch -> map -> upsert contract, with a legacy window-fetch
seam), ROAD-R6 (one adapter, zero consumer change), UPS-R6 (one transaction per
batch), CON-R3 (graceful partial/degraded coverage on a source error), ARCH-R9 (a
failing source never crashes the run), AUT-R2/SEC-R7 (the credential is resolved
from an opaque ``credential_ref`` only at the point of use), ARCH-R2 (no source-name
branching outside adapters).

:meth:`SyncOrchestrator.run` flow, for the *server-derived* athlete (AUTH-R3): select
the adapter via the registry (never importing a named adapter — ARCH-R2); on
incremental mode narrow the window forward of the watermark so already-current ranges
are skipped (ADP-R6); resolve the opaque ``credential_ref`` only at the point of use
(SEC-R7); drive the five-phase pipeline (or the legacy fetch) with per-record map
ISOLATION (MAP-R1) so a single un-mappable record becomes a range-precise gap while
the good records still commit (ING-GAP-R5 / ING-UPS-R3); land the batch through
:class:`IngestService` in ONE transaction that ALSO advances the watermark (SYN-R3),
opens every typed gap, and self-heals covered transient gaps (ING-GAP-R4) — store,
cursor, and gap state stay mutually consistent (ING-UPS-R2). The engine REFUSES an
upsert of any GBO type the adapter did not declare (ADP-R3). A source that errors
degrades WITH a persisted typed gap (ING-R3) and never crashes the others
(CON-R3 / ARCH-R9). An auth break flips the Connection to ``reauth_required``, emits
a typed §7 gap, and STOPS scheduling that source until re-auth (AUT-R4). Every run
emits the per-phase trace + operational metrics (ING-OBS-R1/R2).

Layer: L3 ingestion/sync — the ONLY writer to the store (ARCH-R3, via
:class:`IngestService`); imports NO named adapter module (ARCH-R2) and NO L6 edge.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from sqlalchemy import select

from wattwise_core.domain.enums import ConnectionStatus
from wattwise_core.ingestion._sync_discover import (
    DiscoverFetch,
    DiscoverOutput,
    PhaseStats,
    emit_run_trace,
)
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
    SessionFactory,
    SourceSyncResult,
    SyncOutcome,
    SyncRun,
    SyncWindow,
    _ConnectionTarget,
    _uid,
    degrade_with_gap,
    degraded,
    handle_reauth,
    skipped,
)
from wattwise_core.ingestion.backfill import (
    advance_backfill_cursor,
    chunk_windows,
    read_backfill_cursor,
)
from wattwise_core.ingestion.base import AuthError
from wattwise_core.ingestion.registry import AdapterRegistry, UnknownSourceError
from wattwise_core.persistence.models import Connection, SourceDescriptor
from wattwise_core.persistence.types import utcnow, uuid7
from wattwise_core.seams import SessionProvider
from wattwise_core.security.credentials import CredentialStore

# Default lookback for a connection sync when the caller gives no explicit window.
_DEFAULT_LOOKBACK = _dt.timedelta(days=42)


class SyncOrchestrator:
    """Drives on-demand sync from authorized connections into the canonical store.

    Source-blind: selects adapters through the injected :class:`AdapterRegistry` (by
    ``source_key``) and resolves credentials through the :class:`CredentialStore`,
    never importing or branching on a named source (ARCH-R2). ``now`` is injectable so
    the built fetch context is deterministic in tests. Every canonical-store open
    flows through the ONE engine-owned :class:`SessionProvider` seam (SEAM-R11 / ARCH-R31),
    keyed on the server-derived athlete ``subject`` (ARCH-R16), never around it. The
    per-source execution lives in :mod:`wattwise_core.ingestion._sync_run`.
    """

    def __init__(
        self,
        sessions: SessionProvider,
        *,
        registry: AdapterRegistry,
        credential_store: CredentialStore | None = None,
        now: Any = None,
    ) -> None:
        self._sessions = sessions
        self._registry = registry
        self._credentials = credential_store
        self._now = now or utcnow
        self._ctx = RunContext(sessions=sessions, credentials=credential_store, now=self._now)

    def _factory_for(self, athlete_id: str) -> SessionFactory:
        """A zero-arg :class:`SessionFactory` over the provider seam, subject-bound (SEAM-R11)."""
        return self._ctx.factory_for(athlete_id)

    async def run(
        self,
        athlete_id: str,
        *,
        connection_id: str | None = None,
        source: str | None = None,
        window: SyncWindow | None = None,
    ) -> SyncRun:
        """Sync one athlete's connections on demand (SYN-R*; no scheduler).

        ``connection_id`` syncs that connection; ``source`` syncs the connection for
        that ``source_key``; neither syncs every authorized connection. ``athlete_id``
        MUST be the server-derived identity (AUTH-R3), never a model/client value. Each
        source runs in its own transaction so one failure degrades, never crashes the
        others (CON-R3 / ARCH-R9).
        """
        run = SyncRun(athlete_id=athlete_id, sync_run_id=str(uuid7()), started_at=self._now())
        fetched_at = run.started_at
        win = window or SyncWindow(
            oldest=(fetched_at - _DEFAULT_LOOKBACK).date().isoformat(),
            newest=fetched_at.date().isoformat(),
        )
        explicit_window = window is not None
        connections = await self._select_connections(athlete_id, connection_id, source)
        for conn in connections:
            result = await self._sync_one(
                athlete_id, conn, win, fetched_at, run.sync_run_id, explicit_window
            )
            run.results.append(result)
        return run

    async def _select_connections(
        self, athlete_id: str, connection_id: str | None, source: str | None
    ) -> list[_ConnectionTarget]:
        """Resolve the connections to sync into source-agnostic targets (read-only).

        Excludes a Connection in an athlete-actionable terminal state — ``reauth_required``
        (AUT-R4: a revoked credential re-hits the same 401/403, never self-heals) and
        ``disconnected`` (ONB-R5) — gating on the PERSISTED status until re-auth.
        """
        excluded = (ConnectionStatus.REAUTH_REQUIRED, ConnectionStatus.DISCONNECTED)
        async with self._sessions.session(subject=athlete_id) as session:
            stmt = (
                select(Connection, SourceDescriptor)
                .join(
                    SourceDescriptor,
                    Connection.source_descriptor_id == SourceDescriptor.source_descriptor_id,
                )
                .where(Connection.athlete_id == _uid(athlete_id))
                .where(Connection.status.notin_(excluded))
            )
            if connection_id is not None:
                stmt = stmt.where(Connection.connection_id == _uid(connection_id))
            if source is not None:
                stmt = stmt.where(SourceDescriptor.source_key == source)
            rows = (await session.execute(stmt)).all()
            return [_ConnectionTarget.of(conn, desc) for conn, desc in rows]

    async def backfill(
        self,
        athlete_id: str,
        *,
        window: SyncWindow,
        chunk_days: int,
        connection_id: str | None = None,
        source: str | None = None,
    ) -> SyncRun:
        """On-demand historical backfill in bounded oldest-first windows (SYN-R5/R6).

        The requested range is chunked into ``chunk_days`` windows, OLDEST-FIRST, each
        landed + committed independently with the per-window watermark advance riding
        the landing transaction (SYN-R5). A persisted, RANGE-SCOPED per-source backfill
        cursor advances after each committed window, so the run is CANCELLABLE between
        windows and RESUMABLE without re-downloading already-committed windows
        (SYN-R6): a re-run of the same range skips every window at or before the
        cursor, and a window failure stops the source's loop with the cursor still at
        the last committed window. ``chunk_days`` is configuration
        (``ingestion.backfill_window_days``, CFG-R1a) supplied by the caller. The
        automatic pacing/prioritisation around this mechanism is commercial
        orchestration (COMM-R19), not shipped here.
        """
        run = SyncRun(athlete_id=athlete_id, sync_run_id=str(uuid7()), started_at=self._now())
        for target in await self._select_connections(athlete_id, connection_id, source):
            run.results.extend(
                await self._backfill_one(athlete_id, target, window, chunk_days, run.sync_run_id)
            )
        return run

    async def _backfill_one(
        self,
        athlete_id: str,
        target: _ConnectionTarget,
        window: SyncWindow,
        chunk_days: int,
        sync_run_id: str,
    ) -> list[SourceSyncResult]:
        """Backfill ONE source oldest-first, skipping windows the cursor already committed."""
        factory = self._factory_for(athlete_id)
        athlete = _uid(athlete_id)
        descriptor = _uid(target.source_descriptor_id)
        cursor = await read_backfill_cursor(
            factory, athlete, descriptor, range_oldest=window.oldest
        )
        results: list[SourceSyncResult] = []
        for win in chunk_windows(window, chunk_days):
            if cursor is not None and _dt.date.fromisoformat(win.newest) <= cursor:
                continue  # SYN-R6: committed window — never re-downloaded
            result = await self._sync_one(
                athlete_id, target, win, self._now(), sync_run_id, explicit_window=True
            )
            results.append(result)
            if result.outcome is not SyncOutcome.OK:
                break  # resume point = last committed window; re-run continues here
            await advance_backfill_cursor(
                factory, athlete, descriptor, range_oldest=window.oldest,
                through=_dt.date.fromisoformat(win.newest), ingest_run_id=_uid(sync_run_id),
            )
        return results

    async def _sync_one(
        self,
        athlete_id: str,
        target: _ConnectionTarget,
        window: SyncWindow,
        fetched_at: _dt.datetime,
        sync_run_id: str,
        explicit_window: bool,
    ) -> SourceSyncResult:
        """Authorize -> discover -> fetch -> map -> land ONE source, never raising past it.

        Prefers the full five-phase :class:`DiscoverFetch` contract (ADP-R4/R5/R7);
        falls back to the legacy window-fetch seam for adapters that expose only
        ``fetch``. A failure degrades WITH a persisted typed gap (ING-R3), never a
        crash (CON-R3 / ARCH-R9) and never a swallowed-into-a-string outcome.
        """
        try:
            adapter = self._registry.get(target.source_key)
        except UnknownSourceError:
            return degraded(target, "source is not installed")
        if not isinstance(adapter, DiscoverFetch | AdapterFetch):
            # No direct-API fetch seam (e.g. connectionless file upload): nothing to
            # pull on demand. Skipped, not degraded — this is the expected shape.
            return skipped(target, "source has no on-demand fetch")
        since: _dt.datetime | None = None
        if not explicit_window:
            window, since = await narrow_incremental(self._ctx, athlete_id, target, window)
        stats = PhaseStats()
        out: DiscoverOutput | None = None
        try:
            if isinstance(adapter, DiscoverFetch):
                out = await discover_batch(
                    self._ctx, adapter, target, window, fetched_at, sync_run_id, since
                )
                batch, stats = out.batch, out.stats
            else:
                batch, stats = await fetch_and_map(
                    self._ctx, adapter, target, window, fetched_at, sync_run_id
                )
        except AuthError as exc:  # credential revoked/expired -> reauth, stop the source (AUT-R4)
            result = await handle_reauth(
                self._factory_for(athlete_id), athlete_id, target, exc, seen_at=fetched_at
            )
            emit_run_trace(target.source_key, result.outcome.value, stats, gaps_opened=1)
            return result
        except Exception as exc:  # isolate the failure; degrade + typed gap (ARCH-R9/ING-R3)
            result = await degrade_with_gap(
                self._factory_for(athlete_id), athlete_id, target, window, exc,
                seen_at=fetched_at, detail="source fetch or mapping failed",
            )
            emit_run_trace(target.source_key, result.outcome.value, stats, gaps_opened=1)
            return result
        originals = adapter.original_files() if isinstance(adapter, OriginalArtifactSource) else []
        return await land(
            self._ctx, athlete_id, target, batch, window, sync_run_id, fetched_at, originals,
            adapter=adapter, discover=out, stats=stats,
        )


__all__ = [
    "AdapterFetch",
    "OriginalArtifactSource",
    "SessionFactory",
    "SourceSyncResult",
    "SyncOrchestrator",
    "SyncOutcome",
    "SyncRun",
    "SyncWindow",
]
