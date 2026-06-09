"""On-demand sync orchestrator — connection -> fetch -> map -> canonical (doc 30).

Owning requirements: SYN-R* (on-demand sync, no scheduler), ADP-R* (pluggable
adapters), ROAD-R6 (one adapter, zero consumer change), UPS-R6 (one transaction per
batch), CON-R3 (graceful partial/degraded coverage on a source error), ARCH-R9 (a
failing source never crashes the run), AUT-R2/SEC-R7 (the credential is resolved from
an opaque ``credential_ref`` only at the point of use), ARCH-R2 (no source-name
branching outside adapters).

:meth:`SyncOrchestrator.run` flow, for the *server-derived* athlete (AUTH-R3): select the
adapter via the registry (never importing a named adapter — ARCH-R2); on incremental mode
narrow the window forward of the watermark so already-current ranges are skipped (ADP-R6);
resolve the opaque ``credential_ref`` only at the point of use (SEC-R7); fetch ASBOs
(impure) and pure-map each in ISOLATION (MAP-R1) so a single un-mappable record becomes a
range-precise gap while the good records still commit (ING-GAP-R5 / ING-UPS-R3); land the
batch through :class:`IngestService` in ONE transaction that ALSO advances the watermark
(SYN-R3) and self-heals covered transient gaps (ING-GAP-R4) — store, cursor, and gap state
stay mutually consistent (ING-UPS-R2). A source that errors degrades and never crashes the
others (CON-R3 / ARCH-R9), rolling that source back without fabricating a partial record. The
fetch is driven by the adapter's resilient typed client (exponential backoff + full jitter under
a per-source budget CLI-R6, a client-side token-bucket limiter honouring ``Retry-After`` with
adaptive rate reduction CLI-R10/R11, and a typed ``FetchError`` on schema mismatch CLI-R2); a
401/403 surfaces as an :class:`AuthError`, which flips the Connection to ``reauth_required``,
emits a typed §7 gap, and STOPS scheduling that source until re-auth (AUT-R4) — never swallowed
as a transient degrade.

Layer: L3 ingestion/sync — the ONLY writer to the store (ARCH-R3, via
:class:`IngestService`); imports NO named adapter module (ARCH-R2) and NO L6 edge.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import select

from wattwise_core.domain.enums import ConnectionStatus
from wattwise_core.ingestion._sync_records import (
    MappedBatch,
    incremental_floor_date,
    map_records_isolated,
    open_record_gaps,
)
from wattwise_core.ingestion._sync_targets import (
    SessionFactory,
    SourceSyncResult,
    SyncOutcome,
    SyncRun,
    SyncWindow,
    _ConnectionTarget,
    _uid,
    degraded,
    handle_reauth,
    resolve_api_key,
    skipped,
    synced_range,
)
from wattwise_core.ingestion.base import (
    AuthError,
    FetchContext,
    SourceAdapter,
    SourceDescriptorRef,
)
from wattwise_core.ingestion.ingest import IngestService, OriginalFile
from wattwise_core.ingestion.registry import AdapterRegistry, UnknownSourceError
from wattwise_core.observability.logging import get_logger
from wattwise_core.persistence.models import Connection, SourceDescriptor
from wattwise_core.persistence.types import utcnow, uuid7
from wattwise_core.seams import SessionProvider
from wattwise_core.security.credentials import CredentialStore

_log = get_logger(__name__)


@runtime_checkable
class AdapterFetch(SourceAdapter, Protocol):
    """A :class:`SourceAdapter` that ALSO exposes a direct-API fetch seam (ADP-R8).

    The orchestrator drives fetch polymorphically through this structural shape,
    never by naming a source (ARCH-R2). ``fetch`` is the IMPURE side (network/file
    I/O) kept strictly out of the pure :meth:`SourceAdapter.map`; it returns
    source-shaped objects (ASBOs) the adapter's ``map`` then turns into canonical
    candidates. A ``file_upload`` adapter (connectionless) need not implement this,
    so the orchestrator skips a source that is not :class:`AdapterFetch`.
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
    created (FIL-R1). A direct-API source (e.g. Intervals.icu, which returns JSON, not
    a recording file) does NOT implement this, so the orchestrator captures NO file for
    it — exactly the intended shape (a direct-API observation has no original artifact).
    """

    def original_files(self) -> list[OriginalFile]:
        """The verbatim originals acquired in the last fetch (empty if none)."""
        ...


# Default lookback for a connection sync when the caller gives no explicit window.
_DEFAULT_LOOKBACK = _dt.timedelta(days=42)


class SyncOrchestrator:
    """Drives on-demand sync from authorized connections into the canonical store.

    Source-blind: selects adapters through the injected :class:`AdapterRegistry` (by
    ``source_key``) and resolves credentials through the :class:`CredentialStore`,
    never importing or branching on a named source (ARCH-R2). ``now`` is injectable so
    the built :class:`FetchContext` is deterministic in tests. Every canonical-store open
    flows through the ONE engine-owned :class:`SessionProvider` seam (SEAM-R11 / ARCH-R31),
    keyed on the server-derived athlete ``subject`` (ARCH-R16), never around it.
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

    def _factory_for(self, athlete_id: str) -> SessionFactory:
        """A zero-arg :class:`SessionFactory` over the provider seam, subject-bound (SEAM-R11)."""
        return lambda: self._sessions.session(subject=athlete_id)

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

    async def _sync_one(
        self,
        athlete_id: str,
        target: _ConnectionTarget,
        window: SyncWindow,
        fetched_at: _dt.datetime,
        sync_run_id: str,
        explicit_window: bool,
    ) -> SourceSyncResult:
        """Fetch -> map -> land ONE source, never raising past it (CON-R3 / ARCH-R9)."""
        try:
            adapter = self._registry.get(target.source_key)
        except UnknownSourceError:
            return degraded(target, "source is not installed")
        if not isinstance(adapter, AdapterFetch):
            # No direct-API fetch seam (e.g. connectionless file upload): nothing to
            # pull on demand. Skipped, not degraded — this is the expected shape.
            return skipped(target, "source has no on-demand fetch")
        # ADP-R6: on incremental mode (no explicit window) skip the already-watermarked
        # range — fetch only forward of the source's high-water cursor.
        if not explicit_window:
            floor = await incremental_floor_date(
                self._factory_for(athlete_id), _uid(athlete_id),
                _uid(target.source_descriptor_id), window.oldest,
            )
            window = SyncWindow(oldest=floor, newest=window.newest)
        try:
            batch = await self._fetch_and_map(adapter, target, window, fetched_at, sync_run_id)
        except AuthError as exc:  # credential revoked/expired -> reauth, stop the source (AUT-R4)
            return await handle_reauth(
                self._factory_for(athlete_id), athlete_id, target, exc, seen_at=fetched_at
            )
        except Exception as exc:  # isolate the source failure; degrade not crash (ARCH-R9)
            _log.warning(
                "sync.source_degraded",
                source_key=target.source_key,
                connection_id=target.connection_id,
                error_type=type(exc).__name__,
            )
            return degraded(target, "source fetch or mapping failed")
        originals = adapter.original_files() if isinstance(adapter, OriginalArtifactSource) else []
        return await self._land(
            athlete_id, target, batch, window, sync_run_id, fetched_at, originals
        )

    async def _fetch_and_map(
        self,
        adapter: AdapterFetch,
        target: _ConnectionTarget,
        window: SyncWindow,
        fetched_at: _dt.datetime,
        sync_run_id: str,
    ) -> MappedBatch:
        """Fetch ASBOs (impure), then pure-map each in ISOLATION (ING-GAP-R5/ING-UPS-R3)."""
        api_key = resolve_api_key(self._credentials, target)
        asbos = await adapter.fetch(
            api_key=api_key,
            athlete_native_id=target.athlete_native_id,
            window=window,
        )
        ctx = FetchContext(
            ingest_run_id=sync_run_id, fetched_at=fetched_at, connection_id=target.connection_id
        )
        ref = SourceDescriptorRef(
            source_descriptor_id=target.source_descriptor_id,
            source_key=target.source_key,
            kind=target.kind,
        )
        return map_records_isolated(adapter, asbos, ref, ctx, source_key=target.source_key)

    async def _land(
        self,
        athlete_id: str,
        target: _ConnectionTarget,
        batch: MappedBatch,
        window: SyncWindow,
        sync_run_id: str,
        fetched_at: _dt.datetime,
        original_files: list[OriginalFile],
    ) -> SourceSyncResult:
        """Land the batch in ONE transaction with the cursor + gap bookkeeping (ING-UPS-R2).

        The upsert, the watermark advance (SYN-R3), the self-heal of transient gaps the
        synced range covers (ING-GAP-R4), the per-record range-precise gap (ING-GAP-R5),
        and the tier-1 original capture (ING-R8/FIL-R1) ALL commit in the SAME transaction.
        """
        if not batch.candidates and not batch.failed:
            return SourceSyncResult.ok(target, candidates_mapped=0)
        synced = synced_range(window, self._now())
        try:
            async with self._sessions.session(subject=athlete_id) as session:
                outcome = await IngestService(session).ingest(
                    athlete_id,
                    target.source_descriptor_id,
                    batch.candidates,
                    connection_id=target.connection_id,
                    ingest_run_id=_uid(sync_run_id),
                    original_files=original_files or None,
                    synced_range=synced,
                )
                await open_record_gaps(
                    session, _uid(athlete_id), _uid(target.source_descriptor_id), batch.failed,
                    ingest_run_id=_uid(sync_run_id), seen_at=fetched_at,
                )
        except Exception as exc:  # rolled back by the session ctx; degrade not crash (ARCH-R9)
            _log.warning(
                "sync.ingest_degraded",
                source_key=target.source_key,
                connection_id=target.connection_id,
                error_type=type(exc).__name__,
            )
            return degraded(target, "writing the canonical batch failed")
        if batch.failed:  # partial failure: good records landed, failed range gap-marked
            return SourceSyncResult(
                source_key=target.source_key,
                connection_id=target.connection_id,
                outcome=SyncOutcome.DEGRADED,
                candidates_mapped=len(batch.candidates),
                activities_written=len(outcome.activities_written),
                wellness_written=outcome.wellness_written,
                detail="some records could not be mapped",
            )
        return SourceSyncResult.ok(
            target,
            candidates_mapped=len(batch.candidates),
            activities_written=len(outcome.activities_written),
            wellness_written=outcome.wellness_written,
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
