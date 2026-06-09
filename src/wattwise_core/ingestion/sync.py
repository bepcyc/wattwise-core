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
others (CON-R3 / ARCH-R9), rolling that source back without fabricating a partial record.

Layer: L3 ingestion/sync — the ONLY writer to the store (ARCH-R3, via
:class:`IngestService`); imports NO named adapter module (ARCH-R2) and NO L6 edge.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.enums import AuthArchetype
from wattwise_core.ingestion._sync_records import (
    MappedBatch,
    incremental_floor_date,
    map_records_isolated,
    open_record_gaps,
)
from wattwise_core.ingestion.base import FetchContext, SourceAdapter, SourceDescriptorRef
from wattwise_core.ingestion.ingest import IngestService, OriginalFile
from wattwise_core.ingestion.registry import AdapterRegistry, UnknownSourceError
from wattwise_core.ingestion.watermark import SyncedRange
from wattwise_core.observability.logging import get_logger
from wattwise_core.persistence.models import Connection, SourceDescriptor
from wattwise_core.persistence.types import utcnow, uuid7
from wattwise_core.security.credentials import CredentialStore

_log = get_logger(__name__)


class SyncOutcome(StrEnum):
    """Per-source result of a sync attempt (CON-R3 graceful degradation vocab)."""

    OK = "ok"
    DEGRADED = "degraded"  # the source errored; other sources are unaffected (ARCH-R9)
    SKIPPED = "skipped"  # nothing to do (no connection / unauthorized / no fetcher)


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


class SessionFactory(Protocol):
    """A callable yielding a transactional :class:`AsyncSession` context (UPS-R6).

    Matches :meth:`wattwise_core.persistence.Database.session`: entering opens a
    transaction, a clean exit commits, an exception rolls back. One ``run`` uses one
    such context per source so a degraded source rolls back in isolation (ARCH-R9).
    """

    def __call__(self) -> AbstractAsyncContextManager[AsyncSession]: ...


@dataclass(frozen=True, slots=True)
class SyncWindow:
    """The inclusive ISO-date window a fetch covers (ADP-R5; deterministic input)."""

    oldest: str
    newest: str


@dataclass(frozen=True, slots=True)
class SourceSyncResult:
    """The outcome of syncing ONE source for one athlete (typed summary)."""

    source_key: str
    connection_id: str | None
    outcome: SyncOutcome
    candidates_mapped: int = 0
    activities_written: int = 0
    wellness_written: int = 0
    detail: str | None = None  # non-secret reason for a DEGRADED/SKIPPED outcome

    @classmethod
    def ok(
        cls,
        target: _ConnectionTarget,
        *,
        candidates_mapped: int = 0,
        activities_written: int = 0,
        wellness_written: int = 0,
    ) -> SourceSyncResult:
        return cls(
            source_key=target.source_key,
            connection_id=target.connection_id,
            outcome=SyncOutcome.OK,
            candidates_mapped=candidates_mapped,
            activities_written=activities_written,
            wellness_written=wellness_written,
        )

    @classmethod
    def non_ok(
        cls, target: _ConnectionTarget, outcome: SyncOutcome, detail: str
    ) -> SourceSyncResult:
        """A DEGRADED/SKIPPED result with a non-secret reason (CON-R3)."""
        return cls(
            source_key=target.source_key,
            connection_id=target.connection_id,
            outcome=outcome,
            detail=detail,
        )


@dataclass(slots=True)
class SyncRun:
    """The typed summary a :meth:`SyncOrchestrator.run` returns (on-demand sync)."""

    athlete_id: str
    sync_run_id: str
    started_at: _dt.datetime
    results: list[SourceSyncResult] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        """True when any source degraded — the caller can surface partial coverage."""
        return any(r.outcome is SyncOutcome.DEGRADED for r in self.results)

    @property
    def activities_written(self) -> int:
        return sum(r.activities_written for r in self.results)

    @property
    def wellness_written(self) -> int:
        return sum(r.wellness_written for r in self.results)


# Default lookback for a connection sync when the caller gives no explicit window.
_DEFAULT_LOOKBACK = _dt.timedelta(days=42)


class SyncOrchestrator:
    """Drives on-demand sync from authorized connections into the canonical store.

    Source-blind: selects adapters through the injected :class:`AdapterRegistry` (by
    ``source_key``) and resolves credentials through the :class:`CredentialStore`,
    never importing or branching on a named source (ARCH-R2). ``now`` is injectable so
    the built :class:`FetchContext` is deterministic in tests.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        registry: AdapterRegistry,
        credential_store: CredentialStore | None = None,
        now: Any = None,
    ) -> None:
        self._session_factory = session_factory
        self._registry = registry
        self._credentials = credential_store
        self._now = now or utcnow

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
        """Resolve the connections to sync into source-agnostic targets (read-only)."""
        async with self._session_factory() as session:
            stmt = (
                select(Connection, SourceDescriptor)
                .join(
                    SourceDescriptor,
                    Connection.source_descriptor_id == SourceDescriptor.source_descriptor_id,
                )
                .where(Connection.athlete_id == _uid(athlete_id))
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
            return _degraded(target, "source is not installed")
        if not isinstance(adapter, AdapterFetch):
            # No direct-API fetch seam (e.g. connectionless file upload): nothing to
            # pull on demand. Skipped, not degraded — this is the expected shape.
            return _skipped(target, "source has no on-demand fetch")
        # ADP-R6: on incremental mode (no explicit window) skip the already-watermarked
        # range — fetch only forward of the source's high-water cursor.
        if not explicit_window:
            floor = await incremental_floor_date(
                self._session_factory, _uid(athlete_id), _uid(target.source_descriptor_id),
                window.oldest,
            )
            window = SyncWindow(oldest=floor, newest=window.newest)
        try:
            batch = await self._fetch_and_map(adapter, target, window, fetched_at, sync_run_id)
        except Exception as exc:  # isolate the source failure; degrade not crash (ARCH-R9)
            _log.warning(
                "sync.source_degraded",
                source_key=target.source_key,
                connection_id=target.connection_id,
                error_type=type(exc).__name__,
            )
            return _degraded(target, "source fetch or mapping failed")
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
        api_key = self._resolve_api_key(target)
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

    def _resolve_api_key(self, target: _ConnectionTarget) -> str | None:
        """Resolve the opaque ``credential_ref`` to the live secret (CLI-R13, SEC-R7).

        Only ``api_key`` connections carry a usable key; it is decrypted in-memory at
        the point of use and never logged. ``None`` if connectionless or no store.
        """
        if target.auth_archetype is not AuthArchetype.API_KEY:
            return None
        if self._credentials is None or target.credential_ref is None:
            return None
        return self._credentials.resolve(target.credential_ref).get_secret_value()

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
        synced = _synced_range(window, self._now())
        try:
            async with self._session_factory() as session:
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
            return _degraded(target, "writing the canonical batch failed")
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


@dataclass(frozen=True, slots=True)
class _ConnectionTarget:
    """A source-agnostic view of one connection the orchestrator acts on.

    Carries ONLY source identity (``source_key`` / ``kind``), the archetype (consumers
    branch on archetype, never source name — GBO-R48), and the opaque ``credential_ref``
    (never the secret).
    """

    source_key: str
    kind: Any
    source_descriptor_id: str
    connection_id: str | None
    auth_archetype: AuthArchetype
    credential_ref: str | None
    athlete_native_id: str | None

    @classmethod
    def of(cls, conn: Connection, desc: SourceDescriptor) -> _ConnectionTarget:
        return cls(
            source_key=desc.source_key,
            kind=desc.kind,
            source_descriptor_id=str(desc.source_descriptor_id),
            connection_id=str(conn.connection_id),
            auth_archetype=conn.auth_archetype,
            credential_ref=conn.credential_ref,
            athlete_native_id=None,
        )


def _degraded(target: _ConnectionTarget, detail: str) -> SourceSyncResult:
    return SourceSyncResult.non_ok(target, SyncOutcome.DEGRADED, detail)


def _skipped(target: _ConnectionTarget, detail: str) -> SourceSyncResult:
    return SourceSyncResult.non_ok(target, SyncOutcome.SKIPPED, detail)


def _uid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _synced_range(window: SyncWindow, now: _dt.datetime) -> SyncedRange:
    """The committed [oldest, newest end-of-day] time range a sync covered (ING-GAP-R4)."""
    start = _dt.datetime.fromisoformat(window.oldest).replace(tzinfo=_dt.UTC)
    end = _dt.datetime.fromisoformat(window.newest).replace(tzinfo=_dt.UTC)
    return SyncedRange(oldest=start, newest=end + _dt.timedelta(days=1) - _dt.timedelta(seconds=1),
                       now=now)


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
