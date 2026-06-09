"""Sync-orchestrator value types + per-source outcome helpers.

A focused split of the orchestrator's connection-target and result concerns (QUAL-R9):
the source-agnostic :class:`_ConnectionTarget` view of one connection, the typed
:class:`SourceSyncResult` / :class:`SyncRun` summaries, the graceful-degradation
:class:`SyncOutcome` vocabulary (CON-R3), the credential resolution at the point of use
(SEC-R7), and the re-auth transition that flips a Connection to ``reauth_required`` while
emitting a typed §7 gap and stopping scheduling (AUT-R4). These are L3 ingestion-side
helpers; they import only the rankless domain enums, the adapter-contract seam, the
watermark range type, the canonical ORM, and the credential store — never the
orchestrator (no cycle) and never a consumer layer.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.enums import (
    AuthArchetype,
    ConnectionStatus,
    GapReason,
    GboType,
    Severity,
)
from wattwise_core.ingestion.base import AuthError, AuthGapSignal
from wattwise_core.ingestion.watermark import SyncedRange, open_gap
from wattwise_core.observability.logging import get_logger
from wattwise_core.persistence.models import Connection, SourceDescriptor
from wattwise_core.security.credentials import CredentialStore

_log = get_logger(__name__)


class SyncOutcome(StrEnum):
    """Per-source result of a sync attempt (CON-R3 graceful degradation vocab)."""

    OK = "ok"
    DEGRADED = "degraded"  # a transient/source error; other sources unaffected (ARCH-R9)
    SKIPPED = "skipped"  # nothing to do (no connection / unauthorized / no fetcher)
    REAUTH_REQUIRED = "reauth_required"  # credential revoked/expired; source stopped (AUT-R4)


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
    # the §7 auth-gap signal for a REAUTH_REQUIRED outcome (AUT-R4)
    gap: AuthGapSignal | None = None

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

    @classmethod
    def reauth(cls, target: _ConnectionTarget, gap: AuthGapSignal) -> SourceSyncResult:
        """A REAUTH_REQUIRED result carrying the §7 auth-gap signal (AUT-R4)."""
        return cls(
            source_key=target.source_key,
            connection_id=target.connection_id,
            outcome=SyncOutcome.REAUTH_REQUIRED,
            detail=gap.detail or None,
            gap=gap,
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


def resolve_api_key(
    credentials: CredentialStore | None, target: _ConnectionTarget
) -> str | None:
    """Resolve the opaque ``credential_ref`` to the live secret (CLI-R13, SEC-R7).

    Only ``api_key`` connections carry a usable key; it is decrypted in-memory at
    the point of use and never logged. ``None`` if connectionless or no store.
    """
    if target.auth_archetype is not AuthArchetype.API_KEY:
        return None
    if credentials is None or target.credential_ref is None:
        return None
    return credentials.resolve(target.credential_ref).get_secret_value()


async def handle_reauth(
    session_factory: SessionFactory,
    athlete_id: str,
    target: _ConnectionTarget,
    exc: AuthError,
    *,
    seen_at: _dt.datetime,
) -> SourceSyncResult:
    """Flip the Connection to reauth_required, PERSIST a typed gap, stop the source (AUT-R4).

    Set ``status=reauth_required`` (canonical enum, doc 20), open a PERSISTED typed gap
    (§7) so the failure is queryable by downstream consumers (AUT-R4 / ING-GAP-R1) — never
    merely an in-memory signal — and stop scheduling, without deleting prior data (ING-R4;
    only the status row is touched + the gap row is opened). The status flip and the gap
    open ride the SAME session/transaction so they commit atomically (ING-UPS-R2). The gap
    is TERMINAL (``transient=False``): a revoked/expired credential re-hits the same 401/403
    and never self-heals (AUT-R4) — it needs the athlete to re-authorize, so the transient
    self-heal (ING-GAP-R4) MUST NOT auto-close it. Future runs skip the source:
    ``_select_connections`` excludes the persisted ``reauth_required`` status.
    """
    _log.warning(
        "sync.source_reauth_required",
        source_key=target.source_key,
        connection_id=target.connection_id,
        reason=exc.kind.value,
    )
    async with session_factory() as session:
        if target.connection_id is not None:
            conn = await session.get(Connection, _uid(target.connection_id))
            if conn is not None:
                conn.status = ConnectionStatus.REAUTH_REQUIRED
        await open_gap(
            session,
            _uid(athlete_id),
            _uid(target.source_descriptor_id),
            GboType.ACTIVITY,
            reason=GapReason.NEEDS_REAUTH,
            seen_at=seen_at,
            severity=Severity.WARNING,
            transient=False,
        )
    return SourceSyncResult.reauth(
        target, AuthGapSignal.needs_reauth("source credential needs re-authorization")
    )


def degraded(target: _ConnectionTarget, detail: str) -> SourceSyncResult:
    return SourceSyncResult.non_ok(target, SyncOutcome.DEGRADED, detail)


def skipped(target: _ConnectionTarget, detail: str) -> SourceSyncResult:
    return SourceSyncResult.non_ok(target, SyncOutcome.SKIPPED, detail)


def _uid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def synced_range(window: SyncWindow, now: _dt.datetime) -> SyncedRange:
    """The committed [oldest, newest end-of-day] time range a sync covered (ING-GAP-R4)."""
    start = _dt.datetime.fromisoformat(window.oldest).replace(tzinfo=_dt.UTC)
    end = _dt.datetime.fromisoformat(window.newest).replace(tzinfo=_dt.UTC)
    return SyncedRange(
        oldest=start,
        newest=end + _dt.timedelta(days=1) - _dt.timedelta(seconds=1),
        now=now,
    )


__all__ = [
    "SessionFactory",
    "SourceSyncResult",
    "SyncOutcome",
    "SyncRun",
    "SyncWindow",
    "_ConnectionTarget",
    "_uid",
    "degraded",
    "handle_reauth",
    "resolve_api_key",
    "skipped",
    "synced_range",
]
