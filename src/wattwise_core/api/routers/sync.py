"""Sync router — owner-triggered on-demand ingestion (``POST /v1/sync/run``, API-R46a).

OSS sync is **manual only**: creating an ``api_key`` connection does NOT auto-enqueue a
sync (API-R46 / COMM-R19), so the athlete (or an operator cron) pulls fresh data by
calling this route. Automatic cadence, webhooks, and on-connect fast-first-value are a
commercial overlay and are not mounted here.

This is an operational surface where a source name is a legitimate part of the contract
(AUTH-R15): the athlete may scope a run to a single connected source. The request scope
is ``sync`` (AUTH-R13). The run is handed to the ingestion sync orchestration — an
injectable seam (:data:`sync_orchestrator`) the app factory overrides with the real
on-demand sync service — so this router never imports a concrete source adapter
(ARCH-R22 / ONB-R4). Acting identity is server-derived (AUTH-R3); the request carries
no caller-identity field (AUTH-R18).

Endpoint:

- ``POST /v1/sync/run`` (``sync``) — body ``SyncRunRequest {connection_id?, source?}``
  (mutually exclusive); an unknown ``connection_id`` → ``404``; an absent/empty body =
  every owner connection. Returns ``202 SyncRun`` (a handle to the started run).

Requirement IDs: API-R46, API-R46a, AUTH-R3, AUTH-R13, AUTH-R15, AUTH-R18, SCHEMA-R4,
ERR-R8.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.copy import message as _copy
from wattwise_core.api.deps import CurrentPrincipal, DbSession, RateLimit
from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.persistence.models import Connection, SourceDescriptor

router = APIRouter(prefix="/v1/sync", tags=["sync"], dependencies=[RateLimit])


# --------------------------------------------------------- sync-orchestrator seam


@dataclass(frozen=True, slots=True)
class SyncTarget:
    """The server-resolved scope of one sync run (API-R46a).

    Either a single ``connection_id`` or, when the request named no scope, ``None`` =
    every owner connection. ``athlete_id`` is the server-derived owner (AUTH-R3); the
    orchestrator never reads identity from client input.
    """

    athlete_id: str
    connection_id: str | None


#: The sync-orchestration seam (API-R46a): start an on-demand sync for the resolved
#: target and return a started-run handle (its id + started_at). The app factory
#: overrides this with the OSS on-demand sync service so this router never imports a
#: named adapter (ARCH-R22 / ONB-R4); webhooks/orchestrator call the same service
#: directly (server-resolved target, no bearer). Tests inject a fake.
SyncOrchestrator = Callable[[SyncTarget], Awaitable["SyncRun"]]


async def _unconfigured_orchestrator(target: SyncTarget) -> SyncRun:
    """Fail-closed default: refuse to start a run until the factory wires the service."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def sync_orchestrator() -> SyncOrchestrator:
    """Provide the sync-orchestration seam; the app factory overrides it (API-R46a)."""
    return _unconfigured_orchestrator


OrchestratorDep = Annotated[SyncOrchestrator, Depends(sync_orchestrator)]


# --------------------------------------------------------------------------- wire shapes


class SyncRunRequest(BaseModel):
    """Scope for an on-demand sync run (API-R46).

    ``connection_id`` and ``source`` are mutually exclusive; supplying both → ``422``.
    An absent/empty body means every owner connection. Carries NO caller-identity field
    (AUTH-R18); ``additionalProperties:false`` (SCHEMA-R4) rejects a forged
    ``athlete_id`` outright.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    connection_id: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def _exclusive(self) -> SyncRunRequest:
        """Reject naming both a connection and a source (mutually exclusive, API-R46)."""
        if self.connection_id is not None and self.source is not None:
            raise ValueError("Choose either a single source or all of them, not both.")
        return self


class SyncRun(BaseModel):
    """A sync-run handle (``202``, API-R46a/API-R46c).

    ``status`` is a stable machine token a client branches on (QUAL-R13(d), never the
    sentence): ``accepted`` when a connected source was actually started, ``nothing_to_sync``
    when the owner has NO connected source to pull from so the run did nothing (API-R46c —
    no false "we're pulling your training" reassurance). ``status_text`` is the matching
    jargon-free athlete copy (API-R21/HLT-R7). No secret, watermark, or adapter internal is
    exposed.
    """

    sync_run_id: str
    status: Literal["accepted", "running", "nothing_to_sync"]
    started_at: datetime
    status_text: str


# --------------------------------------------------------------------------- route


@router.post(
    "/run",
    response_model=SyncRun,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="runSync",
    dependencies=[Depends(require_scopes(Scope.SYNC))],
)
async def run_sync(
    principal: CurrentPrincipal,
    session: DbSession,
    orchestrator: OrchestratorDep,
    body: SyncRunRequest | None = None,
) -> SyncRun:
    """Start an on-demand sync for the owner, optionally scoped to one connection (API-R46).

    A ``connection_id`` is resolved against the owner's connections (unknown → ``404``,
    API-R51); a ``source`` scopes to the owner's matching connection; an absent body
    syncs every owner connection. The resolved target (server-derived owner +
    connection) is handed to the orchestration seam, which returns the started-run
    handle (``202``). OSS never auto-enqueues — this manual route is the only trigger
    (API-R46).
    """
    request = body or SyncRunRequest()
    target = await _resolve_target(session, principal.athlete_id, request)
    return await orchestrator(target)


# --------------------------------------------------------------------------- helpers


async def _resolve_target(
    session: AsyncSession, athlete_id: str, request: SyncRunRequest
) -> SyncTarget:
    """Resolve the request to a server-side :class:`SyncTarget` (AUTH-R3/API-R51).

    A named ``connection_id`` must belong to the owner and exist (else ``404``); a
    named ``source`` resolves to the owner's connection for that source (else ``404``);
    an empty request targets every owner connection. Identity always comes from the
    verified principal, never from the body.
    """
    athlete_uuid = uuid.UUID(athlete_id)
    if request.connection_id is not None:
        connection = await _owned_connection_by_id(session, athlete_uuid, request.connection_id)
        return SyncTarget(athlete_id=athlete_id, connection_id=str(connection.connection_id))
    if request.source is not None:
        connection = await _owned_connection_by_source(session, athlete_uuid, request.source)
        return SyncTarget(athlete_id=athlete_id, connection_id=str(connection.connection_id))
    return SyncTarget(athlete_id=athlete_id, connection_id=None)


async def _owned_connection_by_id(
    session: AsyncSession, athlete_id: uuid.UUID, connection_id: str
) -> Connection:
    """Fetch the owner's connection by id; unknown/malformed → ``404`` (API-R51)."""
    try:
        target_id = uuid.UUID(connection_id)
    except ValueError as exc:
        raise ProblemError("not-found") from exc
    stmt = select(Connection).where(
        Connection.connection_id == target_id,
        Connection.athlete_id == athlete_id,
    )
    connection = (await session.execute(stmt)).scalar_one_or_none()
    if connection is None:
        raise ProblemError("not-found")
    return connection


async def _owned_connection_by_source(
    session: AsyncSession, athlete_id: uuid.UUID, source: str
) -> Connection:
    """Fetch the owner's connection for a source key; none → ``404`` (API-R51).

    Joins the connection to its registered descriptor by ``source_key`` so a request
    may scope by the source name the athlete sees on the Connections surface (AUTH-R15)
    without exposing any descriptor internals.
    """
    stmt = (
        select(Connection)
        .join(
            SourceDescriptor,
            SourceDescriptor.source_descriptor_id == Connection.source_descriptor_id,
        )
        .where(Connection.athlete_id == athlete_id, SourceDescriptor.source_key == source)
    )
    connection = (await session.execute(stmt)).scalar_one_or_none()
    if connection is None:
        raise ProblemError(
            "not-found",
            errors=[
                FieldError(
                    code="unknown_source",
                    message=_copy("sync.unknown_source"),
                    pointer="/source",
                )
            ],
        )
    return connection


def started_run(sync_run_id: str) -> SyncRun:
    """Build an accepted :class:`SyncRun` handle (a connected source was started).

    A small constructor the orchestration seam reuses so the accepted status + athlete
    copy are defined once here (API-R21), not at the wiring site. Used ONLY when the run
    actually started against at least one connected source; the no-connected-source case
    uses :func:`nothing_to_sync_run` so the reassurance is never falsely shown (API-R46c).
    """
    return SyncRun(
        sync_run_id=sync_run_id,
        status="accepted",
        started_at=datetime.now(UTC),
        status_text="We're pulling in your latest training now.",
    )


def nothing_to_sync_run(sync_run_id: str) -> SyncRun:
    """Build an HONEST :class:`SyncRun` handle for the no-connected-source case (API-R46c).

    When the owner has no connected source to pull from, the run touched nothing — so the
    handle MUST NOT claim a sync is happening (issue #118: the canned "we're pulling your
    training" reassurance is misleading on this path). The distinct ``nothing_to_sync``
    status (a stable machine token, QUAL-R13(d)) carries warm, athlete-native copy
    (API-R21/HLT-R7) that states the no-source state and the next step, without blame
    (QUAL-R13(e)), so the intervals.icu onboarding user can tell it did nothing.
    """
    return SyncRun(
        sync_run_id=sync_run_id,
        status="nothing_to_sync",
        started_at=datetime.now(UTC),
        status_text=(
            "No sources are connected yet, so there's nothing to bring in right now. "
            "Connect a source or upload a file to get started."
        ),
    )


__all__ = [
    "SyncOrchestrator",
    "SyncRun",
    "SyncRunRequest",
    "SyncTarget",
    "nothing_to_sync_run",
    "router",
    "started_run",
    "sync_orchestrator",
]
