"""Connection management router — the ``/v1/connections`` list / get / disconnect surface.

The READ + lifecycle half of the connections surface (API-R27), split out of the connect
flow (``connections.py``) so each module stays within the QUAL-R9 module-size ceiling. It
reuses that module's ``_owned_connection`` ownership lookup (API-R51) so the existence-not-
ownership resolution is defined once.

This is one of the three surfaces where a source name is a legitimate part of the consumer
contract (AUTH-R15): the athlete is managing a data source, so the source key + display
name appear here (and only here, on Sync and Data-health).

- ``GET /v1/connections`` (``read``) — the athlete's connections (API-R27) in the PAGE-R4
  envelope (``{data, page}``); each row is
  ``{connection_id, source, display_name, status, connected_at, last_synced_at, scopes}``
  PLUS the two API-R47 additive read-only fields ``auth_archetype`` and ``first_sync_state``.
  The ``status`` is the canonical four-member ``connection.status`` enum (doc 20 §3.11,
  verbatim); ``scopes`` is the canonical ``connection.scopes`` projection (no
  ``scopes_granted`` rename).
- ``GET /v1/connections/{connection_id}`` (``read``) — one connection with its typed
  status (API-R27); an unknown/foreign id → ``404 not-found`` (API-R51).
- ``POST /v1/connections/{connection_id}/disconnect`` (``write``) → ``204`` — a
  DATA-PRESERVING disconnect (API-R27 / API-R29): sets ``status=disconnected`` and drops
  the credential ref, but NEVER deletes the athlete's already-ingested data — analytics
  re-resolve to the next-best data and surface reduced precision rather than erroring
  (the graceful-degradation guarantee).

Identity is server-derived from the bearer token (AUTH-R3); the client never supplies an
athlete id. Scopes gate capability (AUTH-R7/R11): reads need ``read``, disconnect needs
``write``.

Requirement IDs: API-R27, API-R29, API-R46, API-R47, API-R51, AUTH-R3, AUTH-R7, AUTH-R11,
AUTH-R15, AUTH-R18, GBO-R43, PAGE-R1, PAGE-R4.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.activity_schemas import Page
from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.deps import CurrentPrincipal, DbSession, RateLimit
from wattwise_core.api.errors import ProblemError
from wattwise_core.api.routers.connections import _owned_connection
from wattwise_core.api.routers.onboarding import (
    FirstSyncState,
    OnboardingConnection,
    _derive_first_sync_state,
    _has_canonical_activity,
)
from wattwise_core.domain.enums import AuthArchetype, ConnectionStatus
from wattwise_core.persistence.models import Connection, SourceDescriptor

router = APIRouter(prefix="/v1/connections", tags=["connections"], dependencies=[RateLimit])

_ConnectionId = Annotated[str, Path(description="An existing connection id.")]


class ConnectionSummary(BaseModel):
    """The ``Connection`` resource for the list/detail surface (API-R27 / API-R47).

    The exact API-R27 field set ``{ connection_id, source, display_name, status,
    connected_at, last_synced_at, scopes }`` PLUS the API-R47 additive read-only fields
    ``auth_archetype`` (GBO-R48) so a client renders the right reconnect affordance and
    ``first_sync_state`` (API-R46) so the same onboarding progression is visible per
    connection. The ``status`` is the canonical four-member ``connection.status`` enum
    (doc 20 §3.11, referenced verbatim); ``scopes`` is the canonical ``connection.scopes``
    projection (no ``scopes_granted`` rename). The ``source``/``display_name`` are resolved
    from the canonical ``source_descriptor`` — the one consumer list where a named source
    is a legitimate field (the AUTH-R15 exception).
    """

    connection_id: str
    source: str
    display_name: str
    status: ConnectionStatus
    auth_archetype: AuthArchetype
    first_sync_state: FirstSyncState
    connected_at: datetime | None
    last_synced_at: datetime | None
    scopes: list[str]


class ConnectionList(BaseModel):
    """Paginated ``GET /v1/connections`` envelope (API-R27, PAGE-R4).

    The project-wide cursor-pagination envelope: ``data`` items + a ``page`` sub-object
    (``limit``/``next_cursor``/``has_more``), identical to every other paginated list
    (``ActivityList``, ``GoalList``, …). The OSS connections list is a small bounded
    collection, so it is returned as a single page (``has_more=False``, ``next_cursor=None``).
    """

    data: list[ConnectionSummary]
    page: Page


@router.get(
    "",
    response_model=ConnectionList,
    operation_id="listConnections",
    dependencies=[Depends(require_scopes(Scope.READ))],
)
async def list_connections(principal: CurrentPrincipal, session: DbSession) -> ConnectionList:
    """List the athlete's source connections (API-R27).

    The one consumer list where a named ``source`` is a legitimate field (the AUTH-R15
    exception). Each row carries the canonical four-member ``status`` enum, the
    ``scopes`` projection, the API-R47 ``auth_archetype`` and ``first_sync_state``.
    Scoped to the one server-derived athlete (AUTH-R3); ordered by ``connection_id`` for
    a stable, backend-portable view. Returned in the project-wide PAGE-R4 envelope as a
    single page (the OSS connections collection is small and bounded).
    """
    athlete_uuid = uuid.UUID(principal.athlete_id)
    stmt = (
        select(Connection)
        .where(Connection.athlete_id == athlete_uuid)
        .order_by(Connection.connection_id)
    )
    rows = (await session.execute(stmt)).scalars().all()
    first_sync_state = await _athlete_first_sync_state(session, athlete_uuid)
    items = [await _to_summary(session, row, first_sync_state) for row in rows]
    page = Page(limit=len(items), next_cursor=None, has_more=False)
    return ConnectionList(data=items, page=page)


@router.get(
    "/{connection_id}",
    response_model=ConnectionSummary,
    operation_id="getConnection",
    dependencies=[Depends(require_scopes(Scope.READ))],
)
async def get_connection(
    connection_id: _ConnectionId,
    principal: CurrentPrincipal,
    session: DbSession,
) -> ConnectionSummary:
    """Return one connection with its typed status (API-R27).

    An unknown/malformed/foreign id → ``404 not-found`` (API-R51). The ``status`` is the
    canonical connection status; the source key/display name appear here per AUTH-R15.
    """
    athlete_uuid = uuid.UUID(principal.athlete_id)
    connection = await _owned_connection(session, athlete_uuid, connection_id)
    first_sync_state = await _athlete_first_sync_state(session, athlete_uuid)
    return await _to_summary(session, connection, first_sync_state)


@router.post(
    "/{connection_id}/disconnect",
    status_code=204,
    operation_id="disconnectConnection",
    dependencies=[Depends(require_scopes(Scope.WRITE))],
)
async def disconnect_connection(
    connection_id: _ConnectionId,
    principal: CurrentPrincipal,
    session: DbSession,
) -> None:
    """Disconnect a source — data-preserving (API-R27 / API-R29) → ``204``.

    Sets the canonical ``status`` to ``disconnected`` and drops the stored credential
    ref; it MUST NOT delete the athlete's already-ingested data — analytics re-resolve
    to the next-best available data and surface reduced precision rather than erroring
    (the graceful-degradation guarantee, API-R29). An unknown/foreign id → ``404``
    (API-R51).
    """
    athlete_uuid = uuid.UUID(principal.athlete_id)
    connection = await _owned_connection(session, athlete_uuid, connection_id)
    connection.status = ConnectionStatus.DISCONNECTED
    connection.credential_ref = None
    await session.flush()


async def _athlete_first_sync_state(session: AsyncSession, athlete_id: uuid.UUID) -> FirstSyncState:
    """The athlete's onboarding first-sync progression (API-R46), reused for API-R47.

    Derived from the SAME canonical state as ``OnboardingStatus`` (the athlete's
    connections + whether any canonical activity has landed), so the ``first_sync_state``
    exposed per-connection on the ``Connection`` resource (API-R47) matches the value the
    onboarding surface reports — not a second, divergent derivation. The OSS carve-out
    holds: a fresh ``api_key`` connection with no manual sync yet stays ``not_started``.
    """
    stmt = select(Connection).where(Connection.athlete_id == athlete_id)
    rows = (await session.execute(stmt)).scalars().all()
    onboarding_rows = [
        OnboardingConnection(
            connection_id=str(row.connection_id),
            status=row.status,
            auth_archetype=row.auth_archetype,
        )
        for row in rows
    ]
    first_data_ready = await _has_canonical_activity(session, athlete_id)
    return _derive_first_sync_state(onboarding_rows, first_data_ready=first_data_ready)


async def _to_summary(
    session: AsyncSession, connection: Connection, first_sync_state: FirstSyncState
) -> ConnectionSummary:
    """Render a persisted connection to the API-R27 ``Connection`` resource (API-R27/R47).

    The ``source``/``display_name`` are resolved from the canonical ``source_descriptor``
    (the AUTH-R15-exception fields); the four-member ``status`` enum, the ``scopes``
    projection, ``connected_at``/``last_synced_at``, and the API-R47 ``auth_archetype``
    are read straight off the canonical row. ``first_sync_state`` is the athlete-level
    onboarding progression (API-R46), passed in so the per-request derivation runs once.
    """
    descriptor = await session.get(SourceDescriptor, connection.source_descriptor_id)
    if descriptor is None:
        raise ProblemError("internal-error")
    return ConnectionSummary(
        connection_id=str(connection.connection_id),
        source=descriptor.source_key,
        display_name=descriptor.display_name,
        status=connection.status,
        auth_archetype=connection.auth_archetype,
        first_sync_state=first_sync_state,
        connected_at=connection.connected_at,
        last_synced_at=connection.last_synced_at,
        scopes=list(connection.scopes),
    )


__all__ = ["ConnectionList", "ConnectionSummary", "router"]
