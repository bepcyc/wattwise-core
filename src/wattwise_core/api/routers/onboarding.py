"""Onboarding router — first-connection setup status (``GET /v1/onboarding/status``).

A composed, read-only view (API-R46) that tells the client where the athlete is in
first-time setup: whether any source is connected, the connected sources, the
first-sync progress, whether the first useful data is ready, and the suggested next
step. It is DERIVED from canonical state (connections + whether any activity has
landed), never a hand-maintained counter.

OSS carve-out (API-R46): creating an ``api_key`` connection does NOT auto-enqueue a
sync, so ``first_sync_state`` stays ``not_started`` until the athlete runs a manual
``POST /v1/sync/run`` and data lands. The file-upload cohort can reach
``first_data_ready=true`` with no connection at all (``has_connection`` may be
``false`` after a first import).

This surface MAY name a source (it composes the Connections list, AUTH-R15). Acting
identity is server-derived (AUTH-R3); the request carries no caller-identity field.

Requirement IDs: API-R46, API-R47, AUTH-R3, AUTH-R11, AUTH-R15.
"""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.deps import CurrentPrincipal, DbSession
from wattwise_core.domain.enums import AuthArchetype, ConnectionStatus
from wattwise_core.persistence.models import Activity, Connection, SourceDescriptor

router = APIRouter(prefix="/v1/onboarding", tags=["onboarding"])


#: Closed first-sync progress vocabulary (SCHEMA-R3 ``first_sync_state``).
FirstSyncState = Literal["not_started", "syncing_recent", "recent_ready", "backfilling", "complete"]


# --------------------------------------------------------------------------- wire shapes


class OnboardingConnection(BaseModel):
    """One connected source in the onboarding view (composes Connections, AUTH-R15)."""

    connection_id: str
    source: str
    display_name: str
    status: ConnectionStatus
    auth_archetype: AuthArchetype


class OnboardingStatus(BaseModel):
    """The composed first-setup status (``GET /status``, API-R46).

    ``first_sync_state`` is the closed progress enum; ``suggested_next_step`` is a
    machine token a client maps to athlete-native copy. All fields are derived from
    canonical state — no source-shaped internals, no model/tier/cost.
    """

    has_connection: bool
    connections: list[OnboardingConnection]
    first_sync_state: FirstSyncState
    first_data_ready: bool
    suggested_next_step: Literal[
        "connect_a_source", "run_first_sync", "waiting_for_data", "all_set"
    ]


# --------------------------------------------------------------------------- route


@router.get(
    "/status",
    response_model=OnboardingStatus,
    operation_id="getOnboardingStatus",
    dependencies=[Depends(require_scopes(Scope.READ))],
)
async def onboarding_status(principal: CurrentPrincipal, session: DbSession) -> OnboardingStatus:
    """Return the composed, derived first-setup status for the owner (API-R46).

    Reads the owner's connections and whether any canonical activity has landed, then
    derives the first-sync progress and the suggested next step. The OSS carve-out
    holds: a fresh ``api_key`` connection with no manual sync yet reports
    ``first_sync_state="not_started"`` (no auto-enqueue, API-R46).
    """
    athlete_uuid = uuid.UUID(principal.athlete_id)
    connections = await _owner_connections(session, athlete_uuid)
    first_data_ready = await _has_canonical_activity(session, athlete_uuid)
    state = _derive_first_sync_state(connections, first_data_ready=first_data_ready)
    return OnboardingStatus(
        has_connection=bool(connections),
        connections=connections,
        first_sync_state=state,
        first_data_ready=first_data_ready,
        suggested_next_step=_suggest_next_step(
            connections, state, first_data_ready=first_data_ready
        ),
    )


# --------------------------------------------------------------------------- helpers


async def _owner_connections(
    session: AsyncSession, athlete_id: uuid.UUID
) -> list[OnboardingConnection]:
    """Return the owner's connections joined to their descriptor (source name, AUTH-R15)."""
    stmt = (
        select(Connection, SourceDescriptor)
        .join(
            SourceDescriptor,
            SourceDescriptor.source_descriptor_id == Connection.source_descriptor_id,
        )
        .where(Connection.athlete_id == athlete_id)
    )
    rows = (await session.execute(stmt)).all()
    return [
        OnboardingConnection(
            connection_id=str(conn.connection_id),
            source=descriptor.source_key,
            display_name=descriptor.display_name,
            status=conn.status,
            auth_archetype=conn.auth_archetype,
        )
        for conn, descriptor in rows
    ]


async def _has_canonical_activity(session: AsyncSession, athlete_id: uuid.UUID) -> bool:
    """True once at least one canonical activity exists for the owner (first useful data).

    Derived from the canonical store (HLT-R2): the first activity landing is the
    first-value signal, whether it came from a sync run or a file import.
    """
    stmt = select(func.count()).select_from(Activity).where(Activity.athlete_id == athlete_id)
    count = (await session.execute(stmt)).scalar_one()
    return bool(count)


def _derive_first_sync_state(
    connections: list[OnboardingConnection], *, first_data_ready: bool
) -> FirstSyncState:
    """Derive the first-sync progress from canonical state (API-R46 carve-out).

    Once canonical data exists, setup has produced its first value → ``complete``.
    Otherwise it is ``not_started``: OSS never auto-enqueues a sync on connect, so a
    fresh connection with no data yet stays ``not_started`` until a manual
    ``POST /v1/sync/run`` lands data (API-R46). (``syncing_recent``/``backfilling`` are
    transient states a commercial orchestrator surfaces; OSS on-demand sync is
    short-lived and reports the terminal states only.)
    """
    if first_data_ready:
        return "complete"
    return "not_started"


def _suggest_next_step(
    connections: list[OnboardingConnection],
    state: FirstSyncState,
    *,
    first_data_ready: bool,
) -> Literal["connect_a_source", "run_first_sync", "waiting_for_data", "all_set"]:
    """Suggest the next setup action as a machine token (client maps to copy, API-R21).

    No connection and no data yet → connect a source (or upload a file). Connected but
    no data → run the first sync (OSS is manual, API-R46). Data ready → all set.
    """
    if first_data_ready:
        return "all_set"
    if not connections:
        return "connect_a_source"
    return "run_first_sync"


__all__ = [
    "FirstSyncState",
    "OnboardingConnection",
    "OnboardingStatus",
    "router",
]
