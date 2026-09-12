"""Onboarding router — first-connection setup status (``GET /v1/onboarding/status``).

A composed, read-only view (API-R46) that tells the client where the athlete is in
first-time setup: whether any source is connected, the connected sources, the
first-sync progress, whether the first useful data is ready, and the suggested next
step. It is DERIVED from canonical state (connections + whether any activity has
landed + whether the FTP fitness signature is set), never a hand-maintained counter.

Setting the FTP fitness signature is a first-run prerequisite the suggested next step
surfaces (API-R46b): power TSS/IF (and therefore the PMC load series) require an
effective FTP, so an athlete whose first rides have landed but whose current-sport FTP
is unset would otherwise read an all-zero chart with no hint why. When canonical data is
ready but FTP is NOT set, ``suggested_next_step`` is ``set_ftp`` (PUT /v1/athlete/signature),
NOT ``all_set``. The FTP-set check mirrors the analytics resolution (ANL-R9 / GBO-R27):
the latest-effective signature for the athlete's ``current_sport`` as-of today must exist
AND carry a non-NULL ``ftp_w`` — a signature row for a different sport, or one whose
``ftp_w`` is NULL, still counts as "FTP unset".

OSS carve-out (API-R46): creating an ``api_key`` connection does NOT auto-enqueue a
sync, so ``first_sync_state`` stays ``not_started`` until the athlete runs a manual
``POST /v1/sync/run`` and data lands. The file-upload cohort can reach
``first_data_ready=true`` with no connection at all (``has_connection`` may be
``false`` after a first import).

This surface MUST NOT name a source: it is NOT one of the three source-identity-exempt
surfaces (AUTH-R15/API-R28 permits a provider/source name only on /v1/connections,
/v1/sync, /v1/data-health), so the connection rows here carry the opaque connection id +
generic status + auth archetype, never the source key/display name. Acting identity is
server-derived (AUTH-R3); the request carries no caller-identity field.

Requirement IDs: API-R46, API-R46b, API-R47, AUTH-R3, AUTH-R11, AUTH-R15, API-R28.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.deps import CurrentPrincipal, DbSession, RateLimit
from wattwise_core.domain.enums import AuthArchetype, ConnectionStatus
from wattwise_core.persistence.models import Activity, Athlete, Connection, FitnessSignature

router = APIRouter(prefix="/v1/onboarding", tags=["onboarding"], dependencies=[RateLimit])


#: Closed first-sync progress vocabulary (SCHEMA-R3 ``first_sync_state``).
FirstSyncState = Literal["not_started", "syncing_recent", "recent_ready", "backfilling", "complete"]


# --------------------------------------------------------------------------- wire shapes


class OnboardingConnection(BaseModel):
    """One connection in the onboarding progress view.

    Onboarding is NOT one of the three source-identity-exempt surfaces (AUTH-R15/API-R28
    permits a provider/source NAME only on /v1/connections, /v1/sync, /v1/data-health), so
    this composed progress view exposes the connection's opaque id + generic status + auth
    archetype, but NOT the source key or its display name. A client that needs the provider
    identity reads it from /v1/connections.
    """

    connection_id: str
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
        "connect_a_source", "run_first_sync", "waiting_for_data", "set_ftp", "all_set"
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

    Reads the owner's connections, whether any canonical activity has landed, and
    whether the current-sport FTP signature is set, then derives the first-sync
    progress and the suggested next step. The OSS carve-out holds: a fresh ``api_key``
    connection with no manual sync yet reports ``first_sync_state="not_started"`` (no
    auto-enqueue, API-R46). When data is ready but FTP is unset, the next step is
    ``set_ftp`` so the first chart is not a silent zero (API-R46b).
    """
    athlete_uuid = uuid.UUID(principal.athlete_id)
    connections = await _owner_connections(session, athlete_uuid)
    first_data_ready = await _has_canonical_activity(session, athlete_uuid)
    ftp_ready = await _ftp_is_set(session, athlete_uuid)
    state = _derive_first_sync_state(connections, first_data_ready=first_data_ready)
    return OnboardingStatus(
        has_connection=bool(connections),
        connections=connections,
        first_sync_state=state,
        first_data_ready=first_data_ready,
        suggested_next_step=_suggest_next_step(
            connections, state, first_data_ready=first_data_ready, ftp_ready=ftp_ready
        ),
    )


# --------------------------------------------------------------------------- helpers


async def _owner_connections(
    session: AsyncSession, athlete_id: uuid.UUID
) -> list[OnboardingConnection]:
    """Return the owner's connections as progress rows — no source identity (AUTH-R15)."""
    stmt = select(Connection).where(Connection.athlete_id == athlete_id)
    conns = (await session.execute(stmt)).scalars().all()
    return [
        OnboardingConnection(
            connection_id=str(conn.connection_id),
            status=conn.status,
            auth_archetype=conn.auth_archetype,
        )
        for conn in conns
    ]


async def _has_canonical_activity(session: AsyncSession, athlete_id: uuid.UUID) -> bool:
    """True once at least one canonical activity exists for the owner (first useful data).

    Derived from the canonical store (HLT-R2): the first activity landing is the
    first-value signal, whether it came from a sync run or a file import.
    """
    stmt = select(func.count()).select_from(Activity).where(Activity.athlete_id == athlete_id)
    count = (await session.execute(stmt)).scalar_one()
    return bool(count)


async def _ftp_is_set(session: AsyncSession, athlete_id: uuid.UUID) -> bool:
    """True once an effective FTP exists for the owner's CURRENT sport (API-R46b).

    Mirrors the analytics resolution the power stack grounds on (ANL-R9 / GBO-R27): the
    FTP that feeds TSS/IF is the latest-effective ``fitness_signature`` for the athlete's
    ``current_sport`` as-of today, honoring the closed-interval ``[effective_date,
    effective_to)`` so a superseded row never shadows its successor. Gating on the actual
    resolved ``ftp_w`` — NOT on the mere presence of a signature row — is deliberate:

    * ``current_sport`` unset (NULL) → nothing resolves → FTP is NOT set;
    * a signature exists only for a DIFFERENT sport (not ``current_sport``) → it never
      matches the scope → FTP is NOT set;
    * the effective signature for ``current_sport`` exists but its ``ftp_w`` is NULL →
      FTP is NOT set.

    Only an effective current-sport signature whose ``ftp_w IS NOT NULL`` counts as set —
    exactly the condition under which the power analytics can produce a non-zero TSS.
    """
    sport = (
        await session.execute(select(Athlete.current_sport).where(Athlete.athlete_id == athlete_id))
    ).scalar_one_or_none()
    if sport is None:
        return False
    today = _dt.datetime.now(tz=_dt.UTC).date()
    as_of_instant = _dt.datetime.combine(today, _dt.time.min, tzinfo=_dt.UTC)
    ftp_w = (
        await session.execute(
            select(FitnessSignature.ftp_w)
            .where(
                FitnessSignature.athlete_id == athlete_id,
                FitnessSignature.signature_type == sport,
                FitnessSignature.effective_date <= today,
                or_(
                    FitnessSignature.effective_to.is_(None),
                    FitnessSignature.effective_to > as_of_instant,
                ),
            )
            .order_by(FitnessSignature.effective_date.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return ftp_w is not None


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
    ftp_ready: bool,
) -> Literal["connect_a_source", "run_first_sync", "waiting_for_data", "set_ftp", "all_set"]:
    """Suggest the next setup action as a machine token (client maps to copy, API-R21).

    No connection and no data yet → connect a source (or upload a file). Connected but
    no data → run the first sync (OSS is manual, API-R46). Data ready but the current-sport
    FTP is unset → set the FTP signature so the first chart is not a silent zero (API-R46b:
    power TSS/IF and the PMC need an effective FTP). Data ready AND FTP set → all set.
    """
    if first_data_ready:
        return "all_set" if ftp_ready else "set_ftp"
    if not connections:
        return "connect_a_source"
    return "run_first_sync"


__all__ = [
    "FirstSyncState",
    "OnboardingConnection",
    "OnboardingStatus",
    "router",
]
