"""Users router — the single owner's account self-service surface (``/v1/users/me``).

Serves the one server-derived owner's own account (doc 60 §8; retention §11):

- ``GET /v1/users/me`` (``read``) — the readable :class:`UserAccount`: the account's
  per-channel delivery/notification routes (GBO-R49), the captured ``email``, and the email
  ``verified`` flag that GATES the digest email channel (a digest e-mail is delivered only
  when the email route is BOTH verified AND enabled).
- ``PATCH /v1/users/me`` (``write``) — capture/verify the digest email. The address is bound
  to the canonical ``email`` :class:`NotificationRoute` (``address_ref`` = the address); a
  NEW/changed address resets ``verified`` to ``false`` so an unverified address can never
  gate the email channel open (fail-closed, GBO-R49). The ``verified`` state is
  SERVER-controlled — never accepted from the client body (a spoofed flag is rejected by
  ``additionalProperties:false``, SCHEMA-R4).
- ``DELETE /v1/users/me`` (``write``) — request asynchronous account deletion (retention
  §11). It does NOT hard-delete inline: it records an erasure request that a background
  right-to-be-forgotten path fulfils across every store (PRIV-R8), and disables the account's
  delivery channels immediately so no notification leaks while erasure is pending. Returns a
  durable ``pending_deletion`` acknowledgement.

Boundary contract: identity is server-derived from the bearer token (AUTH-R3 / AUTH-R18) and
every read/write acts ONLY on that one owner id — no writable caller-identity field exists on
any body (SCHEMA-R4). Reads require ``read``; mutations require ``write`` (AUTH-R11), so a
token without ``write`` is ``403 insufficient-scope`` (AUTH-R7). No field is source-shaped or
carries a provider name (AUTH-R15), and no response carries a model/tier/catalog (API-R38).

The deletion-erasure handoff is an injectable seam (:data:`deletion_requester`) the app
factory overrides with the registered background-erasure recorder, so this router never
imports the concrete erasure machinery (ARCH-R22); its fail-closed default refuses every
request until wired, so a DELETE NEVER silently no-ops an erasure (fail-closed).

Requirement IDs: API-R51, AUTH-R3, AUTH-R7, AUTH-R11, AUTH-R15, AUTH-R18, GBO-R49, PRIV-R8,
SCHEMA-R4, ERR-R5, ERR-R6, ERR-R8.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.deps import CurrentPrincipal, DbSession, RateLimit
from wattwise_core.api.errors import ProblemError
from wattwise_core.api.routers.users_schemas import (
    AccountDeletionAck,
    EmailCaptureRequest,
    NotificationRouteOut,
    UserAccount,
)
from wattwise_core.domain.enums import DeliveryChannel
from wattwise_core.persistence.models import NotificationRoute

router = APIRouter(prefix="/v1/users", tags=["users"], dependencies=[RateLimit])


# --------------------------------------------------------- deletion-erasure seam


#: The async erasure handoff: given the SERVER-DERIVED athlete id + the request instant,
#: durably record an account-deletion request for the background right-to-be-forgotten path
#: (PRIV-R8). Returns nothing on success; raises on a recording failure. The app factory
#: overrides it with the registered recorder so this router never imports the concrete
#: erasure machinery (ARCH-R22); the fail-closed default refuses until wired.
DeletionRequester = Callable[[str, _dt.datetime], Awaitable[None]]


async def _unconfigured_deletion_requester(athlete_id: str, requested_at: _dt.datetime) -> None:
    """Fail-closed default: refuse the request until the factory wires a recorder.

    A DELETE must NEVER silently succeed without the erasure being recorded somewhere
    durable (fail-closed): until the real background-erasure recorder is injected, every
    deletion request surfaces a generic internal error rather than acknowledging an erasure
    that was never enqueued (ERR-R5 — no leak of why).
    """
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def deletion_requester() -> DeletionRequester:
    """Provide the async-erasure recorder seam; the app factory overrides it (PRIV-R8)."""
    return _unconfigured_deletion_requester


DeletionRequesterDep = Annotated[DeletionRequester, Depends(deletion_requester)]


# --------------------------------------------------------------------------- routes


@router.get(
    "/me",
    response_model=UserAccount,
    operation_id="getCurrentUserAccount",
    dependencies=[Depends(require_scopes(Scope.READ))],
)
async def get_me(principal: CurrentPrincipal, session: DbSession) -> UserAccount:
    """Read the one owner's account: delivery routes + captured email + verified (doc 60 §8).

    Derived from the canonical :class:`NotificationRoute` rows (GBO-R49); the ``email`` field
    mirrors the address on the ``email`` channel and ``verified`` is that channel's verified
    flag — the gate the digest email path checks. Acts ONLY on the server-derived owner id
    (AUTH-R3). An owner with no routes yet reads an honest empty/`null` account, never an
    error.
    """
    return await _account_for(session, _uid(principal.athlete_id))


@router.patch(
    "/me",
    response_model=UserAccount,
    operation_id="captureUserEmail",
    dependencies=[Depends(require_scopes(Scope.WRITE))],
)
async def patch_me(
    body: EmailCaptureRequest, principal: CurrentPrincipal, session: DbSession
) -> UserAccount:
    """Capture/verify the digest email — the gate on the digest email channel (GBO-R49).

    Binds the address to the canonical ``email`` :class:`NotificationRoute` for the owner
    (``address_ref`` = the address). Capturing a NEW/changed address resets ``verified`` to
    ``false`` (the verification flow flips it true out of band) so an unverified address can
    NEVER gate the email channel open; re-capturing the SAME address leaves an already-verified
    route untouched. ``verified`` is server-controlled — it is not a body field (a spoofed flag
    is rejected by ``additionalProperties:false``, SCHEMA-R4). Acts ONLY on the server-derived
    owner id (AUTH-R3); no caller-identity field exists.
    """
    athlete_uuid = _uid(principal.athlete_id)
    existing = await _route_row(session, athlete_uuid, DeliveryChannel.EMAIL)
    if existing is None:
        session.add(_new_email_route(athlete_uuid, body.email))
    elif existing.address_ref != body.email:
        existing.address_ref = body.email
        existing.verified = False
        existing.enabled = True
    await session.flush()
    return await _account_for(session, athlete_uuid)


@router.delete(
    "/me",
    response_model=AccountDeletionAck,
    operation_id="requestAccountDeletion",
    dependencies=[Depends(require_scopes(Scope.WRITE))],
)
async def delete_me(
    principal: CurrentPrincipal, session: DbSession, requester: DeletionRequesterDep
) -> AccountDeletionAck:
    """Request ASYNC account deletion — mark for erasure, do NOT hard-delete inline (§11).

    Records an account-deletion request the background right-to-be-forgotten path fulfils
    across every store (PRIV-R8) via the injected :data:`deletion_requester` seam (fail-closed
    until wired). It does NOT delete the owner row or canonical data inline. To prevent any
    notification leaking while erasure is pending it disables the account's delivery channels
    immediately. Acts ONLY on the server-derived owner id (AUTH-R3). Returns a durable
    ``pending_deletion`` acknowledgement stamped with the server time.
    """
    athlete_uuid = _uid(principal.athlete_id)
    requested_at = _dt.datetime.now(tz=_dt.UTC)
    await requester(principal.athlete_id, requested_at)
    await _disable_routes(session, athlete_uuid)
    await session.flush()
    return AccountDeletionAck(status="pending_deletion", requested_at=requested_at)


# --------------------------------------------------------------------------- helpers


def _uid(value: str) -> uuid.UUID:
    """Coerce the server-derived athlete id; an unparsable id is an internal error.

    The id is server-derived from the verified token (AUTH-R3), so it is always a valid
    UUID in practice; a malformed value indicates a broken auth seam and fails closed as a
    generic internal error (no leak), never a client-probeable error.
    """
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:  # pragma: no cover - server-derived id is valid
        raise ProblemError("internal-error") from exc


async def _account_for(session: AsyncSession, athlete_id: uuid.UUID) -> UserAccount:
    """Project the owner's loaded notification routes onto the readable account (doc 60 §8).

    The single read+write projection: ``email``/``verified`` mirror the ``email`` channel
    route (the digest-email gate, GBO-R49), and every route is surfaced with a masked
    address hint (ERR-R5). An owner with no routes reads an honest empty/`null` account.
    """
    routes = await _owner_routes(session, athlete_id)
    email_route = _route_for(routes, DeliveryChannel.EMAIL)
    return UserAccount(
        email=email_route.address_ref if email_route is not None else None,
        verified=email_route.verified if email_route is not None else False,
        notification_routes=[_route_out(route) for route in routes],
    )


async def _owner_routes(
    session: AsyncSession, athlete_id: uuid.UUID
) -> list[NotificationRoute]:
    """All of the owner's per-channel notification routes, channel-ordered (GBO-R49)."""
    stmt = (
        select(NotificationRoute)
        .where(NotificationRoute.athlete_id == athlete_id)
        .order_by(NotificationRoute.channel)
    )
    return list((await session.execute(stmt)).scalars().all())


async def _route_row(
    session: AsyncSession, athlete_id: uuid.UUID, channel: DeliveryChannel
) -> NotificationRoute | None:
    """The owner's route for ``channel`` (the UNIQUE ``(athlete_id, channel)`` row), if any."""
    stmt = select(NotificationRoute).where(
        NotificationRoute.athlete_id == athlete_id,
        NotificationRoute.channel == channel,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def _disable_routes(session: AsyncSession, athlete_id: uuid.UUID) -> None:
    """Disable every delivery channel so nothing is delivered while erasure is pending."""
    for route in await _owner_routes(session, athlete_id):
        route.enabled = False


def _new_email_route(athlete_id: uuid.UUID, address: str) -> NotificationRoute:
    """A freshly-captured, UNVERIFIED, enabled email route (GBO-R49 fail-closed gate)."""
    return NotificationRoute(
        athlete_id=athlete_id,
        channel=DeliveryChannel.EMAIL,
        address_ref=address,
        verified=False,
        enabled=True,
    )


def _route_for(
    routes: list[NotificationRoute], channel: DeliveryChannel
) -> NotificationRoute | None:
    """Pick the in-memory route for ``channel`` from an already-loaded list, if present."""
    for route in routes:
        if route.channel == channel:
            return route
    return None


def _route_out(route: NotificationRoute) -> NotificationRouteOut:
    """Project a canonical route onto the wire shape with a masked address hint (ERR-R5)."""
    return NotificationRouteOut(
        channel=route.channel,
        enabled=route.enabled,
        verified=route.verified,
        address_hint=_mask_address(route.address_ref),
    )


def _mask_address(address: str | None) -> str | None:
    """A redaction-safe hint for a bound address — never the full value (API-R19 / ERR-R5).

    The notification-route LIST returns only a non-reversible mask (first character +
    domain) so a read that just needs "is a channel bound" cannot exfiltrate the whole
    address; the full email is available only on the dedicated account ``email`` field.
    """
    if not address:
        return None
    local, _, domain = address.partition("@")
    if not domain:
        return "***"
    head = local[0] if local else ""
    return f"{head}***@{domain}"


__all__ = [
    "DeletionRequester",
    "deletion_requester",
    "router",
]
