"""The auth feature router — token issue/refresh/revoke + bot account-link (§7A, API-R23).

The full ``/v1/auth`` surface (the canonical Auth group, API-R10):

- ``POST /v1/auth/token`` (public, AUTH-R10) — first-party owner sign-in -> ``AuthTokens``
  with a REAL rotating ``refresh_token`` (a fresh refresh-token family).
- ``POST /v1/auth/refresh`` (not public — the presented refresh token IS the credential) —
  rotating single-use exchange; replay of a consumed token -> ``401`` AND revokes the
  whole family (reuse detection).
- ``POST /v1/auth/revoke`` (not public) — revoke the presented token + its family -> ``204``.
- ``POST /v1/auth/link/start`` (public) — mint a short-lived single-use
  ``LinkChallenge { link_code, expires_at }``; the code is NOT itself a credential.
- ``POST /v1/auth/link/approve`` (bearer) — the AUTHENTICATED owner proves control of the
  WattWise account by approving the code in-app (the AUTH-R8 proof-of-control step).
- ``POST /v1/auth/link/complete`` (public) — redeem a PROVEN code -> per-athlete delegated
  ``AuthTokens`` (the bot token, AUTH-R8); expired/used/unproven -> ``409``, forged -> ``401``.

Refresh-token families + link-challenge state are OPERATIONAL state on the dedicated
agent-state store (amended ARCH-R13), hash-only at rest (AUTH-R9). Token responses and
link challenges carry no object contents / internal ids / secret material beyond the
issued credential itself (API-R24 / AUTH-R9). Lifetimes are config-loaded (CFG-R1a).

Requirement IDs: API-R23, API-R24, AUTH-R8, AUTH-R9, AUTH-R10, ARCH-R13, CFG-R1a.
"""

from __future__ import annotations

import hmac
from typing import Annotated, Final

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.auth_state import (
    approve_link_challenge,
    complete_link_challenge,
    consume_refresh_token,
    find_refresh_token,
    issue_refresh_token,
    revoke_family,
    start_link_challenge,
)
from wattwise_core.api.auth import (
    DELEGATED_CLIENT,
    Principal,
    Scope,
    authenticate,
    issue_access_token,
)
from wattwise_core.api.deps import AppSettings, PublicRateLimit, get_agent_state_session
from wattwise_core.api.errors import ProblemError
from wattwise_core.config import Settings
from wattwise_core.identity import OWNER_SUBJECT
from wattwise_core.observability.audit import audit_event

#: The WWW-Authenticate challenge returned with an invalid sign-in (AUTH-R1/API-R23).
_BEARER_CHALLENGE: Final = {"WWW-Authenticate": "Bearer"}

#: Scopes the OSS first-party token grants the single owner (every in-OSS capability).
OWNER_SCOPES: Final = (
    Scope.READ,
    Scope.WRITE,
    Scope.AGENT,
    Scope.SYNC,
    Scope.EXPORT,
    Scope.ADMIN,
)

#: Scopes a DELEGATED (bot-link) token grants: every athlete capability EXCEPT the
#: operator/admin surface — a delegated client fronts the athlete, not the operator.
DELEGATED_SCOPES: Final = (
    Scope.READ,
    Scope.WRITE,
    Scope.AGENT,
    Scope.SYNC,
    Scope.EXPORT,
)

router = APIRouter(prefix="/v1/auth", tags=["auth"], dependencies=[PublicRateLimit])

StateSession = Annotated[AsyncSession, Depends(get_agent_state_session)]


# ----------------------------------------------------------------- request bodies


class TokenRequest(BaseModel):
    """The first-party sign-in exchange body for ``POST /v1/auth/token`` (API-R23).

    Carries ONLY the owner sign-in secret — the platform's first-party credential —
    and no caller-identity field (the subject is fixed server-side, AUTH-R3).
    ``additionalProperties:false`` (SCHEMA-R4) rejects any forged extra property.
    """

    model_config = ConfigDict(extra="forbid")

    owner_secret: str = Field(min_length=1, max_length=512)


class RefreshRequest(BaseModel):
    """``POST /v1/auth/refresh`` / ``/revoke`` body: the presented refresh token only."""

    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(min_length=1, max_length=512)


class LinkCodeRequest(BaseModel):
    """``POST /v1/auth/link/approve`` / ``/link/complete`` body: the link code only."""

    model_config = ConfigDict(extra="forbid")

    link_code: str = Field(min_length=1, max_length=64)


class LinkChallenge(BaseModel):
    """The ``POST /v1/auth/link/start`` response (API-R23 / API-R24).

    Carries ONLY the short-lived single-use ``link_code`` + its expiry — no object
    contents, no internal ids, no secret material (AUTH-R9): the code is NOT itself a
    credential until the owner proves account control and the bot redeems it.
    """

    model_config = ConfigDict(extra="forbid")

    link_code: str
    expires_at: str


class LinkApproval(BaseModel):
    """The ``POST /v1/auth/link/approve`` acknowledgement (the proof-of-control step)."""

    model_config = ConfigDict(extra="forbid")

    status: str = "approved"


# ----------------------------------------------------------------------- helpers


def _verify_owner_secret(settings: Settings, presented: str) -> None:
    """Constant-time-verify the first-party owner secret, else ``401`` (API-R23).

    OSS is single-owner: the first-party credential is the configured
    ``token_signing_key`` secret (the operator's boot secret). A mismatch — or an
    absent configured secret — yields ``401 unauthenticated`` with NO unknown-user /
    wrong-secret distinction and no credential echo (API-R23 / AUTH-R9). Only a verified
    secret proceeds to mint a token (no fail-open issuance).
    """
    configured = settings.token_signing_key
    if configured is None or not hmac.compare_digest(
        presented.encode(), configured.get_secret_value().encode()
    ):
        raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE)


# ------------------------------------------------------------------------- routes


@router.post("/token", operation_id="issueToken")
async def issue_token(
    body: TokenRequest, settings: AppSettings, session: StateSession
) -> dict[str, object]:
    """Exchange the first-party owner credential for ``AuthTokens`` (API-R23/R24).

    Public (pre-token, AUTH-R10) but NOT fail-open: an invalid/absent credential is
    ``401``. A successful sign-in mints an access token scoped to the one owner PLUS a
    REAL rotating ``refresh_token`` starting a fresh refresh-token family (stored
    hash-only on the agent-state store, ARCH-R13/AUTH-R9).
    """
    _verify_owner_secret(settings, body.owner_secret)
    tokens = issue_access_token(settings, subject=OWNER_SUBJECT, scopes=OWNER_SCOPES)
    refresh = await issue_refresh_token(
        session,
        subject=OWNER_SUBJECT,
        scopes=tuple(s.value for s in OWNER_SCOPES),
        ttl_seconds=settings.auth__refresh_ttl_seconds,
    )
    # LOG-R6.2: authentication events ride the tamper-evident audit stream.
    audit_event("auth_token_issued", athlete_id=OWNER_SUBJECT)
    payload = tokens.to_dict()
    payload["refresh_token"] = refresh
    return payload


@router.post("/refresh", operation_id="refreshToken")
async def refresh_token(
    body: RefreshRequest, settings: AppSettings, session: StateSession
) -> dict[str, object]:
    """Rotate a valid single-use refresh token for fresh ``AuthTokens`` (API-R23).

    Not public — the presented refresh token IS the credential. The presented token is
    single-use and revoked on use (rotation); REPLAY of a consumed token yields ``401``
    AND revokes the entire family (reuse detection). Unknown/expired -> ``401`` with no
    detail distinguishing the cases (AUTH-R9).
    """
    outcome = await consume_refresh_token(
        session, presented=body.refresh_token, ttl_seconds=settings.auth__refresh_ttl_seconds
    )
    if outcome.status != "ok" or outcome.subject is None or outcome.new_secret is None:
        # LOG-R6.2: a replayed (reuse-detected) refresh revokes its whole family —
        # an auditable authentication event; a plain invalid token is audited too.
        audit_event("auth_refresh_rejected", reason=outcome.status)
        raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE)
    scopes = tuple(Scope(value) for value in outcome.scopes)
    # Re-mint the family's client claim (AUTH-R8a): a delegated (bot-link) family
    # stays ``client: delegated`` across EVERY rotation, or the X-Service-Auth
    # factor enforcement keyed on that claim would silently lapse after the first
    # refresh.
    tokens = issue_access_token(
        settings, subject=outcome.subject, scopes=scopes, client=outcome.client
    )
    audit_event("auth_token_refreshed", athlete_id=outcome.subject)
    payload = tokens.to_dict()
    payload["refresh_token"] = outcome.new_secret
    return payload


@router.post("/revoke", operation_id="revokeToken", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_token(body: RefreshRequest, session: StateSession) -> None:
    """Revoke the presented refresh token AND its whole family -> ``204`` (API-R23).

    Not public — the presented refresh token is the credential. An unknown token is
    ``401`` (no fail-open acknowledgement of a forged credential, AUTH-R9).
    """
    row = await find_refresh_token(session, presented=body.refresh_token)
    if row is None:
        raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE)
    await revoke_family(session, family_id=row.family_id)
    audit_event("auth_family_revoked", athlete_id=row.subject)  # LOG-R6.2


@router.post("/link/start", response_model=LinkChallenge, operation_id="startAccountLink")
async def link_start(settings: AppSettings, session: StateSession) -> LinkChallenge:
    """Begin account-linking: mint a short-lived single-use ``LinkChallenge`` (AUTH-R8).

    Public (pre-token, AUTH-R10): an external client (e.g. the bot) requests a code and
    shows it to the athlete; the code is NOT itself a credential — it mints nothing
    until the athlete proves control of the WattWise account by approving it in-app.
    The lifetime is config-loaded (CFG-R1a).
    """
    code, expires = await start_link_challenge(session, ttl_seconds=settings.auth__link_ttl_seconds)
    return LinkChallenge(link_code=code, expires_at=expires.isoformat())


@router.post("/link/approve", response_model=LinkApproval, operation_id="approveAccountLink")
async def link_approve(
    body: LinkCodeRequest,
    session: StateSession,
    principal: Annotated[Principal, Depends(authenticate)],
) -> LinkApproval:
    """Prove control of the WattWise account by approving a pending link code (AUTH-R8).

    Bearer-authenticated: the verified owner session IS the proof of account control the
    link flow requires; approving binds the challenge to the server-derived subject
    (AUTH-R3). An unknown/expired/already-handled code -> ``409 conflict`` (no detail
    distinguishing the cases beyond the typed slug, AUTH-R9).
    """
    approved = await approve_link_challenge(
        session, link_code=body.link_code, subject=principal.athlete_id
    )
    if not approved:
        raise ProblemError("conflict")
    return LinkApproval()


@router.post("/link/complete", operation_id="completeAccountLink")
async def link_complete(
    body: LinkCodeRequest, settings: AppSettings, session: StateSession
) -> dict[str, object]:
    """Redeem a PROVEN link code for the per-athlete delegated ``AuthTokens`` (AUTH-R8).

    Public (pre-token, AUTH-R10): the external client redeems the code the athlete
    approved, minting delegated tokens bound to exactly that one athlete/owner — with a
    rotating refresh family of their own. An expired/used/unproven code -> ``409``; a
    forged/unknown code -> ``401`` (API-R23).
    """
    redemption = await complete_link_challenge(session, link_code=body.link_code)
    if redemption.status == "invalid":
        raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE)
    if redemption.status != "ok" or redemption.subject is None:
        raise ProblemError("conflict")
    tokens = issue_access_token(
        settings, subject=redemption.subject, scopes=DELEGATED_SCOPES, client=DELEGATED_CLIENT
    )
    refresh = await issue_refresh_token(
        session,
        subject=redemption.subject,
        scopes=tuple(s.value for s in DELEGATED_SCOPES),
        ttl_seconds=settings.auth__refresh_ttl_seconds,
        client=DELEGATED_CLIENT,
    )
    payload = tokens.to_dict()
    payload["refresh_token"] = refresh
    return payload


__all__ = [
    "DELEGATED_SCOPES",
    "OWNER_SCOPES",
    "LinkChallenge",
    "router",
]
