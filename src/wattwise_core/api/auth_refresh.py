"""Separate, revocable refresh-token issuance / rotation / revocation (SEC-R2.3).

SEC-R2.3: the access token is short-lived (config-bounded ≤ 60 minutes, validated at
config load) and refresh happens via a SEPARATE, revocable opaque token. This module
owns that refresh leg:

- :func:`mint_refresh_token` — mint one opaque high-entropy refresh credential; the
  store keeps only its SHA-256 hash (never the secret, SEC-R12), the server-derived
  subject, the granted scopes, a finite expiry, and the rotation ``family_id``.
- ``POST /v1/auth/refresh`` — exchange a valid refresh token for a NEW access token +
  a ROTATED refresh token. Rotation is reuse-detecting: presenting an already-rotated
  (revoked) member revokes the WHOLE family fail-closed (a replayed stolen token kills
  the chain) and yields ``401``.
- ``POST /v1/auth/revoke`` — revoke the presented token's whole family (sign-out).

Every issuance/refresh/revocation/reuse event is recorded on the tamper-evident audit
stream (LOG-R6.2: authentication events). Identity is always the SUBJECT PERSISTED at
mint time — server-derived (AUTH-R3), never read from the request beyond the opaque
credential itself.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid as _uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.auth import AuthTokens, Scope, issue_access_token
from wattwise_core.api.errors import ProblemError
from wattwise_core.config import Settings
from wattwise_core.observability.audit import audit_event
from wattwise_core.persistence import Database
from wattwise_core.persistence.models.auth import AuthRefreshToken
from wattwise_core.persistence.types import uuid7
from wattwise_core.seams import SYSTEM_SUBJECT, EngineSessionProvider

#: The WWW-Authenticate challenge returned with every ``401`` (AUTH-R1).
_BEARER_CHALLENGE: Final = {"WWW-Authenticate": "Bearer"}

#: Bytes of entropy in the opaque refresh secret (well above the 256-bit floor).
_REFRESH_SECRET_BYTES: Final = 32


def _hash(token: str) -> str:
    """SHA-256 hex of the opaque refresh secret — the only form ever persisted (SEC-R12)."""
    return hashlib.sha256(token.encode()).hexdigest()


async def mint_refresh_token(
    session: AsyncSession,
    settings: Settings,
    *,
    subject: str,
    scopes: tuple[str, ...],
    family_id: Any | None = None,
) -> str:
    """Mint + persist ONE opaque refresh credential; return the secret (SEC-R2.3).

    The opaque secret is returned to the caller exactly once; the store keeps only its
    hash. ``family_id`` chains a rotation to its sign-in family (a fresh sign-in starts
    a new family). The expiry window is loaded config (``auth__refresh_ttl_days``).
    """
    opaque = secrets.token_urlsafe(_REFRESH_SECRET_BYTES)
    row = AuthRefreshToken(
        refresh_token_id=uuid7(),
        athlete_id=_subject_uuid(subject),
        token_hash=_hash(opaque),
        family_id=family_id if family_id is not None else uuid7(),
        scopes=" ".join(scopes),
        expires_at=datetime.now(UTC) + timedelta(days=settings.auth__refresh_ttl_days),
        revoked=False,
    )
    session.add(row)
    await session.flush()
    return opaque


def _subject_uuid(subject: str) -> Any:
    """Coerce the server-derived subject to the stored UUID, failing closed."""
    try:
        return _uuid.UUID(subject)
    except ValueError as exc:  # a non-UUID subject is an assembly error, never a hint
        raise ProblemError("internal-error") from exc


async def _revoke_family(session: AsyncSession, family_id: Any) -> None:
    """Revoke EVERY member of a rotation family (reuse detection / sign-out)."""
    await session.execute(
        update(AuthRefreshToken).where(AuthRefreshToken.family_id == family_id).values(revoked=True)
    )


async def rotate_refresh_token(
    session: AsyncSession, settings: Settings, presented: str
) -> AuthTokens | None:
    """Exchange a valid refresh token for new access + ROTATED refresh tokens.

    Fail-closed paths return ``None`` (the caller answers ``401``): an unknown token;
    an expired token; and a REVOKED token (already rotated or signed out) — reuse
    detection, which ALSO revokes the WHOLE family. The revocation is a state write the
    caller's session COMMITS on normal exit (returning ``None`` instead of raising keeps
    the commit), so a replayed stolen token durably kills the chain. On success the
    presented member is revoked and a new member of the SAME family is minted, so at
    most one live token per family.
    """
    row = await _lookup(session, presented)
    if row is None:
        return None
    now = datetime.now(UTC)
    if row.revoked:
        await _revoke_family(session, row.family_id)
        audit_event("auth_refresh_reuse_detected", athlete_id=str(row.athlete_id))
        return None
    if _as_utc(row.expires_at) <= now:
        return None
    row.revoked = True
    scopes = tuple(row.scopes.split())
    new_refresh = await mint_refresh_token(
        session,
        settings,
        subject=str(row.athlete_id),
        scopes=scopes,
        family_id=row.family_id,
    )
    access = issue_access_token(
        settings,
        subject=str(row.athlete_id),
        scopes=[Scope(s) for s in scopes],
        ttl_seconds=settings.auth__access_ttl_seconds,
    )
    audit_event("auth_token_refreshed", athlete_id=str(row.athlete_id))
    return AuthTokens(
        access_token=access.access_token,
        refresh_token=new_refresh,
        expires_in=access.expires_in,
        scopes=access.scopes,
    )


async def revoke_refresh_token(session: AsyncSession, presented: str) -> bool:
    """Revoke the presented token's WHOLE family (sign-out; SEC-R2.3 revocability).

    Returns ``False`` for an unknown credential (the caller answers ``401`` without
    distinguishing unknown from revoked — AUTH-R9: no oracle).
    """
    row = await _lookup(session, presented)
    if row is None:
        return False
    await _revoke_family(session, row.family_id)
    audit_event("auth_token_revoked", athlete_id=str(row.athlete_id))
    return True


async def _lookup(session: AsyncSession, presented: str) -> AuthRefreshToken | None:
    """Resolve the presented opaque secret to its stored row (``None`` if unknown)."""
    return (
        await session.execute(
            select(AuthRefreshToken).where(AuthRefreshToken.token_hash == _hash(presented))
        )
    ).scalar_one_or_none()


def _as_utc(value: datetime) -> datetime:
    """Coerce a possibly-naive stored timestamp to aware UTC for comparison."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class _RefreshRequest(BaseModel):
    """The refresh/revoke exchange body: ONLY the opaque credential (AUTH-R3)."""

    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(min_length=16, max_length=512)


def build_refresh_router(prefix: str) -> APIRouter:
    """Build the public refresh/revoke router mounted under ``{prefix}/auth`` (SEC-R2.3).

    Pre-token endpoints (AUTH-R10): the presented opaque refresh credential IS the
    verified secret — identity comes only from its persisted, server-derived subject.
    The lookup transaction opens through the ONE engine-owned session-provider seam
    with the request-less ``SYSTEM_SUBJECT`` (like the readiness probe): the acting
    subject is not known until the credential resolves.
    """
    router = APIRouter(prefix=f"{prefix}/auth", tags=["auth"])

    @router.post("/refresh", operation_id="refreshToken")
    async def refresh(body: _RefreshRequest, request: Request) -> dict[str, object]:
        """Rotate a valid refresh token into new access + refresh tokens (SEC-R2.3).

        The ``401`` is raised AFTER the session commits so a reuse-detection family
        revocation is durably persisted even though the request itself is refused.
        """
        async with _session(request) as session:
            tokens = await rotate_refresh_token(session, _settings(request), body.refresh_token)
        if tokens is None:
            raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE)
        return tokens.to_dict()

    @router.post("/revoke", operation_id="revokeToken", status_code=204)
    async def revoke(body: _RefreshRequest, request: Request) -> None:
        """Revoke the presented refresh token's whole family (SEC-R2.3)."""
        async with _session(request) as session:
            known = await revoke_refresh_token(session, body.refresh_token)
        if not known:
            raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE)

    return router


def _session(request: Request) -> Any:
    """A SYSTEM-subject canonical session context for the pre-token exchange."""
    database = getattr(request.app.state, "database", None)
    if not isinstance(database, Database):
        raise ProblemError("internal-error")
    return EngineSessionProvider(database).session(subject=SYSTEM_SUBJECT)


def _settings(request: Request) -> Settings:
    """The resolved settings bound to app state at startup (fail-closed)."""
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise ProblemError("internal-error")
    return settings


__all__ = [
    "build_refresh_router",
    "mint_refresh_token",
    "revoke_refresh_token",
    "rotate_refresh_token",
]
