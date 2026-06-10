"""Refresh-token families + account-link challenges on the agent-state store (API-R23).

OPERATIONAL auth state per the amended ARCH-R13: refresh-token family rows and the
account-link challenge state are operational API state and live on the dedicated
agent-state store (:class:`AgentStateBase`) — NEVER the canonical GBO store, accessed
through the agent-state role.

Security shape (API-R23 / AUTH-R8):

- A refresh token is an opaque high-entropy secret; only its SHA-256 HASH is stored
  (a DB leak never yields a usable credential). Tokens ROTATE: each ``/refresh`` marks
  the presented row used and mints a new row in the SAME ``family_id``; replay of a
  consumed token revokes the WHOLE family (reuse detection).
- A link challenge mints a short single-use ``link_code`` (NOT itself a credential);
  the athlete proves control of the WattWise account by APPROVING it over an
  authenticated session, after which ``/link/complete`` may redeem it exactly once.

Requirement IDs: API-R23, AUTH-R8, ARCH-R13, AUTH-R9 (no secret material persisted).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import secrets
import uuid
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import String, UniqueConstraint, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.agent.state_store import AGENT_STATE_PREFIX, AgentStateBase
from wattwise_core.persistence.types import UtcDateTime, utcnow, uuid7


class AuthRefreshToken(AgentStateBase):
    """One rotating refresh-token row (API-R23): hash-only, single-use, family-bound.

    ``token_hash`` is the SHA-256 of the opaque secret (never the secret itself,
    AUTH-R9). ``family_id`` groups every rotation of one sign-in; reuse detection
    revokes the family. ``used_at``/``revoked_at`` are terminal markers — a row with
    either set can never mint again (fail-closed).
    """

    __tablename__ = AGENT_STATE_PREFIX + "auth_refresh_token"
    __table_args__ = (UniqueConstraint("token_hash", name="uq_agent_auth_refresh_token_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    family_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(64), nullable=False)
    scopes: Mapped[str] = mapped_column(String(256), nullable=False)
    expires_at: Mapped[_dt.datetime] = mapped_column(UtcDateTime(), nullable=False)
    used_at: Mapped[_dt.datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    revoked_at: Mapped[_dt.datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )


class AuthLinkChallenge(AgentStateBase):
    """One account-link challenge (AUTH-R8): short-lived, single-use, hash-only.

    ``code_hash`` is the SHA-256 of the ``link_code`` (the code itself is never stored).
    ``proven_at`` is set when the AUTHENTICATED owner approves the code in-app (proof of
    WattWise-account control); only a proven, unexpired, unused challenge can be
    redeemed, exactly once (``used_at``).
    """

    __tablename__ = AGENT_STATE_PREFIX + "auth_link_challenge"
    __table_args__ = (UniqueConstraint("code_hash", name="uq_agent_auth_link_code_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    subject: Mapped[str | None] = mapped_column(String(64), nullable=True)
    proven_at: Mapped[_dt.datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    used_at: Mapped[_dt.datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    expires_at: Mapped[_dt.datetime] = mapped_column(UtcDateTime(), nullable=False)
    created_at: Mapped[_dt.datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )


def hash_token(token: str) -> str:
    """The stored SHA-256 hex digest of an opaque credential (hash-only at rest)."""
    return hashlib.sha256(token.encode()).hexdigest()


def mint_refresh_secret() -> str:
    """A fresh high-entropy opaque refresh-token secret (256 bits, URL-safe)."""
    return secrets.token_urlsafe(32)


def mint_link_code() -> str:
    """A fresh short single-use link code (NOT itself a credential, API-R23)."""
    return secrets.token_urlsafe(9)


@dataclass(frozen=True, slots=True)
class RefreshOutcome:
    """The classified result of presenting a refresh token (API-R23)."""

    status: str  # "ok" | "reuse" | "invalid"
    subject: str | None = None
    scopes: tuple[str, ...] = ()
    new_secret: str | None = None


async def issue_refresh_token(
    session: AsyncSession,
    *,
    subject: str,
    scopes: tuple[str, ...],
    ttl_seconds: int,
    family_id: str | None = None,
) -> str:
    """Mint + persist a refresh-token row; returns the OPAQUE secret (stored hash-only).

    A fresh sign-in starts a NEW family; a rotation passes the existing ``family_id``.
    """
    secret = mint_refresh_secret()
    session.add(
        AuthRefreshToken(
            token_hash=hash_token(secret),
            family_id=family_id or str(uuid.uuid4()),
            subject=subject,
            scopes=" ".join(scopes),
            expires_at=utcnow() + _dt.timedelta(seconds=ttl_seconds),
        )
    )
    await session.flush()
    return secret


async def consume_refresh_token(
    session: AsyncSession, *, presented: str, ttl_seconds: int
) -> RefreshOutcome:
    """Rotate a presented refresh token per API-R23 — single-use + family reuse-detection.

    Valid + unused + unexpired -> mark used, mint the successor in the SAME family and
    return ``ok`` with the new secret. A CONSUMED/REVOKED token -> revoke the ENTIRE
    family and return ``reuse`` (replay detection). Unknown/expired -> ``invalid``.
    The used-marking is an atomic guarded UPDATE so two concurrent presentations of the
    same token can never both rotate (the loser reads as reuse).
    """
    row = (
        await session.execute(
            select(AuthRefreshToken).where(AuthRefreshToken.token_hash == hash_token(presented))
        )
    ).scalar_one_or_none()
    if row is None or row.expires_at < utcnow():
        return RefreshOutcome(status="invalid")
    if row.used_at is not None or row.revoked_at is not None:
        await revoke_family(session, family_id=row.family_id)
        return RefreshOutcome(status="reuse")
    claimed = cast(
        "CursorResult[Any]",
        await session.execute(
            update(AuthRefreshToken)
            .where(AuthRefreshToken.id == row.id, AuthRefreshToken.used_at.is_(None))
            .values(used_at=utcnow())
        ),
    )
    if claimed.rowcount != 1:  # a concurrent presentation won the race -> replay
        await revoke_family(session, family_id=row.family_id)
        return RefreshOutcome(status="reuse")
    scopes = tuple(row.scopes.split())
    successor = await issue_refresh_token(
        session,
        subject=row.subject,
        scopes=scopes,
        ttl_seconds=ttl_seconds,
        family_id=row.family_id,
    )
    return RefreshOutcome(
        status="ok", subject=row.subject, scopes=scopes, new_secret=successor
    )


async def revoke_family(session: AsyncSession, *, family_id: str) -> None:
    """Revoke EVERY token in a refresh-token family (reuse detection / explicit revoke).

    COMMITS immediately: the reuse-detection path raises ``401`` right after revoking,
    and the request-scoped session would otherwise ROLL BACK on that exception — losing
    the very revocation the replay must trigger (fail-closed means the family dies even
    though the request errors).
    """
    await session.execute(
        update(AuthRefreshToken)
        .where(AuthRefreshToken.family_id == family_id, AuthRefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    await session.commit()


async def find_refresh_token(
    session: AsyncSession, *, presented: str
) -> AuthRefreshToken | None:
    """Look up a presented refresh token by its stored hash (revoke path)."""
    return (
        await session.execute(
            select(AuthRefreshToken).where(AuthRefreshToken.token_hash == hash_token(presented))
        )
    ).scalar_one_or_none()


async def start_link_challenge(
    session: AsyncSession, *, ttl_seconds: int
) -> tuple[str, _dt.datetime]:
    """Mint + persist a link challenge; returns ``(link_code, expires_at)`` (AUTH-R8)."""
    code = mint_link_code()
    expires = utcnow() + _dt.timedelta(seconds=ttl_seconds)
    session.add(AuthLinkChallenge(code_hash=hash_token(code), expires_at=expires))
    await session.flush()
    return code, expires


async def approve_link_challenge(
    session: AsyncSession, *, link_code: str, subject: str
) -> bool:
    """Bind a pending challenge to the AUTHENTICATED owner (proof of control, AUTH-R8).

    Only a pending (unproven, unused, unexpired) challenge can be approved; the guarded
    UPDATE makes approval idempotent-safe and race-safe (rowcount decides).
    """
    result = cast(
        "CursorResult[Any]",
        await session.execute(
            update(AuthLinkChallenge)
            .where(
                AuthLinkChallenge.code_hash == hash_token(link_code),
                AuthLinkChallenge.proven_at.is_(None),
                AuthLinkChallenge.used_at.is_(None),
                AuthLinkChallenge.expires_at >= utcnow(),
            )
            .values(proven_at=utcnow(), subject=subject)
        ),
    )
    await session.flush()
    return result.rowcount == 1


@dataclass(frozen=True, slots=True)
class LinkRedemption:
    """The classified result of redeeming a link code (API-R23)."""

    status: str  # "ok" | "conflict" | "invalid"
    subject: str | None = None


async def complete_link_challenge(
    session: AsyncSession, *, link_code: str
) -> LinkRedemption:
    """Redeem a PROVEN link code exactly once (API-R23 ``/link/complete``).

    Proven + unused + unexpired -> mark used, return the bound subject. A known but
    used/expired/unproven code -> ``conflict`` (409); an unknown/forged code ->
    ``invalid`` (401). The single-use marking is an atomic guarded UPDATE.
    """
    row = (
        await session.execute(
            select(AuthLinkChallenge).where(
                AuthLinkChallenge.code_hash == hash_token(link_code)
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return LinkRedemption(status="invalid")
    claimed = cast(
        "CursorResult[Any]",
        await session.execute(
            update(AuthLinkChallenge)
            .where(
                AuthLinkChallenge.id == row.id,
                AuthLinkChallenge.proven_at.is_not(None),
                AuthLinkChallenge.used_at.is_(None),
                AuthLinkChallenge.expires_at >= utcnow(),
            )
            .values(used_at=utcnow())
        ),
    )
    await session.flush()
    if claimed.rowcount != 1:
        return LinkRedemption(status="conflict")
    return LinkRedemption(status="ok", subject=row.subject)


__all__ = [
    "AuthLinkChallenge",
    "AuthRefreshToken",
    "LinkRedemption",
    "RefreshOutcome",
    "approve_link_challenge",
    "complete_link_challenge",
    "consume_refresh_token",
    "find_refresh_token",
    "hash_token",
    "issue_refresh_token",
    "mint_link_code",
    "mint_refresh_secret",
    "revoke_family",
    "start_link_challenge",
]
