"""First-party refresh-token store (SEC-R2.3).

SEC-R2.3 requires refresh via a SEPARATE, revocable token — never a long-lived access
token. Each row is ONE opaque refresh credential: the relational store holds only the
SHA-256 ``token_hash`` of the opaque secret (never the secret itself, SEC-R12), the
server-derived subject it was minted for (AUTH-R3), its granted scopes, a finite
``expires_at``, and the revocation state. Rotation is reuse-detecting: every mint in a
refresh chain shares a ``family_id``; presenting an already-rotated (revoked) member
revokes the WHOLE family, so a stolen-then-replayed refresh token kills the chain
fail-closed instead of silently coexisting with the attacker's copy.

``athlete_id`` is the owning athlete (plain UUID, no FK — a token may be minted before
the athlete profile row exists), so the per-athlete erasure executor's athlete-scoped
table scan covers this store too (PRIV-R8: erasure removes the athlete's credentials).
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import Boolean, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.types import pk_column, timestamptz_column


class AuthRefreshToken(Base, TimestampMixin):
    """ONE revocable opaque refresh credential (SEC-R2.3).

    ``token_hash`` (SHA-256 hex of the opaque secret) is the lookup key; the secret
    itself is never persisted. ``family_id`` links every rotation of one sign-in so
    reuse of a rotated member revokes the whole family (reuse detection).
    """

    __tablename__ = "auth_refresh_token"
    __table_args__ = (Index("ix_auth_refresh_token_family", "family_id"),)

    refresh_token_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    family_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    # Space-delimited granted scope tokens (the closed AUTH-R7 vocabulary).
    scopes: Mapped[str] = mapped_column(String(256), nullable=False)
    expires_at: Mapped[_dt.datetime] = timestamptz_column(nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


__all__ = ["AuthRefreshToken"]
