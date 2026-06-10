"""First-party revocable refresh-token store (SEC-R2.3).

Creates ``auth_refresh_token``: one row per opaque refresh credential, holding only
the SHA-256 ``token_hash`` (never the secret, SEC-R12), the server-derived subject's
``athlete_id``, the rotation ``family_id`` (reuse detection revokes the whole family),
the granted scopes, a finite ``expires_at``, and the revocation flag.

PORTABLE (GBO-R8b): only ``sa.Uuid`` / ``sa.String`` / ``sa.Boolean`` / ``sa.DateTime``
primitives; runs unchanged on SQLite / PostgreSQL / MariaDB.

Revision ID: 0012
Revises: 0011
Create Date: 2026-06-10 15:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    """Create the ``auth_refresh_token`` table (SEC-R2.3)."""
    op.create_table(
        "auth_refresh_token",
        sa.Column("refresh_token_id", sa.Uuid(), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("family_id", sa.Uuid(), nullable=False),
        sa.Column("scopes", sa.String(length=256), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("refresh_token_id", name=op.f("pk_auth_refresh_token")),
        sa.UniqueConstraint("token_hash", name=op.f("uq_auth_refresh_token_token_hash")),
    )
    op.create_index(op.f("ix_auth_refresh_token_athlete_id"), "auth_refresh_token", ["athlete_id"])
    op.create_index("ix_auth_refresh_token_family", "auth_refresh_token", ["family_id"])


def downgrade() -> None:
    """Drop the ``auth_refresh_token`` table."""
    op.drop_index("ix_auth_refresh_token_family", table_name="auth_refresh_token")
    op.drop_index(op.f("ix_auth_refresh_token_athlete_id"), table_name="auth_refresh_token")
    op.drop_table("auth_refresh_token")
