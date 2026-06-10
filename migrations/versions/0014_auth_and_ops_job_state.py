"""auth-flow + import/export job operational state (API-R23 / API-R33 / API-R34).

Creates four AGENT-STATE tables (amended ARCH-R13 — operational API state lives on the
dedicated agent-state store, never the canonical GBO schema):

- ``agent_auth_refresh_token`` — rotating refresh-token families, hash-only at rest
  (API-R23 / AUTH-R9): single-use rows grouped by ``family_id`` so replay of a consumed
  token can revoke the whole family (reuse detection).
- ``agent_auth_link_challenge`` — short-lived, single-use account-link challenges
  (AUTH-R8): ``code_hash`` only, with the proof-of-control / redemption markers.
- ``agent_import_job`` — the upload-job bookkeeping rows backing the paginated
  ``GET /v1/imports`` read surface (API-R33); never the landed canonical activity.
- ``agent_export_job`` — export jobs + the one-time nonce that seeds the single-use,
  owner-bound signed download URL (API-R34).

All columns are the portable types the column factories produce (GBO-R8b): the same
revision runs unchanged on SQLite / PostgreSQL / MariaDB — only the DSN differs.

Revision ID: 0013
Revises: 0012
Create Date: 2026-06-10 00:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from wattwise_core.persistence.types import UtcDateTime

# revision identifiers, used by Alembic.
revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the auth-flow + import/export job agent-state tables (ARCH-R13)."""
    op.create_table(
        "agent_auth_refresh_token",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("family_id", sa.String(length=36), nullable=False),
        sa.Column("subject", sa.String(length=64), nullable=False),
        sa.Column("scopes", sa.String(length=256), nullable=False),
        sa.Column("client", sa.String(length=32), nullable=True),
        sa.Column("expires_at", UtcDateTime(), nullable=False),
        sa.Column("used_at", UtcDateTime(), nullable=True),
        sa.Column("revoked_at", UtcDateTime(), nullable=True),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_agent_auth_refresh_token_hash"),
    )
    op.create_index(
        op.f("ix_agent_auth_refresh_token_family_id"),
        "agent_auth_refresh_token",
        ["family_id"],
        unique=False,
    )
    op.create_table(
        "agent_auth_link_challenge",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("code_hash", sa.String(length=64), nullable=False),
        sa.Column("subject", sa.String(length=64), nullable=True),
        sa.Column("proven_at", UtcDateTime(), nullable=True),
        sa.Column("used_at", UtcDateTime(), nullable=True),
        sa.Column("expires_at", UtcDateTime(), nullable=False),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code_hash", name="uq_agent_auth_link_code_hash"),
    )
    op.create_table(
        "agent_import_job",
        sa.Column("import_job_id", sa.String(length=64), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("filename", sa.String(length=256), nullable=True),
        sa.Column("status_text", sa.String(length=256), nullable=False),
        sa.Column("received_at", UtcDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("import_job_id"),
    )
    op.create_index(
        op.f("ix_agent_import_job_athlete_id"), "agent_import_job", ["athlete_id"], unique=False
    )
    op.create_table(
        "agent_export_job",
        sa.Column("export_job_id", sa.Uuid(), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("format", sa.String(length=8), nullable=False),
        sa.Column("from_date", sa.String(length=10), nullable=True),
        sa.Column("to_date", sa.String(length=10), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("nonce", sa.String(length=64), nullable=False),
        sa.Column("nonce_used_at", UtcDateTime(), nullable=True),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("export_job_id"),
    )
    op.create_index(
        op.f("ix_agent_export_job_athlete_id"), "agent_export_job", ["athlete_id"], unique=False
    )


def downgrade() -> None:
    """Drop the auth-flow + import/export job agent-state tables."""
    op.drop_index(op.f("ix_agent_export_job_athlete_id"), table_name="agent_export_job")
    op.drop_table("agent_export_job")
    op.drop_index(op.f("ix_agent_import_job_athlete_id"), table_name="agent_import_job")
    op.drop_table("agent_import_job")
    op.drop_table("agent_auth_link_challenge")
    op.drop_index(
        op.f("ix_agent_auth_refresh_token_family_id"), table_name="agent_auth_refresh_token"
    )
    op.drop_table("agent_auth_refresh_token")


__all__ = ["downgrade", "upgrade"]
