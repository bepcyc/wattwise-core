"""agent weekly-review history store (agent_digest_record, API-R14).

Creates the agent-state ``agent_digest_record`` table backing the paginated weekly-review
history surface (``GET /v1/agent/digest/list``, doc 60 API-R14): one stored grounded
review per ``(athlete, week_end)``, upserted when a review is generated and replayed
VERBATIM on read (GROUND-R7 — never recomputed). The table lives on the AGENT-STATE store
(``AgentStateBase``) per ARCH-R13 — operational deliverable state is NEVER canonical GBO
master data: ``athlete_id`` is an agent-state-side scope column, NOT a foreign key into
the canonical ``athlete`` table. The deliverable body is portable ``JSON`` (text-backed on
SQLite/MariaDB, JSON on PostgreSQL) so the same revision runs unchanged on all three
backends (GBO-R8b, only the DSN differs).

Revision ID: 0012
Revises: 0011
Create Date: 2026-06-10 00:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from wattwise_core.persistence.types import UtcDateTime

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "agent_digest_record"


def upgrade() -> None:
    """Create the agent-state weekly-review history table (API-R14 / ARCH-R13)."""
    op.create_table(
        _TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("week_end", sa.String(length=10), nullable=False),
        sa.Column("body", sa.JSON(), nullable=False),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("athlete_id", "week_end", name="uq_agent_digest_athlete_week"),
    )
    op.create_index(
        op.f("ix_agent_digest_record_athlete_id"), _TABLE, ["athlete_id"], unique=False
    )


def downgrade() -> None:
    """Drop the weekly-review history table."""
    op.drop_index(op.f("ix_agent_digest_record_athlete_id"), table_name=_TABLE)
    op.drop_table(_TABLE)
