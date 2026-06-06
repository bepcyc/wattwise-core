"""agent durable memory store (agent_memory_item).

Creates the dedicated agent-state ``agent_memory_item`` table (doc 50 MEM-R1..R5):
durable, athlete-scoped, ground-truth-preserving memory of goals/constraints/
preferences/episodes. The table lives on the AGENT-STATE store (``AgentStateBase``),
NEVER the canonical GBO store (MEM-R3/ARCH-R13): ``athlete_id`` is an agent-state-side
scope column, NOT a foreign key into the canonical ``athlete`` table, so memory never
joins canonical metadata or shares its write credential. The ``kind`` column is the
closed ``memory_item_kind`` enum (MEM-R5), stored portably as text + CHECK (NOT a native
PG ENUM), so the same revision runs unchanged on SQLite / PostgreSQL / MariaDB (GBO-R8b,
only the DSN differs). There is deliberately NO numeric column: the store cannot hold a
canonical analytic value (MEM-R1) — those are always read LIVE from analytics (doc 40).

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-06 16:20:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "agent_memory_item"

# Closed memory_item_kind enum (MEM-R5); stored as text + CHECK for portability. The
# constraint name matches the project's naming convention (ck_<table>_<column>) so
# ``alembic check`` against AgentStateBase.metadata stays clean.
_MEMORY_ITEM_KIND = sa.Enum(
    "goal",
    "constraint",
    "load_response",
    "preference",
    "language",
    "plan_history",
    name="ck_agent_memory_item_kind",
    native_enum=False,
    length=64,
    create_constraint=True,
)


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("memory_item_id", sa.Uuid(), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("kind", _MEMORY_ITEM_KIND, nullable=False),
        sa.Column("content", sa.String(length=2048), nullable=False),
        sa.Column("inferred", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("memory_item_id", name=op.f("pk_agent_memory_item")),
    )
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_agent_memory_item_athlete_id"), ["athlete_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_agent_memory_item_athlete_id"))
    op.drop_table(_TABLE)
