"""agent durable memory store (memory_item).

Creates the dedicated agent-state ``memory_item`` table (doc 50 MEM-R1..R5): durable,
athlete-scoped, ground-truth-preserving memory of goals/constraints/preferences/
episodes. The ``kind`` column is the closed ``memory_item_kind`` enum (MEM-R5),
stored portably as text + CHECK (NOT a native PG ENUM), so the same revision runs
unchanged on SQLite / PostgreSQL / MariaDB (GBO-R8b, only the DSN differs). There is
deliberately NO numeric column: the store cannot hold a canonical analytic value
(MEM-R1) — those are always read LIVE from the analytics service (doc 40).

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

# Closed memory_item_kind enum (MEM-R5); stored as text + CHECK for portability.
_MEMORY_ITEM_KIND = sa.Enum(
    "goal",
    "constraint",
    "load_response",
    "preference",
    "language",
    "plan_history",
    name="memoryitemkind",
    native_enum=False,
    length=64,
)


def upgrade() -> None:
    op.create_table(
        "memory_item",
        sa.Column("memory_item_id", sa.Uuid(), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("kind", _MEMORY_ITEM_KIND, nullable=False),
        sa.Column("content", sa.String(length=2048), nullable=False),
        sa.Column("inferred", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["athlete_id"],
            ["athlete.athlete_id"],
            name=op.f("fk_memory_item_athlete_id_athlete"),
        ),
        sa.PrimaryKeyConstraint("memory_item_id", name=op.f("pk_memory_item")),
    )
    with op.batch_alter_table("memory_item", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_memory_item_athlete_id"), ["athlete_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("memory_item", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_memory_item_athlete_id"))
    op.drop_table("memory_item")
