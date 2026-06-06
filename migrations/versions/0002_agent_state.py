"""agent-state store schema (checkpoints / threads / writes).

Adds the dedicated **agent-state** tables that hold the agent graph's durable state —
checkpoints, threads, and pending intermediate writes (doc 50 §4 CKPT-R1/-R2/-R3/-R7,
doc 10 ARCH-R13). These tables are the agent orchestrator's store and are NEVER part of
the canonical GBO schema; they map the ORM models in
:mod:`wattwise_core.agent.state_store`, which carry their own metadata so canonical and
agent state cannot share a schema or write credential (ARCH-R13, DEPLOY-R4).

PORTABLE (ARCH-R13 / GBO-R8b): every column uses only the portable primitives the
canonical layer uses — ``sa.Uuid``, ``sa.DateTime(timezone=True)``, portable ``sa.JSON``,
``sa.LargeBinary`` (BLOB / BYTEA / LONGBLOB), ``sa.String`` / ``sa.Integer`` — so this
revision runs unchanged on SQLite / PostgreSQL / MariaDB (DSN-only difference). Batch mode
keeps index creation portable on SQLite without code change.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-06 16:30:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_thread",
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("thread_id", name=op.f("pk_agent_thread")),
        sa.UniqueConstraint(
            "athlete_id",
            "conversation_id",
            name="uq_agent_thread_athlete_conversation",
        ),
    )
    with op.batch_alter_table("agent_thread", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_agent_thread_athlete_id"), ["athlete_id"], unique=False
        )

    op.create_table(
        "agent_checkpoint",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("checkpoint_ns", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=128), nullable=False),
        sa.Column("parent_checkpoint_id", sa.String(length=128), nullable=True),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("checkpoint_type", sa.String(length=64), nullable=False),
        sa.Column("checkpoint_blob", sa.LargeBinary(), nullable=False),
        sa.Column("metadata_blob", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["agent_thread.thread_id"],
            name=op.f("fk_agent_checkpoint_thread_id_agent_thread"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_checkpoint")),
        sa.UniqueConstraint(
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            name="uq_agent_checkpoint_thread_ns_id",
        ),
    )
    with op.batch_alter_table("agent_checkpoint", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_agent_checkpoint_thread_id"), ["thread_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_agent_checkpoint_athlete_id"), ["athlete_id"], unique=False
        )
        batch_op.create_index(
            "ix_agent_checkpoint_thread_ns_created",
            ["thread_id", "checkpoint_ns", "created_at"],
            unique=False,
        )

    op.create_table(
        "agent_write",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("checkpoint_ns", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_id", sa.String(length=128), nullable=False),
        sa.Column("task_id", sa.String(length=128), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("channel", sa.String(length=255), nullable=False),
        sa.Column("value_type", sa.String(length=64), nullable=False),
        sa.Column("value_blob", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["agent_thread.thread_id"],
            name=op.f("fk_agent_write_thread_id_agent_thread"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_write")),
        sa.UniqueConstraint(
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            name="uq_agent_write_identity",
        ),
    )
    with op.batch_alter_table("agent_write", schema=None) as batch_op:
        batch_op.create_index(
            "ix_agent_write_checkpoint",
            ["thread_id", "checkpoint_ns", "checkpoint_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("agent_write", schema=None) as batch_op:
        batch_op.drop_index("ix_agent_write_checkpoint")
    op.drop_table("agent_write")

    with op.batch_alter_table("agent_checkpoint", schema=None) as batch_op:
        batch_op.drop_index("ix_agent_checkpoint_thread_ns_created")
        batch_op.drop_index(batch_op.f("ix_agent_checkpoint_athlete_id"))
        batch_op.drop_index(batch_op.f("ix_agent_checkpoint_thread_id"))
    op.drop_table("agent_checkpoint")

    with op.batch_alter_table("agent_thread", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_agent_thread_athlete_id"))
    op.drop_table("agent_thread")
