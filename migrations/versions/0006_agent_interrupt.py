"""agent-interrupt ledger (HITL approval-gate live/consumed rows, CKPT-R9 / D-P2).

Adds the dedicated **agent-interrupt** ledger table backing durable human-in-the-loop
approval gates (doc 50 §4 CKPT-R9, doc 10 ARCH-R13). When the graph raises a langgraph
interrupt at an approval-gated plan, the interrupt-gate persists a ``live`` row here; a
``POST …/decision`` then consumes it via an ATOMIC guarded UPDATE
(``SET status='consumed' WHERE thread_id=? AND interrupt_id=? AND athlete_id=? AND
status='live'``) whose ``rowcount`` decides resume-vs-409 (CKPT-R9, fail-closed). Like
``agent_checkpoint`` / ``agent_write`` it is part of the SEPARATE agent-state store and is
NEVER part of the canonical GBO schema (ARCH-R13, DEPLOY-R4); it maps the
:class:`~wattwise_core.agent.state_store.AgentInterrupt` ORM on ``AgentStateBase``. The
``athlete_id`` is duplicated here (as on ``agent_checkpoint``) as defence-in-depth so a row
is independently identity-scoped and joins the per-athlete erasure target set (CKPT-R8 /
PRIV-R8) even when read outside the thread join.

PORTABLE (ARCH-R13 / GBO-R8b): every column uses only the portable primitives the canonical
and agent-state layers already use — ``sa.Uuid``, ``UtcDateTime()``, ``sa.String`` — so this
revision runs unchanged on SQLite / PostgreSQL / MariaDB (DSN-only difference). No
server-side defaults are emitted: ``id`` (uuid7), ``status`` ('live') and ``created_at``
(utcnow) are Python-side ORM defaults, exactly as on ``agent_thread`` / ``agent_write``, so
the migrated schema matches the live ORM and ``alembic check`` finds no drift (BOOT-R3
parity gate). Batch mode keeps index creation portable on SQLite without code change.

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-09 09:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from wattwise_core.persistence.types import UtcDateTime

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "agent_interrupt"


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("interrupt_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["agent_thread.thread_id"],
            name=op.f("fk_agent_interrupt_thread_id_agent_thread"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_interrupt")),
        sa.UniqueConstraint(
            "thread_id",
            "interrupt_id",
            name="uq_agent_interrupt_thread_interrupt",
        ),
    )
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_agent_interrupt_thread_id"), ["thread_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_agent_interrupt_athlete_id"), ["athlete_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_agent_interrupt_athlete_id"))
        batch_op.drop_index(batch_op.f("ix_agent_interrupt_thread_id"))
    op.drop_table(_TABLE)
