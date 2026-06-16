"""agent constraint lifecycle columns (severity/status/effective_until on agent_memory_item).

Adds the CONSTRAINT-lifecycle columns to the dedicated agent-state ``agent_memory_item`` table
(doc 50 MEM-R6/MEM-R7, GROUND-R14; ADR 0008): a ``severity`` (HARD|SOFT — the absolute-vs-relative
contraindication ontology that selects veto vs caution at the grounding gate), a ``status``
(ACTIVE|LIFTED|EXPIRED — the return-to-sport lifecycle), and an optional ``effective_until``
self-expiry instant ("no running for 6 months"). All three are NULLABLE and meaningful ONLY on a
CONSTRAINT-kind row (NULL for every other kind); a NULL ``status`` reads as ACTIVE so rows written
before this revision keep gating (backward-compat).

The two enums are stored portably as text + CHECK (``sa.Enum(native_enum=False,
create_constraint=True)``), NOT a PG-native ENUM, so the same revision runs unchanged on SQLite /
PostgreSQL / MariaDB (GBO-R8b, only the DSN differs) — exactly like the ``kind`` enum in 0003. The
constraint names follow the project's ``ck_<table>_<column>`` convention so ``alembic check``
against ``AgentStateBase.metadata`` stays clean.

Revision ID: 0016
Revises: 0015
Create Date: 2026-06-15 12:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from wattwise_core.persistence.types import UtcDateTime
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "agent_memory_item"

# Closed CONSTRAINT-severity / -status enums (GROUND-R14 / MEM-R7); stored as text + CHECK for
# portability (native_enum=False), mirroring the kind enum in 0003. The constraint names match the
# project convention so ``alembic check`` against AgentStateBase.metadata stays clean.
_CONSTRAINT_SEVERITY = sa.Enum(
    "hard",
    "soft",
    name="ck_agent_memory_item_severity",
    native_enum=False,
    length=64,
    create_constraint=True,
)
_CONSTRAINT_STATUS = sa.Enum(
    "active",
    "lifted",
    "expired",
    name="ck_agent_memory_item_status",
    native_enum=False,
    length=64,
    create_constraint=True,
)


def upgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.add_column(sa.Column("severity", _CONSTRAINT_SEVERITY, nullable=True))
        batch_op.add_column(sa.Column("status", _CONSTRAINT_STATUS, nullable=True))
        batch_op.add_column(sa.Column("effective_until", UtcDateTime(), nullable=True))


def downgrade() -> None:
    # The portable copy-and-recreate (SQLite cannot DROP COLUMN inline) reflects the existing
    # constraints — including the two text+CHECK enum constraints on the columns being dropped — so
    # those CHECKs MUST be dropped in the SAME batch, else the recreated table references a column
    # that no longer exists. Naming them explicitly keeps the drop portable across SQLite/PG/MariaDB.
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_constraint("ck_agent_memory_item_severity", type_="check")
        batch_op.drop_constraint("ck_agent_memory_item_status", type_="check")
        batch_op.drop_column("effective_until")
        batch_op.drop_column("status")
        batch_op.drop_column("severity")
