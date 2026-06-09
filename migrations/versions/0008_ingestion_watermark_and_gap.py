"""ingestion watermark + typed gap entities (SYN-R2, ING-GAP-R2/R3).

Creates the two source-derived canonical entities the trustworthy-ingestion cluster
needs, both written ONLY by the Ingestion/Sync service (ARCH-R3 canonical-write
partition):

* ``ingestion_watermark`` — the idempotent incremental cursor per
  ``(athlete_id, source_descriptor_id, gbo_type, stream)`` (SYN-R2): a high-water
  timestamp/cursor PLUS a content hint so changed-but-not-new records are re-fetched.
  Advanced transactionally with the batch it represents (SYN-R3) and honored by discover
  for incremental mode (ADP-R6).
* ``ingestion_gap`` — the first-class typed gap recording a partial failure
  (ING-GAP-R1..R6): the ten-member ``GapReason`` taxonomy (ING-GAP-R3), open/closed
  ``state`` + ``closed_at`` closure timestamp (ING-GAP-R4), the covered time/record range
  (ING-GAP-R5), severity, ingest-run id, first/last-seen, transient/terminal.

PORTABLE (GBO-R8b / BOOT-R3): a plain ``create_table`` emitting only the portable types
the column factories produce — ``sa.Uuid``, ``sa.Enum(native_enum=False,
create_constraint=True)`` (text + named CHECK, NOT a native PG ENUM), ``UtcDateTime()``
— so the SAME revision runs unchanged on SQLite / PostgreSQL / MariaDB (DSN-only
difference). The Enum spellings, the named CHECKs, and the constraint/index names match
the ORM models + naming convention so ``alembic check`` finds no drift (BOOT-R3 parity).

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-09 20:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from wattwise_core.persistence.types import UtcDateTime

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WATERMARK = "ingestion_watermark"
_GAP = "ingestion_gap"

# Enum value sets spelled exactly as enum_column(...) renders them (native_enum=False ->
# VARCHAR + named CHECK on every backend; name = lowercase enum-class name).
_GBO_TYPE = sa.Enum(
    "activity",
    "activity_lap",
    "activity_file",
    "stream_channel",
    "daily_wellness",
    "wellness_stream_set",
    "fitness_signature",
    name="gbotype",
    native_enum=False,
    create_constraint=True,
    length=64,
)
_GAP_REASON = sa.Enum(
    "auth_revoked",
    "needs_reauth",
    "rate_limited",
    "source_unavailable",
    "discovery_incomplete",
    "fetch_failed",
    "schema_mismatch",
    "mapping_field_missing",
    "source_removed",
    "coverage_stale",
    name="gapreason",
    native_enum=False,
    create_constraint=True,
    length=64,
)
_GAP_STATE = sa.Enum(
    "open",
    "closed",
    name="gapstate",
    native_enum=False,
    create_constraint=True,
    length=64,
)
_SEVERITY = sa.Enum(
    "info",
    "warning",
    "critical",
    name="severity",
    native_enum=False,
    create_constraint=True,
    length=64,
)


def upgrade() -> None:
    op.create_table(
        _WATERMARK,
        sa.Column("ingestion_watermark_id", sa.Uuid(), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("source_descriptor_id", sa.Uuid(), nullable=False),
        sa.Column("gbo_type", _GBO_TYPE, nullable=False),
        sa.Column("stream", sa.String(length=64), nullable=False),
        sa.Column("high_water_at", UtcDateTime(), nullable=True),
        sa.Column("cursor", sa.String(length=512), nullable=True),
        sa.Column("content_hint", sa.String(length=128), nullable=True),
        sa.Column("ingest_run_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.Column("updated_at", UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["athlete_id"],
            ["athlete.athlete_id"],
            name=op.f("fk_ingestion_watermark_athlete_id_athlete"),
        ),
        sa.ForeignKeyConstraint(
            ["source_descriptor_id"],
            ["source_descriptor.source_descriptor_id"],
            name=op.f("fk_ingestion_watermark_source_descriptor_id_source_descriptor"),
        ),
        sa.PrimaryKeyConstraint("ingestion_watermark_id", name=op.f("pk_ingestion_watermark")),
        sa.UniqueConstraint(
            "athlete_id",
            "source_descriptor_id",
            "gbo_type",
            "stream",
            name="uq_ingestion_watermark_scope",
        ),
    )
    with op.batch_alter_table(_WATERMARK, schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_ingestion_watermark_athlete_id"), ["athlete_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_ingestion_watermark_source_descriptor_id"),
            ["source_descriptor_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_ingestion_watermark_scope",
            ["athlete_id", "source_descriptor_id", "gbo_type"],
            unique=False,
        )

    op.create_table(
        _GAP,
        sa.Column("ingestion_gap_id", sa.Uuid(), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("source_descriptor_id", sa.Uuid(), nullable=True),
        sa.Column("gbo_type", _GBO_TYPE, nullable=False),
        sa.Column("reason", _GAP_REASON, nullable=False),
        sa.Column("severity", _SEVERITY, nullable=False),
        sa.Column("state", _GAP_STATE, nullable=False),
        sa.Column("transient", sa.Boolean(), nullable=False),
        sa.Column("range_start_at", UtcDateTime(), nullable=True),
        sa.Column("range_end_at", UtcDateTime(), nullable=True),
        sa.Column("range_start_token", sa.String(length=256), nullable=True),
        sa.Column("range_end_token", sa.String(length=256), nullable=True),
        sa.Column("ingest_run_id", sa.Uuid(), nullable=True),
        sa.Column("first_seen_at", UtcDateTime(), nullable=True),
        sa.Column("last_seen_at", UtcDateTime(), nullable=True),
        sa.Column("closed_at", UtcDateTime(), nullable=True),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.Column("updated_at", UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["athlete_id"],
            ["athlete.athlete_id"],
            name=op.f("fk_ingestion_gap_athlete_id_athlete"),
        ),
        sa.ForeignKeyConstraint(
            ["source_descriptor_id"],
            ["source_descriptor.source_descriptor_id"],
            name=op.f("fk_ingestion_gap_source_descriptor_id_source_descriptor"),
        ),
        sa.PrimaryKeyConstraint("ingestion_gap_id", name=op.f("pk_ingestion_gap")),
    )
    with op.batch_alter_table(_GAP, schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_ingestion_gap_athlete_id"), ["athlete_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_ingestion_gap_source_descriptor_id"),
            ["source_descriptor_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_ingestion_gap_athlete_source_gbo",
            ["athlete_id", "source_descriptor_id", "gbo_type"],
            unique=False,
        )
        batch_op.create_index("ix_ingestion_gap_state", ["state"], unique=False)
        batch_op.create_index("ix_ingestion_gap_ingest_run_id", ["ingest_run_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table(_GAP, schema=None) as batch_op:
        batch_op.drop_index("ix_ingestion_gap_ingest_run_id")
        batch_op.drop_index("ix_ingestion_gap_state")
        batch_op.drop_index("ix_ingestion_gap_athlete_source_gbo")
        batch_op.drop_index(batch_op.f("ix_ingestion_gap_source_descriptor_id"))
        batch_op.drop_index(batch_op.f("ix_ingestion_gap_athlete_id"))
    op.drop_table(_GAP)
    with op.batch_alter_table(_WATERMARK, schema=None) as batch_op:
        batch_op.drop_index("ix_ingestion_watermark_scope")
        batch_op.drop_index(batch_op.f("ix_ingestion_watermark_source_descriptor_id"))
        batch_op.drop_index(batch_op.f("ix_ingestion_watermark_athlete_id"))
    op.drop_table(_WATERMARK)
