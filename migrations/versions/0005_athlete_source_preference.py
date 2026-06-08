"""per-athlete source-trust override (athlete_source_preference, PRV-R7 / CONF-R1).

Creates the ``athlete_source_preference`` table: configuration DATA letting an athlete
override the per-channel ``trust_tier`` a source declares in its descriptor
``trust_profile`` (LIN-R1), without a code change (PRV-R7). One row binds
``(athlete_id, source_descriptor_id, channel)`` to an effective ``trust_tier``;
``channel = "*"`` is the whole-source default, a concrete channel name overrides only
that field/channel. The conflict resolver consults these rows as the highest-precedence
layer of its effective-tier resolution; an empty table means no overrides, so default
resolution stays byte-identical (the opt-in invariant).

PORTABLE (GBO-R8b / BOOT-R3): a plain ``create_table`` emitting only the portable types
the column factories produce — ``sa.Uuid``, ``sa.Enum(native_enum=False,
create_constraint=True)`` (text + named CHECK, NOT a native PG ENUM), ``UtcDateTime()``
— so the SAME revision runs unchanged on SQLite / PostgreSQL / MariaDB (DSN-only
difference). A fresh table needs no CHECK-rename gymnastics (cf. 0004): the named CHECK
is created with the full ``Fidelity`` value set up front. The Enum spelling, the named
CHECK (``ck_athlete_source_preference_trust_tier``), and the constraint/index names
match the ORM ``AthleteSourcePreference`` + naming convention so ``alembic check`` finds
no drift between this migration and the live model (BOOT-R3 parity gate).

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-08 10:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from wattwise_core.persistence.types import UtcDateTime

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "athlete_source_preference"

# The canonical Fidelity value set, spelled exactly as enum_column(Fidelity) renders it
# (native_enum=False -> VARCHAR + named CHECK on every backend, GBO-R12). The named CHECK
# follows the project naming convention (ck_<table>_<column>) so it matches the live ORM.
_FIDELITY = sa.Enum(
    "raw_stream",
    "device_computed",
    "platform_computed",
    "modeled",
    "summary_only",
    "substituted",
    "absent_true",
    "absent_failed",
    name="fidelity",
    native_enum=False,
    create_constraint=True,
    length=64,
)


def upgrade() -> None:
    op.create_table(
        _TABLE,
        sa.Column("athlete_source_preference_id", sa.Uuid(), nullable=False),
        sa.Column("athlete_id", sa.Uuid(), nullable=False),
        sa.Column("source_descriptor_id", sa.Uuid(), nullable=False),
        sa.Column("channel", sa.String(length=64), nullable=False),
        sa.Column("trust_tier", _FIDELITY, nullable=False),
        sa.Column("created_at", UtcDateTime(), nullable=False),
        sa.Column("updated_at", UtcDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["athlete_id"],
            ["athlete.athlete_id"],
            name=op.f("fk_athlete_source_preference_athlete_id_athlete"),
        ),
        sa.ForeignKeyConstraint(
            ["source_descriptor_id"],
            ["source_descriptor.source_descriptor_id"],
            name=op.f(
                "fk_athlete_source_preference_source_descriptor_id_source_descriptor"
            ),
        ),
        sa.PrimaryKeyConstraint(
            "athlete_source_preference_id", name=op.f("pk_athlete_source_preference")
        ),
        sa.UniqueConstraint(
            "athlete_id",
            "source_descriptor_id",
            "channel",
            name="uq_athlete_source_preference_athlete_source_channel",
        ),
    )
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_athlete_source_preference_athlete_id"),
            ["athlete_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_athlete_source_preference_source_descriptor_id"),
            ["source_descriptor_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_athlete_source_preference_source_descriptor_id"))
        batch_op.drop_index(batch_op.f("ix_athlete_source_preference_athlete_id"))
    op.drop_table(_TABLE)
