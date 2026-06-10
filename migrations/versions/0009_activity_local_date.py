"""activity local_date — canonical athlete-LOCAL day-attribution bucket (GBO-R33/R34/R35).

Adds the nullable ``activity.local_date`` column: the athlete-LOCAL calendar date of the
activity's UTC ``start_time``, projected through the athlete's effective-dated reference
timezone (§3.8). It is the reproducible day-attribution bucket the analytics day-rollups
(daily-load → PMC/CTL-ATL) read (GBO-R35), recomputable purely from the UTC instant + the
as-of reference-timezone metadata (GBO-R34) — the UTC ``start_time`` stays the source of
truth (GBO-R32). It mirrors ``daily_wellness.local_date`` (a ``Date``, NOT a UTC date) so
both day-bucketed entities key on the same local-calendar rule.

A composite ``(athlete_id, local_date)`` index backs the local-day range query the
analytics layer issues (the day-bucket scan), matching the existing
``ix_daily_wellness_athlete_local_date_desc`` shape on the wellness side.

PORTABLE (GBO-R8b): uses only the portable ``sa.Date`` primitive (the athlete/wellness
profiles already spell ``Date``), so this revision runs unchanged on SQLite / PostgreSQL /
MariaDB (DSN-only difference). The add + index creation run inside ``batch_alter_table`` so
SQLite performs its portable copy-and-recreate, identical to the other revisions. The
column is nullable with no server-side default — a row ingested before a reference timezone
existed surfaces a typed absence rather than a fabricated date (GBO-R7), and the migrated
schema matches the bare ``mapped_column(Date, nullable=True)`` ORM column exactly so
``alembic check`` finds no drift (BOOT-R3 parity gate).

Revision ID: 0009
Revises: 0008
Create Date: 2026-06-10 06:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "activity"
_COLUMN = "local_date"
_INDEX = "ix_activity_athlete_local_date"


def upgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.add_column(sa.Column(_COLUMN, sa.Date(), nullable=True))
        batch_op.create_index(_INDEX, ["athlete_id", _COLUMN], unique=False)


def downgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_index(_INDEX)
        batch_op.drop_column(_COLUMN)
