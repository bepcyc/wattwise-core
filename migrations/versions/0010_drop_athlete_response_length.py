"""drop the canonical athlete.default_response_length column (VOICE-R8 §382 store-split fix).

The persisted answer-length default (``GET``/``PUT /v1/user-settings/response-length``, doc 60
§8.10 / API-R11f) is — per doc 50 VOICE-R8 §382 — an agent-interaction preference held in the
dedicated AGENT-STATE store (a ``preference``-kind memory item, MEM-R1), NOT a canonical §3
master-data entity like ``language``/``primary_locale``. Migration 0007 wrongly added a canonical
``athlete.default_response_length`` column to back it. That created a store-split bug (the HIGH
finding): the run path READ the agent-state preference (the engine scans the ``response_length=``
``PREFERENCE`` item) while the user-settings ``PUT`` WROTE this canonical column — so a saved
preference never reached the run.

This revision drops the now-dead canonical column. The user-settings GET/PUT and the run-path
default all read/write the SINGLE agent-state preference through the engine seam, so the value an
athlete sets is exactly the run-path default (VOICE-R8 single source). No data migration is needed:
OSS is single-tenant and the column carried at most one transient owner preference now superseded by
the agent-state preference; the downgrade re-adds the bare nullable column (no value backfill).

PORTABLE (GBO-R8b): the drop is done inside ``batch_alter_table`` so SQLite (which cannot
``ALTER TABLE ... DROP COLUMN`` inline on older engines) performs its portable copy-and-recreate,
identical on PostgreSQL / MariaDB (DSN-only difference). After this revision the live ORM no longer
declares the column, so ``alembic check`` against the migrated schema finds NO drift (BOOT-R3).

Revision ID: 0010
Revises: 0009
Create Date: 2026-06-10 00:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "athlete"
_COLUMN = "default_response_length"


def upgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_column(_COLUMN)


def downgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.add_column(sa.Column(_COLUMN, sa.String(length=16), nullable=True))
