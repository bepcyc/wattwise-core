"""activity perceived_exertion + feel — athlete-reported session exertion (SRPE-R1).

Adds two nullable columns to ``activity``:

- ``perceived_exertion`` — the athlete-reported session exertion on the CR-10 scale
  (0..10). It is the only internal-load input that exists for power-less, HR-less
  sessions (strength work, most swims), and the primitive the ``srpe_load`` member of
  the ``training_load`` equivalence class is computed from (LOAD-R3 last resort).
- ``feel`` — the athlete-reported session feel, the intervals.icu 1..5 ordinal
  (1 = strong, 5 = weak). A subjective summary token, not a load input.

Both are captured from sources that already carry them (FIT ``perceived_exertion``,
intervals.icu ``icu_rpe``/``feel``) through the standard candidate -> trust-policy
resolution, so they ride the existing ``coverage`` / ``field_resolution`` lineage.

PORTABLE (GBO-R8b): ``perceived_exertion`` uses the canonical ``sa.Numeric(18, 6)``
primitive every ``numeric_column`` produces, ``feel`` the ``sa.SmallInteger`` of
``smallint_column``, so this revision runs unchanged on SQLite / PostgreSQL / MariaDB
(DSN-only difference). The adds run inside ``batch_alter_table`` so SQLite performs its
portable copy-and-recreate, identical to the other revisions. Both columns are nullable
with no server-side default — an unreported session surfaces a typed absence, never an
imputed neutral value (GBO-R7) — and match the ORM columns exactly so ``alembic check``
finds no drift (BOOT-R3 parity gate).

Revision ID: 0015
Revises: 0014
Create Date: 2026-06-11 00:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "activity"
_RPE_COLUMN = "perceived_exertion"
_FEEL_COLUMN = "feel"


def upgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(_RPE_COLUMN, sa.Numeric(precision=18, scale=6), nullable=True)
        )
        batch_op.add_column(sa.Column(_FEEL_COLUMN, sa.SmallInteger(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_column(_FEEL_COLUMN)
        batch_op.drop_column(_RPE_COLUMN)
