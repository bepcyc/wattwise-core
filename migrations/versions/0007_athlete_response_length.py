"""athlete persisted answer-length default (user-settings response-length, API-R11f).

Adds the nullable ``athlete.default_response_length`` column backing the persisted
answer-length preference the user-settings surface reads/writes
(``GET``/``PUT /v1/user-settings/response-length``, doc 60 §8.10 / API-R11f): the default
applied to every athlete-facing agent answer + deliverable when an ``AgentAskRequest``
gives no per-request ``response_length``. It holds one of the ``ResponseLength`` tokens
(``short``/``standard``/``detailed``); ``NULL`` means the system default (``standard``).
It is an agent-interaction preference — verbosity only, never analytics truth (VOICE-R8) —
held on the single-owner profile row rather than a separate table, because the OSS
single-tenant store has exactly ONE of these (no per-tenant fan-out).

PORTABLE (GBO-R8b): the column uses only the portable ``sa.String`` primitive the athlete
profile already spells, so this revision runs unchanged on SQLite / PostgreSQL / MariaDB
(DSN-only difference). The add is done in ``batch_alter_table`` so SQLite (which cannot
``ALTER TABLE ... ADD COLUMN`` with every option inline) performs the portable
copy-and-recreate, identical to the other revisions. No server-side default and no CHECK
is emitted: the model column is a bare ``mapped_column(String(16), nullable=True)`` (NOT an
``enum_column``), so the migrated schema matches the live ORM exactly and ``alembic check``
finds no drift (BOOT-R3 parity gate). The plain string (no CHECK) mirrors the sibling
``default_training_load_model`` column; the application layer validates the token set.

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-09 12:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "athlete"
_COLUMN = "default_response_length"


def upgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.add_column(sa.Column(_COLUMN, sa.String(length=16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_column(_COLUMN)
