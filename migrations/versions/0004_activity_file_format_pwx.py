"""activity_file.format CHECK — add 'pwx' (B-P2 PWX file-format adapter).

The ``ActivityFileFormat`` enum gained ``pwx`` (a new file-upload decoder, ROAD-R6), but
``activity_file.format`` is a ``VARCHAR + CHECK`` column (``native_enum=False``) whose CHECK
was pinned to the original value set in the initial migration. Without this migration a PWX
import decodes + maps fine and then fails at INSERT with a CHECK-constraint violation, so no
PWX activity could ever be ingested against a migration-built database. Re-type the column to
the new Enum so its named CHECK is regenerated with the full value set including ``pwx``.

Portability (BOOT-R3) needs a per-dialect path, verified against real PostgreSQL + MariaDB:

- **SQLite / PostgreSQL** — ``batch_alter_table(recreate="always")`` re-types the column to the
  new Enum, regenerating its CHECK. SQLite cannot ``DROP`` a named CHECK, so the table rebuild
  is required; PostgreSQL handles the rebuild cleanly.
- **MariaDB / MySQL** — the rebuild FAILS there because the new table would carry a CHECK with
  the SAME name (``activityfileformat``) while the old table still exists (a duplicate-name
  error). MariaDB enforces unique constraint names AND supports ``DROP CONSTRAINT`` (10.2+), so
  we drop the named CHECK and re-add it in place instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "activity_file"
_CHECK = "activityfileformat"  # the named CHECK the Enum emits (verified via SHOW CREATE)
_WITHOUT_PWX = ("fit", "gpx", "tcx", "json", "other")
_WITH_PWX = ("fit", "gpx", "tcx", "pwx", "json", "other")


def _format_enum(values: tuple[str, ...]) -> sa.Enum:
    """The ``activity_file.format`` Enum (VARCHAR + named CHECK) for a given value set.

    Matches the ORM ``enum_column(ActivityFileFormat)`` spelling (``native_enum=False`` so
    it is a portable VARCHAR + CHECK on every backend, GBO-R12).
    """
    return sa.Enum(
        *values, name=_CHECK, native_enum=False, create_constraint=True, length=64
    )


def _condition(values: tuple[str, ...]) -> str:
    """The ``format IN ('a', 'b', ...)`` CHECK body for the MariaDB/MySQL in-place re-add."""
    return "format IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def _retype(old_values: tuple[str, ...], new_values: tuple[str, ...]) -> None:
    """Move ``activity_file.format`` from ``old_values`` to ``new_values`` portably (BOOT-R3)."""
    if op.get_bind().dialect.name in ("mysql", "mariadb"):
        # MariaDB/MySQL: drop + re-add the named CHECK in place (a table rebuild collides on
        # the reused constraint name).
        op.drop_constraint(_CHECK, _TABLE, type_="check")
        op.create_check_constraint(_CHECK, _TABLE, _condition(new_values))
    else:
        # SQLite (must rebuild — no named-CHECK drop) + PostgreSQL: re-type the column.
        with op.batch_alter_table(_TABLE, schema=None, recreate="always") as batch_op:
            batch_op.alter_column(
                "format",
                existing_type=_format_enum(old_values),
                type_=_format_enum(new_values),
                existing_nullable=False,
            )


def upgrade() -> None:
    _retype(_WITHOUT_PWX, _WITH_PWX)


def downgrade() -> None:
    _retype(_WITH_PWX, _WITHOUT_PWX)
