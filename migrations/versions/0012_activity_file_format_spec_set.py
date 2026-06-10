"""activity_file.format CHECK — restore the spec-closed 5-member set (SCHEMA-R3 / API-R33).

The canonical ``activity_file_format`` enum is spec-CLOSED to ``fit|gpx|tcx|json|other``
(doc 60 SCHEMA-R3, matching doc 20 §3.2.2 verbatim), and the API-R33 upload allowlist is
exactly ``.fit/.fit.gz/.gpx/.tcx``. Migration 0004 widened the CHECK with an unsanctioned
``pwx`` member; the ORM enum and the ingestion accept lists have been narrowed back to the
spec set, so this revision regenerates the named CHECK without ``pwx`` (the inverse of
0004, reusing its verified per-dialect path).

Fail-closed note: if a deployment ever ingested a ``pwx`` row, re-adding the narrowed
CHECK fails the upgrade rather than silently stranding a value outside the constraint —
the operator must re-import that activity from a sanctioned format first.

Portability (BOOT-R3), verified shape per 0004:

- **SQLite / PostgreSQL** — ``batch_alter_table(recreate="always")`` re-types the column,
  regenerating its named CHECK.
- **MariaDB / MySQL** — drop + re-add the named CHECK in place (the rebuild would collide
  on the reused constraint name).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "activity_file"
_CHECK = "activityfileformat"  # the named CHECK the Enum emits (see 0004)
_SPEC_SET = ("fit", "gpx", "tcx", "json", "other")
_WITH_PWX = ("fit", "gpx", "tcx", "pwx", "json", "other")


def _format_enum(values: tuple[str, ...]) -> sa.Enum:
    """The ``activity_file.format`` Enum (VARCHAR + named CHECK) for a given value set."""
    return sa.Enum(
        *values, name=_CHECK, native_enum=False, create_constraint=True, length=64
    )


def _condition(values: tuple[str, ...]) -> str:
    """The ``format IN ('a', 'b', ...)`` CHECK body for the MariaDB/MySQL in-place re-add."""
    return "format IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def _retype(old_values: tuple[str, ...], new_values: tuple[str, ...]) -> None:
    """Move ``activity_file.format`` between value sets portably (BOOT-R3; mirrors 0004)."""
    if op.get_bind().dialect.name in ("mysql", "mariadb"):
        op.drop_constraint(_CHECK, _TABLE, type_="check")
        op.create_check_constraint(_CHECK, _TABLE, _condition(new_values))
        return
    with op.batch_alter_table(_TABLE, recreate="always") as batch:
        batch.alter_column(
            "format",
            existing_type=_format_enum(old_values),
            type_=_format_enum(new_values),
            existing_nullable=False,
        )


def upgrade() -> None:
    """Narrow the CHECK back to the spec-closed 5-member set (drop ``pwx``)."""
    _retype(_WITH_PWX, _SPEC_SET)


def downgrade() -> None:
    """Re-widen the CHECK with ``pwx`` (the 0004 state)."""
    _retype(_SPEC_SET, _WITH_PWX)
