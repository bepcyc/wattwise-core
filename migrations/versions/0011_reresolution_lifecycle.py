"""Re-resolution lifecycle + lineage columns (CONF-R6, EVOL-R2, UPS-R5, MAP-R6/R10/R12, LIN-R3).

Adds the additive columns the multi-source re-resolution lifecycle needs:

* ``source_descriptor.is_active`` — EVOL-R2: disabling a source is a CONFIGURATION
  action (deactivate the descriptor); inactive descriptors' candidates stop
  contributing and affected canonical records re-resolve from the remainder.
* ``source_candidate.is_tombstone`` — UPS-R5: a source-side deletion is a typed
  tombstone candidate (removes that source's contribution; never a cascade delete).
* ``source_candidate.strong_fingerprint`` (+ lookup index) — MAP-R10: the typed
  shared device/file fingerprint that matches regardless of the time window.
* ``source_candidate.quarantine_rule_id`` — MAP-R6: a candidate failing canonical
  schema/invariant validation is quarantined with the failing rule id.
* ``source_candidate.identity_resolution`` — MAP-R12: the recorded identity decision
  (rule fired, match score, matched ids) so a merge is explainable and splittable.
* ``activity.policy_version`` / ``daily_wellness.policy_version`` — CONF-R6: the
  conflict-policy version that produced the resolved values is recorded.
* ``activity.field_resolution`` / ``daily_wellness.field_resolution`` — LIN-R3: the
  per-field resolution record (winner/considered candidate pointers + deciding rule).

PORTABLE (GBO-R8b): only ``sa.Boolean`` / ``sa.String`` / ``sa.JSON`` primitives, all
nullable or server-defaulted, applied inside ``batch_alter_table`` so SQLite performs
its copy-and-recreate; runs unchanged on SQLite / PostgreSQL / MariaDB. The boolean
columns carry a server default so existing rows backfill, matching the ORM defaults.

Revision ID: 0011
Revises: 0010
Create Date: 2026-06-10 09:05:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FP_INDEX = "ix_source_candidate_strong_fingerprint"


def upgrade() -> None:
    with op.batch_alter_table("source_descriptor", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "is_active",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
    with op.batch_alter_table("source_candidate", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "is_tombstone",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch_op.add_column(
            sa.Column("strong_fingerprint", sa.String(length=256), nullable=True)
        )
        batch_op.add_column(
            sa.Column("quarantine_rule_id", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(sa.Column("identity_resolution", sa.JSON(), nullable=True))
        batch_op.create_index(
            _FP_INDEX, ["athlete_id", "strong_fingerprint"], unique=False
        )
    for table in ("activity", "daily_wellness"):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("policy_version", sa.String(length=64), nullable=True)
            )
            batch_op.add_column(sa.Column("field_resolution", sa.JSON(), nullable=True))


def downgrade() -> None:
    for table in ("activity", "daily_wellness"):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_column("field_resolution")
            batch_op.drop_column("policy_version")
    with op.batch_alter_table("source_candidate", schema=None) as batch_op:
        batch_op.drop_index(_FP_INDEX)
        batch_op.drop_column("identity_resolution")
        batch_op.drop_column("quarantine_rule_id")
        batch_op.drop_column("strong_fingerprint")
        batch_op.drop_column("is_tombstone")
    with op.batch_alter_table("source_descriptor", schema=None) as batch_op:
        batch_op.drop_column("is_active")
