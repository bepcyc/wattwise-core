"""The single sanctioned dialect-aware upsert seam (UPS-R2).

SQLAlchemy has no backend-agnostic upsert, so the atomic insert-or-update on a
natural key lives here, in ONE place, branching on the dialect (PostgreSQL/SQLite
``ON CONFLICT ... DO UPDATE`` vs MariaDB ``ON DUPLICATE KEY UPDATE``). This is the
**only** module in application code permitted to branch on the SQL dialect — the
``no-vendor-SQL`` gate (RUN-R7-AC) whitelists exactly this file. No other module may
import a dialect-specific construct or branch on ``dialect.name``.

The upsert is atomic (never check-then-write, so there is no race; UPS-R7/R2) and is
the mechanism the dedup resolver and ingestion path use to land candidates and
resolved canonical rows idempotently (UPS-R3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Table
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.dml import Insert


class UnsupportedDialectError(RuntimeError):
    """Raised when the configured backend is not one of the three supported ones."""


def _dialect_name(session: AsyncSession) -> str:
    bind = session.get_bind()
    return bind.dialect.name


def _as_rows(values: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Normalise ``values`` to a non-empty list of row mappings (fail-closed on empty)."""
    rows = [values] if isinstance(values, Mapping) else list(values)
    if not rows:
        raise ValueError("upsert requires at least one row (empty batch has no column set)")
    return rows


def build_upsert(
    dialect: str,
    table: Table,
    values: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    conflict_keys: Sequence[str],
    update_columns: Sequence[str] | None,
) -> Insert:
    """Build a dialect-specific atomic upsert statement.

    ``values`` is either ONE row mapping or a non-empty SEQUENCE of row mappings; a
    sequence compiles to a single multi-row ``VALUES`` clause — one round-trip per
    batch (PERF-R1), not a per-row insert loop. Every row in a batch carries the SAME
    column set, taken from the first row.

    ``update_columns`` are the columns refreshed on conflict; when ``None`` every
    inserted column except the conflict keys is updated. Pass an empty sequence for
    insert-or-ignore semantics (used for byte-identical re-ingest no-ops, UPS-R3).
    """
    rows = _as_rows(values)
    cols = list(rows[0].keys())
    if update_columns is None:
        # On conflict, refresh every supplied value column EXCEPT the conflict keys, the
        # surrogate primary key, and created_at — clobbering the PK would rewrite the
        # identity that source_candidate.resolved_*_id back-pointers reference, and
        # bumping created_at would churn an unchanged row (UPS-R3 idempotency, GBO-AC-1).
        protected = set(conflict_keys) | set(table.primary_key.columns.keys()) | {"created_at"}
        update_columns = [c for c in cols if c not in protected]

    # --- the ONE sanctioned dialect branch (UPS-R2) ---
    # Reference the conflict-row columns by SUBSCRIPT (``excluded[c]`` / ``inserted[c]``),
    # never ``getattr`` — a column literally named ``values`` would otherwise resolve to the
    # ColumnCollection ``.values()`` METHOD instead of the column (a silent corruption).
    if dialect in ("postgresql", "sqlite"):
        insert_fn = pg_insert if dialect == "postgresql" else sqlite_insert
        stmt = insert_fn(table).values(rows)
        if update_columns:
            set_ = {c: stmt.excluded[c] for c in update_columns}
            stmt = stmt.on_conflict_do_update(index_elements=list(conflict_keys), set_=set_)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=list(conflict_keys))
        return stmt
    if dialect in ("mysql", "mariadb"):
        mstmt = mysql_insert(table).values(rows)
        if update_columns:
            set_ = {c: mstmt.inserted[c] for c in update_columns}
            return mstmt.on_duplicate_key_update(**set_)
        # MariaDB insert-or-ignore: update a conflict key to itself (a no-op).
        first_key = conflict_keys[0]
        return mstmt.on_duplicate_key_update(**{first_key: mstmt.inserted[first_key]})
    raise UnsupportedDialectError(
        f"unsupported dialect {dialect!r}; wattwise-core supports sqlite, postgresql, mariadb"
    )


async def upsert(
    session: AsyncSession,
    table: Table,
    values: Mapping[str, Any],
    *,
    conflict_keys: Sequence[str],
    update_columns: Sequence[str] | None = None,
) -> None:
    """Execute an atomic insert-or-update on ``conflict_keys`` (UPS-R2).

    Atomic at the database level — no check-then-write, so there is no
    time-of-check/time-of-use race when two ingest runs land the same natural key.
    """
    dialect = _dialect_name(session)
    stmt = build_upsert(dialect, table, values, conflict_keys, update_columns)
    await session.execute(stmt)


async def upsert_many(
    session: AsyncSession,
    table: Table,
    rows: Sequence[Mapping[str, Any]],
    *,
    conflict_keys: Sequence[str],
    update_columns: Sequence[str] | None = None,
) -> None:
    """Execute a BATCHED atomic insert-or-update in a single round-trip (PERF-R1).

    ``rows`` is upserted with ONE multi-row ``VALUES`` statement — never a per-row
    insert loop (PERF-R1). An empty ``rows`` is a no-op (no statement issued). Each row
    carries the same column set; conflicts on ``conflict_keys`` update in place, so
    re-ingest is idempotent (UPS-R3) and concurrent runs cannot race (UPS-R2).
    """
    if not rows:
        return
    dialect = _dialect_name(session)
    stmt = build_upsert(dialect, table, list(rows), conflict_keys, update_columns)
    await session.execute(stmt)


__all__ = ["UnsupportedDialectError", "build_upsert", "upsert", "upsert_many"]
