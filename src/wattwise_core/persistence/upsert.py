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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
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
        # MariaDB insert-or-ignore: a TRUE ``INSERT IGNORE`` — never the
        # update-a-key-to-itself ``ON DUPLICATE KEY UPDATE`` trick. The self-assignment
        # still WRITES the conflicting row, and under MariaDB >= 11.6's default
        # ``innodb_snapshot_isolation=ON`` (REPEATABLE READ) writing a row whose
        # committed version is newer than the transaction's read view fails with
        # error 1020 ``ER_CHECKREAD`` — exactly the concurrent get-or-create race this
        # branch exists to absorb. ``INSERT IGNORE`` skips the conflicting row without
        # writing it, so both racers succeed (UPS-R2 atomicity, no catch-and-retry).
        return mstmt.prefix_with("IGNORE")
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


async def ensure_row(
    session_factory: async_sessionmaker[AsyncSession],
    table: Table,
    values: Mapping[str, Any],
    *,
    conflict_keys: Sequence[str],
) -> None:
    """Atomically guarantee a row EXISTS (insert-or-ignore) in its OWN short transaction.

    The get-or-create seam for concurrent first-touches (UPS-R2): both racers issue ONE
    atomic statement and both succeed — never a plain ``INSERT`` whose loser raises, and
    never catch-and-retry. The statement runs on a FRESH session committed immediately,
    NOT inside the caller's (possibly long-lived) transaction, because of a
    MySQL-family semantic this seam must own: MariaDB >= 11.6 defaults
    ``innodb_snapshot_isolation=ON``, under which a REPEATABLE READ statement that
    touches a conflict row committed AFTER the transaction's read view fails with
    error 1020 ``ER_CHECKREAD`` ("Record has changed since last read") — this hits
    plain ``INSERT``, ``INSERT IGNORE`` and ``ON DUPLICATE KEY UPDATE`` alike. The
    MySQL-family leg therefore runs the statement at ``READ COMMITTED`` (per-statement
    read view; ``innodb_snapshot_isolation`` applies only to REPEATABLE READ), which
    removes the failure mode at the root instead of retrying around it. This is dialect
    knowledge, so it lives HERE — the one sanctioned dialect-aware module (RUN-R7-AC).

    On return the row is committed and visible to NEW snapshots; a caller inside an
    older REPEATABLE READ snapshot must re-read through a fresh session/connection.
    """
    async with session_factory() as session:
        dialect = _dialect_name(session)
        if dialect in ("mysql", "mariadb"):
            await session.connection(execution_options={"isolation_level": "READ COMMITTED"})
        stmt = build_upsert(dialect, table, values, conflict_keys, update_columns=[])
        await session.execute(stmt)
        await session.commit()


__all__ = ["UnsupportedDialectError", "build_upsert", "ensure_row", "upsert", "upsert_many"]
