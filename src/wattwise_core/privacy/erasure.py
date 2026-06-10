"""Whole-athlete erasure EXECUTOR — the "right to be forgotten" fulfilment (PRIV-R8/-R11).

PRIV-R8 requires that an authenticated erasure request delete the athlete's canonical
records, durable agent memory, agent checkpoints/threads/writes/interrupts, source
candidates/connections, the retained original-file objects, and source secrets — and
"produce an auditable record that erasure completed". This module is the executable
fulfilment: :func:`erase_athlete` deletes every athlete-scoped row across BOTH stores and
returns an :class:`ErasureReceipt` carrying the per-table residual-zero counts.

Two stores, two metadatas (ARCH-R13). Durable agent state (checkpoints/threads/writes/
interrupts/memory) lives in a store that is structurally NEVER the canonical GBO store and
owns its OWN engine/pool (``state_db``). The executor therefore takes the two session seams
as INJECTED deps — one per store — and runs each store's deletions inside that session's
transaction (commit on success, roll back on any error: fail-closed, no partial silent
success). The two stores are distinct databases, so this is one transaction PER store, not a
single distributed transaction; the receipt records each store's outcome so a partial failure
is auditable, never masked.

Exhaustive by construction (not a hand-kept list). The set of athlete-scoped tables is
DERIVED from the live ORM metadata: every table carrying an ``athlete_id`` column is scoped
directly; the few tables that hold athlete personal data only TRANSITIVELY (``activity_lap``,
``activity_stream_set``, ``derived_activity_metric`` via ``activity``; ``stream_channel`` via
its activity/wellness stream-set parent; ``agent_write`` via ``agent_thread``) are scoped
through their parent's athlete-owned keys. A canonical/agent-state table that is neither
directly nor transitively mapped raises at call time (fail-closed) rather than being silently
skipped — so a NEW athlete-bearing table cannot quietly escape erasure. Shared registry/config
tables (``sport``, ``sub_sport``, ``source_descriptor``) and the NULL-``athlete_id`` shared
``workout`` library templates are never touched (the ``WHERE athlete_id = :id`` predicate
already excludes ``NULL``).

Deletion order is FK-correct by construction: the per-store delete order is the metadata's
topological ``sorted_tables`` (parents-first) REVERSED (children-first), so a child row is
always removed before its parent even when foreign keys are RESTRICT (the ORM declares no
``ON DELETE CASCADE``; SQLite enforces FKs via ``PRAGMA foreign_keys=ON``).

Idempotent: a second run finds nothing and returns a receipt with all-zero counts (no error).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import Select, Table, delete, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

import wattwise_core.agent.memory  # noqa: F401  (registers agent_memory_item on AgentStateBase)
from wattwise_core.agent.state_store import AgentStateBase
from wattwise_core.persistence.models import Base
from wattwise_core.storage import ObjectStore

# Canonical tables that are NOT athlete-scoped at all (shared registry / global config).
# They hold no personal data and MUST NOT be touched by a per-athlete erasure.
_CANONICAL_SHARED_TABLES: frozenset[str] = frozenset({"sport", "sub_sport", "source_descriptor"})

# Scope path for a table that carries NO ``athlete_id`` column: the child column to match
# against the athlete-owned ids produced by following one or more PARENT HOPS down to an
# ``athlete_id`` column.
#
# ``child_column``  — the column on the table being deleted to filter on (an FK / parent ref);
# ``hops``          — a chain ``[(table, select_column, match_column), ...]`` evaluated
#                     OUTER-to-inner: the FINAL hop's table carries ``athlete_id`` and is
#                     filtered by it; each earlier hop selects its ``select_column`` where its
#                     ``match_column`` is in the next inner hop's selected ids. The first hop's
#                     ``select_column`` is the id set the ``child_column`` must be ``IN``.
#
# This resolves athlete-owned ids from the LIVE parent rows independently of deletion order,
# so the transitive children are deleted correctly even though their parents are not yet gone.
_ScopePath = tuple[str, tuple[tuple[str, str, str], ...]]


@dataclass(frozen=True, slots=True)
class _Transitive:
    """How to scope a no-``athlete_id`` table to the athlete: one or more :data:`_ScopePath`."""

    paths: tuple[_ScopePath, ...]


# Canonical tables that hold athlete personal data only transitively. ``activity_lap`` /
# ``activity_stream_set`` / ``derived_activity_metric`` hang off ``activity`` (one hop).
# ``stream_channel`` has NO foreign key and TWO possible parents (an activity- or a
# wellness-stream-set, discriminated by ``set_kind``): the activity branch is TWO hops
# (stream-set ids whose ``activity_id`` belongs to the athlete's activities) because
# ``activity_stream_set`` itself carries no ``athlete_id``; the wellness branch is one hop
# (``wellness_stream_set`` IS athlete-scoped).
_CANONICAL_TRANSITIVE: Mapping[str, _Transitive] = {
    "activity_lap": _Transitive((("activity_id", (("activity", "activity_id", "athlete_id"),)),)),
    "activity_stream_set": _Transitive(
        (("activity_id", (("activity", "activity_id", "athlete_id"),)),)
    ),
    "derived_activity_metric": _Transitive(
        (("activity_id", (("activity", "activity_id", "athlete_id"),)),)
    ),
    "stream_channel": _Transitive(
        (
            (
                "stream_set_id",
                (
                    ("activity_stream_set", "stream_set_id", "activity_id"),
                    ("activity", "activity_id", "athlete_id"),
                ),
            ),
            (
                "stream_set_id",
                (("wellness_stream_set", "wellness_stream_set_id", "athlete_id"),),
            ),
        )
    ),
}

# Agent-state tables that carry no ``athlete_id`` but are scoped via ``agent_thread``.
_AGENT_TRANSITIVE: Mapping[str, _Transitive] = {
    "agent_write": _Transitive((("thread_id", (("agent_thread", "thread_id", "athlete_id"),)),)),
}

# Agent-state OPERATIONAL auth tables (amended ARCH-R13) scoped by a STRING ``subject``
# column carrying the owner id (API-R23 refresh families / AUTH-R8 link challenges): a
# per-athlete erasure deletes the rows whose subject is the athlete; an UNBOUND link
# challenge (NULL subject) holds no personal data and simply expires.
_AGENT_SUBJECT_SCOPED: Mapping[str, str] = {
    "agent_auth_refresh_token": "subject",
    "agent_auth_link_challenge": "subject",
}

# Tables that MUST be deleted before topological order would otherwise place them: a no-FK
# child whose scope subquery reads a parent that the topo order would delete first. The
# un-FK'd ``stream_channel`` (no FK -> topo-sorted as a root, hence deleted LAST) reads
# ``activity_stream_set`` / ``wellness_stream_set``; it MUST run while those parents still
# exist, so it is forced to the FRONT of the canonical delete order.
_CANONICAL_DELETE_FIRST: tuple[str, ...] = ("stream_channel",)


@dataclass(frozen=True, slots=True)
class StoreErasureReport:
    """Per-store erasure outcome: ``{table_name: rows_deleted}`` plus the object count.

    ``deleted_rows`` is the auditable residual proof — after a successful erasure a re-count
    of every listed table for the athlete MUST be zero (PRIV-R8-AC). ``deleted_objects`` is
    the number of original-file objects whose BYTES were deleted from the object store
    (PRIV-R11.3); it is ``0`` in the agent-state report (no objects there).
    """

    store: str
    deleted_rows: Mapping[str, int]
    deleted_objects: int = 0


@dataclass(frozen=True, slots=True)
class ErasureReceipt:
    """Auditable completion record for a whole-athlete erasure (PRIV-R8).

    Returned by :func:`erase_athlete` after BOTH stores have been erased and committed. It is
    the "auditable record that erasure completed": it names the subject, the instant, and the
    per-store / per-table row counts removed (and original-file objects deleted). The wiring
    layer persists/logs it; this module mints it but takes no opinion on where it is stored.
    """

    athlete_id: uuid.UUID
    completed_at: _dt.datetime
    stores: Sequence[StoreErasureReport] = field(default_factory=tuple)

    @property
    def total_rows_deleted(self) -> int:
        """Total athlete-scoped rows removed across every store (audit summary)."""
        return sum(n for store in self.stores for n in store.deleted_rows.values())

    @property
    def total_objects_deleted(self) -> int:
        """Total original-file objects whose bytes were deleted (PRIV-R11.3)."""
        return sum(store.deleted_objects for store in self.stores)


def _coerce_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """Coerce a string id to a UUID at the query boundary (portable UUID binds UUIDs)."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(value)


def _ordered_scoped_tables(
    metadata_tables_order: Sequence[Table],
    *,
    shared: frozenset[str],
    transitive: Mapping[str, _Transitive],
    delete_first: tuple[str, ...] = (),
) -> list[Table]:
    """Children-first delete order of every athlete-scoped table in one metadata.

    Order = the metadata's topological ``sorted_tables`` (parents-first) REVERSED, filtered to
    tables that are athlete-scoped: a direct ``athlete_id`` column, or a known transitive
    mapping. A table that is neither shared, directly scoped, nor transitively mapped raises —
    fail-closed against a NEW athlete-bearing table silently escaping erasure.

    ``delete_first`` names tables hoisted to the FRONT (in order): a no-FK child whose scope
    reads a parent that the bare topo order would delete first must run while that parent is
    still present (``stream_channel`` reads the stream-set tables it has no FK to).
    """
    ordered: list[Table] = []
    for table in reversed(list(metadata_tables_order)):
        name = table.name
        if name in shared:
            continue
        if "athlete_id" in table.columns or name in transitive or name in _AGENT_SUBJECT_SCOPED:
            ordered.append(table)
            continue
        raise RuntimeError(
            f"fail-closed: table {name!r} is neither shared, athlete-scoped, nor "
            "transitively mapped for per-athlete erasure (PRIV-R8)"
        )
    by_name = {t.name: t for t in ordered}
    hoisted = [by_name[name] for name in delete_first]
    remaining = [t for t in ordered if t.name not in set(delete_first)]
    return hoisted + remaining


async def _delete_direct(session: AsyncSession, table: Table, athlete_id: uuid.UUID) -> int:
    """Delete (and count) rows of an ``athlete_id``-bearing table for the athlete.

    The ``WHERE athlete_id = :id`` predicate excludes NULL-``athlete_id`` rows (e.g. the
    shared ``workout`` library templates), so shared rows are never erased.
    """
    column = table.c["athlete_id"]
    result = cast(
        CursorResult[Any],
        await session.execute(delete(table).where(column == athlete_id)),
    )
    return int(result.rowcount or 0)


async def _delete_subject_scoped(
    session: AsyncSession, table: Table, athlete_id: uuid.UUID, column_name: str
) -> int:
    """Delete (and count) operational auth rows scoped by the STRING ``subject`` column.

    The subject is the server-derived owner id rendered as a string (API-R23 token
    families / AUTH-R8 link challenges); NULL-subject rows (unbound challenges) carry no
    personal data and are left to expire.
    """
    column = table.c[column_name]
    result = cast(
        CursorResult[Any],
        await session.execute(delete(table).where(column == str(athlete_id))),
    )
    return int(result.rowcount or 0)


def _owned_id_subquery(
    hops: tuple[tuple[str, str, str], ...],
    athlete_id: uuid.UUID,
    tables: Mapping[str, Table],
) -> Select[Any]:
    """Build the subquery of athlete-owned ids for a transitive scope path.

    Hops are evaluated INNERMOST-first: the final hop's table carries ``athlete_id`` and is
    filtered by it; each earlier hop selects its ``select_column`` where its ``match_column``
    is ``IN`` the next-inner hop's ids. The outermost hop's ids are what the child column must
    match — resolved from LIVE parent rows, so deletion order of the parents is irrelevant.
    """
    inner: Select[Any] | None = None
    for hop_table, select_col, match_col in reversed(hops):
        parent = tables[hop_table]
        if inner is None:  # final (deepest) hop: filter by athlete_id directly
            inner = select(parent.c[select_col]).where(parent.c["athlete_id"] == athlete_id)
        else:
            inner = select(parent.c[select_col]).where(parent.c[match_col].in_(inner))
    if inner is None:  # an empty hop chain is a wiring bug, never a silent no-op (fail-closed)
        raise RuntimeError("fail-closed: transitive scope path has no hops (PRIV-R8)")
    return inner


async def _delete_transitive(
    session: AsyncSession,
    table: Table,
    athlete_id: uuid.UUID,
    spec: _Transitive,
    tables: Mapping[str, Table],
) -> int:
    """Delete rows scoped to the athlete via one or more parent-hop paths (no ``athlete_id``).

    For each scope path the athlete-owned parent ids are resolved by :func:`_owned_id_subquery`
    and the child rows whose ``child_column`` is ``IN`` that set are deleted. With no owned ids
    the subquery is empty and nothing is deleted (the athlete had no such rows). The paths
    resolve from LIVE parent rows, so a no-FK child (``stream_channel``) is erased correctly as
    long as it runs before its parents (it is hoisted to the front of the delete order).
    """
    deleted = 0
    for child_column, hops in spec.paths:
        owned_ids = _owned_id_subquery(hops, athlete_id, tables)
        result = cast(
            CursorResult[Any],
            await session.execute(delete(table).where(table.c[child_column].in_(owned_ids))),
        )
        deleted += int(result.rowcount or 0)
    return deleted


async def _erase_object_bytes(
    session: AsyncSession, athlete_id: uuid.UUID, object_store: ObjectStore
) -> int:
    """Delete the BYTES of every retained original-file object for the athlete (PRIV-R11.3).

    Reads the athlete's ``activity_file.object_ref`` handles BEFORE the relational rows are
    deleted, then deletes the object bytes (``ObjectStore.delete`` is idempotent). Returns the
    number of distinct objects deleted so the receipt proves the bytes are gone, not just the
    relational reference.
    """
    activity_file = Base.metadata.tables["activity_file"]
    refs = (
        (
            await session.execute(
                select(activity_file.c["object_ref"]).where(
                    activity_file.c["athlete_id"] == athlete_id
                )
            )
        )
        .scalars()
        .all()
    )
    distinct_refs = set(refs)
    for ref in distinct_refs:
        object_store.delete(ref)
    return len(distinct_refs)


async def _erase_store(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    *,
    store_name: str,
    order: Sequence[Table],
    tables: Mapping[str, Table],
    transitive: Mapping[str, _Transitive],
    deleted_objects: int,
) -> StoreErasureReport:
    """Delete every athlete-scoped row in one store, children-first, in this transaction.

    Iterates the precomputed children-first ``order``; each table is deleted directly (it has
    an ``athlete_id`` column) or transitively (via its parent keys). Records the per-table
    rowcount for the auditable receipt. The caller owns the commit/rollback boundary.
    """
    deleted_rows: dict[str, int] = {}
    for table in order:
        if table.name in _AGENT_SUBJECT_SCOPED:
            deleted_rows[table.name] = await _delete_subject_scoped(
                session, table, athlete_id, _AGENT_SUBJECT_SCOPED[table.name]
            )
        elif "athlete_id" in table.columns:
            deleted_rows[table.name] = await _delete_direct(session, table, athlete_id)
        else:
            deleted_rows[table.name] = await _delete_transitive(
                session, table, athlete_id, transitive[table.name], tables
            )
    return StoreErasureReport(
        store=store_name, deleted_rows=deleted_rows, deleted_objects=deleted_objects
    )


async def erase_athlete(
    athlete_id: str | uuid.UUID,
    *,
    canonical_session: AsyncSession,
    agent_state_session: AsyncSession,
    object_store: ObjectStore | None = None,
) -> ErasureReceipt:
    """Erase EVERY athlete-scoped record across both stores and return an audit receipt.

    The fulfilment of PRIV-R8 / PRIV-R11.3 / MEM-R3: it deletes the athlete's canonical
    records (activities + laps/streams/derived metrics, wellness, signatures, plans, goals,
    connections, source candidates, preferences, notifications, zones), the durable agent
    memory + checkpoints/threads/writes/interrupts, and — when ``object_store`` is supplied —
    the retained original-file object BYTES (PRIV-R11.3), then returns an
    :class:`ErasureReceipt` recording the residual-zero counts per store/table.

    Identity is the SERVER-DERIVED subject the caller resolved (AUTH-R18); this executor never
    reads identity from a model/tool/payload. The two stores are distinct databases on
    distinct pools (ARCH-R13), so each is erased inside ITS OWN transaction: object bytes first
    (so a handle is never dropped before its bytes), then the canonical rows are deleted and
    committed, then the agent-state rows are deleted and committed. On any error the active
    session rolls back (fail-closed); the receipt is only minted once both commits succeed.
    Idempotent: a re-run deletes nothing and returns all-zero counts.
    """
    aid = _coerce_uuid(athlete_id)

    canonical_tables = dict(Base.metadata.tables)
    canonical_order = _ordered_scoped_tables(
        Base.metadata.sorted_tables,
        shared=_CANONICAL_SHARED_TABLES,
        transitive=_CANONICAL_TRANSITIVE,
        delete_first=_CANONICAL_DELETE_FIRST,
    )
    agent_tables = dict(AgentStateBase.metadata.tables)
    agent_order = _ordered_scoped_tables(
        AgentStateBase.metadata.sorted_tables,
        shared=frozenset(),
        transitive=_AGENT_TRANSITIVE,
    )

    # Canonical store: delete object bytes first, then every athlete-scoped row, then commit.
    try:
        objects_deleted = (
            await _erase_object_bytes(canonical_session, aid, object_store)
            if object_store is not None
            else 0
        )
        canonical_report = await _erase_store(
            canonical_session,
            aid,
            store_name="canonical",
            order=canonical_order,
            tables=canonical_tables,
            transitive=_CANONICAL_TRANSITIVE,
            deleted_objects=objects_deleted,
        )
        await canonical_session.commit()
    except Exception:
        await canonical_session.rollback()
        raise

    # Agent-state store: separate database / pool / transaction (ARCH-R13).
    try:
        agent_report = await _erase_store(
            agent_state_session,
            aid,
            store_name="agent_state",
            order=agent_order,
            tables=agent_tables,
            transitive=_AGENT_TRANSITIVE,
            deleted_objects=0,
        )
        await agent_state_session.commit()
    except Exception:
        await agent_state_session.rollback()
        raise

    return ErasureReceipt(
        athlete_id=aid,
        completed_at=_dt.datetime.now(_dt.UTC),
        stores=(canonical_report, agent_report),
    )


__all__ = ["ErasureReceipt", "StoreErasureReport", "erase_athlete"]
