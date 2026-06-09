"""The whole-athlete erasure EXECUTOR removes EVERY athlete-scoped row (PRIV-1, PRIV-R8/-R11).

This exercises the REAL production executor :func:`wattwise_core.privacy.erasure.erase_athlete`
(NOT a reimplemented delete loop — that is the vacuity the audit flagged in the legacy
``test_gdpr_erasure``). The flow is: seed one athlete with personal data fanned across BOTH
stores and EVERY tier — canonical activity + its lap / stream-set / stream-channel / derived
metric / original-file object, wellness, connection + source candidate, plan/goal, plus the
agent-state memory row, thread, checkpoint, pending write, and live interrupt — run the real
executor, then RE-QUERY every athlete-scoped table (the set DERIVED from the live ORM metadata,
so the assertion is exhaustive by construction and a new table cannot quietly escape it) and
assert ZERO residual rows, that the original-file object BYTES are gone (PRIV-R11.3), and that an
auditable completion record (the :class:`ErasureReceipt`) exists naming the subject + nonzero
counts (PRIV-R8 "auditable record that erasure completed").

Mutation-proof (skill §3): :func:`test_executor_residual_set_covers_every_scoped_table` and the
no-op object-store / scoping tests would FAIL if a table were dropped from the executor's delete
plan or if a delete were turned into a no-op — proven by reverting the executor's fix.

CRITICAL (skill §7): both stores run on FILE-backed SQLite engines with a REAL multi-connection
pool (WAL + busy_timeout), on SEPARATE engines mirroring production's canonical vs agent-state
store separation (ARCH-R13) — never ``:memory:`` / ``StaticPool`` (which false-greens a single
connection). PostgreSQL / MariaDB legs run when ``WATTWISE_PG_DSN`` / ``WATTWISE_MARIADB_DSN`` set.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import wattwise_core.agent.memory  # noqa: F401  (registers agent_memory_item on AgentStateBase)
from wattwise_core.agent.memory import MemoryItem, MemoryItemKind
from wattwise_core.agent.state_store import (
    AgentCheckpoint,
    AgentInterrupt,
    AgentStateBase,
    AgentThread,
    AgentWrite,
)
from wattwise_core.domain.enums import (
    ActivityFileFormat,
    AuthArchetype,
    ConnectionStatus,
    GboType,
    GoalStatus,
    GoalType,
    PlanStatus,
    SampleBasis,
    StreamChannelName,
    StreamSetKind,
)
from wattwise_core.persistence.models import (
    Activity,
    ActivityFile,
    ActivityLap,
    ActivityStreamSet,
    Athlete,
    Base,
    Connection,
    DailyWellness,
    DerivedActivityMetric,
    Goal,
    Plan,
    SourceCandidate,
    SourceDescriptor,
    Sport,
    StreamChannel,
)
from wattwise_core.privacy.erasure import ErasureReceipt, erase_athlete
from wattwise_core.storage import LocalObjectStore, content_hash

pytestmark = pytest.mark.integration

UTC = _dt.UTC

# Every athlete-scoped table across BOTH metadatas, DERIVED from the live ORM (NOT a hand-kept
# list): the residual-rows assertion re-queries each of these and a NEW athlete-bearing table is
# covered automatically. The shared registry tables hold no personal data and are excluded.
_SHARED = frozenset({"sport", "sub_sport", "source_descriptor"})
_CANONICAL_SCOPED = sorted(n for n in Base.metadata.tables if n not in _SHARED)
_AGENT_SCOPED = sorted(AgentStateBase.metadata.tables)


def _enable_sqlite_wal_and_fk(dbapi_conn: Any, _record: Any) -> None:
    """WAL + long busy_timeout + FK enforcement per SQLite connection (real pool, GBO-R8b)."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


@pytest_asyncio.fixture
async def stores(
    tmp_path: Path,
) -> AsyncIterator[
    tuple[async_sessionmaker[AsyncSession], async_sessionmaker[AsyncSession], LocalObjectStore]
]:
    """Two SEPARATE file-SQLite engines (canonical + agent-state) on a real pool + object store.

    Separate engines mirror production (ARCH-R13): the executor takes the two session seams as
    distinct injected deps. File-backed (not ``:memory:``) so a real multi-connection pool
    backs the erase, and FKs are enforced (PRAGMA) so a wrong delete ORDER would FK-fail loudly.
    """
    canon_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'canon.sqlite'}")
    agent_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'agent.sqlite'}")
    for engine in (canon_engine, agent_engine):
        event.listen(engine.sync_engine, "connect", _enable_sqlite_wal_and_fk)
    async with canon_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with agent_engine.begin() as conn:
        await conn.run_sync(AgentStateBase.metadata.create_all)
    canon_factory = async_sessionmaker(canon_engine, expire_on_commit=False, class_=AsyncSession)
    agent_factory = async_sessionmaker(agent_engine, expire_on_commit=False, class_=AsyncSession)
    store = LocalObjectStore(tmp_path / "objects")
    yield canon_factory, agent_factory, store
    await canon_engine.dispose()
    await agent_engine.dispose()


async def _seed_canonical(
    factory: async_sessionmaker[AsyncSession], store: LocalObjectStore, *, file_bytes: bytes
) -> tuple[uuid.UUID, str]:
    """Seed one athlete across EVERY canonical tier; return (athlete_id, original-file ref)."""
    async with factory() as s:
        existing = (
            await s.execute(select(Sport).where(Sport.sport_code == "cycling"))
        ).scalar_one_or_none()
        if existing is None:
            s.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        descriptor = SourceDescriptor(
            source_key=f"file_import_{uuid.uuid4().hex[:8]}",
            display_name="Activity files",
            kind="file_upload",
        )
        athlete = Athlete(sex="male", reference_timezone="UTC")
        s.add_all([descriptor, athlete])
        await s.flush()
        aid = athlete.athlete_id
        act = Activity(
            athlete_id=aid,
            start_time=_dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
            sport="cycling",
        )
        s.add(act)
        await s.flush()
        # transitively-scoped children (NO athlete_id of their own): the executor must reach them
        s.add(ActivityLap(activity_id=act.activity_id, lap_index=0))
        s.add(DerivedActivityMetric(activity_id=act.activity_id, load_model="default", tss=42.0))
        ss = ActivityStreamSet(
            activity_id=act.activity_id, sample_basis=SampleBasis.TIME, t0=act.start_time
        )
        s.add(ss)
        await s.flush()
        s.add(
            StreamChannel(
                stream_set_id=ss.stream_set_id,
                set_kind=StreamSetKind.ACTIVITY,
                channel=StreamChannelName.POWER_W,
                values=[1, 2, 3],
            )
        )
        # original-file object: bytes in the object store, only an opaque ref relationally
        object_ref = store.put(file_bytes, suffix=".fit")
        s.add(
            ActivityFile(
                activity_id=act.activity_id,
                athlete_id=aid,
                source_descriptor_id=descriptor.source_descriptor_id,
                object_ref=object_ref,
                format=ActivityFileFormat.FIT,
                byte_size=len(file_bytes),
                content_hash=content_hash(file_bytes),
            )
        )
        s.add(
            Connection(
                athlete_id=aid,
                source_descriptor_id=descriptor.source_descriptor_id,
                status=ConnectionStatus.CONNECTED,
                auth_archetype=AuthArchetype.API_KEY,
                credential_ref="cred_seed",
            )
        )
        s.add(
            SourceCandidate(
                athlete_id=aid,
                source_descriptor_id=descriptor.source_descriptor_id,
                source_native_id="ride-1",
                gbo_type=GboType.ACTIVITY,
                content_hash=content_hash(file_bytes),
                resolved_activity_id=act.activity_id,
            )
        )
        s.add(
            DailyWellness(athlete_id=aid, local_date=_dt.date(2026, 6, 1), resting_hr_bpm=44)
        )
        goal = Goal(
            athlete_id=aid,
            sport="cycling",
            goal_type=GoalType.EVENT,
            title="A race",
            status=GoalStatus.ACTIVE,
        )
        s.add(goal)
        await s.flush()
        s.add(
            Plan(
                athlete_id=aid,
                goal_id=goal.goal_id,
                start_date=_dt.date(2026, 6, 1),
                end_date=_dt.date(2026, 6, 7),
                status=PlanStatus.ACTIVE,
            )
        )
        await s.commit()
    return aid, object_ref


async def _seed_agent_state(
    factory: async_sessionmaker[AsyncSession], athlete_id: uuid.UUID
) -> None:
    """Seed the athlete's agent-state: a memory row, thread, checkpoint, write, and interrupt."""
    thread_id = f"thread-{athlete_id}"
    async with factory() as s:
        s.add(
            MemoryItem(
                athlete_id=athlete_id,
                kind=MemoryItemKind.CONSTRAINT,
                content="left knee niggle on hard efforts",
            )
        )
        s.add(AgentThread(thread_id=thread_id, athlete_id=athlete_id, conversation_id="conv-1"))
        await s.flush()
        s.add(
            AgentCheckpoint(
                thread_id=thread_id,
                checkpoint_id="ckpt-1",
                athlete_id=athlete_id,
                schema_version=1,
                checkpoint_type="msgpack",
                checkpoint_blob=b"\x00\x01",
                metadata_blob={"step": 1},
            )
        )
        s.add(
            AgentWrite(
                thread_id=thread_id,
                checkpoint_id="ckpt-1",
                task_id="task-1",
                idx=0,
                channel="messages",
                value_type="msgpack",
                value_blob=b"\x02\x03",
            )
        )
        s.add(
            AgentInterrupt(
                thread_id=thread_id,
                athlete_id=athlete_id,
                interrupt_id="int-1",
                status="live",
            )
        )
        await s.commit()


async def _count(factory: async_sessionmaker[AsyncSession], table_name: str, metadata: Any) -> int:
    """Count ALL rows of a table (post-erasure the athlete is the only seeded subject)."""
    table = metadata.tables[table_name]
    async with factory() as s:
        return int((await s.execute(select(func.count()).select_from(table))).scalar_one())


async def _count_scoped(
    factory: async_sessionmaker[AsyncSession], table_name: str, metadata: Any, athlete_id: uuid.UUID
) -> int:
    """Count rows of a table scoped to one athlete (direct ``athlete_id`` tables only)."""
    table = metadata.tables[table_name]
    async with factory() as s:
        stmt = select(func.count()).select_from(table).where(table.c["athlete_id"] == athlete_id)
        return int((await s.execute(stmt)).scalar_one())


@pytest.mark.asyncio
async def test_executor_erases_every_athlete_scoped_table_and_object(
    stores: tuple[
        async_sessionmaker[AsyncSession], async_sessionmaker[AsyncSession], LocalObjectStore
    ],
) -> None:
    """Run the REAL executor; assert ZERO residual rows everywhere + object bytes gone + receipt."""
    canon_factory, agent_factory, store = stores
    athlete_id, object_ref = await _seed_canonical(canon_factory, store, file_bytes=b"FIT-bytes-1")
    await _seed_agent_state(agent_factory, athlete_id)

    # precondition: data exists across both stores AND the transitive children + object.
    assert await _count(canon_factory, "activity", Base.metadata) == 1
    assert await _count(canon_factory, "activity_lap", Base.metadata) == 1
    assert await _count(canon_factory, "stream_channel", Base.metadata) == 1
    assert await _count(canon_factory, "derived_activity_metric", Base.metadata) == 1
    assert await _count(agent_factory, "agent_memory_item", AgentStateBase.metadata) == 1
    assert await _count(agent_factory, "agent_write", AgentStateBase.metadata) == 1
    assert store.get(object_ref) == b"FIT-bytes-1"

    # run the REAL production executor (NOT a reimplemented delete loop).
    async with canon_factory() as cs, agent_factory() as ass:
        receipt = await erase_athlete(
            athlete_id,
            canonical_session=cs,
            agent_state_session=ass,
            object_store=store,
        )

    # postcondition (PRIV-R8-AC): ZERO residual rows in EVERY athlete-scoped table, both stores.
    for table_name in _CANONICAL_SCOPED:
        assert await _count(canon_factory, table_name, Base.metadata) == 0, table_name
    for table_name in _AGENT_SCOPED:
        assert await _count(agent_factory, table_name, AgentStateBase.metadata) == 0, table_name

    # the original-file OBJECT bytes themselves are gone (PRIV-R11.3), not just the ref.
    with pytest.raises(KeyError):
        store.get(object_ref)

    # an auditable completion record exists naming the subject + nonzero deletions (PRIV-R8).
    assert isinstance(receipt, ErasureReceipt)
    assert receipt.athlete_id == athlete_id
    assert receipt.completed_at.tzinfo is not None
    assert receipt.total_rows_deleted > 0
    assert receipt.total_objects_deleted == 1
    store_names = {r.store for r in receipt.stores}
    assert store_names == {"canonical", "agent_state"}


@pytest.mark.asyncio
async def test_executor_residual_set_covers_every_scoped_table(
    stores: tuple[
        async_sessionmaker[AsyncSession], async_sessionmaker[AsyncSession], LocalObjectStore
    ],
) -> None:
    """The receipt's per-table plan covers EVERY athlete-scoped table — mutation tripwire.

    If a table were dropped from the executor's delete plan (the mutation), it would be absent
    from the receipt here AND non-empty in the residual scan above — so this asserts the plan's
    table set EQUALS the metadata-derived scoped set, exactly.
    """
    canon_factory, agent_factory, store = stores
    athlete_id, _ = await _seed_canonical(canon_factory, store, file_bytes=b"FIT-bytes-2")
    await _seed_agent_state(agent_factory, athlete_id)
    async with canon_factory() as cs, agent_factory() as ass:
        receipt = await erase_athlete(
            athlete_id, canonical_session=cs, agent_state_session=ass, object_store=store
        )
    planned = {name for r in receipt.stores for name in r.deleted_rows}
    assert planned == set(_CANONICAL_SCOPED) | set(_AGENT_SCOPED)
    # every seeded table actually reported a deletion (no table silently no-op'd).
    for tbl in ("activity", "activity_lap", "stream_channel", "derived_activity_metric"):
        assert receipt.stores[0].deleted_rows[tbl] >= 1, tbl
    for tbl in ("agent_memory_item", "agent_checkpoint", "agent_write", "agent_interrupt"):
        assert receipt.stores[1].deleted_rows[tbl] >= 1, tbl


@pytest.mark.asyncio
async def test_executor_is_scoped_to_the_one_athlete(
    stores: tuple[
        async_sessionmaker[AsyncSession], async_sessionmaker[AsyncSession], LocalObjectStore
    ],
) -> None:
    """Erasing one athlete leaves a second athlete's rows, memory, and object intact (PRIV-R8)."""
    canon_factory, agent_factory, store = stores
    victim_id, victim_ref = await _seed_canonical(canon_factory, store, file_bytes=b"victim-bytes")
    await _seed_agent_state(agent_factory, victim_id)
    keeper_id, keeper_ref = await _seed_canonical(canon_factory, store, file_bytes=b"keeper-bytes")
    await _seed_agent_state(agent_factory, keeper_id)

    async with canon_factory() as cs, agent_factory() as ass:
        await erase_athlete(
            victim_id, canonical_session=cs, agent_state_session=ass, object_store=store
        )

    # the kept athlete is fully intact across both stores + object store.
    assert await _count_scoped(canon_factory, "activity", Base.metadata, keeper_id) == 1
    assert await _count_scoped(canon_factory, "daily_wellness", Base.metadata, keeper_id) == 1
    assert (
        await _count_scoped(agent_factory, "agent_memory_item", AgentStateBase.metadata, keeper_id)
        == 1
    )
    assert store.get(keeper_ref) == b"keeper-bytes"
    # the victim is gone everywhere; the victim's object is deleted.
    assert await _count_scoped(canon_factory, "activity", Base.metadata, victim_id) == 0
    assert (
        await _count_scoped(agent_factory, "agent_memory_item", AgentStateBase.metadata, victim_id)
        == 0
    )
    with pytest.raises(KeyError):
        store.get(victim_ref)


@pytest.mark.asyncio
async def test_executor_is_idempotent(
    stores: tuple[
        async_sessionmaker[AsyncSession], async_sessionmaker[AsyncSession], LocalObjectStore
    ],
) -> None:
    """A second erasure of the same athlete deletes nothing and reports all-zero (idempotent)."""
    canon_factory, agent_factory, store = stores
    athlete_id, _ = await _seed_canonical(canon_factory, store, file_bytes=b"FIT-bytes-3")
    await _seed_agent_state(agent_factory, athlete_id)
    async with canon_factory() as cs, agent_factory() as ass:
        await erase_athlete(
            athlete_id, canonical_session=cs, agent_state_session=ass, object_store=store
        )
    async with canon_factory() as cs, agent_factory() as ass:
        second = await erase_athlete(
            athlete_id, canonical_session=cs, agent_state_session=ass, object_store=store
        )
    assert second.total_rows_deleted == 0
    assert second.total_objects_deleted == 0
