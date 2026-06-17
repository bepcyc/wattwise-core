"""Per-athlete erasure wipes EVERY store, including the original-file objects (PRIV-R8 / R11.3).

Cites: doc 70 PRIV-R7 (every category has a retention window), PRIV-R8 / PRIV-R8-AC ("right to
be forgotten": delete the athlete's canonical records, durable memory, agent checkpoints, raw
candidates, source secrets, AND every retained original recording file in the object store —
DELETING the object bytes, not merely the relational reference), PRIV-R11.3 (erasure deletes the
underlying object). Also doc 80 RAW-T-R2(e) (erasure deletes the object too, no orphan).

This is the executable form of PRIV-R8-AC: after an authenticated erasure request, ZERO residual
rows / memory entries / checkpoints / original-file objects remain for the athlete across all
stores — and a SECOND athlete's data is untouched (erasure is scoped to the one subject).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.memory import MemoryItem, MemoryItemKind, OssMemoryStore
from wattwise_core.agent.state_store import (
    AgentCheckpoint,
    AgentStateBase,
    AgentThread,
)
from wattwise_core.domain.enums import ActivityFileFormat, GboType
from wattwise_core.persistence.models import (
    Activity,
    ActivityFile,
    Athlete,
    Base,
    SourceCandidate,
    SourceDescriptor,
    Sport,
)
from wattwise_core.privacy.erasure import erase_athlete
from wattwise_core.storage import LocalObjectStore, content_hash

UTC = _dt.UTC

# Canonical tables that carry an ``athlete_id`` column and therefore hold personal data subject to
# per-athlete erasure (PRIV-R8). Derived from the mapped metadata so a new athlete-scoped table is
# covered automatically — the erasure scan is exhaustive by construction, not a hand-kept list.
_ATHLETE_SCOPED_TABLES = sorted(
    name
    for name, table in Base.metadata.tables.items()
    if "athlete_id" in table.columns and name != "athlete"
)


@pytest_asyncio.fixture
async def factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory over a fresh canonical + agent-state schema (one in-memory engine)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(AgentStateBase.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


async def _seed_athlete(
    session: AsyncSession, store: LocalObjectStore, *, file_bytes: bytes
) -> tuple[uuid.UUID, str]:
    """Seed one athlete with an activity, a candidate, memory, a checkpoint, and a stored file.

    Returns the athlete id and the object-store ref of the retained original file so the test can
    assert the bytes themselves are gone after erasure (PRIV-R11.3), not just the relational row.
    """
    # The cycling sport is a shared registry row, not athlete-scoped — seed it once.
    existing_sport = (
        await session.execute(select(Sport).where(Sport.sport_code == "cycling"))
    ).scalar_one_or_none()
    if existing_sport is None:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    descriptor = SourceDescriptor(
        source_key=f"file_import_{uuid.uuid4().hex[:8]}",
        display_name="Activity files",
        kind="file_upload",
    )
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add_all([descriptor, athlete])
    await session.flush()

    activity = Activity(
        athlete_id=athlete.athlete_id,
        start_time=_dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
        sport="cycling",
    )
    session.add(activity)
    await session.flush()

    # tier-1 original file: bytes in the object store, only an opaque ref in the relational store.
    object_ref = store.put(file_bytes, suffix=".fit")
    session.add(
        ActivityFile(
            activity_id=activity.activity_id,
            athlete_id=athlete.athlete_id,
            source_descriptor_id=descriptor.source_descriptor_id,
            object_ref=object_ref,
            format=ActivityFileFormat.FIT,
            byte_size=len(file_bytes),
            content_hash=content_hash(file_bytes),
        )
    )
    # tier-2 candidate (raw lineage envelope).
    session.add(
        SourceCandidate(
            athlete_id=athlete.athlete_id,
            source_descriptor_id=descriptor.source_descriptor_id,
            source_native_id="ride-1",
            gbo_type=GboType.ACTIVITY,
            content_hash=content_hash(file_bytes),
            resolved_activity_id=activity.activity_id,
        )
    )
    # durable athlete memory (special-category personalization).
    session.add(
        MemoryItem(
            athlete_id=athlete.athlete_id,
            kind=MemoryItemKind.CONSTRAINT,
            content="left knee niggle on hard efforts",
        )
    )
    # an agent-state thread + checkpoint owned by the athlete.
    thread_id = f"thread-{athlete.athlete_id}"
    session.add(
        AgentThread(
            thread_id=thread_id,
            athlete_id=athlete.athlete_id,
            conversation_id="conv-1",
        )
    )
    await session.flush()
    session.add(
        AgentCheckpoint(
            thread_id=thread_id,
            checkpoint_id="ckpt-1",
            athlete_id=athlete.athlete_id,
            schema_version=1,
            checkpoint_type="msgpack",
            checkpoint_blob=b"\x00\x01",
            metadata_blob={"step": 1},
        )
    )
    await session.commit()
    return athlete.athlete_id, object_ref


async def _erase_athlete(
    session: AsyncSession, store: LocalObjectStore, athlete_id: uuid.UUID
) -> None:
    """Drive the REAL production erasure executor (PRIV-R8), NOT a re-implemented delete loop.

    Previously this test asserted against a hand-rolled deletion loop, so a broken production
    ``erase_athlete`` (wrong delete order → FK violation, a missed transitive table, an undeleted
    object) still passed — it tested the test (issue #93). Now it calls the production executor
    directly: this single-DB integration carries BOTH the canonical (``Base``) and agent-state
    (``AgentStateBase``) tables on one engine, so the one session is passed as BOTH the canonical
    and agent-state session; the production executor owns the children-first ordering, transitive
    deletions, object-bytes-first sequencing, and its own commits.
    """
    await erase_athlete(
        athlete_id,
        canonical_session=session,
        agent_state_session=session,
        object_store=store,
    )


async def _count(session: AsyncSession, table_name: str, athlete_id: uuid.UUID) -> int:
    """Count rows for an athlete in a canonical athlete-scoped table."""
    table = Base.metadata.tables[table_name]
    stmt = select(func.count()).select_from(table).where(table.c.athlete_id == athlete_id)
    return int((await session.execute(stmt)).scalar_one())


async def _count_memory(session: AsyncSession, athlete_id: uuid.UUID) -> int:
    """Count durable memory rows for an athlete in the agent-state store (MEM-R3)."""
    stmt = select(func.count()).select_from(MemoryItem).where(MemoryItem.athlete_id == athlete_id)
    return int((await session.execute(stmt)).scalar_one())


@pytest.mark.integration
async def test_erasure_removes_all_personal_rows_and_objects(
    factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    """After erasure, zero athlete rows remain in any store and object bytes are gone (PRIV-R8)."""
    store = LocalObjectStore(tmp_path / "objects")
    async with factory() as session:
        athlete_id, object_ref = await _seed_athlete(session, store, file_bytes=b"FIT-bytes-1")

    # precondition: data exists across the stores before erasure.
    async with factory() as session:
        assert await _count(session, "activity", athlete_id) == 1
        assert await _count(session, "activity_file", athlete_id) == 1
        assert await _count(session, "source_candidate", athlete_id) == 1
        assert await _count_memory(session, athlete_id) == 1
        assert store.get(object_ref) == b"FIT-bytes-1"

    async with factory() as session:
        await _erase_athlete(session, store, athlete_id)

    # postcondition: every athlete-scoped canonical table is empty for the athlete (PRIV-R8-AC).
    async with factory() as session:
        for table_name in _ATHLETE_SCOPED_TABLES:
            assert await _count(session, table_name, athlete_id) == 0, table_name
        # the athlete profile itself is gone.
        gone = (
            await session.execute(select(Athlete).where(Athlete.athlete_id == athlete_id))
        ).scalar_one_or_none()
        assert gone is None
        # agent-state checkpoints + threads gone.
        ckpts = (
            await session.execute(
                select(func.count())
                .select_from(AgentCheckpoint)
                .where(AgentCheckpoint.athlete_id == athlete_id)
            )
        ).scalar_one()
        assert ckpts == 0
        threads = (
            await session.execute(
                select(func.count())
                .select_from(AgentThread)
                .where(AgentThread.athlete_id == athlete_id)
            )
        ).scalar_one()
        assert threads == 0

    # the original-file OBJECT bytes themselves are deleted (PRIV-R11.3), not just the ref.
    with pytest.raises(KeyError):
        store.get(object_ref)


@pytest.mark.integration
async def test_erasure_is_scoped_to_the_one_athlete(
    factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    """Erasing one athlete leaves a second athlete's rows, memory, and objects intact (PRIV-R8)."""
    store = LocalObjectStore(tmp_path / "objects")
    async with factory() as session:
        victim_id, victim_ref = await _seed_athlete(session, store, file_bytes=b"victim-bytes")
        keeper_id, keeper_ref = await _seed_athlete(session, store, file_bytes=b"keeper-bytes")

    async with factory() as session:
        await _erase_athlete(session, store, victim_id)

    async with factory() as session:
        # the kept athlete is fully intact.
        assert await _count(session, "activity", keeper_id) == 1
        assert await _count_memory(session, keeper_id) == 1
        store_async = OssMemoryStore(session)
        kept = await store_async.fetch_relevant(athlete_id=str(keeper_id), query="knee")
        assert len(kept) == 1
        # the erased athlete is gone everywhere.
        assert await _count(session, "activity", victim_id) == 0
        assert await _count_memory(session, victim_id) == 0

    # the keeper's object survives; the victim's object is deleted.
    assert store.get(keeper_ref) == b"keeper-bytes"
    with pytest.raises(KeyError):
        store.get(victim_ref)
