"""Import-job bookkeeping is idempotent on a re-upload (ING-R6 / FIL-R5, API-R33).

A retrying or double-clicking user sends the same file twice. Once the first upload has
landed its canonical activity, the upsert re-derives the SAME activity id and thus the
SAME ``import_job_id`` — so the second upload's bookkeeping INSERT collides on the
primary key. ``_record_job`` MUST absorb that collision and converge, not surface a raw
``500`` (the onboarding papercut this regression guards). Sequential, single-session —
no concurrency claim, so ``:memory:`` is fine here (no race to hide).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.ops_jobs import ImportJobRecord
from wattwise_core.agent.state_store import AgentStateBase
from wattwise_core.api.routers.imports import ImportJob, _record_job

pytestmark = pytest.mark.unit

_ATHLETE = uuid.UUID("0c2742a2-cc08-5d7c-8c43-0898ccbb1392")
_JOB_ID = "019ec644-2710-7000-87b9-472080722e22"


def _job() -> ImportJob:
    """A freshly-queued ImportJob identical to what the processor returns on accept."""
    return ImportJob(
        import_job_id=_JOB_ID,
        status="queued",
        filename="ride.fit",
        received_at=datetime.now(UTC),
        status_text="We've got your file and we're bringing it in.",
    )


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory over a fresh in-memory agent-state schema (ARCH-R13 store)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(AgentStateBase.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


async def test_record_job_inserts_first_upload(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The first upload of a file records exactly one job row (the happy path)."""
    async with session_factory() as session, session.begin():
        await _record_job(session, str(_ATHLETE), _job())
    async with session_factory() as session:
        rows = (await session.execute(select(ImportJobRecord))).scalars().all()
    assert len(rows) == 1
    assert rows[0].import_job_id == _JOB_ID


async def test_record_job_is_idempotent_on_reupload(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A re-upload whose job id already exists converges, not a 500 (ING-R6 / FIL-R5).

    The activity has landed (post-sync), so the upsert re-derives the same activity id
    and the second upload carries the SAME import_job_id. The bookkeeping INSERT must
    absorb the primary-key collision and leave exactly one row — never raise, never a
    500 (the onboarding retry/double-click papercut).
    """
    async with session_factory() as session, session.begin():
        # First upload: the job lands.
        await _record_job(session, str(_ATHLETE), _job())
    async with session_factory() as session, session.begin():
        # Second upload: same job id (activity-derived) → would collide without idempotency.
        await _record_job(session, str(_ATHLETE), _job())
    async with session_factory() as session:
        rows = (await session.execute(select(ImportJobRecord))).scalars().all()
    assert len(rows) == 1, "re-upload must not duplicate the job row"
    assert rows[0].import_job_id == _JOB_ID


async def test_record_job_collision_does_not_poison_outer_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A colliding re-upload rolls back only the duplicate add, not the outer work.

    The SAVEPOINT (begin_nested) must confine the IntegrityError so a caller that did
    other writes in the same transaction is not left with a poisoned session.
    """
    async with session_factory() as session, session.begin():
        # An unrelated write in the SAME transaction.
        other = ImportJobRecord(
            import_job_id="019ec644-9999-7000-aaaa-222222222222",
            athlete_id=_ATHLETE,
            status="done",
            filename="other.fit",
            status_text="Done.",
            received_at=datetime.now(UTC),
        )
        session.add(other)
        # The colliding re-upload inside the same transaction.
        await _record_job(session, str(_ATHLETE), _job())
    async with session_factory() as session:
        rows = (await session.execute(select(ImportJobRecord))).scalars().all()
    ids = {r.import_job_id for r in rows}
    assert _JOB_ID in ids and "019ec644-9999-7000-aaaa-222222222222" in ids
