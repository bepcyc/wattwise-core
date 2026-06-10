"""Property: the goals list query is athlete-scoped and status-filter-exact (API-R35 / API-R51).

Where the integration tests pin specific cases, this asserts the INVARIANTS over generated stores:

    For a generated set of goals split across the owner and a foreign athlete, across a generated
    set of statuses, the owner-scoped list query (the one the ``/v1/goals`` GET drives):
      * returns ONLY the owner's goals — a foreign athlete's goals are NEVER returned, however many
        exist (athlete scoping, AUTH-R3 / API-R51);
      * when filtered by a status, returns EXACTLY the owner's goals carrying that status — no more,
        no fewer (typed status filter, API-R35);
      * with no filter, returns EXACTLY the owner's full goal set.

The query is driven against a fresh real store per example (file-SQLite, isolated tmp file) through
the SAME ``_query_goals`` the router uses, so the property holds over the real SQL, not a model. A
broken athlete scope (dropping the ``athlete_id`` predicate) or a broken status filter would
immediately falsify these.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.api.routers.goals import _query_goals
from wattwise_core.domain.enums import GoalStatus, GoalType
from wattwise_core.persistence.models import Athlete, Base, Goal, Sport

pytestmark = pytest.mark.property

UTC = _dt.UTC
_OWNER = uuid.UUID("00000000-0000-7000-8000-0000000000b1")
_FOREIGN = uuid.UUID("00000000-0000-7000-8000-0000000000b2")
_STATUSES = list(GoalStatus)
_CURSOR_KEY = "property-cursor-key-0123456789ab"

_status_strat = st.sampled_from(_STATUSES)
# A list of (status) for the owner + a list for the foreign athlete; bounded for cost (ANL-R30).
_goals_strat = st.lists(_status_strat, min_size=0, max_size=8)


def _enable_wal(dbapi_conn: object, _record: object) -> None:
    cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


async def _seed(session: AsyncSession, owner: list[GoalStatus], foreign: list[GoalStatus]) -> None:
    """Seed the registry, both athletes, and one goal per generated status for each."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    for aid in (_OWNER, _FOREIGN):
        session.add(Athlete(athlete_id=aid, sex="male", reference_timezone="UTC"))
    await session.flush()
    for i, status in enumerate(owner):
        session.add(_goal(_OWNER, status, f"owner-{i}"))
    for i, status in enumerate(foreign):
        session.add(_goal(_FOREIGN, status, f"foreign-{i}"))
    await session.commit()


def _goal(athlete_id: uuid.UUID, status: GoalStatus, label: str) -> Goal:
    return Goal(
        goal_id=uuid.uuid4(),
        athlete_id=athlete_id,
        sport="cycling",
        goal_type=GoalType.EVENT,
        title=label,
        status=status,
        target_date=_dt.date(2026, 6, 1),
    )


async def _run(owner: list[GoalStatus], foreign: list[GoalStatus], tmp: Path) -> None:
    dsn = f"sqlite+aiosqlite:///{tmp}/goals_prop_{uuid.uuid4().hex}.sqlite"
    engine = create_async_engine(dsn)
    event.listen(engine.sync_engine, "connect", _enable_wal)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        async with factory() as session:
            await _seed(session, owner, foreign)
        async with factory() as session:
            # No filter: exactly the owner's full set, never a foreign goal.
            allrows = await _query_goals(
                session,
                str(_OWNER),
                status=None,
                sport=None,
                frm=None,
                to=None,
                sort="created_at",
                order="asc",
                cursor=None,
                key=_CURSOR_KEY,
                limit=1000,
            )
            assert all(r.athlete_id == _OWNER for r in allrows)
            assert len(allrows) == len(owner)
            # Per-status filter: exactly the owner's goals carrying that status.
            for status in _STATUSES:
                rows = await _query_goals(
                    session,
                    str(_OWNER),
                    status=status,
                    sport=None,
                    frm=None,
                    to=None,
                    sort="created_at",
                    order="asc",
                    cursor=None,
                    key=_CURSOR_KEY,
                    limit=1000,
                )
                assert all(r.athlete_id == _OWNER and r.status is status for r in rows)
                assert len(rows) == sum(1 for s in owner if s is status)
    finally:
        await engine.dispose()


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(owner=_goals_strat, foreign=_goals_strat)
def test_list_is_athlete_scoped_and_status_filter_exact(
    owner: list[GoalStatus], foreign: list[GoalStatus], tmp_path: Path
) -> None:
    """Owner-scoped + status-exact over generated stores (athlete scoping + typed filter)."""
    asyncio.run(_run(owner, foreign, tmp_path))
