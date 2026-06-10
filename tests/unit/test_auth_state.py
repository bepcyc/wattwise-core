"""Unit tests for the refresh-token family state machine (API-R23 / AUTH-R9).

Pins the fail-closed expiry contract of :func:`consume_refresh_token`:

* an ALREADY-expired presented token is ``invalid`` and mints nothing;
* a token that expires BETWEEN the validity read and the guarded used-marking UPDATE
  can never mint a successor — the claim guard re-asserts ``expires_at`` itself, so
  the lapsed claim fails closed instead of rotating a dead credential.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent import auth_state
from wattwise_core.agent.auth_state import (
    AuthRefreshToken,
    consume_refresh_token,
    issue_refresh_token,
)
from wattwise_core.agent.state_store import AgentStateBase
from wattwise_core.persistence.types import utcnow


@pytest_asyncio.fixture
async def session(tmp_path) -> AsyncIterator[AsyncSession]:
    """One session over a fresh FILE-backed agent-state schema (never the canonical)."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'auth_state.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(AgentStateBase.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        yield sess
    await engine.dispose()


async def test_expired_token_is_invalid_and_mints_nothing(session: AsyncSession) -> None:
    """A presented token past its ``expires_at`` reads ``invalid`` — no successor row."""
    secret = await issue_refresh_token(session, subject="owner", scopes=("read",), ttl_seconds=-1)
    outcome = await consume_refresh_token(session, presented=secret, ttl_seconds=3600)
    assert outcome.status == "invalid"
    assert outcome.new_secret is None
    rows = (await session.execute(select(AuthRefreshToken))).scalars().all()
    assert len(rows) == 1  # only the original row — nothing was minted


async def test_expiry_between_check_and_claim_never_mints_a_successor(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Token expires between check and consume → the guarded claim refuses to rotate.

    A stepping clock models the exact window the UPDATE's own ``expires_at`` predicate
    must close: the read-side validity check sees ``now`` (the token still looks live),
    while the guarded used-marking UPDATE evaluates AFTER expiry (fail-closed, API-R23).
    """
    # Issue under the REAL clock — only the consume path runs on the stepping clock.
    secret = await issue_refresh_token(session, subject="owner", scopes=("read",), ttl_seconds=600)
    real_now = utcnow()

    def _steps() -> Iterator[_dt.datetime]:
        yield real_now  # call 1: the read-side `row.expires_at < utcnow()` check passes
        while True:  # every later call (the claim UPDATE onward) is past expiry
            yield real_now + _dt.timedelta(hours=1)

    ticks = _steps()
    monkeypatch.setattr(auth_state, "utcnow", lambda: next(ticks))
    outcome = await consume_refresh_token(session, presented=secret, ttl_seconds=3600)
    assert outcome.status != "ok"  # never a successful rotation off a dead credential
    assert outcome.new_secret is None
    rows = (await session.execute(select(AuthRefreshToken))).scalars().all()
    assert len(rows) == 1  # NO successor row was minted in the family
