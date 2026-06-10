"""On-demand backfill: bounded oldest-first windows + resumable cursor (SYN-R5/SYN-R6).

Exercises :meth:`SyncOrchestrator.backfill` with an in-process window-recording fake:
the historical range is chunked into bounded windows walked OLDEST-FIRST with a
per-window commit (SYN-R5); the persisted range-scoped cursor makes an interrupted or
cancelled backfill RESUMABLE without re-downloading already-committed windows
(SYN-R6); a window failure stops the loop at the resume point; cancellation between
windows keeps committed windows; and a backfill of a DIFFERENT range never reuses the
cursor (skipping never-downloaded windows would silently lose data).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tests.integration._fake_capability import fake_capability
from tests.integration._session_provider import FactorySessionProvider
from tests.integration.test_sync import FakeApiAdapter, _cred_store, _seed_connection
from wattwise_core.ingestion.backfill import chunk_windows
from wattwise_core.ingestion.capability import CapabilityDescriptor
from wattwise_core.ingestion.registry import registry_from_adapters
from wattwise_core.ingestion.sync import SyncOrchestrator, SyncOutcome, SyncWindow
from wattwise_core.persistence.models import Base

UTC = _dt.UTC
_FIXED_NOW = _dt.datetime(2026, 6, 5, 9, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[Any]:
    """A transactional session-factory over a fresh single-connection canonical schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    @asynccontextmanager
    async def factory() -> AsyncIterator[AsyncSession]:
        async with sessionmaker() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    yield factory
    await engine.dispose()



class WindowRecordingAdapter(FakeApiAdapter):
    """A fake that records every fetched window and can fail/cancel a given window."""

    source_key: ClassVar[str] = "fake_api"
    capability: ClassVar[CapabilityDescriptor] = fake_capability("fake_api")

    def __init__(
        self, *, fail_on_window: int | None = None, cancel_on_window: int | None = None
    ) -> None:
        super().__init__()
        self.windows: list[SyncWindow] = []
        self.fail_on_window = fail_on_window
        self.cancel_on_window = cancel_on_window

    async def fetch(
        self, *, api_key: str | None, athlete_native_id: str | None, window: SyncWindow
    ) -> list[Any]:
        index = len(self.windows)
        self.windows.append(window)
        if self.fail_on_window is not None and index == self.fail_on_window:
            raise RuntimeError("window fetch exploded")
        if self.cancel_on_window is not None and index == self.cancel_on_window:
            raise asyncio.CancelledError
        return []  # no records needed; the windows walked are the assertion target


def _orch(session_factory: Any, adapter: Any, store: Any) -> SyncOrchestrator:
    return SyncOrchestrator(
        FactorySessionProvider(session_factory),
        registry=registry_from_adapters([adapter]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )


@pytest.mark.unit
def test_chunk_windows_is_bounded_contiguous_oldest_first() -> None:
    """The range chunks into bounded, contiguous, non-overlapping oldest-first windows."""
    wins = chunk_windows(SyncWindow("2026-01-01", "2026-03-15"), 30)
    assert wins[0] == SyncWindow("2026-01-01", "2026-01-30")
    assert wins[-1].newest == "2026-03-15"
    for prev, nxt in itertools.pairwise(wins):
        prev_end = _dt.date.fromisoformat(prev.newest)
        assert _dt.date.fromisoformat(nxt.oldest) == prev_end + _dt.timedelta(days=1)
        assert (prev_end - _dt.date.fromisoformat(prev.oldest)).days < 30
    with pytest.raises(ValueError):
        chunk_windows(SyncWindow("2026-02-01", "2026-01-01"), 30)


@pytest.mark.integration
async def test_backfill_walks_windows_oldest_first(session_factory: Any) -> None:
    """SYN-R5: the backfill fetches bounded windows OLDEST-FIRST across the range."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="fake_api", credential_ref=ref
    )
    adapter = WindowRecordingAdapter()

    run = await _orch(session_factory, adapter, store).backfill(
        athlete_id, window=SyncWindow("2026-01-01", "2026-03-01"), chunk_days=30, source="fake_api"
    )

    assert all(r.outcome is SyncOutcome.OK for r in run.results)
    # 2026-01-31 + 29 days = 2026-03-01, so the 60-day range chunks into TWO windows.
    assert [(w.oldest, w.newest) for w in adapter.windows] == [
        ("2026-01-01", "2026-01-30"),
        ("2026-01-31", "2026-03-01"),
    ]


@pytest.mark.integration
async def test_backfill_resumes_without_redownloading(session_factory: Any) -> None:
    """SYN-R6: a re-run of the same range skips every already-committed window."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="fake_api", credential_ref=ref
    )
    window = SyncWindow("2026-01-01", "2026-02-28")
    first = WindowRecordingAdapter(fail_on_window=1)  # window 0 commits, window 1 fails
    orch = _orch(session_factory, first, store)

    run1 = await orch.backfill(athlete_id, window=window, chunk_days=30, source="fake_api")

    outcomes = [r.outcome for r in run1.results]
    assert outcomes[0] is SyncOutcome.OK and outcomes[1] is SyncOutcome.DEGRADED
    assert len(run1.results) == 2  # the loop stopped at the failed window (resume point)

    second = WindowRecordingAdapter()
    run2 = await _orch(session_factory, second, store).backfill(
        athlete_id, window=window, chunk_days=30, source="fake_api"
    )
    # The committed window 0 is NEVER re-downloaded; the walk resumes at window 1.
    assert second.windows[0].oldest == "2026-01-31"
    assert all(r.outcome is SyncOutcome.OK for r in run2.results)


@pytest.mark.integration
async def test_backfill_cancellation_keeps_committed_windows(session_factory: Any) -> None:
    """SYN-R6/CLI-R5: cancelling mid-backfill keeps committed windows + the resume cursor."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="fake_api", credential_ref=ref
    )
    window = SyncWindow("2026-01-01", "2026-02-28")
    cancelling = WindowRecordingAdapter(cancel_on_window=1)
    orch = _orch(session_factory, cancelling, store)

    with pytest.raises(asyncio.CancelledError):
        await orch.backfill(athlete_id, window=window, chunk_days=30, source="fake_api")

    # Window 0 committed before the cancellation; resume skips it (no re-download).
    resumed = WindowRecordingAdapter()
    await _orch(session_factory, resumed, store).backfill(
        athlete_id, window=window, chunk_days=30, source="fake_api"
    )
    assert resumed.windows and resumed.windows[0].oldest == "2026-01-31"


@pytest.mark.integration
async def test_backfill_cursor_is_range_scoped(session_factory: Any) -> None:
    """A backfill of a DIFFERENT (older) range ignores the prior range's cursor.

    Reusing the cursor across ranges would skip windows that were NEVER downloaded —
    silent data loss; the cursor is scoped to the range it was advancing.
    """
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="fake_api", credential_ref=ref
    )
    orch_adapter = WindowRecordingAdapter()
    orch = _orch(session_factory, orch_adapter, store)
    await orch.backfill(
        athlete_id, window=SyncWindow("2026-01-01", "2026-02-28"), chunk_days=30, source="fake_api"
    )

    older = WindowRecordingAdapter()
    await _orch(session_factory, older, store).backfill(
        athlete_id, window=SyncWindow("2025-01-01", "2025-02-28"), chunk_days=30, source="fake_api"
    )

    # Every window of the older range was walked — none skipped via the other cursor.
    assert older.windows and older.windows[0].oldest == "2025-01-01"
    assert len(older.windows) == 2
