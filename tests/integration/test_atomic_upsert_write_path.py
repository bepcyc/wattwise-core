"""Atomic-upsert canonical write-path conformance (UPS-R2, ING-UPS-R1/R3, PERF-R1).

Pins the foundation that the entire canonical write path is an atomic insert-or-update
through ``persistence/upsert.py`` — never check-then-write — and that bulk ingestion is
batched (single round-trip per batch), fault-isolated per record, and never rolls the
whole source batch back on one bad row:

* UPS-R2     — two concurrent ingest runs landing the SAME natural key over a real
  multi-connection pool produce exactly ONE row, with no check-then-write race.
* PERF-R1    — a bulk import issues O(activities / batch_size) write round-trips, not a
  per-row insert loop (asserted by counting INSERT statements).
* ING-UPS-R1 — candidates are upserted by their candidate key (re-ingest never duplicates).
* ING-UPS-R3 — one failing record is isolated; committed records survive (no whole-run
  rollback on a single record failure).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity
from wattwise_core.ingestion import _ingest_steps as ingest_steps_mod
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    Base,
    SourceCandidate,
    SourceDescriptor,
    Sport,
)
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.integration

UTC = _dt.UTC
_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)


def _enable_wal(dbapi_conn: object, _record: object) -> None:
    """WAL + busy_timeout per connection so concurrent runs hit a REAL pool, not StaticPool."""
    cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


@pytest_asyncio.fixture
async def pool(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A REAL file-SQLite QueuePool (WAL + busy_timeout) — never :memory:/StaticPool."""
    dsn = f"sqlite+aiosqlite:///{tmp_path}/upsert_{uuid.uuid4().hex}.sqlite"
    engine = create_async_engine(dsn)
    event.listen(engine.sync_engine, "connect", _enable_wal)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield factory
    finally:
        await engine.dispose()


async def _seed(factory: async_sessionmaker[AsyncSession]) -> tuple[str, str]:
    """Seed the cycling sport, an athlete, and one source descriptor."""
    async with factory() as s:
        s.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        athlete = Athlete(sex="male", reference_timezone="UTC")
        descriptor = SourceDescriptor(
            source_key="other_src", display_name="Other", kind="oauth_api"
        )
        s.add_all([athlete, descriptor])
        await s.flush()
        ids = (str(athlete.athlete_id), str(descriptor.source_descriptor_id))
        await s.commit()
    return ids


def _ride(
    native_id: str, *, watts: float = 200.0, seconds: int = 1800, start: _dt.datetime = _START
) -> GboCandidate:
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(f"{native_id}:{watts}:{seconds}".encode()),
        payload={
            "start_time": start,
            "sport": "cycling",
            "elapsed_time_s": seconds,
            "avg_power_w": watts,
        },
        trust_tier=Fidelity.RAW_STREAM,
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )


# --------------------------------------------------------------- UPS-R2: no race


async def test_concurrent_same_key_ingest_is_one_row_no_race(
    pool: async_sessionmaker[AsyncSession],
) -> None:
    """Two concurrent runs landing the SAME candidate key produce ONE row (UPS-R2).

    Over a real multi-connection pool, the atomic insert-or-update means there is no
    time-of-check/time-of-use window: a SELECT-then-add path would let both runs insert
    and either duplicate or deadlock; the atomic seam yields exactly one canonical row.
    """
    athlete_id, descriptor = await _seed(pool)
    cand = _ride("ride-shared")

    async def run() -> None:
        async with pool() as session:
            await IngestService(session).ingest(athlete_id, descriptor, [cand])
            await session.commit()

    await asyncio.gather(run(), run())
    async with pool() as session:
        n_act = (await session.execute(select(func.count()).select_from(Activity))).scalar_one()
        n_cand = (
            await session.execute(select(func.count()).select_from(SourceCandidate))
        ).scalar_one()
    assert n_act == 1  # one resolved activity, never a duplicate from a lost race
    assert n_cand == 1  # one candidate row on the natural key (idempotent, ING-UPS-R1)


# ------------------------------------------------------ PERF-R1: batched round-trips


async def test_bulk_lap_array_is_one_round_trip_not_per_row_loop(
    pool: async_sessionmaker[AsyncSession],
) -> None:
    """A 20-lap array lands in ONE multi-row VALUES upsert, never a per-row loop (PERF-R1).

    Per-row insert loops for bulk data are prohibited. The lap array — the bulk per-activity
    payload — is batched into a SINGLE round-trip through the seam; a loop would issue one
    INSERT per lap. Every canonical INSERT also carries ON CONFLICT, proving the writes are
    atomic insert-or-update (the seam), not a check-then-write select-then-add (UPS-R2).
    """
    athlete_id, descriptor = await _seed(pool)
    laps = [{"lap_index": i, "duration_s": 60, "avg_power_w": 200.0} for i in range(20)]
    cand = _ride("ride-laps")
    cand.payload["laps"] = laps

    inserts: list[str] = []

    def _count(_conn: object, _cur: object, statement: str, *_a: object) -> None:
        if statement.lstrip().upper().startswith("INSERT INTO"):
            inserts.append(statement)

    async with pool() as session:
        bind = session.get_bind()  # the proxied sync Engine the async session runs over
        event.listen(bind, "before_cursor_execute", _count)
        try:
            await IngestService(session, batch_size=500).ingest(athlete_id, descriptor, [cand])
            await session.commit()
        finally:
            event.remove(bind, "before_cursor_execute", _count)

    # 20 laps land in ONE multi-row VALUES INSERT — never 20 separate lap INSERTs (PERF-R1).
    lap_inserts = [s for s in inserts if "activity_lap" in s]
    assert len(lap_inserts) == 1
    # Every canonical-table write is an atomic upsert (ON CONFLICT), not a plain INSERT that
    # would imply a preceding existence SELECT (check-then-write); UPS-R2 forbids the latter.
    canonical = [s for s in inserts if any(t in s for t in ("activity", "source_candidate"))]
    assert canonical and all("ON CONFLICT" in s for s in canonical)


# --------------------------------------------------- ING-UPS-R3: no whole-run rollback


async def test_one_bad_record_does_not_roll_back_the_run(
    pool: async_sessionmaker[AsyncSession],
) -> None:
    """A single malformed candidate is isolated; the good records still land (ING-UPS-R3).

    The prohibited behaviour is whole-run rollback on one record failure. With per-record
    SAVEPOINT isolation, the bad row rolls back only itself and the surrounding good rows
    are persisted.
    """
    athlete_id, descriptor = await _seed(pool)
    good_a = _ride("ride-good-a")
    good_b = _ride("ride-good-b")
    # Distinct sessions: shift B well beyond the ±2h identity window so it is its OWN
    # activity (otherwise the windowed matcher would resolve both to one activity).
    good_b.payload["start_time"] = _START + _dt.timedelta(hours=6)
    # A malformed candidate: a daily_wellness with NO local_date -> KeyError mid-persist.
    bad = GboCandidate(
        gbo_type="daily_wellness",
        source_descriptor_id="placeholder",
        source_native_id="bad-wellness",
        content_hash=content_hash(b"bad"),
        payload={"resting_hr_bpm": 50},  # missing the required local_date
        trust_tier=Fidelity.SUMMARY_ONLY,
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )
    async with pool() as session:
        result = await IngestService(session).ingest(
            athlete_id, descriptor, [good_a, bad, good_b]
        )
        await session.commit()
    assert result.candidates_failed == 1  # the bad row was isolated, not fatal
    async with pool() as session:
        n_act = (await session.execute(select(func.count()).select_from(Activity))).scalar_one()
        # The malformed wellness candidate was rejected BEFORE the bulk insert: it never
        # leaves an orphan candidate row (only the two good activity candidates persist).
        n_cand = (
            await session.execute(select(func.count()).select_from(SourceCandidate))
        ).scalar_one()
    assert n_act == 2  # both good activities survived the bad row (no whole-run rollback)
    assert n_cand == 2  # only the two good candidates landed; the bad one left no row


async def test_candidate_batch_is_one_multi_row_round_trip_not_per_row(
    pool: async_sessionmaker[AsyncSession],
) -> None:
    """A batch of N candidates lands in ONE multi-row VALUES upsert, not N inserts (ING-UPS-R1).

    ING-UPS-R1 requires candidate writes be bulk/batched — one round-trip per batch, never
    a per-candidate INSERT loop (the prior implementation issued N single-row upserts). A
    batch of three distinct candidates must compile to exactly ONE ``source_candidate``
    INSERT, and it must be an atomic ``ON CONFLICT`` upsert (UPS-R2), not a plain INSERT.
    """
    athlete_id, descriptor = await _seed(pool)
    # Three distinct activities, each its OWN window so resolution never collapses them.
    cands = [
        _ride(f"ride-{i}", start=_START + _dt.timedelta(hours=6 * i)) for i in range(3)
    ]

    inserts: list[str] = []

    def _count(_conn: object, _cur: object, statement: str, *_a: object) -> None:
        if statement.lstrip().upper().startswith("INSERT INTO") and "source_candidate" in statement:
            inserts.append(statement)

    async with pool() as session:
        bind = session.get_bind()
        event.listen(bind, "before_cursor_execute", _count)
        try:
            await IngestService(session, batch_size=500).ingest(athlete_id, descriptor, cands)
        finally:
            event.remove(bind, "before_cursor_execute", _count)

    # All three candidates land in ONE multi-row VALUES INSERT — never three separate
    # single-row candidate INSERTs (a per-row loop). The batch is one round-trip (PERF-R1).
    assert len(inserts) == 1
    assert "ON CONFLICT" in inserts[0]  # atomic insert-or-update, not check-then-write
    async with pool() as session:
        n_cand = (
            await session.execute(select(func.count()).select_from(SourceCandidate))
        ).scalar_one()
    assert n_cand == 3  # all three distinct candidates persisted in the one round-trip


# ----------------------------------------- ING-UPS-R3: DURABLE batch granularity (ACC-4)


async def test_committed_batch_survives_a_later_batch_failure(
    pool: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earlier batch stays durable in a SEPARATE session when a later batch fails (ING-UPS-R3).

    This is the durable-batch-granularity guarantee, stronger than within-transaction
    SAVEPOINT isolation: with ``batch_size=1`` each candidate is its own batch and is
    committed before the next begins. The SECOND batch's candidate write is made to raise
    (a failure OUTSIDE the per-record savepoint, so it aborts the run) — yet the FIRST
    batch's activity must already be **committed** and therefore visible from a brand-new
    session on a fresh connection (a flushed-but-uncommitted row would be invisible there).
    SAVEPOINTs alone would be discarded on the outer rollback, failing this assertion — so
    it would catch a regression to non-durable batch semantics.
    """
    athlete_id, descriptor = await _seed(pool)
    first = _ride("ride-first")
    second = _ride("ride-second", start=_START + _dt.timedelta(hours=6))

    # Fail the SECOND batch's candidate write — a non-savepointed error that aborts the run.
    real_bulk = ingest_steps_mod.persist_candidates_bulk
    calls = {"n": 0}

    async def _bulk_fail_second(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected later-batch failure")
        return await real_bulk(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ingest_steps_mod, "persist_candidates_bulk", _bulk_fail_second)

    with pytest.raises(RuntimeError, match="injected later-batch failure"):
        async with pool() as session:
            # batch_size=1 => [first] is batch 1 (committed), [second] is batch 2 (fails).
            await IngestService(session, batch_size=1).ingest(
                athlete_id, descriptor, [first, second]
            )

    # A BRAND-NEW session on a fresh pooled connection: it can only see COMMITTED rows.
    async with pool() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
        n_cand = (
            await session.execute(select(func.count()).select_from(SourceCandidate))
        ).scalar_one()
    # The first batch is durably committed; the failed second batch left nothing.
    assert len(acts) == 1
    assert n_cand == 1
