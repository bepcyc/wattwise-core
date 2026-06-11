"""Five-phase discover pipeline + fail-closed sync gaps (ADP-R3..R7, ING-R3, ING-OBS-R1/R2).

Exercises :class:`SyncOrchestrator` against an in-process :class:`DiscoverFetch` fake
(no network, ADP-R17): cursor-paginated discovery (ADP-R7), engine-side watermark
honoring (ADP-R6), per-ref fetch isolation with token-precise transient gaps that
SELF-HEAL when the record later lands (ING-GAP-R4/R5), a mid-pagination break
recorded as a ``discovery_incomplete`` gap while page-1 records still commit
(ADP-R7/ING-UPS-R3), the typed ``ensure_authorized`` reauth path (ADP-R4/AUT-R4),
the ADP-R3 declared-GBO-type REFUSAL (terminal ``schema_mismatch`` gap, nothing
written — the fail-open silent drop is dead), the ING-R3 persisted typed gap on a
degraded legacy fetch, and the ING-OBS-R1/R2 per-run trace metrics.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tests.integration._fake_capability import fake_capability
from tests.integration._session_provider import FactorySessionProvider
from tests.integration.test_sync import FailingAdapter, _cred_store, _seed_connection
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import (
    AuthArchetype,
    ConnectionStatus,
    Fidelity,
    GapReason,
    GapState,
    GboType,
    SourceKind,
)
from wattwise_core.ingestion.base import AuthError, FetchContext, SourceDescriptorRef
from wattwise_core.ingestion.capability import (
    AuthContext,
    CapabilityDescriptor,
    DiscoveryPage,
    DiscoveryRef,
)
from wattwise_core.ingestion.registry import registry_from_adapters
from wattwise_core.ingestion.sync import SyncOrchestrator, SyncOutcome
from wattwise_core.observability import metrics as _metrics
from wattwise_core.persistence.models import Activity, Base, Connection
from wattwise_core.persistence.models.source import IngestionGap, IngestionWatermark
from wattwise_core.storage import content_hash

UTC = _dt.UTC
_FIXED_NOW = _dt.datetime(2026, 6, 5, 9, 0, tzinfo=UTC)
_RIDE_DAYS = (
    _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
    _dt.datetime(2026, 6, 2, 8, 0, tzinfo=UTC),
)


class _Ride:
    """A trivial source-shaped object the fake's map consumes."""

    def __init__(self, native_id: str, start: _dt.datetime) -> None:
        self.native_id = native_id
        self.start = start


class PagedAdapter:
    """A five-phase fake: ensure_authorized + cursor-paged discover + per-ref fetch.

    ``page_size`` forces multi-page discovery (ADP-R7); ``fail_fetch`` makes single
    refs un-fetchable (token-gap isolation); ``fail_page`` breaks pagination at that
    page index (partial discovery); ``auth_fail`` raises the typed AuthError
    (ADP-R4). The fake does NOT filter by ``since_watermark`` so the tests prove the
    ENGINE-side ADP-R6 skip. ``wrong_type`` makes map emit an undeclared GBO type
    (the ADP-R3 refusal probe).
    """

    source_key: ClassVar[str] = "paged_api"
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.API_KEY
    kind: ClassVar[SourceKind] = SourceKind.OAUTH_API
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"
    capability: ClassVar[CapabilityDescriptor] = fake_capability(
        "paged_api", gbo_types=frozenset({GboType.ACTIVITY})
    )

    def __init__(
        self,
        rides: list[tuple[str, _dt.datetime]] | None = None,
        *,
        page_size: int = 1,
        fail_fetch: frozenset[str] = frozenset(),
        fail_page: int | None = None,
        auth_fail: bool = False,
        wrong_type: bool = False,
    ) -> None:
        default_rides = [("r-1", _RIDE_DAYS[0]), ("r-2", _RIDE_DAYS[1])]
        self.rides = rides if rides is not None else default_rides
        self.page_size = page_size
        self.fail_fetch = fail_fetch
        self.fail_page = fail_page
        self.auth_fail = auth_fail
        self.wrong_type = wrong_type
        self.fetched: list[str] = []
        self.discover_pages = 0

    async def ensure_authorized(
        self, *, api_key: str | None, athlete_native_id: str | None
    ) -> AuthContext:
        if self.auth_fail:
            raise AuthError(detail="key revoked upstream")
        return AuthContext(athlete_native_id=athlete_native_id or "self", api_key=api_key)

    async def discover(
        self,
        ctx: AuthContext,
        window: Any,
        *,
        cursor: str | None = None,
        since_watermark: _dt.datetime | None = None,
    ) -> DiscoveryPage:
        offset = int(cursor or 0)
        page_index = offset // self.page_size
        if self.fail_page is not None and page_index == self.fail_page:
            raise RuntimeError("listing endpoint exploded mid-pagination")
        self.discover_pages += 1
        refs = tuple(
            DiscoveryRef(source_native_id=nid, gbo_type=GboType.ACTIVITY, last_modified=start)
            for nid, start in self.rides[offset : offset + self.page_size]
        )
        end = offset + self.page_size
        return DiscoveryPage(refs=refs, next_cursor=str(end) if end < len(self.rides) else None)

    async def fetch_ref(self, ctx: AuthContext, ref: DiscoveryRef) -> Any:
        if ref.source_native_id in self.fail_fetch:
            raise RuntimeError("record endpoint exploded")
        self.fetched.append(ref.source_native_id)
        start = dict(self.rides)[ref.source_native_id]
        return _Ride(ref.source_native_id, start)

    def map(
        self, asbo: Any, source_descriptor: SourceDescriptorRef, fetch_context: FetchContext
    ) -> list[GboCandidate]:
        if not isinstance(asbo, _Ride):
            return []
        payload: dict[str, Any] = {
            "start_time": asbo.start,
            "sport": "cycling",
            "elapsed_time_s": 3600,
            "avg_power_w": 250.0,
        }
        return [
            GboCandidate(
                gbo_type="daily_wellness" if self.wrong_type else "activity",
                source_descriptor_id=source_descriptor.source_descriptor_id,
                source_native_id=asbo.native_id,
                content_hash=content_hash(asbo.native_id.encode()),
                payload=payload if not self.wrong_type else {"local_date": "2026-06-01"},
                observed_at=asbo.start,
                fetched_at=fetch_context.fetched_at,
                trust_tier=Fidelity.PLATFORM_COMPUTED,
                connection_id=fetch_context.connection_id,
                adapter_version=self.adapter_version,
                mapping_version=self.mapping_version,
            )
        ]


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


def _orch(session_factory: Any, adapter: Any, store: Any) -> SyncOrchestrator:
    return SyncOrchestrator(
        FactorySessionProvider(session_factory),
        registry=registry_from_adapters([adapter]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )


async def _gaps(session_factory: Any) -> list[IngestionGap]:
    async with session_factory() as session:
        return list((await session.execute(select(IngestionGap))).scalars().all())


@pytest.mark.integration
async def test_discover_pipeline_paginates_and_lands(session_factory: Any) -> None:
    """Cursor-paged discovery fetches every ref and lands them canonically (ADP-R5/R7)."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="paged_api", credential_ref=ref
    )
    adapter = PagedAdapter(page_size=1)  # 2 rides -> 2 pages, cursor surfaced between
    before = _metrics.get_registry().counter_value(
        _metrics.INGEST_SOURCE_RUNS, labels={"source_key": "paged_api", "outcome": "ok"}
    )

    run = await _orch(session_factory, adapter, store).run(athlete_id, source="paged_api")

    assert run.results[0].outcome is SyncOutcome.OK
    assert adapter.discover_pages == 2  # pagination actually happened (ADP-R7)
    assert sorted(adapter.fetched) == ["r-1", "r-2"]
    async with session_factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
        marks = (await session.execute(select(IngestionWatermark))).scalars().all()
    assert len(acts) == 2
    # SYN-R2/R3: the activity watermark advanced to the newest ingested instant.
    assert any(m.high_water_at is not None for m in marks)
    # ING-OBS-R1/R2: the per-run trace recorded the run outcome on the metrics surface.
    after = _metrics.get_registry().counter_value(
        _metrics.INGEST_SOURCE_RUNS, labels={"source_key": "paged_api", "outcome": "ok"}
    )
    assert after == before + 1


@pytest.mark.integration
async def test_second_run_skips_watermark_current_refs(session_factory: Any) -> None:
    """The engine skips refs already current per the watermark — no re-fetch (ADP-R6).

    The fake does NOT filter by ``since_watermark``, so zero ``fetch_ref`` calls on
    the second run prove the ENGINE-side skip, not adapter politeness.
    """
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="paged_api", credential_ref=ref
    )
    adapter = PagedAdapter(page_size=2)
    orch = _orch(session_factory, adapter, store)
    await orch.run(athlete_id, source="paged_api")
    adapter.fetched.clear()

    run2 = await orch.run(athlete_id, source="paged_api")

    assert run2.results[0].outcome is SyncOutcome.OK
    assert adapter.fetched == []  # every ref was already current per the watermark
    async with session_factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
    assert len(acts) == 2  # idempotent: nothing duplicated (ING-R6)


@pytest.mark.integration
async def test_failed_ref_opens_token_gap_then_self_heals(session_factory: Any) -> None:
    """A per-ref fetch failure gap-marks exactly that record; a later success closes it.

    ING-GAP-R5 (range-precise: only the failed token) + ING-GAP-R4 (transient gaps
    self-heal with a recorded closure time when the record finally lands).
    """
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="paged_api", credential_ref=ref
    )
    adapter = PagedAdapter(page_size=2, fail_fetch=frozenset({"r-2"}))
    orch = _orch(session_factory, adapter, store)

    run = await orch.run(athlete_id, source="paged_api")

    assert run.results[0].outcome is SyncOutcome.DEGRADED  # partial: r-1 landed, r-2 gap
    async with session_factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
    assert len(acts) == 1  # the good record committed (ING-UPS-R3)
    gaps = await _gaps(session_factory)
    token_gaps = [g for g in gaps if g.reason is GapReason.FETCH_FAILED]
    assert len(token_gaps) == 1
    gap = token_gaps[0]
    assert gap.transient is True
    assert gap.range_start_token == "r-2" and gap.range_end_token == "r-2"
    assert gap.state is GapState.OPEN

    # The source recovers: the next run fetches r-2 and the token gap self-heals.
    healed = PagedAdapter(page_size=2)
    run2 = await _orch(session_factory, healed, store).run(athlete_id, source="paged_api")
    assert run2.results[0].outcome is SyncOutcome.OK
    gaps_after = await _gaps(session_factory)
    healed_gap = next(g for g in gaps_after if g.reason is GapReason.FETCH_FAILED)
    assert healed_gap.state is GapState.CLOSED
    assert healed_gap.closed_at is not None  # closure time recorded (ING-GAP-R4)


@pytest.mark.integration
async def test_partial_discovery_opens_incomplete_gap_keeps_page_one(session_factory: Any) -> None:
    """Page N+1 failing yields a ``discovery_incomplete`` gap; page N still lands (ADP-R7)."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="paged_api", credential_ref=ref
    )
    adapter = PagedAdapter(page_size=1, fail_page=1)  # page 0 ok, page 1 explodes

    run = await _orch(session_factory, adapter, store).run(athlete_id, source="paged_api")

    assert run.results[0].outcome is SyncOutcome.DEGRADED
    async with session_factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
    assert len(acts) == 1  # page-1 record still committed (ING-UPS-R3)
    gaps = await _gaps(session_factory)
    incomplete = [g for g in gaps if g.reason is GapReason.DISCOVERY_INCOMPLETE]
    assert len(incomplete) == 1
    assert incomplete[0].transient is True
    assert incomplete[0].range_end_at is not None  # bounded to the window end


@pytest.mark.integration
async def test_ensure_authorized_failure_flips_reauth(session_factory: Any) -> None:
    """A typed AuthError from ensure_authorized stops the source via reauth (ADP-R4/AUT-R4)."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="paged_api", credential_ref=ref
    )
    adapter = PagedAdapter(auth_fail=True)

    run = await _orch(session_factory, adapter, store).run(athlete_id, source="paged_api")

    assert run.results[0].outcome is SyncOutcome.REAUTH_REQUIRED
    async with session_factory() as session:
        conn = (await session.execute(select(Connection))).scalars().one()
    assert conn.status is ConnectionStatus.REAUTH_REQUIRED
    gaps = await _gaps(session_factory)
    assert any(g.reason is GapReason.NEEDS_REAUTH and g.transient is False for g in gaps)


@pytest.mark.integration
async def test_undeclared_gbo_type_is_refused_never_dropped(session_factory: Any) -> None:
    """ADP-R3 fail-closed: an undeclared GBO type REFUSES the upsert with a terminal gap.

    The adapter declares only ``activity`` but maps a ``daily_wellness`` candidate.
    Before the fix this was silently dropped from the canonical store (fail-open data
    loss); now NOTHING lands, the result degrades, and a TERMINAL ``schema_mismatch``
    gap records the refusal — mutation-proof against reverting the declared-type guard.
    """
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="paged_api", credential_ref=ref
    )
    adapter = PagedAdapter(page_size=2, wrong_type=True)

    run = await _orch(session_factory, adapter, store).run(athlete_id, source="paged_api")

    assert run.results[0].outcome is SyncOutcome.DEGRADED
    assert run.results[0].detail == "adapter emitted an undeclared record type"
    async with session_factory() as session:
        acts = (await session.execute(select(Activity))).scalars().all()
    assert acts == []  # refusal, not partial landing
    gaps = await _gaps(session_factory)
    refusals = [g for g in gaps if g.reason is GapReason.SCHEMA_MISMATCH]
    assert len(refusals) == 1
    assert refusals[0].transient is False  # terminal: needs an operator/code fix


@pytest.mark.integration
async def test_degraded_fetch_persists_typed_gap(session_factory: Any) -> None:
    """ING-R3: un-obtainable data records a PERSISTED typed gap, never a string-only degrade.

    The legacy window-fetch adapter raising used to produce only a DEGRADED summary
    (the audited ING-R3 deviation); now a transient ``fetch_failed`` gap covering the
    attempted window is queryable by downstream consumers (ING-GAP-R1).
    """
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="broken_api", credential_ref=ref
    )

    run = await _orch(session_factory, FailingAdapter(), store).run(athlete_id, source="broken_api")

    assert run.results[0].outcome is SyncOutcome.DEGRADED
    gaps = await _gaps(session_factory)
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.reason is GapReason.FETCH_FAILED
    assert gap.transient is True
    assert gap.range_start_at is not None and gap.range_end_at is not None
    assert gap.state is GapState.OPEN
