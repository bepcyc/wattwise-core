"""On-demand sync orchestrator integration journey (SYN-R*, CON-R3, ARCH-R9, UPS-R6).

Exercises :class:`SyncOrchestrator` end to end on the portable substrate with a fake,
in-process adapter (no network, ADP-R17): a connection is resolved to its adapter via
the registry (by ``source_key``, never by importing a named adapter — ARCH-R2), the
adapter's impure ``fetch`` yields source-shaped objects, its pure ``map`` turns them
into canonical candidates, and :class:`IngestService` lands them into the canonical
store in ONE transaction (UPS-R6). A second adapter that raises in ``fetch`` proves a
source error DEGRADES gracefully (CON-R3) and never crashes the run nor the other
sources (ARCH-R9). The api-key secret is resolved from an opaque ``credential_ref``
via the credential store (SEC-R7), never passed in plaintext.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import AuthArchetype, ConnectionStatus, Fidelity, SourceKind
from wattwise_core.ingestion.base import FetchContext, SourceDescriptorRef
from wattwise_core.ingestion.registry import registry_from_adapters
from wattwise_core.ingestion.sync import (
    SyncOrchestrator,
    SyncOutcome,
    SyncWindow,
)
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    Base,
    Connection,
    SourceDescriptor,
    Sport,
)
from wattwise_core.security.credentials import InMemoryCredentialStore
from wattwise_core.security.crypto import EnvelopeCipher
from wattwise_core.storage import content_hash

UTC = _dt.UTC
_FIXED_NOW = _dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- fakes


class _FakeRideAsbo:
    """A trivial source-shaped object the fake adapter's ``map`` consumes."""

    def __init__(self, native_id: str, watts: float, seconds: int) -> None:
        self.native_id = native_id
        self.watts = watts
        self.seconds = seconds


class FakeApiAdapter:
    """An in-process api-key adapter: impure ``fetch`` + pure ``map`` (ADP-R*, ADP-R17).

    Records the api_key it was handed so the test can assert the credential was
    resolved from the opaque ref (SEC-R7) and passed to the fetch side only.
    """

    source_key: ClassVar[str] = "fake_api"
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.API_KEY
    kind: ClassVar[SourceKind] = SourceKind.OAUTH_API
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"

    def __init__(self) -> None:
        self.seen_api_key: str | None = None
        self.seen_window: SyncWindow | None = None

    async def fetch(
        self, *, api_key: str | None, athlete_native_id: str | None, window: SyncWindow
    ) -> Iterable[Any]:
        self.seen_api_key = api_key
        self.seen_window = window
        return [_FakeRideAsbo("ride-1", watts=250.0, seconds=3600)]

    def map(
        self, asbo: Any, source_descriptor: SourceDescriptorRef, fetch_context: FetchContext
    ) -> list[GboCandidate]:
        if not isinstance(asbo, _FakeRideAsbo):
            return []
        start = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
        payload: dict[str, Any] = {
            "start_time": start,
            "sport": "cycling",
            "elapsed_time_s": asbo.seconds,
            "moving_time_s": asbo.seconds,
            "avg_power_w": asbo.watts,
            "streams": {
                "power_w": {
                    "values": [asbo.watts] * asbo.seconds,
                    "sample_basis": "time",
                    "sample_rate_hz": 1.0,
                }
            },
        }
        return [
            GboCandidate(
                gbo_type="activity",
                source_descriptor_id=source_descriptor.source_descriptor_id,
                source_native_id=asbo.native_id,
                content_hash=content_hash(f"{asbo.native_id}:{asbo.watts}".encode()),
                payload=payload,
                observed_at=start,
                fetched_at=fetch_context.fetched_at,
                trust_tier=Fidelity.RAW_STREAM,
                connection_id=fetch_context.connection_id,
                adapter_version=self.adapter_version,
                mapping_version=self.mapping_version,
            )
        ]


class FailingAdapter:
    """An api-key adapter whose impure ``fetch`` always raises (CON-R3 / ARCH-R9)."""

    source_key: ClassVar[str] = "broken_api"
    auth_archetype: ClassVar[AuthArchetype] = AuthArchetype.API_KEY
    kind: ClassVar[SourceKind] = SourceKind.OAUTH_API
    adapter_version: ClassVar[str] = "1"
    mapping_version: ClassVar[str] = "1"

    async def fetch(
        self, *, api_key: str | None, athlete_native_id: str | None, window: SyncWindow
    ) -> Iterable[Any]:
        raise RuntimeError("source upstream is down")

    def map(
        self, asbo: Any, source_descriptor: SourceDescriptorRef, fetch_context: FetchContext
    ) -> list[GboCandidate]:
        return []


# --------------------------------------------------------------------------- harness


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[Any]:
    """A transactional session-factory over a fresh in-memory canonical schema."""
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


async def _seed_connection(
    factory: Any, *, source_key: str, credential_ref: str | None
) -> tuple[str, str]:
    """Seed the athlete, the cycling sport, a source descriptor, and an api-key connection."""
    async with factory() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        athlete = Athlete(sex="male", reference_timezone="UTC")
        session.add(athlete)
        descriptor = SourceDescriptor(
            source_key=source_key, display_name=source_key, kind="oauth_api"
        )
        session.add(descriptor)
        await session.flush()
        session.add(
            Connection(
                athlete_id=athlete.athlete_id,
                source_descriptor_id=descriptor.source_descriptor_id,
                status=ConnectionStatus.CONNECTED,
                credential_ref=credential_ref,
                auth_archetype=AuthArchetype.API_KEY,
            )
        )
        await session.flush()
        return str(athlete.athlete_id), str(descriptor.source_descriptor_id)


def _cred_store() -> tuple[InMemoryCredentialStore, str]:
    """A credential store holding one api key, returning the store + its opaque ref."""
    cipher = EnvelopeCipher(EnvelopeCipher.generate_root_key())
    store = InMemoryCredentialStore(cipher)
    ref = store.store("secret-api-key-123")
    return store, ref


# --------------------------------------------------------------------------- tests


@pytest.mark.integration
async def test_fake_adapter_sync_writes_canonical(session_factory: Any) -> None:
    """A fake adapter's fetched ride is mapped and landed as ONE canonical activity."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="fake_api", credential_ref=ref
    )
    adapter = FakeApiAdapter()
    orch = SyncOrchestrator(
        session_factory,
        registry=registry_from_adapters([adapter]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    run = await orch.run(
        athlete_id, source="fake_api", window=SyncWindow("2026-05-01", "2026-06-01")
    )

    assert not run.degraded
    assert run.activities_written == 1
    assert len(run.results) == 1
    assert run.results[0].outcome is SyncOutcome.OK
    assert run.results[0].candidates_mapped == 1
    # The opaque credential_ref was resolved to the live secret at the point of use.
    assert adapter.seen_api_key == "secret-api-key-123"
    assert adapter.seen_window == SyncWindow("2026-05-01", "2026-06-01")
    # Exactly one canonical activity landed (single-count DEDUP-R1).
    async with session_factory() as session:
        activities = (await session.execute(select(Activity))).scalars().all()
    assert len(activities) == 1
    assert float(activities[0].avg_power_w) == pytest.approx(250.0)


@pytest.mark.integration
async def test_resync_is_idempotent(session_factory: Any) -> None:
    """Re-running sync on unchanged content writes no second activity (UPS-R3)."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="fake_api", credential_ref=ref
    )
    orch = SyncOrchestrator(
        session_factory,
        registry=registry_from_adapters([FakeApiAdapter()]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )
    await orch.run(athlete_id, source="fake_api")
    await orch.run(athlete_id, source="fake_api")
    async with session_factory() as session:
        count = len((await session.execute(select(Activity))).scalars().all())
    assert count == 1


@pytest.mark.integration
async def test_failing_adapter_degrades_not_crashes(session_factory: Any) -> None:
    """A source that raises in fetch degrades gracefully (CON-R3) and never crashes."""
    store, ref = _cred_store()
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="broken_api", credential_ref=ref
    )
    orch = SyncOrchestrator(
        session_factory,
        registry=registry_from_adapters([FailingAdapter()]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    # The run completes — the exception does NOT propagate past the source (ARCH-R9).
    run = await orch.run(athlete_id, source="broken_api")

    assert run.degraded
    assert run.activities_written == 0
    assert len(run.results) == 1
    assert run.results[0].outcome is SyncOutcome.DEGRADED
    assert run.results[0].detail is not None
    # Nothing was written for the failed source.
    async with session_factory() as session:
        activities = (await session.execute(select(Activity))).scalars().all()
    assert activities == []


@pytest.mark.integration
async def test_one_failing_source_does_not_crash_the_other(session_factory: Any) -> None:
    """With two connections, a failing source degrades while the healthy one still lands."""
    store, ref = _cred_store()
    # Seed a healthy fake_api connection (its athlete), then add a broken_api connection
    # for the SAME athlete so one run covers both sources.
    athlete_id, _ = await _seed_connection(
        session_factory, source_key="fake_api", credential_ref=ref
    )
    async with session_factory() as session:
        broken = SourceDescriptor(
            source_key="broken_api", display_name="broken_api", kind="oauth_api"
        )
        session.add(broken)
        await session.flush()
        session.add(
            Connection(
                athlete_id=uuid.UUID(athlete_id),
                source_descriptor_id=broken.source_descriptor_id,
                status=ConnectionStatus.CONNECTED,
                credential_ref=ref,
                auth_archetype=AuthArchetype.API_KEY,
            )
        )
        await session.flush()

    orch = SyncOrchestrator(
        session_factory,
        registry=registry_from_adapters([FakeApiAdapter(), FailingAdapter()]),
        credential_store=store,
        now=lambda: _FIXED_NOW,
    )

    run = await orch.run(athlete_id)  # no source filter -> both connections

    assert run.degraded  # the broken source degraded
    assert run.activities_written == 1  # the healthy source still landed its ride
    outcomes = {r.source_key: r.outcome for r in run.results}
    assert outcomes["fake_api"] is SyncOutcome.OK
    assert outcomes["broken_api"] is SyncOutcome.DEGRADED
