"""Integration tests for the data-quality / coverage DIAGNOSIS deliverable (API-R15).

These exercise the DETERMINISTIC diagnosis the deployable
:class:`~wattwise_core.agent.engine.GraphAgentEngine` exposes (``engine.diagnose``) plus the
underlying :func:`~wattwise_core.agent.diagnose_deliverable.diagnose_coverage` projection against a
REAL canonical store. They pin the three guarantees that make the diagnosis trustworthy:

* a SEEDED athlete (rides + an FTP signature for their CURRENT sport) reports the canonical inputs
  the analytics service actually computes as ``present`` and the run is ``completed`` (no model
  call, no fabrication); the signature probe is SPORT-KEYED off the athlete's current sport (H3 /
  CFG-R1a) — a runner's running signature reads present, a stale cycling one does not;
* an EMPTY athlete (no rides, no signature) reports every input ``missing`` and the run
  ``degrades`` with a typed ``no_canonical_coverage`` caveat — it never invents a present input or a
  number (fail-closed, GROUND-R7 / OUTCOME-R3);
* the diagnosis surfaces NO athlete-facing numeric value on any coverage line (a diagnosis reports
  coverage, never a canonical metric, VOICE-R7).

The diagnosis is model-free, so a bare :class:`FakeModel` (never invoked) backs the engine — the
test proves the coverage narration comes only from the canonical ``Computed``/``Unavailable``
envelope.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.diagnose_deliverable import (
    InputCoverage,
    InputStatus,
    diagnose_coverage,
)
from wattwise_core.agent.engine import GraphAgentEngine
from wattwise_core.agent.model import FakeModel
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, SignatureOrigin
from wattwise_core.identity import OWNER_ATHLETE_ID, OWNER_SUBJECT
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.persistence.models import (
    Athlete,
    Base,
    FitnessSignature,
    SourceDescriptor,
    Sport,
)
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.integration

UTC = _dt.UTC


class _DatabaseStub:
    """A minimal canonical ``Database`` substitute over one engine (the engine reads only)."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    def session(self) -> _SessionCtx:
        return _SessionCtx(self._factory)


class _SessionCtx:
    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        self._session = self._factory()
        return self._session

    async def __aexit__(self, *exc: object) -> None:
        assert self._session is not None
        await self._session.close()


def _ride(native_id: str, day: _dt.date, *, watts: float = 250.0) -> GboCandidate:
    """A constant-``watts``, 1 h cycling ride so the canonical analytics have load to compute."""
    seconds = 3600
    payload = {
        "start_time": _dt.datetime(day.year, day.month, day.day, 8, 0, tzinfo=UTC),
        "sport": "cycling",
        "elapsed_time_s": seconds,
        "moving_time_s": seconds,
        "avg_power_w": watts,
        "streams": {
            "power_w": {"values": [watts] * seconds, "sample_basis": "time", "sample_rate_hz": 1.0}
        },
        "laps": [
            {"lap_index": 0, "start_offset_s": 0, "duration_s": seconds, "avg_power_w": watts}
        ],
    }
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(native_id.encode()),
        payload=payload,
        trust_tier=Fidelity.RAW_STREAM,
        fetched_at=_dt.datetime(day.year, day.month, day.day, 9, 0, tzinfo=UTC),
    )


async def _new_canonical(
    *,
    seed_signature: bool,
    rides: int,
    current_sport: str | None = "cycling",
    signature_sport: str = "cycling",
) -> _DatabaseStub:
    """A fresh in-memory canonical store; optionally an FTP signature + recent rides.

    Rides are seeded relative to ``utcnow()`` so they fall inside the diagnosis recent window
    regardless of the wall clock. ``seed_signature=False`` / ``rides=0`` produces the empty athlete
    whose every canonical input fails closed. ``current_sport`` is the athlete's profile sport the
    signature probe keys on (H3 / CFG-R1a — resolved from data, never hardcoded);
    ``signature_sport`` is the sport the seeded signature is FOR, so a mismatch (a runner whose only
    signature is cycling) models a stale cross-sport signature that must read MISSING.
    ``current_sport=None`` models an athlete with no current sport (the signature reads MISSING).
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        session.add(Sport(sport_code="running", display_name="Running", has_mechanical_power=False))
        session.add(
            Athlete(
                athlete_id=OWNER_ATHLETE_ID,
                sex="male",
                reference_timezone="UTC",
                current_sport=current_sport,
            )
        )
        descriptor = SourceDescriptor(
            source_key="file_import", display_name="Activity files", kind="file_upload"
        )
        session.add(descriptor)
        if seed_signature:
            session.add(
                FitnessSignature(
                    athlete_id=OWNER_ATHLETE_ID,
                    signature_type=signature_sport,
                    effective_date=_dt.date(2024, 1, 1),
                    ftp_w=250.0,
                    origin=SignatureOrigin.MEASURED,
                )
            )
        await session.flush()
        if rides:
            ingest = IngestService(session)
            today = _dt.datetime.now(UTC).date()
            for i in range(rides):
                day = today - _dt.timedelta(days=i)
                await ingest.ingest(
                    str(OWNER_ATHLETE_ID),
                    str(descriptor.source_descriptor_id),
                    [_ride(f"d{i}", day)],
                )
        await session.commit()
    return _DatabaseStub(factory)


@pytest_asyncio.fixture
async def seeded_db() -> AsyncIterator[_DatabaseStub]:
    """Canonical store with an FTP signature + three recent rides (present coverage)."""
    yield await _new_canonical(seed_signature=True, rides=3)


@pytest_asyncio.fixture
async def empty_db() -> AsyncIterator[_DatabaseStub]:
    """Canonical store with the athlete only — no signature, no rides (no coverage)."""
    yield await _new_canonical(seed_signature=False, rides=0)


def _by_key(inputs: tuple[InputCoverage, ...]) -> dict[str, InputCoverage]:
    """Index the per-input coverage lines by their stable machine key."""
    return {i.key: i for i in inputs}


async def test_diagnose_reports_present_inputs_for_seeded_athlete(
    seeded_db: _DatabaseStub,
) -> None:
    """A seeded athlete -> training-load + signature present, run completed (API-R15)."""
    engine = GraphAgentEngine(seeded_db, FakeModel())  # type: ignore[arg-type]
    diagnosis = await engine.diagnose(athlete_id=OWNER_SUBJECT)
    assert diagnosis.status is RunStatus.COMPLETED
    inputs = _by_key(diagnosis.inputs)
    # The canonical PMC computes a load series from the seeded rides; the FTP signature resolves.
    assert inputs["training_load"].status is InputStatus.PRESENT
    assert inputs["fitness_signature"].status is InputStatus.PRESENT
    assert diagnosis.coverage_caveat is None  # at least one input present -> no degrade caveat


async def test_diagnose_signature_present_for_runner_with_running_signature() -> None:
    """H3: a runner with a RUNNING signature reports the signature PRESENT (CFG-R1a, sport-keyed).

    The signature probe must key on the athlete's CURRENT sport resolved from canonical data — not a
    hardcoded ``"cycling"``. A runner (``current_sport="running"``) whose seeded signature is a
    running signature must read PRESENT; a hardcoded-cycling probe would falsely report it MISSING.
    """
    db = await _new_canonical(
        seed_signature=True, rides=0, current_sport="running", signature_sport="running"
    )
    engine = GraphAgentEngine(db, FakeModel())  # type: ignore[arg-type]
    diagnosis = await engine.diagnose(athlete_id=OWNER_SUBJECT)
    inputs = _by_key(diagnosis.inputs)
    assert inputs["fitness_signature"].status is InputStatus.PRESENT


async def test_diagnose_signature_missing_for_runner_with_only_cycling_signature() -> None:
    """H3: a runner with ONLY a stale cycling signature reports the signature MISSING (fail-closed).

    The mirror of the prior case: a runner (``current_sport="running"``) whose only seeded signature
    is a CYCLING one has no signature for their current sport — a hardcoded-cycling probe would
    falsely report it PRESENT. The sport-keyed probe correctly reads MISSING (a stale cross-sport
    signature never grounds the current sport, GROUND-R7).
    """
    db = await _new_canonical(
        seed_signature=True, rides=0, current_sport="running", signature_sport="cycling"
    )
    engine = GraphAgentEngine(db, FakeModel())  # type: ignore[arg-type]
    diagnosis = await engine.diagnose(athlete_id=OWNER_SUBJECT)
    inputs = _by_key(diagnosis.inputs)
    assert inputs["fitness_signature"].status is InputStatus.MISSING


async def test_diagnose_signature_missing_when_no_current_sport() -> None:
    """H3: with no current sport set there is no sport to ground against -> MISSING (fail-closed).

    A cycling signature exists but the athlete has no ``current_sport``; with nothing to key on, the
    probe must NOT guess a sport — it reports the signature MISSING with the typed
    ``no_current_sport`` reason rather than fabricating cycling coverage (CFG-R1a fail-closed).
    """
    db = await _new_canonical(
        seed_signature=True, rides=0, current_sport=None, signature_sport="cycling"
    )
    engine = GraphAgentEngine(db, FakeModel())  # type: ignore[arg-type]
    diagnosis = await engine.diagnose(athlete_id=OWNER_SUBJECT)
    inputs = _by_key(diagnosis.inputs)
    assert inputs["fitness_signature"].status is InputStatus.MISSING
    assert inputs["fitness_signature"].reason == "no_current_sport"


async def test_diagnose_degrades_with_caveat_when_no_canonical_coverage(
    empty_db: _DatabaseStub,
) -> None:
    """An empty athlete -> every input missing, degraded + typed caveat, no fabrication (GROUND-R7).

    With no rides and no signature, every canonical probe fails closed to ``Unavailable``; the
    diagnosis reports each input ``missing`` (or ``stale``) and the run degrades with a typed
    ``no_canonical_coverage`` caveat naming the unavailable inputs — it never invents a present
    input or a number (OUTCOME-R3/-R4).
    """
    engine = GraphAgentEngine(empty_db, FakeModel())  # type: ignore[arg-type]
    diagnosis = await engine.diagnose(athlete_id=OWNER_SUBJECT)
    assert diagnosis.status is RunStatus.DEGRADED
    assert all(i.status is not InputStatus.PRESENT for i in diagnosis.inputs)
    assert diagnosis.coverage_caveat is not None
    assert diagnosis.coverage_caveat["reason"] == "no_canonical_coverage"
    assert "fitness_signature" in diagnosis.coverage_caveat["inputs_unavailable"]
    assert "hrv" in diagnosis.coverage_caveat["inputs_unavailable"]  # no wellness row seeded


async def test_diagnose_surfaces_no_athlete_facing_number(empty_db: _DatabaseStub) -> None:
    """No coverage line carries a numeric value — a diagnosis reports coverage, never a metric.

    The :class:`InputCoverage` dataclass deliberately has no numeric field; this asserts the typed
    contract (VOICE-R7) so a future field that smuggles a canonical value in would fail the test.
    """
    engine = GraphAgentEngine(empty_db, FakeModel())  # type: ignore[arg-type]
    diagnosis = await engine.diagnose(athlete_id=OWNER_SUBJECT)
    assert InputCoverage.__slots__ == ("key", "label", "status", "reason")
    assert all(i.reason is None or isinstance(i.reason, str) for i in diagnosis.inputs)


async def test_diagnose_coverage_is_deterministic_over_canonical(empty_db: _DatabaseStub) -> None:
    """The projection reads the canonical envelope directly + is stable across calls (GROUND-R7).

    Drives :func:`diagnose_coverage` against the analytics service twice for the SAME pinned date
    and asserts the per-input coverage is identical — the diagnosis is a deterministic function of
    canonical analytics, with no model call and no run-to-run variation.
    """
    today = _dt.date(2026, 6, 9)
    async with empty_db.session() as session:
        svc = AnalyticsService(session)
        first = await diagnose_coverage(svc, OWNER_SUBJECT, today=today)
        second = await diagnose_coverage(svc, OWNER_SUBJECT, today=today)
    assert first.as_of == today.isoformat() == second.as_of
    assert first.inputs == second.inputs  # frozen dataclasses compare by value
    assert first.status is second.status
