"""Unit tests for the SEAM-R11 / ARCH-R31 / CONF-R7 / DEDUP-R6 engine seams.

SEAM-R11 (doc 10) and ARCH-R31's positive clause require ALL canonical-store access —
reads AND writes — to flow through ONE engine-owned session/repository provider obtained
via a typed ``SessionProvider`` Protocol declared in ``wattwise_core.seams``. The provider
takes the server-derived ``subject`` context; the OSS default performs NO tenant scoping
but IS the single attach point the commercial tenant-scoping overlay (COMM-R22) mounts on.

CONF-R7/DEDUP-R6 require the cross-source identity + field-conflict resolver to be a
pluggable strategy INJECTED into ``IngestService`` behind that same seam — never a direct
module import — so the advanced commercial resolver (DEDUP-R8) rides it without code edits.
"""

from __future__ import annotations

import ast
import datetime as _dt
import inspect
import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.engine import GraphAgentEngine
from wattwise_core.api.auth import Principal
from wattwise_core.api.deps import get_db, request_subject
from wattwise_core.config import load_settings
from wattwise_core.domain.candidate import FieldCandidate
from wattwise_core.ingestion.dedup import ResolvedField
from wattwise_core.ingestion.ingest import IngestService
from wattwise_core.persistence import Database
from wattwise_core.persistence.base import Base
from wattwise_core.persistence.models import Athlete  # noqa: F401  populate Base.metadata
from wattwise_core.seams import (
    SYSTEM_SUBJECT,
    ConflictResolver,
    DefaultConflictResolver,
    EngineSessionProvider,
    SessionProvider,
)

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[2] / "src" / "wattwise_core"

# The ONLY modules permitted to construct a canonical-store session factory: the
# engine-owned canonical provider (SEAM-R11/ARCH-R31) and the structurally SEPARATE
# agent-state store (ARCH-R13 — its own engine/pool, never the canonical store).
_SANCTIONED_FACTORY_SITES: frozenset[str] = frozenset(
    {"persistence/engine.py", "agent/state_db.py"}
)

# The ONLY module permitted to open a RAW canonical ``Database.session()`` (no provider seam):
# the OSS default :class:`EngineSessionProvider` itself, which wraps the engine-owned ``Database``
# (SEAM-R11). The canonical ``Database.session`` DEFINITION lives in persistence/engine.py and is
# excluded by receiver classification, not by file. Every other layer MUST open through the
# provider (a ``.session(subject=...)`` call), never around it.
_SANCTIONED_RAW_CANONICAL_OPEN_SITES: frozenset[str] = frozenset({"seams.py"})

# Receiver source-text that resolves to the CANONICAL ``Database`` (the store SEAM-R11 governs).
# A bare ``.session()`` on one of these — WITHOUT the provider's ``subject`` keyword — is a raw
# canonical open. The dedicated agent-state store (``state_db`` / ``self._state_db``, ARCH-R13) is
# a SEPARATE store and is deliberately NOT in this set, so its opens are never flagged.
_CANONICAL_DB_RECEIVERS: frozenset[str] = frozenset(
    {"database", "db", "self._db", "self._database"}
)


def _is_canonical_session_attr(node: ast.expr) -> bool:
    """True when ``node`` is a canonical ``Database.session`` attribute (the SEAM-R11 store).

    i.e. ``X.session`` where ``X`` resolves to the canonical :class:`Database`
    (:data:`_CANONICAL_DB_RECEIVERS`) — the data-access the provider seam governs. The
    structurally separate agent-state store (``state_db`` / ``self._state_db``, ARCH-R13) is
    deliberately NOT in the receiver set, so its ``.session`` is never matched.
    """
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "session"
        and ast.unparse(node.value) in _CANONICAL_DB_RECEIVERS
    )


def _canonical_session_open_offenders() -> list[str]:
    """Every canonical ``Database.session`` reached AROUND the SEAM-R11 provider.

    Walks ``src/wattwise_core`` and flags two distinct bypass shapes, both outside the one
    sanctioned provider module (``seams.py``):

    1. A RAW open CALL ``X.session(...)`` where the receiver ``X`` resolves to the canonical
       :class:`Database` (:data:`_CANONICAL_DB_RECEIVERS`) and the call does NOT pass the
       provider's server-derived ``subject`` keyword — i.e. the store was opened around the
       provider rather than through it.
    2. The canonical bound method ``X.session`` PASSED as a Call ARGUMENT (positional or
       keyword) — e.g. ``SyncOrchestrator(database.session, ...)``. Handing the raw factory to
       another layer lets it open the store around the provider just as surely as calling it
       here, so method-passing is a bypass too. (A provider-seam call ``X.session(subject=...)``
       is the Call's own ``.func``, NOT an argument, so it is correctly never flagged here.)

    A provider call (``.session(subject=...)``) and an agent-state ``state_db.session()``
    (a SEPARATE store, ARCH-R13) are both correctly NOT flagged.
    """
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        rel = path.relative_to(_SRC).as_posix()
        sanctioned = path.name in _SANCTIONED_RAW_CANONICAL_OPEN_SITES
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Shape 2: the canonical bound method handed to a callee as an argument.
            if not sanctioned:
                arg_values = [*node.args, *(kw.value for kw in node.keywords)]
                for arg in arg_values:
                    if _is_canonical_session_attr(arg):
                        offenders.append(f"{rel}:{arg.lineno}")
            # Shape 1: a raw canonical open call lacking the provider's ``subject`` keyword.
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "session":
                continue
            # A provider seam call carries the server-derived ``subject`` — the sanctioned entry
            # point; never a bypass regardless of where it appears.
            if any(kw.arg == "subject" for kw in node.keywords):
                continue
            receiver = ast.unparse(node.func.value)
            if receiver not in _CANONICAL_DB_RECEIVERS:
                continue  # not the canonical store (e.g. the agent-state store, ARCH-R13)
            if sanctioned:
                continue  # the OSS provider itself wrapping the engine ``Database`` (SEAM-R11)
            offenders.append(f"{rel}:{node.lineno}")
    return offenders


@pytest_asyncio.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    """A canonical :class:`Database` over a fresh file-backed SQLite schema."""
    settings = load_settings(
        database_dsn=f"sqlite+aiosqlite:///{tmp_path / 'seam.sqlite'}",
        app__environment="development",
    )
    db = Database(settings)
    try:
        yield db
    finally:
        await db.dispose()


def test_engine_session_provider_satisfies_protocol(database: Database) -> None:
    """The OSS default provider structurally satisfies the SessionProvider seam (SEAM-R11)."""
    assert isinstance(EngineSessionProvider(database), SessionProvider)


def test_session_provider_session_takes_subject() -> None:
    """SEAM-R11: the provider's session entry point takes the server-derived subject."""
    assert "subject" in EngineSessionProvider.session.__annotations__


async def test_default_provider_yields_canonical_session(database: Database) -> None:
    """SEAM-R11: the OSS provider yields a transactional canonical session for a subject.

    No tenant scoping is applied (single-athlete OSS, ARCH-R31): the subject is accepted
    and carried, but the session reaches the un-scoped canonical store.
    """
    provider = EngineSessionProvider(database)
    async with provider.session(subject="athlete-1") as session:
        assert isinstance(session, AsyncSession)


def test_get_db_keys_on_resolved_subject_not_a_baked_value() -> None:
    """ARCH-R31 (finding #3): the request session is keyed on the resolved subject, not coupled.

    ``request_subject`` is the thin seam that yields the verified principal's ``subject`` so the
    canonical session is keyed on the SERVER-DERIVED identity (ARCH-R16) WITHOUT making subject
    resolution a side-effect of the data-access dependency. It returns exactly the principal's
    subject — never a hardcoded/ambient value — so a tenant-scoped overlay reading the same subject
    sees the real identity. ``get_db`` depends on it (not on ``authenticate`` directly), keeping the
    provider decoupled from the auth mechanism (so agent callers with no FastAPI Principal still use
    the same provider seam).
    """
    principal = Principal(subject="athlete-xyz", scopes=frozenset())
    assert request_subject(principal) == "athlete-xyz"
    # Structural guard (decoupling): get_db depends on the subject seam, NOT on authenticate
    # directly — pulling authenticate into the data-access dep was the finding #3 over-reach.
    src = inspect.getsource(get_db)
    assert "Depends(request_subject)" in src
    assert "Depends(authenticate)" not in src


def test_system_subject_is_a_distinct_non_athlete_marker() -> None:
    """SEAM-R11/ARCH-R31: system/probe opens carry an explicit non-scoped subject, not an athlete.

    The readiness probe + the operator erasure executor open through the SAME provider seam but are
    not bound to a request athlete, so they carry :data:`SYSTEM_SUBJECT` — a named, sanctioned
    marker that is deliberately not a valid UUID athlete id (never confusable with a real subject).
    """
    assert SYSTEM_SUBJECT == "_system"
    with pytest.raises(ValueError, match="badly formed"):
        uuid.UUID(SYSTEM_SUBJECT)


class _RecordingSessionProvider:
    """A :class:`SessionProvider` spy: records every ``subject`` then delegates to the real one.

    Proves the engine reaches the canonical store ONLY through the injected provider seam (SEAM-R11)
    — it wraps the real :class:`EngineSessionProvider` so the canonical read still succeeds, while
    recording the server-derived ``subject`` the engine keyed the open on.
    """

    def __init__(self, inner: SessionProvider) -> None:
        self._inner = inner
        self.subjects: list[str] = []

    def session(self, *, subject: str) -> AbstractAsyncContextManager[AsyncSession]:
        self.subjects.append(subject)
        return self._inner.session(subject=subject)


class _SilentModel:
    """A no-op ``ChatModel`` stub: the deterministic ``diagnose`` path never calls the model."""

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        raise AssertionError("diagnose is deterministic and must not call the model")

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        raise AssertionError("diagnose is deterministic and must not call the model")


async def test_engine_canonical_read_flows_through_injected_provider(database: Database) -> None:
    """SEAM-R11 / ARCH-R31: the agent's canonical read goes THROUGH the engine's provider seam.

    The DETERMINISTIC ``diagnose`` is the agent's pure canonical read. A spy ``SessionProvider`` is
    injected into :class:`GraphAgentEngine`; running ``diagnose`` MUST open the canonical store via
    that provider keyed on the server-derived ``athlete_id`` — proving the engine never reaches the
    store around the seam (the bug finding #1 named: ``self._db.session()`` bypasses). If the engine
    reverted to opening ``self._db.session()`` directly, the spy would record NO subject and this
    fails.
    """
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    spy = _RecordingSessionProvider(EngineSessionProvider(database))
    engine = GraphAgentEngine(database, _SilentModel(), sessions=spy)  # type: ignore[arg-type]
    athlete_id = str(uuid.uuid4())
    await engine.diagnose(athlete_id=athlete_id)
    assert spy.subjects == [athlete_id], (
        "the engine MUST open the canonical store through the injected provider seam, "
        f"keyed on the server-derived athlete subject; recorded={spy.subjects!r}"
    )


def test_no_canonical_session_open_outside_provider_seam() -> None:
    """SEAM-R11 / ARCH-R31: every canonical-store open flows THROUGH the provider, never around it.

    The single-choke-point rule: no layer may open its own canonical ``Database.session()``. A
    repo-wide AST scan flags every ``database.session()`` / ``self._db.session()`` call that lacks
    the provider's server-derived ``subject`` keyword (i.e. reaches the store around the seam) AND
    every canonical bound method ``database.session`` PASSED as a Call argument (e.g. handing
    ``database.session`` to a ``SyncOrchestrator`` so it opens around the seam) — both outside the
    ONE sanctioned site — the OSS :class:`EngineSessionProvider` in ``seams.py`` that wraps the
    engine-owned ``Database``. The historical bypass sites (engine.py answer/digest/plan/decision,
    engine_extras.py diagnose/readiness, unconfigured.py diagnose, security.py probe+erasure,
    wiring.py import AND the sync-orchestrator construction) MUST now all route through the provider
    (``.session(subject=...)`` / an injected ``SessionProvider``); an agent-state
    ``state_db.session()`` (a SEPARATE store, ARCH-R13) is correctly NOT flagged.
    """
    offenders = _canonical_session_open_offenders()
    assert offenders == [], (
        "canonical store opened around the SEAM-R11 provider (no `subject`): "
        + ", ".join(offenders)
    )


def test_no_session_factory_constructed_outside_engine_modules() -> None:
    """ARCH-R31: only the two engine store modules mint a session factory (no rogue factories).

    A complementary AST scan finds every ``async_sessionmaker(...)`` construction; the only
    sanctioned sites are the engine-owned canonical engine module (persistence/engine.py — the
    SEAM-R11 choke point's factory) and the structurally separate agent-state store
    (agent/state_db.py, ARCH-R13). Any other layer minting its own factory could then open the
    store around the provider, so this guards the factory layer below
    :func:`test_no_canonical_session_open_outside_provider_seam`.
    """
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        rel = path.relative_to(_SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "async_sessionmaker"
                and rel not in _SANCTIONED_FACTORY_SITES
            ):
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == [], (
        "canonical-store session factory constructed outside the engine store modules: "
        + ", ".join(offenders)
    )


def test_conflict_resolver_default_satisfies_protocol() -> None:
    """CONF-R7/DEDUP-R6: the OSS default resolver is a typed, injectable seam impl."""
    assert isinstance(DefaultConflictResolver(), ConflictResolver)


async def test_ingest_service_uses_injected_resolver() -> None:
    """CONF-R7/DEDUP-R6: IngestService consults the INJECTED resolver, not a hard import.

    A spy resolver whose ``resolve_activity_identity`` records its call and forces a
    NON-match decision is injected; the service uses it for identity resolution. The
    default conservative resolver WOULD merge the windowed same-sport activity, so the
    forced non-match minting a NEW id proves the injected resolver — not a directly
    imported function — is the identity-resolution authority (swappable behind the seam).
    """
    calls: list[str] = []

    class _SpyResolver:
        def resolve_field(
            self,
            candidates: list[FieldCandidate],
            *,
            dispute_tolerance: float | None = None,
        ) -> ResolvedField | None:
            return DefaultConflictResolver().resolve_field(
                candidates, dispute_tolerance=dispute_tolerance
            )

        def resolve_activity_identity(self, *args: object, **kwargs: object) -> bool:
            calls.append(str(args[2]))  # record the sport arg the service passed in
            return False  # force a NEW id regardless of the windowed match

    existing = _StubActivity(
        activity_id=uuid.uuid4(),
        start_time=_dt.datetime(2026, 6, 1, 8, 0, tzinfo=_dt.UTC),
        elapsed_time_s=3600,
        sport="cycling",
    )
    service = IngestService(_StubSession([existing]), resolver=_SpyResolver())  # type: ignore[arg-type]
    cand = _StubCandidate(
        {"start_time": "2026-06-01T08:00:00+00:00", "elapsed_time_s": 3600, "sport": "cycling"}
    )
    resolved, decision = await service._resolve_activity_id(uuid.uuid4(), cand)  # type: ignore[arg-type]
    assert calls, "the injected resolver MUST be the identity-resolution authority"
    assert resolved != existing.activity_id  # spy forced a new id, not the windowed match
    assert decision["rule"] == "no_match_new_record"  # the MAP-R12 decision is recorded


class _StubActivity:
    """A minimal stand-in for a persisted ``Activity`` row the windowed query returns."""

    def __init__(
        self, *, activity_id: object, start_time: object, elapsed_time_s: int, sport: str
    ) -> None:
        self.activity_id = activity_id
        self.start_time = start_time
        self.elapsed_time_s = elapsed_time_s
        self.sport = sport


class _StubCandidate:
    """A minimal candidate carrying only the payload the identity path reads."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.strong_fingerprint: str | None = None  # no typed fingerprint on the stub


class _StubResult:
    """A SQLAlchemy-result stand-in for the windowed-activities query."""

    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _StubResult:
        return self

    def all(self) -> list[object]:
        return self._rows


class _StubSession:
    """An ``AsyncSession`` stand-in returning a fixed windowed-activity set."""

    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    async def execute(self, _stmt: object) -> _StubResult:
        return _StubResult(self._rows)
