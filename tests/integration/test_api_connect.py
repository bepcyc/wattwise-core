"""Integration tests for the connect -> import -> sync -> onboarding surface (doc 60).

Exercises the OSS connection lifecycle end-to-end over the portable in-memory
substrate, with the credential probe, credential store, import processor, and sync
orchestrator wired as in-test fakes (the seams the app factory overrides in
production). Proves the load-bearing invariants of the p4 slice:

- **API-R42** ``GET /v1/connections/available`` returns the fixed OSS catalog (a
  ``file_upload`` importer + one ``api_key`` source); no OAuth archetype appears.
- **API-R43 / SCHEMA-R10** ``initiate`` returns the archetype-discriminated next step;
  an unknown source -> ``404``.
- **API-R44 / AUTH-R16 / AUTH-R17** ``complete`` runs the MANDATORY probe BEFORE
  ``connected``; a good key stores an opaque ref + writes a ``connected`` row; a bad
  key -> ``422 credential-invalid`` with NO half-connected row, and the raw secret is
  never persisted.
- **API-R33 / LIMIT-R5** ``POST /v1/imports`` accepts a supported file (``202``),
  rejects an unsupported extension (``415``), an oversized upload (``413``), and a
  structurally invalid file (``422 import-rejected``); the import lands as a
  connectionless ``file_import`` candidate.
- **API-R46** ``POST /v1/sync/run`` is the only (manual) trigger -> ``202``; mutually
  exclusive scope -> ``422``; unknown connection -> ``404``. ``api_key`` connect does
  NOT auto-enqueue a sync.
- **API-R46 (onboarding)** ``GET /v1/onboarding/status`` is derived; a fresh ``api_key``
  connection reports ``first_sync_state="not_started"`` until data lands.

Tier: T-INTEGRATION (offline; in-process ASGI via ``TestClient`` over a fresh
in-memory canonical schema; the auth/db/seam dependencies are overridden in-test).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

import pytest
from anyio.from_thread import BlockingPortal
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.api import connection_catalog
from wattwise_core.api.auth import Principal, Scope, authenticate
from wattwise_core.api.deps import get_db, get_settings
from wattwise_core.api.errors import install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import connections as connections_router
from wattwise_core.api.routers import imports as imports_router
from wattwise_core.api.routers import onboarding as onboarding_router
from wattwise_core.api.routers import sync as sync_router
from wattwise_core.config import Settings, load_settings
from wattwise_core.domain.enums import ConnectionStatus
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    Base,
    Connection,
    SourceDescriptor,
    Sport,
)

pytestmark = pytest.mark.integration

UTC = _dt.UTC

#: The fixed owner subject the in-test auth seam resolves (server-derived, AUTH-R3).
_OWNER_SCOPES = frozenset(
    {Scope.READ, Scope.WRITE, Scope.AGENT, Scope.SYNC, Scope.EXPORT, Scope.ADMIN}
)

#: A small, valid FIT-shaped header so the file-upload extension check passes; the
#: in-test processor decides accept/reject, so the bytes need only be non-empty.
_FIT_BYTES = b"\x0c\x10\x00\x00\x00\x00\x00\x00.FITabcd"


# --------------------------------------------------------------------------- in-test fakes


class _FakeProbe:
    """A credential probe that accepts one known-good key and records calls (AUTH-R17)."""

    def __init__(self, good_key: str) -> None:
        self._good = good_key
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, source: str, secret: str) -> None:
        self.calls.append((source, secret))
        if secret != self._good:
            raise connections_router.CredentialProbeError(source)


class _FakeSink:
    """A credential store that issues an opaque ref and never returns the raw secret."""

    def __init__(self) -> None:
        self.stored: list[str] = []

    def store(self, raw_secret: str) -> str:
        self.stored.append(raw_secret)
        return f"cred_ref_{len(self.stored)}"


class _FakeProcessor:
    """An import processor that rejects a known-bad sentinel and accepts otherwise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None]] = []

    async def __call__(
        self, athlete_id: str, data: bytes, filename: str | None
    ) -> imports_router.ImportJob:
        self.calls.append((athlete_id, len(data), filename))
        if data.startswith(b"CORRUPT"):
            raise imports_router.ImportRejected(
                code="unreadable_file", reason="We couldn't read that file."
            )
        return imports_router.queued_job(f"import_{len(self.calls)}", filename)


class _FakeOrchestrator:
    """A sync orchestrator that records the resolved target and returns a started run."""

    def __init__(self) -> None:
        self.targets: list[sync_router.SyncTarget] = []

    async def __call__(self, target: sync_router.SyncTarget) -> sync_router.SyncRun:
        self.targets.append(target)
        return sync_router.started_run(f"sync_{len(self.targets)}")


# --------------------------------------------------------------------------- harness


class _Harness:
    """Bundles the app, client, athlete id, and in-test seams for one test scenario.

    Every DB read/write goes through :meth:`run` so it executes on the SAME event loop
    the ASGI app (and the shared :class:`AsyncSession`) runs on — the TestClient's
    anyio portal. Mixing a foreign loop with the request loop would detach the session.
    """

    def __init__(
        self,
        client: TestClient,
        session: AsyncSession,
        athlete_id: str,
        settings: Settings,
        probe: _FakeProbe,
        sink: _FakeSink,
        processor: _FakeProcessor,
        orchestrator: _FakeOrchestrator,
    ) -> None:
        self.client = client
        self._session = session
        self.athlete_id = athlete_id
        self.settings = settings
        self.probe = probe
        self.sink = sink
        self.processor = processor
        self.orchestrator = orchestrator

    def run[T](self, coro_fn: Callable[[AsyncSession], Awaitable[T]]) -> T:
        """Run a DB coroutine against the shared session on the app's event loop."""
        return _portal(self.client).call(coro_fn, self._session)


@pytest.fixture
def harness() -> Iterator[_Harness]:
    """Build the app + client with the four routers mounted and the seams overridden.

    The engine/session/seed are created INSIDE the TestClient's portal (on its event
    loop) so the shared session and the request handlers never cross loops.
    """
    settings = load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="integration-test-key",
    )
    probe, sink = _FakeProbe(good_key="good-key"), _FakeSink()
    processor, orchestrator = _FakeProcessor(), _FakeOrchestrator()
    holder: dict[str, AsyncSession] = {}
    app = _build_app(settings, holder, probe, sink, processor, orchestrator)

    with TestClient(app, raise_server_exceptions=False) as client:
        engine, session, athlete_id = _portal(client).call(_open_db)
        holder["session"] = session
        try:
            yield _Harness(
                client, session, athlete_id, settings, probe, sink, processor, orchestrator
            )
        finally:
            _portal(client).call(_close_db, session, engine)


def _portal(client: TestClient) -> BlockingPortal:
    """Return the TestClient's anyio portal (started on ``__enter__``); never None here."""
    portal = client.portal
    assert portal is not None  # the portal exists inside the ``with TestClient(...)`` block
    return portal


async def _open_db() -> tuple[object, AsyncSession, str]:
    """Create the in-memory schema, open a session, and seed the owner (on the app loop)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    session = factory()
    athlete_id = await _seed(session)
    session.info["athlete_id"] = athlete_id  # the in-test auth seam reads this
    return engine, session, athlete_id


async def _close_db(session: AsyncSession, engine: object) -> None:
    """Close the session and dispose the engine (on the app loop)."""
    await session.close()
    await engine.dispose()  # type: ignore[attr-defined]


def _build_app(
    settings: Settings,
    holder: dict[str, AsyncSession],
    probe: _FakeProbe,
    sink: _FakeSink,
    processor: _FakeProcessor,
    orchestrator: _FakeOrchestrator,
) -> FastAPI:
    """Assemble a minimal app: error handlers + the four routers + overridden seams."""
    app = FastAPI()
    app.state.settings = settings
    app.state.rate_limiter = RateLimiter()  # the per-athlete read/mutating buckets (LIMIT-R1)
    install_error_handlers(app)
    for module in (connections_router, imports_router, sync_router, onboarding_router):
        app.include_router(module.router)

    async def _session() -> AsyncIterator[AsyncSession]:
        yield holder["session"]

    app.dependency_overrides[authenticate] = lambda: Principal(
        subject=holder["session"].info.get("athlete_id", ""), scopes=_OWNER_SCOPES
    )
    app.dependency_overrides[get_db] = _session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[connections_router.credential_probe] = lambda: probe
    app.dependency_overrides[connections_router.credential_sink] = lambda: (
        connections_router.CredentialSink(store=sink.store)
    )
    app.dependency_overrides[imports_router.import_processor] = lambda: processor
    app.dependency_overrides[sync_router.sync_orchestrator] = lambda: orchestrator
    return app


async def _seed(session: AsyncSession) -> str:
    """Seed the single athlete, the cycling sport, and the two OSS source descriptors."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    session.add(
        SourceDescriptor(
            source_key="file_import", display_name="Activity files", kind="file_upload"
        )
    )
    session.add(
        SourceDescriptor(source_key="intervals_icu", display_name="Intervals.icu", kind="oauth_api")
    )
    await session.commit()
    return str(athlete.athlete_id)


def _auth() -> dict[str, str]:
    """A bearer header so the route's security extractor is satisfied (value unused)."""
    return {"Authorization": "Bearer test"}


# --------------------------------------------------------------------------- catalog


def test_available_catalog_is_the_two_oss_archetypes(harness: _Harness) -> None:
    """GET /available lists exactly the file-upload + api_key sources, no OAuth (API-R42)."""
    resp = harness.client.get("/v1/connections/available", headers=_auth())
    assert resp.status_code == 200
    sources = {s["source"]: s for s in resp.json()["sources"]}
    assert set(sources) == {"file_import", "intervals_icu"}
    assert sources["file_import"]["auth_archetype"] == "file_upload"
    assert sources["intervals_icu"]["auth_archetype"] == "api_key"
    archetypes = {s["auth_archetype"] for s in sources.values()}
    assert "oauth_redirect" not in archetypes  # OAuth is commercial-only (COMM-R18)


# --------------------------------------------------------------------------- initiate


def test_initiate_api_key_returns_key_next_step(harness: _Harness) -> None:
    """initiate for an api_key source returns the {label, hint_url} next step (API-R43)."""
    resp = harness.client.post("/v1/connections/intervals_icu/initiate", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "api_key"
    assert body["label"] and body["hint_url"]


def test_initiate_file_upload_returns_accepted_formats(harness: _Harness) -> None:
    """initiate for the file-upload source returns the accepted formats (API-R43)."""
    resp = harness.client.post("/v1/connections/file_import/initiate", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "file_upload"
    # Pin every advertised format so dropping one (e.g. a regression removing .pwx) fails here.
    assert {".fit", ".fit.gz", ".gpx", ".tcx", ".pwx"} <= set(body["accepted_formats"])


def test_initiate_unknown_source_is_404(harness: _Harness) -> None:
    """initiate for a source not in the catalog -> 404 not-found (API-R51)."""
    resp = harness.client.post("/v1/connections/garmin/initiate", headers=_auth())
    assert resp.status_code == 404
    assert resp.json()["type"].endswith("/not-found")


# --------------------------------------------------------------------------- complete


def test_complete_good_key_probes_then_connects(harness: _Harness) -> None:
    """A good key probes FIRST, stores an opaque ref, and writes a connected row (API-R44)."""
    resp = harness.client.post(
        "/v1/connections/intervals_icu/complete",
        headers=_auth(),
        json={"api_key": "good-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "connected"
    assert body["source"] == "intervals_icu"
    # The probe ran before connecting (AUTH-R17) and the raw key was handed to the store.
    assert harness.probe.calls == [("intervals_icu", "good-key")]
    assert harness.sink.stored == ["good-key"]
    # The persisted row holds the OPAQUE ref, never the raw secret (AUTH-R16).
    conn = _only_connection(harness)
    assert conn.status is ConnectionStatus.CONNECTED
    assert conn.credential_ref == "cred_ref_1"
    assert conn.credential_ref != "good-key"


def test_complete_bad_key_is_422_with_no_half_connected_row(harness: _Harness) -> None:
    """A bad key -> 422 credential-invalid and NO connection row is created (AUTH-R17)."""
    resp = harness.client.post(
        "/v1/connections/intervals_icu/complete",
        headers=_auth(),
        json={"api_key": "wrong-key"},
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/credential-invalid")
    # The probe rejected it, so nothing was stored and no half-connected row exists.
    assert harness.sink.stored == []
    assert _connection_count(harness) == 0


def test_complete_rejects_unknown_property_in_body(harness: _Harness) -> None:
    """A forged caller-identity property in the body is rejected (AUTH-R3 / SCHEMA-R4)."""
    resp = harness.client.post(
        "/v1/connections/intervals_icu/complete",
        headers=_auth(),
        json={"api_key": "good-key", "athlete_id": "impostor"},
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


def test_complete_file_upload_source_is_422(harness: _Harness) -> None:
    """Completing the file-upload source (no api_key step) -> 422 (API-R44)."""
    resp = harness.client.post(
        "/v1/connections/file_import/complete",
        headers=_auth(),
        json={"api_key": "good-key"},
    )
    assert resp.status_code == 422


def test_reconnect_replaces_ref_without_duplicating_row(harness: _Harness) -> None:
    """Re-completing the same source replaces the ref atomically, one row (API-R44)."""
    first = harness.client.post(
        "/v1/connections/intervals_icu/complete",
        headers=_auth(),
        json={"api_key": "good-key"},
    )
    second = harness.client.post(
        "/v1/connections/intervals_icu/complete",
        headers=_auth(),
        json={"api_key": "good-key"},
    )
    assert first.status_code == second.status_code == 200
    assert _connection_count(harness) == 1
    assert _only_connection(harness).credential_ref == "cred_ref_2"


def test_reconnect_route_recovers_an_errored_connection(harness: _Harness) -> None:
    """POST /{connection_id}/reconnect re-auths an api_key connection in place (API-R45)."""
    harness.client.post(
        "/v1/connections/intervals_icu/complete", headers=_auth(), json={"api_key": "good-key"}
    )
    connection_id = str(_only_connection(harness).connection_id)
    resp = harness.client.post(
        f"/v1/connections/{connection_id}/reconnect",
        headers=_auth(),
        json={"api_key": "good-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "connected"
    # The probe ran before reconnecting (AUTH-R17) and the ref was atomically replaced,
    # with no duplicate row minted (one connection, history preserved).
    assert _connection_count(harness) == 1
    assert _only_connection(harness).credential_ref == "cred_ref_2"


def test_reconnect_bad_key_is_422_credential_invalid(harness: _Harness) -> None:
    """A bad key on reconnect -> 422 credential-invalid; the row is untouched (API-R45/R17)."""
    harness.client.post(
        "/v1/connections/intervals_icu/complete", headers=_auth(), json={"api_key": "good-key"}
    )
    connection_id = str(_only_connection(harness).connection_id)
    resp = harness.client.post(
        f"/v1/connections/{connection_id}/reconnect", headers=_auth(), json={"api_key": "bad"}
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/credential-invalid")
    assert _only_connection(harness).credential_ref == "cred_ref_1"  # untouched


def test_reconnect_unknown_connection_is_404(harness: _Harness) -> None:
    """Reconnecting an unknown/foreign connection id -> 404 (API-R51)."""
    resp = harness.client.post(
        "/v1/connections/00000000-0000-0000-0000-000000000000/reconnect",
        headers=_auth(),
        json={"api_key": "good-key"},
    )
    assert resp.status_code == 404


def test_initiate_complete_accept_write_only_token(harness: _Harness) -> None:
    """initiate/complete require WRITE only (not READ+WRITE), per API-R43/R44/AUTH-R11."""
    # Re-bind the auth seam to a WRITE-only principal (no READ scope).
    harness.client.app.dependency_overrides[authenticate] = lambda: Principal(
        subject=harness.athlete_id, scopes=frozenset({Scope.WRITE, Scope.SYNC})
    )
    try:
        init = harness.client.post("/v1/connections/intervals_icu/initiate", headers=_auth())
        assert init.status_code == 200  # a write-only token is accepted (no extra read needed)
        done = harness.client.post(
            "/v1/connections/intervals_icu/complete", headers=_auth(), json={"api_key": "good-key"}
        )
        assert done.status_code == 200
    finally:
        harness.client.app.dependency_overrides[authenticate] = lambda: Principal(
            subject=harness.athlete_id, scopes=_OWNER_SCOPES
        )


# --------------------------------------------------------------------------- imports


def test_import_supported_file_is_202(harness: _Harness) -> None:
    """A supported file is accepted (202) and processed into a queued ImportJob (API-R33)."""
    resp = harness.client.post(
        "/v1/imports",
        headers=_auth(),
        files={"file": ("ride.fit", _FIT_BYTES, "application/octet-stream")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["filename"] == "ride.fit"
    # The processor saw the verbatim bytes and the acting athlete id (AUTH-R3).
    assert harness.processor.calls[0][0] == harness.athlete_id


def test_import_unsupported_extension_is_415(harness: _Harness) -> None:
    """An unsupported extension -> 415 unsupported-media-type before any parse (API-R33)."""
    resp = harness.client.post(
        "/v1/imports",
        headers=_auth(),
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415
    assert resp.json()["type"].endswith("/unsupported-media-type")
    assert harness.processor.calls == []  # never reached the processor


def test_import_oversized_file_is_413(harness: _Harness) -> None:
    """An upload past the cap -> 413 payload-too-large (LIMIT-R5)."""
    harness.settings.api__request_max_bytes = 1024
    big = b"\x0c\x10\x00\x00\x00\x00\x00\x00.FIT" + b"0" * 4096
    resp = harness.client.post(
        "/v1/imports",
        headers=_auth(),
        files={"file": ("ride.fit", big, "application/octet-stream")},
    )
    assert resp.status_code == 413
    assert resp.json()["type"].endswith("/payload-too-large")


def test_import_corrupt_file_is_422_import_rejected(harness: _Harness) -> None:
    """A structurally invalid file -> 422 import-rejected with a machine code (API-R33)."""
    resp = harness.client.post(
        "/v1/imports",
        headers=_auth(),
        files={"file": ("ride.fit", b"CORRUPT-not-a-fit", "application/octet-stream")},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["type"].endswith("/import-rejected")
    assert any(e["code"] == "unreadable_file" for e in body["errors"])


def test_import_accepts_double_extension_fit_gz(harness: _Harness) -> None:
    """A .fit.gz upload is accepted as a whole-suffix match (API-R33)."""
    resp = harness.client.post(
        "/v1/imports",
        headers=_auth(),
        files={"file": ("ride.fit.gz", _FIT_BYTES, "application/gzip")},
    )
    assert resp.status_code == 202


def test_import_pwx_extension_is_accepted_202(harness: _Harness) -> None:
    """A .pwx upload passes the extension gate -> 202 (pins imports.ACCEPTED_EXTENSIONS, API-R33).

    The advertised-format list and the actual upload gate are duplicated constants; this
    pins the gate side so dropping .pwx from imports.ACCEPTED_EXTENSIONS can no longer slip a
    real .pwx upload into a silent 415 while initiate still advertises it.
    """
    resp = harness.client.post(
        "/v1/imports",
        headers=_auth(),
        files={"file": ("ride.pwx", b"<pwx>data</pwx>", "application/xml")},
    )
    assert resp.status_code == 202


def test_accepted_format_lists_agree() -> None:
    """The upload gate and advertised-format list MUST stay in sync (no advertise/accept gap)."""
    assert set(imports_router.ACCEPTED_EXTENSIONS) == set(connection_catalog.ACCEPTED_FILE_FORMATS)


# --------------------------------------------------------------------------- sync


def test_sync_run_all_connections_is_202(harness: _Harness) -> None:
    """POST /sync/run with no body syncs every owner connection -> 202 (API-R46)."""
    resp = harness.client.post("/v1/sync/run", headers=_auth())
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"
    # The orchestrator received the server-derived owner + no connection scope.
    target = harness.orchestrator.targets[0]
    assert target.athlete_id == harness.athlete_id
    assert target.connection_id is None


def test_sync_run_scoped_to_known_connection_is_202(harness: _Harness) -> None:
    """A run scoped to an existing owner connection resolves and starts (API-R46)."""
    harness.client.post(
        "/v1/connections/intervals_icu/complete",
        headers=_auth(),
        json={"api_key": "good-key"},
    )
    connection_id = str(_only_connection(harness).connection_id)
    resp = harness.client.post(
        "/v1/sync/run", headers=_auth(), json={"connection_id": connection_id}
    )
    assert resp.status_code == 202
    assert harness.orchestrator.targets[-1].connection_id == connection_id


def test_sync_run_unknown_connection_is_404(harness: _Harness) -> None:
    """A run scoped to a connection the owner doesn't have -> 404 (API-R51)."""
    resp = harness.client.post(
        "/v1/sync/run",
        headers=_auth(),
        json={"connection_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert resp.status_code == 404
    assert harness.orchestrator.targets == []  # never reached the orchestrator


def test_sync_run_both_scopes_is_422(harness: _Harness) -> None:
    """Naming both a connection and a source is mutually exclusive -> 422 (API-R46)."""
    resp = harness.client.post(
        "/v1/sync/run",
        headers=_auth(),
        json={"connection_id": "abc", "source": "intervals_icu"},
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


# --------------------------------------------------------------------------- onboarding


def test_onboarding_no_connection_suggests_connecting(harness: _Harness) -> None:
    """With nothing connected, onboarding suggests connecting a source (API-R46)."""
    resp = harness.client.get("/v1/onboarding/status", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_connection"] is False
    assert body["first_sync_state"] == "not_started"
    assert body["first_data_ready"] is False
    assert body["suggested_next_step"] == "connect_a_source"


def test_onboarding_api_key_connect_does_not_auto_enqueue_sync(harness: _Harness) -> None:
    """A fresh api_key connection stays first_sync_state=not_started (OSS carve-out, API-R46)."""
    harness.client.post(
        "/v1/connections/intervals_icu/complete",
        headers=_auth(),
        json={"api_key": "good-key"},
    )
    resp = harness.client.get("/v1/onboarding/status", headers=_auth())
    body = resp.json()
    assert body["has_connection"] is True
    # No sync was auto-enqueued and no data has landed -> still not_started (API-R46).
    assert body["first_sync_state"] == "not_started"
    assert body["first_data_ready"] is False
    assert body["suggested_next_step"] == "run_first_sync"
    assert harness.orchestrator.targets == []  # connect did NOT trigger a sync


def test_onboarding_complete_once_data_lands(harness: _Harness) -> None:
    """Once a canonical activity exists, onboarding reports complete + all_set (API-R46)."""
    _land_activity(harness)
    resp = harness.client.get("/v1/onboarding/status", headers=_auth())
    body = resp.json()
    assert body["first_data_ready"] is True
    assert body["first_sync_state"] == "complete"
    assert body["suggested_next_step"] == "all_set"


# --------------------------------------------------------------------------- db helpers


def _connection_count(harness: _Harness) -> int:
    """Count the owner's persisted connection rows (half-connected-row assertion)."""
    return len(harness.run(_fetch_connections))


def _only_connection(harness: _Harness) -> Connection:
    """Return the single persisted connection (asserts exactly one)."""
    rows = harness.run(_fetch_connections)
    assert len(rows) == 1
    return rows[0]


async def _fetch_connections(session: AsyncSession) -> list[Connection]:
    """Read every connection row from the test session."""
    return list((await session.execute(select(Connection))).scalars().all())


def _land_activity(harness: _Harness) -> None:
    """Insert one canonical activity for the owner (simulates first data landing)."""
    harness.run(_insert_activity)


async def _insert_activity(session: AsyncSession) -> None:
    """Persist a minimal canonical activity for the owner."""
    session.add(
        Activity(
            athlete_id=uuid.UUID(session.info["athlete_id"]),
            sport="cycling",
            start_time=_dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
            elapsed_time_s=3600,
        )
    )
    await session.commit()
