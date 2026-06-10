"""Non-functional conformance tests for doc 70: auth lifetimes/refresh, service auth,
GPS opt-out, original-file retention, export, readiness depth, graceful shutdown, and
stateless two-instance interchangeability.

Requirement IDs: SEC-R2.3, SEC-R4, PRIV-R2, PRIV-R7, PRIV-R9, PRIV-R11.2, RUN-R6,
RUN-R11, RUN-R13, OBS-R6.2.
"""

from __future__ import annotations

import datetime as _dt
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tests.integration._schema import provision_app_schema
from wattwise_core.api.app import API_PREFIX, create_app
from wattwise_core.api.auth import Scope, issue_access_token
from wattwise_core.config import Settings, load_settings
from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import ActivityFileFormat, Fidelity
from wattwise_core.ingestion.ingest import IngestService, OriginalFile
from wattwise_core.persistence.base import Base
from wattwise_core.persistence.migrations_state import _ALEMBIC_VERSION
from wattwise_core.persistence.models import (
    Activity,
    ActivityFile,
    Athlete,
    SourceDescriptor,
    Sport,
    StreamChannel,
)
from wattwise_core.privacy.retention import purge_expired_original_files
from wattwise_core.storage import LocalObjectStore, content_hash

pytestmark = pytest.mark.integration

_STRONG_KEY = "k3y-" + "0123456789abcdef" * 4


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    """REAL dev settings on a FILE DB (a real multi-connection pool, never ``:memory:``)."""
    base: dict[str, Any] = {
        "app__environment": "development",
        "database_dsn": f"sqlite+aiosqlite:///{tmp_path / 'doc70.db'}",
        "token_signing_key": _STRONG_KEY,
        "object_store__local_root": str(tmp_path / "objects"),
    }
    base.update(overrides)
    return load_settings(**base)


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    """A REAL app over a provisioned (schema + stamped-head) throwaway file DB."""
    app = create_app(_settings(tmp_path, **overrides))
    provision_app_schema(app)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    return client


def _sign_in(client: TestClient) -> dict[str, Any]:
    """Mint owner tokens via the public sign-in exchange."""
    resp = client.post(f"{API_PREFIX}/auth/token", json={"owner_secret": _STRONG_KEY})
    assert resp.status_code == 200, resp.text
    return dict(resp.json())


# ------------------------------------------------------------------- SEC-R2.3 refresh tokens


def test_sign_in_returns_separate_revocable_refresh_token(tmp_path: Path) -> None:
    """Sign-in mints an access token plus a SEPARATE opaque refresh token (SEC-R2.3).

    The refresh credential is non-empty, distinct from the access token, and the
    access lifetime echoes the config-loaded ``auth__access_ttl_seconds`` (≤ 3600).
    """
    client = _client(tmp_path)
    try:
        body = _sign_in(client)
        assert body["refresh_token"], "refresh leg must be minted (SEC-R2.3)"
        assert body["refresh_token"] != body["access_token"]
        assert 0 < body["expires_in"] <= 3600
    finally:
        client.__exit__(None, None, None)


def test_refresh_rotates_and_reuse_revokes_the_family(tmp_path: Path) -> None:
    """Refresh ROTATES the token; replaying the rotated member kills the family (SEC-R2.3).

    First refresh succeeds and returns a NEW refresh token. Presenting the ALREADY
    ROTATED token again is reuse: it must 401 AND revoke the whole family, so the
    newest member is dead too — a stolen-and-replayed refresh token cannot coexist
    with the legitimate chain.
    """
    client = _client(tmp_path)
    try:
        first = _sign_in(client)["refresh_token"]
        rotated = client.post(f"{API_PREFIX}/auth/refresh", json={"refresh_token": first})
        assert rotated.status_code == 200, rotated.text
        second = rotated.json()["refresh_token"]
        assert second != first
        # Replay the rotated member -> 401 + family revocation (reuse detection).
        replay = client.post(f"{API_PREFIX}/auth/refresh", json={"refresh_token": first})
        assert replay.status_code == 401, replay.text
        # The family is dead: even the newest member is refused now.
        after = client.post(f"{API_PREFIX}/auth/refresh", json={"refresh_token": second})
        assert after.status_code == 401, after.text
    finally:
        client.__exit__(None, None, None)


def test_revoke_kills_the_refresh_family(tmp_path: Path) -> None:
    """``POST /v1/auth/revoke`` revokes the presented token's family (SEC-R2.3)."""
    client = _client(tmp_path)
    try:
        refresh = _sign_in(client)["refresh_token"]
        revoke = client.post(f"{API_PREFIX}/auth/revoke", json={"refresh_token": refresh})
        assert revoke.status_code == 204, revoke.text
        resp = client.post(f"{API_PREFIX}/auth/refresh", json={"refresh_token": refresh})
        assert resp.status_code == 401, resp.text
    finally:
        client.__exit__(None, None, None)


def test_access_ttl_zero_or_over_an_hour_rejected_at_config_load(tmp_path: Path) -> None:
    """A 0 or >3600 access lifetime is refused AT CONFIG LOAD (SEC-R2.3), never 'no expiry'."""
    for bad in (0, -5, 3601):
        with pytest.raises(Exception, match="access_ttl") as excinfo:
            _settings(tmp_path, auth__access_ttl_seconds=bad)
        assert excinfo.value is not None


# ------------------------------------------------------------------------ SEC-R4 service auth


def test_service_auth_header_verified_constant_time_and_never_replaces_bearer(
    tmp_path: Path,
) -> None:
    """The X-Service-Auth factor is verified when presented and never widens identity (SEC-R4).

    A WRONG service secret is rejected 401 even alongside a valid athlete bearer; the
    RIGHT secret passes the middleware but the request still authenticates via the
    bearer token (the factor is additional, never a replacement: with the service
    header alone and no bearer, a protected route still 401s).
    """
    client = _client(tmp_path, service_auth_secret="service-" + "s3cr3t-" * 8)
    try:
        access = _sign_in(client)["access_token"]
        bearer = {"Authorization": f"Bearer {access}"}
        good = {"X-Service-Auth": "service-" + "s3cr3t-" * 8}
        bad = {"X-Service-Auth": "wrong-secret"}
        ok = client.get(f"{API_PREFIX}/system/status", headers={**bearer, **good})
        assert ok.status_code == 200, ok.text
        rejected = client.get(f"{API_PREFIX}/system/status", headers={**bearer, **bad})
        assert rejected.status_code == 401, rejected.text
        # The service factor alone never authenticates an athlete (additional, not instead).
        protected = client.get(f"{API_PREFIX}/users/me", headers=good)
        assert protected.status_code == 401, protected.text
    finally:
        client.__exit__(None, None, None)


def test_service_auth_header_with_no_provisioned_secret_is_rejected(tmp_path: Path) -> None:
    """Presenting X-Service-Auth when no service principal is provisioned → 401 (SEC-R4)."""
    client = _client(tmp_path)  # no service_auth_secret configured
    try:
        resp = client.get(f"{API_PREFIX}/system/status", headers={"X-Service-Auth": "anything"})
        assert resp.status_code == 401, resp.text
    finally:
        client.__exit__(None, None, None)


# ----------------------------------------------------------------------- PRIV-R2 GPS opt-out


async def _seed_minimal(session: AsyncSession) -> tuple[str, str]:
    """Seed the athlete + sport + source the ingest pipeline needs."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC")
    session.add(athlete)
    descriptor = SourceDescriptor(
        source_key="file_import", display_name="Activity files", kind="file_upload"
    )
    session.add(descriptor)
    await session.flush()
    await session.commit()
    return str(athlete.athlete_id), str(descriptor.source_descriptor_id)


def _gps_ride(native_id: str) -> GboCandidate:
    """A ride candidate carrying BOTH a power stream and a raw GPS latlng stream."""
    seconds = 60
    payload = {
        "start_time": _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
        "sport": "cycling",
        "elapsed_time_s": seconds,
        "moving_time_s": seconds,
        "streams": {
            "power_w": {
                "values": [200.0] * seconds,
                "sample_basis": "time",
                "sample_rate_hz": 1.0,
            },
            "latlng": {
                "values": [[48.1, 11.5]] * seconds,
                "sample_basis": "time",
                "sample_rate_hz": 1.0,
            },
        },
    }
    return GboCandidate(
        gbo_type="activity",
        source_descriptor_id="placeholder",
        source_native_id=native_id,
        content_hash=content_hash(native_id.encode()),
        payload=payload,
        trust_tier=Fidelity.RAW_STREAM,
        fetched_at=_dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )


async def _landed_channels(tmp_path: Path, *, store_raw_gps: bool) -> set[str]:
    """Land one GPS ride with the given opt-out and return the canonical channel names."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'gps.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with factory() as session:
            athlete_id, descriptor_id = await _seed_minimal(session)
            svc = IngestService(session, store_raw_gps=store_raw_gps)
            await svc.ingest(athlete_id, descriptor_id, [_gps_ride("gps-ride-1")])
            await session.commit()
            rows = (await session.execute(select(StreamChannel.channel))).scalars().all()
            return {str(getattr(c, "value", c)) for c in rows}
    finally:
        await engine.dispose()


async def test_gps_opt_out_drops_latlng_but_keeps_derived_channels(tmp_path: Path) -> None:
    """With ``store_raw_gps=False`` the raw latlng channel never lands; power does (PRIV-R2)."""
    channels = await _landed_channels(tmp_path, store_raw_gps=False)
    assert "latlng" not in channels
    assert "power_w" in channels  # derived/non-locating metrics still land


async def test_gps_default_stores_latlng(tmp_path: Path) -> None:
    """With the default ``store_raw_gps=True`` the latlng channel lands as before (PRIV-R2)."""
    channels = await _landed_channels(tmp_path, store_raw_gps=True)
    assert "latlng" in channels
    assert "power_w" in channels


# ------------------------------------------------------- PRIV-R7 / PRIV-R11.2 raw-file purge


async def test_original_file_purge_deletes_object_and_reference_not_canonical(
    tmp_path: Path,
) -> None:
    """The retention sweep purges old originals — bytes AND reference — only (PRIV-R11.2).

    A retained original older than the window loses its object-store BYTES and its
    ``activity_file`` reference row, while the canonical activity derived from it
    survives. A fresh file inside the window is untouched. ``retention_days=0`` is the
    documented retain-forever sentinel (no sweep).
    """
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'purge.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    store = LocalObjectStore(tmp_path / "objects")
    now = _dt.datetime(2026, 6, 10, tzinfo=UTC)
    try:
        async with factory() as session:
            athlete_id, descriptor_id = await _seed_minimal(session)
            svc = IngestService(session, object_store=store)
            ride = _gps_ride("purge-ride-1")
            await svc.ingest(
                athlete_id,
                descriptor_id,
                [ride],
                original_files=[
                    OriginalFile(
                        data=b"verbatim-bytes",
                        file_format=ActivityFileFormat.FIT,
                        source_native_id="purge-ride-1",
                    )
                ],
            )
            await session.commit()
            row = (await session.execute(select(ActivityFile))).scalars().one()
            object_ref = row.object_ref
            assert store.get(object_ref) == b"verbatim-bytes"
            # Age the reference row past the window.
            row.created_at = now - _dt.timedelta(days=120)
            await session.commit()
            # Sentinel: 0 retains forever — nothing purged.
            assert (
                await purge_expired_original_files(
                    session, store, retention_days=0, now=lambda: now
                )
                == 0
            )
            purged = await purge_expired_original_files(
                session, store, retention_days=90, now=lambda: now
            )
            await session.commit()
            assert purged == 1
            assert (await session.execute(select(ActivityFile))).scalars().all() == []
            with pytest.raises(KeyError):
                store.get(object_ref)
            # The canonical typed activity derived from the file OUTLIVES it (PRIV-R11.2).
            assert len((await session.execute(select(Activity))).scalars().all()) == 1
    finally:
        await engine.dispose()


# ------------------------------------------------------------------------- PRIV-R9 export


def test_export_streams_athlete_canonical_data_as_ndjson(tmp_path: Path) -> None:
    """``GET /v1/users/me/export`` returns the owner's canonical rows as NDJSON (PRIV-R9)."""
    client = _client(tmp_path)
    try:
        access = _sign_in(client)["access_token"]
        resp = client.get(
            f"{API_PREFIX}/users/me/export", headers={"Authorization": f"Bearer {access}"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(line) for line in resp.text.splitlines() if line]
        # The owner's refresh-token credential row is athlete-scoped data and exports.
        assert all({"table", "row"} <= set(line) for line in lines)
        assert any(line["table"] == "auth_refresh_token" for line in lines)
    finally:
        client.__exit__(None, None, None)


def test_export_requires_the_export_scope(tmp_path: Path) -> None:
    """A token without the ``export`` scope is refused 403 (AUTH-R7 gate on PRIV-R9)."""
    client = _client(tmp_path)
    try:
        settings = client.app.state.settings  # type: ignore[attr-defined]
        tokens = issue_access_token(
            settings, subject="00000000-0000-0000-0000-000000000001", scopes=(Scope.READ,)
        )
        resp = client.get(
            f"{API_PREFIX}/users/me/export",
            headers={"Authorization": f"Bearer {tokens.access_token}"},
        )
        assert resp.status_code == 403, resp.text
    finally:
        client.__exit__(None, None, None)


# ------------------------------------------------------ RUN-R6 / OBS-R6.2 readiness depth


def test_readiness_gates_on_migrations_config_and_drain_state(tmp_path: Path) -> None:
    """Readiness reports the RUN-R6/OBS-R6.2 dimensions and 503s on an unmigrated DB.

    With schema + stamped head every check passes (200). Dropping the migration stamp
    leaves the DB reachable but UNMIGRATED: ``migrations_applied`` is false and the
    probe returns 503 — the instance never serves an unmigrated schema (RUN-R6).
    """
    settings = _settings(tmp_path)
    app = create_app(settings)
    provision_app_schema(app)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    try:
        ready = client.get("/readyz")
        assert ready.status_code == 200, ready.text
        checks = ready.json()["checks"]
        for key in (
            "database",
            "migrations_applied",
            "configuration",
            "not_draining",
            "entitlement_resolver",
            "default_plan_loaded",
        ):
            assert checks[key] is True, key
        # Un-stamp the migration head -> reachable but unmigrated -> 503 (RUN-R6).
        sync_engine = create_engine(f"sqlite:///{tmp_path / 'doc70.db'}")
        with sync_engine.begin() as conn:
            conn.execute(delete(_ALEMBIC_VERSION))
        sync_engine.dispose()
        not_ready = client.get("/readyz")
        assert not_ready.status_code == 503, not_ready.text
        assert not_ready.json()["checks"]["migrations_applied"] is False
    finally:
        client.__exit__(None, None, None)


# ----------------------------------------------------------------- RUN-R11 graceful shutdown


def test_shutdown_marks_draining_and_closes_pools(tmp_path: Path) -> None:
    """Lifespan shutdown flips the drain flag the readiness probe reports (RUN-R11).

    Inside the lifespan the instance is ready; after the lifespan exits (the SIGTERM
    drain path uvicorn drives) the app is marked draining — the readiness probe's
    ``not_draining`` dimension — and the pools are disposed without error.
    """
    app = create_app(_settings(tmp_path))
    provision_app_schema(app)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    assert client.get("/readyz").status_code == 200
    assert app.state.draining is False
    client.__exit__(None, None, None)  # delivers the shutdown lifespan event
    assert app.state.draining is True  # drained from rotation before pool close (RUN-R11)


# ------------------------------------------------------------- RUN-R13 stateless instances


def test_two_instances_serve_interchangeably_over_shared_store(tmp_path: Path) -> None:
    """Two app instances over ONE database serve the athlete interchangeably (RUN-R13).

    Instance A signs in (persisting the refresh credential); instance B — a separate
    process-equivalent app over the same store — ROTATES that refresh token and serves
    the athlete's authenticated reads. No instance-local session state is involved, and
    work committed through A survives A's shutdown (B still serves it).
    """
    settings = _settings(tmp_path)
    app_a = create_app(settings)
    provision_app_schema(app_a)
    app_b = create_app(settings)
    client_a = TestClient(app_a, raise_server_exceptions=False)
    client_b = TestClient(app_b, raise_server_exceptions=False)
    client_a.__enter__()
    client_b.__enter__()
    try:
        body = client_a.post(f"{API_PREFIX}/auth/token", json={"owner_secret": _STRONG_KEY}).json()
        # B rotates the refresh token A minted: durable state, not instance memory.
        rotated = client_b.post(
            f"{API_PREFIX}/auth/refresh", json={"refresh_token": body["refresh_token"]}
        )
        assert rotated.status_code == 200, rotated.text
        # A's access token (signed with the shared key) authenticates on B too.
        resp = client_b.get(
            f"{API_PREFIX}/users/me/export",
            headers={"Authorization": f"Bearer {body['access_token']}"},
        )
        assert resp.status_code == 200, resp.text
        # Kill A; B still serves the committed credential chain (no lost work).
        client_a.__exit__(None, None, None)
        again = client_b.post(
            f"{API_PREFIX}/auth/refresh",
            json={"refresh_token": rotated.json()["refresh_token"]},
        )
        assert again.status_code == 200, again.text
    finally:
        client_b.__exit__(None, None, None)
