"""Integration tests for the users router (``/v1/users/me``, doc 60 §8 / retention §11).

Builds a minimal ASGI app that mounts the users router and overrides the shared dependency
seams (server-derived identity AUTH-R3, the request session, settings, and rate limiter) plus
the router-local async-erasure recorder, against a seeded canonical store with exactly one
owner. Asserts the load-bearing invariants of the ``/v1/users/me`` slice:

* ``GET`` reads the account derived from the canonical :class:`NotificationRoute` rows — an
  owner with no email captured reads ``email=null``, ``verified=false`` (an honest empty
  account, never an error);
* ``PATCH`` captures the digest email by binding it to the ``email`` channel route, persists
  it (a fresh ``GET`` reflects it — it is stored, not echoed), and gates fail-closed: a freshly
  captured address is ``verified=false`` (GBO-R49); changing an already-verified address RESETS
  ``verified`` to ``false``; re-capturing the SAME address leaves a verified route untouched;
* a malformed email is a ``422 validation-error`` (the ``pattern`` constraint) and a forged
  ``verified`` body property is a ``422`` (``additionalProperties:false``, SCHEMA-R4) — neither
  can spoof the server-controlled verified state;
* ``DELETE`` records the async erasure request through the injected recorder, disables the
  delivery channels (no notification leaks while pending), and does NOT hard-delete inline (the
  owner row + canonical data survive) — returning ``status=pending_deletion`` (retention §11);
* scope is enforced — a ``write`` mutation with only the ``read`` scope is ``403`` — and no
  account response leaks a full address in the route list (only a masked hint) or a model/tier.

Tier: T-INTEGRATION (offline; in-process ASGI over a fresh in-memory canonical schema; the
auth/db/seam dependencies are overridden in-test). Runs on in-memory SQLite (GBO-R8b).
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.api.auth import Principal, Scope, authenticate
from wattwise_core.api.deps import get_db, get_rate_limiter, get_settings
from wattwise_core.api.errors import install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import users as users_router
from wattwise_core.config import load_settings
from wattwise_core.domain.enums import DeliveryChannel
from wattwise_core.persistence.models import Athlete, Base, NotificationRoute

pytestmark = pytest.mark.integration

#: Model/tier/catalog tokens that MUST NOT appear on any account response (API-R38).
_FORBIDDEN_MODEL_FIELDS = ("model_tier", "reasoning", "model_name", "model_catalog", "frontier")

#: The full scope grant the in-test auth seam resolves (server-derived owner, AUTH-R3).
_FULL_SCOPES = frozenset({Scope.READ, Scope.WRITE, Scope.AGENT})
_READ_ONLY_SCOPES = frozenset({Scope.READ})


@dataclass
class _Recorder:
    """An in-test async-erasure recorder capturing the (athlete_id, requested_at) calls."""

    calls: list[tuple[str, _dt.datetime]] = field(default_factory=list)

    async def __call__(self, athlete_id: str, requested_at: _dt.datetime) -> None:
        self.calls.append((athlete_id, requested_at))


@dataclass
class Env:
    """The wired app + its client/session/recorder for one seeded scenario."""

    client: AsyncClient
    app: FastAPI
    session: AsyncSession
    athlete_id: str
    recorder: _Recorder


@pytest_asyncio.fixture
async def seeded() -> AsyncIterator[Env]:
    """An app wired to a seeded canonical store with exactly one owner (no email captured)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        athlete = Athlete(sex="male", reference_timezone="UTC")
        session.add(athlete)
        await session.flush()
        athlete_id = str(athlete.athlete_id)
        await session.commit()
        recorder = _Recorder()
        app = _build_app(session, athlete_id, recorder, scopes=_FULL_SCOPES)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield Env(client, app, session, athlete_id, recorder)
    await engine.dispose()


def _build_app(
    session: AsyncSession,
    athlete_id: str,
    recorder: _Recorder,
    *,
    scopes: frozenset[Scope],
) -> FastAPI:
    """Mount the users router and override the shared identity/session/seam dependencies."""
    settings = load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="integration-test-key",
    )
    app = FastAPI()
    app.state.settings = settings
    app.state.rate_limiter = RateLimiter()
    install_error_handlers(app)
    app.include_router(users_router.router)

    async def _session() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[authenticate] = lambda: Principal(subject=athlete_id, scopes=scopes)
    app.dependency_overrides[get_db] = _session
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rate_limiter] = lambda: app.state.rate_limiter
    app.dependency_overrides[users_router.deletion_requester] = lambda: recorder
    return app


def _auth() -> dict[str, str]:
    """A bearer header so the route's security extractor is satisfied (value unused in-test)."""
    return {"Authorization": "Bearer test"}


async def _email_route(session: AsyncSession, athlete_id: str) -> NotificationRoute | None:
    """Read the owner's persisted ``email`` notification route (the captured-email anchor)."""
    stmt = select(NotificationRoute).where(
        NotificationRoute.athlete_id == uuid.UUID(athlete_id),
        NotificationRoute.channel == DeliveryChannel.EMAIL,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


# --- GET /v1/users/me: the readable account (doc 60 §8) --------------------------


async def test_get_account_empty_when_no_email_captured(seeded: Env) -> None:
    """An owner with no email captured reads an honest null/empty account (doc 60 §8)."""
    resp = await seeded.client.get("/v1/users/me", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] is None
    assert body["verified"] is False
    assert body["notification_routes"] == []


# --- PATCH /v1/users/me: capture/verify email (gates the digest channel) ---------


async def test_capture_email_persists_unverified(seeded: Env) -> None:
    """PATCH captures the email on the email channel and persists it UNVERIFIED (GBO-R49)."""
    resp = await seeded.client.patch(
        "/v1/users/me", json={"email": "owner@example.com"}, headers=_auth()
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "owner@example.com"
    assert body["verified"] is False  # fail-closed: a fresh address never gates the channel open
    # persisted, not merely echoed: a fresh GET reflects it
    again = await seeded.client.get("/v1/users/me", headers=_auth())
    assert again.json()["email"] == "owner@example.com"
    # backed by a real canonical email NotificationRoute (API-R32)
    route = await _email_route(seeded.session, seeded.athlete_id)
    assert route is not None and route.address_ref == "owner@example.com"
    assert route.verified is False


async def test_changing_verified_email_resets_verified(seeded: Env) -> None:
    """Changing an already-verified address resets verified to false (fail-closed, GBO-R49)."""
    await seeded.client.patch("/v1/users/me", json={"email": "first@example.com"}, headers=_auth())
    # simulate the out-of-band verification flow marking the route verified
    route = await _email_route(seeded.session, seeded.athlete_id)
    assert route is not None
    route.verified = True
    await seeded.session.commit()
    assert (await seeded.client.get("/v1/users/me", headers=_auth())).json()["verified"] is True
    # changing the address must drop verification
    changed = await seeded.client.patch(
        "/v1/users/me", json={"email": "second@example.com"}, headers=_auth()
    )
    assert changed.status_code == 200
    assert changed.json()["email"] == "second@example.com"
    assert changed.json()["verified"] is False


async def test_recapturing_same_email_keeps_verified(seeded: Env) -> None:
    """Re-capturing the SAME address leaves an already-verified route untouched (GBO-R49)."""
    await seeded.client.patch("/v1/users/me", json={"email": "owner@example.com"}, headers=_auth())
    route = await _email_route(seeded.session, seeded.athlete_id)
    assert route is not None
    route.verified = True
    await seeded.session.commit()
    again = await seeded.client.patch(
        "/v1/users/me", json={"email": "owner@example.com"}, headers=_auth()
    )
    assert again.status_code == 200
    assert again.json()["verified"] is True  # unchanged address keeps the verified state


async def test_malformed_email_is_422(seeded: Env) -> None:
    """A malformed email is rejected 422 by the pattern constraint, never a silent accept."""
    resp = await seeded.client.patch(
        "/v1/users/me", json={"email": "not-an-email"}, headers=_auth()
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


async def test_forged_verified_property_is_422(seeded: Env) -> None:
    """A forged ``verified`` body property is rejected 422 — server controls it (SCHEMA-R4)."""
    resp = await seeded.client.patch(
        "/v1/users/me",
        json={"email": "owner@example.com", "verified": True},
        headers=_auth(),
    )
    assert resp.status_code == 422


async def test_account_route_list_masks_the_full_address(seeded: Env) -> None:
    """The route list returns only a masked hint — the full address never leaks there (ERR-R5)."""
    await seeded.client.patch("/v1/users/me", json={"email": "owner@example.com"}, headers=_auth())
    body = (await seeded.client.get("/v1/users/me", headers=_auth())).json()
    email_routes = [r for r in body["notification_routes"] if r["channel"] == "email"]
    assert email_routes, "the captured email must surface as a route"
    hint = email_routes[0]["address_hint"]
    assert hint is not None and "owner@example.com" not in hint
    assert hint.endswith("@example.com")  # the domain hint is preserved, the local part is masked


async def test_account_carries_no_model_machinery(seeded: Env) -> None:
    """No model/tier/catalog control appears on the account surface (API-R38)."""
    await seeded.client.patch("/v1/users/me", json={"email": "owner@example.com"}, headers=_auth())
    flat = json.dumps((await seeded.client.get("/v1/users/me", headers=_auth())).json())
    for token in _FORBIDDEN_MODEL_FIELDS:
        assert token not in flat, f"model-selection token {token!r} leaked (API-R38)"


# --- DELETE /v1/users/me: async account-deletion request (retention §11) ---------


async def test_delete_records_async_request_without_hard_delete(seeded: Env) -> None:
    """DELETE records the async erasure request and does NOT hard-delete inline (retention §11)."""
    # capture an email first so there is a delivery channel to disable
    await seeded.client.patch("/v1/users/me", json={"email": "owner@example.com"}, headers=_auth())
    resp = await seeded.client.delete("/v1/users/me", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending_deletion"
    assert body["requested_at"]  # server-stamped instant
    # the async recorder was invoked for the SERVER-DERIVED owner id (PRIV-R8)
    assert len(seeded.recorder.calls) == 1
    assert seeded.recorder.calls[0][0] == seeded.athlete_id
    # NOT hard-deleted inline: the owner row + the email route survive
    owner = await seeded.session.get(Athlete, uuid.UUID(seeded.athlete_id))
    assert owner is not None, "the owner row must survive (deletion is async, not inline)"
    route = await _email_route(seeded.session, seeded.athlete_id)
    assert route is not None, "the route row survives erasure-pending (not hard-deleted)"
    # but the channel is disabled so no notification leaks while erasure is pending
    assert route.enabled is False


async def test_delete_fails_closed_when_recorder_unwired(seeded: Env) -> None:
    """With the erasure recorder unwired, DELETE fails closed — it never silently no-ops."""
    # rebuild the app WITHOUT overriding the deletion-requester seam (its fail-closed default)
    app = _build_app(seeded.session, seeded.athlete_id, seeded.recorder, scopes=_FULL_SCOPES)
    del app.dependency_overrides[users_router.deletion_requester]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.delete("/v1/users/me", headers=_auth())
    assert resp.status_code == 500
    assert resp.json()["type"].endswith("/internal-error")
    # fail-closed: nothing recorded, channels untouched (the transaction rolled back)
    assert seeded.recorder.calls == []


# --- AUTH-R11: write scope enforcement -------------------------------------------


async def test_writes_without_write_scope_are_403(seeded: Env) -> None:
    """A PATCH/DELETE with only the read scope is 403 insufficient-scope; reads still work."""
    read_only = _build_app(
        seeded.session, seeded.athlete_id, seeded.recorder, scopes=_READ_ONLY_SCOPES
    )
    async with AsyncClient(transport=ASGITransport(app=read_only), base_url="http://t") as client:
        patch = await client.patch(
            "/v1/users/me", json={"email": "owner@example.com"}, headers=_auth()
        )
        delete = await client.delete("/v1/users/me", headers=_auth())
        read_ok = await client.get("/v1/users/me", headers=_auth())
    assert patch.status_code == 403
    assert patch.json()["type"].endswith("/insufficient-scope")
    assert delete.status_code == 403
    assert read_ok.status_code == 200
