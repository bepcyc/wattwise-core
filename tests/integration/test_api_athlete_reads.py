"""Integration tests for the athlete READ sub-resources (doc 60 §8.1).

Covers the read-scope, server-derived, cursor-paginated signature-history list
(``GET /v1/athlete/fitness-signature/history``) and the explicit change-sport action
(``POST /v1/athlete/change-sport``) added on top of the profile/signature surface.

Builds a minimal ASGI app that mounts ONLY the athlete router and overrides its
dependency seams (server-derived identity AUTH-R3, ``read``/``write`` scopes AUTH-R11,
the shared session, and the opaque-cursor HMAC signing key PAGE-R5) against a seeded
canonical store. Asserts that:

* ``GET /v1/athlete/fitness-signature/history`` lists the owner's effective-dated
  :class:`FitnessSignature` rows newest-first (GBO-R27), cursor-paginates correctly across
  pages with a stable, non-overlapping keyset (PAGE-R1/R7), clamps the limit (PAGE-R3),
  fails closed on a tampered cursor (``invalid-cursor``), returns ``data: []`` for an owner
  with no signatures (never a ``404``), requires the ``read`` scope (``403`` without it),
  and leaks no source/provider name (AUTH-R15);
* ``POST /v1/athlete/change-sport`` sets the current sport (reflected by a subsequent
  ``GET /v1/athlete``), rejects an unregistered sport ``422`` with
  ``errors[].code == "unknown_sport"`` leaving the seeded sport untouched (no partial
  mutation, API-R40), rejects a forged ``athlete_id`` body field ``422``
  (additionalProperties:false, SCHEMA-R4/AUTH-R3), and requires the ``write`` scope.

Runs on in-memory SQLite (the portable substrate, GBO-R8b); the reads here are single-
session and not concurrency-shaped, so the in-memory engine is sufficient.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.api.auth import Principal, Scope, authenticate
from wattwise_core.api.errors import ProblemError, install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import athlete as athlete_router
from wattwise_core.domain.enums import SignatureOrigin
from wattwise_core.persistence.models import Athlete, Base, FitnessSignature, Sport

pytestmark = pytest.mark.integration

UTC = _dt.UTC
_CURSOR_KEY = "test-cursor-signing-key"
# Three effective-dated cycling signatures, chronologically ascending FTP (newest last).
_HISTORY: tuple[tuple[_dt.date, float], ...] = (
    (_dt.date(2026, 1, 1), 230.0),
    (_dt.date(2026, 3, 1), 250.0),
    (_dt.date(2026, 5, 1), 265.0),
)


@dataclass
class Env:
    """The wired app + its client/session for one seeded scenario."""

    client: AsyncClient
    app: FastAPI
    session: AsyncSession
    athlete_id: str


def _insufficient_scope() -> None:
    """The unwired scope seam stand-in: deny (used to assert the 403 path)."""
    raise ProblemError("insufficient-scope")


@pytest_asyncio.fixture
async def seeded() -> AsyncIterator[Env]:
    """An app wired to a seeded store: one owner (cycling) with three FTP signatures."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        athlete_id = await _seed(session, with_signatures=True)
        app = _build_app(session, athlete_id, read_allowed=True, write_allowed=True)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield Env(client, app, session, athlete_id)
    await engine.dispose()


@pytest_asyncio.fixture
async def empty() -> AsyncIterator[Env]:
    """An app wired to a seeded owner with NO signatures (the empty-history case)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        athlete_id = await _seed(session, with_signatures=False)
        app = _build_app(session, athlete_id, read_allowed=True, write_allowed=True)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield Env(client, app, session, athlete_id)
    await engine.dispose()


def _build_app(
    session: AsyncSession, athlete_id: str, *, read_allowed: bool, write_allowed: bool
) -> FastAPI:
    """Mount the athlete router and override identity/scope/session/cursor-key seams."""
    app = FastAPI()
    app.state.rate_limiter = RateLimiter()  # the per-athlete read/write buckets (LIMIT-R1)
    install_error_handlers(app)
    app.include_router(athlete_router.router)
    read_seam = (lambda: None) if read_allowed else _insufficient_scope
    write_seam = (lambda: None) if write_allowed else _insufficient_scope
    app.dependency_overrides.update(
        {
            # The router attaches the per-subject RateLimit gate, which derives identity from
            # ``authenticate`` (AUTH-R18); bind it to the seeded owner so the bucket is keyed
            # server-side, mirroring the assembled app's wiring (LIMIT-R1/R6).
            authenticate: lambda: Principal(subject=athlete_id, scopes=frozenset(Scope)),
            athlete_router.require_read_scope: read_seam,
            athlete_router.require_write_scope: write_seam,
            athlete_router.current_athlete_id: lambda: athlete_id,
            athlete_router.current_session: lambda: session,
            athlete_router.cursor_signing_key: lambda: _CURSOR_KEY,
        }
    )
    return app


async def _seed(session: AsyncSession, *, with_signatures: bool) -> str:
    """Seed one owner (current sport cycling) and optionally three effective-dated FTPs."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    session.add(Sport(sport_code="running", display_name="Running", has_mechanical_power=False))
    athlete = Athlete(sex="male", reference_timezone="UTC", current_sport="cycling")
    session.add(athlete)
    await session.flush()
    aid = athlete.athlete_id
    if with_signatures:
        for effective_date, ftp in _HISTORY:
            session.add(
                FitnessSignature(
                    athlete_id=aid,
                    signature_type="cycling",
                    effective_date=effective_date,
                    ftp_w=ftp,
                    origin=SignatureOrigin.USER_ENTERED,
                )
            )
    await session.commit()
    return str(aid)


def _no_source_name(payload: object) -> None:
    text = repr(payload).lower()
    for banned in ("garmin", "strava", "intervals", "wahoo", "source_descriptor", "provider"):
        assert banned not in text


# --- GET /fitness-signature/history ----------------------------------------------


async def test_history_lists_signatures_newest_first(seeded: Env) -> None:
    """The history lists every effective-dated signature, newest effective date first (GBO-R27)."""
    resp = await seeded.client.get("/v1/athlete/fitness-signature/history")
    assert resp.status_code == 200
    body = resp.json()
    dates = [item["effective_date"] for item in body["data"]]
    assert dates == ["2026-05-01", "2026-03-01", "2026-01-01"]  # newest -> oldest
    assert body["data"][0]["ftp_w"] == pytest.approx(265.0)
    assert body["data"][0]["origin"] == "user_entered"
    assert body["page"]["has_more"] is False
    assert body["page"]["next_cursor"] is None
    _no_source_name(body)


async def test_history_empty_owner_returns_empty_list_not_404(empty: Env) -> None:
    """An owner with no signatures yet returns data: [] (typed empty, never a 404)."""
    resp = await empty.client.get("/v1/athlete/fitness-signature/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["page"]["has_more"] is False
    assert body["page"]["next_cursor"] is None


async def test_history_paginates_across_pages_without_overlap(seeded: Env) -> None:
    """A small limit pages forward via the cursor with no gap or overlap (PAGE-R1/R7)."""
    first = await seeded.client.get("/v1/athlete/fitness-signature/history", params={"limit": 2})
    assert first.status_code == 200
    page1 = first.json()
    assert len(page1["data"]) == 2
    assert page1["page"]["has_more"] is True
    cursor = page1["page"]["next_cursor"]
    assert cursor is not None

    second = await seeded.client.get(
        "/v1/athlete/fitness-signature/history", params={"limit": 2, "cursor": cursor}
    )
    assert second.status_code == 200
    page2 = second.json()
    assert len(page2["data"]) == 1
    assert page2["page"]["has_more"] is False
    assert page2["page"]["next_cursor"] is None

    seen = [i["effective_date"] for i in page1["data"]] + [
        i["effective_date"] for i in page2["data"]
    ]
    assert seen == ["2026-05-01", "2026-03-01", "2026-01-01"]  # contiguous, no overlap
    assert len(set(seen)) == 3


async def test_history_rejects_nonpositive_limit(seeded: Env) -> None:
    """A limit < 1 is REJECTED 422 validation-error, never coerced to a default (PAGE-R3)."""
    resp = await seeded.client.get("/v1/athlete/fitness-signature/history", params={"limit": 0})
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


async def test_history_tampered_cursor_is_invalid_cursor(seeded: Env) -> None:
    """A forged/tampered cursor fails closed with the invalid-cursor problem (PAGE-R5)."""
    resp = await seeded.client.get(
        "/v1/athlete/fitness-signature/history", params={"cursor": "not-a-valid-cursor"}
    )
    assert resp.status_code == 400
    assert resp.json()["type"].endswith("/invalid-cursor")


async def test_history_without_read_scope_is_403(seeded: Env) -> None:
    """The history requires the read scope; without it the call is 403 (AUTH-R7/R11)."""
    no_read = _build_app(seeded.session, seeded.athlete_id, read_allowed=False, write_allowed=True)
    async with AsyncClient(transport=ASGITransport(app=no_read), base_url="http://t") as client:
        resp = await client.get("/v1/athlete/fitness-signature/history")
    assert resp.status_code == 403
    assert resp.json()["type"].endswith("/insufficient-scope")


# --- POST /change-sport ----------------------------------------------------------


async def test_change_sport_sets_current_sport(seeded: Env) -> None:
    """POST /change-sport sets the current sport and a subsequent GET reflects it (API-R40)."""
    resp = await seeded.client.post("/v1/athlete/change-sport", json={"sport": "running"})
    assert resp.status_code == 200
    assert resp.json()["current_sport"] == "running"
    again = await seeded.client.get("/v1/athlete")
    assert again.json()["current_sport"] == "running"


async def test_change_sport_unknown_is_422_unknown_sport(seeded: Env) -> None:
    """An unregistered sport is rejected 422 unknown_sport, leaving the seeded sport untouched."""
    resp = await seeded.client.post(
        "/v1/athlete/change-sport", json={"sport": "underwater_basketweaving"}
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["type"].endswith("/validation-error")
    assert any(e.get("code") == "unknown_sport" for e in body.get("errors", []))
    # no partial mutation: the seeded sport is intact
    assert (await seeded.client.get("/v1/athlete")).json()["current_sport"] == "cycling"


async def test_change_sport_rejects_forged_identity_field(seeded: Env) -> None:
    """A forged athlete_id body field is a 422 (additionalProperties:false, SCHEMA-R4/AUTH-R3)."""
    resp = await seeded.client.post(
        "/v1/athlete/change-sport",
        json={"sport": "running", "athlete_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert resp.status_code == 422
    # the rejected request mutated nothing
    assert (await seeded.client.get("/v1/athlete")).json()["current_sport"] == "cycling"


async def test_change_sport_missing_sport_is_422(seeded: Env) -> None:
    """The required sport field is enforced — an empty body is a 422 validation error."""
    resp = await seeded.client.post("/v1/athlete/change-sport", json={})
    assert resp.status_code == 422


async def test_change_sport_without_write_scope_is_403(seeded: Env) -> None:
    """A change-sport mutation with only the read scope is 403 insufficient-scope (AUTH-R7/R11)."""
    no_write = _build_app(seeded.session, seeded.athlete_id, read_allowed=True, write_allowed=False)
    async with AsyncClient(transport=ASGITransport(app=no_write), base_url="http://t") as client:
        resp = await client.post("/v1/athlete/change-sport", json={"sport": "running"})
    assert resp.status_code == 403
    assert resp.json()["type"].endswith("/insufficient-scope")
    # the read surface still works without write
    async with AsyncClient(transport=ASGITransport(app=no_write), base_url="http://t") as client:
        assert (await client.get("/v1/athlete")).status_code == 200
