"""Integration tests for the athlete-profile router (doc 60 §8.1 / API-R40).

Builds a minimal ASGI app that mounts the athlete router (and the performance router for
the FTP-feeds-analytics proof) and overrides their dependency seams (server-derived
identity AUTH-R3, ``read``/``write`` scopes AUTH-R11, the shared session, the analytics
service) against a seeded canonical store. Asserts that:

* ``GET /v1/athlete`` reflects sex / reference timezone / current sport / the effective
  FTP signature; ``PUT`` sets timezone + sport and the read reflects it (API-R40);
* the critical ``PUT /v1/athlete/signature`` writes a ``user_entered`` FTP row that the
  power analytics then ground on — the load-bearing proof that BEFORE the write the
  Coggan TSS is unavailable and AFTER it the chart computes a non-trivial TSS (~100),
  mirroring the direct-``FitnessSignature`` seeding of ``test_agent_engine`` but driving it
  through the API (this is what makes the analytics the agent grounds on meaningful);
* scope is enforced — a ``write`` mutation with only the ``read`` scope is ``403`` — and an
  unknown sport (on either the profile update or the signature) is a ``422`` with
  ``errors[].code == "unknown_sport"`` (API-R40), never a silent accept;
* no response field carries a source/provider name (AUTH-R15).

Runs on in-memory SQLite (the portable substrate, GBO-R8b).
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

from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.api.auth import Principal, Scope, authenticate
from wattwise_core.api.errors import ProblemError, install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import athlete as athlete_router
from wattwise_core.api.routers import performance as perf_router
from wattwise_core.domain.enums import SampleBasis, StreamChannelName, StreamSetKind
from wattwise_core.persistence.models import (
    Activity,
    ActivityStreamSet,
    Athlete,
    Base,
    Sport,
    StreamChannel,
)

pytestmark = pytest.mark.integration

UTC = _dt.UTC
_START = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)


@dataclass
class Env:
    """The wired app + its client/session for one seeded scenario."""

    client: AsyncClient
    app: FastAPI
    session: AsyncSession
    athlete_id: str


def _insufficient_scope() -> None:
    """The unwired ``write`` seam stand-in: deny (used to assert the 403 path)."""
    raise ProblemError("insufficient-scope")


@pytest_asyncio.fixture
async def seeded() -> AsyncIterator[Env]:
    """An app wired to a seeded canonical store (one owner, cycling sport, a 250W ride).

    The ride is seeded WITHOUT a fitness signature, so the FTP-feeds-analytics proof can
    set it through the API and observe the analytics turn on.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        athlete_id = await _seed(session)
        app = _build_app(session, athlete_id, write_allowed=True)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield Env(client, app, session, athlete_id)
    await engine.dispose()


def _build_app(session: AsyncSession, athlete_id: str, *, write_allowed: bool) -> FastAPI:
    """Mount the athlete + performance routers and override the identity/scope/session seams."""
    app = FastAPI()
    app.state.rate_limiter = RateLimiter()  # the per-athlete read/write buckets (LIMIT-R1)
    install_error_handlers(app)
    app.include_router(athlete_router.router)
    app.include_router(perf_router.router)
    write_seam = (lambda: None) if write_allowed else _insufficient_scope
    app.dependency_overrides.update(
        {
            # The routers attach the per-subject RateLimit gate, which derives identity from
            # ``authenticate`` (AUTH-R18); bind it to the seeded owner so the bucket is keyed
            # server-side, mirroring the assembled app's wiring (LIMIT-R1/R6).
            authenticate: lambda: Principal(subject=athlete_id, scopes=frozenset(Scope)),
            athlete_router.require_read_scope: lambda: None,
            athlete_router.require_write_scope: write_seam,
            athlete_router.current_athlete_id: lambda: athlete_id,
            athlete_router.current_session: lambda: session,
            perf_router.require_read_scope: lambda: None,
            perf_router.current_athlete_id: lambda: athlete_id,
            perf_router.analytics_service: lambda: AnalyticsService(session),
        }
    )
    return app


async def _seed(session: AsyncSession) -> str:
    """Seed one owner (current sport cycling) + a 1-hour constant-250W ride, NO signature."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    session.add(Sport(sport_code="running", display_name="Running", has_mechanical_power=False))
    athlete = Athlete(sex="male", reference_timezone="UTC", current_sport="cycling")
    session.add(athlete)
    await session.flush()
    aid = athlete.athlete_id
    activity = Activity(
        athlete_id=aid,
        start_time=_START,
        sport="cycling",
        elapsed_time_s=3600,
        moving_time_s=3600,
        avg_power_w=250.0,
        max_power_w=400.0,
        has_power=True,
        has_hr=False,
        has_gps=False,
    )
    session.add(activity)
    await session.flush()
    stream_set = ActivityStreamSet(
        activity_id=activity.activity_id,
        sample_basis=SampleBasis.TIME,
        sample_rate_hz=1.0,
        sample_count=3600,
        t0=_START,
    )
    session.add(stream_set)
    await session.flush()
    session.add(
        StreamChannel(
            stream_set_id=stream_set.stream_set_id,
            set_kind=StreamSetKind.ACTIVITY,
            channel=StreamChannelName.POWER_W,
            sample_basis=SampleBasis.TIME,
            values=[250.0] * 3600,
            coverage={},
        )
    )
    await session.commit()
    return str(aid)


def _range() -> dict[str, str]:
    return {"from": "2026-06-01", "to": "2026-06-07"}


def _no_source_name(payload: object) -> None:
    text = repr(payload).lower()
    for banned in ("garmin", "strava", "intervals", "wahoo", "source_descriptor", "provider"):
        assert banned not in text


# --- §8.1 GET profile ------------------------------------------------------------


async def test_get_profile_reflects_seeded_owner(seeded: Env) -> None:
    """GET /v1/athlete returns the seeded sex / timezone / current sport, no signature yet."""
    resp = await seeded.client.get("/v1/athlete")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sex"] == "male"
    assert body["reference_timezone"] == "UTC"
    assert body["current_sport"] == "cycling"
    assert body["fitness_signature"] is None
    _no_source_name(body)


# --- §8.1 PUT profile (timezone + change-sport, API-R40) -------------------------


async def test_put_profile_sets_timezone_and_sport(seeded: Env) -> None:
    """PUT /v1/athlete sets timezone + current sport and a subsequent GET reflects both."""
    resp = await seeded.client.put(
        "/v1/athlete", json={"reference_timezone": "Europe/Berlin", "current_sport": "running"}
    )
    assert resp.status_code == 200
    assert resp.json()["reference_timezone"] == "Europe/Berlin"
    assert resp.json()["current_sport"] == "running"
    again = await seeded.client.get("/v1/athlete")
    assert again.json()["reference_timezone"] == "Europe/Berlin"
    assert again.json()["current_sport"] == "running"


async def test_put_profile_unknown_sport_is_422_unknown_sport(seeded: Env) -> None:
    """An unregistered current_sport is rejected 422 with errors[].code unknown_sport (API-R40)."""
    resp = await seeded.client.put(
        "/v1/athlete", json={"current_sport": "underwater_basketweaving"}
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["type"].endswith("/validation-error")
    assert any(e.get("code") == "unknown_sport" for e in body.get("errors", []))
    # the rejected write left the seeded sport untouched (no partial mutation)
    assert (await seeded.client.get("/v1/athlete")).json()["current_sport"] == "cycling"


async def test_put_profile_rejects_forged_identity_field(seeded: Env) -> None:
    """A forged athlete_id body field is a 422 (additionalProperties:false, SCHEMA-R4/AUTH-R3)."""
    resp = await seeded.client.put(
        "/v1/athlete",
        json={"reference_timezone": "UTC", "athlete_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert resp.status_code == 422


# --- §8.1 PUT signature — the critical FTP write + the analytics proof ------------


async def test_set_signature_then_get_reflects_it(seeded: Env) -> None:
    """PUT /v1/athlete/signature writes a user_entered FTP that the profile then surfaces."""
    resp = await seeded.client.put(
        "/v1/athlete/signature", json={"ftp_w": 250.0, "effective_date": "2026-01-01"}
    )
    assert resp.status_code == 200
    sig = resp.json()["fitness_signature"]
    assert sig is not None
    assert sig["ftp_w"] == pytest.approx(250.0)
    assert sig["signature_type"] == "cycling"  # defaulted to the current sport
    assert sig["origin"] == "user_entered"  # stamped server-side, never client-supplied
    # a fresh GET still reflects it (persisted, not just echoed)
    fresh = (await seeded.client.get("/v1/athlete")).json()
    assert fresh["fitness_signature"]["ftp_w"] == pytest.approx(250.0)


async def test_set_ftp_signature_feeds_power_analytics(seeded: Env) -> None:
    """THE proof: with no FTP the Coggan TSS is unavailable; after the API sets it, TSS ~100.

    A 1-hour ride held at exactly the FTP (250 W avg, 250 W FTP) is IF=1.0 → TSS≈100. Before
    any signature exists the power-load chart cannot resolve a threshold, so the per-activity
    TSS is a typed ``null`` (ANL-R3, never a fabricated 0). After the owner sets the FTP
    through ``PUT /v1/athlete/signature``, the SAME chart computes a non-trivial TSS — i.e.
    the API write is what makes the power analytics (the numbers the coaching agent grounds
    on) meaningful.
    """
    before = await seeded.client.get("/v1/performance/coggan", params=_range())
    assert before.status_code == 200
    pre_points = before.json()["items"]
    assert pre_points, "the seeded ride should appear as a Coggan point"
    assert pre_points[0]["values"]["tss"] is None  # no FTP yet -> typed-unavailable, not 0

    setting = await seeded.client.put(
        "/v1/athlete/signature", json={"ftp_w": 250.0, "effective_date": "2026-01-01"}
    )
    assert setting.status_code == 200

    after = await seeded.client.get("/v1/performance/coggan", params=_range())
    assert after.status_code == 200
    post = after.json()["items"][0]
    assert post["values"]["tss"] == pytest.approx(100.0, abs=1.0)
    assert post["values"]["intensity_factor"] == pytest.approx(1.0, abs=0.02)
    _no_source_name(after.json())


async def test_set_signature_is_idempotent_on_same_effective_date(seeded: Env) -> None:
    """Re-PUT for the SAME effective date updates the row in place (no uniqueness violation)."""
    first = await seeded.client.put(
        "/v1/athlete/signature", json={"ftp_w": 200.0, "effective_date": "2026-01-01"}
    )
    assert first.status_code == 200
    second = await seeded.client.put(
        "/v1/athlete/signature", json={"ftp_w": 275.0, "effective_date": "2026-01-01"}
    )
    assert second.status_code == 200
    assert second.json()["fitness_signature"]["ftp_w"] == pytest.approx(275.0)


async def test_set_signature_unknown_sport_is_422(seeded: Env) -> None:
    """An explicit unregistered signature_type is rejected 422 unknown_sport (API-R40)."""
    resp = await seeded.client.put(
        "/v1/athlete/signature", json={"ftp_w": 250.0, "signature_type": "quidditch"}
    )
    assert resp.status_code == 422
    assert any(e.get("code") == "unknown_sport" for e in resp.json().get("errors", []))


async def test_set_signature_rejects_nonpositive_ftp(seeded: Env) -> None:
    """A non-positive FTP fails type validation 422 (never persisted)."""
    resp = await seeded.client.put("/v1/athlete/signature", json={"ftp_w": 0})
    assert resp.status_code == 422


# --- AUTH-R11: write scope enforcement -------------------------------------------


async def test_profile_update_without_write_scope_is_403(seeded: Env) -> None:
    """A PUT mutation with only the read scope is 403 insufficient-scope (AUTH-R7/R11)."""
    no_write = _build_app(seeded.session, seeded.athlete_id, write_allowed=False)
    async with AsyncClient(transport=ASGITransport(app=no_write), base_url="http://t") as client:
        profile = await client.put("/v1/athlete", json={"reference_timezone": "UTC"})
        signature = await client.put("/v1/athlete/signature", json={"ftp_w": 250.0})
    assert profile.status_code == 403
    assert profile.json()["type"].endswith("/insufficient-scope")
    assert signature.status_code == 403
    # the read surface still works without write
    async with AsyncClient(transport=ASGITransport(app=no_write), base_url="http://t") as client:
        assert (await client.get("/v1/athlete")).status_code == 200
