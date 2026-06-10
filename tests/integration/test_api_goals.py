"""Integration tests for the Goals router (doc 60 §8.13, API-R35 / API-R32 / API-R51).

Drives the five ``/v1/goals`` endpoints end to end over the goals router with its dependency
seams overridden against a REAL multi-connection pool (file-SQLite + WAL + busy_timeout, skill
§7 — NEVER ``:memory:``/StaticPool, which is a single connection that hides the athlete-scoping
and uniqueness races a CRUD surface must withstand). Each test asserts the boundary contract the
router owns per the spec:

- **API-R35** ``GET /v1/goals`` paginated + typed-filtered (status/sport/from/to) + typed-sorted
  (``target_date``/``created_at`` allow-list); ``POST`` → ``201`` + ``Location``; ``GET {id}``;
  ``PATCH {id}``; ``DELETE {id}`` → ``204``.
- **GBO-R39** ``DELETE`` is a soft close: it sets a TERMINAL ``status`` (``abandoned``), never a
  hard row delete, so goal history stays auditable (reconciling the ``204`` of API-R35 with the
  "never delete it" of GBO-R39).
- **GBO-R38** ``goal.sport`` is validated against the runtime sport registry; an unregistered code
  → ``422 unknown_sport`` BEFORE any write.
- **AUTH-R3 / SCHEMA-R4** identity is server-derived; a forged ``athlete_id`` body field → ``422``.
- **API-R51** an unknown/foreign ``goal_id`` → ``404 not-found`` on every ``{id}`` verb.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.api.errors import install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import goals as goals_router
from wattwise_core.domain.enums import GoalStatus, GoalType
from wattwise_core.persistence.models import Athlete, Base, Goal, Sport

pytestmark = pytest.mark.integration

UTC = _dt.UTC

#: The cursor/signing key the seam binds; deterministic for a test.
_CURSOR_KEY = "test-cursor-key-0123456789abcdef"


def _enable_sqlite_wal(dbapi_conn: Any, _record: Any) -> None:
    """WAL + long busy_timeout per SQLite connection so the real pool serialises writers."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


@dataclass
class _Env:
    """The wired goals app + its client/session-factory + the seeded owner id."""

    client: AsyncClient
    app: FastAPI
    factory: async_sessionmaker[AsyncSession]
    athlete_id: str
    limiter: RateLimiter


def _build_app(
    factory: async_sessionmaker[AsyncSession], athlete_id: str, limiter: RateLimiter
) -> FastAPI:
    """Mount the goals router and override its identity/scope/session/cursor/limiter seams.

    A per-request session is yielded from the REAL pooled factory (not one shared session) so
    every request runs on its own connection — the multi-connection arrangement the pool exists
    for. The read/write scope gates are satisfied; identity is the server-derived seam value.
    """

    async def _session() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session
            await session.commit()

    app = FastAPI()
    install_error_handlers(app)
    app.include_router(goals_router.router)
    app.dependency_overrides.update(
        {
            goals_router.require_read_scope: lambda: None,
            goals_router.require_write_scope: lambda: None,
            goals_router.current_athlete_id: lambda: athlete_id,
            goals_router.current_session: _session,
            goals_router.cursor_signing_key: lambda: _CURSOR_KEY,
            goals_router.rate_limiter: lambda: limiter,
        }
    )
    return app


async def _seed_owner(session: AsyncSession) -> str:
    """Seed the sport registry + one owner athlete; return the owner id."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    session.add(Sport(sport_code="running", display_name="Running", has_mechanical_power=False))
    athlete = Athlete(sex="male", reference_timezone="UTC", current_sport="cycling")
    session.add(athlete)
    await session.flush()
    aid = str(athlete.athlete_id)
    await session.commit()
    return aid


@pytest_asyncio.fixture
async def env(tmp_path: Path) -> AsyncIterator[_Env]:
    """A goals app over a REAL file-SQLite pool (WAL), seeded with the owner + sport registry."""
    dsn = f"sqlite+aiosqlite:///{tmp_path}/goals.sqlite"
    engine = create_async_engine(dsn)
    event.listen(engine.sync_engine, "connect", _enable_sqlite_wal)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        athlete_id = await _seed_owner(session)
    limiter = RateLimiter()
    app = _build_app(factory, athlete_id, limiter)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        yield _Env(client, app, factory, athlete_id, limiter)
    await engine.dispose()


def _goal_body(**over: object) -> dict[str, object]:
    """A valid ``GoalCreateRequest`` body (no caller-identity field, AUTH-R3)."""
    body: dict[str, object] = {
        "title": "Sub-9 gran fondo",
        "goal_type": "event",
        "sport": "cycling",
        "target_event": "Maratona dles Dolomites",
        "target_date": "2026-07-05",
        "status": "active",
    }
    body.update(over)
    return body


# --- POST /v1/goals -------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_goal_returns_201_with_location_and_body(env: _Env) -> None:
    """POST creates a backing Goal and returns 201 + a Location pointing at it (API-R35)."""
    resp = await env.client.post("/v1/goals", json=_goal_body(), headers={})
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "Sub-9 gran fondo"
    assert body["goal_type"] == "event"
    assert body["status"] == "active"
    assert resp.headers["location"] == f"/v1/goals/{body['goal_id']}"
    # The row really landed in the canonical store under the server-derived owner.
    async with env.factory() as session:
        row = await session.get(Goal, uuid.UUID(body["goal_id"]))
        assert row is not None
        assert str(row.athlete_id) == env.athlete_id  # AUTH-R3 server-derived, not a client value


@pytest.mark.asyncio
async def test_post_goal_metric_target_persists_value(env: _Env) -> None:
    """A target_metric goal persists its canonical-unit target_value (GBO-R38)."""
    resp = await env.client.post(
        "/v1/goals",
        json=_goal_body(
            goal_type="target_metric",
            title="Raise FTP",
            target_event=None,
            target_metric="ftp_w",
            target_value=300.0,
        ),
        headers={},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["target_metric"] == "ftp_w"
    assert body["target_value"] == 300.0


@pytest.mark.asyncio
async def test_post_goal_forged_athlete_id_is_422(env: _Env) -> None:
    """A forged caller-identity body field is rejected (AUTH-R3 / SCHEMA-R4)."""
    resp = await env.client.post("/v1/goals", json=_goal_body(athlete_id="attacker"), headers={})
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


@pytest.mark.asyncio
async def test_post_goal_unknown_sport_is_422_before_write(env: _Env) -> None:
    """An unregistered sport code is rejected 422 unknown_sport, no row written (GBO-R38)."""
    resp = await env.client.post("/v1/goals", json=_goal_body(sport="quidditch"), headers={})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["code"] == "unknown_sport"
    async with env.factory() as session:
        rows = (await session.execute(select(Goal))).scalars().all()
        assert rows == []  # no partial mutation


# --- GET /v1/goals/{id}, PATCH, DELETE ------------------------------------------


async def _create(env: _Env, **over: object) -> dict[str, Any]:
    resp = await env.client.post("/v1/goals", json=_goal_body(**over), headers={})
    assert resp.status_code == 201
    return resp.json()


@pytest.mark.asyncio
async def test_get_one_goal_returns_it(env: _Env) -> None:
    """GET /v1/goals/{id} returns the created goal (API-R35)."""
    created = await _create(env)
    resp = await env.client.get(f"/v1/goals/{created['goal_id']}", headers={})
    assert resp.status_code == 200
    assert resp.json()["goal_id"] == created["goal_id"]


@pytest.mark.asyncio
async def test_get_unknown_goal_is_404(env: _Env) -> None:
    """An unknown goal_id is a 404 not-found, never a silent 200 (API-R51)."""
    resp = await env.client.get(f"/v1/goals/{uuid.uuid4()}", headers={})
    assert resp.status_code == 404
    assert resp.json()["type"].endswith("/not-found")


@pytest.mark.asyncio
async def test_get_malformed_goal_id_is_404(env: _Env) -> None:
    """A non-UUID goal_id resolves to 404, never a 500 (API-R51, fail-closed)."""
    resp = await env.client.get("/v1/goals/not-a-uuid", headers={})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_patch_goal_updates_fields(env: _Env) -> None:
    """PATCH updates only the supplied fields and returns the updated Goal (API-R35)."""
    created = await _create(env)
    resp = await env.client.patch(
        f"/v1/goals/{created['goal_id']}",
        json={"title": "Sub-8:30 gran fondo", "status": "achieved"},
        headers={},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Sub-8:30 gran fondo"
    assert body["status"] == "achieved"
    assert body["target_event"] == "Maratona dles Dolomites"  # untouched field preserved


@pytest.mark.asyncio
async def test_patch_unknown_goal_is_404(env: _Env) -> None:
    """A PATCH to an unknown goal_id is a 404 not-found (API-R51)."""
    resp = await env.client.patch(f"/v1/goals/{uuid.uuid4()}", json={"title": "x"}, headers={})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_patch_unknown_sport_is_422(env: _Env) -> None:
    """A PATCH changing sport to an unregistered code is 422 unknown_sport (GBO-R38)."""
    created = await _create(env)
    resp = await env.client.patch(
        f"/v1/goals/{created['goal_id']}", json={"sport": "quidditch"}, headers={}
    )
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["code"] == "unknown_sport"


@pytest.mark.asyncio
async def test_delete_goal_soft_closes_to_terminal_status(env: _Env) -> None:
    """DELETE returns 204 AND soft-closes to a terminal status, never hard-deleting (GBO-R39).

    Reconciles API-R35's ``DELETE -> 204`` with GBO-R39's "Closing a goal MUST set status to a
    terminal value, never delete it, so goal history is auditable": the row survives with a
    terminal ``abandoned`` status; a subsequent GET still resolves it (200), not a 404.
    """
    created = await _create(env)
    resp = await env.client.delete(f"/v1/goals/{created['goal_id']}", headers={})
    assert resp.status_code == 204
    assert resp.content == b""
    async with env.factory() as session:
        row = await session.get(Goal, uuid.UUID(created["goal_id"]))
        assert row is not None  # history is preserved, NOT hard-deleted (GBO-R39)
        assert row.status is GoalStatus.ABANDONED  # terminal close


@pytest.mark.asyncio
async def test_delete_unknown_goal_is_404(env: _Env) -> None:
    """A DELETE of an unknown goal_id is a 404 not-found (API-R51)."""
    resp = await env.client.delete(f"/v1/goals/{uuid.uuid4()}", headers={})
    assert resp.status_code == 404


# --- GET /v1/goals (list) — filters + sort allow-list ---------------------------


async def _seed_goals(env: _Env) -> None:
    """Create a spread of goals across status / sport / target_date for filter+sort tests."""
    await _create(env, title="A active cyc early", sport="cycling", target_date="2026-03-01")
    await _create(env, title="B active run late", sport="running", target_date="2026-09-01")
    achieved = await _create(env, title="C achieved cyc", sport="cycling", target_date="2026-06-01")
    await env.client.patch(
        f"/v1/goals/{achieved['goal_id']}", json={"status": "achieved"}, headers={}
    )


@pytest.mark.asyncio
async def test_list_filters_by_status(env: _Env) -> None:
    """GET /v1/goals?status=active returns only active goals (API-R35 typed filter)."""
    await _seed_goals(env)
    resp = await env.client.get("/v1/goals?status=active", headers={})
    assert resp.status_code == 200
    statuses = {g["status"] for g in resp.json()["data"]}
    assert statuses == {"active"}
    titles = {g["title"] for g in resp.json()["data"]}
    assert titles == {"A active cyc early", "B active run late"}


@pytest.mark.asyncio
async def test_list_filters_by_sport(env: _Env) -> None:
    """GET /v1/goals?sport=running returns only that sport's goals (API-R35 typed filter)."""
    await _seed_goals(env)
    resp = await env.client.get("/v1/goals?sport=running", headers={})
    assert resp.status_code == 200
    sports = {g["sport"] for g in resp.json()["data"]}
    assert sports == {"running"}


@pytest.mark.asyncio
async def test_list_filters_by_target_date_window(env: _Env) -> None:
    """GET /v1/goals?from=&to= filters on the target_date window (API-R35 typed filter)."""
    await _seed_goals(env)
    resp = await env.client.get("/v1/goals?from=2026-05-01&to=2026-07-01", headers={})
    assert resp.status_code == 200
    titles = {g["title"] for g in resp.json()["data"]}
    assert titles == {"C achieved cyc"}  # only the 2026-06-01 goal falls in the window


@pytest.mark.asyncio
async def test_list_reversed_window_is_422(env: _Env) -> None:
    """from > to on the list window is a 422, never a silent empty list (ERR-R6)."""
    resp = await env.client.get("/v1/goals?from=2026-07-01&to=2026-05-01", headers={})
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


@pytest.mark.asyncio
async def test_list_sorts_by_target_date(env: _Env) -> None:
    """GET /v1/goals?sort=target_date orders by the target_date axis (API-R35 sort allow-list)."""
    await _seed_goals(env)
    resp = await env.client.get("/v1/goals?sort=target_date&order=asc", headers={})
    assert resp.status_code == 200
    dates = [g["target_date"] for g in resp.json()["data"]]
    assert dates == sorted(dates)


@pytest.mark.asyncio
async def test_list_rejects_off_allowlist_sort(env: _Env) -> None:
    """A sort key outside the {target_date, created_at} allow-list is a 422 (API-R35/PAGE-R2)."""
    resp = await env.client.get("/v1/goals?sort=title", headers={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_is_athlete_scoped(env: _Env) -> None:
    """The list returns ONLY the server-derived owner's goals, never a foreign athlete's.

    A second athlete's goal is seeded directly into the store; the owner-scoped list must not
    surface it (AUTH-R3 / API-R51 — access reduces to the one owner's rows).
    """
    mine = await _create(env)
    other_id = uuid.uuid4()
    async with env.factory() as session:
        session.add(Athlete(athlete_id=other_id, sex="female", reference_timezone="UTC"))
        await session.flush()
        session.add(
            Goal(
                goal_id=uuid.uuid4(),
                athlete_id=other_id,
                sport="cycling",
                goal_type=GoalType.EVENT,
                title="Foreign goal",
                status=GoalStatus.ACTIVE,
            )
        )
        await session.commit()
    resp = await env.client.get("/v1/goals", headers={})
    assert resp.status_code == 200
    ids = {g["goal_id"] for g in resp.json()["data"]}
    assert ids == {mine["goal_id"]}  # the foreign goal is invisible to the owner


@pytest.mark.asyncio
async def test_get_foreign_goal_is_404(env: _Env) -> None:
    """A GET of another athlete's goal_id is a 404, never disclosed (AUTH-R3 / API-R51)."""
    other_id = uuid.uuid4()
    foreign_goal_id = uuid.uuid4()
    async with env.factory() as session:
        session.add(Athlete(athlete_id=other_id, sex="female", reference_timezone="UTC"))
        await session.flush()
        session.add(
            Goal(
                goal_id=foreign_goal_id,
                athlete_id=other_id,
                sport="cycling",
                goal_type=GoalType.EVENT,
                title="Foreign goal",
                status=GoalStatus.ACTIVE,
            )
        )
        await session.commit()
    resp = await env.client.get(f"/v1/goals/{foreign_goal_id}", headers={})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_paginates_with_signed_cursor(env: _Env) -> None:
    """A limit=1 page returns one row + an opaque cursor paging the rest (PAGE-R1/R7)."""
    await _seed_goals(env)
    first = await env.client.get("/v1/goals?limit=1", headers={})
    assert first.status_code == 200
    page1 = first.json()
    assert len(page1["data"]) == 1
    assert page1["page"]["has_more"] is True
    cursor = page1["page"]["next_cursor"]
    assert cursor
    second = await env.client.get(f"/v1/goals?limit=1&cursor={cursor}", headers={})
    assert second.status_code == 200
    page2 = second.json()
    assert len(page2["data"]) == 1
    assert page1["data"][0]["goal_id"] != page2["data"][0]["goal_id"]


@pytest.mark.asyncio
async def test_list_target_date_sort_pages_null_dated_goals_losslessly(env: _Env) -> None:
    """A target_date-sorted page walk never DROPS a NULL-target_date goal (PAGE-R7 keyset).

    ``target_date`` is nullable; a naive keyset comparison over it returns NULL for a NULL-dated row
    and silently drops it across pages. Seed dated AND undated goals, then walk every page at
    ``limit=1`` sorted by ``target_date`` — the union of pages MUST equal the full set, undated
    goals included. Mutation-proof: a raw (un-coalesced) ``target_date`` keyset loses undated rows.
    """
    a = await _create(env, title="dated-early", target_date="2026-03-01")
    b = await _create(
        env, title="undated-1", goal_type="process", target_event=None, target_date=None
    )
    c = await _create(env, title="dated-late", target_date="2026-09-01")
    d = await _create(
        env, title="undated-2", goal_type="process", target_event=None, target_date=None
    )
    expected = {a["goal_id"], b["goal_id"], c["goal_id"], d["goal_id"]}
    seen: set[str] = set()
    url: str | None = "/v1/goals?sort=target_date&order=asc&limit=1"
    for _ in range(10):
        if url is None:
            break
        resp = await env.client.get(url, headers={})
        assert resp.status_code == 200
        page = resp.json()
        seen.update(g["goal_id"] for g in page["data"])
        nxt = page["page"]["next_cursor"]
        url = (
            f"/v1/goals?sort=target_date&order=asc&limit=1&cursor={nxt}"
            if page["page"]["has_more"] and nxt
            else None
        )
    assert seen == expected  # every goal — including both undated ones — paged through


@pytest.mark.asyncio
async def test_list_no_goals_is_typed_empty(env: _Env) -> None:
    """An owner with no goals gets a typed empty list, never a 404 (API-R35)."""
    resp = await env.client.get("/v1/goals", headers={})
    assert resp.status_code == 200
    assert resp.json()["data"] == []


@pytest.mark.asyncio
async def test_response_carries_no_athlete_id_field(env: _Env) -> None:
    """The Goal response shape carries no athlete_id (identity is the caller's, AUTH-R3)."""
    created = await _create(env)
    flat = json.dumps(created)
    assert '"athlete_id"' not in flat
