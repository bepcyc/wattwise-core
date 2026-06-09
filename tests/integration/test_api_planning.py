"""Integration tests for the planning router (doc 60 §planning, API-R32 / API-R12a).

Drives the three ``/v1/planning/*`` endpoints end-to-end over the assembled ASGI app with the
router's dependency seams overridden against a FAKE :class:`PlanningEngine` and a seeded canonical
store, asserting the boundary contract the router owns:

- **API-R32 / API-R12a** ``POST /v1/planning/workouts`` drives the engine's ``plan_deliverable`` and
  renders the status-discriminated ``AgentAskResponse`` union — an approval-gated PLAN comes back
  ``awaiting_approval`` carrying the ``interrupt_id`` + grounded plan body so the EXISTING
  ``POST /v1/agent/threads/{id}/decision`` endpoint can approve/resume it (CKPT-R9).
- **RUN-R4.1 / phase-gating** an engine without ``plan_deliverable`` (the OSS no-LLM engine) yields
  a typed ``degraded`` "not yet available" answer, never an error and never a fabricated plan.
- **AUTH-R3 / SCHEMA-R4** identity is server-derived; a forged ``athlete_id`` body field is a
  ``422`` before the engine; the server-derived id is what reaches the engine.
- **API-R13 / SCHEMA-R7** the rendered ``plan_html`` is server-side sanitized inert.
- **API-R11c** no athlete-facing response carries billing/model/token machinery.
- **API-R32 read views** ``GET /v1/planning/workouts`` paginates the canonical workout library
  (owned + shared); ``GET /v1/planning/schedule`` reads the active plan's immutable days, returns a
  typed empty view (never ``404``) with no plan, and rejects ``from > to`` with ``422``.

The engine is a stand-in for the multi-day PLAN deliverable projection (ARCH-R21): the router is the
unit under test, not the grounding engine. Read views run on a seeded SQLite store (single session,
no concurrency — the portable substrate, GBO-R8b).
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.testclient import TestClient

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import Citation, Observation, Plan
from wattwise_core.api.app import create_app
from wattwise_core.api.auth import Scope, issue_access_token
from wattwise_core.api.errors import install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import planning as planning_router
from wattwise_core.config import Environment, load_settings
from wattwise_core.domain.enums import PlanDayIntent, PlanStatus
from wattwise_core.persistence.models import Athlete, Base, PlanDay, Sport, Workout
from wattwise_core.persistence.models import Plan as PlanRow

pytestmark = pytest.mark.integration

UTC = _dt.UTC

#: The forbidden billing/budget/model machinery fields (API-R11c).
FORBIDDEN_FIELDS = (
    "usage", "cost_remaining_usd", "cost_usd_estimate", "input_tokens",
    "output_tokens", "model_tier", "reasoning", "model",
)


# --- POST /v1/planning/workouts — fake-engine boundary tests ----------------------


class _FakePlanEngine:
    """A controllable stand-in for the multi-day PLAN engine (ARCH-R21 seam, API-R12a).

    Returns a preset :class:`Plan` so the router's boundary behavior — not the grounding — is what
    is exercised. ``athlete_id``/``request`` are recorded to assert the router passes the
    server-derived id (never a client value) and the request text through.
    """

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.seen_athlete_id: str | None = None
        self.seen_request: str | None = None
        self.seen_requires_approval: bool | None = None

    async def plan_deliverable(
        self,
        *,
        athlete_id: str,
        request: str,
        thread_id: str | None,
        locale: str,
        response_length: str,
        requires_approval: bool,
    ) -> Plan:
        self.seen_athlete_id = athlete_id
        self.seen_request = request
        self.seen_requires_approval = requires_approval
        return self._plan


class _NoPlanEngine:
    """The OSS no-LLM engine stand-in: no ``plan_deliverable`` method (phase-gated, RUN-R4.1)."""


def _awaiting_plan(
    *, plan_html: str = "<p>Week 1: build aerobic base.</p>", interrupt_id: str = "01INT"
) -> Plan:
    """An approval-gated PLAN paused at the durable interrupt-gate (CKPT-R9)."""
    return Plan(
        status=RunStatus.AWAITING_APPROVAL,
        thread_id="owner:01CONV",
        plan_html=plan_html,
        plan_text="Week 1: build aerobic base.",
        interrupt_id=interrupt_id,
        observations=(Observation(observation_id="01OBS", text="Build aerobic base."),),
        citations=(Citation(record_id="01CIT", metric="ctl", value=42.0, as_of="2026-06-07"),),
        suggested_followups=("Make it harder",),
    )


def _completed_plan() -> Plan:
    """A non-paused terminal PLAN (the no-approval / already-resolved path)."""
    return Plan(
        status=RunStatus.COMPLETED,
        thread_id="owner:01CONV",
        plan_html="<p>Recovery week.</p>",
        plan_text="Recovery week.",
        citations=(Citation(record_id="01CIT", metric="ctl", value=42.0, as_of="2026-06-07"),),
    )


def _build_plan_app(engine: object, *, limiter: RateLimiter | None = None) -> FastAPI:
    """Build the REAL app, mount the planning router, and override its agent-path seams.

    Uses :func:`create_app` so the full app is assembled (the contract's "build the app and drive
    POST"); the planning router's engine/identity/scope/limiter seams are overridden so the fake
    engine — not the live LangGraph — is exercised.
    """
    settings = load_settings(
        app__environment=Environment.DEVELOPMENT,
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="test-signing-key-0123456789abcdef",
    )
    app = create_app(settings)
    app.include_router(planning_router.router)
    bucket = limiter or RateLimiter()
    app.dependency_overrides[planning_router.require_agent_scope] = lambda: None
    app.dependency_overrides[planning_router.current_athlete_id] = lambda: "owner"
    app.dependency_overrides[planning_router.planning_engine] = lambda: engine
    app.dependency_overrides[planning_router.rate_limiter] = lambda: bucket
    return app


def _token(app: FastAPI) -> str:
    settings = app.state.settings
    tokens = issue_access_token(settings, subject="owner", scopes=(Scope.AGENT, Scope.READ))
    return tokens.access_token


def _auth(app: FastAPI) -> dict[str, str]:
    return {"Authorization": f"Bearer {_token(app)}"}


def test_post_workouts_reaches_awaiting_approval_with_interrupt_id() -> None:
    """The contract drive: POST → awaiting_approval with interrupt_id + grounded plan (API-R12a).

    The approval-gated PLAN comes back ``awaiting_approval`` with the ``interrupt_id`` the EXISTING
    ``POST /v1/agent/threads/{id}/decision`` endpoint consumes (CKPT-R9) and the grounded plan body
    in BOTH ``plan_*`` and ``answer_*`` fields, so a status-agnostic client still renders the prose.
    """
    engine = _FakePlanEngine(_awaiting_plan())
    app = _build_plan_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/planning/workouts",
            json={"request": "give me a week plan"},
            headers=_auth(app),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "awaiting_approval"
    assert body["interrupt_id"] == "01INT"  # the decision endpoint consumes this (CKPT-R9)
    assert body["grounding"]["grounded"] is True
    assert body["plan_text"] == "Week 1: build aerobic base."
    assert body["answer_text"] == body["plan_text"]  # mirrored for status-agnostic clients
    assert engine.seen_athlete_id == "owner"  # AUTH-R3: server-derived, not a client value
    assert engine.seen_request == "give me a week plan"
    assert engine.seen_requires_approval is True  # approval-gated by default (COACH-R2)
    flat = json.dumps(body)
    for field in FORBIDDEN_FIELDS:
        assert f'"{field}"' not in flat, f"forbidden field {field!r} leaked (API-R11c)"


def test_post_workouts_awaiting_plan_html_is_sanitized_inert() -> None:
    """The awaiting-approval plan_html is server-side sanitized inert before return (API-R13)."""
    engine = _FakePlanEngine(
        _awaiting_plan(plan_html="<p>Build.</p><script>alert(1)</script><img src=x onerror=y>")
    )
    app = _build_plan_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/planning/workouts", json={"request": "plan"}, headers=_auth(app)
        )
    assert resp.status_code == 200
    html = resp.json()["plan_html"].lower()
    assert "<script" not in html
    assert "onerror" not in html
    assert resp.json()["answer_html"] == resp.json()["plan_html"]


def test_post_workouts_completed_plan_renders_completed_union() -> None:
    """A non-paused terminal plan renders the ``completed`` member, grounded (API-R11a)."""
    app = _build_plan_app(_FakePlanEngine(_completed_plan()))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/planning/workouts", json={"request": "easy week"}, headers=_auth(app)
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["grounding"]["grounded"] is True
    assert body["interrupt_id"] is None


def test_post_workouts_phase_gated_when_engine_cannot_generate() -> None:
    """A no-LLM engine (no ``plan_deliverable``) yields a typed degraded answer (RUN-R4.1)."""
    app = _build_plan_app(_NoPlanEngine())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/planning/workouts", json={"request": "plan please"}, headers=_auth(app)
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["degraded"]["coverage_caveat"]["reason"] == "agent_unconfigured"
    assert body["answer_text"]  # a jargon-free sentence, not an error


def test_post_workouts_forged_athlete_id_is_422_before_engine() -> None:
    """A forged caller-identity body field is rejected before the engine (AUTH-R3 / SCHEMA-R4)."""
    engine = _FakePlanEngine(_awaiting_plan())
    app = _build_plan_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/planning/workouts",
            json={"request": "plan", "athlete_id": "attacker"},
            headers=_auth(app),
        )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")
    assert engine.seen_athlete_id is None  # the forged field never reached the engine


def test_post_workouts_missing_request_is_422() -> None:
    """A request body with no ``request`` is a 422 validation error (SCHEMA-R4)."""
    app = _build_plan_app(_FakePlanEngine(_awaiting_plan()))
    with TestClient(app) as client:
        resp = client.post("/v1/planning/workouts", json={}, headers=_auth(app))
    assert resp.status_code == 422


def test_post_workouts_rate_limited_returns_429() -> None:
    """The 21st plan call in a window is 429 — the agent bucket is debited (LIMIT-R2)."""
    app = _build_plan_app(_FakePlanEngine(_awaiting_plan()), limiter=RateLimiter())
    with TestClient(app) as client:
        headers = _auth(app)
        last = None
        for _ in range(21):
            last = client.post(
                "/v1/planning/workouts", json={"request": "plan"}, headers=headers
            )
    assert last is not None
    assert last.status_code == 429
    assert last.json()["type"].endswith("/rate-limited")


def test_post_workouts_requires_agent_scope() -> None:
    """Without the scope override, the unwired agent-scope seam fails closed 403 (AUTH-R13)."""
    app = _build_plan_app(_FakePlanEngine(_awaiting_plan()))
    # Drop the agent-scope override so the router's OWN fail-closed seam (insufficient-scope) runs.
    del app.dependency_overrides[planning_router.require_agent_scope]
    with TestClient(app) as client:
        resp = client.post(
            "/v1/planning/workouts", json={"request": "plan"}, headers=_auth(app)
        )
    assert resp.status_code == 403
    assert resp.json()["type"].endswith("/insufficient-scope")


# --- GET /v1/planning/workouts + /schedule — seeded read views --------------------


@dataclass
class _ReadEnv:
    """The wired read-view app + its client/session for one seeded scenario."""

    client: AsyncClient
    app: FastAPI
    session: AsyncSession
    athlete_id: str


@pytest_asyncio.fixture
async def seeded() -> AsyncIterator[_ReadEnv]:
    """An app on a seeded store: one owner, an owned + a shared workout, an active 2-day plan."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        athlete_id = await _seed(session)
        app = _build_read_app(session, athlete_id)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield _ReadEnv(client, app, session, athlete_id)
    await engine.dispose()


def _build_read_app(session: AsyncSession, athlete_id: str) -> FastAPI:
    """Mount the planning router and override the read-view identity/scope/session/cursor seams."""
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(planning_router.router)
    app.dependency_overrides.update(
        {
            planning_router.require_read_scope: lambda: None,
            planning_router.current_athlete_id: lambda: athlete_id,
            planning_router.current_session: lambda: session,
            planning_router.cursor_signing_key: lambda: "test-cursor-key-0123456789abcdef",
        }
    )
    return app


async def _seed(session: AsyncSession) -> str:
    """Seed one owner, an owned + a shared workout, and an active plan with two immutable days."""
    session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
    athlete = Athlete(sex="male", reference_timezone="UTC", current_sport="cycling")
    session.add(athlete)
    await session.flush()
    aid = athlete.athlete_id
    owned_wid = uuid.uuid4()
    session.add(
        Workout(
            workout_id=owned_wid,
            athlete_id=aid,
            name="Sweet Spot 2x20",
            sport="cycling",
            steps=[
                {"intent": "warmup", "duration_s": 600},
                {"intent": "work", "duration_s": 1200, "target_low": 0.88,
                 "target_high": 0.94, "target_unit": "ftp"},
            ],
        )
    )
    session.add(
        Workout(
            workout_id=uuid.uuid4(),
            athlete_id=None,  # NULL-athlete shared library template (TEN-R1)
            name="Library Endurance Ride",
            sport="cycling",
            steps=[{"intent": "steady", "duration_s": 3600}],
        )
    )
    plan_id = uuid.uuid4()
    session.add(
        PlanRow(
            plan_id=plan_id,
            athlete_id=aid,
            start_date=_dt.date(2026, 6, 8),
            end_date=_dt.date(2026, 6, 14),
            status=PlanStatus.ACTIVE,
            lineage={},
        )
    )
    session.add(
        PlanDay(
            plan_day_id=uuid.uuid4(),
            plan_id=plan_id,
            plan_date=_dt.date(2026, 6, 9),
            athlete_id=aid,
            workout_id=owned_wid,
            intent=PlanDayIntent.THRESHOLD,
            rationale="Build threshold.",
        )
    )
    session.add(
        PlanDay(
            plan_day_id=uuid.uuid4(),
            plan_id=plan_id,
            plan_date=_dt.date(2026, 6, 10),
            athlete_id=aid,
            workout_id=None,  # rest marker
            intent=PlanDayIntent.REST,
            rationale=None,
        )
    )
    await session.flush()
    return str(aid)


@pytest.mark.asyncio
async def test_get_workouts_lists_owned_and_shared_templates(seeded: _ReadEnv) -> None:
    """The read view lists the athlete's own + the shared library templates (API-R32 / GBO-R29)."""
    resp = await seeded.client.get("/v1/planning/workouts", headers={})
    assert resp.status_code == 200
    body = resp.json()
    names = {w["name"] for w in body["data"]}
    assert names == {"Sweet Spot 2x20", "Library Endurance Ride"}
    shared = {w["name"]: w["shared"] for w in body["data"]}
    assert shared["Library Endurance Ride"] is True  # NULL-athlete template marked shared (TEN-R1)
    assert shared["Sweet Spot 2x20"] is False
    sweet = next(w for w in body["data"] if w["name"] == "Sweet Spot 2x20")
    work = next(s for s in sweet["steps"] if s["intent"] == "work")
    assert work["target_low"] == 0.88 and work["target_unit"] == "ftp"  # target zones surfaced
    flat = json.dumps(body)
    for field in FORBIDDEN_FIELDS:
        assert f'"{field}"' not in flat


@pytest.mark.asyncio
async def test_get_workouts_paginates_with_signed_cursor(seeded: _ReadEnv) -> None:
    """A ``limit=1`` page returns one row + an opaque next cursor paging the rest (PAGE-R1/R7)."""
    first = await seeded.client.get("/v1/planning/workouts?limit=1", headers={})
    assert first.status_code == 200
    page1 = first.json()
    assert len(page1["data"]) == 1
    assert page1["page"]["has_more"] is True
    cursor = page1["page"]["next_cursor"]
    assert cursor
    second = await seeded.client.get(
        f"/v1/planning/workouts?limit=1&cursor={cursor}", headers={}
    )
    assert second.status_code == 200
    page2 = second.json()
    assert len(page2["data"]) == 1
    # The two pages are disjoint — the cursor advanced past the first row (PAGE-R7).
    assert page1["data"][0]["workout_id"] != page2["data"][0]["workout_id"]


@pytest.mark.asyncio
async def test_get_schedule_reads_active_plan_immutable_days(seeded: _ReadEnv) -> None:
    """The schedule view reads the active plan's immutable days in range (API-R32 / GBO-R30b)."""
    resp = await seeded.client.get(
        "/v1/planning/schedule?from=2026-06-08&to=2026-06-14", headers={}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["plan_id"] is not None
    assert body["status"] == "active"
    assert [d["plan_date"] for d in body["days"]] == ["2026-06-09", "2026-06-10"]
    threshold = body["days"][0]
    assert threshold["intent"] == "threshold"
    assert threshold["workout_id"] is not None
    assert threshold["rationale"] == "Build threshold."
    rest = body["days"][1]
    assert rest["intent"] == "rest"
    assert rest["workout_id"] is None  # a rest marker carries no workout


@pytest.mark.asyncio
async def test_get_schedule_range_filters_days(seeded: _ReadEnv) -> None:
    """A narrowed ``from``/``to`` returns only the in-range immutable days (API-R32)."""
    resp = await seeded.client.get(
        "/v1/planning/schedule?from=2026-06-10&to=2026-06-10", headers={}
    )
    assert resp.status_code == 200
    assert [d["plan_date"] for d in resp.json()["days"]] == ["2026-06-10"]


@pytest.mark.asyncio
async def test_get_schedule_reversed_range_is_422(seeded: _ReadEnv) -> None:
    """``from > to`` is a 422 validation error, never a silent empty view (ERR-R6)."""
    resp = await seeded.client.get(
        "/v1/planning/schedule?from=2026-06-14&to=2026-06-08", headers={}
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


@pytest.mark.asyncio
async def test_get_schedule_no_active_plan_is_typed_empty_not_404() -> None:
    """With no active plan the schedule is a typed empty view, never a 404 (API-R32 / GBO-R30a)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        athlete = Athlete(sex="male", reference_timezone="UTC", current_sport="cycling")
        session.add(athlete)
        await session.flush()
        aid = str(athlete.athlete_id)
        app = _build_read_app(session, aid)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            resp = await client.get(
                "/v1/planning/schedule?from=2026-06-01&to=2026-06-30", headers={}
            )
    await engine.dispose()
    assert resp.status_code == 200
    body = resp.json()
    assert body["plan_id"] is None
    assert body["days"] == []
