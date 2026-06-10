"""Integration tests for the agent BREADTH surface — diagnose / digest / memory (doc 60 §6/§7).

Drives the ``/v1/agent`` breadth endpoints end-to-end over the assembled ASGI app with a FAKE
:class:`BreadthEngine` injected through the router's override seams (and a REAL canonical session
for the digest-subscription persistence + the email-verified gate), asserting the boundary contract
the router owns:

- **API-R15** ``POST /v1/agent/diagnose`` returns the DETERMINISTIC coverage narration (no made-up
  number); ``degraded`` when no canonical input is present, scoped to the server-derived id
  (AUTH-R3).
- **API-R14** the digest subscription CRUD persists ONE standing schedule for the server-derived
  owner; a forged ``athlete_id`` / a weekly cadence with no weekday is ``422``; an unknown / foreign
  subscription id is ``404``; the ``email`` channel is GATED on the verified email (GBO-R49) — a
  subscription naming ``email`` before verification is refused ``422`` (fail-closed).
- **API-R14** ``GET /v1/agent/digest/last`` is server-side sanitized (API-R13) and abstains visibly.
- **API-R15a / MEM-R3 MUST** the per-item memory GET/DELETE is owner-scoped; a per-item erase
  removes the residual row (PRIV-R8) so a re-GET is ``404``; a foreign / unknown id is ``404``
  (never disclosed). The memory surface is NON-LLM and outside the agent cost gate.
- **API-R11c** no athlete-facing breadth response carries billing/model/token machinery.

The fake engine stands in for the deliverables projection (ARCH-R21): the router is the unit under
test, not the grounding engine.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.agent.contracts import RunStatus
from wattwise_core.agent.deliverables import Citation, Digest, Observation
from wattwise_core.agent.diagnose_deliverable import AgentDiagnosis, InputCoverage, InputStatus
from wattwise_core.agent.memory import MemoryItemKind, RecalledItem
from wattwise_core.api.app import create_app
from wattwise_core.api.routers import agent_routes
from wattwise_core.config import Environment, load_settings
from wattwise_core.domain.enums import DeliveryChannel, DigestStatus
from wattwise_core.persistence.models import Athlete, Base, DigestSubscription, NotificationRoute

pytestmark = pytest.mark.integration

#: Billing/model/token machinery — none may appear on any athlete-facing response (API-R11c).
FORBIDDEN_FIELDS = (
    "usage",
    "cost_remaining_usd",
    "input_tokens",
    "model_tier",
    "reasoning",
    "model",
)


# --- fake engine (the diagnose / digest / memory deliverables projection, ARCH-R21) ---


@dataclass
class _FakeBreadthEngine:
    """A controllable stand-in for the breadth engine seam (ARCH-R21).

    Returns preset deliverables so the router's boundary behaviour — not the grounding — is what is
    exercised. ``seen_athlete_id`` records the id the router passed so a test can assert it is the
    server-derived one (AUTH-R3), never a client value. Memory rows are held in-process and scoped
    by athlete id, modelling the owner-scoped store the real engine drives.
    """

    diagnosis: AgentDiagnosis
    digest_body: Digest
    memory: dict[str, list[RecalledItem]] = field(default_factory=dict)
    seen_athlete_id: str | None = None

    async def diagnose(self, *, athlete_id: str, locale: str) -> AgentDiagnosis:
        self.seen_athlete_id = athlete_id
        return self.diagnosis

    async def digest(self, *, athlete_id: str, week_end: str, entitlement: Any = None) -> Digest:
        self.seen_athlete_id = athlete_id
        return self.digest_body

    async def list_memory(
        self, *, athlete_id: str, limit: int, offset: int
    ) -> Sequence[RecalledItem]:
        self.seen_athlete_id = athlete_id
        return self.memory.get(athlete_id, [])[offset : offset + limit]

    async def get_memory(self, *, athlete_id: str, memory_item_id: str) -> RecalledItem | None:
        self.seen_athlete_id = athlete_id
        for item in self.memory.get(athlete_id, []):
            if item.memory_item_id == memory_item_id:
                return item
        return None

    async def delete_memory(self, *, athlete_id: str, memory_item_id: str) -> bool:
        self.seen_athlete_id = athlete_id
        rows = self.memory.get(athlete_id, [])
        kept = [r for r in rows if r.memory_item_id != memory_item_id]
        if len(kept) == len(rows):
            return False
        self.memory[athlete_id] = kept
        return True


def _recalled(content: str, *, item_id: str | None = None) -> RecalledItem:
    """A trusted PREFERENCE memory item (personalization context only, never a number, MEM-R1)."""
    return RecalledItem(
        memory_item_id=item_id or str(uuid.uuid4()),
        kind=MemoryItemKind.PREFERENCE,
        content=content,
        inferred=False,
        recorded_at=_dt.datetime(2026, 6, 1, 12, 0, tzinfo=_dt.UTC),
    )


def _diagnosis(*, present: bool) -> AgentDiagnosis:
    """A diagnosis: ``completed`` with a present input, or ``degraded`` w/ no coverage (API-R15)."""
    if present:
        return AgentDiagnosis(
            status=RunStatus.COMPLETED,
            athlete_id="owner",
            as_of="2026-06-08",
            inputs=(
                InputCoverage("training_load", "Training load", InputStatus.PRESENT),
                InputCoverage("hrv", "Recovery (HRV)", InputStatus.MISSING, reason="missing_input"),
            ),
        )
    return AgentDiagnosis(
        status=RunStatus.DEGRADED,
        athlete_id="owner",
        as_of="2026-06-08",
        inputs=(
            InputCoverage("hrv", "Recovery (HRV)", InputStatus.MISSING, reason="missing_input"),
        ),
        coverage_caveat={"reason": "no_canonical_coverage", "inputs_unavailable": ["hrv"]},
    )


def _digest(*, status: RunStatus = RunStatus.COMPLETED, html: str = "<p>Solid week.</p>") -> Digest:
    """A grounded weekly digest with a stable observation + citation (COACH-R1 #1)."""
    return Digest(
        status=status,
        thread_id="owner:digest:2026-06-07",
        week_end="2026-06-07",
        digest_html=html,
        digest_text="Solid week.",
        observations=(Observation(observation_id="01OBS", text="You built aerobic base."),),
        citations=(Citation(record_id="01CIT", metric="ctl", value=42.0, as_of="2026-06-07"),),
        coverage_caveat={"inputs": [{"input": "hrv", "state": "missing"}]}
        if status is RunStatus.DEGRADED
        else None,
    )


# --- app wiring (mounts agent_routes, which includes the breadth router) ----------


@dataclass
class _Env:
    """The wired app + client + canonical session + fake engine for one scenario."""

    client: AsyncClient
    engine: _FakeBreadthEngine
    session: AsyncSession
    athlete_id: str


def _build_app(engine: _FakeBreadthEngine, session: AsyncSession, athlete_id: str) -> FastAPI:
    """Assemble the app with the agent router + the breadth seams overridden (mirrors the contract).

    Overrides the SHARED agent seams (scope / server-derived id / engine / rate limiter) and the new
    ``current_session`` DB seam the digest persistence + email gate need — exactly the seams the app
    factory wires in production.
    """
    settings = load_settings(
        app__environment=Environment.DEVELOPMENT,
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="test-signing-key-0123456789abcdef",
    )
    app = create_app(settings)
    app.include_router(agent_routes.router)

    async def _session() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[agent_routes.require_agent_scope] = lambda: None
    app.dependency_overrides[agent_routes.current_athlete_id] = lambda: athlete_id
    app.dependency_overrides[agent_routes.agent_engine] = lambda: engine
    app.dependency_overrides[agent_routes.current_session] = _session
    return app


@pytest_asyncio.fixture
async def env() -> AsyncIterator[_Env]:
    """An app over a seeded canonical store (one owner, no email captured) + a fake engine."""
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
        fake = _FakeBreadthEngine(diagnosis=_diagnosis(present=True), digest_body=_digest())
        app = _build_app(fake, session, athlete_id)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            yield _Env(client, fake, session, athlete_id)
    await engine.dispose()


def _auth() -> dict[str, str]:
    """A bearer header so the route security extractor is satisfied (value unused in-test)."""
    return {"Authorization": "Bearer test"}


async def _seed_email_route(session: AsyncSession, athlete_id: str, *, verified: bool) -> None:
    """Seed the owner's ``email`` notification route in the given verified state (GBO-R49)."""
    session.add(
        NotificationRoute(
            athlete_id=uuid.UUID(athlete_id),
            channel=DeliveryChannel.EMAIL,
            address_ref="rider@example.com",
            verified=verified,
            enabled=True,
        )
    )
    await session.commit()


def _no_forbidden(body: dict[str, object]) -> None:
    """Assert no billing/model/token machinery leaked onto an athlete-facing response (API-R11c)."""
    flat = json.dumps(body)
    for field_name in FORBIDDEN_FIELDS:
        assert f'"{field_name}"' not in flat, f"forbidden field {field_name!r} leaked (API-R11c)"


# --- POST /v1/agent/diagnose (API-R15) -------------------------------------------


async def test_diagnose_completed_reports_coverage_no_number(env: _Env) -> None:
    """A present athlete -> completed coverage lines, no athlete-facing number (VOICE-R7)."""
    resp = await env.client.post("/v1/agent/diagnose", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    keys = {i["key"]: i for i in body["inputs"]}
    assert keys["training_load"]["status"] == "present"
    assert keys["hrv"]["status"] == "missing"
    assert "value" not in json.dumps(body)  # a diagnosis reports coverage, never a metric value
    assert env.engine.seen_athlete_id == env.athlete_id  # server-derived identity (AUTH-R3)
    _no_forbidden(body)


async def test_diagnose_degraded_carries_typed_caveat(env: _Env) -> None:
    """No canonical coverage -> degraded + typed no_canonical_coverage caveat (OUTCOME-R3)."""
    env.engine.diagnosis = _diagnosis(present=False)
    resp = await env.client.post("/v1/agent/diagnose", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["coverage_caveat"]["reason"] == "no_canonical_coverage"
    assert "hrv" in body["coverage_caveat"]["inputs_unavailable"]


# --- GET /v1/agent/digest/last (API-R14) -----------------------------------------


async def test_digest_last_is_sanitized_and_grounded(env: _Env) -> None:
    """The digest body html is server-side sanitized + grounded (API-R13 / SCHEMA-R7 / API-R14)."""
    env.engine.digest_body = _digest(html="<p>Week.</p><script>steal()</script>")
    resp = await env.client.get("/v1/agent/digest/last", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert "<script" not in body["digest_html"].lower()
    assert body["grounding"]["grounded"] is True
    assert body["grounding"]["citations"][0]["metric"] == "ctl"
    assert env.engine.seen_athlete_id == env.athlete_id  # AUTH-R3
    _no_forbidden(body)


async def test_digest_last_degraded_localizes_caveat(env: _Env) -> None:
    """A degraded week surfaces the localized human caveat over the typed note (API-R37)."""
    env.engine.digest_body = _digest(status=RunStatus.DEGRADED)
    resp = await env.client.get(
        "/v1/agent/digest/last", headers={**_auth(), "Accept-Language": "de"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert "vorhandenen Daten" in body["degraded"]["reason_text"]  # German localization


# --- POST /v1/agent/digest/subscribe + list + delete (API-R14 / GBO-R46) ---------


async def test_subscribe_persists_then_lists_then_cancels(env: _Env) -> None:
    """A web subscription persists, lists, and a DELETE sets the terminal cancelled status."""
    resp = await env.client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "weekly", "weekday": "mon", "hour_local": 7, "channels": ["web"]},
        headers=_auth(),
    )
    assert resp.status_code == 200
    sub = resp.json()
    assert sub["cadence"] == "weekly"
    assert sub["weekday"] == "mon"
    assert sub["status"] == "active"
    sub_id = sub["subscription_id"]
    # it lists for the owner
    listed = await env.client.get("/v1/agent/digest/list", headers=_auth())
    assert listed.status_code == 200
    assert [s["subscription_id"] for s in listed.json()["data"]] == [sub_id]
    # delete -> 204 and the row is terminally cancelled (GBO-R47)
    deleted = await env.client.delete(f"/v1/agent/digest/subscribe/{sub_id}", headers=_auth())
    assert deleted.status_code == 204
    row = await env.session.get(DigestSubscription, uuid.UUID(sub_id))
    assert row is not None and row.status is DigestStatus.CANCELLED


async def test_resubscribe_replaces_keeping_one_active_row(env: _Env) -> None:
    """M5: two POSTs leave ONE active standing schedule — re-subscribe UPDATES, not duplicates."""
    first = await env.client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "weekly", "weekday": "mon", "hour_local": 7, "channels": ["web"]},
        headers=_auth(),
    )
    assert first.status_code == 200
    first_id = first.json()["subscription_id"]
    # a SECOND subscribe with different settings must replace the standing schedule, not add a row
    second = await env.client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "daily", "hour_local": 18, "channels": ["web"]},
        headers=_auth(),
    )
    assert second.status_code == 200
    # the same standing row is updated in place (same id), now carrying the new settings
    assert second.json()["subscription_id"] == first_id
    assert second.json()["cadence"] == "daily"
    assert second.json()["hour_local"] == 18
    # exactly ONE active row exists for the owner (GBO-R46 — one standing schedule)
    rows = (
        (
            await env.session.execute(
                select(DigestSubscription).where(
                    DigestSubscription.athlete_id == uuid.UUID(env.athlete_id),
                    DigestSubscription.status == DigestStatus.ACTIVE,
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    # and the list surfaces exactly one schedule
    listed = await env.client.get("/v1/agent/digest/list", headers=_auth())
    assert len(listed.json()["data"]) == 1


async def test_subscribe_forged_athlete_id_is_422(env: _Env) -> None:
    """A forged caller-identity body field is rejected before persistence (SCHEMA-R4 / AUTH-R3)."""
    resp = await env.client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "daily", "hour_local": 7, "channels": ["web"], "athlete_id": "attacker"},
        headers=_auth(),
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


async def test_subscribe_weekly_without_weekday_is_422(env: _Env) -> None:
    """A weekly cadence with no weekday is a 422 cross-field validation error (GBO-R46b)."""
    resp = await env.client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "weekly", "hour_local": 7, "channels": ["web"]},
        headers=_auth(),
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


async def test_delete_unknown_subscription_is_404(env: _Env) -> None:
    """Cancelling an unknown / non-UUID subscription id is 404 not-found (API-R51)."""
    unknown = await env.client.delete(f"/v1/agent/digest/subscribe/{uuid.uuid4()}", headers=_auth())
    assert unknown.status_code == 404
    assert unknown.json()["type"].endswith("/not-found")
    bad = await env.client.delete("/v1/agent/digest/subscribe/not-a-uuid", headers=_auth())
    assert bad.status_code == 404


async def test_subscribe_email_channel_refused_until_verified(env: _Env) -> None:
    """The email channel is GATED: an unverified email -> 422, a verified one -> 200 (GBO-R49)."""
    # no email route at all -> the email channel is not yet verified -> refused
    refused = await env.client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "daily", "hour_local": 7, "channels": ["web", "email"]},
        headers=_auth(),
    )
    assert refused.status_code == 422
    assert refused.json()["type"].endswith("/validation-error")
    # an UNVERIFIED captured email is still refused (fail-closed)
    await _seed_email_route(env.session, env.athlete_id, verified=False)
    still_refused = await env.client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "daily", "hour_local": 7, "channels": ["email"]},
        headers=_auth(),
    )
    assert still_refused.status_code == 422
    # once the email is verified the email channel is allowed
    route = (
        await env.session.execute(
            select(NotificationRoute).where(
                NotificationRoute.athlete_id == uuid.UUID(env.athlete_id),
                NotificationRoute.channel == DeliveryChannel.EMAIL,
            )
        )
    ).scalar_one()
    route.verified = True
    await env.session.commit()
    allowed = await env.client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "daily", "hour_local": 7, "channels": ["email"]},
        headers=_auth(),
    )
    assert allowed.status_code == 200
    assert allowed.json()["channels"] == ["email"]


# --- GET/DELETE /v1/agent/memory (API-R15a / MEM-R3 MUST) ------------------------


async def test_memory_list_is_owner_scoped(env: _Env) -> None:
    """The memory list returns the owner's rows only; a foreign athlete's never list (MEM-R3)."""
    env.engine.memory = {
        env.athlete_id: [_recalled("prefers morning rides"), _recalled("hates the trainer")],
        "other-athlete": [_recalled("foreign secret")],
    }
    resp = await env.client.get("/v1/agent/memory", headers=_auth())
    assert resp.status_code == 200
    contents = [r["content"] for r in resp.json()["data"]]
    assert contents == ["prefers morning rides", "hates the trainer"]
    assert "foreign secret" not in contents
    _no_forbidden(resp.json())


async def test_memory_get_and_erase_then_regets_404(env: _Env) -> None:
    """A per-item erase removes the residual row so a re-GET is 404 (MEM-R3 MUST / PRIV-R8)."""
    item = _recalled("prefers tempo work", item_id="11111111-1111-7000-8000-000000000001")
    env.engine.memory = {env.athlete_id: [item]}
    got = await env.client.get(f"/v1/agent/memory/{item.memory_item_id}", headers=_auth())
    assert got.status_code == 200
    assert got.json()["content"] == "prefers tempo work"
    assert "value" not in json.dumps(got.json())  # personalization only, never a number (MEM-R1)
    # erase -> the row is gone (residual-row erasure, PRIV-R8)
    erased = await env.client.delete(f"/v1/agent/memory/{item.memory_item_id}", headers=_auth())
    assert erased.status_code == 200
    assert erased.json()["status"] == "erased"
    # PRIV-R8: a re-GET of the erased id is 404 (the residual row is truly gone)
    regot = await env.client.get(f"/v1/agent/memory/{item.memory_item_id}", headers=_auth())
    assert regot.status_code == 404
    assert regot.json()["type"].endswith("/not-found")


async def test_memory_get_foreign_or_unknown_is_404(env: _Env) -> None:
    """A foreign / unknown / non-UUID memory id is 404, never disclosed (MEM-R3 fail-closed)."""
    foreign_id = "22222222-2222-7000-8000-000000000002"
    env.engine.memory = {"other-athlete": [_recalled("foreign secret", item_id=foreign_id)]}
    foreign = await env.client.get(f"/v1/agent/memory/{foreign_id}", headers=_auth())
    assert foreign.status_code == 404  # B's row queried under A is absent, not disclosed
    bad = await env.client.get("/v1/agent/memory/not-a-uuid", headers=_auth())
    assert bad.status_code == 404


async def test_memory_delete_foreign_or_unknown_is_404(env: _Env) -> None:
    """A cross-athlete / unknown per-item delete erases nothing and is 404 (PRIV-R8 fail-closed)."""
    foreign_id = "33333333-3333-7000-8000-000000000003"
    env.engine.memory = {"other-athlete": [_recalled("foreign secret", item_id=foreign_id)]}
    foreign = await env.client.delete(f"/v1/agent/memory/{foreign_id}", headers=_auth())
    assert foreign.status_code == 404
    # B's row still exists (the cross-athlete delete erased nothing).
    assert env.engine.memory["other-athlete"][0].content == "foreign secret"
