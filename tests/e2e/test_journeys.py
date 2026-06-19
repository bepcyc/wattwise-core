"""API-level Phase-1 end-to-end journeys on the BUILT stack (E2E-R1 a-d / DOD-R5).

The OSS engine ships no GUI client (the web app + Telegram bot are commercial ``athload``,
ROAD-R1), so the Phase-1 E2E journeys are exercised through the assembled REST API — the
REAL :func:`wattwise_core.api.app.create_app` app with its real seam wiring (auth, the
analytics service, the file-upload import processor, the on-demand sync orchestrator, the
credential store) over a file-backed SQLite database that all request-scoped sessions
share. The single external boundary is the LLM: the coaching agent is driven by a
deterministic, network-free :class:`FakeModel` so the grounded-answer journey is
reproducible (the model never self-certifies — code grounds every claim, OUTCOME-R5).

The four required Phase-1 journeys (E2E-R1):

- **(a) connect → sync.** Upload an activity file (``POST /v1/imports``) — the OSS
  file-upload "connect", which lands a connectionless ``file_import`` canonical activity
  through the real ingest path — and trigger a manual sync (``POST /v1/sync/run``). The
  uploaded ride then appears on the canonical analytics surface.
- **(b) headline metric.** Read the Performance Management Chart (``GET
  /v1/performance/load-fitness``) and a headline load metric (``GET
  /v1/performance/coggan``) over the seeded canonical activities.
- **(c) grounded API ask.** Ask the agent a question (``POST /v1/agent/ask``) and receive
  a grounded, status-discriminated answer; a fabricated number fails closed to ``degraded``
  rather than shipping an unverified value (API-R12 / GROUND-R6).
- **(d) token-issuance → grounded query.** Mint a first-party token via the real
  ``POST /v1/auth/token`` exchange and drive a grounded query through the same canonical
  API path with it — proving any client (the commercial bot included) can consume the
  surface and that the server-derived subject is the single owner athlete (AUTH-R3/R18).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Iterator
from pathlib import Path

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import event

from wattwise_core.agent.contracts import ClaimKind, ReflectDecision, ReflectVerdict
from wattwise_core.agent.engine import (
    GraphAgentEngine,
    _ClaimSchema,
    _ExtractedClaim,
    _PlanSchema,
)
from wattwise_core.agent.model import FakeModel
from wattwise_core.agent.state_db import AgentStateDatabase, build_agent_state_database
from wattwise_core.api.app import create_app
from wattwise_core.api.auth import Scope, require_scopes
from wattwise_core.api.deps import get_rate_limiter
from wattwise_core.api.routers import agent_routes
from wattwise_core.api.routers import planning as planning_routes
from wattwise_core.config import Settings, load_settings
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
from wattwise_core.security.crypto import EnvelopeCipher
from wattwise_core.storage import content_hash

pytestmark = pytest.mark.e2e

UTC = _dt.UTC

#: The first-party owner sign-in secret (the configured ``token_signing_key``, API-R23).
_SIGNING_KEY = "e2e-owner-secret-0123456789abcdef0123456789"

#: A reference "today" the seeded rides sit just behind, so the agent's trailing-window
#: plan (today minus 42 days) covers them and the PMC range is populated.
_TODAY = _dt.date(2026, 6, 8)
_RIDE_DAYS = (_dt.date(2026, 6, 1), _dt.date(2026, 6, 2), _dt.date(2026, 6, 3))

#: The committed file-upload fixture decoded by the real import processor (journey a).
_RIDE_FIT = (
    Path(__file__).resolve().parents[1] / "contract" / "fixtures" / "file_upload" / "ride.fit"
)


def _completed_model() -> FakeModel:
    """A FakeModel scripting a grounded, COMPLETED weekly-load answer (the default path).

    The planner selects ``weekly_load`` (so the gather resolves canonical PMC); the claim
    extractor points at one non-prescriptive state observation, which publishes as a
    grounded ``complementary`` (GROUND-R9) — a number-free, leads-with-state coach answer.
    """
    return FakeModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.STATEMENT,
                        text="your training is trending in a good direction",
                    )
                ]
            ),
            # Script the reflect verdict too: the REFLECT-R4 fall-through (an exhausted REGENERATE
            # re-plans while reflection budget remains, §225/§451) can now reach the ``reflect``
            # node on a perpetually-contradicted draft, so the structured verdict must be present.
            "ReflectDecision": ReflectDecision(verdict=ReflectVerdict.GIVE_UP_GRACEFULLY),
        },
        prose="Your training is trending in a good direction this week — nice, steady work.",
    )


class _Journey:
    """The assembled client + the scriptable model for one E2E scenario."""

    def __init__(self, client: TestClient, app: FastAPI, model: FakeModel, token: str) -> None:
        self.client = client
        self.app = app
        self.model = model
        self.token = token

    @property
    def auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


@pytest.fixture
def journey(tmp_path: Path) -> Iterator[_Journey]:
    """Build the REAL app on a shared file DB, seed canonical data, and mint a real token.

    A temp-file DSN means every request-scoped session (the analytics read, the import
    write, the agent read) shares ONE database — unlike ``:memory:``. The agent engine seam
    is overridden with a :class:`FakeModel`-backed :class:`GraphAgentEngine` (the only
    external boundary doubled); everything else is the real wiring.
    """
    settings = _settings(tmp_path)
    app = create_app(settings)
    model = _completed_model()
    app.dependency_overrides[agent_routes.agent_engine] = lambda: GraphAgentEngine(
        app.state.database, model
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.portal.call(_seed, app)  # type: ignore[union-attr]
        token = _issue_owner_token(client)
        yield _Journey(client, app, model, token)


def _settings(tmp_path: Path) -> Settings:
    """Dev settings on a file DB with a real envelope key (the full built-stack wiring)."""
    return load_settings(
        app__environment="development",
        database_dsn=f"sqlite+aiosqlite:///{tmp_path / 'e2e.db'}",
        token_signing_key=_SIGNING_KEY,
        encryption_root_key=EnvelopeCipher.generate_root_key(),
        object_store__local_root=str(tmp_path / "objects"),
    )


async def _seed(app: FastAPI) -> None:
    """Create the schema + seed the owner, registries, an FTP signature, and rides.

    Mirrors what the initial migration seeds (the single owner athlete + the file-import
    descriptor) plus the per-athlete data a fresh sign-in would have (an FTP signature and
    a few canonical rides) so the PMC/headline reads are populated.
    """
    database = app.state.database
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with database.session() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        session.add(Athlete(athlete_id=OWNER_ATHLETE_ID, sex="male", reference_timezone="UTC"))
        descriptor = SourceDescriptor(
            source_key="file_import", display_name="Activity files", kind="file_upload"
        )
        session.add(descriptor)
        session.add(
            SourceDescriptor(
                source_key="intervals_icu", display_name="Intervals.icu", kind="oauth_api"
            )
        )
        session.add(
            FitnessSignature(
                athlete_id=OWNER_ATHLETE_ID,
                signature_type="cycling",
                effective_date=_dt.date(2024, 1, 1),
                ftp_w=250.0,
                origin=SignatureOrigin.MEASURED,
            )
        )
        await session.flush()
        ingest = IngestService(session)
        for i, day in enumerate(_RIDE_DAYS):
            await ingest.ingest(
                str(OWNER_ATHLETE_ID),
                str(descriptor.source_descriptor_id),
                [_ride_candidate(f"seed-{i}", day)],
            )


def _ride_candidate(native_id: str, day: _dt.date) -> GboCandidate:
    """A deterministic constant-250 W, 1 h cycling ride (TSS == 100 at FTP 250)."""
    seconds = 3600
    watts = 250.0
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
        fetched_at=_dt.datetime(2026, 6, 4, 9, 0, tzinfo=UTC),
    )


def _issue_owner_token(client: TestClient) -> str:
    """Exchange the first-party owner secret for an access token (the real route, API-R23)."""
    resp = client.post("/v1/auth/token", json={"owner_secret": _SIGNING_KEY})
    assert resp.status_code == 200, resp.text
    token: str = resp.json()["access_token"]
    return token


# --- (a) connect → sync ----------------------------------------------------------


def test_journey_a_connect_sync_lands_canonical_data(journey: _Journey) -> None:
    """Upload an activity file + trigger a sync; the ride lands on the canonical surface."""
    # Before the import, the 2024 window holds nothing (the seeded rides are in 2026).
    before = journey.client.get(
        "/v1/performance/coggan",
        params={"from": "2024-01-01", "to": "2024-01-03"},
        headers=journey.auth,
    )
    assert before.status_code == 200
    assert before.json()["items"] == []

    upload = journey.client.post(
        "/v1/imports",
        headers=journey.auth,
        files={"file": ("ride.fit", _RIDE_FIT.read_bytes(), "application/octet-stream")},
    )
    assert upload.status_code == 202, upload.text
    # The OSS import ingests SYNCHRONOUSLY in-request, so the job is already TERMINAL by the
    # time the 202 lands: it reads "done", never a stranded "queued" (API-R33a, #115).
    assert upload.json()["status"] == "done"

    # A manual sync is the only OSS trigger (API-R46). The file upload is CONNECTIONLESS
    # (LIN-R1.1) — it created no connection — so there is no connected source to pull from:
    # the run honestly reports "nothing_to_sync" rather than falsely claiming a sync is
    # happening (API-R46c, #118). The upload's data already landed via the import path above.
    run = journey.client.post("/v1/sync/run", headers=journey.auth)
    assert run.status_code == 202, run.text
    assert run.json()["status"] == "nothing_to_sync"
    assert run.json()["status_text"] != "We're pulling in your latest training now."

    # The uploaded ride (2024-01-02) is now a canonical activity on the analytics surface.
    after = journey.client.get(
        "/v1/performance/coggan",
        params={"from": "2024-01-01", "to": "2024-01-03"},
        headers=journey.auth,
    )
    assert after.status_code == 200
    items = after.json()["items"]
    assert len(items) == 1, "exactly the one uploaded activity must appear (no double-land)"


# --- (b) headline metric ---------------------------------------------------------


def test_journey_b_pmc_and_headline_metric(journey: _Journey) -> None:
    """The PMC and a headline load metric read the seeded canonical activities (API-R30)."""
    pmc = journey.client.get(
        "/v1/performance/load-fitness",
        params={"from": "2026-05-25", "to": "2026-06-08"},
        headers=journey.auth,
    )
    assert pmc.status_code == 200, pmc.text
    body = pmc.json()
    assert body["method"] == "pmc_ewma"
    assert body["summary"]["fitness"] is not None and body["summary"]["fitness"] > 0

    coggan = journey.client.get(
        "/v1/performance/coggan",
        params={"from": "2026-06-01", "to": "2026-06-03"},
        headers=journey.auth,
    )
    assert coggan.status_code == 200, coggan.text
    tss_values = [it["values"]["tss"] for it in coggan.json()["items"]]
    assert any(v is not None and abs(v - 100.0) < 1.0 for v in tss_values), tss_values


# --- (c) grounded API ask --------------------------------------------------------


def _live_fitness(journey: _Journey) -> float:
    """The seeded athlete's canonical current fitness (CTL) read off the PMC surface."""
    pmc = journey.client.get(
        "/v1/performance/load-fitness",
        params={"from": "2026-05-25", "to": "2026-06-08"},
        headers=journey.auth,
    )
    ctl = pmc.json()["summary"]["fitness"]
    assert ctl is not None and ctl > 0
    return float(ctl)


def test_journey_c_grounded_agent_ask(journey: _Journey) -> None:
    """A grounded question returns a COMPLETED, grounded answer through the agent surface.

    The scripted claim states the CANONICAL current fitness so the run completes WITH a grounded
    citation (STATUS-R1: a data-grounded answer needs grounded substance to complete — the
    number-free completed answer this journey once pinned was the issue-44 defect).
    """
    ctl = _live_fitness(journey)
    journey.model.set_response(
        _ClaimSchema(
            claims=[
                _ExtractedClaim(
                    kind=ClaimKind.NUMBER,
                    text=f"your fitness is around {ctl:.1f}",
                    metric="ctl",
                    value=ctl,
                    as_of="2026-06-08",
                )
            ]
        )
    )
    resp = journey.client.post(
        "/v1/agent/ask", json={"question": "How is my training going?"}, headers=journey.auth
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert body["grounding"]["grounded"] is True
    assert body["grounding"]["citations"], "a completed data-grounded answer carries a citation"
    assert body["answer_text"].strip()
    assert "<p>" in body["answer_html"]


def test_journey_c_grounded_number_carries_a_citation(journey: _Journey) -> None:
    """A number that matches canonical data grounds and ships WITH its citation (GROUND-R5/R7).

    Reads the current canonical fitness (CTL) off the PMC surface, scripts the agent to claim
    exactly that value as-of the same date, and asserts the API answer is ``completed`` and
    carries a grounding citation for ``ctl`` — exercising the numeric-grounding fix end to end.
    """
    pmc = journey.client.get(
        "/v1/performance/load-fitness",
        params={"from": "2026-05-25", "to": "2026-06-08"},
        headers=journey.auth,
    )
    ctl = pmc.json()["summary"]["fitness"]
    assert ctl is not None and ctl > 0
    journey.model.set_response(
        _ClaimSchema(
            claims=[
                _ExtractedClaim(
                    kind=ClaimKind.NUMBER,
                    text=f"your fitness is around {ctl:.1f}",
                    metric="ctl",
                    value=ctl,
                    as_of="2026-06-08",
                )
            ]
        )
    )
    resp = journey.client.post(
        "/v1/agent/ask", json={"question": "What's my fitness number?"}, headers=journey.auth
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    citations = body["grounding"]["citations"]
    assert citations, "a grounded number must reach the API answer with a citation (GROUND-R5)"
    assert citations[0]["metric"] == "ctl"
    assert abs(citations[0]["value"] - ctl) < 1e-6  # verbatim canonical value (GROUND-R7)


def test_journey_c_fabricated_number_fails_closed_to_degraded(journey: _Journey) -> None:
    """A model-invented number can never ship: it scrubs out and the run degrades (GROUND-R6)."""
    journey.model.set_response(
        _ClaimSchema(
            claims=[
                _ExtractedClaim(
                    kind=ClaimKind.NUMBER, text="your CTL is 999", metric="ctl", value=999.0
                )
            ]
        )
    )
    resp = journey.client.post(
        "/v1/agent/ask", json={"question": "What exactly is my CTL?"}, headers=journey.auth
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "degraded"
    assert "999" not in body["answer_text"]  # the fabricated number never reaches the athlete


# --- (d) token-issuance → grounded query -----------------------------------------


def test_journey_d_token_issuance_drives_a_grounded_query(journey: _Journey) -> None:
    """The real token exchange yields the owner subject and drives a grounded query (E2E-R1d)."""
    # The minted token's subject is the single owner athlete id — a real UUID, server-derived
    # (AUTH-R3/R18), NOT a placeholder string; this is what makes every canonical read resolve.
    claims = jwt.decode(journey.token, options={"verify_signature": False})
    assert claims["sub"] == OWNER_SUBJECT
    assert uuid.UUID(claims["sub"]) == OWNER_ATHLETE_ID

    # The SAME token drives a grounded query through the canonical API path. The scripted claim
    # states the canonical fitness so the run completes WITH grounded substance (STATUS-R1).
    ctl = _live_fitness(journey)
    journey.model.set_response(
        _ClaimSchema(
            claims=[
                _ExtractedClaim(
                    kind=ClaimKind.NUMBER,
                    text=f"your fitness is around {ctl:.1f}",
                    metric="ctl",
                    value=ctl,
                    as_of="2026-06-08",
                )
            ]
        )
    )
    resp = journey.client.post(
        "/v1/agent/ask", json={"question": "Give me a quick read on my form."}, headers=journey.auth
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"

    # And the canonical analytics surface is reachable with the very same issued token.
    pmc = journey.client.get(
        "/v1/performance/load-fitness",
        params={"from": "2026-05-25", "to": "2026-06-08"},
        headers=journey.auth,
    )
    assert pmc.status_code == 200


# --- (E2E-R1a) approval-gated multi-day PLAN → durable HITL resume ----------------
#
# The ROAD-R2 journey (doc-80:256 / DOD-R5): over the BUILT app, a FakeModel scripted to a
# multi-day PLAN whose prescribed workout name is canonical (so it GROUNDS, not scrubbed)
# pauses at the durable interrupt-gate; the EXISTING decision endpoint resumes the SAME
# durable thread from the checkpoint WITHOUT recomputation — approve finalizes, reject is
# not adopted, an edit re-grounds. The planning engine and the decision engine are the SAME
# GraphAgentEngine over ONE durable state store (the only thing that makes resume possible).


class _CountingModel(FakeModel):
    """A :class:`FakeModel` that counts model calls, to PROVE resume does not recompute.

    The plan body is composed + the claims extracted exactly once, BEFORE the interrupt pauses
    the run. A decision (``approve``/``reject``) drives ``Command(resume)`` over the durable
    checkpoint, replaying the pre-interrupt nodes from state — so it must add ZERO further
    ``compose``/``structured`` calls (CKPT-R2 no-recompute). An ``edit`` is the sole exception:
    it RE-GROUNDS the edited body (one extra ``structured`` extraction), by design (GROUND-R3).
    """

    def __init__(self, *, scripted: dict[str, BaseModel], prose: str) -> None:
        super().__init__(scripted=scripted, prose=prose)
        self.compose_calls = 0
        self.structured_calls = 0

    async def compose(self, *, system: str, context: str, max_tokens: int = 1024) -> str:
        self.compose_calls += 1
        return await super().compose(system=system, context=context, max_tokens=max_tokens)

    async def structured[M: BaseModel](self, *, system: str, data: str, schema: type[M]) -> M:
        self.structured_calls += 1
        return await super().structured(system=system, data=data, schema=schema)


def _plan_model() -> _CountingModel:
    """A model scripting a grounded, approval-gated MULTI-DAY plan (canonical workout name).

    The ``NAME`` claim names ``endurance ride`` — a CANONICAL workout (it grounds, GROUND-R2),
    so the paused plan body is non-empty and grounded; the prose spans multiple days. The
    counting wrapper lets the journey assert the durable resume adds no model work.
    """
    return _CountingModel(
        scripted={
            "_PlanSchema": _PlanSchema(capabilities=["weekly_load"], window_days=42),
            "_ClaimSchema": _ClaimSchema(
                claims=[
                    _ExtractedClaim(
                        kind=ClaimKind.NAME, text="endurance ride", as_of="endurance ride"
                    ),
                    _ExtractedClaim(kind=ClaimKind.STATEMENT, text="build aerobic base"),
                ]
            ),
        },
        prose="Day 1: endurance ride. Day 2: rest day. Day 3: endurance ride. Build your base.",
    )


class _PlanJourney(_Journey):
    """A :class:`_Journey` plus the ONE shared engine driving both plan + decision (E2E-R1a)."""

    def __init__(
        self,
        client: TestClient,
        app: FastAPI,
        model: _CountingModel,
        token: str,
        engine: GraphAgentEngine,
    ) -> None:
        super().__init__(client, app, model, token)
        self.counting = model
        self.engine = engine


def _wal(dbapi_conn: object, _record: object) -> None:
    """WAL + busy_timeout per connection so the durable saver runs on a REAL pool (skill §7)."""
    cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


async def _make_state_db(tmp_path: Path) -> AgentStateDatabase:
    """A DEDICATED agent-state store over a file-SQLite REAL pool (NEVER ``:memory:``, skill §7)."""
    state_db = build_agent_state_database(dsn=f"sqlite+aiosqlite:///{tmp_path / 'agent.sqlite'}")
    event.listen(state_db.engine.sync_engine, "connect", _wal)
    await state_db.create_all()
    return state_db


def _wire_plan_seams(app: FastAPI, engine: GraphAgentEngine) -> None:
    """Mount the planning router (idempotent) + point BOTH engine seams at the ONE engine.

    The plan-generation seam (``planning_routes.planning_engine``) and the decision seam
    (``agent_routes.agent_engine``) MUST resolve to the SAME engine over the SAME durable store,
    or the interrupt the plan recorded would be invisible to the decision endpoint. The planning
    router's identity/scope/limiter reuse the agent router's already-wired server-derived seams.
    """
    if "/v1/planning/workouts" not in {r.path for r in app.routes}:  # type: ignore[attr-defined]
        app.include_router(planning_routes.router)
    overrides = app.dependency_overrides
    overrides[agent_routes.agent_engine] = lambda: engine
    overrides[planning_routes.planning_engine] = lambda: engine
    overrides[planning_routes.require_agent_scope] = require_scopes(Scope.AGENT)
    overrides[planning_routes.current_athlete_id] = overrides[agent_routes.current_athlete_id]
    overrides[planning_routes.rate_limiter] = get_rate_limiter


@pytest.fixture
def plan_journey(tmp_path: Path) -> Iterator[_PlanJourney]:
    """The BUILT app whose plan + decision endpoints share ONE durable-saver engine (E2E-R1a).

    Mirrors :func:`journey` (real app, shared file DB, real token), but the agent engine is the
    SAME instance behind both the planning and the agent/decision surfaces, over a dedicated
    file-SQLite agent-state pool — so a paused plan resumes from its durable checkpoint.
    """
    settings = _settings(tmp_path)
    app = create_app(settings)
    model = _plan_model()
    with TestClient(app, raise_server_exceptions=False) as client:
        client.portal.call(_seed, app)  # type: ignore[union-attr]
        state_db = client.portal.call(_make_state_db, tmp_path)  # type: ignore[union-attr]
        engine = GraphAgentEngine(app.state.database, model, state_db=state_db)
        _wire_plan_seams(app, engine)
        token = _issue_owner_token(client)
        yield _PlanJourney(client, app, model, token, engine)


def _generate_plan(journey: _PlanJourney) -> dict[str, object]:
    """POST a plan request and assert it paused ``awaiting_approval``, grounded (API-R12a)."""
    resp = journey.client.post(
        "/v1/planning/workouts",
        json={"request": "Give me a multi-day training plan for this week."},
        headers=journey.auth,
    )
    assert resp.status_code == 200, resp.text
    body: dict[str, object] = resp.json()
    assert body["status"] == "awaiting_approval"
    assert body["interrupt_id"], "a paused plan must carry the interrupt_id the decision consumes"
    assert body["grounding"]["grounded"] is True  # type: ignore[index]
    text = str(body["plan_text"])
    assert "endurance ride" in text, "the canonical workout name must ground into the plan body"
    return body


def test_e2e_plan_approve_resumes_from_durable_checkpoint(plan_journey: _PlanJourney) -> None:
    """E2E-R1a: a paused PLAN approve resumes the durable thread to ``completed``, NO recompute.

    The plan composes + grounds its body ONCE before the interrupt; ``approve`` drives the resume
    over the durable checkpoint and finalizes ``completed`` WITHOUT re-invoking the model (delta
    compose == delta structured == 0) — the run picked up from the saved checkpoint, not a rerun
    (CKPT-R2 / DOD-R5).
    """
    plan = _generate_plan(plan_journey)
    before = (plan_journey.counting.compose_calls, plan_journey.counting.structured_calls)

    resp = plan_journey.client.post(
        f"/v1/agent/threads/{plan['thread_id']}/decision",
        json={"interrupt_id": plan["interrupt_id"], "decision": "approve"},
        headers=plan_journey.auth,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert body["decision"] == "approve"
    assert "endurance ride" in body["plan_text"]  # the same grounded body, resumed verbatim
    after = (plan_journey.counting.compose_calls, plan_journey.counting.structured_calls)
    assert after == before, "approve must resume from the durable checkpoint, never recompute"

    # The interrupt is consumed: a second decision is refused 409 (never resumed twice, CKPT-R9).
    again = plan_journey.client.post(
        f"/v1/agent/threads/{plan['thread_id']}/decision",
        json={"interrupt_id": plan["interrupt_id"], "decision": "approve"},
        headers=plan_journey.auth,
    )
    assert again.status_code == 409
    assert again.json()["type"].endswith("/decision-conflict")


def test_e2e_plan_reject_resumes_without_adopting(plan_journey: _PlanJourney) -> None:
    """E2E-R1a (reject): a ``reject`` resumes the durable thread un-approved and adopts no plan.

    Reject is still a durable resume (it finalizes + consumes the interrupt so it can't be acted on
    twice), but the plan is NOT adopted: no active plan is persisted to the canonical schedule, so
    the read-only schedule surface stays empty (API-R32 / CKPT-R9).
    """
    plan = _generate_plan(plan_journey)
    before = (plan_journey.counting.compose_calls, plan_journey.counting.structured_calls)

    resp = plan_journey.client.post(
        f"/v1/agent/threads/{plan['thread_id']}/decision",
        json={"interrupt_id": plan["interrupt_id"], "decision": "reject"},
        headers=plan_journey.auth,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] in ("completed", "degraded")
    assert resp.json()["decision"] == "reject"
    after = (plan_journey.counting.compose_calls, plan_journey.counting.structured_calls)
    assert after == before, "reject must resume from the durable checkpoint, never recompute"

    # Not adopted: the canonical schedule surface holds no plan day from a rejected plan.
    schedule = plan_journey.client.get(
        "/v1/planning/schedule",
        params={"from": "2026-06-01", "to": "2026-06-30"},
        headers=plan_journey.auth,
    )
    assert schedule.status_code == 200, schedule.text
    assert schedule.json()["plan_id"] is None
    assert schedule.json()["days"] == []


def test_e2e_plan_edit_regrounds_then_completes(plan_journey: _PlanJourney) -> None:
    """E2E-R1a (edit): an ``edit`` naming a CANONICAL workout re-grounds and is adopted (GROUND-R3).

    The edited body names ``recovery ride`` (canonical), so the engine RE-GROUNDS it (one extra
    structured extraction, by design) and the resumed plan carries the edited, re-grounded body —
    the un-edited original is replaced only because the edit fully grounded.
    """
    plan = _generate_plan(plan_journey)
    # The re-grounding extracts a NAME claim per workout the edited body prescribes.
    plan_journey.counting.set_response(
        _ClaimSchema(
            claims=[
                _ExtractedClaim(kind=ClaimKind.NAME, text="recovery ride", as_of="recovery ride")
            ]
        )
    )
    resp = plan_journey.client.post(
        f"/v1/agent/threads/{plan['thread_id']}/decision",
        json={
            "interrupt_id": plan["interrupt_id"],
            "decision": "edit",
            "edited_plan": "Day 1: recovery ride to build aerobic base.",
        },
        headers=plan_journey.auth,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert body["decision"] == "edit"
    assert "recovery ride" in body["plan_text"]  # the re-grounded edit body is adopted (GROUND-R3)


def test_e2e_plan_edit_with_invented_workout_fails_closed(plan_journey: _PlanJourney) -> None:
    """E2E-R1a (edit guard): an edit whose workout does NOT ground degrades, never shipping it.

    The edited body names an INVENTED workout (``magic super workout``); re-grounding scrubs it, so
    the edit does not fully ground — the run resolves ``degraded`` and the unverified edit text
    NEVER reaches the athlete (the pre-edit grounded plan is delivered instead, GROUND-R3 / H3).
    """
    plan = _generate_plan(plan_journey)
    plan_journey.counting.set_response(
        _ClaimSchema(
            claims=[
                _ExtractedClaim(
                    kind=ClaimKind.NAME, text="magic super workout", as_of="magic super workout"
                )
            ]
        )
    )
    resp = plan_journey.client.post(
        f"/v1/agent/threads/{plan['thread_id']}/decision",
        json={
            "interrupt_id": plan["interrupt_id"],
            "decision": "edit",
            "edited_plan": "Day 1: magic super workout.",
        },
        headers=plan_journey.auth,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "degraded", "a non-fully-grounded edit must fail closed, never ship"
    assert "magic super workout" not in body["plan_text"]  # the invented workout never reaches out
