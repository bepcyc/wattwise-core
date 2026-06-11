"""CLUSTER A (security/privacy) conformance — REAL production paths, NON-VACUOUS (audit fix).

Each test exercises the REAL wired production path the 14-agent audit found fail-OPEN /
vacuously tested, and is MUTATION-PROOF (revert the fix -> the test fails):

* **ENT (resolve -> attach -> check).** The agent surface is gated THROUGH the REAL
  ``create_app`` wiring (``security.agent_feature_gate`` bound on the agent router), driven by
  the REAL :class:`OssEntitlementResolver` on app state — NOT a stubbed gate. A plan that
  ungrants ``can_use_agent`` (a commercial-style plan, swapped onto the app's resolver) REFUSES
  ``POST /v1/agent/ask`` fail-closed (403); the OSS default plan ALLOWS it (200). A config
  override of the node-visit ceiling is HONORED by the agent run through the carried entitlement
  (ENT-R1-AC), proven against the PRODUCTION ``build_graph`` ceiling-resolution.
* **AUTH-SEC-1 (SEC-R3).** A 4-char ``WATTWISE_TOKEN_SIGNING_KEY`` in a real (production)
  environment fails the boot non-zero through the REAL ``load_settings``; a strong key boots.
* **AUTH-SEC-2 (SEC-R10/.1/.2/-AC).** Wildcard CORS origin + credentials refuses to start; a
  normal response carries the security headers + a scoped CORS header; a spoofed Host is rejected
  by TrustedHostMiddleware when a concrete allowed-host list is configured.
* **PRIV-1 (PRIV-R8).** ``DELETE /v1/users/me`` runs the REAL erasure executor through the wired
  recorder: a re-query shows the athlete's seeded rows GONE across both stores, and a durable
  completion record (the logged ``ErasureReceipt`` counts) exists.
* **ENT-4 (OBS-R6.2).** The readiness probe reports not-ready (503) without a loaded/validated
  default plan, and ready (200) with it.

Real-pool note (skill §7): the DB-touching tests run on FILE-backed SQLite (a real
multi-connection pool), never ``:memory:`` — the erasure + readiness DB checks open fresh
sessions/engines that a single in-memory connection could not model.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine as _create_sync_engine
from sqlalchemy import func, select
from starlette.testclient import TestClient

from tests.integration._schema import provision_app_schema
from wattwise_core.agent.deliverables import AgentAnswer, Citation, Observation
from wattwise_core.agent.graph import DEFAULT_NODE_VISIT_CEILING
from wattwise_core.agent.memory import MemoryItem, MemoryItemKind
from wattwise_core.agent.seams import (
    AgentServices,
    EntitlementCostGate,
    entitlement_node_visit_ceiling,
)
from wattwise_core.agent.state_db import build_agent_state_database
from wattwise_core.agent.state_store import AgentStateBase
from wattwise_core.api.app import create_app
from wattwise_core.api.routers import agent_routes
from wattwise_core.config import ConfigError, Environment, Settings, load_settings
from wattwise_core.entitlement import Entitlements, OssEntitlementResolver
from wattwise_core.identity import OWNER_ATHLETE_ID
from wattwise_core.persistence.models import Activity, Athlete, Base, Sport
from wattwise_core.security.crypto import EnvelopeCipher

pytestmark = pytest.mark.integration

#: A strong (>= 256-bit, high-distinct-byte) first-party owner secret / signing key.
_STRONG_KEY = "real-cluster-a-signing-key-0123456789abcdef"


# --------------------------------------------------------------------------- app harness


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    """REAL dev settings on a FILE DB so the wired engine + recorder share one database file."""
    base: dict[str, Any] = {
        "app__environment": "development",
        "database_dsn": f"sqlite+aiosqlite:///{tmp_path / 'cluster_a.db'}",
        "token_signing_key": _STRONG_KEY,
        "encryption_root_key": EnvelopeCipher.generate_root_key(),
        "object_store__local_root": str(tmp_path / "objects"),
    }
    base.update(overrides)
    return load_settings(**base)


class _FakeAnswerEngine:
    """A minimal grounded-answer engine so ``/v1/agent/ask`` exercises the GATE, not a model.

    The entitlement gate runs as a FastAPI dependency BEFORE the handler body, so when the gate
    refuses (403) this engine is never called; when the gate admits, it returns a grounded answer
    so the 200 path is real. ``athlete_id`` is recorded to prove the run is server-derived.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.seen_athlete_id: str | None = None

    async def answer(self, *, athlete_id: str, **_: Any) -> AgentAnswer:
        """Record the server-derived id and return a stable grounded answer."""
        self.calls += 1
        self.seen_athlete_id = athlete_id
        return AgentAnswer(
            status="completed",  # type: ignore[arg-type]
            thread_id="01THREAD",
            answer_html="<p>You're fresh and ready.</p>",
            answer_text="You're fresh and ready.",
            observations=(Observation(observation_id="01OBS", text="You're recovered."),),
            citations=(Citation(record_id="01CIT", metric="tsb", value=6.2, as_of="2026-06-05"),),
            suggested_followups=("Tell me more",),
        )


def _create_canonical_schema(db_file: Path) -> None:
    """Create the canonical schema on the harness FILE DB (what migrations do in prod).

    ``POST /v1/agent/ask`` now legitimately reads the canonical ``athlete`` row for the
    persisted language default (API-R37), so the harness database must carry the
    canonical schema even when the test seeds no rows.
    """
    sync_engine = _create_sync_engine(f"sqlite:///{db_file}")
    try:
        Base.metadata.create_all(sync_engine)
    finally:
        sync_engine.dispose()


def _app_with_fake_engine(tmp_path: Path, **overrides: Any) -> tuple[TestClient, FastAPI]:
    """The REAL ``create_app`` (real entitlement gate wiring) with ONLY the engine faked.

    Crucially the agent-scope gate is NOT overridden: ``POST /v1/agent/ask`` runs through the
    REAL ``security.agent_feature_gate`` -> ``resolve_entitlement`` -> the app's resolver. Only
    the engine is swapped (so no LLM/network), so a 200 vs 403 is decided by the REAL gate.
    """
    settings = _settings(tmp_path, **overrides)
    _create_canonical_schema(tmp_path / "cluster_a.db")
    app = create_app(settings)
    provision_app_schema(app)  # real schema + stamped head (the readiness gate, RUN-R6)
    engine = _FakeAnswerEngine()
    # Replace ONLY the engine seam; the entitlement gate stays the real wired one.
    app.dependency_overrides[agent_routes.agent_engine] = lambda: engine
    app.state._fake_engine = engine
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    return client, app


def _token(client: TestClient) -> dict[str, str]:
    """Mint a real owner access token via the public token-issuance route (API-R23)."""
    resp = client.post("/v1/auth/token", json={"owner_secret": _STRONG_KEY})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


# ---------------------------------------------------- ENT: resolve -> attach -> check (agent gate)


def test_oss_default_plan_allows_agent_ask(tmp_path: Path) -> None:
    """The OSS all-permissive default plan ADMITS POST /v1/agent/ask through the real gate.

    The gate runs ``resolve_entitlement`` against the REAL ``OssEntitlementResolver`` on app
    state (``can_use_agent=True``), so the agent surface is permitted and the faked engine
    returns a grounded 200 — the resolve -> attach -> check seam permits under OSS.
    """
    client, app = _app_with_fake_engine(tmp_path)
    try:
        resp = client.post("/v1/agent/ask", json={"question": "How am I?"}, headers=_token(client))
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "completed"
        assert app.state._fake_engine.calls == 1  # the gate admitted -> the engine ran
    finally:
        client.__exit__(None, None, None)


def test_plan_ungranting_agent_refuses_ask_fail_closed(tmp_path: Path) -> None:
    """A plan with ``can_use_agent=False`` REFUSES POST /v1/agent/ask fail-closed (AGT-ENT-R3).

    Swaps the app's REAL resolver for one carrying a commercial-style plan that UNGRANTS the
    agent feature (every bound still positive + valid). The request flows through the IDENTICAL
    real gate (``agent_feature_gate`` -> ``resolve_entitlement``) — not a stubbed gate — and is
    refused 403 BEFORE the engine runs. This is the fail-OPEN defect the audit flagged: the gate
    now reads the carried flag and fails closed.
    """
    client, app = _app_with_fake_engine(tmp_path)
    try:
        ungranted = Entitlements(
            can_use_agent=False,
            node_visit_ceiling=60,
            max_output_tokens=8192,
            wall_clock_seconds=120,
            max_tool_iterations=16,
            request_rate_per_minute=120,
        )
        app.state.entitlement_resolver = OssEntitlementResolver(ungranted)
        resp = client.post("/v1/agent/ask", json={"question": "How am I?"}, headers=_token(client))
        assert resp.status_code == 403, resp.text  # fail-closed refusal through the real gate
        assert app.state._fake_engine.calls == 0  # the gate refused BEFORE the engine ran
    finally:
        client.__exit__(None, None, None)


def test_config_override_of_node_ceiling_is_honored_by_agent_run(tmp_path: Path) -> None:
    """A config override of the node-visit ceiling is HONORED by the agent run (ENT-R1-AC).

    Builds the PRODUCTION entitlement resolver from settings whose node-visit-ceiling config is
    overridden to a non-default value, attaches it to the OSS cost gate, and asserts the
    PRODUCTION ceiling-resolution (the SAME function ``build_graph`` calls) reads the OVERRIDE from
    the carried entitlement — not the hardcoded ``DEFAULT_NODE_VISIT_CEILING``. The engine passes
    the module default explicitly, so the entitlement's config value governs the run.
    """
    override = DEFAULT_NODE_VISIT_CEILING + 25  # a value provably distinct from the default
    settings = _settings(tmp_path, **{"entitlement__node_visit_ceiling": override})
    plan = OssEntitlementResolver.from_settings(settings).resolve(str(OWNER_ATHLETE_ID))
    assert plan.node_visit_ceiling == override  # config override carried on the resolved plan
    svc = _services_with_gate(EntitlementCostGate(plan))
    # The production ceiling-resolution reads the OVERRIDE from the entitlement (engine passes
    # the module default explicitly -> the carried bound governs, ENT-R1-AC).
    resolved = entitlement_node_visit_ceiling(
        svc, DEFAULT_NODE_VISIT_CEILING, DEFAULT_NODE_VISIT_CEILING
    )
    assert resolved == override


def _services_with_gate(gate: EntitlementCostGate) -> AgentServices:
    """A minimal ``AgentServices`` carrying ``gate`` (only the cost gate drives the ceiling)."""

    class _Stub:
        async def plan(self, **_: Any) -> list[Any]:
            return []

        async def gather(self, **_: Any) -> dict[str, Any]:
            return {}

        def assess(self, **_: Any) -> set[str]:
            return set()

        async def ground(self, **_: Any) -> Any:  # pragma: no cover - unused for ceiling test
            raise NotImplementedError

    stub = _Stub()
    return AgentServices(
        planner=stub,
        gateway=stub,
        coverage=stub,
        grounder=stub,
        cost_gate=gate,
    )


# ------------------------------------------------------------- AUTH-SEC-1: signing-key entropy


def test_short_signing_key_fails_boot_in_production() -> None:
    """A 4-char WATTWISE_TOKEN_SIGNING_KEY refuses to boot non-zero in production (SEC-R3).

    Through the REAL ``load_settings`` (the production-environment strict gate): a key carrying
    far fewer than 256 bits of entropy raises :class:`ConfigError` so the process exits non-zero
    rather than signing tokens under a guessable key. This is the fail-OPEN the audit flagged.
    """
    with pytest.raises(ConfigError, match="TOKEN_SIGNING_KEY"):
        load_settings(
            app__environment=Environment.PRODUCTION,
            database_dsn="sqlite+aiosqlite:///x.db",
            token_signing_key="abcd",  # 4 chars = 32 bits << the 256-bit floor
            encryption_root_key=EnvelopeCipher.generate_root_key(),
        )


def test_trivially_weak_long_signing_key_fails_boot_in_production() -> None:
    """A long-but-degenerate signing key (one repeated byte) refuses to boot (SEC-R3)."""
    with pytest.raises(ConfigError, match="TOKEN_SIGNING_KEY"):
        load_settings(
            app__environment=Environment.PRODUCTION,
            database_dsn="sqlite+aiosqlite:///x.db",
            token_signing_key="k" * 64,  # 64 bytes but only 1 distinct byte = no real entropy
            encryption_root_key=EnvelopeCipher.generate_root_key(),
        )


def test_strong_signing_key_boots_in_production() -> None:
    """A strong (>= 256-bit, high-entropy) signing key boots cleanly in production (SEC-R3)."""
    settings = load_settings(
        app__environment=Environment.PRODUCTION,
        database_dsn="sqlite+aiosqlite:///x.db",
        token_signing_key=_STRONG_KEY,
        encryption_root_key=EnvelopeCipher.generate_root_key(),
    )
    assert settings.app__environment is Environment.PRODUCTION


# ------------------------------------------------------- AUTH-SEC-2: CORS / headers / host


def test_wildcard_cors_with_credentials_fails_boot() -> None:
    """Wildcard CORS origin '*' + allow_credentials=true refuses to start (SEC-R10-AC).

    The always-insecure configuration cliff is rejected at config load in EVERY environment —
    asserted through the REAL ``load_settings``, so the service can never serve under it.
    """
    with pytest.raises(ConfigError, match="wildcard origin"):
        load_settings(
            app__environment="development",
            database_dsn="sqlite+aiosqlite:///x.db",
            token_signing_key=_STRONG_KEY,
            security__cors_allow_origins=["*"],
            security__cors_allow_credentials=True,
        )


def test_response_carries_security_headers_and_scoped_cors(tmp_path: Path) -> None:
    """A normal response carries the security headers + a SCOPED (non-wildcard) CORS origin.

    Drives a real request through the assembled app's middleware stack: the security-headers
    middleware attaches HSTS / nosniff / Referrer-Policy / CSP (SEC-R10.1), and CORS echoes the
    configured concrete allowed origin (SEC-R10) — never the wildcard.
    """
    origin = "https://app.example.test"
    client, _app = _app_with_fake_engine(
        tmp_path,
        **{
            "security__cors_allow_origins": [origin],
            "security__cors_allow_credentials": True,
        },
    )
    try:
        resp = client.get("/v1/system/status", headers={"Origin": origin})
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("referrer-policy") == "no-referrer"
        assert "max-age=" in resp.headers.get("strict-transport-security", "")
        assert resp.headers.get("content-security-policy")  # a restrictive CSP is present
        # CORS echoes the configured CONCRETE origin (scoped), never '*'.
        assert resp.headers.get("access-control-allow-origin") == origin
    finally:
        client.__exit__(None, None, None)


def test_spoofed_host_is_rejected_when_allowed_hosts_configured(tmp_path: Path) -> None:
    """A request with a Host NOT on the configured allowed-host list is rejected (SEC-R10.2).

    With a concrete ``security.allowed_hosts`` (the production posture), TrustedHostMiddleware
    rejects a spoofed ``Host`` header with 400 — the host-header-attack guard. The configured
    host is accepted; an off-list host is refused.
    """
    client, _app = _app_with_fake_engine(
        tmp_path, **{"security__allowed_hosts": ["good.example.test"]}
    )
    try:
        ok = client.get("/v1/system/status", headers={"Host": "good.example.test"})
        assert ok.status_code == 200, ok.text
        spoofed = client.get("/v1/system/status", headers={"Host": "evil.example.test"})
        assert spoofed.status_code == 400, spoofed.text  # TrustedHostMiddleware refuses
    finally:
        client.__exit__(None, None, None)


# ----------------------------------------------------------- ENT-4: readiness probe (OBS-R6.2)


def test_readiness_not_ready_without_loaded_plan(tmp_path: Path) -> None:
    """The readiness probe reports not-ready (503) without a loaded/validated default plan.

    Drops the validated plan off app state (simulating an instance that has not loaded/validated
    the default plan) and asserts the probe returns 503 ``not_ready`` — there is no fake-healthy
    ready before the resolver+plan checks pass (OBS-R6.2 / ENT-R6). With the plan present it is
    200 ``ready``.
    """
    client, app = _app_with_fake_engine(tmp_path)
    try:
        ready = client.get("/v1/health/ready")
        assert ready.status_code == 200, ready.text
        assert ready.json()["status"] == "ready"
        # Now remove the loaded plan -> the probe must report not-ready (503).
        app.state.entitlement_plan = None
        not_ready = client.get("/readyz")
        assert not_ready.status_code == 503, not_ready.text
        assert not_ready.json()["status"] == "not_ready"
        assert not_ready.json()["checks"]["default_plan_loaded"] is False
    finally:
        client.__exit__(None, None, None)


# ----------------------------------------------------------------- PRIV-1: real erasure on DELETE


def _make_seed(app: FastAPI, settings: Settings) -> Any:
    """Build the async seeder: the OWNER + a canonical activity + an agent-state memory row.

    Two stores get a row apiece (PRIV-1 fixture) so the erasure must clear BOTH: a canonical
    ``activity`` and a durable ``agent_memory_item`` — proving the executor spans both stores.
    """
    database = app.state.database

    async def _seed() -> None:
        async with database.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with database.session() as s:
            s.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
            await s.flush()
            s.add(
                Athlete(
                    athlete_id=OWNER_ATHLETE_ID,
                    sex="male",
                    reference_timezone="UTC",
                    current_sport="cycling",
                )
            )
            await s.flush()
            s.add(
                Activity(
                    athlete_id=OWNER_ATHLETE_ID,
                    start_time=_dt.datetime(2026, 6, 1, 8, 0, tzinfo=_dt.UTC),
                    sport="cycling",
                )
            )
        state_db = build_agent_state_database(settings)
        await state_db.create_all()
        async with state_db.session() as s:
            s.add(
                MemoryItem(
                    memory_item_id=uuid.UUID("aaaaaaaa-aaaa-7000-8000-000000000099"),
                    athlete_id=OWNER_ATHLETE_ID,
                    kind=MemoryItemKind.PREFERENCE,
                    content="prefers morning rides",
                    inferred=False,
                )
            )
        await state_db.dispose()

    return _seed


def test_delete_me_runs_real_erasure_and_removes_rows(tmp_path: Path) -> None:
    """DELETE /v1/users/me runs the REAL erasure executor; a re-query shows the rows GONE (PRIV-1).

    Builds the REAL app (the wired ``build_deletion_requester`` recorder), seeds the OWNER with
    canonical rows (activity + wellness) AND an agent-state memory row, then DELETEs the account.
    The endpoint returns the async ``pending_deletion`` ack and the recorder runs the production
    ``erase_athlete`` across BOTH stores — so a re-query of the canonical + agent-state tables
    finds ZERO residual rows for the athlete. This is the log-only no-op the audit flagged: the
    erasure now ACTUALLY happens.
    """
    settings = _settings(tmp_path)
    app = create_app(settings)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    try:
        client.portal.call(_make_seed(app, settings))  # type: ignore[union-attr]
        auth = _token(client)

        async def _count_activities() -> int:
            async with app.state.database.session() as s:
                n = await s.scalar(
                    select(func.count())
                    .select_from(Activity)
                    .where(Activity.athlete_id == OWNER_ATHLETE_ID)
                )
            return int(n or 0)

        async def _count_memory() -> int:
            state_db = build_agent_state_database(settings)
            try:
                mem_table = AgentStateBase.metadata.tables["agent_memory_item"]
                async with state_db.session() as s:
                    n = await s.scalar(
                        select(func.count())
                        .select_from(mem_table)
                        .where(mem_table.c["athlete_id"] == OWNER_ATHLETE_ID)
                    )
                return int(n or 0)
            finally:
                await state_db.dispose()

        before = (
            client.portal.call(_count_activities),  # type: ignore[union-attr]
            client.portal.call(_count_memory),  # type: ignore[union-attr]
        )
        assert before == (1, 1), "fixture seeded one canonical + one agent-state row"

        resp = client.delete("/v1/users/me", headers=auth)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "pending_deletion"  # async ack

        # PRIV-1: the REAL executor ran -> ZERO residual rows across BOTH stores.
        after = (
            client.portal.call(_count_activities),  # type: ignore[union-attr]
            client.portal.call(_count_memory),  # type: ignore[union-attr]
        )
        assert after == (0, 0), "erasure removed every row across both stores (PRIV-R8)"
    finally:
        client.__exit__(None, None, None)
