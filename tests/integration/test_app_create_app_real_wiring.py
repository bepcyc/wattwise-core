"""The REAL ``create_app()`` serves every ROAD-R2-EXIT endpoint — NO manual seam overrides.

This is the production-wiring guard the unit/router tests could NOT provide: those drive each new
``/v1/agent`` breadth / ``/v1/users`` / ``/v1/planning`` endpoint through the router's *override
seams* with a fake engine + an injected session, so a router that 500s in the REAL
:func:`wattwise_core.api.app.create_app` (because the factory never wired that seam) still passes
them. Here the app is assembled by the factory ALONE — its real ``_wire_router_seams`` is the only
wiring — and a real first-party owner token (``POST /v1/auth/token``) drives each endpoint over a
shared file-sqlite DB. The single assertion that matters: NONE of the advertised endpoints returns
``500``; each returns its spec status (200/202/404/422).

The 500s this would have caught (each a seam the factory failed to bind):

- **H1** — ``agent_breadth.current_session`` unwired -> ``POST /v1/agent/digest/subscribe`` /
  ``GET /v1/agent/digest/list`` / ``DELETE …/digest/subscribe/{id}`` fell through to the fail-closed
  default and 500'd.
- **H2** — on a no-LLM deployment the factory bound an :class:`UnconfiguredAgentEngine` that only
  implemented ``answer``/``readiness``, so ``POST /v1/agent/diagnose`` and ``GET/DELETE
  /v1/agent/memory`` 500'd (no ``diagnose``/``list_memory``/``get_memory``/``delete_memory``).
- **H4** — ``users.deletion_requester`` unwired -> ``DELETE /v1/users/me`` hit its fail-closed
  default and 500'd.

The two variants pin the same surface with and WITHOUT an LLM key. The no-LLM variant is the one the
findings target: it binds the :class:`UnconfiguredAgentEngine`, and it MUST still serve the NON-LLM
surfaces — the deterministic ``diagnose`` (API-R15), the per-item memory read/erase (MEM-R3 /
PRIV-R8, a privacy MUST that can never require a model — the erase actually removes the row so a
re-GET is 404), and a degraded (never-500) ``digest`` — while plan generation phase-gates to a
typed ``degraded`` answer (RUN-R4.1). Identity is server-derived from the real token throughout
(AUTH-R3); no seam is overridden.

Requirement IDs: API-R4, API-R14, API-R15, API-R15a, API-R32, AUTH-R3, MEM-R3, PRIV-R8, RUN-R4.1.
"""

from __future__ import annotations

import datetime as _dt
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from wattwise_core.agent.memory import MemoryItem, MemoryItemKind
from wattwise_core.agent.state_db import build_agent_state_database
from wattwise_core.api.app import create_app
from wattwise_core.api.connection_catalog import FILE_IMPORT_SOURCE_KEY
from wattwise_core.config import Settings, load_settings
from wattwise_core.domain.enums import SourceKind
from wattwise_core.identity import OWNER_ATHLETE_ID
from wattwise_core.persistence.models import Athlete, Base, SourceDescriptor, Sport
from wattwise_core.security.crypto import EnvelopeCipher

# ``tools`` lives at the repo root (not an installed package); forge a REAL decodable FIT the
# same way test_fit_forge does (sys.path bootstrap, any import-mode) so the import below drives
# the REAL file-import adapter + IngestService — not a stub the fake processor would accept.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.fit_forge import forge_ride  # noqa: E402  (after the sys.path bootstrap)

pytestmark = pytest.mark.integration

#: The first-party owner sign-in secret (the configured ``token_signing_key``, API-R23).
_SIGNING_KEY = "real-wiring-signing-key-0123456789abcdef"
#: A memory row seeded into the agent-state store so the no-LLM erase has a real row to remove.
_MEMORY_ID = uuid.UUID("aaaaaaaa-aaaa-7000-8000-000000000001")


def _settings(tmp_path: Path, *, with_llm: bool) -> Settings:
    """Dev settings on a file DB; ``with_llm`` toggles whether the live coach engine binds.

    A FILE DSN (not ``:memory:``) so every request-scoped session AND the separate agent-state
    engine the factory builds share ONE database file — the only way the seeded owner + memory row
    are visible to the wired engine. ``with_llm`` sets ``llm_api_key`` so the factory binds the
    live ``GraphAgentEngine``; without it the factory binds the :class:`UnconfiguredAgentEngine`
    (the no-LLM path the findings target).
    """
    extra = {"agent__model": "deepseek/deepseek-v4-flash", "llm_api_key": "sk-not-called-offline"}
    return load_settings(
        app__environment="development",
        database_dsn=f"sqlite+aiosqlite:///{tmp_path / 'real.db'}",
        token_signing_key=_SIGNING_KEY,
        encryption_root_key=EnvelopeCipher.generate_root_key(),
        object_store__local_root=str(tmp_path / "objects"),
        **(extra if with_llm else {}),
    )


async def _seed(app: FastAPI, settings: Settings) -> None:
    """Create both schemas + seed the owner (canonical) and one memory row (agent-state store).

    Mirrors the initial migration's owner seed plus a current sport (so the sport-keyed signature
    probe has a sport to resolve, H3) and one durable memory item in the SAME agent-state store the
    wired engine reads — so the no-LLM memory list/get/erase operate on a real row.
    """
    database = app.state.database
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with database.session() as session:
        session.add(Sport(sport_code="cycling", display_name="Cycling", has_mechanical_power=True))
        # The built-in file_import descriptor the initial migration seeds (LIN-R1.1) — required
        # for the real import processor to land an upload; create_all builds the schema only.
        session.add(
            SourceDescriptor(
                source_key=FILE_IMPORT_SOURCE_KEY,
                display_name="Activity files",
                kind=SourceKind.FILE_UPLOAD,
            )
        )
        await session.flush()  # the sport row must exist before the athlete's current_sport FK
        session.add(
            Athlete(
                athlete_id=OWNER_ATHLETE_ID,
                sex="male",
                reference_timezone="UTC",
                current_sport="cycling",
            )
        )
    state_db = build_agent_state_database(settings)
    await state_db.create_all()
    async with state_db.session() as session:
        session.add(
            MemoryItem(
                memory_item_id=_MEMORY_ID,
                athlete_id=OWNER_ATHLETE_ID,
                kind=MemoryItemKind.PREFERENCE,
                content="prefers morning rides",
                inferred=False,
            )
        )
    await state_db.dispose()


def _client(tmp_path: Path, *, with_llm: bool) -> tuple[TestClient, FastAPI]:
    """Assemble the REAL app (no seam overrides) on a file DB and seed it."""
    settings = _settings(tmp_path, with_llm=with_llm)
    app = create_app(settings)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    client.portal.call(_seed, app, settings)  # type: ignore[union-attr]
    return client, app


def _token(client: TestClient) -> dict[str, str]:
    """Mint a real owner access token via the public token-issuance route (API-R23)."""
    resp = client.post("/v1/auth/token", json={"owner_secret": _SIGNING_KEY})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def no_llm(tmp_path: Path) -> Iterator[tuple[TestClient, dict[str, str]]]:
    """The REAL app on the NO-LLM path (UnconfiguredAgentEngine) + a real owner token."""
    client, _app = _client(tmp_path, with_llm=False)
    try:
        yield client, _token(client)
    finally:
        client.__exit__(None, None, None)


# --- the no-LLM path: every NON-LLM surface MUST work, NOTHING 500s (H1/H2/H4/M5) ---


def test_diagnose_does_not_500_on_no_llm(no_llm: tuple[TestClient, dict[str, str]]) -> None:
    """H2: ``POST /v1/agent/diagnose`` is deterministic and works with no LLM (API-R15)."""
    client, auth = no_llm
    resp = client.post("/v1/agent/diagnose", headers=auth)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["overall_ok"], bool)
    assert {c["code"] for c in body["checks"]} >= {"training_load", "fitness_signature"}


def test_digest_crud_does_not_500_on_no_llm(no_llm: tuple[TestClient, dict[str, str]]) -> None:
    """H1 + M5: subscribe / list / delete are wired; re-subscribe keeps ONE active row."""
    client, auth = no_llm
    sub = client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "weekly", "weekday": "mon", "hour_local": 7, "channels": ["web"]},
        headers=auth,
    )
    assert sub.status_code == 201, sub.text
    first_id = sub.json()["subscription_id"]
    # M5: a second subscribe UPDATES the standing row in place (same id), not a duplicate
    again = client.post(
        "/v1/agent/digest/subscribe",
        json={"cadence": "daily", "hour_local": 18, "channels": ["web"]},
        headers=auth,
    )
    assert again.status_code == 201, again.text
    assert again.json()["subscription_id"] == first_id
    listed = client.get("/v1/agent/digest/subscriptions", headers=auth)
    assert listed.status_code == 200, listed.text
    assert len(listed.json()["data"]) == 1, "exactly ONE standing schedule (GBO-R46 / M5)"
    deleted = client.delete(f"/v1/agent/digest/subscribe/{first_id}", headers=auth)
    assert deleted.status_code == 204, deleted.text


def test_digest_last_degrades_not_500_on_no_llm(no_llm: tuple[TestClient, dict[str, str]]) -> None:
    """H2: ``GET /v1/agent/digest/last`` degrades visibly with no LLM (never 500, RUN-R4.1)."""
    client, auth = no_llm
    resp = client.get("/v1/agent/digest/last", headers=auth)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "degraded"


def test_memory_read_then_erase_then_404_on_no_llm(
    no_llm: tuple[TestClient, dict[str, str]],
) -> None:
    """H2 + MEM-R3 + PRIV-R8: list/get/erase work with NO LLM and the erase REMOVES the row.

    The per-item memory surface is NON-LLM and a privacy MUST that can never depend on an LLM. On
    the no-LLM path it lists the seeded row, gets it by id, and a DELETE actually removes the
    residual row (PRIV-R8) so a re-GET is 404 — none of which may 500.
    """
    client, auth = no_llm
    listed = client.get("/v1/agent/memory", headers=auth)
    assert listed.status_code == 200, listed.text
    assert [r["summary_text"] for r in listed.json()["data"]] == ["prefers morning rides"]
    got = client.get(f"/v1/agent/memory/{_MEMORY_ID}", headers=auth)
    assert got.status_code == 200, got.text
    erased = client.delete(f"/v1/agent/memory/{_MEMORY_ID}", headers=auth)
    assert erased.status_code == 204, erased.text
    # PRIV-R8: the residual row is truly gone -> a re-GET is 404 (not a 500, not still present)
    regot = client.get(f"/v1/agent/memory/{_MEMORY_ID}", headers=auth)
    assert regot.status_code == 404, regot.text


def test_users_me_get_and_delete_do_not_500_on_no_llm(
    no_llm: tuple[TestClient, dict[str, str]],
) -> None:
    """H4: ``GET`` reads the account; ``DELETE`` records an ASYNC erasure request (never 500)."""
    client, auth = no_llm
    read = client.get("/v1/users/me", headers=auth)
    assert read.status_code == 200, read.text
    assert read.json()["email"] is None  # honest empty account, no email captured
    deleted = client.delete("/v1/users/me", headers=auth)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["status"] == "pending_deletion"  # async ack, not an inline delete/500


def test_planning_phase_gates_not_500_on_no_llm(
    no_llm: tuple[TestClient, dict[str, str]],
) -> None:
    """RUN-R4.1: plan generation phase-gates to a degraded answer with no LLM (never 500)."""
    client, auth = no_llm
    gen = client.post(
        "/v1/planning/workouts", json={"request": "build me a base week"}, headers=auth
    )
    assert gen.status_code == 200, gen.text
    assert gen.json()["status"] == "degraded"
    # the keyset read view is wired too (empty library, but a real 200 — not a 500)
    read = client.get("/v1/planning/workouts", headers=auth)
    assert read.status_code == 200, read.text


def test_no_advertised_endpoint_500s_on_no_llm(
    no_llm: tuple[TestClient, dict[str, str]],
) -> None:
    """The headline guard: NONE of the ROAD-R2-EXIT endpoints 500 in the real factory (no override).

    A compact sweep over every newly-wired surface asserting each returns a SPEC status, never a
    ``500`` — the single regression this whole file exists to catch.
    """
    client, auth = no_llm
    probes = [
        client.post("/v1/agent/diagnose", headers=auth),
        client.get("/v1/agent/digest/list", headers=auth),
        client.get("/v1/agent/digest/last", headers=auth),
        client.get("/v1/agent/memory", headers=auth),
        client.get(f"/v1/agent/memory/{uuid.uuid4()}", headers=auth),
        client.delete(f"/v1/agent/memory/{uuid.uuid4()}", headers=auth),
        client.get("/v1/users/me", headers=auth),
        client.get("/v1/planning/workouts", headers=auth),
    ]
    for resp in probes:
        assert resp.status_code != 500, (resp.request.url, resp.text)
        assert resp.status_code in {200, 201, 202, 204, 404, 422}, (
            resp.request.url,
            resp.status_code,
        )


def test_import_job_reaches_done_after_synchronous_ingest(
    no_llm: tuple[TestClient, dict[str, str]],
) -> None:
    """API-R33a regression (#115): a successful import reaches the TERMINAL ``done`` status.

    Drives the REAL factory-wired import processor (api.wiring) over a REAL decodable FIT, so
    the synchronous decode→ingest actually runs in-request and lands a canonical activity. The
    persisted job (the ``GET /v1/imports/{id}`` read surface) MUST then read ``done`` — never the
    non-terminal ``queued`` the bug stranded it at, even though the activity was already written.
    Asserts the OBSERVABLE behavior end-to-end: the 202 body's terminal status AND the
    polled-after read both report ``done`` (and the imported ride is canonically present).
    """
    client, auth = no_llm
    fit = forge_ride(start=_dt.datetime(2026, 6, 1, 10, 0, tzinfo=_dt.UTC))

    created = client.post(
        "/v1/imports",
        headers=auth,
        files={"file": ("ride.fit", fit, "application/octet-stream")},
    )
    assert created.status_code == 202, created.text
    body = created.json()
    # The synchronous OSS path reaches its terminal status in one step (API-R33a): NOT queued.
    assert body["status"] == "done", body
    job_id = body["import_job_id"]

    # The polled read surface — the exact route the tester reported stuck at "queued" — agrees.
    polled = client.get(f"/v1/imports/{job_id}", headers=auth)
    assert polled.status_code == 200, polled.text
    assert polled.json()["status"] == "done", polled.json()

    # And the work the "done" claims is real: the imported ride landed as a canonical activity.
    activities = client.get("/v1/activities", headers=auth)
    assert activities.status_code == 200, activities.text
    assert activities.json()["data"], "the imported ride must be canonically present"


def test_import_job_reaches_failed_when_ingest_raises_post_acceptance(
    no_llm: tuple[TestClient, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API-R33a fail-closed (#115): a post-acceptance ingest failure reaches TERMINAL ``failed``.

    The upload passes extension/size/decode validation (so it is ACCEPTED, 202), then the real
    factory-wired processor's synchronous ``IngestService.ingest`` raises an INTERNAL error. The
    job MUST be persisted reading ``failed`` — never stranded at ``queued`` and never a bare 5xx
    with no job row to poll. Asserts the OBSERVABLE behavior: the 202 body AND the polled read
    surface both report ``failed``. ``IngestService.ingest`` is patched (not the seam) so the REAL
    ``api.wiring`` exception handling is exercised.
    """
    client, auth = no_llm

    async def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("ingest blew up mid-write")

    # Patch the engine method the real processor calls; the wiring's except-branch must catch it.
    monkeypatch.setattr("wattwise_core.ingestion.ingest.IngestService.ingest", _boom, raising=True)
    fit = forge_ride(start=_dt.datetime(2026, 6, 2, 10, 0, tzinfo=_dt.UTC))

    created = client.post(
        "/v1/imports",
        headers=auth,
        files={"file": ("ride.fit", fit, "application/octet-stream")},
    )
    # Accepted (passed pre-ingest validation), then the ingest failed -> terminal "failed".
    assert created.status_code == 202, created.text
    body = created.json()
    assert body["status"] == "failed", body
    job_id = body["import_job_id"]

    polled = client.get(f"/v1/imports/{job_id}", headers=auth)
    assert polled.status_code == 200, polled.text
    assert polled.json()["status"] == "failed", polled.json()

    # Idempotency (API-R33a): the "please try again" the failed copy invites must CONVERGE,
    # not orphan a new row each retry. Re-uploading the SAME failing file re-derives the SAME
    # deterministic job id, so the list shows exactly one failed job, not two.
    retry = client.post(
        "/v1/imports",
        headers=auth,
        files={"file": ("ride.fit", fit, "application/octet-stream")},
    )
    assert retry.status_code == 202, retry.text
    assert retry.json()["import_job_id"] == job_id, "a same-file retry must converge to one row"
    listed = client.get("/v1/imports", headers=auth)
    assert listed.status_code == 200, listed.text
    assert [j["import_job_id"] for j in listed.json()["data"]] == [job_id], listed.json()


# --- the WITH-LLM path: the SAME factory binds the live engine without 500-ing the wiring ---


def test_with_llm_factory_wires_breadth_and_users_without_500(tmp_path: Path) -> None:
    """The same factory on the WITH-LLM path binds the live engine + all breadth/users seams.

    Proves the seam WIRING (not the model) is correct on the live-engine path too: the breadth DB
    session seam, the users deletion recorder, and the engine seam are all bound, so the NON-LLM
    surfaces that touch no model (diagnose / memory list / users GET+DELETE / digest CRUD) return
    their spec status rather than 500-ing on an unwired seam. The model-driven ``/ask`` is NOT
    called here (it would hit the network); this isolates the wiring from the model.
    """
    client, _app = _client(tmp_path, with_llm=True)
    try:
        auth = _token(client)
        probes = [
            client.post("/v1/agent/diagnose", headers=auth),
            client.post(
                "/v1/agent/digest/subscribe",
                json={"cadence": "daily", "hour_local": 7, "channels": ["web"]},
                headers=auth,
            ),
            client.get("/v1/agent/digest/list", headers=auth),
            client.get("/v1/agent/memory", headers=auth),
            client.get("/v1/users/me", headers=auth),
            client.delete("/v1/users/me", headers=auth),
        ]
        for resp in probes:
            assert resp.status_code != 500, (resp.request.url, resp.text)
            assert resp.status_code in {200, 201, 202, 204, 404, 422}, (
                resp.request.url,
                resp.status_code,
            )
    finally:
        client.__exit__(None, None, None)
