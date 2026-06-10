"""Integration tests for the doc-60 surface groups landed in this slice.

Covers, against the REAL ``create_app`` wiring (only no LLM is configured):

* **Exports (§8.15, API-R34)** — job create/list/read; the bearer download; the
  short-lived, SINGLE-USE, owner-bound signed download URL (reuse + tamper → ``403
  invalid-signed-url``); not-found isolation (API-R51).
* **Imports (API-R33)** — a real GPX upload lands a job row; ``GET /v1/imports`` +
  ``GET /v1/imports/{id}`` read it back; a foreign id is ``404``.
* **Help (AUTH-R10)** — public topics list/detail, no token required; PAGE-R3 bounds.
* **Dashboard / Data health (§8.2/§8.3)** — composed reads answer with typed nulls on
  an empty history, never fabricated numbers, and respect the ``read`` scope.
* **Admin + system diagnose (AUTH-R12)** — the ``admin`` scope gates the operator
  surface: the owner token passes, a delegated (bot-link) token gets ``403``; the
  plan PUT validates through the fail-closed gate and takes effect on app state.
* **AUTH-R8a** — the ``X-Service-Auth`` service-principal factor: wrong/unverifiable
  header → ``401``; a delegated token MUST present it when configured; it never
  substitutes for the bearer.

The canonical DB is a real FILE SQLite (never ``:memory:``) so the app's own engines
read the seeded schema; the agent-state store is the factory's own (ARCH-R13).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine as _create_sync_engine
from starlette.testclient import TestClient

from wattwise_core.api.app import create_app
from wattwise_core.config import Settings, load_settings
from wattwise_core.identity import OWNER_ATHLETE_ID
from wattwise_core.persistence.models import Athlete, Base, SourceDescriptor, Sport
from wattwise_core.security.crypto import EnvelopeCipher

pytestmark = pytest.mark.integration

#: The first-party owner sign-in secret (the configured ``token_signing_key``, API-R23).
_KEY = "doc60-surfaces-signing-key-0123456789abcdef"

#: The built-in ``file_import`` descriptor id seeded by migration 0001 (LIN-R1.1).
_FILE_IMPORT_DESCRIPTOR_ID = uuid.UUID("01890000-0000-7000-8000-000000000001")

_REPO = Path(__file__).resolve().parents[2]
_GPX = _REPO / "tests" / "contract" / "fixtures" / "file_upload" / "ride.gpx"


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    """REAL dev settings on a FILE DB (a real pool — never ``:memory:``)."""
    base: dict[str, Any] = {
        "app__environment": "development",
        "database_dsn": f"sqlite+aiosqlite:///{tmp_path / 'doc60.db'}",
        "token_signing_key": _KEY,
        "encryption_root_key": EnvelopeCipher.generate_root_key(),
        "object_store__local_root": str(tmp_path / "objects"),
    }
    base.update(overrides)
    return load_settings(**base)


def _seed_canonical(tmp_path: Path) -> None:
    """Create the canonical schema + seed the owner and the cycling sport."""
    engine = _create_sync_engine(f"sqlite:///{tmp_path / 'doc60.db'}")
    try:
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(
                Sport.__table__.insert().values(
                    sport_code="cycling", display_name="Cycling", has_mechanical_power=True
                )
            )
            conn.execute(
                Athlete.__table__.insert().values(
                    athlete_id=OWNER_ATHLETE_ID,
                    sex="male",
                    reference_timezone="UTC",
                    current_sport="cycling",
                )
            )
            conn.execute(
                SourceDescriptor.__table__.insert().values(
                    source_descriptor_id=_FILE_IMPORT_DESCRIPTOR_ID,
                    source_key="file_import",
                    display_name="Activity files",
                    kind="file_upload",
                    trust_profile={},
                    default_fidelity=None,
                )
            )
    finally:
        engine.dispose()


def _app(tmp_path: Path, **overrides: Any) -> tuple[TestClient, FastAPI]:
    """The real assembled app on a seeded file DB, with a non-raising client."""
    settings = _settings(tmp_path, **overrides)
    _seed_canonical(tmp_path)
    app = create_app(settings)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    return client, app


def _owner_auth(client: TestClient) -> dict[str, str]:
    """Mint a real owner token via the public token route (API-R23)."""
    resp = client.post("/v1/auth/token", json={"owner_secret": _KEY})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _delegated_auth(client: TestClient, owner: dict[str, str]) -> dict[str, str]:
    """Mint a DELEGATED (bot-link) token through the full link flow (AUTH-R8)."""
    challenge = client.post("/v1/auth/link/start")
    assert challenge.status_code == 200, challenge.text
    code = challenge.json()["link_code"]
    approved = client.post("/v1/auth/link/approve", json={"link_code": code}, headers=owner)
    assert approved.status_code == 200, approved.text
    completed = client.post("/v1/auth/link/complete", json={"link_code": code})
    assert completed.status_code == 200, completed.text
    return {"Authorization": f"Bearer {completed.json()['access_token']}"}


# ------------------------------------------------------------------ exports (API-R34)


def test_export_job_create_list_read_and_bearer_download(tmp_path: Path) -> None:
    """An export job is created 202/ready, listed, read, and downloadable via bearer."""
    client, _ = _app(tmp_path)
    try:
        auth = _owner_auth(client)
        created = client.post(
            "/v1/exports", json={"scope": "activities", "format": "json"}, headers=auth
        )
        assert created.status_code == 202, created.text
        job = created.json()
        assert job["status"] == "ready"
        assert job["download"] is not None  # the signed handle on a ready job (API-R34)
        listed = client.get("/v1/exports", headers=auth)
        assert listed.status_code == 200
        assert [j["export_job_id"] for j in listed.json()["data"]] == [job["export_job_id"]]
        got = client.get(f"/v1/exports/{job['export_job_id']}", headers=auth)
        assert got.status_code == 200
        download = client.get(f"/v1/exports/{job['export_job_id']}/download", headers=auth)
        assert download.status_code == 200, download.text
        assert download.headers["content-disposition"].startswith("attachment")
    finally:
        client.__exit__(None, None, None)


def test_signed_export_url_is_single_use_and_tamper_proof(tmp_path: Path) -> None:
    """The signed URL works bearer-FREE exactly once; reuse/tamper → 403 (API-R34)."""
    client, _ = _app(tmp_path)
    try:
        auth = _owner_auth(client)
        job = client.post(
            "/v1/exports", json={"scope": "analytics", "format": "json"}, headers=auth
        ).json()
        url = job["download"]["url"]
        first = client.get(url)  # NO bearer header: the documented exception path
        assert first.status_code == 200, first.text
        replay = client.get(url)
        assert replay.status_code == 403  # the one-time nonce is consumed (single-use)
        assert replay.json()["type"].endswith("invalid-signed-url")
        tampered = url[:-4] + ("aaaa" if not url.endswith("aaaa") else "bbbb")
        assert client.get(tampered).status_code == 403  # signature mismatch → no leak
    finally:
        client.__exit__(None, None, None)


def test_unknown_export_job_is_404(tmp_path: Path) -> None:
    """An unknown/foreign job id reads as absent → 404 not-found (API-R51)."""
    client, _ = _app(tmp_path)
    try:
        auth = _owner_auth(client)
        resp = client.get(
            "/v1/exports/00000000-0000-7000-8000-0000000000aa", headers=auth
        )
        assert resp.status_code == 404
    finally:
        client.__exit__(None, None, None)


# ------------------------------------------------------------------ imports (API-R33)


def test_import_upload_then_list_and_detail(tmp_path: Path) -> None:
    """A real GPX upload lands a job row readable via GET list + detail (API-R33)."""
    client, _ = _app(tmp_path)
    try:
        auth = _owner_auth(client)
        accepted = client.post(
            "/v1/imports",
            headers=auth,
            files={"file": ("ride.gpx", _GPX.read_bytes(), "application/gpx+xml")},
        )
        assert accepted.status_code == 202, accepted.text
        job_id = accepted.json()["import_job_id"]
        listed = client.get("/v1/imports", headers=auth)
        assert listed.status_code == 200, listed.text
        assert job_id in [j["import_job_id"] for j in listed.json()["data"]]
        got = client.get(f"/v1/imports/{job_id}", headers=auth)
        assert got.status_code == 200
        assert got.json()["status"] in {"queued", "processing", "done"}
        assert got.json()["status_text"]  # athlete-native copy (API-R21)
        missing = client.get("/v1/imports/definitely-not-a-job", headers=auth)
        assert missing.status_code == 404
    finally:
        client.__exit__(None, None, None)


# ------------------------------------------------------------------- help (AUTH-R10)


def test_help_topics_are_public_and_bounded(tmp_path: Path) -> None:
    """Help topics serve WITHOUT a token (AUTH-R10); PAGE-R3 bounds the list."""
    client, _ = _app(tmp_path)
    try:
        listed = client.get("/v1/help/topics")  # no Authorization header at all
        assert listed.status_code == 200, listed.text
        topics = listed.json()["data"]
        assert topics, "the static help catalog is non-empty"
        one = client.get(f"/v1/help/topics/{topics[0]['topic_id']}")
        assert one.status_code == 200
        assert client.get("/v1/help/topics/no-such-topic").status_code == 404
        assert client.get("/v1/help/topics", params={"limit": 0}).status_code == 422
    finally:
        client.__exit__(None, None, None)


# --------------------------------------------------- dashboard + data health (§8.2/§8.3)


def test_dashboard_metrics_and_alerts_on_empty_history(tmp_path: Path) -> None:
    """The composed dashboard answers with typed nulls on an empty history (§8.2)."""
    client, _ = _app(tmp_path)
    try:
        auth = _owner_auth(client)
        assert client.get("/v1/dashboard/metrics").status_code == 401  # AUTH-R1
        metrics = client.get("/v1/dashboard/metrics", headers=auth)
        assert metrics.status_code == 200, metrics.text
        body = metrics.json()
        assert body["last_activity"] is None  # nothing fabricated for an empty history
        assert body["computed_at"]
        alerts = client.get("/v1/dashboard/alerts", headers=auth)
        assert alerts.status_code == 200, alerts.text
        for alert in alerts.json()["data"]:
            assert alert["severity"] in {"info", "warning", "critical"}  # SCHEMA-R3
    finally:
        client.__exit__(None, None, None)


def test_data_health_summary_matrix_and_issues(tmp_path: Path) -> None:
    """Data-health reads derive from the real coverage checks, never invented (§8.3)."""
    client, _ = _app(tmp_path)
    try:
        auth = _owner_auth(client)
        summary = client.get("/v1/data-health/summary", headers=auth)
        assert summary.status_code == 200, summary.text
        body = summary.json()
        assert 0.0 <= body["completeness_score"] <= 1.0
        assert body["headline_text"]
        matrix = client.get("/v1/data-health/coverage-matrix", headers=auth)
        assert matrix.status_code == 200, matrix.text
        domains = {cell["domain"] for cell in matrix.json()["domains"]}
        assert {"training_load", "fitness_signature"} <= domains
        issues = client.get("/v1/data-health/issues", headers=auth)
        assert issues.status_code == 200, issues.text
        for issue in issues.json()["data"]:
            assert issue["severity"] in {"info", "warning", "critical"}
            assert issue["message_text"]
    finally:
        client.__exit__(None, None, None)


# ----------------------------------------------- admin + system diagnose (AUTH-R12)


def test_admin_surface_requires_the_admin_scope(tmp_path: Path) -> None:
    """The owner token (admin) passes; a delegated bot token gets 403 (AUTH-R12)."""
    client, _ = _app(tmp_path)
    try:
        owner = _owner_auth(client)
        delegated = _delegated_auth(client, owner)
        assert client.get("/v1/admin/plans", headers=owner).status_code == 200
        refused = client.get("/v1/admin/plans", headers=delegated)
        assert refused.status_code == 403
        assert refused.json()["type"].endswith("insufficient-scope")
        assert client.get("/v1/system/diagnose", headers=delegated).status_code == 403
        diagnose = client.get("/v1/system/diagnose", headers=owner)
        assert diagnose.status_code == 200, diagnose.text
        codes = {check["code"] for check in diagnose.json()["checks"]}
        assert {"database", "default_plan_loaded", "signing_key"} <= codes
    finally:
        client.__exit__(None, None, None)


def test_admin_plan_put_validates_and_takes_effect(tmp_path: Path) -> None:
    """A plan PUT runs the fail-closed validation and swaps the live plan (§8.16)."""
    client, app = _app(tmp_path)
    try:
        owner = _owner_auth(client)
        plan = client.get("/v1/admin/plans", headers=owner).json()["data"][0]
        update = {k: v for k, v in plan.items() if k != "plan_id"}
        update["request_rate_per_minute"] = 7
        ok = client.put("/v1/admin/plans/default", json=update, headers=owner)
        assert ok.status_code == 200, ok.text
        assert app.state.entitlement_plan.request_rate_per_minute == 7  # live swap
        assert client.put("/v1/admin/plans/other", json=update, headers=owner).status_code == 404
        update["node_visit_ceiling"] = 0  # invalid bound → rejected by validation
        assert client.put("/v1/admin/plans/default", json=update, headers=owner).status_code == 422
    finally:
        client.__exit__(None, None, None)


def test_admin_model_policy_roundtrip_and_catalog(tmp_path: Path) -> None:
    """Model policy GET/PUT and the read-only catalog (§8.16, closed enums)."""
    client, _ = _app(tmp_path)
    try:
        owner = _owner_auth(client)
        policy = client.get("/v1/admin/model-policy", headers=owner)
        assert policy.status_code == 200, policy.text
        put = client.put(
            "/v1/admin/model-policy",
            json={"allowed_tiers": ["flash"], "default_tier": "flash", "reasoning_ceiling": "low"},
            headers=owner,
        )
        assert put.status_code == 200, put.text
        bad = client.put(
            "/v1/admin/model-policy",
            json={"allowed_tiers": ["flash"], "default_tier": "turbo", "reasoning_ceiling": "low"},
            headers=owner,
        )
        assert bad.status_code == 422  # unknown enum member → closed enum (SCHEMA-R3)
        catalog = client.get("/v1/admin/model-catalog", headers=owner)
        assert catalog.status_code == 200
        assert catalog.json()["data"], "the one configured OSS model seam is reported"
    finally:
        client.__exit__(None, None, None)


# ------------------------------------------------------------------ AUTH-R8a factor


def test_service_auth_header_is_verified_constant_time(tmp_path: Path) -> None:
    """A presented X-Service-Auth must match the configured secret, else 401 (AUTH-R8a)."""
    client, _ = _app(
        tmp_path, security__service_auth_secret="service-factor-secret-0123456789abcdef"
    )
    try:
        owner = _owner_auth(client)
        factor = {"X-Service-Auth": "service-factor-secret-0123456789abcdef"}
        good = client.get("/v1/athlete", headers={**owner, **factor})
        assert good.status_code in {200, 404}  # the factor verifies; the route proceeds
        wrong = client.get("/v1/athlete", headers={**owner, "X-Service-Auth": "wrong"})
        assert wrong.status_code == 401  # mismatched factor fails closed
    finally:
        client.__exit__(None, None, None)


def test_presented_factor_with_no_configured_secret_fails_closed(tmp_path: Path) -> None:
    """A presented header that CANNOT be verified (no secret configured) → 401."""
    client, _ = _app(tmp_path)
    try:
        owner = _owner_auth(client)
        resp = client.get("/v1/athlete", headers={**owner, "X-Service-Auth": "anything"})
        assert resp.status_code == 401
    finally:
        client.__exit__(None, None, None)


def test_delegated_token_requires_the_service_factor_when_configured(tmp_path: Path) -> None:
    """A delegated (bot) token MUST carry the factor when configured (AUTH-R8a)."""
    secret = "service-factor-secret-0123456789abcdef"
    client, _ = _app(tmp_path, security__service_auth_secret=secret)
    try:
        owner = _owner_auth(client)
        delegated = _delegated_auth(client, owner)
        bare = client.get("/v1/athlete", headers=delegated)
        assert bare.status_code == 401  # the second layer is mandatory for the service
        with_factor = client.get("/v1/athlete", headers={**delegated, "X-Service-Auth": secret})
        assert with_factor.status_code in {200, 404}  # both layers verify; never a superuser
        # The factor alone (no bearer) grants nothing: it never substitutes (AUTH-R8a).
        assert client.get("/v1/athlete", headers={"X-Service-Auth": secret}).status_code == 401
    finally:
        client.__exit__(None, None, None)


# ----------------------------------------------------------- public rate limit (LIMIT-R1)


def test_public_auth_endpoints_are_rate_limited(tmp_path: Path) -> None:
    """The pre-token surface debits a shared bucket and 429s when exhausted (LIMIT-R1)."""
    client, _ = _app(tmp_path)
    try:
        # The mutating public bucket is 30/min: hammer link/start until it trips.
        statuses = [client.post("/v1/auth/link/start").status_code for _ in range(35)]
        assert 429 in statuses, "the public pre-token bucket never tripped (LIMIT-R1)"
        assert statuses[0] == 200  # the first call was served normally
    finally:
        client.__exit__(None, None, None)
