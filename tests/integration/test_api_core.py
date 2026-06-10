"""Integration regressions for the cross-cutting ``/v1`` contract (ERR/DOC/LIMIT/AUTH).

End-to-end over the assembled :func:`create_app` ASGI app, asserting the convergence
fixes the whole surface shares:

- **DOC-R3/R4/R5** the OpenAPI carries reusable ``Problem`` + ``PageEnvelope`` components,
  every operation declares a stable ``operationId``, the connections next-step union is
  ``discriminator``-tagged, and every operation documents the Problem error responses;
- **AUTH-R3/R18** no request schema in the published document exposes a writable
  caller-identity field (a contract scan of every request model);
- **ERR-R7** a framework ``404``/``405`` keeps its originating status (not collapsed to a
  ``422``/``404`` by the status->slug table) and stays an RFC 9457 problem;
- **LIMIT-R5/R6** an oversized JSON body is rejected ``413`` from the streamed bytes;
- **LIMIT-R1/R2/R3** the read/mutating buckets are enforced per athlete with the
  ``RateLimit-*`` + ``Retry-After`` headers;
- **API-R3/AUTH-R1** the factory wires the performance/activities/agent seams so the
  surface is functional and auth is actually enforced (a tokenless call is ``401``).

Tier: T-INTEGRATION (offline, in-process ASGI via the FastAPI ``TestClient``).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from wattwise_core.api.app import API_PREFIX, create_app
from wattwise_core.api.auth import TOKEN_ALGORITHM, TOKEN_AUDIENCE, TOKEN_ISSUER
from wattwise_core.api.errors import PROBLEM_BASE_URI, PROBLEM_MEDIA_TYPE
from wattwise_core.api.middleware import DEFAULT_JSON_MAX_BYTES
from wattwise_core.config import Settings, load_settings
from wattwise_core.persistence.models import Athlete, Base

pytestmark = pytest.mark.integration

_SIGNING_KEY = "core-integration-signing-key-0123456789"

#: Banned writable caller-identity fields no request schema may declare (AUTH-R3/R18).
_BANNED_IDENTITY_FIELDS = {"athlete_id", "user_id", "subject", "owner_id", "principal_id"}


def _settings(dsn: str = "sqlite+aiosqlite:///:memory:") -> Settings:
    return load_settings(
        app__environment="development",
        database_dsn=dsn,
        token_signing_key=_SIGNING_KEY,
    )


def _client() -> TestClient:
    return TestClient(create_app(_settings()), raise_server_exceptions=False)


@pytest.fixture
def db_client(tmp_path: Path) -> Iterator[tuple[TestClient, str]]:
    """An app on a real file-backed SQLite DB (schema created + one owner seeded).

    The feature read/mutating routes touch the canonical store, so the rate-limit and
    wired-surface regressions need a real schema; a temp-file DSN shares one DB across
    the app's connections (unlike ``:memory:``). Yields the client + the owner id the
    seeded athlete uses as the token subject.
    """
    dsn = f"sqlite+aiosqlite:///{tmp_path / 'core.sqlite'}"
    app = create_app(_settings(dsn))
    with TestClient(app, raise_server_exceptions=False) as client:
        athlete_id = client.portal.call(_prepare_db, app)  # type: ignore[union-attr]
        yield client, athlete_id


async def _prepare_db(app: Any) -> str:
    """Create the schema on the app's engine and seed one athlete (the token subject)."""
    database = app.state.database
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    athlete_id = uuid.uuid4()
    async with database.session() as session:
        session.add(Athlete(athlete_id=athlete_id, sex="male", reference_timezone="UTC"))
    return str(athlete_id)


def _owner_token(athlete_id: str, *, scopes: list[str]) -> dict[str, str]:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        "sub": athlete_id,
        "scope": " ".join(scopes),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
    }
    token = jwt.encode(payload, _SIGNING_KEY, algorithm=TOKEN_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


def _token(*, scopes: list[str]) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        "sub": "owner",
        "scope": " ".join(scopes),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
    }
    return jwt.encode(payload, _SIGNING_KEY, algorithm=TOKEN_ALGORITHM)


def _auth(*, scopes: list[str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {_token(scopes=scopes)}"}


# --- DOC-R3/R4/R5: OpenAPI is client-generatable ---------------------------------


def test_openapi_has_problem_and_pageenvelope_components() -> None:
    """The document registers reusable Problem + PageEnvelope components (DOC-R4)."""
    spec = create_app(_settings()).openapi()
    schemas = spec["components"]["schemas"]
    assert "Problem" in schemas and "PageEnvelope" in schemas
    assert set(schemas["Problem"]["required"]) >= {
        "type",
        "title",
        "status",
        "detail",
        "instance",
        "trace_id",
    }


def test_every_operation_has_stable_operation_id_and_error_responses() -> None:
    """Every operation declares a stable operationId + documented Problem errors (DOC-R3)."""
    spec = create_app(_settings()).openapi()
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            assert "operationId" in op, f"{method} {path} lacks a stable operationId"
            # An auto-mangled, path-embedded id is not stable (DOC-R3).
            assert "_v1_" not in op["operationId"], f"{path} has an auto-mangled operationId"
            responses = op["responses"]
            assert "422" in responses
            ref = responses["422"]["content"][PROBLEM_MEDIA_TYPE]["schema"]["$ref"]
            assert ref.endswith("/Problem")


def test_connection_next_step_union_is_discriminated() -> None:
    """The initiate response union carries an OpenAPI discriminator (SCHEMA-R10/DOC-R5)."""
    spec = create_app(_settings()).openapi()
    op = spec["paths"]["/v1/connections/{source}/initiate"]["post"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    # FastAPI emits the discriminated union as an allOf/$ref with a discriminator block.
    flat = str(schema)
    assert "discriminator" in flat


def test_no_request_schema_exposes_caller_identity_field() -> None:
    """No request body schema declares a writable caller-identity field (AUTH-R3/R18)."""
    spec = create_app(_settings()).openapi()
    for name, component in spec["components"]["schemas"].items():
        props = set(component.get("properties", {}))
        assert _BANNED_IDENTITY_FIELDS.isdisjoint(props), f"{name} exposes a caller-identity field"


# --- ERR-R7: framework status is preserved ---------------------------------------


def test_unknown_route_is_404_problem_not_422() -> None:
    """An unmatched route stays a 404 problem (not collapsed to validation-error, ERR-R7)."""
    resp = _client().get(f"{API_PREFIX}/does/not/exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == 404
    assert body["type"] == f"{PROBLEM_BASE_URI}not-found"


def test_wrong_method_is_405_problem() -> None:
    """A wrong HTTP method keeps its 405 status (not rewritten to 404), as a problem (ERR-R7)."""
    resp = _client().get(f"{API_PREFIX}/auth/token")  # token is POST-only
    assert resp.status_code == 405
    assert resp.json()["status"] == 405
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)


# --- LIMIT-R5/R6: JSON body size cap ---------------------------------------------


def test_oversized_json_body_is_413() -> None:
    """A JSON body past the 256 KiB cap is rejected 413 from the streamed bytes (LIMIT-R5)."""
    huge = "x" * (DEFAULT_JSON_MAX_BYTES + 1024)
    resp = _client().post(
        f"{API_PREFIX}/agent/ask",
        content=f'{{"question": "{huge}"}}',
        headers={"Content-Type": "application/json", **_auth(scopes=["agent"])},
    )
    assert resp.status_code == 413
    assert resp.json()["type"].endswith("/payload-too-large")


# --- LIMIT-R1/R2/R3: read/mutating rate limits on the feature surface ------------


def test_read_endpoints_are_rate_limited_per_athlete(db_client: tuple[TestClient, str]) -> None:
    """Read endpoints debit the 120/min read bucket and 429 past it (LIMIT-R1/R2/R3)."""
    client, athlete_id = db_client
    headers = _owner_token(athlete_id, scopes=["read"])
    limited = None
    # The read bucket is 120/min; a burst exhausts it. Allow a small margin for the
    # token-bucket's continuous refill (≈2 tokens/sec) over the loop's wall time.
    for _ in range(160):
        resp = client.get(f"{API_PREFIX}/onboarding/status", headers=headers)
        if resp.status_code == 429:
            limited = resp
            break
    assert limited is not None, "the read bucket never rate-limited within the burst (LIMIT-R2)"
    assert limited.json()["type"].endswith("/rate-limited")
    assert int(limited.headers["Retry-After"]) >= 1
    assert limited.headers["RateLimit-Limit"] == "120"
    assert limited.headers["RateLimit-Remaining"] == "0"


def test_read_endpoint_emits_ratelimit_headers_on_success(
    db_client: tuple[TestClient, str],
) -> None:
    """A served read carries the RateLimit-* headers for the post-debit state (LIMIT-R3)."""
    client, athlete_id = db_client
    headers = _owner_token(athlete_id, scopes=["read"])
    resp = client.get(f"{API_PREFIX}/onboarding/status", headers=headers)
    assert resp.status_code == 200
    assert resp.headers["RateLimit-Limit"] == "120"
    assert int(resp.headers["RateLimit-Remaining"]) <= 120


def test_activities_list_is_rate_limited_per_athlete(db_client: tuple[TestClient, str]) -> None:
    """GET /v1/activities debits the per-athlete read bucket and 429s past it (LIMIT-R1/R2).

    The activities/athlete/performance/user-settings feature surfaces attached only scope
    gates and were unthrottled (no per-subject rate limit) — LIMIT-R1 requires EVERY endpoint
    be rate-limited per athlete/owner. This drives the REAL assembled ``create_app`` (no stubbed
    limiter): a burst on a SUCCESSFULLY-served (200) read endpoint exhausts the 120/min read
    bucket keyed on the server-derived id and yields the catalog ``429 rate-limited`` problem
    with the ``Retry-After`` + ``RateLimit-*`` headers (LIMIT-R3). Mutation-proof: drop the
    router-level ``dependencies=[RateLimit]`` on ``activities`` and this burst never 429s.
    """
    client, athlete_id = db_client
    headers = _owner_token(athlete_id, scopes=["read"])
    # Sanity: the endpoint really serves 200 (so the 429 is the limiter, not an auth/wiring 401).
    first = client.get(f"{API_PREFIX}/activities", headers=headers)
    assert first.status_code == 200, "GET /v1/activities must serve before the rate-limit assertion"
    limited = None
    # The read bucket is 120/min; a burst exhausts it. A margin past 120 absorbs the
    # token-bucket's continuous refill (≈2 tokens/sec) over the loop's wall time.
    for _ in range(160):
        resp = client.get(f"{API_PREFIX}/activities", headers=headers)
        if resp.status_code == 429:
            limited = resp
            break
    assert limited is not None, "GET /v1/activities was never rate-limited in the burst (LIMIT-R1)"
    assert limited.json()["type"].endswith("/rate-limited")
    assert int(limited.headers["Retry-After"]) >= 1
    assert limited.headers["RateLimit-Limit"] == "120"
    assert limited.headers["RateLimit-Remaining"] == "0"


# --- API-R3 / AUTH-R1: the factory wires the seam routers ------------------------


def test_performance_surface_is_wired_and_auth_enforced() -> None:
    """The factory wires the performance seams; a tokenless call is 401, not 403/500 (API-R3)."""
    resp = _client().get(
        f"{API_PREFIX}/performance/load-fitness", params={"from": "2026-06-01", "to": "2026-06-07"}
    )
    # Wired but unauthenticated -> the real bearer gate fires (AUTH-R1), not the
    # unwired fail-closed 403/500 the seam default would raise.
    assert resp.status_code == 401
    assert resp.json()["type"].endswith("/unauthenticated")


def test_activities_surface_is_wired_and_auth_enforced() -> None:
    """The activities seams are wired; a tokenless call is a uniform 401 (API-R3/AUTH-R1)."""
    resp = _client().get(f"{API_PREFIX}/activities")
    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)


# --- API-R41 / AUTH-R13: GET /v1/agent/readiness is wired and gated --------------
#
# The readiness endpoint (API-R41) shares the agent router's bearer + scope gates,
# but the factory-level "wired and auth enforced" coverage its sibling surfaces have
# (test_performance_surface_is_wired_and_auth_enforced /
# test_activities_surface_is_wired_and_auth_enforced) was missing for it. These three
# assert the real gate over the assembled create_app (no override of the scope/identity
# seams), so a tokenless call fails 401 and a token lacking ``agent`` fails 403 — the
# real authn/authz seam, not the router's unwired fail-closed default. Deterministic:
# the factory's agent engine falls back to the unconfigured (no-model) engine, so the
# served 200 needs no live model.

#: Numeric readiness KPI/score field names that must never appear on the response
#: (API-R41 / COACH-R7: readiness is a typed verdict, never a number).
_FORBIDDEN_READINESS_KPI_FIELDS = ("readiness", "readiness_score", "score")
#: Thread/continuation field names — the readiness surface is stateless this phase
#: (no thread_id, no continuation follow-up chip), so none may appear on the response.
_FORBIDDEN_READINESS_THREAD_FIELDS = ("thread_id", "continuation", "follow_up")


def test_agent_readiness_without_token_is_401() -> None:
    """A tokenless GET /v1/agent/readiness hits the real bearer gate -> 401 (AUTH-R1/R13).

    The factory wires the agent scope gate + server-derived identity, so an unauthenticated
    call yields the uniform ``401 unauthenticated`` (the real bearer gate, AUTH-R1), never
    the router's unwired fail-closed 401/403 seam default.
    """
    resp = _client().get(f"{API_PREFIX}/agent/readiness")
    assert resp.status_code == 401
    assert resp.json()["type"].endswith("/unauthenticated")
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)


def test_agent_readiness_with_wrong_scope_is_403() -> None:
    """A valid token lacking the ``agent`` scope is 403 insufficient-scope (AUTH-R7/R13).

    Authentication passes (a real, well-formed bearer with the ``read`` scope), so the
    failure is an authorization gap on the wired ``require_scopes(Scope.AGENT)`` gate — a
    403 listing the required scope, not a 401.
    """
    resp = _client().get(f"{API_PREFIX}/agent/readiness", headers=_auth(scopes=["read"]))
    assert resp.status_code == 403
    body = resp.json()
    assert body["type"].endswith("/insufficient-scope")
    # the machine-readable required-scope hint names the missing capability (AUTH-R7/R9)
    assert any(err.get("message") == "agent" for err in body.get("errors", []))


def test_agent_readiness_with_agent_scope_is_200_typed_verdict_no_kpi_no_thread() -> None:
    """A valid ``agent``-scoped token gets 200 with a typed verdict, no KPI, no thread (API-R41).

    Over the assembled factory (the agent engine falls back to the unconfigured no-model
    engine, so this is deterministic without a live model), an authorized readiness read
    returns the typed :class:`ReadinessResponse`: a verdict member (``null`` here, since the
    engine is unconfigured) and NO numeric readiness KPI, NO ``thread_id``, and NO
    continuation follow-up chip — readiness is a stateless typed verdict this phase.
    """
    resp = _client().get(f"{API_PREFIX}/agent/readiness", headers=_auth(scopes=["agent"]))
    assert resp.status_code == 200
    body = resp.json()
    # a typed verdict member (null when unassessable), never a number (API-R41 / COACH-R7)
    assert "verdict" in body
    assert body["verdict"] in (None, "go", "maintain", "ease", "rest")
    assert isinstance(body["summary_text"], str) and body["summary_text"]
    flat = json.dumps(body)
    for field in _FORBIDDEN_READINESS_KPI_FIELDS:
        assert f'"{field}"' not in flat, f"numeric readiness field {field!r} leaked (API-R41)"
    for field in _FORBIDDEN_READINESS_THREAD_FIELDS:
        assert f'"{field}"' not in flat, f"stateful thread/continuation field {field!r} leaked"
    # no continuation chip is promised on the stateless readiness surface this phase
    assert body.get("suggested_followups", []) == []
