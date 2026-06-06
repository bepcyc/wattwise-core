"""Unit tests for the API core: app factory, bearer auth, scopes, and errors.

Covers the load-bearing security + error invariants of the ``/v1`` surface that the
factory, auth, errors, and deps modules must uphold (doc 60):

- **AUTH-R1** a protected route without a token -> ``401`` + ``WWW-Authenticate: Bearer``.
- **AUTH-R7** a token missing the required scope -> ``403 insufficient-scope`` listing
  the required scopes.
- **AUTH-R3/R18** no ``/v1`` request schema carries a writable caller-identity field;
  a client-supplied ``athlete_id``/``user_id`` is never trusted to widen access.
- **ERR-R1/R2/R3/R4** every non-2xx is one ``application/problem+json`` document with
  the six required members and a stable catalog ``type`` URI.
- **ERR-R6** request-validation failure -> ``422 validation-error`` with ``errors[]``.
- **OBS-R6.1** ``GET /healthz`` returns ``200`` (liveness; no external dependency).
- **DOC-R1** the OpenAPI document is published (public) at ``GET /v1/openapi.json``.
- **AUTH-R6** an expired token -> ``401`` (verification runs every request).

Tier: T-UNIT (offline, in-process ASGI via the FastAPI ``TestClient``; no real DB
connection is opened — the routes under test never touch persistence).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient
from pydantic import BaseModel

from wattwise_core.api.app import API_PREFIX, create_app
from wattwise_core.api.auth import (
    TOKEN_ALGORITHM,
    TOKEN_AUDIENCE,
    TOKEN_ISSUER,
    Principal,
    Scope,
)
from wattwise_core.api.deps import require_scope
from wattwise_core.api.errors import PROBLEM_BASE_URI, PROBLEM_MEDIA_TYPE
from wattwise_core.config import Settings, load_settings

pytestmark = pytest.mark.unit

_SIGNING_KEY = "unit-test-signing-key-not-a-real-secret"


def _settings() -> Settings:
    """Build dev settings with an in-memory DSN + a deterministic signing key."""
    return load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key=_SIGNING_KEY,
    )


def _token(*, scopes: list[str], ttl_seconds: int = 3600, subject: str = "owner") -> str:
    """Mint a signed access token directly (mirrors the issuer) for a given scope set."""
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        "sub": subject,
        "scope": " ".join(scopes),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
    }
    return jwt.encode(payload, _SIGNING_KEY, algorithm=TOKEN_ALGORITHM)


class _Echo(BaseModel):
    """A trivial request body used to probe for a writable caller-identity field."""

    note: str


# Module-level annotated dependencies so FastAPI's type-hint resolution sees them.
_ReadPrincipal = Annotated[Principal, Depends(require_scope(Scope.READ))]
_WritePrincipal = Annotated[Principal, Depends(require_scope(Scope.READ, Scope.WRITE))]


def _app_with_probe_routes() -> TestClient:
    """Build the app and attach read/write probe routes guarded by the scope gate."""
    app = create_app(_settings())
    router = APIRouter(prefix=API_PREFIX, tags=["test"])

    @router.get("/_probe/read", operation_id="probeRead")
    async def probe_read(principal: _ReadPrincipal) -> dict[str, str]:
        """A read route requiring the ``read`` scope; echoes the resolved subject."""
        return {"subject": principal.subject}

    @router.post("/_probe/write", operation_id="probeWrite")
    async def probe_write(body: _Echo, principal: _WritePrincipal) -> dict[str, str]:
        """A write route requiring ``read``+``write``; echoes the server subject only."""
        return {"subject": principal.subject, "note": body.note}

    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------- auth


def test_protected_route_without_token_is_401_with_challenge() -> None:
    """A protected route with no bearer token -> 401 + WWW-Authenticate: Bearer (AUTH-R1)."""
    client = _app_with_probe_routes()
    resp = client.get(f"{API_PREFIX}/_probe/read")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    body = resp.json()
    assert body["type"] == f"{PROBLEM_BASE_URI}unauthenticated"


def test_token_missing_scope_is_403_insufficient_scope() -> None:
    """A token lacking the required scope -> 403 insufficient-scope listing it (AUTH-R7)."""
    client = _app_with_probe_routes()
    token = _token(scopes=["read"])  # has read, lacks write
    resp = client.post(
        f"{API_PREFIX}/_probe/write",
        headers={"Authorization": f"Bearer {token}"},
        json={"note": "hi"},
    )
    assert resp.status_code == 403
    body = resp.json()
    assert body["type"] == f"{PROBLEM_BASE_URI}insufficient-scope"
    required = {e["message"] for e in body["errors"]}
    assert "write" in required


def test_valid_scope_passes_and_subject_is_server_derived() -> None:
    """A token with the right scope passes; the subject comes from the token (AUTH-R3)."""
    client = _app_with_probe_routes()
    token = _token(scopes=["read"])
    resp = client.get(
        f"{API_PREFIX}/_probe/read", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"subject": "owner"}


def test_expired_token_is_401() -> None:
    """An expired token -> 401 (signature/expiry verified every request, AUTH-R6)."""
    client = _app_with_probe_routes()
    token = _token(scopes=["read"], ttl_seconds=-10)
    resp = client.get(
        f"{API_PREFIX}/_probe/read", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 401
    assert resp.json()["type"] == f"{PROBLEM_BASE_URI}unauthenticated"


def test_non_bearer_scheme_is_401() -> None:
    """A non-bearer Authorization scheme is rejected as unauthenticated (AUTH-R2)."""
    client = _app_with_probe_routes()
    resp = client.get(
        f"{API_PREFIX}/_probe/read", headers={"Authorization": "Basic abc123"}
    )
    assert resp.status_code == 401


def test_wrong_audience_token_is_401() -> None:
    """A token signed for a different audience fails verification -> 401 (AUTH-R6)."""
    client = _app_with_probe_routes()
    now = datetime.now(UTC)
    bad = jwt.encode(
        {
            "iss": TOKEN_ISSUER,
            "aud": "someone-else",
            "sub": "owner",
            "scope": "read",
            "exp": int((now + timedelta(hours=1)).timestamp()),
        },
        _SIGNING_KEY,
        algorithm=TOKEN_ALGORITHM,
    )
    resp = client.get(
        f"{API_PREFIX}/_probe/read", headers={"Authorization": f"Bearer {bad}"}
    )
    assert resp.status_code == 401


# ------------------------------------------------------------- caller-identity ban


def test_client_supplied_identity_is_not_trusted() -> None:
    """A body carrying user_id/athlete_id never widens access; subject is server-side (AUTH-R3)."""
    client = _app_with_probe_routes()
    token = _token(scopes=["read", "write"], subject="owner")
    resp = client.post(
        f"{API_PREFIX}/_probe/write",
        headers={"Authorization": f"Bearer {token}"},
        json={"note": "hi", "athlete_id": "impostor", "user_id": "impostor"},
    )
    assert resp.status_code == 200
    # The resolved subject is the token's, never the forged body fields (AUTH-R3).
    assert resp.json()["subject"] == "owner"


def test_no_request_schema_exposes_caller_identity_field() -> None:
    """No request body schema in the OpenAPI doc declares a caller-identity field (AUTH-R3)."""
    client = _app_with_probe_routes()
    schema = client.get(f"{API_PREFIX}/openapi.json").json()
    banned = {"athlete_id", "user_id", "subject", "owner_id", "principal_id"}
    for name, component in schema.get("components", {}).get("schemas", {}).items():
        props = set(component.get("properties", {}))
        assert banned.isdisjoint(props), f"schema {name} exposes a caller-identity field"


# --------------------------------------------------------------------------- errors


def test_validation_error_is_422_problem_with_errors_pointer() -> None:
    """A malformed body -> 422 validation-error with a populated errors[] pointer (ERR-R6)."""
    client = _app_with_probe_routes()
    token = _token(scopes=["read", "write"])
    resp = client.post(
        f"{API_PREFIX}/_probe/write",
        headers={"Authorization": f"Bearer {token}"},
        json={},  # missing required "note"
    )
    assert resp.status_code == 422
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    body = resp.json()
    assert body["type"] == f"{PROBLEM_BASE_URI}validation-error"
    assert body["errors"]
    assert any(e.get("pointer") == "/note" for e in body["errors"])


def test_problem_document_has_all_required_members() -> None:
    """Every non-2xx carries type/title/status/detail/instance/trace_id (ERR-R2/R4)."""
    client = _app_with_probe_routes()
    resp = client.get(f"{API_PREFIX}/_probe/read")
    body = resp.json()
    for member in ("type", "title", "status", "detail", "instance", "trace_id"):
        assert member in body, f"missing required problem member {member!r}"
    assert body["status"] == resp.status_code  # ERR-R4: body status mirrors the line
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert body["instance"] == f"{API_PREFIX}/_probe/read"
    assert isinstance(body["trace_id"], str) and body["trace_id"]


def test_unknown_route_is_uniform_not_found_problem() -> None:
    """An unmatched route returns the uniform not-found problem, not a default body (ERR-R1)."""
    client = _app_with_probe_routes()
    resp = client.get(f"{API_PREFIX}/does/not/exist")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert resp.json()["type"] == f"{PROBLEM_BASE_URI}not-found"


# --------------------------------------------------------------- public + liveness


def test_healthz_is_200_liveness() -> None:
    """GET /healthz returns 200 with no token (process liveness only, OBS-R6.1)."""
    client = _app_with_probe_routes()
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_system_status_is_public_and_carries_no_user_data() -> None:
    """GET /v1/system/status is public and exposes no per-user data (AUTH-R10)."""
    client = _app_with_probe_routes()
    resp = client.get(f"{API_PREFIX}/system/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "athlete_id" not in body and "subject" not in body


def test_openapi_document_is_public() -> None:
    """GET /v1/openapi.json is served publicly as a valid OpenAPI 3.x doc (DOC-R1)."""
    client = _app_with_probe_routes()
    resp = client.get(f"{API_PREFIX}/openapi.json")
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["openapi"].startswith("3.")
    assert f"{API_PREFIX}/system/status" in doc["paths"]


def test_token_issuance_returns_auth_tokens_shape() -> None:
    """POST /v1/auth/token (public) issues an AuthTokens body usable for auth (API-R23)."""
    client = _app_with_probe_routes()
    resp = client.post(f"{API_PREFIX}/auth/token")
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    assert "read" in body["scopes"]
    # The issued token actually authenticates against a protected route.
    authed = client.get(
        f"{API_PREFIX}/_probe/read",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert authed.status_code == 200
