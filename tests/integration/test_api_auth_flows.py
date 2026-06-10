"""Integration tests for the auth feature router (§7A, API-R23 / AUTH-R8 / AUTH-R10).

Drives the full ``/v1/auth`` surface over the assembled app with the REAL token issuance,
the REAL agent-state-backed refresh-token families, and the REAL link-challenge flow:

- ``/token`` issues ``AuthTokens`` with a REAL rotating ``refresh_token`` (API-R23).
- ``/refresh`` rotates single-use; REPLAY of a consumed token -> ``401`` AND revokes the
  whole family (reuse detection) — checklist item 12 (API-R20).
- ``/revoke`` -> ``204`` and the family can no longer refresh.
- ``/link/start`` (public) mints a short-lived single-use ``LinkChallenge``; an
  authenticated ``/link/approve`` is the proof-of-control step (AUTH-R8); a proven code
  redeems ONCE via ``/link/complete`` into delegated tokens WITHOUT the admin scope;
  an unproven/used code -> ``409``; a forged code -> ``401``.
- Token responses and link challenges follow AUTH-R9/API-R24 (no internals, no secrets
  beyond the issued credential).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from wattwise_core.api.app import create_app
from wattwise_core.config import Environment, load_settings

pytestmark = pytest.mark.integration

_SIGNING_KEY = "auth-flows-test-signing-key-0123456789"


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    """The assembled app over a file-backed store (real pool — never :memory: races)."""
    settings = load_settings(
        app__environment=Environment.DEVELOPMENT,
        database_dsn=f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}",
        token_signing_key=_SIGNING_KEY,
    )
    app: FastAPI = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def _sign_in(client: TestClient) -> dict[str, object]:
    """A successful first-party sign-in -> the AuthTokens payload (API-R23)."""
    resp = client.post("/v1/auth/token", json={"owner_secret": _SIGNING_KEY})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_token_issues_real_rotating_refresh_token(client: TestClient) -> None:
    """/token returns a NON-empty refresh_token alongside the access token (API-R23)."""
    body = _sign_in(client)
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert isinstance(body["refresh_token"], str) and len(body["refresh_token"]) >= 32
    # AUTH-R9 / API-R24: no object contents / internal ids beyond the credential itself.
    assert set(body) == {"access_token", "token_type", "expires_in", "refresh_token", "scopes"}


def test_refresh_rotates_and_replay_revokes_the_family(client: TestClient) -> None:
    """The refresh token is single-use; replay -> 401 AND the family dies (API-R23)."""
    first = _sign_in(client)["refresh_token"]
    rotated = client.post("/v1/auth/refresh", json={"refresh_token": first})
    assert rotated.status_code == 200, rotated.text
    second = rotated.json()["refresh_token"]
    assert second and second != first  # rotation minted a NEW opaque secret
    # REPLAY of the consumed first token -> 401 + the whole family is revoked.
    replay = client.post("/v1/auth/refresh", json={"refresh_token": first})
    assert replay.status_code == 401
    assert replay.json()["type"].endswith("/unauthenticated")
    # The successor (same family) is dead too — reuse detection revoked the family.
    after = client.post("/v1/auth/refresh", json={"refresh_token": second})
    assert after.status_code == 401


def test_revoke_kills_the_family_and_returns_204(client: TestClient) -> None:
    """/revoke -> 204; the revoked token (and family) can no longer refresh (API-R23)."""
    token = _sign_in(client)["refresh_token"]
    revoked = client.post("/v1/auth/revoke", json={"refresh_token": token})
    assert revoked.status_code == 204
    assert revoked.content == b""
    dead = client.post("/v1/auth/refresh", json={"refresh_token": token})
    assert dead.status_code == 401


def test_refresh_unknown_token_is_401_without_detail(client: TestClient) -> None:
    """An unknown/forged refresh token -> 401 with no distinguishing detail (AUTH-R9)."""
    resp = client.post("/v1/auth/refresh", json={"refresh_token": "forged-token"})
    assert resp.status_code == 401
    assert "forged-token" not in resp.text  # the credential is never echoed


def test_link_flow_start_approve_complete_mints_delegated_tokens(
    client: TestClient,
) -> None:
    """start (public) -> approve (bearer proof of control) -> complete (AUTH-R8)."""
    owner = _sign_in(client)
    started = client.post("/v1/auth/link/start")
    assert started.status_code == 200, started.text
    challenge = started.json()
    assert set(challenge) == {"link_code", "expires_at"}  # AUTH-R9: nothing else leaks
    code = challenge["link_code"]
    # an UNPROVEN code cannot redeem (the code alone is not a credential, API-R23)
    early = client.post("/v1/auth/link/complete", json={"link_code": code})
    assert early.status_code == 409
    # the AUTHENTICATED owner approves it in-app — the proof-of-control step (AUTH-R8)
    approved = client.post(
        "/v1/auth/link/approve",
        json={"link_code": code},
        headers={"Authorization": f"Bearer {owner['access_token']}"},
    )
    assert approved.status_code == 200, approved.text
    # the external client redeems the PROVEN code exactly once
    completed = client.post("/v1/auth/link/complete", json={"link_code": code})
    assert completed.status_code == 200, completed.text
    delegated = completed.json()
    assert delegated["access_token"] and delegated["refresh_token"]
    assert "admin" not in delegated["scopes"]  # delegated client is never the operator
    # single-use: a second redemption of the same code -> 409 conflict
    again = client.post("/v1/auth/link/complete", json={"link_code": code})
    assert again.status_code == 409


def test_link_complete_forged_code_is_401(client: TestClient) -> None:
    """A forged/unknown link code -> 401 (API-R23); nothing is minted."""
    resp = client.post("/v1/auth/link/complete", json={"link_code": "forged-code"})
    assert resp.status_code == 401


def test_link_approve_requires_bearer(client: TestClient) -> None:
    """The proof-of-control approval is bearer-only — anonymous approval is 401 (AUTH-R8)."""
    started = client.post("/v1/auth/link/start")
    resp = client.post("/v1/auth/link/approve", json={"link_code": started.json()["link_code"]})
    assert resp.status_code == 401
