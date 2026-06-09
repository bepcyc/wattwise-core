"""Integration tests for the user-settings router (doc 60 §8.10 / API-R11f / API-R37).

Builds a minimal ASGI app that mounts the user-settings router and overrides its dependency
seams (server-derived identity AUTH-R3, ``read``/``write`` scopes AUTH-R11, the shared
session) against a seeded canonical store. Asserts that:

* each setting round-trips through its ``GET``/``PUT`` pair — the persisted answer-length
  (``response_length``), the language, the training zones, and the default load model — and
  a fresh ``GET`` reflects what the ``PUT`` stored (it is persisted, not merely echoed);
* defaults are honest — an unset answer-length reads ``standard`` and an unset language
  reads ``en`` (API-R11f/API-R37);
* an unsupported value is a ``422`` (an out-of-set ``response_length``/``language`` via the
  typed enum, and an out-of-set ``default_load_model`` via the LOAD-R2 set check) — never a
  silent accept;
* scope is enforced — a ``write`` ``PUT`` with only the ``read`` scope is ``403`` — and no
  end-user settings response carries an LLM model/tier/catalog control (API-R38).

Runs on in-memory SQLite (the portable substrate, GBO-R8b).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wattwise_core.api.auth import Principal, Scope, authenticate
from wattwise_core.api.errors import ProblemError, install_error_handlers
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import user_settings as settings_router
from wattwise_core.persistence.models import Athlete, Base

pytestmark = pytest.mark.integration

#: Model/tier/catalog tokens that MUST NOT appear on any end-user settings response (API-R38).
_FORBIDDEN_MODEL_FIELDS = (
    "model_tier", "reasoning", "model_name", "model_catalog", "flash", "frontier",
)


@dataclass
class Env:
    """The wired app + its client/session for one seeded scenario."""

    client: AsyncClient
    app: FastAPI
    session: AsyncSession
    athlete_id: str


def _insufficient_scope() -> None:
    """The unwired ``write`` seam stand-in: deny (used to assert the 403 path)."""
    raise ProblemError("insufficient-scope")


@pytest_asyncio.fixture
async def seeded() -> AsyncIterator[Env]:
    """An app wired to a seeded canonical store with exactly one owner (no preferences set)."""
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
        app = _build_app(session, athlete_id, write_allowed=True)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as client:
            yield Env(client, app, session, athlete_id)
    await engine.dispose()


def _build_app(session: AsyncSession, athlete_id: str, *, write_allowed: bool) -> FastAPI:
    """Mount the user-settings router and override the identity/scope/session seams."""
    app = FastAPI()
    app.state.rate_limiter = RateLimiter()  # the per-athlete read/write buckets (LIMIT-R1)
    install_error_handlers(app)
    app.include_router(settings_router.router)
    write_seam = (lambda: None) if write_allowed else _insufficient_scope
    app.dependency_overrides.update(
        {
            # The router attaches the per-subject RateLimit gate, which derives identity from
            # ``authenticate`` (AUTH-R18); bind it to the seeded owner so the bucket is keyed
            # server-side, mirroring the assembled app's wiring (LIMIT-R1/R6).
            authenticate: lambda: Principal(subject=athlete_id, scopes=frozenset(Scope)),
            settings_router.require_read_scope: lambda: None,
            settings_router.require_write_scope: write_seam,
            settings_router.current_athlete_id: lambda: athlete_id,
            settings_router.current_session: lambda: session,
        }
    )
    return app


# --- §8.10 response-length (the persisted answer-length default, API-R11f) --------


async def test_response_length_defaults_to_standard(seeded: Env) -> None:
    """An unset answer-length reads the honest default ``standard`` (API-R11f)."""
    resp = await seeded.client.get("/v1/user-settings/response-length")
    assert resp.status_code == 200
    assert resp.json()["response_length"] == "standard"


async def test_set_and_get_response_length(seeded: Env) -> None:
    """PUT response-length persists; a fresh GET reflects the stored value (API-R11f)."""
    put = await seeded.client.put(
        "/v1/user-settings/response-length", json={"response_length": "detailed"}
    )
    assert put.status_code == 200
    assert put.json()["response_length"] == "detailed"
    again = await seeded.client.get("/v1/user-settings/response-length")
    assert again.json()["response_length"] == "detailed"


async def test_unsupported_response_length_is_422(seeded: Env) -> None:
    """An out-of-set answer-length (the former ``concise``) is rejected 422 (API-R11f)."""
    resp = await seeded.client.put(
        "/v1/user-settings/response-length", json={"response_length": "concise"}
    )
    assert resp.status_code == 422
    assert resp.json()["type"].endswith("/validation-error")


async def test_response_length_carries_no_model_machinery(seeded: Env) -> None:
    """No model/tier/catalog control appears on the answer-length surface (API-R38/API-R11c)."""
    await seeded.client.put(
        "/v1/user-settings/response-length", json={"response_length": "short"}
    )
    flat = json.dumps((await seeded.client.get("/v1/user-settings/response-length")).json())
    for field in _FORBIDDEN_MODEL_FIELDS:
        assert field not in flat, f"model-selection token {field!r} leaked (API-R38)"


# --- §8.10 language --------------------------------------------------------------


async def test_language_defaults_to_en(seeded: Env) -> None:
    """An unset language reads the default ``en`` (API-R37)."""
    resp = await seeded.client.get("/v1/user-settings/language")
    assert resp.status_code == 200
    assert resp.json()["language"] == "en"


async def test_set_and_get_language(seeded: Env) -> None:
    """PUT language persists; a fresh GET reflects the stored value (API-R37)."""
    put = await seeded.client.put("/v1/user-settings/language", json={"language": "de"})
    assert put.status_code == 200
    assert put.json()["language"] == "de"
    assert (await seeded.client.get("/v1/user-settings/language")).json()["language"] == "de"


async def test_unsupported_language_is_422(seeded: Env) -> None:
    """A PUT of an unsupported language is rejected 422 (API-R37)."""
    resp = await seeded.client.put("/v1/user-settings/language", json={"language": "fr"})
    assert resp.status_code == 422


# --- §8.10 zones -----------------------------------------------------------------


async def test_zones_round_trip(seeded: Env) -> None:
    """PUT zones persists a today-effective TrainingZoneSet that GET then reflects (GBO-R13d)."""
    zones = {
        "kind": "power",
        "basis": "absolute",
        "boundaries": [
            {"zone_index": 0, "label": "Z1", "lower": 0.0, "upper": 150.0},
            {"zone_index": 1, "label": "Z2", "lower": 150.0, "upper": 250.0},
        ],
    }
    put = await seeded.client.put("/v1/user-settings/zones", json=zones)
    assert put.status_code == 200
    got = await seeded.client.get("/v1/user-settings/zones")
    assert got.status_code == 200
    body = got.json()
    assert body["kind"] == "power" and body["basis"] == "absolute"
    assert [b["label"] for b in body["boundaries"]] == ["Z1", "Z2"]


async def test_zones_default_empty(seeded: Env) -> None:
    """An owner with no zones set reads an empty boundary list, never an error (GBO-R13d)."""
    resp = await seeded.client.get("/v1/user-settings/zones")
    assert resp.status_code == 200
    assert resp.json()["boundaries"] == []


# --- §8.10 default load model (LOAD-R2 set; NOT a model tier) ---------------------


async def test_default_load_model_round_trip(seeded: Env) -> None:
    """PUT default-load-model persists a LOAD-R2 member that GET reflects."""
    put = await seeded.client.put(
        "/v1/user-settings/default-load-model", json={"default_load_model": "hr_load_zonal"}
    )
    assert put.status_code == 200
    got = await seeded.client.get("/v1/user-settings/default-load-model")
    assert got.json()["default_load_model"] == "hr_load_zonal"


async def test_default_load_model_rejects_non_load_r2_token(seeded: Env) -> None:
    """A token outside the LOAD-R2 set is rejected 422 unsupported_load_model (API-R38)."""
    resp = await seeded.client.put(
        "/v1/user-settings/default-load-model", json={"default_load_model": "gpt5"}
    )
    assert resp.status_code == 422
    assert any(e.get("code") == "unsupported_load_model" for e in resp.json().get("errors", []))


async def test_default_load_model_null_clears(seeded: Env) -> None:
    """PUT null clears the preference so the automatic LOAD-R3 selection applies."""
    await seeded.client.put(
        "/v1/user-settings/default-load-model", json={"default_load_model": "power_tss"}
    )
    cleared = await seeded.client.put(
        "/v1/user-settings/default-load-model", json={"default_load_model": None}
    )
    assert cleared.status_code == 200
    assert cleared.json()["default_load_model"] is None


# --- AUTH-R11: write scope enforcement -------------------------------------------


async def test_writes_without_write_scope_are_403(seeded: Env) -> None:
    """Every settings PUT with only the read scope is 403 insufficient-scope (AUTH-R7/R11)."""
    no_write = _build_app(seeded.session, seeded.athlete_id, write_allowed=False)
    async with AsyncClient(transport=ASGITransport(app=no_write), base_url="http://t") as client:
        length = await client.put(
            "/v1/user-settings/response-length", json={"response_length": "short"}
        )
        lang = await client.put("/v1/user-settings/language", json={"language": "de"})
        loadm = await client.put(
            "/v1/user-settings/default-load-model", json={"default_load_model": "power_tss"}
        )
        # reads still work without write
        read_ok = await client.get("/v1/user-settings/response-length")
    assert length.status_code == 403
    assert length.json()["type"].endswith("/insufficient-scope")
    assert lang.status_code == 403
    assert loadm.status_code == 403
    assert read_ok.status_code == 200
