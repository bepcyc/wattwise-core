"""The ``wattwise-core`` ASGI application factory.

:func:`create_app` assembles the single-owner ``/v1`` REST surface: it validates
configuration at boot (fail-closed), binds the shared :class:`Settings` and
:class:`Database` to app state, installs the uniform RFC 9457 error handlers, mounts
the always-present public endpoints (service status, liveness, OpenAPI), wires the
token-issuance endpoint, and includes whatever feature routers are registered.

Requirements realized here (doc 60 / doc 70):

- **API-R4** Everything is served under a single ``/v1`` prefix; the liveness probe
  (``GET /healthz``) is the one intentional out-of-version operational route.
- **AUTH-R10** Only the enumerated pre-token/public endpoints are unauthenticated —
  here: ``GET /v1/system/status``, ``GET /v1/openapi.json``, and ``POST
  /v1/auth/token``. Every feature router mounts behind the bearer + scope gates.
- **ERR-R1** The uniform ``application/problem+json`` handlers are installed so no
  raw framework error can escape (errors module).
- **OBS-R6.1** ``GET /healthz`` reflects only that the process/event-loop is alive; it
  depends on NO external service (a DB blip must not restart-loop the container). The
  Dockerfile health-check probes exactly this route.
- **DOC-R1** A complete OpenAPI 3.1 document is published (public) at
  ``GET /v1/openapi.json`` with human docs at ``GET /v1/docs``.
- **RUN-R4.1** Invalid configuration fails the boot closed (``create_app`` raises via
  the settings loader) rather than starting in an undefined state.
"""

from __future__ import annotations

import hmac
from importlib import import_module
from typing import Annotated, Final

from fastapi import APIRouter, Depends, FastAPI
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.api.auth import (
    Principal,
    Scope,
    authenticate,
    issue_access_token,
    require_scopes,
)
from wattwise_core.api.deps import AppSettings, get_db, get_rate_limiter
from wattwise_core.api.errors import ProblemError, install_error_handlers
from wattwise_core.api.middleware import JSONBodySizeLimitMiddleware
from wattwise_core.api.openapi import install_openapi
from wattwise_core.api.ratelimit import RateLimiter
from wattwise_core.api.routers import activities as activities_router
from wattwise_core.api.routers import agent_routes as agent_router
from wattwise_core.api.routers import performance as performance_router
from wattwise_core.config import Settings, get_settings
from wattwise_core.observability.logging import configure_logging
from wattwise_core.persistence import Database

#: The WWW-Authenticate challenge returned with an invalid sign-in (AUTH-R1/API-R23).
_BEARER_CHALLENGE: Final = {"WWW-Authenticate": "Bearer"}

#: The single major-version prefix every resource path is mounted under (API-R4).
API_PREFIX: Final = "/v1"

#: Public OpenAPI + human-docs locations (DOC-R1); both are unauthenticated.
_OPENAPI_PATH: Final = f"{API_PREFIX}/openapi.json"
_DOCS_PATH: Final = f"{API_PREFIX}/docs"

#: Scopes the OSS first-party token grants the single owner (every in-OSS capability).
_OWNER_SCOPES: Final = (
    Scope.READ,
    Scope.WRITE,
    Scope.AGENT,
    Scope.SYNC,
    Scope.EXPORT,
    Scope.ADMIN,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the configured ``wattwise-core`` ASGI app (fail-closed boot).

    Resolving settings validates configuration; an invalid/insecure config raises
    here (RUN-R4.1) so the process never serves traffic in an undefined state. The
    resolved settings and the shared :class:`Database` are bound to app state for the
    dependency providers; the uniform error handlers, public endpoints, and feature
    routers are then installed.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.app__log_level)

    app = FastAPI(
        title="WattWise",
        version="1",
        openapi_url=_OPENAPI_PATH,
        docs_url=_DOCS_PATH,
        redoc_url=None,
    )
    app.state.settings = resolved
    app.state.database = Database(resolved)
    app.state.rate_limiter = RateLimiter()

    app.add_middleware(JSONBodySizeLimitMiddleware)
    install_error_handlers(app)
    _mount_liveness(app)
    app.include_router(_public_router())
    app.include_router(_auth_router())
    register_routers(app)
    _wire_router_seams(app)
    install_openapi(app)
    return app


def _wire_router_seams(app: FastAPI) -> None:
    """Bind the seam-based routers' fail-closed dependencies to real providers (API-R3).

    The performance/activities/agent routers ship with override seams that fail closed
    (403/401/500) so a router mounted without wiring never serves ungated. The factory
    binds the REAL gates here so the surface is functional AND auth is actually enforced
    (AUTH-R1/R7): the scope gate runs the bearer+scope check, the acting athlete id is
    derived server-side from the verified principal (AUTH-R3), the analytics service is
    built per request from the shared DB session, and the agent rate limiter is the
    process limiter. The agent engine remains an injectable seam the runtime supplies
    (the OSS factory has no model config to construct it).
    """
    overrides = app.dependency_overrides
    overrides[performance_router.require_read_scope] = require_scopes(Scope.READ)
    overrides[performance_router.current_athlete_id] = _athlete_id_seam
    overrides[performance_router.analytics_service] = _analytics_seam
    overrides[activities_router.current_session] = get_db
    overrides[activities_router.cursor_signing_key] = _cursor_signing_key_seam
    overrides[agent_router.require_agent_scope] = require_scopes(Scope.AGENT)
    overrides[agent_router.current_athlete_id] = _athlete_id_seam
    overrides[agent_router.rate_limiter] = get_rate_limiter


def _athlete_id_seam(principal: Annotated[Principal, Depends(authenticate)]) -> str:
    """Server-derived acting athlete id from the verified principal (AUTH-R3)."""
    return principal.athlete_id


def _analytics_seam(session: Annotated[AsyncSession, Depends(get_db)]) -> AnalyticsService:
    """Build the request-scoped :class:`AnalyticsService` from the shared session."""
    return AnalyticsService(session)


def _cursor_signing_key_seam(settings: AppSettings) -> str:
    """The HMAC key the opaque pagination cursor is signed with (PAGE-R5).

    Bound to the engine ``token_signing_key`` so a cursor is tamper-evident and not
    client-constructible. An absent key is an operator misconfiguration surfaced
    fail-closed as a generic internal error (ERR-R5), never as a hint.
    """
    key = settings.token_signing_key
    if key is None:
        raise ProblemError("internal-error")
    return key.get_secret_value()


def _mount_liveness(app: FastAPI) -> None:
    """Mount ``GET /healthz`` — process/event-loop liveness only (OBS-R6.1).

    This route is deliberately OUTSIDE ``/v1``: it is an operational probe (the
    Dockerfile health-check calls it), not part of the versioned product contract. It
    returns ``200`` as long as the event loop can service the request and depends on
    NO external dependency, so a transient database/secret-store blip never triggers a
    restart loop. Readiness (which DOES check dependencies) is a distinct surface.
    """

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up and its event loop is responsive (OBS-R6.1)."""
        return {"status": "alive"}


def _public_router() -> APIRouter:
    """Build the router for the unauthenticated public endpoints (AUTH-R10).

    Currently the service-status summary, which carries NO per-user data and leaks no
    internal detail (OBS-R6.3): a self-hoster / load balancer can read it without a
    token. The OpenAPI document endpoints are published by FastAPI itself (DOC-R1).
    """
    router = APIRouter(prefix=API_PREFIX, tags=["system"])

    @router.get("/system/status", operation_id="getSystemStatus")
    async def system_status() -> dict[str, str]:
        """Public service-status summary — no per-user data, no internal detail."""
        return {"status": "ok", "service": "wattwise"}

    return router


class _TokenRequest(BaseModel):
    """The first-party sign-in exchange body for ``POST /v1/auth/token`` (API-R23).

    Carries ONLY the owner sign-in secret — the platform's first-party credential —
    and no caller-identity field (the subject is fixed server-side, AUTH-R3).
    ``additionalProperties:false`` (SCHEMA-R4) rejects any forged extra property.
    """

    model_config = ConfigDict(extra="forbid")

    owner_secret: str = Field(min_length=1, max_length=512)


def _verify_owner_secret(settings: Settings, presented: str) -> None:
    """Constant-time-verify the first-party owner secret, else ``401`` (API-R23).

    OSS is single-owner: the first-party credential is the configured
    ``token_signing_key`` secret (the operator's boot secret). A mismatch — or an
    absent configured secret — yields ``401 unauthenticated`` with NO unknown-user /
    wrong-secret distinction and no credential echo (API-R23 / AUTH-R9). Only a verified
    secret proceeds to mint a token (no fail-open issuance).
    """
    configured = settings.token_signing_key
    if configured is None or not hmac.compare_digest(
        presented.encode(), configured.get_secret_value().encode()
    ):
        raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE)


def _auth_router() -> APIRouter:
    """Build the public token-issuance router (``POST /v1/auth/token``, API-R23).

    OSS is single-owner: a successful sign-in exchange (a valid first-party credential)
    mints an access token scoped to the one owner with every in-OSS capability (the
    commercial layer narrows scopes per plan). The endpoint is public (pre-token,
    AUTH-R10) but NOT fail-open: an invalid/absent credential is rejected ``401``;
    refresh/revoke and the bot-link flow are mounted by the auth feature router.
    """
    router = APIRouter(prefix=f"{API_PREFIX}/auth", tags=["auth"])

    @router.post("/token", operation_id="issueToken")
    async def issue_token(body: _TokenRequest, settings: AppSettings) -> dict[str, object]:
        """Exchange the first-party owner credential for an access token (API-R23/R24)."""
        _verify_owner_secret(settings, body.owner_secret)
        tokens = issue_access_token(settings, subject="owner", scopes=_OWNER_SCOPES)
        return tokens.to_dict()

    return router


def register_routers(app: FastAPI) -> None:
    """Include every registered feature router, tolerating an empty registry.

    Feature routers are collected on the ``wattwise_core.api.routers`` package as a
    ``ROUTERS`` list (each an :class:`APIRouter`); the registry starts empty and is
    populated as feature slices land. This factory stays agnostic to which routers
    exist — it includes whatever is registered and is a no-op when none are, so the
    app boots with just the public + liveness surface during early bring-up. Each
    router already carries its own full ``/v1/...`` prefix, so it is included verbatim.
    """
    for router in _discover_routers():
        app.include_router(router)


def _discover_routers() -> tuple[APIRouter, ...]:
    """Return the registered feature routers from the routers package (or none).

    Reads an optional ``ROUTERS`` attribute (a sequence of :class:`APIRouter`). A
    missing attribute or a non-sequence value yields an empty tuple — the registry is
    intentionally permitted to be empty during bring-up.
    """
    package = import_module("wattwise_core.api.routers")
    registered = getattr(package, "ROUTERS", ())
    if not isinstance(registered, (list, tuple)):
        return ()
    return tuple(item for item in registered if isinstance(item, APIRouter))


__all__ = ["API_PREFIX", "create_app", "register_routers"]
