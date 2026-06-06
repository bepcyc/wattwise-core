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

from importlib import import_module
from typing import Final

from fastapi import APIRouter, FastAPI

from wattwise_core.api.auth import Scope, issue_access_token
from wattwise_core.api.deps import AppSettings
from wattwise_core.api.errors import install_error_handlers
from wattwise_core.config import Settings, get_settings
from wattwise_core.observability.logging import configure_logging
from wattwise_core.persistence import Database

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

    install_error_handlers(app)
    _mount_liveness(app)
    app.include_router(_public_router())
    app.include_router(_auth_router())
    register_routers(app)
    return app


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


def _auth_router() -> APIRouter:
    """Build the public token-issuance router (``POST /v1/auth/token``, API-R23).

    OSS is single-owner: a successful issuance mints an access token scoped to the one
    owner with every in-OSS capability (the commercial layer narrows scopes per plan).
    The endpoint is public (pre-token, AUTH-R10); refresh/revoke and the bot-link flow
    are mounted by the auth feature router and are NOT public.
    """
    router = APIRouter(prefix=f"{API_PREFIX}/auth", tags=["auth"])

    @router.post("/token", operation_id="issueToken")
    async def issue_token(settings: AppSettings) -> dict[str, object]:
        """Issue a first-party access token for the single owner (API-R23/R24)."""
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
