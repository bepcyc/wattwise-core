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

import uuid
from collections.abc import Callable
from importlib import import_module
from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.engine import UnconfiguredAgentEngine, build_agent_engine
from wattwise_core.agent.state_db import build_agent_state_database
from wattwise_core.analytics.service import AnalyticsService
from wattwise_core.api.auth import Principal, Scope, authenticate, require_scopes
from wattwise_core.api.deps import AppSettings, get_db, get_master_data_db, get_rate_limiter
from wattwise_core.api.errors import ProblemError, install_error_handlers
from wattwise_core.api.lifecycle import build_lifespan
from wattwise_core.api.middleware import (
    EndpointMetricsMiddleware,
    JSONBodySizeLimitMiddleware,
    RequestContextMiddleware,
)
from wattwise_core.api.openapi import install_openapi
from wattwise_core.api.ratelimit import LimitClass, RateLimiter
from wattwise_core.api.routers import activities as activities_router
from wattwise_core.api.routers import agent_breadth as agent_breadth_router
from wattwise_core.api.routers import agent_routes as agent_router
from wattwise_core.api.routers import athlete as athlete_router
from wattwise_core.api.routers import connections as connections_router
from wattwise_core.api.routers import goals as goals_router
from wattwise_core.api.routers import imports as imports_router
from wattwise_core.api.routers import performance as performance_router
from wattwise_core.api.routers import planning as planning_router
from wattwise_core.api.routers import sync as sync_router
from wattwise_core.api.routers import user_settings as user_settings_router
from wattwise_core.api.routers import users as users_router
from wattwise_core.api.security import (
    agent_feature_gate,
    build_deletion_requester,
    install_security_middleware,
    mount_readiness,
    resolve_entitlement,
)
from wattwise_core.api.wiring import build_ingestion_seams
from wattwise_core.config import Settings, get_settings
from wattwise_core.entitlement import OssEntitlementResolver, validate_plan
from wattwise_core.identity import OWNER_SUBJECT
from wattwise_core.observability.logging import configure_logging
from wattwise_core.observability.metrics import get_registry
from wattwise_core.persistence import Database
from wattwise_core.persistence.models import Athlete

#: The single major-version prefix every resource path is mounted under (API-R4).
API_PREFIX: Final = "/v1"

#: Public OpenAPI + human-docs locations (DOC-R1); both are unauthenticated.
_OPENAPI_PATH: Final = f"{API_PREFIX}/openapi.json"
_DOCS_PATH: Final = f"{API_PREFIX}/docs"


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
        lifespan=build_lifespan(),  # RUN-R11: drain-marked shutdown + clean pool close
    )
    app.state.settings = resolved
    app.state.database = Database(resolved)
    # The MASTER-DATA-WRITE role's Database (ARCH-R3b / DEPLOY-R4): when the deployment
    # configures the distinct role DSN, the API's master-data endpoints (profile/signature,
    # zones/language/default-load-model user-settings, goals) open their sessions on it — a
    # credential that can write ONLY the master-data tables (reciprocal denial). Unset, it IS
    # the shared canonical Database (single-operator self-host: one credential, one pool —
    # the structural split is an opt-in deploy choice, and a second pool on e.g. an in-memory
    # SQLite dev DSN would otherwise see a different, empty database).
    if resolved.database_master_data_dsn is not None:
        app.state.master_data_database = Database(resolved, dsn=resolved.master_data_write_dsn())
    else:
        app.state.master_data_database = app.state.database
    app.state.rate_limiter = _build_rate_limiter(resolved)
    # The dedicated agent-state store for OPERATIONAL API state (amended ARCH-R13):
    # refresh-token families + link challenges (API-R23), export jobs + signed-URL
    # nonces (API-R34), and import-job rows (API-R33) — its OWN engine/pool, never the
    # canonical Database. Schema is ensured lazily on first use (a no-op when migrated).
    app.state.agent_state_db = build_agent_state_database(resolved)
    app.state.agent_state_schema_ready = False
    # Resolve + VALIDATE the OSS default entitlement plan at boot (ENT-4 / AGT-ENT-R4): an
    # invalid/missing plan fails the boot CLOSED here (RUN-R4.1) — the engine never starts
    # under a silently-permissive or unvalidated plan. The resolver + the validated plan are
    # bound to app state so the HTTP gate resolves -> attaches -> checks (AGT-ENT-R1/-R3) and
    # the readiness probe (OBS-R6.2) can confirm they are ready.
    app.state.entitlement_resolver = OssEntitlementResolver.from_settings(resolved)
    app.state.entitlement_plan = validate_plan(
        app.state.entitlement_resolver.resolve(OWNER_SUBJECT)
    )

    install_security_middleware(app, resolved)
    app.add_middleware(JSONBodySizeLimitMiddleware)
    # LOG-R3: bind/clear the per-request correlation context (request_id) around every
    # request so each log line emitted while serving it carries the correlation ids.
    app.add_middleware(RequestContextMiddleware)
    # Added LAST so it is the OUTERMOST middleware: it times the WHOLE request (including a
    # body-size rejection or a security-middleware short-circuit) and reads the matched route the
    # router binds into the shared scope, recording the OBS-R5 per-endpoint request/latency/error
    # metrics onto the /metrics surface for every request.
    app.add_middleware(EndpointMetricsMiddleware)
    install_error_handlers(app)
    _mount_liveness(app)
    _mount_metrics(app)
    mount_readiness(app)
    app.include_router(_public_router())
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
    process limiter. The athlete-profile and user-settings routers bind the same
    server-derived identity + ``read``/``write`` scope gates over the MASTER-DATA session
    (the master-data-write role's Database — ARCH-R3b/DEPLOY-R4), so
    the owner can set their profile (sex/timezone/current sport), their FTP fitness
    signature (the threshold the power analytics ground on), and their preferences
    (zones/language/answer-length/default load model) behind the bearer + scope gates.
    The agent engine is built from settings (the LangGraph coach when a
    model is configured, else a graceful unconfigured fallback) so /v1/agent is live. The
    connect/import/sync routers' seams are bound to the real OSS ingestion services (the
    file-upload import processor, the on-demand sync orchestrator, and the credential
    store) via :func:`wattwise_core.api.wiring.build_ingestion_seams` so the built stack
    can connect → sync → land canonical data (ARCH-R22 — routers stay source-blind). The
    planning router's agent-path + read-view seams are bound by :func:`_wire_planning` to the
    SAME server-derived identity, scope gates, agent engine, session, cursor key, and process
    limiter (ARCH-R21). The agent-breadth surfaces (diagnose / digest / memory) reuse the agent
    router's identity/scope/engine/limiter overrides; only the breadth-local ``current_session``
    is bound here to the shared session so the digest CRUD + email-verified gate are live (H1).
    The users router self-wires its read/write surfaces through the shared ``deps`` providers
    (server-derived principal + scopes + session + rate limit); its deletion-erasure seam is bound
    here to a DURABLE audit-log-backed async recorder so DELETE /v1/users/me records a
    ``pending_deletion`` erasure request (PRIV-R8 / retention §11) rather than 500-ing on its
    fail-closed default (H4) — it never hard-deletes inline.
    """
    overrides = app.dependency_overrides
    overrides[performance_router.require_read_scope] = require_scopes(Scope.READ)
    overrides[performance_router.current_athlete_id] = _athlete_id_seam
    overrides[performance_router.analytics_service] = _analytics_seam
    overrides[activities_router.current_session] = get_db
    overrides[activities_router.cursor_signing_key] = _cursor_signing_key_seam
    overrides[athlete_router.require_read_scope] = require_scopes(Scope.READ)
    overrides[athlete_router.require_write_scope] = require_scopes(Scope.WRITE)
    overrides[athlete_router.current_athlete_id] = _athlete_id_seam
    # The athlete-profile / user-settings / goals routers ARE the API's master-data mutation
    # surface (ARCH-R3b: profile + fitness signature, zones/language/default-load-model,
    # goals), so their sessions open on the MASTER-DATA-WRITE role's Database (DEPLOY-R4) —
    # under a per-role deployment that credential can write ONLY the master-data tables and is
    # read-only on the source-derived canonical tables it joins (e.g. the sport registry).
    overrides[athlete_router.current_session] = get_master_data_db
    overrides[user_settings_router.require_read_scope] = require_scopes(Scope.READ)
    overrides[user_settings_router.require_write_scope] = require_scopes(Scope.WRITE)
    overrides[user_settings_router.current_athlete_id] = _athlete_id_seam
    overrides[user_settings_router.current_session] = get_master_data_db
    # The Goals CRUD surface (/v1/goals, API-R35) binds the SAME server-derived identity + read/
    # write scope gates over the MASTER-DATA session (goals are athlete-authored master-data,
    # ARCH-R3b), the engine-keyed signed cursor (PAGE-R5), and the process limiter so the owner
    # can author the training goals the agent plans toward (GBO-R38).
    overrides[goals_router.require_read_scope] = require_scopes(Scope.READ)
    overrides[goals_router.require_write_scope] = require_scopes(Scope.WRITE)
    overrides[goals_router.current_athlete_id] = _athlete_id_seam
    overrides[goals_router.current_session] = get_master_data_db
    overrides[goals_router.rate_limiter] = get_rate_limiter
    overrides[goals_router.cursor_signing_key] = _cursor_signing_key_seam
    # The agent gate is the CHECK half of the entitlement seam: it runs the bearer+agent-scope
    # gate AND reads the resolved entitlement, failing closed when ``can_use_agent`` is ungranted
    # (AGT-ENT-R3). Under the OSS all-permissive plan it permits; a commercial plan that ungrants
    # the feature is enforced here without touching the agent router (resolve -> attach -> check).
    overrides[agent_router.require_agent_scope] = agent_feature_gate
    # The persisted language default (API-R37): the agent + planning surfaces resolve the
    # response language body -> Accept-Language -> the stored ``athlete.primary_locale``
    # subtag -> ``en``; this binds the seam to the canonical store reader (one override —
    # planning re-uses the same seam object).
    overrides[agent_router.persisted_locale] = _persisted_locale_seam
    overrides[agent_router.current_athlete_id] = _athlete_id_seam
    overrides[agent_router.rate_limiter] = get_rate_limiter
    engine = _build_engine(app)
    overrides[agent_router.agent_engine] = lambda: engine
    # The persisted response-length preference is an agent-interaction preference in the AGENT-STATE
    # store (doc 50 VOICE-R8 §382 / MEM-R1), NOT canonical master-data like language/zones — so its
    # GET/PUT reach the SAME shared engine the run path reads its default from, NOT the canonical
    # ``current_session``. Binding it to ``engine`` makes the value the athlete sets the value a run
    # applies (the VOICE-R8 store-split single source); the other user-settings stay canonical.
    overrides[user_settings_router.response_length_store] = lambda: engine
    # The breadth surfaces (diagnose / digest / memory) reuse the agent router's identity/scope/
    # engine/limiter overrides (above); only the breadth-local DB-session seam is new — bind it to
    # the shared transactional session so digest persistence + the email-verified gate are live
    # (H1: an unwired ``current_session`` fails closed and 500s the digest CRUD).
    overrides[agent_breadth_router.current_session] = get_db
    # The weekly-review history list pages behind the SAME engine-keyed signed cursor every
    # other collection uses (PAGE-R5); bind the breadth router's cursor-key seam to it.
    overrides[agent_breadth_router.cursor_signing_key] = _cursor_signing_key_seam
    # The owner account-deletion endpoint invokes the REAL whole-athlete erasure EXECUTOR via the
    # recorder bound here (PRIV-1 / PRIV-R8): DELETE /v1/users/me erases every athlete-scoped row
    # across BOTH stores (canonical + agent-state) + the retained original-file object bytes, mints
    # an auditable completion record (the ErasureReceipt), and returns the async pending_deletion
    # ack. It never silently no-ops (fail-closed) and never 500s on an unwired seam (H4).
    overrides[users_router.deletion_requester] = lambda: build_deletion_requester(app)
    _wire_planning(overrides, engine)
    seams = build_ingestion_seams(app.state.database, app.state.settings)
    overrides[imports_router.import_processor] = lambda: seams.import_processor
    overrides[sync_router.sync_orchestrator] = lambda: seams.sync_orchestrator
    if seams.credential_sink is not None:
        sink = seams.credential_sink
        overrides[connections_router.credential_sink] = lambda: sink


def _wire_planning(overrides: dict[Callable[..., Any], Callable[..., Any]], engine: object) -> None:
    """Bind the planning router's seams to the real gates + the shared agent engine (API-R32).

    The planning surface mirrors the agent/performance routers: GENERATION (``POST
    /v1/planning/workouts``) is gated on the ``agent`` scope and the READ views
    (``GET /v1/planning/workouts``, ``/schedule``) on ``read`` (AUTH-R11/R13); the acting
    athlete id is server-derived from the verified principal (AUTH-R3 — never a client value);
    the plan-generation seam is the SAME ``GraphAgentEngine`` the ``/v1/agent`` surface drives
    (ARCH-R21), so an OSS deployment with no LLM phase-gates to a degraded answer rather than
    fabricating a plan (RUN-R4.1). The session is the shared transactional one, the keyset cursor
    is signed with the engine key (PAGE-R5), and the per-athlete ``agent`` bucket is the process
    limiter (LIMIT-R2).
    """
    overrides[planning_router.require_agent_scope] = require_scopes(Scope.AGENT)
    overrides[planning_router.require_read_scope] = require_scopes(Scope.READ)
    overrides[planning_router.current_athlete_id] = _athlete_id_seam
    overrides[planning_router.planning_engine] = lambda: engine
    overrides[planning_router.current_session] = get_db
    overrides[planning_router.rate_limiter] = get_rate_limiter
    overrides[planning_router.cursor_signing_key] = _cursor_signing_key_seam


def _build_engine(app: FastAPI) -> object:
    """Build the agent engine the API drives — the live coach, else the no-LLM fallback (RUN-R4.1).

    With an LLM key configured this is the live :class:`GraphAgentEngine`; with none it is the
    :class:`UnconfiguredAgentEngine` wired with the canonical ``Database`` (read-only for the
    DETERMINISTIC ``diagnose``, API-R15) and a DEDICATED agent-state store on its OWN engine/pool
    (ARCH-R13/DEPLOY-R4) so the NON-LLM memory seam (MEM-R3 / PRIV-R8 — a privacy MUST that never
    requires a model) reads/erases durable memory even with no LLM configured (H2). The same single
    engine instance is shared by the ``/v1/agent`` (incl. breadth) and ``/v1/planning`` surfaces.
    """
    live = build_agent_engine(app.state.database, app.state.settings)
    if live is not None:
        return live
    return UnconfiguredAgentEngine(
        app.state.database,
        state_db=build_agent_state_database(app.state.settings),
    )


def _build_rate_limiter(settings: Settings) -> RateLimiter:
    """Build the process RateLimiter with the CONFIG-sourced per-class ceilings (LIMIT-R2/CFG-R1a).

    No rate value is a code literal: the ``agent`` class ceiling is the entitlement-governed
    ``entitlement__request_rate_per_minute`` (so the agent surface's request rate IS the OSS
    plan's non-monetary request-rate guard, AGT-ENT-R1/-R4), and the ``read`` / ``mutating``
    ceilings are the loaded ``ratelimit__*`` values (defaults.toml). The limiter's
    ``DEFAULT_LIMITS`` is used only by an isolated unit that constructs ``RateLimiter()`` with no
    settings; the production app always sources from config here.
    """
    limits = {
        LimitClass.READ: settings.ratelimit__read_per_minute,
        LimitClass.MUTATING: settings.ratelimit__mutating_per_minute,
        LimitClass.AGENT: settings.entitlement__request_rate_per_minute,
    }
    return RateLimiter(limits)


async def _persisted_locale_seam(
    session: Annotated[AsyncSession, Depends(get_db)],
    principal: Annotated[Principal, Depends(authenticate)],
) -> str | None:
    """The owner's persisted language SUBTAG from ``athlete.primary_locale`` (API-R37).

    ``primary_locale`` is the single canonical home of the language preference; reading
    returns its language subtag (``de-DE`` -> ``de``) so the agent/planning resolvers can
    apply the stored default when no per-request override is given. ``None`` (unset)
    falls through to the engine ``en`` baseline — NULL means the athlete has made no
    explicit choice yet.
    """
    try:
        owner_id = uuid.UUID(principal.athlete_id)
    except (ValueError, AttributeError):
        return None  # a non-UUID subject has no canonical profile row to read
    owner = await session.get(Athlete, owner_id)
    if owner is None or not owner.primary_locale:
        return None
    return owner.primary_locale.split("-", 1)[0].lower()


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


def _mount_metrics(app: FastAPI) -> None:
    """Mount ``GET /metrics`` — the production metrics scrape surface (OBS-R5/-R4, AGT-OBS-R7).

    Deliberately OUTSIDE ``/v1`` (an operational route, like ``/healthz``): the platform's
    metrics collector scrapes it. It exposes the process-local registry in Prometheus text
    format — operational metrics (per-endpoint request/latency) AND the agent quality/health
    signals (grounding-scrub, structured-validation-failure, reflection-exhaustion,
    ``degraded``/``budget_exceeded`` rates, p50/p95 latency, cost per run) — so a sustained
    regression in any signal is alertable in production (AGT-OBS-R7), not only in CI (CI-R8).
    It carries no per-athlete payload and leaks no internal detail (OBS-R6.3).
    """

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        """Render the process metrics registry in Prometheus text format (OBS-R5)."""
        return PlainTextResponse(get_registry().render())


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


__all__ = [
    "API_PREFIX",
    "create_app",
    "register_routers",
    "resolve_entitlement",
]
