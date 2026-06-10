"""Shared FastAPI dependency providers for the ``/v1`` surface.

Routers depend on these instead of reaching for globals, so the app factory owns the
lifecycle (one :class:`Database`, one resolved :class:`Settings`) and tests can
override a single seam. Identity/scope dependencies re-export the auth gate so a
router writes ``Depends(require_scope(Scope.READ))`` without importing the auth
internals directly.

Requirements realized here (doc 60):

- **AUTH-R3 / AUTH-R18** The acting principal is resolved only by the auth dependency
  (server-derived subject); no provider here reads a caller-supplied identity.
- **AUTH-R7 / AUTH-R11** :func:`require_scope` is the scope gate routes compose
  (``read`` for reads, ``write`` + endpoint-specific for mutations).
- Database sessions come from the engine seam (``Database.session``) as an
  async-context dependency so each request gets one transactional session that is
  committed on success / rolled back on error and always closed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.auth import Principal, Scope, authenticate, require_scopes
from wattwise_core.api.errors import ProblemError
from wattwise_core.api.ratelimit import LimitClass, RateLimiter
from wattwise_core.config import Settings
from wattwise_core.persistence import Database
from wattwise_core.seams import EngineSessionProvider

#: The HTTP methods that debit the mutating bucket; all others debit read (LIMIT-R2).
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def get_settings(request: Request) -> Settings:
    """Return the process-wide resolved settings bound to the app (fail-closed).

    The factory validates and attaches :class:`Settings` at startup; a request that
    arrives without it indicates the app was assembled incorrectly and surfaces as a
    generic internal error rather than booting in an undefined state (RUN-R4.1).
    """
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise ProblemError("internal-error")
    return settings


def get_database(request: Request) -> Database:
    """Return the shared :class:`Database` bound to the app at startup."""
    database = getattr(request.app.state, "database", None)
    if not isinstance(database, Database):
        raise ProblemError("internal-error")
    return database


def get_master_data_database(request: Request) -> Database:
    """Return the :class:`Database` bound to the MASTER-DATA-WRITE role (DEPLOY-R4).

    ARCH-R3(b): athlete-authored master-data (profile/signature, zones/language/
    default-load-model user-settings, goals) is written ONLY through the API's distinct
    master-data-write role. The factory binds this Database at startup — to its own
    engine/pool when the deployment configures the distinct role DSN, else to the shared
    canonical Database (single-operator self-host: one credential; the structural role
    split is an opt-in deploy choice). Its absence means the app was assembled
    incorrectly and surfaces fail-closed as a generic internal error.
    """
    database = getattr(request.app.state, "master_data_database", None)
    if not isinstance(database, Database):
        raise ProblemError("internal-error")
    return database


def request_subject(
    principal: Annotated[Principal, Depends(authenticate)],
) -> str:
    """The server-derived ``subject`` the request's canonical session is keyed on (ARCH-R16).

    A thin seam over :func:`authenticate` (FastAPI caches it per request, so a route that also
    declares :data:`CurrentPrincipal` shares the SAME resolved principal — auth is resolved once,
    never doubled). It exists so :func:`get_db` is keyed on the subject WITHOUT making subject
    resolution a side-effect of the data-access dependency: the session provider stays decoupled
    from the authentication mechanism (SEAM-R11 keys on a server-derived subject; ARCH-R16 owns how
    that subject is derived). The subject is the verified token identity — never client-asserted.
    """
    return principal.subject


async def get_db(
    database: Annotated[Database, Depends(get_database)],
    subject: Annotated[str, Depends(request_subject)],
) -> AsyncIterator[AsyncSession]:
    """Yield one transactional canonical :class:`AsyncSession` for the request (SEAM-R11).

    Canonical-store access flows through the ONE engine-owned ``SessionProvider`` seam
    (SEAM-R11 / ARCH-R31), obtained with the server-derived ``subject`` established at the
    L6 edge (ARCH-R16) and threaded in via :func:`request_subject` — the provider is keyed on the
    subject but is NOT itself coupled to the auth mechanism. The OSS default provider applies NO
    tenant scoping (single-athlete) but IS the single attach point the commercial tenant-scoped
    overlay mounts on. Routes receive an :class:`AsyncSession` and never open the store around it.
    """
    provider = EngineSessionProvider(database)
    async with provider.session(subject=subject) as session:
        yield session


async def get_master_data_db(
    database: Annotated[Database, Depends(get_master_data_database)],
    subject: Annotated[str, Depends(request_subject)],
) -> AsyncIterator[AsyncSession]:
    """Yield one transactional MASTER-DATA session for the request (ARCH-R3b / DEPLOY-R4).

    Identical to :func:`get_db` (the ONE engine-owned ``SessionProvider`` seam, keyed on the
    server-derived subject) except it opens on the master-data-write role's Database — the
    only write surface for athlete-authored master-data. Under a per-role deployment this
    credential cannot write the source-derived canonical tables or the agent-state store
    (reciprocal denial, DEPLOY-R4); on a single-credential self-host it is the shared
    canonical Database (the split is structural-by-config, opt-in at deploy).
    """
    provider = EngineSessionProvider(database)
    async with provider.session(subject=subject) as session:
        yield session


def require_scope(*scopes: Scope) -> object:
    """Re-export the auth scope gate as the dependency routes compose (AUTH-R7/R11).

    Thin pass-through to :func:`wattwise_core.api.auth.require_scopes` so routers
    depend on the API layer's stable surface (``deps``) rather than auth internals.
    """
    return require_scopes(*scopes)


def get_rate_limiter(request: Request) -> RateLimiter:
    """Return the process-wide :class:`RateLimiter` bound to the app (LIMIT-R1).

    The factory installs one limiter on app state at startup; its absence means the app
    was assembled incorrectly and surfaces fail-closed as a generic internal error.
    """
    limiter = getattr(request.app.state, "rate_limiter", None)
    if not isinstance(limiter, RateLimiter):
        raise ProblemError("internal-error")
    return limiter


def enforce_rate_limit(
    request: Request,
    response: Response,
    principal: Annotated[Principal, Depends(authenticate)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> None:
    """Debit the per-athlete read/mutating bucket for this request (LIMIT-R1/R2/R3).

    Classifies the route by HTTP method (``POST``/``PUT``/``PATCH``/``DELETE`` ->
    ``mutating`` ``30/min``; otherwise ``read`` ``120/min``) and debits the bucket keyed
    on the SERVER-DERIVED athlete id (AUTH-R3) — never a client header (LIMIT-R6). On
    success the ``RateLimit-*`` headers are attached; an exhausted bucket raises ``429``
    ``rate-limited`` with ``Retry-After`` + ``RateLimit-*`` (LIMIT-R3).
    """
    limit_class = (
        LimitClass.MUTATING if request.method.upper() in _MUTATING_METHODS else LimitClass.READ
    )
    headers = limiter.check(principal.athlete_id, limit_class)
    response.headers.update(headers.to_dict())


#: A typed annotation routes can reuse for the authenticated principal (AUTH-R3).
CurrentPrincipal = Annotated[Principal, Depends(authenticate)]

#: A typed annotation routes can reuse for the request-scoped DB session.
DbSession = Annotated[AsyncSession, Depends(get_db)]

#: A typed annotation routes can reuse for the resolved settings.
AppSettings = Annotated[Settings, Depends(get_settings)]

#: A router-level dependency that debits the per-athlete read/mutating bucket (LIMIT-R1).
RateLimit = Depends(enforce_rate_limit)


__all__ = [
    "AppSettings",
    "CurrentPrincipal",
    "DbSession",
    "RateLimit",
    "enforce_rate_limit",
    "get_database",
    "get_db",
    "get_master_data_database",
    "get_master_data_db",
    "get_rate_limiter",
    "get_settings",
    "request_subject",
    "require_scope",
]
