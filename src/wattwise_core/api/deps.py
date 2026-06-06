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

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.auth import Principal, Scope, authenticate, require_scopes
from wattwise_core.api.errors import ProblemError
from wattwise_core.config import Settings
from wattwise_core.persistence import Database


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


async def get_db(
    database: Annotated[Database, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    """Yield one transactional :class:`AsyncSession` for the request (committed/rolled back).

    The session is scoped to a single request; the engine seam commits on success and
    rolls back on error. Routes receive an :class:`AsyncSession` and never touch the
    engine/factory directly, keeping the persistence lifecycle in one place.
    """
    async with database.session() as session:
        yield session


def require_scope(*scopes: Scope) -> object:
    """Re-export the auth scope gate as the dependency routes compose (AUTH-R7/R11).

    Thin pass-through to :func:`wattwise_core.api.auth.require_scopes` so routers
    depend on the API layer's stable surface (``deps``) rather than auth internals.
    """
    return require_scopes(*scopes)


#: A typed annotation routes can reuse for the authenticated principal (AUTH-R3).
CurrentPrincipal = Annotated[Principal, Depends(authenticate)]

#: A typed annotation routes can reuse for the request-scoped DB session.
DbSession = Annotated[AsyncSession, Depends(get_db)]

#: A typed annotation routes can reuse for the resolved settings.
AppSettings = Annotated[Settings, Depends(get_settings)]


__all__ = [
    "AppSettings",
    "CurrentPrincipal",
    "DbSession",
    "get_database",
    "get_db",
    "get_settings",
    "require_scope",
]
