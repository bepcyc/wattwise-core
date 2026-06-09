"""API router registry (doc 60).

Each feature-router module exposes a module-level ``router: APIRouter`` carrying its
own full ``/v1/...`` path prefix; the app factory includes everything listed in
``ROUTERS``. Routers are aggregated here so :func:`wattwise_core.api.app.create_app`
stays thin and agnostic to which slices exist.
"""

from __future__ import annotations

from fastapi import APIRouter

from wattwise_core.api.routers import (
    activities,
    agent_routes,
    athlete,
    connections,
    imports,
    onboarding,
    performance,
    sync,
    user_settings,
)

#: Every feature router, in mount order. Each already carries its full ``/v1/...``
#: prefix, so the factory includes them verbatim (no extra prefix).
ROUTERS: list[APIRouter] = [
    performance.router,
    activities.router,
    agent_routes.router,
    athlete.router,
    user_settings.router,
    connections.router,
    imports.router,
    sync.router,
    onboarding.router,
]

__all__ = ["ROUTERS"]
