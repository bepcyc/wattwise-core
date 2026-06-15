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
    admin,
    agent_routes,
    athlete,
    auth_flows,
    connections,
    connections_management,
    dashboard,
    data_health,
    export,
    exports,
    goals,
    help,
    imports,
    onboarding,
    performance,
    performance_history,
    planning,
    sync,
    user_settings,
    user_settings_constraints,
    users,
)

#: Every feature router, in mount order. Each already carries its full ``/v1/...``
#: prefix, so the factory includes them verbatim (no extra prefix).
ROUTERS: list[APIRouter] = [
    auth_flows.router,
    export.router,
    performance.router,
    performance_history.router,
    activities.router,
    agent_routes.router,
    athlete.router,
    user_settings.router,
    user_settings_constraints.router,
    users.router,
    planning.router,
    goals.router,
    connections.router,
    connections_management.router,
    imports.router,
    exports.router,
    sync.router,
    onboarding.router,
    dashboard.router,
    data_health.router,
    help.router,
    admin.router,
    admin.system_router,
]

__all__ = ["ROUTERS"]
