"""The ROAD-R2-EXIT mount contract: every registered feature router is on the live surface.

This is the app-factory wiring guard for the exit slice (doc 60 §planning / §8): it asserts that
:func:`wattwise_core.api.app.create_app` actually mounts the newly-registered feature routers under
the single ``/v1`` prefix (API-R4), so a registration regression in
:mod:`wattwise_core.api.routers` (a router dropped from ``ROUTERS``) or a broken include in the
factory is caught loudly rather than surfacing as a silent ``404``.

What it pins:

- the two routers this slice newly registers are mounted unconditionally: the agent-backed plan
  surface ``POST /v1/planning/workouts`` (API-R32) and the owner self-service account
  ``GET /v1/users/me`` (doc 60 §8);
- the convergent ROAD-R2-EXIT surface — ``/v1/agent/diagnose`` (API-R15) and ``/v1/agent/memory``
  (MEM-R3) — is mounted the moment its owning (already-registered) router DECLARES it: the
  assertion is gated on the route being declared by a router so this guard is green while a sibling
  slice is still landing, then enforces the full exit surface as soon as each route exists. A route
  that a registered router declares but the factory fails to mount is a hard failure (the
  registration/include regression this test exists to catch).

No external dependency is touched: the app is assembled with an in-memory SQLite DSN and a generated
envelope key (BOOT-R4), and only the route table is inspected — no request is issued.

Requirement IDs: API-R4, API-R15, API-R32, MEM-R3, AUTH-R10.
"""

from __future__ import annotations

import pytest

from wattwise_core.api.app import create_app
from wattwise_core.api.routers import ROUTERS
from wattwise_core.config import Settings, load_settings
from wattwise_core.security.crypto import EnvelopeCipher

pytestmark = pytest.mark.integration

#: Routes this slice registers + wires; they MUST be mounted (unconditional contract).
_REQUIRED_ROUTES = (
    "/v1/planning/workouts",
    "/v1/planning/schedule",
    "/v1/users/me",
)

#: The convergent ROAD-R2-EXIT routes owned by sibling (already-registered) routers; each is
#: enforced as soon as ITS owning router declares it, so this guard is green while a sibling
#: slice is still landing and turns hard once the route exists.
_EXIT_ROUTES = (
    "/v1/agent/diagnose",
    "/v1/agent/memory",
)


def _settings() -> Settings:
    """A fail-closed-valid in-memory config (no external dependency; BOOT-R4)."""
    return load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="k" * 32,
        encryption_root_key=EnvelopeCipher.generate_root_key(),
    )


def _mounted_paths() -> set[str]:
    """The set of paths the assembled app actually exposes on its route table (API-R4)."""
    app = create_app(_settings())
    return {path for route in app.routes if (path := getattr(route, "path", ""))}


def _declared_paths() -> set[str]:
    """Every path DECLARED by a registered feature router (the mount contract's source set)."""
    return {
        path
        for router in ROUTERS
        for route in router.routes
        if (path := getattr(route, "path", ""))
    }


@pytest.mark.parametrize("path", _REQUIRED_ROUTES)
def test_required_route_is_mounted(path: str) -> None:
    """Each newly-registered planning/users route is mounted under ``/v1`` (API-R32 / doc 60 §8)."""
    assert path in _mounted_paths()


@pytest.mark.parametrize("path", _EXIT_ROUTES)
def test_declared_exit_route_is_mounted(path: str) -> None:
    """A declared exit route MUST be mounted; skip only while its sibling slice has not landed."""
    if path not in _declared_paths():
        pytest.skip(f"{path} not yet declared by any registered router (sibling slice pending)")
    assert path in _mounted_paths()


def test_every_registered_router_route_is_mounted() -> None:
    """No registered router silently drops off the live surface (the registration guard, API-R4)."""
    missing = _declared_paths() - _mounted_paths()
    assert not missing, f"registered router routes not mounted by the factory: {sorted(missing)}"
