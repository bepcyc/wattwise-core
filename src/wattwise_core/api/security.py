"""Security + entitlement + readiness wiring for ``/v1`` (SEC-R10*, AGT-ENT-R*, OBS-R6.2).

Factored out of the app factory (QUAL-R9 module-size split): the app factory composes these
pieces, this module owns their bodies. Everything here is CONFIG-DRIVEN (CFG-R1a) and
fail-closed:

* :class:`SecurityHeadersMiddleware` + :func:`install_security_middleware` — the config-driven
  CORS allowlist, TrustedHost allowed-host validation, and the transport security headers
  (HSTS / nosniff / Referrer-Policy / CSP) the first-party web client needs (SEC-R10/.1/.2).
* :func:`resolve_entitlement` + :func:`agent_feature_gate` — the HTTP half of the entitlement
  resolve -> attach -> check seam: resolve the plan from the SERVER-DERIVED subject (AUTH-R18),
  attach it to the request, and gate the agent surface fail-closed on ``can_use_agent``
  (AGT-ENT-R1/-R3).
* :func:`mount_readiness` — the distinct readiness probe (``/readyz`` + ``/v1/health/ready``)
  that returns 503 until the DB is reachable AND the entitlement resolver + validated default
  plan are loaded (OBS-R6.2 / ENT-R6), leaking no internals (OBS-R6.3).
* :func:`build_deletion_requester` — the recorder that invokes the REAL whole-athlete erasure
  executor for DELETE /v1/users/me (PRIV-1 / PRIV-R8).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import literal, select
from starlette.middleware.trustedhost import TrustedHostMiddleware

from wattwise_core.agent.state_db import build_agent_state_database
from wattwise_core.api.auth import Principal, Scope, authenticate, require_scopes
from wattwise_core.api.errors import ProblemError
from wattwise_core.config import Settings
from wattwise_core.entitlement import (
    Entitlements,
    OssEntitlementResolver,
    plan_bounds_summary,
    validate_plan,
)
from wattwise_core.observability.logging import get_logger
from wattwise_core.persistence import Database
from wattwise_core.privacy.erasure import erase_athlete
from wattwise_core.seams import SYSTEM_SUBJECT, EngineSessionProvider
from wattwise_core.storage import create_object_store

#: The in-version readiness path; ``/readyz`` is the conventional operational alias (OBS-R6.2).
READINESS_PATH = "/v1/health/ready"

#: The durable audit logger the account-deletion erasure completion record is written to (PRIV-R8).
_DELETION_LOGGER = get_logger("wattwise_core.api.users.deletion")

#: The deletion-erasure recorder signature the users router's seam expects (PRIV-R8).
DeletionRequester = Callable[[str, _dt.datetime], Awaitable[None]]


# --------------------------------------------------------------- transport security (SEC-R10*)


class SecurityHeadersMiddleware:
    """Attach the transport security headers to every response (SEC-R10.1).

    A pure-ASGI middleware that injects ``Strict-Transport-Security`` (HSTS),
    ``X-Content-Type-Options: nosniff``, ``Referrer-Policy``, and a restrictive
    ``Content-Security-Policy`` onto the outgoing response-start headers — on by default for
    the first-party client (SEC-R10.2), values CONFIG-DRIVEN (CFG-R1a), never code literals.
    It only rewrites the ``http.response.start`` message's header list, so it adds no buffering
    and is SSE-safe (it does not touch the streamed body). HSTS is the standard
    ``max-age=<n>; includeSubDomains`` (the operator terminates TLS, SEC-R10.1).
    """

    def __init__(
        self,
        app: Any,
        *,
        hsts_max_age: int,
        referrer_policy: str,
        content_security_policy: str,
    ) -> None:
        self._app = app
        self._headers: tuple[tuple[bytes, bytes], ...] = (
            (b"strict-transport-security", f"max-age={hsts_max_age}; includeSubDomains".encode()),
            (b"x-content-type-options", b"nosniff"),
            (b"referrer-policy", referrer_policy.encode()),
            (b"content-security-policy", content_security_policy.encode()),
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Pass through non-HTTP scopes; otherwise add the security headers to the start frame."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def _send(message: Any) -> None:
            if message["type"] == "http.response.start":
                existing = list(message.get("headers", []))
                present = {name.lower() for name, _ in existing}
                existing.extend(h for h in self._headers if h[0] not in present)
                message["headers"] = existing
            await send(message)

        await self._app(scope, receive, _send)


def install_security_middleware(app: FastAPI, settings: Settings) -> None:
    """Install the config-driven CORS / allowed-host / security-header middleware (SEC-R10/.1/.2).

    All values are CONFIG-DRIVEN (CFG-R1a / SEC-R10.2 — no per-deployment values baked into
    code): the CORS origin allowlist + credentials + methods + headers (SEC-R10), the
    TrustedHostMiddleware allowed-host list that rejects a spoofed ``Host`` header (SEC-R10.2),
    and the transport security headers (HSTS, ``X-Content-Type-Options: nosniff``,
    ``Referrer-Policy``, a restrictive ``Content-Security-Policy``, SEC-R10.1) — all on by
    default for a first-party web client. The always-insecure wildcard-origin-with-credentials
    combination is already rejected fail-closed at config load (SEC-R10-AC, settings
    ``_fail_closed``), so it can never reach this wiring. Middleware added LAST runs FIRST, so
    TrustedHost (added last) validates the host before CORS / the app.
    """
    app.add_middleware(
        SecurityHeadersMiddleware,
        hsts_max_age=settings.security__hsts_max_age_seconds,
        referrer_policy=settings.security__referrer_policy,
        content_security_policy=settings.security__content_security_policy,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.security__cors_allow_origins),
        allow_credentials=settings.security__cors_allow_credentials,
        allow_methods=list(settings.security__cors_allow_methods),
        allow_headers=list(settings.security__cors_allow_headers),
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.security__allowed_hosts))


# --------------------------------------------------------- entitlement resolve -> attach -> check


def resolve_entitlement(
    request: Request, principal: Annotated[Principal, Depends(authenticate)]
) -> Entitlements:
    """Resolve + ATTACH the entitlement for the server-derived subject (AGT-ENT-R1).

    The HTTP half of the resolve -> attach -> check seam: it resolves the entitlement from the
    SERVER-DERIVED subject (the verified principal's ``athlete_id`` — AUTH-R18, never a client
    value) through the app's bound :class:`OssEntitlementResolver`, then attaches it to the
    request (``request.state.entitlement``) so any downstream gate/handler reads the SAME
    resolved plan. An app assembled without a resolver fails closed as a generic internal error
    (RUN-R4.1) rather than serving ungated.
    """
    resolver = getattr(request.app.state, "entitlement_resolver", None)
    if not isinstance(resolver, OssEntitlementResolver):
        raise ProblemError("internal-error")
    entitlement = resolver.resolve(principal.athlete_id)
    request.state.entitlement = entitlement
    return entitlement


def attached_entitlement(request: Request) -> Entitlements | None:
    """The per-request resolved entitlement the agent gate attached, else ``None`` (MED-2).

    The read half of resolve -> attach -> check the agent routers call: ``agent_feature_gate``
    runs :func:`resolve_entitlement`, which attaches the SERVER-DERIVED resolved plan to
    ``request.state.entitlement`` (AGT-ENT-R1); the engine then reads its non-monetary bounds FROM
    this plan for the run, so the seam is REAL end to end. Read defensively (``getattr``): a
    test/router that stubs the gate without resolving attaches none, so this yields ``None`` and the
    engine falls back to its config-resolved default — identical in OSS, backward-compatible.
    """
    return getattr(request.state, "entitlement", None)


def agent_feature_gate(
    _scopes: Annotated[None, Depends(require_scopes(Scope.AGENT))],
    entitlement: Annotated[Entitlements, Depends(resolve_entitlement)],
) -> None:
    """Gate the agent surface on the ``agent`` scope AND the resolved entitlement (AGT-ENT-R3).

    The real CHECK half of resolve -> attach -> check for the HTTP agent surface: it runs the
    bearer + ``agent``-scope gate (AUTH-R13) AND reads the carried entitlement, FAILING CLOSED
    (``403``) when ``can_use_agent`` is not granted. Under the OSS all-permissive default plan
    the flag is granted, so the agent surface is permitted; a commercial plan that ungrants the
    feature IS enforced here WITHOUT touching the agent router. The refusal is a typed,
    source-agnostic ``insufficient-scope`` problem (AGT-ENT-R3) — never a plan/budget exposition.
    """
    if not entitlement.can_use_agent:
        raise ProblemError("insufficient-scope")


# --------------------------------------------------------------- readiness probe (OBS-R6.2)


def mount_readiness(app: FastAPI) -> None:
    """Mount the readiness probe at ``/v1/health/ready`` and ``/readyz`` (OBS-R6.2 / ENT-4).

    DISTINCT from liveness: readiness reflects whether the instance can serve traffic NOW —
    the database is reachable, AND the Entitlement resolver is initialized with the default
    plan loaded + validated (ENT-R6). It returns ``503`` (``not_ready``) until every check
    passes (a not-ready instance is drained, never killed; there is no fake-healthy route that
    reports ready first, OBS-R6.2). It leaks NO internal detail (OBS-R6.3): the body is a small
    status object (the non-secret plan-bounds summary names only the flags + numeric bounds — no
    DSN, no version, no stack trace). Mounted at both ``/readyz`` (the conventional operational
    name) and the in-version ``/v1/health/ready`` (the documented readiness surface).
    """

    async def _payload(request: Request, response: Response) -> dict[str, Any]:
        """Run the readiness checks; set ``503`` until the resolver+plan AND DB are ready."""
        resolver = getattr(request.app.state, "entitlement_resolver", None)
        plan = getattr(request.app.state, "entitlement_plan", None)
        checks: dict[str, bool] = {
            "entitlement_resolver": isinstance(resolver, OssEntitlementResolver),
            "default_plan_loaded": _plan_is_valid(plan),
            "database": await _database_reachable(request.app),
        }
        ready = all(checks.values())
        if not ready:
            response.status_code = 503
        body: dict[str, Any] = {"status": "ready" if ready else "not_ready", "checks": checks}
        if isinstance(plan, Entitlements):
            body["plan"] = plan_bounds_summary(plan)
        return body

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request, response: Response) -> dict[str, Any]:
        """Readiness: DB + Entitlement resolver/plan ready, else 503 (OBS-R6.2 / ENT-R6)."""
        return await _payload(request, response)

    @app.get(READINESS_PATH, include_in_schema=False)
    async def health_ready(request: Request, response: Response) -> dict[str, Any]:
        """The in-version readiness surface; identical checks to ``/readyz`` (OBS-R6.2)."""
        return await _payload(request, response)


def _plan_is_valid(plan: object) -> bool:
    """True iff ``plan`` is an :class:`Entitlements` that passes fail-closed validation (ENT-R6)."""
    if not isinstance(plan, Entitlements):
        return False
    try:
        validate_plan(plan)
    except Exception:  # any validation failure means not-ready, never a 500 (OBS-R6.2)
        return False
    return True


async def _database_reachable(app: FastAPI) -> bool:
    """True iff a trivial probe query succeeds on the canonical DB (OBS-R6.2 dependency check).

    Readiness depends on the database (unlike liveness, OBS-R6.1): an unreachable DB reports
    not-ready (drain from rotation) rather than a fake-healthy ready. The probe is the portable
    query-builder ``select(literal(1))`` (no dialect-specific raw statement, BOOT-R3). Any error
    is swallowed into ``False`` so the probe never 500s — it reports not-ready and the
    orchestrator drains.
    """
    database = getattr(app.state, "database", None)
    if not isinstance(database, Database):
        return False
    try:
        # The reachability probe is a request-less SYSTEM open, but it STILL flows through the ONE
        # engine-owned session provider seam (SEAM-R11 / ARCH-R31) — never around it. Not bound to a
        # request athlete, it carries the non-scoped ``SYSTEM_SUBJECT`` marker (inert in the OSS
        # provider, which does no scoping; the operator/system context in the commercial one).
        async with EngineSessionProvider(database).session(subject=SYSTEM_SUBJECT) as session:
            await session.execute(select(literal(1)))
        return True
    except Exception:  # an unreachable DB is not-ready, never a 500 (OBS-R6.2)
        return False


# ------------------------------------------------------------- real erasure recorder (PRIV-1)


def build_deletion_requester(app: FastAPI) -> DeletionRequester:
    """Build the recorder that invokes the REAL whole-athlete erasure executor (PRIV-1 / PRIV-R8).

    Wires DELETE /v1/users/me to the executable right-to-be-forgotten fulfilment
    (:func:`wattwise_core.privacy.erasure.erase_athlete`) instead of a log-only no-op: it opens a
    canonical session (the shared ``Database``) AND a session on a DEDICATED agent-state store (its
    own engine/pool, ARCH-R13), passes the local object store (so the retained original-file BYTES
    are erased, PRIV-R11.3), and runs the executor — which deletes EVERY athlete-scoped row across
    BOTH stores inside each store's own transaction (fail-closed: rolls back on any error) and
    returns an auditable :class:`ErasureReceipt`. The receipt is then logged as the durable
    completion record ("auditable record that erasure completed", PRIV-R8). Identity is the
    SERVER-DERIVED owner id (AUTH-R18) the endpoint passed; the executor never reads identity from a
    payload. The agent-state DB is disposed after the run so the app holds no extra pool.
    """
    settings: Settings = app.state.settings
    database: Database = app.state.database

    async def _erase(athlete_id: str, requested_at: _dt.datetime) -> None:
        """Erase the athlete across both stores + object bytes, then log the receipt (PRIV-R8)."""
        object_store = create_object_store(settings)
        state_db = build_agent_state_database(settings)
        try:
            await state_db.create_all()
            # The CANONICAL side of the erasure flows through the ONE engine-owned session provider
            # seam (SEAM-R11 / ARCH-R31), keyed on the server-derived ``athlete_id`` (AUTH-R18),
            # never around it. The agent-state store is SEPARATE (ARCH-R13), with its own session.
            async with (
                EngineSessionProvider(database).session(subject=athlete_id) as canonical_session,
                state_db.session() as agent_session,
            ):
                receipt = await erase_athlete(
                    athlete_id,
                    canonical_session=canonical_session,
                    agent_state_session=agent_session,
                    object_store=object_store,
                )
        finally:
            await state_db.dispose()
        # The auditable completion record (PRIV-R8): the subject, the instant, and the
        # residual-zero row/object counts removed — a non-secret summary the operator's durable
        # log sink retains.
        _DELETION_LOGGER.info(
            "account_deletion_completed",
            athlete_id=athlete_id,
            requested_at=requested_at.isoformat(),
            completed_at=receipt.completed_at.isoformat(),
            rows_deleted=receipt.total_rows_deleted,
            objects_deleted=receipt.total_objects_deleted,
            status="erased",
        )

    return _erase


__all__ = [
    "READINESS_PATH",
    "DeletionRequester",
    "SecurityHeadersMiddleware",
    "agent_feature_gate",
    "attached_entitlement",
    "build_deletion_requester",
    "install_security_middleware",
    "mount_readiness",
    "resolve_entitlement",
]
