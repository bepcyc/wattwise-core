"""Service-principal second factor: the ``X-Service-Auth`` header (SEC-R4 / AUTH-R8a).

A first-party service runtime (e.g. the Telegram bot) authenticates to the core API as a
SERVICE PRINCIPAL with a high-entropy shared secret presented in the DEDICATED
``X-Service-Auth`` header and compared in CONSTANT TIME (SEC-R4 option (b); the mTLS
pinned-cert option is a deployment-edge concern). The factor is carried OUTSIDE
``Authorization`` — that header stays reserved for the athlete's bearer token (doc 60
AUTH-R2/AUTH-R8) — and is presented IN ADDITION to the athlete-scoped token, never as a
replacement: this middleware never resolves, widens, or substitutes an identity; the
bearer gate still runs on every protected route.

Fail-closed semantics (config-driven, CFG-R2 — the secret is env/secret-store only):

* secret CONFIGURED + header presented → constant-time compare; mismatch → ``401``.
* secret NOT configured + header presented → ``401`` (a service factor was asserted but
  no service principal is provisioned — never silently ignored).
* header absent → pass through; routes that REQUIRE the service factor depend on
  :func:`require_service_principal`, which rejects the request when the verified marker
  is absent.
"""

from __future__ import annotations

import hmac
from typing import Any, Final

from fastapi import Request

from wattwise_core.api.errors import PROBLEM_MEDIA_TYPE, ProblemError, render_problem_bytes
from wattwise_core.config import Settings

#: The dedicated service-auth header (SEC-R4): NEVER ``Authorization``.
SERVICE_AUTH_HEADER: Final = "x-service-auth"


class ServiceAuthMiddleware:
    """Verify a presented ``X-Service-Auth`` factor in constant time (SEC-R4).

    Pure-ASGI: inspects only the request headers; on success it marks the scope
    (``scope["state"]["service_principal"] = True``) for :func:`require_service_principal`
    and strips nothing — the athlete bearer gate still authenticates the acting identity.
    """

    def __init__(self, app: Any, *, settings: Settings) -> None:
        self._app = app
        secret = settings.service_auth_secret
        self._secret: bytes | None = (
            secret.get_secret_value().encode() if secret is not None else None
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Pass through non-HTTP scopes; verify the service factor when presented."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        presented = _header(scope, SERVICE_AUTH_HEADER)
        if presented is None:
            await self._app(scope, receive, send)
            return
        # Constant-time compare against the configured secret; an unprovisioned factor
        # (no secret configured) or a mismatch is rejected 401 fail-closed (SEC-R4).
        if self._secret is None or not hmac.compare_digest(presented, self._secret):
            await _reject(scope, send)
            return
        state = scope.setdefault("state", {})
        state["service_principal"] = True
        await self._app(scope, receive, send)


def _header(scope: Any, name: str) -> bytes | None:
    """The first value of header ``name`` (lowercase) from the ASGI scope, if any."""
    wanted = name.encode()
    for key, value in scope.get("headers", []):
        if key.lower() == wanted:
            return bytes(value)
    return None


async def _reject(scope: Any, send: Any) -> None:
    """Emit the catalog ``401 unauthenticated`` problem document (ERR-R1/AUTH-R9)."""
    body = render_problem_bytes("unauthenticated", Request(scope))
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", PROBLEM_MEDIA_TYPE.encode()),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def require_service_principal(request: Request) -> None:
    """Dependency for routes reserved to a first-party service runtime (SEC-R4).

    The verified marker is set ONLY by :class:`ServiceAuthMiddleware` after the
    constant-time check; a request without it is refused ``401`` fail-closed. This
    factor gates the SERVICE surface in ADDITION to the athlete bearer gate — it never
    replaces the athlete token (the route still composes the bearer dependency).
    """
    if not getattr(request.state, "service_principal", False):
        raise ProblemError("unauthenticated", headers={"WWW-Authenticate": "Bearer"})


__all__ = ["SERVICE_AUTH_HEADER", "ServiceAuthMiddleware", "require_service_principal"]
