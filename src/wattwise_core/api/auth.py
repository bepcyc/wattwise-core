"""Bearer-only, server-derived authentication and scope authorization.

This module owns the central security invariant of the ``/v1`` surface: the acting
athlete is derived **exclusively** from a verified bearer token, never from anything
the client can write (body/query/path/header). It verifies the token, exposes the
resolved :class:`Principal`, and provides the scope gate used by route dependencies.

Requirements realized here (doc 60):

- **AUTH-R1** Protected endpoints require auth; an unauthenticated request yields
  ``401`` with a ``WWW-Authenticate: Bearer`` header and an RFC 9457 problem body.
- **AUTH-R2** The acting-user credential is ``Authorization: Bearer <token>`` ONLY —
  never query/body/cookie/other location. (The distinct service-principal factor of
  AUTH-R8a lives outside ``Authorization`` and is not this module's concern.)
- **AUTH-R3 / AUTH-R18** Identity (the ``subject``) is derived ONLY from the verified
  token; this module exposes no way to read a caller-supplied identity, and request
  schemas carry no writable caller-identity field. In OSS the subject is the athlete.
- **AUTH-R6** Validate signature, issuer, audience (``wattwise-core``), and expiry on
  every request; an expired or malformed token yields ``401``; a positive decision is
  never cached past expiry (verification runs per request).
- **AUTH-R7** Tokens carry scopes from the closed set
  ``read | write | agent | sync | export | admin``; a missing required scope yields
  ``403`` ``insufficient-scope`` listing the ``required_scopes``.
- **AUTH-R9** Auth failures expose no object contents/internal ids/stack traces/token
  contents (only the generic catalog copy is returned; ERR-R5).

Token issuance (``POST /v1/auth/token``) mints a first-party access token signed with
the engine's ``token_signing_key`` (HS256). Refresh/revoke and the bot-link flow are
mounted by their own routers (out of this module's scope).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Final

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from wattwise_core.api.errors import FieldError, ProblemError
from wattwise_core.config import Settings

#: The audience every first-party token is issued for and verified against (AUTH-R6).
TOKEN_AUDIENCE: Final = "wattwise-core"  # noqa: S105 (a public protocol value, not a secret)

#: The issuer of first-party tokens (the engine signs its own access tokens, AUTH-R6).
TOKEN_ISSUER: Final = "wattwise-core"  # noqa: S105 (a public protocol value, not a secret)

#: Symmetric signing algorithm for the first-party access token (token_signing_key).
TOKEN_ALGORITHM: Final = "HS256"  # noqa: S105 (an algorithm name, not a secret)

#: Default access-token lifetime; the issued ``expires_in`` echoes this (API-R23).
DEFAULT_ACCESS_TTL_SECONDS: Final = 3600

#: The WWW-Authenticate challenge returned with every ``401`` (AUTH-R1).
_BEARER_CHALLENGE: Final = {"WWW-Authenticate": "Bearer"}


class Scope(StrEnum):
    """The closed scope vocabulary tokens may carry (AUTH-R7).

    Scopes gate capability, not tenancy (AUTH-R18). The string values are the
    machine tokens that appear in the JWT ``scope`` claim and in ``required_scopes``.
    """

    READ = "read"
    WRITE = "write"
    AGENT = "agent"
    SYNC = "sync"
    EXPORT = "export"
    ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated acting party, resolved server-side from the token.

    ``subject`` is the server-derived identity (in OSS, the single athlete/owner —
    AUTH-R18). It is NEVER taken from client-controlled input (AUTH-R3). ``scopes``
    is the set of granted capabilities parsed from the verified token (AUTH-R7).
    """

    subject: str
    scopes: frozenset[Scope]

    @property
    def athlete_id(self) -> str:
        """The one athlete this principal acts as (OSS: subject == athlete, AUTH-R18)."""
        return self.subject

    def has_scope(self, scope: Scope) -> bool:
        """True if the token granted ``scope`` (AUTH-R7)."""
        return scope in self.scopes


@dataclass(frozen=True, slots=True)
class AuthTokens:
    """The token-issuance response shape (API-R23/R24).

    Carries only the issued credentials + their scopes; no object contents, internal
    ids, or secret material beyond the tokens themselves (AUTH-R9).
    """

    access_token: str
    refresh_token: str
    expires_in: int
    scopes: tuple[str, ...]
    token_type: str = "bearer"  # noqa: S105 (the OAuth token_type label, not a secret)

    def to_dict(self) -> dict[str, Any]:
        """Render to the JSON body for ``POST /v1/auth/token``."""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "refresh_token": self.refresh_token,
            "scopes": list(self.scopes),
        }


def _signing_key(settings: Settings) -> str:
    """Return the symmetric signing key, failing closed if it is absent (AUTH-R6).

    The key is a load-bearing secret sourced only from the environment / secret
    manager (BOOT-R4). Its absence is an operator misconfiguration, not a client
    error, so it surfaces as a generic internal error (ERR-R5) — never as a hint.
    """
    key = settings.token_signing_key
    if key is None:
        raise ProblemError("internal-error")
    return key.get_secret_value()


def _parse_scopes(raw: object) -> frozenset[Scope]:
    """Parse the token's ``scope`` claim into the closed :class:`Scope` set (AUTH-R7).

    Accepts either a space-delimited string (OAuth convention) or a list of strings.
    Unknown scope tokens are ignored (forward-compat): only members of the closed set
    grant capability; an unrecognized token can never widen access.
    """
    if isinstance(raw, str):
        tokens: Iterable[str] = raw.split()
    elif isinstance(raw, (list, tuple)):
        tokens = [str(item) for item in raw]
    else:
        tokens = ()
    valid = {member.value for member in Scope}
    return frozenset(Scope(token) for token in tokens if token in valid)


def _decode(token: str, settings: Settings) -> dict[str, Any]:
    """Verify signature/issuer/audience/expiry and return the claims (AUTH-R6).

    Any verification failure — bad signature, wrong issuer/audience, expired, or
    malformed — maps to the same ``401 unauthenticated`` with a ``WWW-Authenticate``
    challenge and no token contents (AUTH-R6/R9). PyJWT validates ``exp``/``aud``/
    ``iss`` when those options are required, so a positive decision can never be
    served past expiry (no caching here — verification runs every request).
    """
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            _signing_key(settings),
            algorithms=[TOKEN_ALGORITHM],
            audience=TOKEN_AUDIENCE,
            issuer=TOKEN_ISSUER,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except jwt.PyJWTError as exc:
        raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE) from exc
    return claims


def _principal_from_claims(claims: dict[str, Any]) -> Principal:
    """Build the :class:`Principal` from verified claims (AUTH-R3/R7).

    The subject comes ONLY from the token's ``sub`` claim (server-derived identity);
    a missing/blank subject is treated as an unauthenticated request, never a
    fallback to any client-supplied value (AUTH-R3).
    """
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE)
    return Principal(subject=subject, scopes=_parse_scopes(claims.get("scope")))


#: Bearer extractor. ``auto_error=False`` so a missing credential routes through our
#: uniform ``401`` problem document (AUTH-R1) instead of FastAPI's default body.
_bearer_scheme: Final = HTTPBearer(auto_error=False, scheme_name="bearer")


def authenticate(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> Principal:
    """Resolve the acting :class:`Principal` from the bearer token (AUTH-R1/R2/R3/R6).

    A missing or non-bearer ``Authorization`` header yields ``401`` with a
    ``WWW-Authenticate: Bearer`` challenge. The verified subject is stashed on the
    request state for correlated logging; the credential itself is never logged or
    echoed (AUTH-R9). This is the single seam every protected route depends on, so
    no route can read identity from anywhere but the verified token.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise ProblemError("unauthenticated", headers=_BEARER_CHALLENGE)
    settings = _settings_of(request)
    principal = _principal_from_claims(_decode(credentials.credentials, settings))
    request.state.athlete_id = principal.athlete_id
    return principal


def _settings_of(request: Request) -> Settings:
    """Fetch the resolved :class:`Settings` placed on app state at startup."""
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise ProblemError("internal-error")
    return settings


def require_scopes(*required: Scope) -> Any:
    """Build a dependency enforcing that the principal holds every ``required`` scope.

    Read routes require :attr:`Scope.READ`; mutating routes require
    :attr:`Scope.WRITE` plus any endpoint-specific scope (AUTH-R11). A principal
    missing any required scope gets ``403 insufficient-scope`` listing the
    ``required_scopes`` (AUTH-R7) — authentication already passed, so this is an
    authorization gap, not a ``401``.
    """

    def _dependency(
        principal: Annotated[Principal, Depends(authenticate)],
    ) -> Principal:
        missing = [scope for scope in required if not principal.has_scope(scope)]
        if missing:
            raise _insufficient_scope(required)
        return principal

    return _dependency


def _insufficient_scope(required: tuple[Scope, ...]) -> ProblemError:
    """Construct the ``403 insufficient-scope`` problem listing required scopes (AUTH-R7).

    ``required_scopes`` is carried as a machine-readable ``errors[]`` member so a
    client learns which capability its token lacks without any object/identity leak
    (AUTH-R9). The codes are the stable scope tokens, not athlete prose.
    """
    field_errors = [
        FieldError(code="missing_scope", message=scope.value, parameter="required_scopes")
        for scope in required
    ]
    return ProblemError("insufficient-scope", errors=field_errors)


def issue_access_token(
    settings: Settings,
    *,
    subject: str,
    scopes: Iterable[Scope],
    ttl_seconds: int = DEFAULT_ACCESS_TTL_SECONDS,
) -> AuthTokens:
    """Mint a signed first-party access token for ``subject`` (API-R23, AUTH-R6).

    The token carries the verified-on-every-request claims (``iss``/``aud``/``exp``/
    ``sub``/``scope``) so :func:`authenticate` can re-derive the principal entirely
    server-side. The refresh token is a separate opaque credential minted by the auth
    router (rotation/reuse-detection live there); this helper issues the access leg.
    """
    granted = tuple(scope.value for scope in scopes)
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        "sub": subject,
        "scope": " ".join(granted),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
    }
    access = jwt.encode(payload, _signing_key(settings), algorithm=TOKEN_ALGORITHM)
    return AuthTokens(
        access_token=access,
        refresh_token="",
        expires_in=ttl_seconds,
        scopes=granted,
    )


__all__ = [
    "DEFAULT_ACCESS_TTL_SECONDS",
    "TOKEN_AUDIENCE",
    "TOKEN_ISSUER",
    "AuthTokens",
    "Principal",
    "Scope",
    "authenticate",
    "issue_access_token",
    "require_scopes",
]
