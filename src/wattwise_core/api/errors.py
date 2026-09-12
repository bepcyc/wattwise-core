"""RFC 9457 ``application/problem+json`` error contract for the ``/v1`` surface.

Implements the engine's single, uniform error shape and its closed problem-type
catalog. Every non-2xx response leaving the API is one of these documents — never a
raw exception, an HTML error page, a framework default body, or a stack trace.

Requirements realized here (doc 60):

- **ERR-R1** Every non-2xx is a single ``application/problem+json`` document
  (RFC 9457); never raw exception text / HTML / framework default / stack trace.
- **ERR-R2** The document carries at least ``type | title | status | detail |
  instance | trace_id`` (plus an OPTIONAL ``errors[]`` for field-level validation).
- **ERR-R3** ``type`` is a stable URI from the closed catalog (ERR-R8); clients
  branch on ``type`` (and optionally ``errors[].code``), never on ``title``/``detail``.
- **ERR-R4** Body ``status`` equals the HTTP status line; ``trace_id`` is present on
  every error and matches the server-side trace/correlation id.
- **ERR-R5** No leakage in ``detail``/``errors[].message`` (no stack/SQL/hostnames/
  source-provider names/secrets/tokens); a forced internal failure yields a generic
  ``internal-error`` ``500`` with no leakage.
- **ERR-R6** Schema/type validation failure -> ``422`` ``validation-error`` with a
  populated ``errors[]`` using a JSON Pointer (``pointer``) into the body or a
  ``parameter`` name for query/path violations.
- **ERR-R7** Status-code usage table (the catalog binds a default status per type).
- **ERR-R8** Closed problem-type catalog (the spec's 21 canonical slugs plus the
  additive engine extras built below); new types are additive, existing meaning frozen.
- **AUTH-R9** Auth failures expose no object contents/internal ids/stack traces/token
  contents (the ``unauthenticated``/``insufficient-scope`` documents carry only the
  generic catalog copy, never the rejected credential).

Athlete-facing ``title``/``detail`` copy (API-R21) is warm and jargon-free; it is held
as catalog constants here (not inline at call sites), which also keeps the API layer
clear of orphan user-literals (doc 80 QUAL-R13(c)).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from wattwise_core.api.redaction import redact_text
from wattwise_core.observability.logging import get_logger

_logger = get_logger("wattwise_core.api.errors")

#: RFC 9457 ``type`` URIs dereference under this base; each slug has a doc page.
PROBLEM_BASE_URI: Final = "https://wattwise.app/problems/"

#: The media type every problem document is served with (ERR-R1).
PROBLEM_MEDIA_TYPE: Final = "application/problem+json"

#: Header carrying the correlation id; surfaces back to the client as ``trace_id``.
TRACE_HEADER: Final = "X-Trace-Id"


@dataclass(frozen=True, slots=True)
class ProblemType:
    """One entry of the closed problem-type catalog (ERR-R8).

    ``title`` is the short, stable, athlete-facing headline for the type; the
    instance-specific ``detail`` is supplied per occurrence (or defaults to
    ``title``). Both are jargon-free consumer copy (API-R21). The catalog is the
    single source of truth so call sites never inline athlete sentences.
    """

    slug: str
    status: int
    title: str

    @property
    def uri(self) -> str:
        """The stable, dereferenceable ``type`` URI for this problem type (ERR-R3)."""
        return f"{PROBLEM_BASE_URI}{self.slug}"


def _catalog() -> dict[str, ProblemType]:
    """Build the closed catalog (ERR-R8) keyed by slug.

    Titles are warm, jargon-free athlete copy (API-R21): no source/provider names,
    no transport jargon. ``detail`` is filled per occurrence at the raise site with
    equally jargon-free, non-leaking text (ERR-R5).
    """
    entries = (
        ProblemType("validation-error", 422, "Check those details and try again"),
        ProblemType("bad-request", 400, "We couldn't read that request"),
        ProblemType("invalid-cursor", 400, "That page link expired"),
        ProblemType("cursor-parameter-mismatch", 400, "That page link no longer matches"),
        ProblemType("unauthenticated", 401, "Please sign in to continue"),
        ProblemType("insufficient-scope", 403, "This needs a different sign-in"),
        ProblemType("not-found", 404, "We couldn't find that"),
        ProblemType("conflict", 409, "That changed while you were working"),
        ProblemType("rate-limited", 429, "Just a moment, you're going a bit fast"),
        ProblemType("cost-limit-exceeded", 429, "You've used this month's coaching"),
        ProblemType("agent-grounding-failed", 422, "We couldn't back that up with your data"),
        ProblemType("upstream-unavailable", 503, "We couldn't reach one of your sources"),
        ProblemType("analytics-precondition-unmet", 422, "We don't have enough to work that out"),
        ProblemType("payload-too-large", 413, "That file is a little too big"),
        ProblemType("unsupported-media-type", 415, "We can't read that kind of file"),
        ProblemType("invalid-signed-url", 403, "That download link is no longer valid"),
        ProblemType("webhook-signature-invalid", 401, "We couldn't verify that update"),
        ProblemType("connection-error", 502, "We couldn't finish connecting that"),
        ProblemType("import-rejected", 422, "We couldn't read that file"),
        ProblemType("decision-conflict", 409, "That plan was already decided"),
        ProblemType("credential-invalid", 422, "Those sign-in details didn't work"),
        ProblemType("connector-unavailable", 422, "This source isn't available to connect here"),
        ProblemType("credential-storage-disabled", 422, "Credential storage is not enabled"),
        ProblemType("internal-error", 500, "Something went wrong on our side"),
    )
    return {entry.slug: entry for entry in entries}


#: The frozen, closed problem-type catalog (ERR-R8).
CATALOG: Final[Mapping[str, ProblemType]] = _catalog()


@dataclass(frozen=True, slots=True)
class FieldError:
    """One field-level validation finding (ERR-R6).

    Exactly one locator is set: ``pointer`` (a JSON Pointer into the request body)
    or ``parameter`` (a query/path parameter name). ``code`` is the stable machine
    code clients branch on; ``message`` is jargon-free, non-leaking copy (ERR-R5).
    """

    code: str
    message: str
    pointer: str | None = None
    parameter: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Render to the ``errors[]`` member shape (ERR-R2/R6)."""
        item: dict[str, str] = {"code": self.code, "message": self.message}
        if self.pointer is not None:
            item["pointer"] = self.pointer
        if self.parameter is not None:
            item["parameter"] = self.parameter
        return item


class ProblemError(Exception):
    """A raisable error that renders to one RFC 9457 problem document (ERR-R1/R2).

    Handlers and dependencies raise this; the registered handler serializes it.
    ``slug`` must be a member of the closed catalog (ERR-R8) — an unknown slug is a
    programming error and is coerced to ``internal-error`` fail-closed.
    """

    def __init__(
        self,
        slug: str,
        *,
        detail: str | None = None,
        errors: Sequence[FieldError] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.problem_type = CATALOG.get(slug, CATALOG["internal-error"])
        self.detail = detail
        self.errors = tuple(errors or ())
        self.headers = dict(headers or {})
        super().__init__(self.problem_type.slug)


def _new_trace_id() -> str:
    """Mint a fresh correlation id (opaque; ERR-R4 / API-R9)."""
    return uuid.uuid4().hex


def resolve_trace_id(request: Request) -> str:
    """Return the request's correlation id, reusing an upstream one if present.

    Prefers a trace id already attached to the request state or carried on the
    inbound ``X-Trace-Id`` header (so a client-reported error correlates to the
    server trace, ERR-R4); otherwise mints a fresh opaque id.
    """
    existing = getattr(request.state, "trace_id", None)
    if isinstance(existing, str) and existing:
        return existing
    header = request.headers.get(TRACE_HEADER)
    if header:
        return header
    return _new_trace_id()


@dataclass(frozen=True, slots=True)
class _Problem:
    """An assembled problem document ready to serialize (ERR-R2)."""

    problem_type: ProblemType
    detail: str
    instance: str
    trace_id: str
    errors: tuple[FieldError, ...] = field(default_factory=tuple)
    status_override: int | None = None

    @property
    def status(self) -> int:
        """The HTTP status for this occurrence (the framework override, else catalog)."""
        if self.status_override is not None:
            return self.status_override
        return self.problem_type.status

    def to_body(self) -> dict[str, Any]:
        """Render the RFC 9457 body (ERR-R2): the six required members + errors[].

        Every free-text member that could echo caller/adapter-supplied text — ``detail``
        and each ``errors[].message`` — is passed through :func:`redact_text` so a
        secret/token/email shape never leaves the process in a problem document
        (ERR-R5 / API-R19). The catalog ``title`` and machine ``code``/locators are
        controlled values and are emitted verbatim.
        """
        body: dict[str, Any] = {
            "type": self.problem_type.uri,
            "title": self.problem_type.title,
            "status": self.status,
            "detail": redact_text(self.detail),
            "instance": self.instance,
            "trace_id": self.trace_id,
        }
        if self.errors:
            body["errors"] = [
                replace(err, message=redact_text(err.message)).to_dict() for err in self.errors
            ]
        return body

    def to_json_bytes(self) -> bytes:
        """Serialize the redacted RFC 9457 body to UTF-8 JSON bytes (for ASGI middleware)."""
        return json.dumps(self.to_body()).encode()


def _assemble(
    request: Request,
    problem_type: ProblemType,
    *,
    detail: str | None,
    errors: tuple[FieldError, ...],
    status_override: int | None = None,
) -> _Problem:
    """Bind a problem type to this occurrence (path + trace), defaulting detail."""
    return _Problem(
        problem_type=problem_type,
        detail=detail if detail is not None else problem_type.title,
        instance=request.url.path,
        trace_id=resolve_trace_id(request),
        errors=errors,
        status_override=status_override,
    )


def _render(problem: _Problem, headers: Mapping[str, str] | None = None) -> JSONResponse:
    """Serialize a problem to a ``problem+json`` response (ERR-R1/R4)."""
    response_headers = {TRACE_HEADER: problem.trace_id, **dict(headers or {})}
    return JSONResponse(
        status_code=problem.status,
        content=problem.to_body(),
        media_type=PROBLEM_MEDIA_TYPE,
        headers=response_headers,
    )


async def _on_problem_error(request: Request, exc: ProblemError) -> JSONResponse:
    """Render a raised :class:`ProblemError` (the primary error path)."""
    problem = _assemble(request, exc.problem_type, detail=exc.detail, errors=exc.errors)
    return _render(problem, headers=exc.headers)


async def _on_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Map a framework ``HTTPException`` onto the catalog (ERR-R1).

    A few statuses framework code raises directly (e.g. ``404`` for an unmatched
    route, ``405``) are normalized to the closed catalog so even framework-origin
    errors emit the uniform document — never a default HTML/JSON shape. The
    ORIGINATING HTTP status is preserved on the response line and the body ``status``
    (ERR-R7/ERR-R4): a framework ``400`` stays a ``400`` (it is NOT silently rewritten
    to the ``422`` ``validation-error`` type), and a ``405`` stays a ``405``. Routers
    raise :class:`ProblemError` for domain errors, so the status-only path here only
    ever sees framework-origin (unmatched route / wrong method / malformed) errors.
    """
    slug = _STATUS_TO_SLUG.get(exc.status_code, "internal-error")
    problem_type = CATALOG[slug]
    # Never echo a framework-supplied detail verbatim (ERR-R5): use catalog copy.
    # Pin the body+line status to the originating exception status so the framework
    # status table is not collapsed into the catalog type's default status (ERR-R7).
    problem = _assemble(
        request, problem_type, detail=None, errors=(), status_override=exc.status_code
    )
    headers = exc.headers or {}
    return _render(problem, headers=headers)


async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Render a request-validation failure as ``422 validation-error`` (ERR-R6)."""
    problem_type = CATALOG["validation-error"]
    errors = tuple(_field_errors(exc))
    problem = _assemble(request, problem_type, detail=None, errors=errors)
    return _render(problem)


async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: any unhandled exception -> generic ``internal-error``.

    The original exception is logged (with the trace id) for the operator; the
    client body carries ONLY the generic catalog copy — no stack trace, no message,
    no internal id (ERR-R5 / AUTH-R9 / OBS-R7).
    """
    problem = _assemble(request, CATALOG["internal-error"], detail=None, errors=())
    _logger.error(
        "unhandled_exception",
        trace_id=problem.trace_id,
        path=request.url.path,
        error_type=type(exc).__name__,
    )
    return _render(problem)


def _field_errors(exc: RequestValidationError) -> list[FieldError]:
    """Translate FastAPI/pydantic validation errors into ``errors[]`` (ERR-R6).

    Body violations get a JSON Pointer (``pointer``); query/path/header violations
    get the offending ``parameter`` name. Only the stable machine ``code`` (the
    pydantic error ``type``) and a short, non-leaking message are surfaced (ERR-R5).
    """
    out: list[FieldError] = []
    for raw in exc.errors():
        loc = tuple(raw.get("loc", ()))
        code = str(raw.get("type", "invalid"))
        message = _safe_validation_message(raw)
        pointer, parameter = _locate(loc)
        out.append(FieldError(code=code, message=message, pointer=pointer, parameter=parameter))
    return out


def _locate(loc: Sequence[Any]) -> tuple[str | None, str | None]:
    """Split a pydantic ``loc`` into a body JSON Pointer or a parameter name."""
    if not loc:
        return None, None
    origin = str(loc[0])
    rest = loc[1:]
    if origin == "body":
        pointer = "/" + "/".join(str(part) for part in rest) if rest else "/"
        return pointer, None
    # query/path/header: the offending parameter is the last segment.
    parameter = str(loc[-1])
    return None, parameter


def _safe_validation_message(raw: Mapping[str, Any]) -> str:
    """Return a short, non-leaking validation message (ERR-R5).

    pydantic ``msg`` strings describe the constraint (e.g. "Field required") and do
    not carry secrets/PII; we pass that constraint description through but never the
    rejected ``input`` value, keeping the field copy leak-free.
    """
    msg = raw.get("msg")
    return str(msg) if msg else "This value isn't valid."


_STATUS_TO_SLUG: Final[Mapping[int, str]] = {
    HTTPStatus.BAD_REQUEST: "bad-request",
    HTTPStatus.UNAUTHORIZED: "unauthenticated",
    HTTPStatus.FORBIDDEN: "insufficient-scope",
    HTTPStatus.NOT_FOUND: "not-found",
    HTTPStatus.METHOD_NOT_ALLOWED: "bad-request",
    HTTPStatus.CONFLICT: "conflict",
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "payload-too-large",
    HTTPStatus.UNSUPPORTED_MEDIA_TYPE: "unsupported-media-type",
    HTTPStatus.UNPROCESSABLE_ENTITY: "validation-error",
    HTTPStatus.TOO_MANY_REQUESTS: "rate-limited",
    HTTPStatus.INTERNAL_SERVER_ERROR: "internal-error",
}


def render_problem_bytes(slug: str, request: Request) -> bytes:
    """Render a catalog problem to redacted ``problem+json`` bytes (for ASGI middleware).

    Used where a problem must be emitted OUTSIDE the FastAPI exception path (an ASGI
    middleware that rejects a request before routing); produces the same RFC 9457 body
    the handlers do (ERR-R1), with ``detail``/``errors[]`` redaction applied (ERR-R5).
    """
    problem = _assemble(request, CATALOG[slug], detail=None, errors=())
    return problem.to_json_bytes()


def install_error_handlers(app: FastAPI) -> None:
    """Register the uniform RFC 9457 handlers on the app (ERR-R1).

    Order is by exception type, not registration: FastAPI dispatches the most
    specific handler. The broad ``Exception`` handler is the fail-closed net so no
    raw error ever escapes (ERR-R5).
    """
    app.add_exception_handler(ProblemError, _typed(_on_problem_error))
    app.add_exception_handler(RequestValidationError, _typed(_on_validation_error))
    app.add_exception_handler(StarletteHTTPException, _typed(_on_http_exception))
    app.add_exception_handler(Exception, _typed(_on_unhandled))


def _typed(handler: Any) -> Any:
    """Identity wrapper so Starlette's loose handler signature satisfies mypy.

    Starlette types exception handlers as ``Callable[[Request, Exception], ...]``;
    our handlers narrow the second argument. This pass-through keeps the precise
    inner signatures while presenting the broad type the registry expects.
    """
    return handler


__all__ = [
    "CATALOG",
    "PROBLEM_BASE_URI",
    "PROBLEM_MEDIA_TYPE",
    "TRACE_HEADER",
    "FieldError",
    "ProblemError",
    "ProblemType",
    "install_error_handlers",
    "resolve_trace_id",
]
