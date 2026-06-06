"""OpenAPI post-processing: reusable Problem + PageEnvelope components (DOC-R3/R4/R5).

FastAPI's generated document carries per-operation request/response schemas but not the
cross-cutting error contract. This module enriches the published document so a typed
client can be generated with NO manual fixups (DOC-R5):

- a reusable ``Problem`` component (RFC 9457: ``type|title|status|detail|instance|
  trace_id`` + optional ``errors[]``) and a ``ProblemFieldError`` sub-schema;
- a reusable ``PageEnvelope`` component (the cursor-pagination wrapper, PAGE-R1);
- every operation gains the closed-catalog error responses (``400/401/403/404/409/
  413/415/422/429/500/503``) as ``application/problem+json`` ``$ref``-ing ``Problem``
  (DOC-R4), so the contract documents how each non-2xx is shaped (DOC-R3).

Requirement IDs: DOC-R3 (documented Problem error responses per op), DOC-R4 (reusable
Problem + PageEnvelope components; all error responses $ref Problem), DOC-R5 (the
document is sufficient to generate a typed client with no fixups).
"""

from __future__ import annotations

from typing import Any, Final

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from wattwise_core.api.errors import PROBLEM_BASE_URI, PROBLEM_MEDIA_TYPE

#: The catalog statuses every operation may emit (the closed error surface, ERR-R8).
_ERROR_STATUSES: Final[tuple[int, ...]] = (400, 401, 403, 404, 409, 413, 415, 422, 429, 500, 503)


def _problem_field_error_schema() -> dict[str, Any]:
    """The ``errors[]`` member schema (ERR-R6): a machine code + locator + message."""
    return {
        "type": "object",
        "required": ["code", "message"],
        "properties": {
            "code": {"type": "string"},
            "message": {"type": "string"},
            "pointer": {"type": "string", "nullable": True},
            "parameter": {"type": "string", "nullable": True},
        },
    }


def _problem_schema() -> dict[str, Any]:
    """The reusable RFC 9457 ``Problem`` component (DOC-R4)."""
    return {
        "type": "object",
        "required": ["type", "title", "status", "detail", "instance", "trace_id"],
        "properties": {
            "type": {"type": "string", "format": "uri", "example": f"{PROBLEM_BASE_URI}not-found"},
            "title": {"type": "string"},
            "status": {"type": "integer"},
            "detail": {"type": "string"},
            "instance": {"type": "string"},
            "trace_id": {"type": "string"},
            "errors": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/ProblemFieldError"},
                "nullable": True,
            },
        },
    }


def _page_envelope_schema() -> dict[str, Any]:
    """The reusable cursor-pagination ``PageEnvelope`` component (PAGE-R1/DOC-R4)."""
    return {
        "type": "object",
        "required": ["limit", "has_more"],
        "properties": {
            "limit": {"type": "integer"},
            "has_more": {"type": "boolean"},
            "next_cursor": {"type": "string", "nullable": True},
        },
    }


def _problem_response(description: str) -> dict[str, Any]:
    """One error-response entry $ref-ing the Problem component (DOC-R4)."""
    return {
        "description": description,
        "content": {PROBLEM_MEDIA_TYPE: {"schema": {"$ref": "#/components/schemas/Problem"}}},
    }


def _attach_error_responses(operation: dict[str, Any]) -> None:
    """Add the closed-catalog Problem error responses to one operation (DOC-R3/R4).

    Every catalog error status is documented as ``application/problem+json`` ``$ref``-ing
    the reusable ``Problem`` component — INCLUDING ``422``, replacing FastAPI's default
    ``HTTPValidationError``/``application/json`` body so ALL error responses share the one
    RFC 9457 shape (DOC-R4) and a generated client never has to branch on two error types.
    """
    responses = operation.setdefault("responses", {})
    for status in _ERROR_STATUSES:
        responses[str(status)] = _problem_response("Error (RFC 9457 problem)")


def build_openapi(app: FastAPI) -> dict[str, Any]:
    """Build the enriched OpenAPI document for ``app`` (DOC-R3/R4/R5).

    Generates the base FastAPI document, registers the reusable ``Problem`` /
    ``ProblemFieldError`` / ``PageEnvelope`` components, and decorates every operation
    with the closed-catalog ``application/problem+json`` error responses so a typed
    client can branch on the uniform error shape with no manual fixups.
    """
    schema = get_openapi(
        title=app.title, version=app.version, routes=app.routes, description=app.description or ""
    )
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    components["ProblemFieldError"] = _problem_field_error_schema()
    components["Problem"] = _problem_schema()
    components["PageEnvelope"] = _page_envelope_schema()
    for path_item in schema.get("paths", {}).values():
        for method, operation in path_item.items():
            if method in {"get", "post", "put", "patch", "delete"} and isinstance(operation, dict):
                _attach_error_responses(operation)
    return schema


def install_openapi(app: FastAPI) -> None:
    """Bind :func:`build_openapi` as the app's cached OpenAPI generator (DOC-R4)."""

    def _openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = build_openapi(app)
        return app.openapi_schema

    app.openapi = _openapi  # type: ignore[method-assign]


__all__ = ["build_openapi", "install_openapi"]
