"""OpenAPI artifact + client-generation contract gates (DOC-R2..R6).

Proves the §13 contract-artifact requirements on the REAL emitted document:

* **DOC-R2** — the running app's emitted schema is byte-identical to the committed
  ``src/wattwise_core/api/openapi.json``; a drift fails this (CI-gated) suite.
* **DOC-R3** — every protected operation declares an ``operationId`` and a
  per-operation ``security`` entry carrying its REQUIRED scope tokens; only the
  enumerated AUTH-R10 public/pre-token/credentialed-non-bearer operations omit it.
* **DOC-R4** — reusable ``Problem`` + ``PageEnvelope`` components exist, the bearer
  scheme is ``http``/``bearer`` with ``bearerFormat: JWT``, and every documented error
  response ``$ref``s the ``Problem`` component.
* **DOC-R5** — the document generates TypeScript interfaces + type guards with zero
  unresolved ``$ref``/unknown types, and the ``uuid``/``email`` format modifiers are
  present in the schema surface.
* **DOC-R6** — the generated artifacts include an ``isProblem(value)`` guard and
  per-resource guards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ``tools`` lives at the repo root (a namespace package, not under ``src/``): bootstrap
# the root onto ``sys.path`` so collection works under any pytest import-mode (the same
# shim ``tests/unit/test_lints.py`` uses for the lint pack).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.client_gen import generate  # noqa: E402  (after the sys.path bootstrap)
from tools.openapi_artifact import (  # noqa: E402  (after the sys.path bootstrap)
    ARTIFACT_PATH,
    build_reference_document,
    render,
)

pytestmark = pytest.mark.contract

#: The only operations allowed to omit per-op security: the AUTH-R10 public set plus
#: the refresh/revoke pair (credentialed by the PRESENTED refresh token, not a bearer).
_NO_SECURITY_ALLOWED = {
    ("get", "/v1/system/status"),
    ("get", "/v1/help/topics"),
    ("get", "/v1/help/topics/{topic_id}"),
    ("post", "/v1/auth/token"),
    ("post", "/v1/auth/link/start"),
    ("post", "/v1/auth/link/complete"),
    ("post", "/v1/auth/refresh"),
    ("post", "/v1/auth/revoke"),
    # The export download is the documented bearer-FREE signed-URL alternative
    # (API-R34); its bearer path authenticates manually through the same verifier.
    ("get", "/v1/exports/{job_id}/download"),
}


#: Bearer-authenticated operations that deliberately require NO scope: the account-link
#: proof-of-control step needs only a valid owner session (AUTH-R8), nothing more.
_AUTH_ONLY = {
    ("post", "/v1/auth/link/approve"),
}


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    """The reference OpenAPI document emitted by the real app factory."""
    return build_reference_document()


def _operations(document: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Every (method, path, operation) triple in the document."""
    out = []
    for path, item in document["paths"].items():
        for method, operation in item.items():
            if method in {"get", "post", "put", "patch", "delete"}:
                out.append((method, path, operation))
    return out


def test_emitted_schema_matches_the_committed_artifact(document: dict[str, Any]) -> None:
    """The emitted schema is byte-identical to the committed openapi.json (DOC-R2)."""
    assert ARTIFACT_PATH.exists(), "the committed OpenAPI artifact is missing (DOC-R2)"
    assert render(document) == ARTIFACT_PATH.read_text(), (
        "OpenAPI drift: run `just openapi` and commit the regenerated artifact (DOC-R2)"
    )


def test_every_operation_declares_operation_id_and_scoped_security(
    document: dict[str, Any],
) -> None:
    """Every op has a stable operationId; protected ops declare their scopes (DOC-R3)."""
    for method, path, operation in _operations(document):
        assert operation.get("operationId"), f"{method.upper()} {path} has no operationId"
        security = operation.get("security")
        if (method, path) in _NO_SECURITY_ALLOWED:
            continue
        assert security, f"{method.upper()} {path} declares no security (DOC-R3)"
        scopes = security[0].get("bearer")
        if (method, path) in _AUTH_ONLY:
            assert scopes == [], f"{method.upper()} {path} should be bearer-only"
            continue
        assert scopes, f"{method.upper()} {path} declares no required scopes (DOC-R3)"


def test_problem_and_page_envelope_components_and_jwt_bearer(
    document: dict[str, Any],
) -> None:
    """Problem + PageEnvelope components exist; bearer scheme is JWT-tagged (DOC-R4)."""
    schemas = document["components"]["schemas"]
    assert "Problem" in schemas and "PageEnvelope" in schemas
    bearer = document["components"]["securitySchemes"]["bearer"]
    assert bearer == {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}


def test_every_error_response_refs_the_problem_component(document: dict[str, Any]) -> None:
    """All documented non-2xx responses $ref the reusable Problem component (DOC-R4)."""
    for method, path, operation in _operations(document):
        for status, response in operation.get("responses", {}).items():
            if not status.startswith(("4", "5")):
                continue
            content = response.get("content", {})
            schema = content.get("application/problem+json", {}).get("schema", {})
            assert schema.get("$ref") == "#/components/schemas/Problem", (
                f"{method.upper()} {path} {status} does not $ref Problem (DOC-R4)"
            )


def test_format_modifiers_present(document: dict[str, Any]) -> None:
    """The uuid/email format modifiers appear in the schema surface (DOC-R5)."""
    blob = json.dumps(document)
    assert '"format": "uuid"' in blob or '"format":"uuid"' in blob
    assert '"format": "email"' in blob or '"format":"email"' in blob


def test_typescript_client_generates_with_guards(document: dict[str, Any]) -> None:
    """TS interfaces + guards generate with no unresolved refs; isProblem exists (DOC-R5/R6).

    ``generate`` raises on any unresolved ``$ref``, unknown schema type, or a required
    field it cannot guard — so a passing run IS the DOC-R5 sufficiency proof. The
    rendered module must carry the DOC-R6 ``isProblem`` guard plus per-resource guards.
    """
    rendered = generate(document)
    assert "export function isProblem(value: unknown): value is Problem" in rendered
    assert rendered.count("export function is") > 50  # per-resource guards, not just one
    assert "export interface PageEnvelope" in rendered
