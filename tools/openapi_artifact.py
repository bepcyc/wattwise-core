"""Build the committed OpenAPI reference artifact (DOC-R2).

The published OpenAPI document is the single source of truth for the contract; a
committed copy lives at ``src/wattwise_core/api/openapi.json`` and the contract suite
fails when the running server's emitted schema diverges from it (DOC-R2 — the CI gate
runs that suite, so drift fails the build). This tool builds the app with fixed,
deterministic reference settings (throwaway dev values; nothing secret reaches the
document) and emits the canonical, sorted-key JSON form.

Usage::

    uv run python -m tools.openapi_artifact          # rewrite the committed artifact
    uv run python -m tools.openapi_artifact --check  # exit 1 when the emitted doc drifts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from wattwise_core.api.app import create_app
from wattwise_core.config import load_settings
from wattwise_core.security.crypto import EnvelopeCipher

#: The committed artifact path (package data, served verbatim nowhere — a reference).
ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "wattwise_core" / "api" / "openapi.json"
)


def build_reference_document() -> dict[str, Any]:
    """Emit the OpenAPI document from a deterministically-configured app (DOC-R2).

    The settings are throwaway reference values: the document depends only on the
    route/schema surface, never on the configured secrets/DSNs, so the emitted JSON is
    reproducible anywhere.
    """
    settings = load_settings(
        app__environment="development",
        database_dsn="sqlite+aiosqlite:///:memory:",
        token_signing_key="openapi-artifact-reference-key-0123456789abcdef",  # noqa: S106 - throwaway reference value, not a credential
        encryption_root_key=EnvelopeCipher.generate_root_key(),
        object_store__local_root="/tmp/wattwise-openapi-artifact",  # noqa: S108 - never written; reference config only
    )
    app = create_app(settings)
    document: dict[str, Any] = app.openapi()
    return document


def render(document: dict[str, Any]) -> str:
    """The canonical, diff-stable text form of the document (sorted keys)."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write (or ``--check``) the committed artifact; non-zero exit on drift."""
    args = argv if argv is not None else sys.argv[1:]
    emitted = render(build_reference_document())
    if "--check" in args:
        committed = ARTIFACT_PATH.read_text() if ARTIFACT_PATH.exists() else ""
        if committed != emitted:
            sys.stderr.write(
                "OpenAPI drift: emitted schema differs from the committed "
                f"{ARTIFACT_PATH.name} (DOC-R2). Run `just openapi` to regenerate.\n"
            )
            return 1
        return 0
    ARTIFACT_PATH.write_text(emitted)
    sys.stdout.write(f"wrote {ARTIFACT_PATH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
