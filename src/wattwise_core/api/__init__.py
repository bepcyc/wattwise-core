"""The OSS HTTP API — part of the wattwise-core package (its ASGI app, doc 60).

The single-owner REST surface + SSE; the OpenAPI spec is generated from it. No GUI
client ships in OSS. Build the app with :func:`wattwise_core.api.app.create_app`.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str) -> object:
    # Lazy re-export so importing the package does not pull in FastAPI eagerly.
    if name == "create_app":
        from wattwise_core.api.app import create_app

        return create_app
    raise AttributeError(name)
