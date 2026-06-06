"""API router registry. Each router module exposes ``router: APIRouter``; the app
factory includes them. Routers are mounted here so the factory stays thin."""

from __future__ import annotations

__all__: list[str] = []
