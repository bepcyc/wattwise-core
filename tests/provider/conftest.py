"""Gate for the manual provider tier: real OpenAI-compatible REST API, never CI.

The tier runs only when ``WATTWISE_PROVIDER_API_KEY`` is set. That variable is distinct from
``WATTWISE_LLM_API_KEY`` on purpose: CI carries the latter as a secret for its nightly live
leg, so gating on it would pull this tier into CI. As a second guard, any run with ``CI`` set
skips the tier even when a key is present.

Point the tier at any OpenAI-compatible endpoint:

* ``WATTWISE_PROVIDER_API_KEY`` — required, the provider key (e.g. an OpenRouter key);
* ``WATTWISE_PROVIDER_BASE_URL`` — default ``https://openrouter.ai/api/v1``;
* ``WATTWISE_PROVIDER_MODEL`` — default ``minimax/minimax-m3.1-flash`` (the shipped default).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from wattwise_core.config import Settings, load_settings

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "minimax/minimax-m3.1-flash"


def _skip_reason() -> str | None:
    """Why the provider tier must not run here, or ``None`` when it may."""
    if os.environ.get("CI"):
        return "provider tier never runs in CI (CI is set)"
    if not os.environ.get("WATTWISE_PROVIDER_API_KEY"):
        return "provider tier: set WATTWISE_PROVIDER_API_KEY (and optionally _BASE_URL/_MODEL)"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every provider-tier test unless the manual gate above allows it."""
    reason = _skip_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("provider") is not None:
            item.add_marker(skip)


@pytest.fixture
def provider_settings() -> Settings:
    """Settings aimed at the configured provider, with the repo's agent defaults otherwise."""
    overrides: dict[str, Any] = {
        "app__environment": "development",
        "database_dsn": "sqlite+aiosqlite:///:memory:",
        "agent__base_url": os.environ.get("WATTWISE_PROVIDER_BASE_URL", DEFAULT_BASE_URL),
        "agent__model": os.environ.get("WATTWISE_PROVIDER_MODEL", DEFAULT_MODEL),
        "llm_api_key": os.environ["WATTWISE_PROVIDER_API_KEY"],
    }
    return load_settings(**overrides)
