"""wattwise-core — the open-source endurance-analytics + coaching-agent engine.

`wattwise-core` is a single-athlete, single-operator self-host engine. It unifies
training data from pluggable source adapters into one canonical record of truth
(the GBO master data), computes sports-science-correct analytics, and runs a
trustworthy LangGraph coaching agent (fail-closed grounding) over that canonical
store. The HTTP API (its ASGI app) is part of this package.

This is the OSS engine of the `wattwise` family (Apache-2.0). The commercial
product `athload` builds multi-tenancy, billing, managed connectors, the web app,
and the Telegram bot additively on top of this engine; none of that lives here.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Single source of version truth for the package and the `wattwise-core:vX.Y.Z`
# image tag; kept in sync with pyproject `[project].version` by the release tool.
__version__ = "0.1.0"
