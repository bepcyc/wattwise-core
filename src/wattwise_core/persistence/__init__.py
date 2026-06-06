"""Persistence package: portable ORM substrate (GBO-R8b, doc 20).

Exposes the engine/session machinery and the upsert seam. The ORM models live in
:mod:`wattwise_core.persistence.models`; the only sanctioned dialect branch is in
:mod:`wattwise_core.persistence.upsert`.
"""

from __future__ import annotations

from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.engine import (
    Database,
    create_engine_from_settings,
    create_session_factory,
)
from wattwise_core.persistence.upsert import upsert

__all__ = [
    "Base",
    "Database",
    "TimestampMixin",
    "create_engine_from_settings",
    "create_session_factory",
    "upsert",
]
