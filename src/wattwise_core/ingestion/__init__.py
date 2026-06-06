"""Ingestion: pluggable source adapters + the canonical write path (doc 30, Principle A/B).

Ingestion uses direct, typed source clients (never MCP, Principle B). Each adapter
maps source-shaped objects into canonical candidates via a pure ``map`` (MAP-R1); the
ingest service persists candidates, resolves identity, runs the conflict resolver, and
writes the resolved canonical records in one transaction (UPS-R6).
"""

from __future__ import annotations

from wattwise_core.ingestion.base import FetchContext, SourceAdapter, SourceDescriptorRef
from wattwise_core.ingestion.dedup import resolve_activity_identity, resolve_field

__all__ = [
    "FetchContext",
    "SourceAdapter",
    "SourceDescriptorRef",
    "resolve_activity_identity",
    "resolve_field",
]
