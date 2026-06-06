"""Source-adapter contract (ADP-R*, MAP-R1, Principle A/B).

Each source is a pluggable adapter — the ONLY code that knows a source's shape,
units, quirks, and ids (GBO-R3). An adapter's :meth:`SourceAdapter.map` is a **pure**
function (no clocks, no randomness, no network; MAP-R1) turning a source-shaped object
(ASBO) into canonical :class:`~wattwise_core.domain.candidate.GboCandidate` list.

Fetching (the impure part — network/file I/O) is separate from mapping, so mapping
stays unit- and golden-testable. Ingestion uses direct typed clients, never MCP
(Principle B); MCP exists only as the agent's runtime tool interface.

Adding a source is one adapter + one descriptor registration (ROAD-R6): no consumer,
analytics, or agent change. Adapters are discovered via the
``wattwise_core.adapters`` entry-point group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import AuthArchetype, SourceKind


@dataclass(frozen=True, slots=True)
class SourceDescriptorRef:
    """The registered-source identity an adapter maps under (LIN-R1).

    This is lineage metadata, opaque to consumers. ``source_key`` and ``kind`` come
    from the registry, not from a hardcoded source name in consumer code.
    """

    source_descriptor_id: str
    source_key: str
    kind: SourceKind


@dataclass(frozen=True, slots=True)
class FetchContext:
    """Deterministic inputs a mapping may need that are NOT wall-clock/network.

    ``ingest_run_id`` and ``fetched_at`` are supplied by the caller (the sync
    orchestrator) so :meth:`SourceAdapter.map` stays pure — it never reads the clock.
    """

    ingest_run_id: str
    fetched_at: Any  # datetime, passed in (never read from the clock inside map)
    connection_id: str | None = None


@runtime_checkable
class SourceAdapter(Protocol):
    """The pluggable ingestion adapter contract (ADP-R*).

    Implementations declare their identity metadata as class attributes and provide a
    pure :meth:`map`. The fetch side (a direct typed client) lives on the concrete
    adapter but is invoked outside :meth:`map`.
    """

    source_key: str
    auth_archetype: AuthArchetype
    kind: SourceKind
    adapter_version: str
    mapping_version: str

    def map(
        self,
        asbo: Any,
        source_descriptor: SourceDescriptorRef,
        fetch_context: FetchContext,
    ) -> list[GboCandidate]:
        """Map one source-shaped object into canonical candidates (MAP-R1).

        MUST be pure and deterministic: no I/O, no clock, no randomness. MUST emit
        only canonical fields + lineage (MAP-R2); convert to canonical units (MAP-R3);
        map source vocab to canonical enums/registries (MAP-R4); preserve real gaps as
        typed missing (MAP-R5); tag free text untrusted (MAP-R7).
        """
        ...


__all__ = ["FetchContext", "SourceAdapter", "SourceDescriptorRef"]
