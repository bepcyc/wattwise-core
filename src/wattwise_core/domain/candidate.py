"""Canonical candidate value objects (MAP-R2, LIN-R2).

A :class:`GboCandidate` is what an adapter's pure ``map`` emits (MAP-R1): canonical
fields ONLY plus lineage metadata — never a source-named field, unit, or enum
(MAP-R2). The candidate store keeps these per-source observations (tier 2); the
resolver collapses them into the canonical record (tier 3) via the conflict policy.

A :class:`FieldCandidate` is one contributing value for one canonical field, carrying
exactly the signals :func:`wattwise_core.ingestion.dedup.resolve_field` ranks on
(CONF-R2). These are pure data — no behaviour, no source-name branching.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

from wattwise_core.domain.coverage import Coverage
from wattwise_core.domain.enums import Fidelity


@dataclass(frozen=True, slots=True)
class FieldCandidate:
    """One source's contribution for one canonical field (CONF-R2 inputs).

    ``trust_tier`` is the ranked fidelity (raw_stream > device_computed > ... >
    summary_only); ``completeness`` is higher for a full stream than a summary-only
    scalar. ``source_descriptor_id`` is the final stable tiebreak (lowest wins) and
    is lineage only — it never reaches a consumer.
    """

    value: Any
    trust_tier: Fidelity
    source_descriptor_id: str
    confidence: float = 1.0
    observed_at: _dt.datetime | None = None
    fetched_at: _dt.datetime | None = None
    completeness: float = 1.0
    # The contributing ``source_candidate`` row id (LIN-R3 pointer); ``None`` only for
    # contributions built outside the candidate store (pure unit tests).
    candidate_id: str | None = None


@dataclass(frozen=True, slots=True)
class GboCandidate:
    """A mapped per-source observation in canonical shape (MAP-R1/R2/R8).

    ``payload`` holds only canonical field names → values (validated; no source-named
    keys). ``source_native_id`` + ``content_hash`` + ``(source_descriptor_id, gbo_type)``
    form the candidate idempotency key (MAP-R8); none of these may appear in a
    canonical key. ``untrusted_content`` flags free-text the agent must treat as data,
    never instructions (MAP-R7).
    """

    gbo_type: str
    source_descriptor_id: str
    source_native_id: str
    content_hash: str
    payload: dict[str, Any]
    observed_at: _dt.datetime | None = None
    fetched_at: _dt.datetime | None = None
    confidence: float = 1.0
    trust_tier: Fidelity = Fidelity.PLATFORM_COMPUTED
    untrusted_content: bool = False
    connection_id: str | None = None
    adapter_version: str = "0"
    mapping_version: str = "0"
    coverage: dict[str, Coverage] = field(default_factory=dict)
    # MAP-R10: the TYPED shared device/file fingerprint (e.g. the FIT ``file_id``
    # fingerprint) — a REAL cross-source identity signal, distinct from the per-source
    # ``source_native_id`` dedup key. ``None`` when the source exposes none.
    strong_fingerprint: str | None = None

    def candidate_key(self, athlete_id: str) -> tuple[str, str, str, str]:
        """The candidate idempotency/dedup key (UPS-R1) — the only key with source id."""
        return (athlete_id, self.source_descriptor_id, self.source_native_id, self.gbo_type)


__all__ = ["FieldCandidate", "GboCandidate"]
