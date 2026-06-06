"""OSS testing deliverables: reusable seam-contract base-classes (GOLD-R5 §6.3a).

Every extension seam (adapter, resolver, MemoryStore, coach-config, entitlement,
sport-registry, MCP tool interface — the QUAL-R9(c) Protocol/ABC seams) ships with an
abstract pytest contract base-class encoding the seam's invariants as impl-agnostic
cases any conforming implementation MUST pass. These are part of every seam's DOD
evidence and are exercised by the OSS default implementations.
"""

from __future__ import annotations

__all__ = ["seam_contracts"]
