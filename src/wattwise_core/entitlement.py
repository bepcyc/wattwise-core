"""Entitlement resolver seam (ENT-R*, SEAM-R*) — OSS default.

The commercial layer (`athload`) mounts a real entitlement system (per-tenant plans,
quotas, feature flags) on this seam. The OSS engine is single-athlete, zero-tenancy
(SCOPE-R12), so its default resolver grants the implicit owner every in-OSS
capability. The seam exists so the commercial layer can resolve → carry → check
entitlements without the OSS engine knowing anything about tenancy.

Keeping this seam clean (not building the commercial side) is the whole point: the
OSS engine boots and passes every gate with zero commercial bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Entitlements:
    """Resolved capabilities for the current principal.

    In OSS this is always the all-allowed grant for the single owner.
    """

    can_use_agent: bool = True
    can_ingest: bool = True
    can_export: bool = True

    def require(self, capability: str) -> None:
        """Raise if ``capability`` is not granted (fail-closed check seam)."""
        if not getattr(self, capability, False):
            raise PermissionError(f"capability not entitled: {capability}")


@runtime_checkable
class EntitlementResolver(Protocol):
    """Resolves the entitlements for a principal (commercial overrides this)."""

    def resolve(self, athlete_id: str) -> Entitlements:
        """Return the resolved entitlements for ``athlete_id``."""
        ...


class OssEntitlementResolver:
    """OSS default: the single owner is entitled to every in-OSS capability."""

    def resolve(self, athlete_id: str) -> Entitlements:
        return Entitlements()


__all__ = [
    "EntitlementResolver",
    "Entitlements",
    "OssEntitlementResolver",
]
