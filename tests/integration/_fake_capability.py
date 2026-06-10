"""Shared test helper: a valid capability descriptor for fake adapters (ADP-R1/R2).

Registration now VALIDATES every adapter's capability descriptor (ONB-R2), so each
test fake declares one. The fakes use the legacy window-``fetch`` seam, which the
phase validation accepts for incremental adapters; the declared GBO types drive the
engine's ADP-R3 declared-type refusal in the tests that exercise it.
"""

from __future__ import annotations

from wattwise_core.domain.enums import AuthArchetype, Fidelity, GboType
from wattwise_core.ingestion.capability import (
    CapabilityDescriptor,
    DiscoveryOrder,
    Granularity,
    SyncMode,
)


def fake_capability(
    source_key: str,
    *,
    gbo_types: frozenset[GboType] = frozenset({GboType.ACTIVITY, GboType.DAILY_WELLNESS}),
    auth_archetype: AuthArchetype = AuthArchetype.API_KEY,
) -> CapabilityDescriptor:
    """A minimal valid descriptor for a fake window-fetch adapter (test-only)."""
    return CapabilityDescriptor(
        source_key=source_key,
        supported_gbo_types=gbo_types,
        sync_modes=frozenset({SyncMode.INCREMENTAL}),
        auth_archetype=auth_archetype,
        server_side_incremental=False,
        discovery_order=DiscoveryOrder.OLDEST_FIRST,
        granularity=dict.fromkeys(gbo_types, Granularity.SUMMARY_ONLY),
        equivalence_classes=(),
        default_trust_profile=Fidelity.PLATFORM_COMPUTED,
        rate_limit_config_section=None,
    )


__all__ = ["fake_capability"]
